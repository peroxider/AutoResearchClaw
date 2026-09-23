"""Minimal, deterministic submission archive alongside the private audit bundle.

Only current manuscript files and approved resources actually read by TeX enter
the ZIP. This establishes payload scope, not anonymity of scientific content.
"""
from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path

from researchclaw.literature.evidence import write_json
from researchclaw.pipeline.evidence_store import content_hash, file_hash


class SubmissionError(ValueError):
    pass


def _read(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise SubmissionError("Expected submission metadata object")
    return result


def payload_files(root: Path) -> dict[str, str]:
    from researchclaw.pipeline.final_acceptance import compilation_inputs
    from researchclaw.pipeline.manuscript import verify_exports
    from researchclaw.pipeline.publication_assets import verify_assets
    from researchclaw.templates.bundle import verify_template, verify_template_resources, inspect_constraints
    verify_exports(root)
    template = verify_template(root)
    verify_template_resources(root)
    assets = verify_assets(root)
    compilation = _read(root / "compilation.json")
    recorder = root / "paper.fls"
    if (compilation.get("success") is not True or not recorder.is_file()
            or compilation.get("recorder_sha256") != file_hash(recorder)
            or compilation.get("inputs") != compilation_inputs(root)
            or compilation.get("pdf_sha256") != file_hash(root / "paper.pdf")):
        raise SubmissionError("Submission requires current successful compilation and its recorder")
    if inspect_constraints(root)["status"] != "passed":
        raise SubmissionError("Template/PDF constraints are not satisfied")
    import fitz
    with fitz.open(root / "paper.pdf") as pdf:
        if pdf.embfile_count():
            raise SubmissionError("Submission PDF contains embedded files")
    policy = template["compiled"]["policy"]
    approved = {"paper.tex", "references.bib", "paper.bbl"}
    approved.update(name for name in assets["outputs"] if Path(name).suffix in {".png", ".pdf", ".svg"})
    approved.update(name for name in template["files"]
                    if name not in {policy["entrypoint"], "template.json"}
                    and Path(name).suffix.lower() not in {".md", ".txt", ".json"})
    # TeX outputs read by later passes are not publication sources.
    # paper.run.xml is biblatex's hand-off to biber: derived from paper.tex and
    # references.bib, re-opened by the recorder pass, never a public input.
    generated = {"paper.aux", "paper.out", "paper.toc", "paper.lof", "paper.lot", "paper.run.xml"}
    selected, read_local = {"paper.pdf", "paper.tex", "references.bib"}, set()
    # BibTeX's style is read by a separate process, so it is absent from .fls.
    style = template["compiled"]["bibliography_style"] + ".bst"
    if style in template["files"]:
        selected.add(style)
    lines = recorder.read_text(encoding="utf-8").splitlines()
    if len(lines) > 100000:
        raise SubmissionError("Compilation recorder exceeds line budget")
    for line in lines:
        if not line.startswith("INPUT "):
            continue
        raw = line[6:].strip().strip('"')
        path = Path(raw)
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root.resolve()):
            if not path.is_absolute():
                raise SubmissionError("Relative compilation input escapes delivery directory")
            # Installed TeX packages/fonts are external runtime dependencies.
            # Their local installation paths never enter the public archive.
            continue
        lexical = path if path.is_absolute() else root / path
        if lexical.is_symlink() or Path(os.path.abspath(lexical)) != resolved:
            raise SubmissionError("Linked publication input is unsupported")
        name = resolved.relative_to(root.resolve()).as_posix()
        read_local.add(name)
        if name in generated:
            continue
        if name not in approved:
            raise SubmissionError("TeX read an unapproved local publication input: " + name)
        selected.add(name)
    if "paper.tex" not in read_local or "paper.bbl" not in read_local:
        raise SubmissionError("Recorder must include manuscript and compiled bibliography")
    if policy["highlights_required"]:
        selected.add("highlights.md")
    result = {}
    for name in sorted(selected):
        path = root / name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise SubmissionError("Publication input missing or linked outside delivery: " + name)
        result[name] = file_hash(path)
    return result


def _zip_bytes(root: Path, files: dict[str, str]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, (root / name).read_bytes())
    return output.getvalue()


def prepare_submission(root: Path) -> dict:
    """Record failure instead of presenting an old ZIP as the current payload."""
    archive = root / "submission.zip"
    if archive.is_symlink() or archive.resolve().parent != root.resolve():
        raise SubmissionError("Submission archive path is outside delivery")
    try:
        files = payload_files(root)
        data = _zip_bytes(root, files)
        archive.write_bytes(data)
        report = {"schema_version": 1, "status": "prepared", "files": files,
                  "archive_sha256": file_hash(archive), "content_anonymity": "unreviewed",
                  "scope": "Current publication payload only; excludes private audit data and unused template files"}
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        # Only remove our fixed, managed output, after checking its parent.
        if archive.exists():
            if archive.is_symlink() or archive.resolve().parent != root.resolve():
                raise SubmissionError("Submission archive path is outside delivery") from exc
            archive.unlink()
        report = {"schema_version": 1, "status": "failed", "files": {}, "archive_sha256": None,
                  "content_anonymity": "unreviewed", "issues": [str(exc)]}
    report["version"] = content_hash(report)
    write_json(root / "submission_bundle.json", report)
    return report


def verify_submission(root: Path) -> dict:
    report = _read(root / "submission_bundle.json")
    payload = dict(report)
    if payload.pop("version", None) != content_hash(payload):
        raise SubmissionError("Submission manifest version changed")
    files = payload_files(root)
    archive = root / "submission.zip"
    if (archive.is_symlink() or report.get("status") != "prepared" or report.get("files") != files
            or report.get("content_anonymity") != "unreviewed"
            or not archive.is_file() or report.get("archive_sha256") != file_hash(archive)
            or archive.read_bytes() != _zip_bytes(root, files)):
        raise SubmissionError("Submission archive is missing, stale or differs from exact allowed payload")
    return report
