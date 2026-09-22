"""Versioned local template bundles and explicit publication constraints.

Template metadata describes the user's supplied rules, not inferred venue
policy. Import never fetches files or executes the supplied TeX.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

from researchclaw.literature.evidence import write_json
from researchclaw.pipeline.evidence_store import content_hash, file_hash


class TemplateError(ValueError):
    pass


MARKERS = ("ARC_TITLE", "ARC_AUTHORS", "ARC_PACKAGES", "ARC_CONTENT", "ARC_BIBLIOGRAPHY")
ROLE_NAMES = {"methods", "theory", "experiments", "results", "related_work", "discussion"}
ALLOWED = {".tex", ".cls", ".sty", ".bst", ".bib", ".def", ".clo", ".cfg", ".png", ".jpg", ".jpeg", ".pdf", ".eps", ".txt", ".md", ".json"}
DEFAULT = {
    "schema_version": 1, "name": "generic-journal", "entrypoint": "main.tex", "engine": "pdflatex",
    "anonymous": True, "max_pages": None, "max_main_pages": None, "appendix_roles": [],
    "highlights_required": False,
    "columns": None,
}
GENERIC = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
{{ARC_PACKAGES}}
\title{ {{ARC_TITLE}} }
\author{ {{ARC_AUTHORS}} }
\date{}
\begin{document}
\maketitle
{{ARC_CONTENT}}
{{ARC_BIBLIOGRAPHY}}
\end{document}
"""


def _name(name: str) -> str:
    if not isinstance(name, str):
        raise TemplateError("Template file name must be text")
    path = PurePosixPath(name)
    if (not isinstance(name, str) or "\\" in name or ":" in name or path.is_absolute()
            or not path.parts or any(p in {"..", "."} for p in path.parts)
            or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", p) or p.endswith(".")
                   or p.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}
                   for p in path.parts)):
        raise TemplateError("Template file paths must be plain relative paths")
    return path.as_posix()


def _files(source: Path) -> dict[str, bytes]:
    files = {}
    if source.is_dir():
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise TemplateError("Template symlinks are unsupported")
            if path.is_file():
                if path.stat().st_size > 20_000_000:
                    raise TemplateError("Template file exceeds 20 MB")
                files[_name(path.relative_to(source).as_posix())] = path.read_bytes()
                if len(files) > 100 or sum(map(len, files.values())) > 20_000_000:
                    raise TemplateError("Template bundle exceeds 100 files or 20 MB")
    elif source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            if len(archive.infolist()) > 100 or sum(i.file_size for i in archive.infolist()) > 20_000_000:
                raise TemplateError("Template archive exceeds 100 entries or 20 MB")
            for info in archive.infolist():
                name = _name(info.filename.rstrip("/"))
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise TemplateError("Template archive contains a symlink")
                if not info.is_dir():
                    if name in files:
                        raise TemplateError("Duplicate template archive entry")
                    files[name] = archive.read(info)
    else:
        raise TemplateError("Template input must be a directory or ZIP file")
    if len({n.casefold() for n in files}) != len(files):
        raise TemplateError("Case-colliding template file names")
    if not files or any(PurePosixPath(n).suffix.lower() not in ALLOWED for n in files):
        raise TemplateError("Empty template or unsupported bundle file type")
    return files


def _policy(data: dict) -> dict:
    if not isinstance(data, dict) or set(data) - set(DEFAULT):
        raise TemplateError("Unknown template policy fields")
    result = {**DEFAULT, **data}
    if type(result["schema_version"]) is not int or result["schema_version"] != 1:
        raise TemplateError("Unsupported template policy schema")
    if not isinstance(result["name"], str) or not result["name"].strip():
        raise TemplateError("Template name is required")
    _name(result["entrypoint"])
    if result["engine"] not in {"pdflatex", "xelatex"}:
        raise TemplateError("Only pdflatex/xelatex compilation is supported")
    if result["columns"] is not None and (type(result["columns"]) is not int or result["columns"] not in {1, 2}):
        raise TemplateError("columns must be 1, 2 or null for class-based inference")
    for key in ("anonymous", "highlights_required"):
        if type(result[key]) is not bool:
            raise TemplateError(f"{key} must be boolean")
    for key in ("max_pages", "max_main_pages"):
        if result[key] is not None and (type(result[key]) is not int or result[key] < 1):
            raise TemplateError(f"{key} must be a positive integer or null")
    roles = result["appendix_roles"]
    if (not isinstance(roles, list) or any(not isinstance(r, str) or r not in ROLE_NAMES for r in roles)
            or len(set(roles)) != len(roles)):
        raise TemplateError("Invalid appendix roles")
    return result


def _metadata(files: dict[str, bytes]) -> dict:
    data = json.loads(files.get("template.json", b"{}"))
    policy = _policy(data)
    if "entrypoint" not in data and policy["entrypoint"] not in files:
        candidates = [name for name, raw in files.items() if name.endswith(".tex") and b"\\documentclass" in raw]
        if len(candidates) == 1:
            policy["entrypoint"] = candidates[0]
    return policy


def _replace_argument(text: str, command: str, replacement: str) -> str:
    matches = list(re.finditer(r"\\" + command + r"\s*\{", text))
    if len(matches) != 1:
        raise TemplateError(f"Raw template needs exactly one \\{command}; use ARC markers for specialized title macros")
    begin = matches[0].end()
    depth, end = 1, begin
    while end < len(text) and depth:
        if text[end] == "\\":
            end += 2
            continue
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
        end += 1
    if depth:
        raise TemplateError("Unbalanced template argument")
    return text[:begin] + replacement + text[end - 1:]


def _compile(files: dict[str, bytes], policy: dict) -> dict:
    entry = policy["entrypoint"]
    if entry not in files or not entry.endswith(".tex"):
        raise TemplateError("Template entrypoint is missing")
    try:
        raw = files[entry].decode("utf-8-sig")
    except UnicodeError as exc:
        raise TemplateError("Template entrypoint must be UTF-8") from exc
    # Ignore ordinary comments while parsing declarations (escaped percent stays).
    raw = re.sub(r"(?m)(?<!\\)%.*$", "", raw)
    classes = re.findall(r"\\documentclass(?:\[([^\]]*)\])?\{([^}]+)\}", raw)
    if len(classes) != 1 or not re.fullmatch(r"[A-Za-z0-9_-]+", classes[0][1]):
        raise TemplateError("Template needs one explicit document class")
    styles = re.findall(r"\\bibliographystyle\{([^}]+)\}", raw)
    if len(styles) > 1 or (styles and not re.fullmatch(r"[A-Za-z0-9_./-]+", styles[0])):
        raise TemplateError("Template has ambiguous bibliography style")
    bibliography_style = styles[0] if styles else "plainnat"
    if ".." in PurePosixPath(bibliography_style).parts or bibliography_style.startswith("/"):
        raise TemplateError("Bibliography style escapes the bundle")
    if re.search(r"\\(?:addbibresource|printbibliography)\b|\{biblatex\}", raw):
        raise TemplateError("BibLaTeX templates require a Biber backend; no silent BibTeX substitution")
    if any("{{" + marker + "}}" in raw for marker in MARKERS):
        skeleton = raw
    else:
        if raw.count(r"\begin{document}") != 1 or raw.count(r"\end{document}") != 1:
            raise TemplateError("Template needs a single document body")
        preamble, body = raw.split(r"\begin{document}", 1)
        preamble = _replace_argument(preamble, "title", "{{ARC_TITLE}}")
        preamble = _replace_argument(preamble, "author", "{{ARC_AUTHORS}}")
        if r"\maketitle" not in body:
            raise TemplateError("Raw template has specialized title layout; supply ARC markers")
        skeleton = (preamble + "\n{{ARC_PACKAGES}}\n\\begin{document}\n\\maketitle\n"
                    "{{ARC_CONTENT}}\n{{ARC_BIBLIOGRAPHY}}\n\\end{document}\n")
    if any(skeleton.count("{{" + marker + "}}") != 1 for marker in MARKERS):
        raise TemplateError("Each ARC template marker must appear exactly once")
    skeleton = re.sub(r"\\bibliographystyle\{[^}]+\}|\\bibliography\{[^}]+\}", "", skeleton)
    if skeleton.count(r"\begin{document}") != 1 or skeleton.count(r"\end{document}") != 1:
        raise TemplateError("Template document boundaries are ambiguous")
    if skeleton.index("{{ARC_PACKAGES}}") > skeleton.index(r"\begin{document}"):
        raise TemplateError("ARC_PACKAGES must occur before the document body")
    if not skeleton.index(r"\begin{document}") < skeleton.index("{{ARC_CONTENT}}") < skeleton.index("{{ARC_BIBLIOGRAPHY}}") < skeleton.index(r"\end{document}"):
        raise TemplateError("Template content/bibliography must be inside the document, in order")
    if policy["anonymous"] and re.search(r"\\(?:affiliation|institute|address|email|thanks)\b|pdfauthor\s*=", skeleton):
        raise TemplateError("Anonymous template retains identifying metadata outside ARC_AUTHORS")
    if re.search(r"\\(?:write18|directlua|openin|read)\b|\^\^", skeleton):
        raise TemplateError("Unsupported executable/file-read template construct")
    for reference in re.findall(r"\\(?:input|include)\{([^}]+)\}", skeleton):
        _name(reference)
        if reference not in files and reference + ".tex" not in files:
            raise TemplateError("Template input is missing from bundle: " + reference)
    for group in re.findall(r"\\(?:usepackage|RequirePackage)(?:\[[^\]]*\])?\{([^}]+)\}", skeleton):
        for reference in group.split(","):
            _name(reference.strip())
    columns = 2 if "twocolumn" in classes[0][0] or r"\twocolumn" in skeleton else policy["columns"] or 1
    if columns == 2 and policy["columns"] == 1:
        raise TemplateError("One-column policy conflicts with two-column source")
    return {"skeleton": skeleton, "document_class": classes[0][1], "class_options": classes[0][0], "columns": columns,
            "bibliography_style": bibliography_style, "policy": policy}


def freeze_template(root: Path, source: Path | None = None, *, authors="Anonymous") -> dict:
    try:
        files = _files(source) if source else {"main.tex": GENERIC.encode(), "template.json": json.dumps(DEFAULT).encode()}
    except (zipfile.BadZipFile, RuntimeError) as exc:
        raise TemplateError("Template ZIP cannot be read") from exc
    policy = _metadata(files)
    compiled = _compile(files, policy)
    if not isinstance(authors, str) or not authors.strip():
        raise TemplateError("Authors must be a nonempty string")
    inventory = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())}
    record = {"schema_version": 1, "compiled": compiled, "files": inventory,
              "authors": "Anonymous" if policy["anonymous"] else authors,
              "origin": "user_bundle" if source else "generic_default"}
    record["version"] = content_hash(record)
    record["directory"] = "publication_templates/" + record["version"]
    for name, data in files.items():
        target = root / record["directory"] / name
        if not target.resolve().is_relative_to(root.resolve()):
            raise TemplateError("Template snapshot target escapes run directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    write_json(root / "publication_template.json", record)
    return record


def verify_template(root: Path) -> dict:
    record = json.loads((root / "publication_template.json").read_text(encoding="utf-8"))
    payload = dict(record)
    directory, version = payload.pop("directory"), payload.pop("version")
    if content_hash(payload) != version or directory != "publication_templates/" + version:
        raise TemplateError("Template contract version changed")
    files = {}
    for name, digest in record["files"].items():
        _name(name)
        path = (root / directory / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or file_hash(path) != digest:
            raise TemplateError("Frozen template file changed or missing")
        files[name] = path.read_bytes()
    policy = _metadata(files)
    if _compile(files, policy) != record["compiled"]:
        raise TemplateError("Compiled template contract changed")
    if policy["anonymous"] and record["authors"] != "Anonymous":
        raise TemplateError("Anonymous contract exposes authors")
    return record


def copy_template_resources(root: Path, destination: Path) -> None:
    record = verify_template(root)
    reserved = {"paper.tex", "paper.pdf", "paper_final.md", "references.bib", "numeric_claims.json", "manuscript_ir.json"}
    for name in record["files"]:
        if name in {record["compiled"]["policy"]["entrypoint"], "template.json"}:
            continue
        if name.casefold() in reserved or Path(name).suffix.lower() in {".md", ".txt", ".json"}:
            if name.casefold() in reserved:
                raise TemplateError("Template resource collides with generated publication file")
            continue
        target = destination / name
        if not target.resolve().is_relative_to(destination.resolve()):
            raise TemplateError("Template resource target escapes publication directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / record["directory"] / name, target)


def verify_template_resources(root: Path) -> None:
    record = verify_template(root)
    for name, digest in record["files"].items():
        if name in {record["compiled"]["policy"]["entrypoint"], "template.json"} or Path(name).suffix.lower() in {".md", ".txt", ".json"}:
            continue
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or file_hash(path) != digest:
            raise TemplateError("Materialized template resource changed: " + name)


def render_frame(root: Path, title: str) -> tuple[str, str, dict]:
    from researchclaw.pipeline.manuscript import _tex
    record = verify_template(root)
    compiled, authors = record["compiled"], record["authors"]
    skeleton = compiled["skeleton"]
    packages = []
    present = {p.strip() for group in re.findall(r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}", skeleton) for p in group.split(",")}
    for package in ("longtable", "natbib", "amsmath", "amssymb", "graphicx", "placeins"):
        if package not in present:
            packages.append("\\usepackage{" + package + "}")
    if compiled["policy"]["engine"] == "pdflatex":
        for package, option in (("fontenc", "T1"), ("inputenc", "utf8")):
            if package not in present:
                packages.append(f"\\usepackage[{option}]{{{package}}}")
    for operator in ("argmax", "argmin"):
        if not re.search(r"\\(?:DeclareMathOperator\*?|newcommand)\s*\{\\" + operator + r"\}", skeleton):
            packages.append("\\ifdefined\\" + operator + "\\else\\DeclareMathOperator*{\\" + operator + "}{arg\\," + operator[3:] + "}\\fi")
    bibliography = "\\bibliographystyle{" + compiled["bibliography_style"] + "}\n\\bibliography{references}"
    skeleton = skeleton.replace("{{ARC_TITLE}}", _tex(title)).replace("{{ARC_AUTHORS}}", _tex(authors))
    skeleton = skeleton.replace("{{ARC_PACKAGES}}", "\n".join(packages)).replace("{{ARC_BIBLIOGRAPHY}}", bibliography)
    prefix, suffix = skeleton.split("{{ARC_CONTENT}}")
    return prefix, suffix, {**compiled["policy"], "columns": compiled["columns"]}


def inspect_constraints(root: Path) -> dict:
    """Check actual PDF page counts. Missing compilation never passes a limit."""
    record = verify_template(root)
    policy = record["compiled"]["policy"]
    issues, pages, main_pages, appendix_pages = [], None, None, []
    pdf = root / "paper.pdf"
    if pdf.is_file():
        try:
            import fitz
            with fitz.open(pdf) as doc:
                pages = doc.page_count
                appendix_pages = [index for index, page in enumerate(doc)
                                  if "Appendix" in [line.strip() for line in page.get_text().splitlines()]]
                author = (doc.metadata or {}).get("author", "").strip()
                if policy["anonymous"] and author not in {"", "Anonymous"}:
                    issues.append("pdf_author_metadata_not_anonymous")
        except (ImportError, OSError, ValueError, RuntimeError):
            issues.append("pdf_page_count_unavailable")
    else:
        issues.append("pdf_missing")
    if policy["appendix_roles"]:
        # Count physical PDF pages, never a TeX page counter that a venue can reset.
        if len(appendix_pages) == 1 and appendix_pages[0] > 0:
            main_pages = appendix_pages[0]
        elif policy["max_main_pages"] is not None:
            issues.append("main_page_count_unavailable")
    else:
        main_pages = pages
    for field, actual in (("max_pages", pages), ("max_main_pages", main_pages)):
        if policy[field] is not None and (actual is None or actual > policy[field]):
            issues.append(field + "_unavailable_or_exceeded")
    return {"template_version": record["version"], "pdf_sha256": file_hash(pdf) if pdf.is_file() else None,
            "pages": pages, "main_pages": main_pages, "status": "passed" if not issues else "failed", "issues": issues,
            "scope": "page limits and PDF author metadata; content anonymity and visual layout require full review"}
