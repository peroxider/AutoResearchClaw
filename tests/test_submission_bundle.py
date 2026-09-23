import json
import zipfile

import pytest

from researchclaw.literature.evidence import write_json
from researchclaw.pipeline.evidence_store import content_hash, file_hash
from researchclaw.pipeline.final_acceptance import compilation_inputs, assess_delivery, inventory
from researchclaw.pipeline.manuscript import build_manuscript, export_manuscript
from researchclaw.pipeline.submission_bundle import prepare_submission, verify_submission, SubmissionError
from researchclaw.templates.bundle import freeze_template
from tests.test_research_inputs import inputs
from tests.test_experiment_protocol import spec
from tests.test_manuscript import study, Writer
from tests.test_template_bundle import bundle


def bind_compilation(root):
    write_json(root / "compilation.json", {"success": True, "inputs": compilation_inputs(root),
               "pdf_sha256": file_hash(root / "paper.pdf"), "recorder_sha256": file_hash(root / "paper.fls")})


@pytest.fixture
def submission(study):
    import fitz
    root, _ = study
    source = bundle(root)
    (source / "unused.tex").write_text("PRIVATE AUTHOR IDENTITY IN TEMPLATE SAMPLE")
    (source / "sample.bib").write_text("PRIVATE SAMPLE BIBLIOGRAPHY")
    (source / "localrefs.bst").write_text("% fixture style supplied to the bibliography process")
    (source / "main.tex").write_text((source / "main.tex").read_text().replace("abbrvnat", "localrefs"))
    freeze_template(root, source)
    build_manuscript(root, "Submission fixture", llm=Writer())
    export_manuscript(root, root)
    (root / "references.bib").write_text("@article{smith2024,title={Fixture}}")
    (root / "paper.bbl").write_text(r"\begin{thebibliography}{1}\bibitem{smith2024}Fixture\end{thebibliography}")
    pdf = fitz.open()
    pdf.new_page().insert_text((72, 72), "Anonymous submission fixture")
    pdf.save(root / "paper.pdf")
    pdf.close()
    assets = json.loads((root / "publication_assets.json").read_text())
    recorded = ["paper.tex", "paper.bbl", "localstyle.sty", "paper.aux"]
    recorded.extend(name for name in assets["outputs"] if name.endswith(".pdf"))
    (root / "paper.aux").write_text("private compilation bookkeeping")
    (root / "paper.fls").write_text("\n".join("INPUT " + name for name in recorded))
    bind_compilation(root)
    return root


def test_archive_excludes_private_sources_unused_templates_and_compiler_metadata(submission):
    root = submission
    report = prepare_submission(root)
    assert report["status"] == "prepared", report
    assert verify_submission(root) == report
    with zipfile.ZipFile(root / "submission.zip") as archive:
        names = archive.namelist()
        assert {"paper.tex", "paper.pdf", "references.bib", "paper.bbl", "localstyle.sty", "localrefs.bst"} <= set(names)
        assert not any(name.startswith(("research_inputs/", "publication_templates/", "evidence_artifacts/")) for name in names)
        assert not set(names) & {"unused.tex", "sample.bib", "main.tex", "paper.aux", "paper.fls", "paper_final.md"}
        assert all("PRIVATE" not in archive.read(name).decode(errors="ignore") for name in names)
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
    first = (root / "submission.zip").read_bytes()
    assert prepare_submission(root) == report
    assert (root / "submission.zip").read_bytes() == first


@pytest.mark.parametrize("change", [
    lambda root: (root / "paper.fls").unlink(),
    lambda root: (root / "paper.fls").write_text("INPUT paper.tex"),
    lambda root: (root / "paper.pdf").write_bytes(b"%PDF-broken"),
    lambda root: (root / "paper.tex").write_text("Unreviewed manuscript"),
    lambda root: (root / "paper.bbl").write_text("Changed bibliography"),
])
def test_stale_or_missing_compilation_invalidates_and_removes_old_archive(submission, change):
    root = submission
    assert prepare_submission(root)["status"] == "prepared"
    change(root)
    report = prepare_submission(root)
    assert report["status"] == "failed" and not (root / "submission.zip").exists()


def test_recorder_cannot_publish_private_evidence_even_with_new_hash(submission):
    root = submission
    private = root / "research_inputs" / "private.csv"
    private.parent.mkdir(exist_ok=True)
    private.write_text("patient,name")
    with (root / "paper.fls").open("a") as stream:
        stream.write("\nINPUT research_inputs/private.csv\n")
    bind_compilation(root)
    report = prepare_submission(root)
    assert report["status"] == "failed" and "unapproved" in report["issues"][0]


def test_biblatex_backend_handoff_is_not_a_publication_input(submission):
    # biblatex re-opens paper.run.xml (its biber hand-off) within the recorded
    # pass; the derived file must not block preparation nor enter the archive.
    root = submission
    (root / "paper.run.xml").write_text(
        "<biblatex>Biber hand-off derived from paper.tex and references.bib</biblatex>")
    with (root / "paper.fls").open("a") as stream:
        stream.write("\nINPUT paper.run.xml\nOUTPUT paper.run.xml\n")
    bind_compilation(root)
    report = prepare_submission(root)
    assert report["status"] == "prepared", report
    assert verify_submission(root) == report
    with zipfile.ZipFile(root / "submission.zip") as archive:
        assert "paper.run.xml" not in archive.namelist()


def test_rehashed_archive_cannot_smuggle_extra_files(submission):
    root = submission
    report = prepare_submission(root)
    with zipfile.ZipFile(root / "submission.zip", "a") as archive:
        archive.writestr("author-identity.txt", "Private author")
    report["archive_sha256"] = file_hash(root / "submission.zip")
    report.pop("version")
    report["version"] = content_hash(report)
    write_json(root / "submission_bundle.json", report)
    with pytest.raises(SubmissionError, match="exact allowed payload"):
        verify_submission(root)


def test_embedded_pdf_attachment_is_not_released(submission):
    import fitz
    root = submission
    with fitz.open(root / "paper.pdf") as doc:
        doc.embfile_add("private.txt", b"private author information")
        doc.saveIncr()
    bind_compilation(root)
    report = prepare_submission(root)
    assert report["status"] == "failed" and "embedded" in report["issues"][0]


def test_anonymous_archive_requires_distinct_bound_content_review(submission):
    root = submission
    prepare_submission(root)
    report = assess_delivery(root)
    assert report["dimensions"]["anonymity"] == "unknown"
    review = {"input_version": content_hash(inventory(root)), "dimensions": {
        "anonymity": {"status": "passed", "checker": "fixture-human", "evidence": "Complete archive inspection"}}}
    write_json(root / "final_reviews.json", review)
    assert assess_delivery(root)["dimensions"]["anonymity"] == "passed"
    (root / "extra-file.txt").write_text("changed bundle")
    assert assess_delivery(root)["dimensions"]["anonymity"] == "unknown"


def test_external_runtime_paths_do_not_enter_archive(submission, tmp_path):
    root = submission
    runtime = tmp_path / "runtime" / "article.cls"
    runtime.parent.mkdir()
    runtime.write_text("fixture external TeX runtime")
    # Use a sibling directory outside the delivery root.
    external = root.parent / "runtime-external.cls"
    external.write_text("external runtime")
    with (root / "paper.fls").open("a") as stream:
        stream.write("\nINPUT " + str(external.resolve()) + "\n")
    bind_compilation(root)
    report = prepare_submission(root)
    assert report["status"] == "prepared"
    assert "runtime-external" not in (root / "submission.zip").read_bytes().decode(errors="ignore")


def test_relative_recorder_escape_is_rejected(submission):
    root = submission
    with (root / "paper.fls").open("a") as stream:
        stream.write("\nINPUT ../private.tex\n")
    bind_compilation(root)
    report = prepare_submission(root)
    assert report["status"] == "failed" and "escapes" in report["issues"][0]
