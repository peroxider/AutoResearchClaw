import json
import zipfile

import pytest

from researchclaw.literature.evidence import write_json
from researchclaw.pipeline.evidence_store import content_hash
from researchclaw.templates.bundle import (
    GENERIC, TemplateError, copy_template_resources, freeze_template, inspect_constraints,
    render_frame, verify_template, verify_template_resources,
)
from researchclaw.pipeline.manuscript import build_manuscript, export_manuscript, render_manuscript, validate_manuscript, verify_exports
from tests.test_research_inputs import inputs
from tests.test_experiment_protocol import spec
from tests.test_manuscript import study, Writer


def bundle(tmp_path, **policy):
    directory = tmp_path / "template-source"
    directory.mkdir(exist_ok=True)
    (directory / "main.tex").write_text(r"""\documentclass[twocolumn]{article}
\usepackage{localstyle}
\title{A sample title}
\author{Real Person\thanks{An identifying grant}}
\begin{document}
\maketitle
EXAMPLE PAPER CONTENT MUST NOT SURVIVE
\bibliographystyle{abbrvnat}
\bibliography{sample}
\end{document}
""", encoding="utf-8")
    (directory / "localstyle.sty").write_text(r"\ProvidesPackage{localstyle}")
    write_json(directory / "template.json", {"schema_version": 1, "name": "fixture-journal", **policy})
    return directory


def test_raw_template_parsing_preserves_class_style_and_removes_sample(tmp_path):
    source = bundle(tmp_path)
    root = tmp_path / "run"
    report = freeze_template(root, source, authors="Private Name")
    assert verify_template(root) == report
    prefix, suffix, policy = render_frame(root, "A & B")
    assert r"\documentclass[twocolumn]{article}" in prefix
    assert "Private Name" not in prefix and "Real Person" not in prefix and "grant" not in prefix
    assert r"\author{Anonymous}" in prefix and r"A \& B" in prefix
    assert "EXAMPLE" not in prefix + suffix and "sample}" not in suffix
    assert r"\bibliographystyle{abbrvnat}" in suffix
    assert policy["columns"] == 2
    copy_template_resources(root, root)
    verify_template_resources(root)
    assert (root / "localstyle.sty").is_file()


def test_zip_import_and_content_addressed_history(tmp_path):
    source = bundle(tmp_path)
    archive = tmp_path / "template.zip"
    with zipfile.ZipFile(archive, "w") as output:
        for path in source.iterdir():
            output.write(path, path.name)
    root = tmp_path / "run"
    first = freeze_template(root, archive)
    (source / "localstyle.sty").write_text(r"\ProvidesPackage{localstyle}[v2]")
    second = freeze_template(root, source)
    assert first["directory"] != second["directory"]
    assert (root / first["directory"] / "localstyle.sty").read_text() == r"\ProvidesPackage{localstyle}"


@pytest.mark.parametrize("name", ["../escape.tex", "/absolute.tex", "C:/outside.tex", "styles/../../escape.tex", "CON.tex"])
def test_zip_path_escape_is_rejected_before_writing(tmp_path, name):
    archive = tmp_path / "template.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(name, "forbidden")
    with pytest.raises(TemplateError):
        freeze_template(tmp_path / "run", archive)
    assert not (tmp_path / "escape.tex").exists()


def test_frozen_and_materialized_style_changes_are_both_detected(tmp_path):
    root = tmp_path / "run"
    record = freeze_template(root, bundle(tmp_path))
    copy_template_resources(root, root)
    (root / "localstyle.sty").write_text("An altered style")
    with pytest.raises(TemplateError, match="Materialized"):
        verify_template_resources(root)
    (root / record["directory"] / "localstyle.sty").write_text("An altered frozen style")
    with pytest.raises(TemplateError, match="Frozen"):
        verify_template(root)


@pytest.mark.parametrize("change", [
    {"max_pages": 0}, {"max_pages": True}, {"engine": "sh"}, {"columns": 3},
    {"max_title_characters": 0}, {"max_abstract_characters": True}, {"max_figures": -1},
    {"max_tables": 1.5},
    {"max_references": 0}, {"max_references": 1.5},
    {"min_page_ink_percent": 0, "max_sparse_pages": 0},
    {"min_page_ink_percent": 101, "max_sparse_pages": 0},
    {"min_page_ink_percent": 5}, {"max_sparse_pages": 1},
    {"min_page_ink_percent": 5, "max_sparse_pages": -1},
    {"min_page_ink_percent": 5, "max_sparse_pages": True},
    {"max_float_reference_page_distance": -1},
    {"max_float_reference_page_distance": True},
    {"min_float_caption_characters": 0},
    {"min_float_caption_characters": True},
    {"min_float_reference_context_characters": -1},
    {"min_float_reference_context_characters": 1.5},
    {"required_sections": ["nonsense"]}, {"required_sections": ["methods", "methods"]},
    {"banned_sections": "methods"}, {"required_sections": [True]},
    {"appendix_roles": ["abstract"]}, {"anonymous": "yes"}, {"unknown_rule": True},
])
def test_invalid_policy_fields_are_not_silently_ignored(tmp_path, change):
    with pytest.raises(TemplateError):
        freeze_template(tmp_path / "run", bundle(tmp_path, **change))


def test_unsupported_bibliography_and_identifying_metadata_fail(tmp_path):
    source = bundle(tmp_path)
    path = source / "main.tex"
    path.write_text(GENERIC.replace("{{ARC_PACKAGES}}", r"\usepackage{biblatex}" + "\n{{ARC_PACKAGES}}"))
    with pytest.raises(TemplateError, match="BibLaTeX"):
        freeze_template(tmp_path / "run", source)
    path.write_text(GENERIC.replace("{{ARC_PACKAGES}}", r"\hypersetup{pdfauthor={Private Name}}" + "\n{{ARC_PACKAGES}}"))
    with pytest.raises(TemplateError, match="identifying metadata"):
        freeze_template(tmp_path / "run", source)


def test_biber_template_is_explicitly_normalized_without_bibtex_substitution(tmp_path):
    source = bundle(tmp_path, bibliography_backend="biber")
    (source / "main.tex").write_text(r"""\documentclass{article}
\usepackage[backend=biber,style=authoryear]{biblatex}
\addbibresource{sample.bib}
\title{Sample}\author{Example}
\begin{document}\maketitle SAMPLE BODY \printbibliography\end{document}
""", encoding="utf-8")
    (source / "localstyle.sty").unlink()
    (source / "localstyle.bbx").write_text(r"\ProvidesFile{localstyle.bbx}", encoding="utf-8")
    root = tmp_path / "run"
    report = freeze_template(root, source)
    assert report["compiled"]["policy"]["bibliography_backend"] == "biber"
    prefix, suffix, _ = render_frame(root, "Biber fixture")
    assert r"\usepackage[backend=biber,style=authoryear]{biblatex}" in prefix
    assert r"\addbibresource{references.bib}" in prefix
    assert r"\printbibliography" in suffix
    assert "sample.bib" not in prefix + suffix and r"\bibliographystyle" not in suffix
    assert r"\usepackage{natbib}" not in prefix
    copy_template_resources(root, root)
    assert (root / "localstyle.bbx").is_file()


def test_biber_policy_and_biblatex_declarations_cannot_disagree(tmp_path):
    with pytest.raises(TemplateError, match="selected together"):
        freeze_template(tmp_path / "run", bundle(tmp_path, bibliography_backend="biber"))


def test_compiler_routes_biblatex_to_biber_and_requires_its_output(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from researchclaw.templates import compiler
    tex = tmp_path / "paper.tex"
    tex.write_text(r"""\documentclass{article}\usepackage[backend=biber]{biblatex}
\addbibresource{references.bib}\begin{document}Text\printbibliography\end{document}""")
    (tmp_path / "references.bib").write_text("@article{x,title={X}}")
    calls = []
    monkeypatch.setattr(compiler.shutil, "which", lambda name: name)
    def latex(*args, **kwargs):
        (tmp_path / "paper.pdf").write_bytes(b"%PDF-fixture")
        return "", True
    monkeypatch.setattr(compiler, "_run_pdflatex", latex)
    def biber(work_dir, stem, timeout=60):
        calls.append((work_dir, stem))
        (work_dir / f"{stem}.bbl").write_text("fixture")
        return True
    monkeypatch.setattr(compiler, "_run_biber", biber)
    monkeypatch.setattr(compiler, "_run_bibtex", lambda *a, **k: pytest.fail("BibTeX substitution"))
    result = compiler.compile_latex(tex, max_attempts=1, allow_repairs=False)
    assert result.success and calls == [(tmp_path, "paper")]


def test_page_limits_use_physical_pdf_pages_not_tex_counters(tmp_path):
    fitz = pytest.importorskip("fitz")
    root = tmp_path / "run"
    freeze_template(root, bundle(tmp_path, max_pages=2, max_main_pages=1, appendix_roles=["results"]))
    with fitz.open() as pdf:
        pdf.new_page()
        pdf.new_page()
        pdf.set_metadata({"author": "Anonymous"})
        pdf.save(root / "paper.pdf")
    missing = inspect_constraints(root)
    assert "main_page_count_unavailable" in missing["issues"]
    with fitz.open(root / "paper.pdf") as pdf:
        pdf[1].insert_text((72, 72), "Appendix")
        pdf.saveIncr()
    checked = inspect_constraints(root)
    assert checked["status"] == "passed" and checked["pages"] == 2 and checked["main_pages"] == 1
    (root / "paper.aux").write_text(r"\newlabel{arc:appendix-start}{{}{99}}")
    assert inspect_constraints(root)["main_pages"] == 1
    with fitz.open(root / "paper.pdf") as pdf:
        pdf[0].insert_text((72, 72), "Appendix")
        pdf.saveIncr()
    assert "main_page_count_unavailable" in inspect_constraints(root)["issues"]


def test_pdf_author_metadata_cannot_pass_anonymous_policy(tmp_path):
    fitz = pytest.importorskip("fitz")
    root = tmp_path / "run"
    freeze_template(root)
    with fitz.open() as pdf:
        pdf.new_page()
        pdf.set_metadata({"author": "Private Name"})
        pdf.save(root / "paper.pdf")
    assert "pdf_author_metadata_not_anonymous" in inspect_constraints(root)["issues"]


def test_appendix_and_highlights_preserve_numeric_spans_in_both_formats(study):
    root, _ = study
    source = bundle(root.parent, appendix_roles=["results"], highlights_required=True)
    freeze_template(root, source)
    ir = build_manuscript(root, "Fixture study", llm=Writer())
    export_manuscript(root, root)
    verify_exports(root)
    texts, bindings = render_manuscript(root, ir)
    assert texts["paper_final.md"].index("# Appendix") > texts["paper_final.md"].index("## Conclusion")
    assert r"\begin{table*}" in texts["paper.tex"] and r"\begin{longtable}" not in texts["paper.tex"]
    assert r"\begin{figure*}" in texts["paper.tex"]
    assert r"\ref{fig:" in texts["paper.tex"] and r"\label{arc-ref:fig:" in texts["paper.tex"]
    assert r"\ref{arc-results-1}\label{arc-ref:arc-results-1}" in texts["paper.tex"]
    assert "zero difference" in (root / "highlights.md").read_text(encoding="utf-8")
    for claim in bindings["numeric"]:
        for name, span in claim["spans"].items():
            assert texts[name][span["start"]:span["end"]] == claim["rendered"]
    (root / "highlights.md").write_text("An unsupported highlight")
    with pytest.raises(ValueError, match="Auxiliary"):
        verify_exports(root)


def test_template_refresh_invalidates_old_manuscript_and_removes_only_managed_highlights(study):
    root, _ = study
    source = bundle(root.parent, highlights_required=True)
    freeze_template(root, source)
    old = build_manuscript(root, "Fixture study", llm=Writer())
    export_manuscript(root, root)
    source = bundle(root.parent, highlights_required=False, max_pages=12)
    freeze_template(root, source)
    with pytest.raises(ValueError, match="dependencies"):
        validate_manuscript(root, old)
    build_manuscript(root, "Fixture study", llm=Writer())
    export_manuscript(root, root)
    assert not (root / "highlights.md").exists()


def test_xelatex_invocation_disables_shell_escape(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from researchclaw.templates import compiler
    calls = []
    monkeypatch.setattr(compiler.subprocess, "run", lambda command, **kwargs: (
        calls.append(command) or SimpleNamespace(stdout=b"", stderr=b"", returncode=0)))
    _, success = compiler._run_pdflatex(tmp_path, "paper.tex", engine="xelatex")
    assert success and calls[0][:2] == ["xelatex", "-no-shell-escape"]


def test_config_template_paths_resolve_from_project_root_and_validate(tmp_path):
    from researchclaw.config import RCConfig, validate_config
    source = bundle(tmp_path)
    data = {"project": {"name": "template-test"}, "research": {"topic": "Fixture"},
        "runtime": {"timezone": "UTC"}, "notifications": {"channel": "local"},
        "knowledge_base": {"root": str(tmp_path / "kb")},
        "llm": {"provider": "openai-compatible", "base_url": "http://localhost:1234/v1", "api_key": "fixture", "api_key_env": "RC_TEST_KEY"},
        "export": {"template_path": "template-source"}}
    cfg = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)
    assert cfg.export.template_path == str(source.resolve())
    invalid = validate_config({"export": {"template_path": 123}}, check_paths=False)
    assert any("export.template_path" in error for error in invalid.errors)


def test_template_with_exact_markers_preserves_special_title_layout(tmp_path):
    source = bundle(tmp_path, anonymous=False, engine="xelatex")
    raw = GENERIC.replace(r"\maketitle", r"\begin{center}\Huge Custom journal heading\end{center}\maketitle")
    (source / "main.tex").write_text(raw)
    root = tmp_path / "run"
    freeze_template(root, source, authors="A & B")
    prefix, _, policy = render_frame(root, "Title")
    assert "Custom journal heading" in prefix and r"A \& B" in prefix
    assert "inputenc" not in prefix and policy["engine"] == "xelatex"


def test_pdf_total_limit_is_enforced_even_without_appendix(tmp_path):
    fitz = pytest.importorskip("fitz")
    root = tmp_path / "run"
    freeze_template(root, bundle(tmp_path, max_pages=1))
    with fitz.open() as pdf:
        pdf.new_page()
        pdf.new_page()
        pdf.save(root / "paper.pdf")
    result = inspect_constraints(root)
    assert result["pages"] == 2 and "max_pages_unavailable_or_exceeded" in result["issues"]


def test_declared_title_abstract_figure_and_table_limits_are_recomputed(tmp_path):
    import fitz
    root = tmp_path / "run"
    freeze_template(root, bundle(tmp_path, max_title_characters=5, max_abstract_characters=5,
                                 max_figures=1, max_tables=1))
    with pytest.raises(TemplateError, match="title exceeds"):
        render_frame(root, "Six chars")
    write_json(root / "manuscript_ir.json", {"title": "Title", "sections": [
        {"task": {"role": "abstract"}, "blocks": [{"text": "123456"}]}]})
    (root / "paper.tex").write_text(
        r"\begin{figure}\end{figure}\begin{figure*}\end{figure*}"
        r"\begin{table}\end{table}\begin{longtable}{l}\end{longtable}"
        "% \\begin{figure} ignored comment\n", encoding="utf-8")
    with fitz.open() as pdf:
        pdf.new_page()
        pdf.save(root / "paper.pdf")
    result = inspect_constraints(root)
    assert result["title_characters"] == 5 and result["abstract_characters"] == 6
    assert result["figures"] == 2 and result["tables"] == 2
    assert {"max_abstract_characters_unavailable_or_exceeded", "max_figures_unavailable_or_exceeded",
            "max_tables_unavailable_or_exceeded"} <= set(result["issues"])


def test_declared_section_and_reference_rules_are_recomputed(tmp_path):
    import fitz
    root = tmp_path / "run"
    freeze_template(root, bundle(tmp_path, required_sections=["methods", "results"],
                                 banned_sections=["theory"], max_references=2))
    # No IR yet: declared section rules must fail closed, not pass vacuously.
    assert "section_rules_unavailable" in inspect_constraints(root)["issues"]
    write_json(root / "manuscript_ir.json", {"title": "Title", "sections": [
        {"task": {"role": "abstract"}, "blocks": [{"text": "x"}]},
        {"task": {"role": "results"}, "blocks": []},
        {"task": {"role": "theory"}, "blocks": []}]})
    (root / "references.bib").write_text(
        "@article{a2024,\n  title = {A},\n}\n"
        "@InProceedings{b2024,\n  title = {B},\n}\n"
        "@string{venue = {Journal}}\n@preamble{\"\"}\n@comment{not an entry}\n",
        encoding="utf-8")
    with fitz.open() as pdf:
        pdf.new_page()
        pdf.save(root / "paper.pdf")
    result = inspect_constraints(root)
    assert result["sections_present"] == ["abstract", "results", "theory"]
    assert result["required_sections_missing"] == ["methods"]
    assert result["banned_sections_present"] == ["theory"]
    assert result["references"] == 2
    assert {"required_sections_missing", "banned_sections_present"} <= set(result["issues"])
    write_json(root / "manuscript_ir.json", {"title": "Title", "sections": [
        {"task": {"role": "abstract"}, "blocks": [{"text": "x"}]},
        {"task": {"role": "methods"}, "blocks": []},
        {"task": {"role": "results"}, "blocks": []},
        {"task": {"role": "discussion"}, "blocks": []}]})
    result = inspect_constraints(root)
    assert result["status"] == "passed"
    assert result["required_sections_missing"] == [] and result["banned_sections_present"] == []
    (root / "references.bib").unlink()
    assert "max_references_unavailable_or_exceeded" in inspect_constraints(root)["issues"]


def test_declared_sparse_page_budget_is_recomputed_from_final_pdf_pixels(tmp_path):
    import fitz
    root = tmp_path / "run"
    freeze_template(root, bundle(tmp_path, min_page_ink_percent=5, max_sparse_pages=1))
    missing = inspect_constraints(root)
    assert "pdf_sparse_page_analysis_unavailable" in missing["issues"]

    def write_pdf(dense_pages):
        target = root / "paper.pdf"
        if target.exists():
            target.unlink()
        with fitz.open() as pdf:
            for index in range(3):
                page = pdf.new_page()
                if index in dense_pages:
                    page.draw_rect(fitz.Rect(50, 50, 545, 790), color=(0, 0, 0), fill=(0, 0, 0))
            pdf.save(target)

    write_pdf({0})
    failed = inspect_constraints(root)
    assert len(failed["page_ink_percent"]) == 3
    assert failed["page_ink_percent"][0] > 5
    assert failed["page_ink_percent"][1:] == [0.0, 0.0]
    assert failed["sparse_pages"] == [2, 3]
    assert "max_sparse_pages_exceeded" in failed["issues"]

    write_pdf({0, 2})
    passed = inspect_constraints(root)
    assert passed["status"] == "passed"
    assert passed["sparse_pages"] == [2]


def test_declared_float_reference_distance_uses_compiled_page_anchors(tmp_path):
    import fitz
    root = tmp_path / "run"
    freeze_template(root, bundle(tmp_path, max_float_reference_page_distance=1))
    (root / "paper.tex").write_text(
        r"See Figure~\ref{fig:one}\label{arc-ref:fig:one}."
        r"\begin{figure}\caption{One}\label{fig:one}\end{figure}", encoding="utf-8")
    with fitz.open() as pdf:
        for _ in range(4):
            pdf.new_page()
        pdf.save(root / "paper.pdf")

    assert "float_reference_analysis_unavailable" in inspect_constraints(root)["issues"]
    (root / "paper.tex").write_text(
        r"See a figure.\begin{figure}\caption{Unlabelled}\end{figure}", encoding="utf-8")
    assert "float_reference_analysis_unavailable" in inspect_constraints(root)["issues"]
    (root / "paper.tex").write_text(
        r"See Figure~\ref{fig:one}\label{arc-ref:fig:one}."
        r"\begin{figure}\caption{One}\label{fig:one}\end{figure}", encoding="utf-8")
    (root / "paper.aux").write_text(
        r"\newlabel{arc-ref:fig:one}{{1}{1}}" "\n"
        r"\newlabel{fig:one}{{1}{3}}" "\n", encoding="utf-8")
    failed = inspect_constraints(root)
    assert failed["float_reference_distances"] == [{
        "label": "fig:one", "float_page": 3,
        "reference_pages": [1], "min_page_distance": 2}]
    assert "max_float_reference_page_distance_exceeded" in failed["issues"]

    (root / "paper.aux").write_text(
        r"\newlabel{arc-ref:fig:one}{{1}{2}}" "\n"
        r"\newlabel{fig:one}{{1}{3}}" "\n", encoding="utf-8")
    passed = inspect_constraints(root)
    assert passed["status"] == "passed"
    assert passed["float_reference_distances"][0]["min_page_distance"] == 1


def test_float_distance_and_reference_count_policies_compose(tmp_path: Path) -> None:
    root = tmp_path / "delivery"
    root.mkdir()
    freeze_template(root, bundle(
        tmp_path, max_float_reference_page_distance=1, max_references=2,
        min_float_caption_characters=1,
        min_float_reference_context_characters=5))
    import fitz
    with fitz.open() as pdf:
        pdf.new_page()
        pdf.new_page()
        pdf.save(root / "paper.pdf")
    (root / "references.bib").write_text(
        "@article{a,title={A}}\n@article{b,title={B}}\n", encoding="utf-8")
    (root / "paper.tex").write_text(
        r"See Figure~\ref{fig:x}\label{arc-ref:fig:x}."
        "\n" r"\begin{figure}\caption{X}\label{fig:x}\end{figure}", encoding="utf-8")
    (root / "paper.aux").write_text(
        r"\newlabel{arc-ref:fig:x}{{}{1}}" "\n" r"\newlabel{fig:x}{{1}{2}}",
        encoding="utf-8")
    result = inspect_constraints(root)
    assert result["status"] == "passed"
    assert result["references"] == 2
    assert result["float_reference_distances"][0]["min_page_distance"] == 1
    assert result["float_semantic_checks"][0]["caption_characters"] == 1


def test_declared_float_caption_and_reference_context_are_measured(tmp_path: Path) -> None:
    root = tmp_path / "delivery"
    root.mkdir()
    freeze_template(root, bundle(
        tmp_path, min_float_caption_characters=20,
        min_float_reference_context_characters=30))
    import fitz
    with fitz.open() as pdf:
        pdf.new_page()
        pdf.save(root / "paper.pdf")
    (root / "paper.tex").write_text(
        "The comparison exposes the stability mechanism under severe shift; see "
        r"Figure~\ref{fig:x}\label{arc-ref:fig:x}."
        "\n\n"
        r"\begin{figure}\caption{Stability under \textbf{severe distribution shift}}"
        r"\label{fig:x}\end{figure}", encoding="utf-8")
    result = inspect_constraints(root)
    assert result["status"] == "passed"
    assert result["float_semantic_checks"] == [{
        "label": "fig:x",
        "caption_characters": len("Stabilityunderseveredistributionshift"),
        "reference_context_characters": [
            len("Thecomparisonexposesthestabilitymechanismundersevereshift;seeFigure~.")],
        "max_reference_context_characters": len(
            "Thecomparisonexposesthestabilitymechanismundersevereshift;seeFigure~."),
    }]


def test_float_semantic_policy_rejects_thin_or_unparseable_context(tmp_path: Path) -> None:
    root = tmp_path / "delivery"
    root.mkdir()
    freeze_template(root, bundle(
        tmp_path, min_float_caption_characters=12,
        min_float_reference_context_characters=12))
    import fitz
    with fitz.open() as pdf:
        pdf.new_page()
        pdf.save(root / "paper.pdf")
    (root / "paper.tex").write_text(
        r"See \ref{fig:x}\label{arc-ref:fig:x}."
        "\n\n" r"\begin{figure}\caption{Tiny}\label{fig:x}\end{figure}",
        encoding="utf-8")
    thin = inspect_constraints(root)
    assert "min_float_caption_characters_unmet" in thin["issues"]
    assert "min_float_reference_context_characters_unmet" in thin["issues"]

    (root / "paper.tex").write_text(
        r"Context without the required marker."
        "\n\n" r"\begin{figure}\caption{A sufficiently detailed caption}\label{fig:x}\end{figure}",
        encoding="utf-8")
    unavailable = inspect_constraints(root)
    assert "float_semantic_analysis_unavailable" in unavailable["issues"]


def test_float_caption_and_reference_context_policies_are_independent(tmp_path: Path) -> None:
    import fitz

    caption_root = tmp_path / "caption"
    caption_root.mkdir()
    freeze_template(caption_root, bundle(tmp_path, min_float_caption_characters=4))
    with fitz.open() as pdf:
        pdf.new_page()
        pdf.save(caption_root / "paper.pdf")
    (caption_root / "paper.tex").write_text(
        r"\begin{figure}\caption{Enough}\label{fig:x}\end{figure}", encoding="utf-8")
    assert inspect_constraints(caption_root)["status"] == "passed"

    context_root = tmp_path / "context"
    context_root.mkdir()
    freeze_template(context_root, bundle(tmp_path, min_float_reference_context_characters=10))
    with fitz.open() as pdf:
        pdf.new_page()
        pdf.save(context_root / "paper.pdf")
    (context_root / "paper.tex").write_text(
        r"Detailed mechanism discussion precedes Figure~\ref{fig:x}\label{arc-ref:fig:x}."
        "\n\n" r"\begin{figure}\label{fig:x}\end{figure}", encoding="utf-8")
    result = inspect_constraints(context_root)
    assert result["status"] == "passed"
    assert result["float_semantic_checks"][0]["caption_characters"] is None


def test_export_refresh_detects_external_template_change(study):
    from dataclasses import replace
    from researchclaw.adapters import AdapterBundle
    from researchclaw.pipeline.stage_impls._review_publish import _execute_export_publish
    from researchclaw.pipeline.stages import StageStatus
    root, cfg = study
    source = bundle(root.parent)
    freeze_template(root, source)
    build_manuscript(root, "Fixture study", llm=Writer())
    cfg = replace(cfg, export=replace(cfg.export, template_path=str(source)))
    stage = root / "stage-22"
    stage.mkdir()
    write_json(source / "template.json", {"max_pages": 12})
    result = _execute_export_publish(stage, root, cfg, AdapterBundle())
    assert result.status == StageStatus.FAILED and "dependencies" in result.error
