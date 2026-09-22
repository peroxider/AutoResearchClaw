"""Opt-in integration against installed TeX binaries; no model/network calls.

ARC_RUN_TEX_INTEGRATION=1 python -m pytest tests/test_real_publication_compile.py
The fixture paper exercises the pipeline, not scientific quality.
"""
import json
import os
import re
import shutil
import zipfile

import pytest

from researchclaw.literature.evidence import write_json
from researchclaw.literature.positioning import contribution_ledger
from researchclaw.pipeline.manuscript import build_manuscript, export_manuscript, package_manuscript, quality_report
from researchclaw.pipeline.research_workbench import compile_method, compile_theory
from researchclaw.pipeline.submission_bundle import verify_submission
from researchclaw.templates.bundle import freeze_template
from tests.test_research_inputs import inputs
from tests.test_experiment_protocol import spec
from tests.test_research_workbench import method, theory
from tests.test_diagram_spec import looping
from tests.test_manuscript import study, Writer
from tests.test_template_bundle import bundle


@pytest.mark.parametrize("engine,columns", [("pdflatex", 1), ("pdflatex", 2), ("xelatex", 1), ("xelatex", 2)])
def test_real_template_diagrams_math_bibliography_and_minimal_archive(study, method, engine, columns):
    if os.environ.get("ARC_RUN_TEX_INTEGRATION") != "1" or not shutil.which(engine) or not shutil.which("bibtex"):
        pytest.skip("Opt-in real TeX compilation requires pdflatex/xelatex and bibtex")
    import fitz
    root, cfg = study
    source = bundle(root, engine=engine, columns=columns, appendix_roles=["theory"], highlights_required=True)
    entry = source / "main.tex"
    if columns == 1:
        entry.write_text(entry.read_text(encoding="utf-8").replace("[twocolumn]", ""), encoding="utf-8")
    freeze_template(root, source)
    write_json(root / "method_spec.json", compile_method(looping(method)))
    write_json(root / "theory_bundle.json", compile_theory(theory()))
    contribution_ledger(root)
    build_manuscript(root, "Reproducible publication integration fixture", llm=Writer())
    for number in (20, 22, 23):
        (root / f"stage-{number}").mkdir()
    export_manuscript(root, root / "stage-22")
    (root / "stage-23/paper_final_verified.md").write_bytes((root / "stage-22/paper_final.md").read_bytes())
    (root / "stage-23/references_verified.bib").write_text(
        "@article{smith2024,author={Smith, Alex},title={Fixture intervention},journal={Fixture Journal},year={2024}}")
    write_json(root / "stage-23/verification_report.json", {"status": "verified"})
    write_json(root / "stage-20/quality_report.json", quality_report(root, cfg.research.quality_threshold))
    dest = package_manuscript(root, "real-tex-fixture", cfg)
    compilation = json.loads((dest / "compilation.json").read_text())
    assert compilation["success"], compilation
    payload = json.loads((dest / "submission_bundle.json").read_text())
    assert payload["status"] == "prepared", payload
    verify_submission(dest)
    with fitz.open(dest / "paper.pdf") as pdf:
        text = "\n".join(page.get_text() for page in pdf)
        assert pdf.page_count >= 2
        for label in ("Methods", "Results", "Conclusion", "References", "Appendix", "Another batch available?"):
            assert label in text
        assert "??" not in text
        assert "evidence group" not in text
        # Every result row stays attached to its method, config and seed. Tables
        # on another page still provide their complete context and legend.
        claims = json.loads((dest / "numeric_claims.json").read_text())["claims"]
        assert text.count("1.000000") == len(claims)
        expected_rows = {(c["key"]["method"], c["configuration_alias"], c["key"]["seed"]) for c in claims}
        for page in pdf:
            page_text = re.sub(r"\s+", " ", page.get_text().replace("\x1c", "fi").replace("-\n", ""))
            if "1.000000" in page_text:
                assert "Configurations:" in page_text and "dataset version:" in page_text
                words = page.get_text("words")
                for value in [w for w in words if w[4] == "1.000000"]:
                    row = sorted((w for w in words if abs(w[1] - value[1]) < 2), key=lambda w: w[0])
                    labels = [w[4].replace("\x1b", "ff") for w in row]
                    assert tuple(labels[:-1]) in expected_rows, labels
        spans = [span for page in pdf for block in page.get_text("dict")["blocks"] if "lines" in block
                 for line in block["lines"] for span in line["spans"]]
        assert min(span["size"] for span in spans) >= 6.9  # Seven-point math subscripts are intentional.
        def positions(label):
            return [(page.number, rect.y0) for page in pdf for rect in page.search_for(label)]

        # Float boundaries are semantic: numerical results precede Discussion,
        # and the method execution diagram precedes the Experiments section.
        assert max(positions("1.000000")) < min(positions("Discussion"))
        assert max(positions("Another batch available?")) < min(positions("Experiments"))
    log = (dest / "paper.log").read_text(encoding="utf-8", errors="replace")
    assert r"\item{} [inference]" in (dest / "paper.tex").read_text(encoding="utf-8")
    assert not any(float(size) > 1 for size in re.findall(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", log))
    with zipfile.ZipFile(dest / "submission.zip") as archive:
        assert "highlights.md" in archive.namelist()
        assert not any(name.startswith(("research_inputs/", "publication_templates/")) for name in archive.namelist())
        rebuilt = root / "archive-rebuild"
        for name in archive.namelist():
            target = rebuilt / name
            assert target.resolve().is_relative_to(rebuilt.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    from researchclaw.templates.compiler import compile_latex
    result = compile_latex(rebuilt / "paper.tex", engine=engine, allow_repairs=False, max_attempts=1)
    assert result.success, result.errors
    with fitz.open(rebuilt / "paper.pdf") as pdf:
        assert "\n".join(page.get_text() for page in pdf) == text
    # Compiling a fixture does not certify research validity or anonymous content.
    assert json.loads((dest / "final_acceptance.json").read_text())["artifact_status"] != "submission_candidate"


@pytest.mark.parametrize("engine", ["pdflatex", "xelatex"])
def test_real_effect_intervals_render_from_frozen_analysis(inputs, spec, engine):
    if os.environ.get("ARC_RUN_TEX_INTEGRATION") != "1" or not shutil.which(engine) or not shutil.which("bibtex"):
        pytest.skip("Opt-in real TeX compilation requires pdflatex/xelatex and bibtex")
    import fitz
    from tests.test_analysis_spec import varied_study
    from researchclaw.pipeline.analysis_spec import verify_analysis
    from researchclaw.pipeline.publication_assets import prepare_assets, render_asset_sections
    from researchclaw.templates.bundle import copy_template_resources, render_frame
    from researchclaw.templates.compiler import compile_latex
    root, _ = varied_study(inputs, spec)
    assets = prepare_assets(root)
    source = bundle(root, engine=engine, columns=1)
    entry = source / "main.tex"
    entry.write_text(entry.read_text(encoding="utf-8").replace("[twocolumn]", ""), encoding="utf-8")
    freeze_template(root, source)
    copy_template_resources(root, root)
    prefix, suffix, _ = render_frame(root, "Conditional paired-seed interval fixture")
    _, fragment = render_asset_sections(root)["results"]
    (root / "paper.tex").write_text(prefix + "\\section{Results}\nFixture observations \\cite{smith2024}.\n"
                                   + fragment + "\\FloatBarrier\n" + suffix, encoding="utf-8")
    (root / "references.bib").write_text(
        "@article{smith2024,author={Smith, Alex},title={Fixture},journal={Fixture Journal},year={2024}}")
    result = compile_latex(root / "paper.tex", engine=engine, allow_repairs=False, max_attempts=1)
    assert result.success, result.errors
    analysis = verify_analysis(root)
    assert all(a["statistics"]["interval"]["status"] == "computed_conditional" for a in analysis["analyses"])
    assert len([f for f in assets["spec"]["figures"] if f["kind"] == "effect_summary"]) == 2
    with fitz.open(root / "paper.pdf") as pdf:
        text = "\n".join(page.get_text() for page in pdf)
        assert text.count("paired effects") >= 2
        assert "95% interval" in text and "Conditional seed-mean interval" in text
        assert "??" not in text
        spans = [s for p in pdf for b in p.get_text("dict")["blocks"] if "lines" in b
                 for line in b["lines"] for s in line["spans"]]
        assert min(s["size"] for s in spans) >= 6.9
    log = (root / "paper.log").read_text(encoding="utf-8", errors="replace")
    assert not any(float(size) > 1 for size in re.findall(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", log))
