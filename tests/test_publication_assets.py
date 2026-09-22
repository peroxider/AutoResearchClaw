import copy
import json

import pytest

from researchclaw.literature.evidence import write_json
from researchclaw.literature.positioning import contribution_ledger
from researchclaw.pipeline.evidence_store import content_hash, file_hash
from researchclaw.pipeline.publication_assets import (
    AssetError, chart_bytes, chart_csv, content_spec, math_latex, polynomial_latex,
    prepare_assets, render_asset_sections, verify_assets,
)
from researchclaw.pipeline.research_workbench import compile_method, compile_theory
from researchclaw.pipeline.manuscript import build_manuscript, render_manuscript, validate_manuscript
from tests.test_research_inputs import inputs
from tests.test_experiment_protocol import spec
from tests.test_research_workbench import method, theory
from tests.test_manuscript import study, Writer


@pytest.mark.parametrize("expression", [
    r"\input{secrets}", r"\write18{echo bad}", r"\csname input\endcsname{secret}",
    r"^^5cinput{secret}", r"\frac{x}{y", "x% hidden", r"\begin{equation}x\end{equation}",
    r"\def\x{bad}", "x} + {y", "x & y", "$x$",
])
def test_math_rejects_commands_outside_bounded_equation_grammar(expression):
    with pytest.raises(AssetError):
        math_latex(expression)


def test_math_preserves_scientific_notation_without_evaluation():
    expression = r"\mathcal{L}=\frac{1}{n}\sum_{i=1}^{n}\left\Vert y_i-X_i W\right\Vert^2"
    assert math_latex(expression) == expression
    assert r"\frac{" in polynomial_latex("x/3 + x/6")
    with pytest.raises(AssetError):
        polynomial_latex("__import__('os').system('bad')")


def test_assets_retain_zero_effect_every_seed_and_complete_result_ids(study):
    root, _ = study
    report = prepare_assets(root)
    assert verify_assets(root) == report
    assert report["visual_review"] == "unavailable"
    figures = report["spec"]["figures"]
    assert len(figures) == 2 and all(f["mean_difference"] == 0 for f in figures)
    assert all({r["seed"] for r in f["rows"]} == {"7", "42"} for f in figures)
    assert all("baseline_result,candidate_result" in chart_csv(f) for f in figures)
    for figure in figures:
        assert (root / f"publication_assets/{figure['id']}.png").read_bytes().startswith(b"\x89PNG")
        assert (root / f"publication_assets/{figure['id']}.pdf").read_bytes().startswith(b"%PDF-")


def test_plots_are_deterministic_and_immune_to_global_style_changes(study):
    import matplotlib
    root, _ = study
    figure = content_spec(root)["figures"][0]
    first = chart_bytes(figure, "png"), chart_bytes(figure, "pdf")
    with matplotlib.rc_context({"axes.facecolor": "red", "font.size": 28}):
        second = chart_bytes(figure, "png"), chart_bytes(figure, "pdf")
    assert first == second


@pytest.mark.parametrize("extension", ["png", "pdf", "csv"])
def test_rehashed_output_cannot_override_plot_data_or_pixels(study, extension):
    root, _ = study
    report = prepare_assets(root)
    name = f"publication_assets/{report['spec']['figures'][0]['id']}.{extension}"
    (root / name).write_bytes(b"A fabricated chart with different values")
    report["outputs"][name] = file_hash(root / name)
    report.pop("version")
    report["version"] = content_hash(report)
    write_json(root / "publication_assets.json", report)
    with pytest.raises(AssetError):
        verify_assets(root)


def test_incomplete_protocol_does_not_produce_selected_seed_plot(study):
    root, _ = study
    path = root / "evidence_store.json"
    from researchclaw.pipeline.evidence_store import EvidenceStore
    store = EvidenceStore.from_dict(json.loads(path.read_text()))
    store.records.pop(next(iter(store.records)))
    write_json(path, store.to_dict())
    with pytest.raises(AssetError, match="incomplete protocol"):
        prepare_assets(root)


def test_method_and_proof_assets_share_exact_source_and_keep_unresolved_status(study, method):
    root, _ = study
    write_json(root / "method_spec.json", compile_method(method))
    proof = theory()
    proof["obligations"].append({"id": "generalization", "required": True, "depends_on": ["identity"],
        "assumptions": ["Data follows the training distribution"], "statement": {"kind": "informal", "text": "Generalization improves"},
        "proof_text": "An incomplete proof sketch; the central bound remains unestablished."})
    write_json(root / "theory_bundle.json", compile_theory(proof))
    contribution_ledger(root)
    report = build_manuscript(root, "Fixture study", llm=Writer())
    texts, _ = render_manuscript(root, report)
    assert r"\label{eq-projection}" in texts["paper.tex"]
    assert "Y = X W" in texts["paper.tex"] and "Y = X W" in texts["paper_final.md"]
    assert "model.py::Linear.predict" in texts["paper_final.md"]
    assert "generalization - unresolved" in texts["paper.tex"]
    assert "incomplete proof sketch" in texts["paper.tex"]
    assert r"\includegraphics" in texts["paper.tex"]
    assert "publication_assets/results-" in texts["paper_final.md"]
    changed = copy.deepcopy(method)
    changed["equations"][0]["latex"] = "Y = 2 X W"
    write_json(root / "method_spec.json", compile_method(changed))
    with pytest.raises(ValueError):
        validate_manuscript(root, report)


def test_label_and_direction_swaps_fail_even_after_rehash(study):
    root, _ = study
    report = prepare_assets(root)
    report["spec"]["figures"][0]["direction"] = "baseline_minus_candidate"
    report.pop("version")
    report["version"] = content_hash(report)
    write_json(root / "publication_assets.json", report)
    with pytest.raises(AssetError, match="specification"):
        verify_assets(root)


def test_real_independent_evaluation_preserves_negative_effects(study):
    import csv
    from researchclaw.pipeline.evidence_store import EvidenceStore
    from researchclaw.pipeline.independent_evaluator import evaluate_predictions
    root, _ = study
    old = EvidenceStore.from_dict(json.loads((root / "evidence_store.json").read_text()))
    changed = EvidenceStore()
    for record in old.records.values():
        if record.key.method != "A_full":
            changed.add(record)
            continue
        labels, predictions, execution = [name for name, _ in record.artifacts]
        with (root / labels).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        with (root / predictions).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["id", "prediction"])
            writer.writeheader()
            writer.writerows({"id": row["id"], "prediction": 1 - float(row["label"])} for row in rows)
        changed.add(evaluate_predictions(root=root, key=record.key, labels=labels, predictions=predictions,
                    execution=execution, expected_labels_sha256=file_hash(root / labels)))
    write_json(root / "evidence_store.json", changed.to_dict())
    report = prepare_assets(root)
    primary = next(f for f in report["spec"]["figures"] if f["question"] == "rq_main")
    assert primary["mean_difference"] == -1.0
    assert all(row["difference"] == -1.0 for row in primary["rows"])
    assert verify_assets(root) == report


def test_method_control_diagrams_are_part_of_exact_manuscript_exports(study, method):
    from tests.test_diagram_spec import looping
    from researchclaw.pipeline.manuscript import export_manuscript, verify_exports
    root, _ = study
    write_json(root / "method_spec.json", compile_method(looping(method)))
    contribution_ledger(root)
    build_manuscript(root, "Control-flow fixture", llm=Writer())
    export_manuscript(root, root)
    verify_exports(root)
    report = verify_assets(root)
    diagrams = report["spec"]["diagrams"]
    assert {d["kind"] for d in diagrams} == {"architecture", "execution"}
    md, tex = ((root / name).read_text(encoding="utf-8") for name in ("paper_final.md", "paper.tex"))
    for diagram in diagrams:
        assert f"publication_assets/{diagram['id']}.png" in md
        assert f"publication_assets/{diagram['id']}.pdf" in tex
    path = root / f"publication_assets/{diagrams[1]['id']}.pdf"
    path.write_bytes(b"different diagram")
    with pytest.raises(AssetError, match="Diagram"):
        verify_exports(root)
