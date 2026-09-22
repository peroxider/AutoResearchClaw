import copy
import json
import xml.etree.ElementTree as ET

import pytest

from researchclaw.literature.evidence import write_json
from researchclaw.pipeline.diagram_spec import (
    DiagramError, diagram_bytes, method_diagrams, validate_diagram,
)
from researchclaw.pipeline.evidence_store import content_hash, file_hash
from researchclaw.pipeline.publication_assets import AssetError, prepare_assets, verify_assets
from researchclaw.pipeline.research_workbench import WorkbenchError, compile_method
from tests.test_research_workbench import method


def looping(method):
    method = copy.deepcopy(method)
    method["control_flow"] = {"entry": "run", "nodes": [
        {"id": "run", "kind": "step", "step": "predict"},
        {"id": "check", "kind": "decision", "phase": "inference", "condition": "Another batch available?"},
        {"id": "done", "kind": "stop", "phase": "inference"}], "edges": [
        {"source": "run", "target": "check", "branch": "next", "loop": False},
        {"source": "check", "target": "run", "branch": "true", "loop": True},
        {"source": "check", "target": "done", "branch": "false", "loop": False}]}
    method["stopping_rule"] = "Stop when no more batches are available"
    return method


def test_source_has_exact_ports_equations_groups_without_domain_defaults(method):
    doc = compile_method(method)
    diagram, = method_diagrams(doc)
    assert diagram["source_method_version"] == doc["version"]
    assert diagram["groups"] == [{"id": "inference", "nodes": ["predict"]}]
    assert diagram["nodes"][0]["ports"] == [
        {"id": "in-X", "direction": "input", "variable": "X"},
        {"id": "in-W", "direction": "input", "variable": "W"},
        {"id": "out-Y", "direction": "output", "variable": "Y"}]
    assert diagram["nodes"][0]["equation_refs"] == ["projection"]
    assert diagram["edges"] == []
    assert "clinical" not in json.dumps(diagram).lower()
    assert validate_diagram(diagram, doc) == diagram


def test_explicit_loop_keeps_true_false_stop_and_unknown_termination(method):
    doc = compile_method(looping(method))
    architecture, flow = method_diagrams(doc)
    assert architecture["kind"] == "architecture" and architecture["edges"] == []
    assert flow["kind"] == "execution" and flow["entry"] == "run"
    assert flow["termination"] == "unproved"
    assert {e["label"] for e in flow["edges"]} == {"true", "false", "next"}
    assert next(e for e in flow["edges"] if e["loop"])["target"] == "run"
    assert flow["stopping_rule"] == doc["spec"]["stopping_rule"]


@pytest.mark.parametrize("change", [
    lambda f: f.update(entry="absent"),
    lambda f: f["nodes"][0].update(step="unknown"),
    lambda f: f["nodes"][1].update(condition=""),
    lambda f: f["nodes"][1].update(phase="both"),
    lambda f: f["nodes"].append(copy.deepcopy(f["nodes"][0])),
    lambda f: f["nodes"].append({"id": "ghost", "kind": "stop", "phase": "train"}),
    lambda f: f["edges"].pop(),
    lambda f: f["edges"][1].update(loop=False),
    lambda f: f["edges"][1].update(loop=1),
    lambda f: f["edges"][1].update(branch="next"),
    lambda f: f["edges"][2].update(target="run"),
    lambda f: f["edges"][2].update(target="unknown"),
    lambda f: f["edges"].append(copy.deepcopy(f["edges"][0])),
    lambda f: f["edges"].append({"source": "done", "target": "run", "branch": "next", "loop": True}),
])
def test_bad_flow_fails_before_it_can_be_rendered(method, change):
    source = looping(method)
    change(source["control_flow"])
    with pytest.raises((WorkbenchError, DiagramError)):
        compile_method(source)


def test_control_path_cannot_skip_a_required_dependency(method):
    source = looping(method)
    prep = copy.deepcopy(source["steps"][0])
    prep.update(id="prepare", equations=[])
    source["steps"].append(prep)
    source["steps"][0]["depends_on"] = ["prepare"]
    flow = source["control_flow"]
    flow["nodes"].append({"id": "prep", "kind": "step", "step": "prepare"})
    flow["edges"][1].update(target="prep", loop=True)
    flow["edges"].append({"source": "prep", "target": "run", "branch": "next", "loop": False})
    with pytest.raises(DiagramError, match="dependency"):
        # Direct checker gives a specific issue; compile_method wraps it.
        from researchclaw.pipeline.diagram_spec import validate_control_flow
        validate_control_flow(flow, {s["id"]: s for s in source["steps"]})


@pytest.mark.parametrize("field,value", [("label", "Fabricated improvement"), ("phase", "train"),
    ("equation_refs", []), ("ports", []), ("source_step", "other")])
def test_rehashed_labels_or_ports_cannot_override_method(method, field, value):
    doc = compile_method(method)
    diagram, = method_diagrams(doc)
    diagram["nodes"][0][field] = value
    diagram["id"] = "diagram-" + content_hash(diagram)[:24]
    with pytest.raises(DiagramError):
        validate_diagram(diagram, doc)


@pytest.mark.parametrize("format", ["svg", "png", "pdf"])
def test_real_renderer_is_deterministic_preserves_labels_and_ignores_global_style(method, format):
    import matplotlib
    diagram = method_diagrams(compile_method(looping(method)))[1]
    first = diagram_bytes(diagram, format)
    with matplotlib.rc_context({"axes.facecolor": "red", "font.size": 40, "text.usetex": True}):
        assert first == diagram_bytes(diagram, format)
    if format == "svg":
        tree = ET.fromstring(first)
        text = " ".join(tree.itertext())
        assert "Another batch available?" in text and "true / loop" in text
        assert "equations: projection" in text and "input: X, W" in text
    elif format == "pdf":
        import fitz
        with fitz.open(stream=first, filetype="pdf") as pdf:
            assert "Another batch available?" in pdf[0].get_text()
    else:
        assert first.startswith(b"\x89PNG")


@pytest.mark.parametrize("format", ["svg", "png", "pdf"])
def test_changed_diagram_pixels_fail_even_with_updated_manifest(tmp_path, method, format):
    write_json(tmp_path / "method_spec.json", compile_method(method))
    report = prepare_assets(tmp_path)
    assert verify_assets(tmp_path) == report
    name = f"publication_assets/{report['spec']['diagrams'][0]['id']}.{format}"
    (tmp_path / name).write_bytes(b"forged topology")
    report["outputs"][name] = file_hash(tmp_path / name)
    report.pop("version")
    report["version"] = content_hash(report)
    write_json(tmp_path / "publication_assets.json", report)
    with pytest.raises(AssetError, match="Diagram"):
        verify_assets(tmp_path)


def test_source_change_invalidates_all_diagram_exports(tmp_path, method):
    write_json(tmp_path / "method_spec.json", compile_method(method))
    old = prepare_assets(tmp_path)
    changed = looping(method)
    write_json(tmp_path / "method_spec.json", compile_method(changed))
    with pytest.raises(AssetError, match="specification"):
        verify_assets(tmp_path)
    new = prepare_assets(tmp_path)
    assert len(new["spec"]["diagrams"]) == 2
    assert not set(d["id"] for d in old["spec"]["diagrams"]) & set(d["id"] for d in new["spec"]["diagrams"])
    assert verify_assets(tmp_path) == new


def test_svg_escapes_untrusted_condition_and_does_not_run_tex(method):
    source = looping(method)
    source["control_flow"]["nodes"][1]["condition"] = r"<script> & $x$ \\input{secret}"
    svg = diagram_bytes(method_diagrams(compile_method(source))[1], "svg")
    assert b"<script>" not in svg and b"&lt;script&gt;" in svg


def test_large_label_fails_explicitly_instead_of_clipping(method):
    diagram, = method_diagrams(compile_method(method))
    diagram["nodes"][0]["label"] = "too long " * 500
    with pytest.raises(DiagramError, match="label budget"):
        diagram_bytes(diagram, "png")


def test_missing_font_glyph_fails_instead_of_exporting_tofu(method):
    source = looping(method)
    source["control_flow"]["nodes"][1]["condition"] = "No glyph: \U0010ffff"
    with pytest.raises(DiagramError, match="character"):
        diagram_bytes(method_diagrams(compile_method(source))[1], "png")


def test_chinese_labels_are_rendered_or_explicitly_rejected(method):
    from researchclaw.pipeline.diagram_spec import _fonts
    source = looping(method)
    source["control_flow"]["nodes"][1]["condition"] = "是否还有批次？"
    diagram = method_diagrams(compile_method(source))[1]
    if len(_fonts()) > 1:
        assert diagram_bytes(diagram, "png").startswith(b"\x89PNG")
    else:
        with pytest.raises(DiagramError, match="character"):
            diagram_bytes(diagram, "png")


def test_rehashed_visual_review_cannot_claim_unperformed_check(tmp_path, method):
    write_json(tmp_path / "method_spec.json", compile_method(method))
    report = prepare_assets(tmp_path)
    report["visual_review"] = "verified"
    report.pop("version")
    report["version"] = content_hash(report)
    write_json(tmp_path / "publication_assets.json", report)
    with pytest.raises(AssetError):
        verify_assets(tmp_path)
