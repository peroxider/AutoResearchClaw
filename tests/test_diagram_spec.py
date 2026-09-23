import copy
import json
import xml.etree.ElementTree as ET

import pytest

from researchclaw.literature.evidence import write_json
from researchclaw.pipeline.diagram_spec import (
    DiagramError, diagram_bytes, diagram_visual_review, method_diagrams, paginate_diagram, validate_diagram,
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
    diagram = next(item for item in method_diagrams(doc) if item["kind"] == "architecture")
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


def test_data_flow_is_bipartite_declared_reads_and_writes_without_producer_guessing(method):
    doc = compile_method(method)
    diagram = next(item for item in method_diagrams(doc) if item["kind"] == "data_flow")
    assert {node["id"] for node in diagram["nodes"]} == {"var-X", "var-W", "var-Y", "step-predict"}
    assert {(edge["source"], edge["target"], edge["type"], edge["label"]) for edge in diagram["edges"]} == {
        ("var-X", "step-predict", "declared_read", "X / read"),
        ("var-W", "step-predict", "declared_read", "W / read"),
        ("step-predict", "var-Y", "declared_write", "Y / write"),
    }
    assert diagram["data_flow_scope"].endswith("not observed runtime tensor flow")
    assert validate_diagram(diagram, doc) == diagram


def test_explicit_loop_keeps_true_false_stop_and_unknown_termination(method):
    doc = compile_method(looping(method))
    diagrams = method_diagrams(doc)
    architecture = next(item for item in diagrams if item["kind"] == "architecture")
    flow = next(item for item in diagrams if item["kind"] == "execution")
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
    diagram = next(item for item in method_diagrams(doc) if item["kind"] == "architecture")
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
    assert len(new["spec"]["diagrams"]) == 3
    assert not set(d["id"] for d in old["spec"]["diagrams"]) & set(d["id"] for d in new["spec"]["diagrams"])
    assert verify_assets(tmp_path) == new


def test_svg_escapes_untrusted_condition_and_does_not_run_tex(method):
    source = looping(method)
    source["control_flow"]["nodes"][1]["condition"] = r"<script> & $x$ \\input{secret}"
    svg = diagram_bytes(method_diagrams(compile_method(source))[1], "svg")
    assert b"<script>" not in svg and b"&lt;script&gt;" in svg


def test_large_label_fails_explicitly_instead_of_clipping(method):
    diagram = next(item for item in method_diagrams(compile_method(method)) if item["kind"] == "architecture")
    diagram["nodes"][0]["label"] = "too long " * 500
    with pytest.raises(DiagramError, match="label budget"):
        diagram_bytes(diagram, "png")


def test_large_graph_is_paginated_without_dropping_nodes_or_cross_page_edges():
    nodes = [{"id": f"n{index}", "kind": "step", "label": f"step {index}", "phase": "train",
              "source_step": f"n{index}", "equation_refs": [], "ports": []}
             for index in range(18)]
    edges = [{"source": f"n{index}", "target": f"n{index + 1}", "type": "dependency",
              "label": "dependency", "loop": False} for index in range(17)]
    full = {"schema_version": 1, "kind": "architecture", "source_method_version": "fixture",
            "method_id": "large", "nodes": nodes, "edges": edges,
            "groups": [{"id": "train", "nodes": [node["id"] for node in nodes]}],
            "implementation_equivalence": "unresolved", "termination": "unproved"}
    full["id"] = "diagram-" + content_hash(full)[:24]
    pages = paginate_diagram(full)
    assert [page["page"] for page in pages] == [1, 2, 3]
    assert all(page["page_count"] == 3 and len(page["nodes"]) <= 8 for page in pages)
    assert {node["id"] for page in pages for node in page["nodes"]} == {node["id"] for node in nodes}
    cross = [edge for page in pages for edge in page["cross_page_edges"]]
    assert len(cross) == 4  # n7→n8 and n15→n16 appear on both endpoint pages.
    assert all(page["source_diagram_id"] == full["id"] for page in pages)
    svg = diagram_bytes(pages[0], "svg")
    assert b"continuation: to page 2: n8 [dependency]" in svg


def test_method_diagrams_automatically_publish_every_large_method_page(monkeypatch):
    steps = [{"id": f"s{index}", "phase": "train", "description": f"step {index}",
              "inputs": ["X"], "outputs": ["Y"],
              "equations": [], "depends_on": [] if index == 0 else [f"s{index - 1}"]}
             for index in range(17)]
    document = {"version": "fixture-version", "step_order": [step["id"] for step in steps],
                "spec": {"method_id": "large_method", "steps": steps,
                         "variables": {"X": {"shape": [1], "description": "input"},
                                       "Y": {"shape": [1], "description": "output"}}}}
    monkeypatch.setattr("researchclaw.pipeline.research_workbench.compile_method", lambda spec: document)
    pages = method_diagrams(document)
    architecture = [page for page in pages if page["kind"] == "architecture"]
    assert len(architecture) == 3
    assert sum(len(page["nodes"]) for page in architecture) == 17
    assert all(validate_diagram(page, document) == page for page in pages)


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
    report["visual_review"] = "human_verified"
    report.pop("version")
    report["version"] = content_hash(report)
    write_json(tmp_path / "publication_assets.json", report)
    with pytest.raises(AssetError):
        verify_assets(tmp_path)


def test_machine_geometry_review_is_bound_to_each_diagram(method):
    diagram = next(item for item in method_diagrams(compile_method(method)) if item["kind"] == "data_flow")
    review = diagram_visual_review(diagram)
    assert review["status"] == "passed"
    assert review["diagram_id"] == diagram["id"]
    assert review["attempt"] <= review["max_attempts"] == 3
    assert review["checks"] == ["node_labels_inside_boxes", "annotations_inside_canvas",
                                "node_boxes_disjoint", "edge_lanes_outside_nodes", "all_declared_edges_rendered"]
    assert review["scope"].startswith("Rendered geometry")


def test_geometry_review_repairs_only_within_fixed_attempt_budget(monkeypatch, method):
    import researchclaw.pipeline.diagram_spec as module
    diagram = next(item for item in method_diagrams(compile_method(method)) if item["kind"] == "architecture")
    original = module._diagram_render

    def fail_widest_once(candidate, format, *, wrap_width):
        if wrap_width == 46:
            raise DiagramError("Node label does not fit; simplify or split the source method")
        return original(candidate, format, wrap_width=wrap_width)

    monkeypatch.setattr(module, "_diagram_render", fail_widest_once)
    review = module.diagram_visual_review(diagram)
    assert review["attempt"] == 2 and review["wrap_width"] == 40
    assert review["repairs"] == [{"attempt": 1, "wrap_width": 46,
                                  "issue": "Node label does not fit; simplify or split the source method"}]
