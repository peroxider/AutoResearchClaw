"""Method-bound diagrams. Graph checks do not prove program equivalence.

The controlled vector backend is used when there is no visual topology reviewer.
Dependency arrows deliberately do not claim tensor flow or execution order.
"""
from __future__ import annotations

import io
import textwrap
import warnings
from functools import lru_cache
from pathlib import Path

from researchclaw.pipeline.evidence_store import content_hash, file_hash
from researchclaw.pipeline.experiment_protocol import _object, _identifier, ProtocolError


class DiagramError(ValueError):
    pass


DIAGRAM_PAGE_NODES = 8


def paginate_diagram(diagram: dict, *, page_nodes: int = DIAGRAM_PAGE_NODES) -> list[dict]:
    """Split a declared graph without dropping nodes or cross-page edges."""
    nodes = diagram["nodes"]
    if len(nodes) <= page_nodes:
        return [diagram]
    if type(page_nodes) is not int or page_nodes < 2:
        raise DiagramError("Diagram page size must be an integer of at least two nodes")
    chunks = [nodes[index:index + page_nodes] for index in range(0, len(nodes), page_nodes)]
    node_page = {node["id"]: index + 1 for index, chunk in enumerate(chunks) for node in chunk}
    if set(node_page) != {node["id"] for node in nodes}:
        raise DiagramError("Diagram pagination requires unique node IDs")
    pages = []
    for page_index, chunk in enumerate(chunks, start=1):
        local = {node["id"] for node in chunk}
        links = {node_id: [] for node_id in local}
        cross = []
        local_edges = []
        for edge in diagram["edges"]:
            source_page, target_page = node_page[edge["source"]], node_page[edge["target"]]
            if source_page == target_page == page_index:
                local_edges.append(edge)
            elif page_index in {source_page, target_page}:
                record = {**edge, "source_page": source_page, "target_page": target_page}
                cross.append(record)
                if edge["source"] in local:
                    links[edge["source"]].append(
                        f"to page {target_page}: {edge['target']} [{edge['label']}]")
                if edge["target"] in local:
                    links[edge["target"]].append(
                        f"from page {source_page}: {edge['source']} [{edge['label']}]")
        page = {key: value for key, value in diagram.items()
                if key not in {"id", "nodes", "edges", "groups", "entry"}}
        page["nodes"] = [{**node, "page_links": sorted(links[node["id"]])} for node in chunk]
        page["edges"] = local_edges
        page["groups"] = [{"id": group["id"], "nodes": [node for node in group["nodes"] if node in local]}
                          for group in diagram.get("groups", []) if any(node in local for node in group["nodes"])]
        page.update({"page": page_index, "page_count": len(chunks), "cross_page_edges": cross,
                     "source_diagram_id": diagram["id"]})
        if diagram.get("entry") in local:
            page["entry"] = diagram["entry"]
        page["id"] = "diagram-" + content_hash(page)[:24]
        pages.append(page)
    # Each original cross-page edge must appear on both endpoint pages.
    expected = [edge for edge in diagram["edges"] if node_page[edge["source"]] != node_page[edge["target"]]]
    if sum(len(page["cross_page_edges"]) for page in pages) != 2 * len(expected):
        raise DiagramError("Diagram pagination lost a cross-page edge")
    return pages


def validate_control_flow(flow: dict, steps: dict) -> list[str]:
    """Validate explicit control edges, reachability and possible termination.

    A path to an exit is not a proof that every execution terminates. Conditions
    are declared prose, never evaluated as Python, and remain unverified.
    """
    from researchclaw.pipeline.research_workbench import _dag, WorkbenchError
    try:
        _object(flow, {"entry", "nodes", "edges"}, {"entry", "nodes", "edges"}, "control flow")
        if not isinstance(flow["nodes"], list) or not 1 <= len(flow["nodes"]) <= 24:
            raise DiagramError("Control flow requires 1–24 nodes")
        if not isinstance(flow["edges"], list) or len(flow["edges"]) > 48:
            raise DiagramError("Control flow supports at most 48 edges")
        nodes, step_refs = {}, []
        for node in flow["nodes"]:
            _object(node, {"id", "kind", "step", "condition", "phase"}, {"id", "kind"}, "control node")
            nid = _identifier(node["id"])
            if nid in nodes:
                raise DiagramError("Duplicate control node")
            if node["kind"] == "step":
                if set(node) != {"id", "kind", "step"} or node["step"] not in steps:
                    raise DiagramError("Unknown or malformed control step")
                step_refs.append(node["step"])
            elif node["kind"] in {"decision", "stop"}:
                fields = {"id", "kind", "phase"} | ({"condition"} if node["kind"] == "decision" else set())
                if set(node) != fields or node["phase"] not in {"train", "inference"}:
                    raise DiagramError("Invalid decision/stop phase or fields")
                if node["kind"] == "decision" and (not isinstance(node["condition"], str)
                        or not node["condition"].strip() or len(node["condition"]) > 240
                        or any(ord(c) < 32 for c in node["condition"])):
                    raise DiagramError("Decision condition must contain 1–240 printable characters")
            else:
                raise DiagramError("Unknown control node kind")
            nodes[nid] = node
        if sorted(step_refs) != sorted(steps):
            raise DiagramError("Control flow must map every method step exactly once")
        if not isinstance(flow["entry"], str) or flow["entry"] not in nodes:
            raise DiagramError("Missing control flow entry")
        outgoing, incoming, dag = ({key: [] for key in nodes} for _ in range(3))
        seen = set()
        for edge in flow["edges"]:
            _object(edge, {"source", "target", "branch", "loop"}, {"source", "target", "branch", "loop"}, "control edge")
            source, target = edge["source"], edge["target"]
            if not isinstance(source, str) or not isinstance(target, str) or source not in nodes or target not in nodes:
                raise DiagramError("Unknown control endpoint")
            if edge["branch"] not in {"next", "true", "false"} or type(edge["loop"]) is not bool:
                raise DiagramError("Invalid branch or loop flag")
            key = source, edge["branch"]
            if key in seen:
                raise DiagramError("Duplicate or ambiguous control branch")
            seen.add(key)
            outgoing[source].append(edge)
            incoming[target].append(source)
            if not edge["loop"]:
                dag[target].append(source)
        for nid, node in nodes.items():
            expected = {"true", "false"} if node["kind"] == "decision" else {"next"} if node["kind"] == "step" else set()
            if {e["branch"] for e in outgoing[nid]} != expected:
                raise DiagramError("Decisions need true/false exits; steps one next; stops no exits")
        order = _dag(dag)
        positions = {nid: i for i, nid in enumerate(order)}
        for edge in flow["edges"]:
            if edge["loop"] and positions[edge["target"]] > positions[edge["source"]]:
                raise DiagramError("Loop edge must return to the same or an earlier node")
        def reachable(seeds, neighbors):
            result, pending = set(), list(seeds)
            while pending:
                nid = pending.pop()
                if nid not in result:
                    result.add(nid)
                    pending.extend(neighbors(nid))
            return result
        for edge in flow["edges"]:
            if edge["loop"] and edge["source"] not in reachable([edge["target"]],
                    lambda nid: [e["target"] for e in outgoing[nid] if not e["loop"]]):
                raise DiagramError("Loop marker must identify an actual feedback cycle")
        reached = reachable([flow["entry"]], lambda nid: [e["target"] for e in outgoing[nid]])
        stops = [nid for nid, node in nodes.items() if node["kind"] == "stop"]
        terminating = reachable(stops, lambda nid: incoming[nid])
        if reached != set(nodes) or terminating != set(nodes):
            raise DiagramError("All control nodes must be reachable and have a path to a stop")
        # A required dependency must execute on every path reaching its consumer.
        dominators = {nid: ({nid} if nid == flow["entry"] else set(nodes)) for nid in nodes}
        changed = True
        while changed:
            changed = False
            for nid in nodes:
                if nid == flow["entry"]:
                    continue
                value = {nid} | set.intersection(*(dominators[p] for p in incoming[nid]))
                if value != dominators[nid]:
                    dominators[nid] = value
                    changed = True
        step_nodes = {node["step"]: nid for nid, node in nodes.items() if node["kind"] == "step"}
        for sid, step in steps.items():
            if any(step_nodes[dep] not in dominators[step_nodes[sid]] for dep in step["depends_on"]):
                raise DiagramError("Control flow can skip a required method dependency")
        return order
    except (TypeError, KeyError, ProtocolError, WorkbenchError) as exc:
        raise DiagramError(f"Invalid control flow: {exc}") from exc


def method_diagrams(method: dict) -> list[dict]:
    from researchclaw.pipeline.research_workbench import compile_method
    if compile_method(method["spec"]) != method:
        raise DiagramError("MethodSpec version or structure changed")
    spec = method["spec"]
    steps = {s["id"]: s for s in spec["steps"]}
    def step_node(nid, step):
        return {"id": nid, "kind": "step", "label": step["id"], "phase": step["phase"],
                "source_step": step["id"], "equation_refs": step["equations"],
                "ports": [{"id": "in-" + v, "direction": "input", "variable": v} for v in step["inputs"]]
                       + [{"id": "out-" + v, "direction": "output", "variable": v} for v in step["outputs"]]}
    def document(kind, nodes, edges):
        result = {"schema_version": 1, "kind": kind, "source_method_version": method["version"],
                  "method_id": spec["method_id"], "nodes": nodes, "edges": edges,
                  "groups": [{"id": phase, "nodes": [n["id"] for n in nodes if n["phase"] == phase]}
                             for phase in ("train", "inference") if any(n["phase"] == phase for n in nodes)],
                  "implementation_equivalence": "unresolved", "termination": "unproved"}
        result["id"] = "diagram-" + content_hash(result)[:24]
        return result
    nodes = [step_node(sid, steps[sid]) for sid in method["step_order"]]
    edges = [{"source": dep, "target": sid, "type": "dependency", "label": "dependency", "loop": False}
             for sid in method["step_order"] for dep in steps[sid]["depends_on"]]
    diagrams = [document("architecture", nodes, edges)]
    if "control_flow" in spec:
        flow = spec["control_flow"]
        by_id = {node["id"]: node for node in flow["nodes"]}
        nodes = []
        for nid in validate_control_flow(flow, steps):
            node = by_id[nid]
            nodes.append(step_node(nid, steps[node["step"]]) if node["kind"] == "step" else
                         {"id": nid, "kind": node["kind"], "label": node.get("condition", "Stop"),
                          "phase": node["phase"], "source_step": None, "equation_refs": [], "ports": []})
        edges = [{"source": e["source"], "target": e["target"], "type": "control", "label": e["branch"], "loop": e["loop"]}
                 for e in flow["edges"]]
        diagram = document("execution", nodes, edges)
        diagram["entry"] = flow["entry"]
        diagram["stopping_rule"] = spec["stopping_rule"]
        diagram["id"] = "diagram-" + content_hash({k: v for k, v in diagram.items() if k != "id"})[:24]
        diagrams.append(diagram)
    phases = {name: {step["phase"] for step in spec["steps"] if name in step["inputs"] + step["outputs"]}
              for name in spec["variables"]}
    variable_nodes = [{"id": "var-" + name, "kind": "variable",
                       "label": (name + "\nshape: " + " × ".join(map(str, variable["shape"]))
                                 + ("\ndtype: " + variable["dtype"] if variable.get("dtype") else "")
                                 + "\n" + variable["description"]),
                       "phase": next(iter(phases[name])) if len(phases[name]) == 1 else "shared",
                       "source_step": None, "equation_refs": [], "ports": []}
                      for name, variable in spec["variables"].items()]
    data_steps = [{**step_node("step-" + sid, steps[sid]), "label": sid + "\n" + steps[sid]["description"]}
                  for sid in method["step_order"]]
    data_edges = []
    for sid in method["step_order"]:
        for variable in steps[sid]["inputs"]:
            data_edges.append({"source": "var-" + variable, "target": "step-" + sid,
                               "type": "declared_read", "label": variable + " / read", "loop": False})
        for variable in steps[sid]["outputs"]:
            data_edges.append({"source": "step-" + sid, "target": "var-" + variable,
                               "type": "declared_write", "label": variable + " / write", "loop": False})
    data_flow = {"schema_version": 1, "kind": "data_flow", "source_method_version": method["version"],
                 "method_id": spec["method_id"], "nodes": [*variable_nodes, *data_steps], "edges": data_edges,
                 "groups": [{"id": phase, "nodes": [node["id"] for node in [*variable_nodes, *data_steps]
                                                     if node["phase"] == phase]}
                            for phase in ("train", "inference", "shared")
                            if any(node["phase"] == phase for node in [*variable_nodes, *data_steps])],
                 "implementation_equivalence": "unresolved", "termination": "unproved",
                 "data_flow_scope": "Declared reads and writes; not observed runtime tensor flow"}
    data_flow["id"] = "diagram-" + content_hash(data_flow)[:24]
    diagrams.append(data_flow)
    return [page for diagram in diagrams for page in paginate_diagram(diagram)]


def experiment_split_diagrams(contract: dict) -> list[dict]:
    """Derive one privacy-aware split diagram per frozen dataset contract."""
    diagrams = []
    for dataset in contract["datasets"]:
        card = dataset["card"]
        sizes = card["split_sizes"]
        nodes = [
            {"id": "source", "kind": "data", "label": f"{card['dataset']} / {card['version']}\n{card['task']} / {card['metric']}",
             "phase": "data", "source_step": None, "equation_refs": [], "ports": []},
            {"id": "train", "kind": "split", "label": f"Train: {sizes['train']} rows\nlabels available; preprocessing fit",
             "phase": "train", "source_step": None, "equation_refs": [], "ports": []},
            {"id": "validation", "kind": "split", "label": f"Validation: {sizes['validation']} rows\nlabels available; model selection only",
             "phase": "validation", "source_step": None, "equation_refs": [], "ports": []},
            {"id": "test_features", "kind": "split", "label": f"Test features: {sizes['test']} rows\nlabels withheld from experiment code",
             "phase": "test", "source_step": None, "equation_refs": [], "ports": []},
            {"id": "test_labels", "kind": "private", "label": f"Private test labels: {sizes['test']} rows\nhost evaluator only",
             "phase": "private", "source_step": None, "equation_refs": [], "ports": []},
        ]
        edges = [
            {"source": "source", "target": name, "type": "split", "label": card["split"]["strategy"], "loop": False}
            for name in ("train", "validation", "test_features", "test_labels")]
        diagram = {"schema_version": 1, "kind": "dataset_split",
                   "source_contract_version": contract["version"], "dataset": card["dataset"],
                   "title": f"{card['dataset']} frozen data split", "nodes": nodes, "edges": edges,
                   "groups": [{"id": "model_visible", "nodes": ["train", "validation", "test_features"]},
                              {"id": "host_private", "nodes": ["test_labels"]}],
                   "privacy_scope": "Contract separation of files; not operating-system isolation",
                   "split_ids_sha256": card["split_ids_sha256"]}
        diagram["id"] = "diagram-" + content_hash(diagram)[:24]
        diagrams.append(diagram)
    return diagrams


def research_overview_diagram(contract: dict, protocol: dict) -> dict:
    """Build a compact overview from frozen datasets, methods and RQs."""
    datasets = [item["manifest"]["dataset"] for item in contract["datasets"]]
    methods = [f"{item['id']} ({item['role']})" for item in protocol["spec"]["methods"]]
    questions = [f"{item['id']} ({item['kind']}): {item['question']}" for item in protocol["spec"]["questions"]]
    nodes = [
        {"id": "datasets", "kind": "evidence", "label": "Frozen datasets\n" + "\n".join(datasets),
         "phase": "data", "source_step": None, "equation_refs": [], "ports": []},
        {"id": "methods", "kind": "declaration", "label": "Declared methods\n" + "\n".join(methods),
         "phase": "method", "source_step": None, "equation_refs": [], "ports": []},
        {"id": "questions", "kind": "questions", "label": "Research questions\n" + "\n".join(questions),
         "phase": "research", "source_step": None, "equation_refs": [], "ports": []},
        {"id": "evidence", "kind": "output", "label": "Host-evaluated evidence\npaired comparisons, analysis, manuscript bindings",
         "phase": "evidence", "source_step": None, "equation_refs": [], "ports": []},
    ]
    if any(len(node["label"]) > 1200 for node in nodes):
        raise DiagramError("Research overview label budget exceeded; split the protocol before publication")
    edges = [
        {"source": "datasets", "target": "questions", "type": "declared_input", "label": "declared input", "loop": False},
        {"source": "methods", "target": "questions", "type": "declared_comparison", "label": "declared comparison", "loop": False},
        {"source": "questions", "target": "evidence", "type": "evaluated", "label": "host evaluation", "loop": False},
    ]
    diagram = {"schema_version": 1, "kind": "research_overview", "title": "Frozen research overview",
               "source_contract_version": contract["version"], "source_protocol_version": protocol["version"],
               "nodes": nodes, "edges": edges,
               "groups": [{"id": "frozen_declarations", "nodes": ["datasets", "methods", "questions"]},
                          {"id": "derived_outputs", "nodes": ["evidence"]}],
               "semantic_scope": "Declared study graph; arrows do not establish causality or scientific validity"}
    diagram["id"] = "diagram-" + content_hash(diagram)[:24]
    return diagram


def validate_diagram(diagram: dict, method: dict) -> dict:
    expected = next((item for item in method_diagrams(method) if item["id"] == diagram.get("id")), None)
    if diagram != expected:
        raise DiagramError("Diagram labels, ports, topology or version differ from MethodSpec")
    return diagram


@lru_cache(maxsize=32)
def _font_digest(path: str, size: int, mtime_ns: int) -> str:
    return file_hash(Path(path))


def _fonts() -> list[dict]:
    from matplotlib import font_manager
    result = []
    for name in ("DejaVu Sans", "Noto Sans CJK SC", "Microsoft YaHei", "SimHei"):
        try:
            path = Path(font_manager.findfont(name, fallback_to_default=False))
        except ValueError:
            continue
        stat = path.stat()
        result.append({"name": name, "sha256": _font_digest(str(path), stat.st_size, stat.st_mtime_ns)})
        if name != "DejaVu Sans":
            break
    return result


def renderer_identity() -> dict:
    import matplotlib
    return {"name": "controlled-method-vector/v2", "matplotlib": matplotlib.__version__,
            "renderer_sha256": file_hash(Path(__file__)), "fonts": _fonts(),
            "visual_review": "machine_checked_geometry",
            "selection_reason": "Controlled vector rendering with bounded geometry review",
            "semantic_scope": "Declared nodes, ports, equations and typed edges; not implementation equivalence"}


def _label(node, *, entry=False, wrap_width=46):
    lines = [f"{node['id']} [{node['phase']}]" + (" [ENTRY]" if entry else "")]
    if node["kind"] != "step":
        lines.append(node["kind"].upper())
    lines.append(node["label"])
    for direction in ("input", "output"):
        ports = [p["variable"] for p in node["ports"] if p["direction"] == direction]
        if ports:
            lines.append(direction + ": " + ", ".join(ports))
    if node["equation_refs"]:
        lines.append("equations: " + ", ".join(node["equation_refs"]))
    for link in node.get("page_links", []):
        lines.append("continuation: " + link)
    return "\n".join(textwrap.fill(line, width=wrap_width, break_long_words=True) for line in lines)


def _diagram_render(diagram: dict, format: str, *, wrap_width: int) -> tuple[bytes, dict]:
    """Render once and return a machine-checkable geometry review."""
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    nodes, edges = diagram["nodes"], diagram["edges"]
    labels = {n["id"]: _label(n, entry=n["id"] == diagram.get("entry"), wrap_width=wrap_width) for n in nodes}
    heights = {nid: max(0.85, 0.23 * (label.count("\n") + 1) + 0.3) for nid, label in labels.items()}
    height, width = sum(heights.values()) + 0.4 * len(nodes) + 1.3, 6 + 0.25 * len(edges)
    if height > 48:
        raise DiagramError("Diagram exceeds rendering height budget")
    settings = {k: v for k, v in matplotlib.rcParamsDefault.items() if k != "backend"}
    settings.update({"font.family": [font["name"] for font in _fonts()], "font.size": 12, "text.usetex": False,
                     "text.parse_math": False, "svg.hashsalt": "researchclaw-diagram-v1", "svg.fonttype": "none"})
    with warnings.catch_warnings(), matplotlib.rc_context(settings):
        warnings.filterwarnings("error", message=r"Glyph .* missing from font")
        fig = Figure(figsize=(width, height), dpi=120)
        FigureCanvasAgg(fig)
        ax = fig.add_axes((0, 0, 1, 1), xlim=(0, width), ylim=(0, height))
        ax.axis("off")
        title = diagram.get("title", diagram.get("method_id", "Declared structure") + " / " + diagram["kind"])
        if diagram.get("page_count", 1) > 1:
            title += f" / page {diagram['page']} of {diagram['page_count']}"
        title_artist = ax.text(0.25, height - 0.2, textwrap.fill(title, 50), va="top", weight="bold")
        footer = (diagram.get("privacy_scope") or diagram.get("semantic_scope") or diagram.get("data_flow_scope") or
                  "Declared structure; implementation equivalence\nand termination remain unverified.")
        footer_artist = ax.text(0.25, 0.12, footer, fontsize=10)
        centers, y, texts, boxes = {}, height - 0.7, [], []
        for node in nodes:
            h = heights[node["id"]]
            center = y - h / 2
            centers[node["id"]] = center
            color = {"train": "#e8f0fc", "validation": "#fff4d6", "test": "#e7f4ed",
                     "private": "#fce8e6", "data": "#eef0f3", "method": "#e8f0fc",
                     "research": "#fff4d6", "evidence": "#e7f4ed", "shared": "#f0e8fc"}.get(node["phase"], "#e7f4ed")
            box = FancyBboxPatch((0.25, y - h), 4.8, h, boxstyle="round,pad=0.04",
                                 facecolor=color, edgecolor="#45566b", linewidth=1)
            ax.add_patch(box)
            text = ax.text(0.38, center, labels[node["id"]], va="center", fontsize=12)
            texts.append(text)
            boxes.append(box)
            y -= h + 0.4
        outgoing = {nid: [i for i, e in enumerate(edges) if e["source"] == nid] for nid in centers}
        incoming = {nid: [i for i, e in enumerate(edges) if e["target"] == nid] for nid in centers}
        edge_texts = []
        for index, edge in enumerate(edges):
            source, target = edge["source"], edge["target"]
            sy = centers[source] - heights[source] * (0.1 + 0.3 * (outgoing[source].index(index) + 1) / (len(outgoing[source]) + 1))
            ty = centers[target] + heights[target] * (0.1 + 0.3 * (incoming[target].index(index) + 1) / (len(incoming[target]) + 1))
            lane = 5.45 + 0.25 * index
            color = "#aa4b13" if edge["loop"] else "#45566b"
            style = "--" if edge["loop"] else "-"
            ax.plot([5.1, lane, lane, 5.18], [sy, sy, ty, ty], color=color, linestyle=style, linewidth=0.85)
            ax.add_patch(FancyArrowPatch((5.18, ty), (5.08, ty), arrowstyle="-|>", mutation_scale=9, color=color))
            edge_texts.append(ax.text(lane + 0.025, (sy + ty) / 2,
                              edge["label"] + (" / loop" if edge["loop"] else ""),
                              rotation=90, va="center", fontsize=10, color=color,
                              bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4}))
        try:
            fig.canvas.draw()
        except UserWarning as exc:
            raise DiagramError("Diagram font cannot render a required character") from exc
        renderer = fig.canvas.get_renderer()
        canvas = fig.bbox
        for text, box in zip(texts, boxes):
            a, b = text.get_window_extent(renderer), box.get_window_extent(renderer)
            if a.x0 < b.x0 or a.x1 > b.x1 or a.y0 < b.y0 or a.y1 > b.y1:
                raise DiagramError("Node label does not fit; simplify or split the source method")
        for artist in [title_artist, footer_artist, *edge_texts]:
            extent = artist.get_window_extent(renderer)
            if extent.x0 < canvas.x0 or extent.x1 > canvas.x1 or extent.y0 < canvas.y0 or extent.y1 > canvas.y1:
                raise DiagramError("Diagram annotation leaves the canvas")
        # Node boxes are vertically disjoint by construction; all edge routes use
        # an x lane beyond the right edge of every node box.
        box_extents = [box.get_window_extent(renderer) for box in boxes]
        if any(left.overlaps(right) for index, left in enumerate(box_extents) for right in box_extents[index + 1:]):
            raise DiagramError("Diagram node boxes overlap")
        rightmost = max(box.x1 for box in box_extents)
        lane_pixels = [ax.transData.transform((5.45 + 0.25 * index, 0))[0] for index in range(len(edges))]
        if any(lane <= rightmost for lane in lane_pixels):
            raise DiagramError("Diagram edge lane crosses a node box")
        buffer = io.BytesIO()
        metadata = ({"Creator": "AutoResearchClaw", "CreationDate": None, "ModDate": None} if format == "pdf" else
                    {"Creator": "AutoResearchClaw", "Date": None} if format == "svg" else {"Software": "AutoResearchClaw"})
        try:
            fig.savefig(buffer, format=format, metadata=metadata)
        except UserWarning as exc:
            raise DiagramError("Diagram font cannot render a required character") from exc
        review = {"schema_version": 1, "checker": "matplotlib_geometry/v1", "status": "passed",
                  "diagram_id": diagram["id"], "wrap_width": wrap_width,
                  "node_count": len(nodes), "edge_count": len(edges),
                  "checks": ["node_labels_inside_boxes", "annotations_inside_canvas",
                             "node_boxes_disjoint", "edge_lanes_outside_nodes", "all_declared_edges_rendered"],
                  "scope": "Rendered geometry and declared topology; not human aesthetic or semantic review"}
        return buffer.getvalue(), review


def _render_with_bounded_review(diagram: dict, format: str) -> tuple[bytes, dict]:
    failures = []
    for attempt, wrap_width in enumerate((46, 40, 34), start=1):
        try:
            data, review = _diagram_render(diagram, format, wrap_width=wrap_width)
            return data, {**review, "attempt": attempt, "max_attempts": 3,
                          "repairs": [{"attempt": item[0], "wrap_width": item[1], "issue": item[2]}
                                      for item in failures]}
        except DiagramError as exc:
            failures.append((attempt, wrap_width, str(exc)))
            if str(exc) not in {"Node label does not fit; simplify or split the source method",
                                "Diagram annotation leaves the canvas"}:
                break
    detail = "; ".join(f"attempt {attempt} width {width}: {issue}" for attempt, width, issue in failures)
    raise DiagramError("Bounded visual geometry review failed: " + detail)


def diagram_visual_review(diagram: dict) -> dict:
    """Return the deterministic geometry review used by publication output."""
    return _render_with_bounded_review(diagram, "png")[1]


def diagram_bytes(diagram: dict, format: str) -> bytes:
    """Render exact labels as vector text, with a separate route for every edge.

    Large diagrams fail explicitly instead of silently dropping nodes or labels.
    SVG hash seeds and PDF metadata are fixed for byte-level reproducibility.
    """
    if format not in {"svg", "png", "pdf"}:
        raise DiagramError("Unsupported diagram format")
    nodes, edges = diagram["nodes"], diagram["edges"]
    if not 1 <= len(nodes) <= 24 or len(edges) > 48:
        raise DiagramError("Diagram exceeds 24 nodes / 48 edges; split MethodSpec before publication")
    labels = {n["id"]: _label(n, entry=n["id"] == diagram.get("entry")) for n in nodes}
    if any(len(label) > 1200 for label in labels.values()):
        raise DiagramError("Diagram node exceeds label budget; split the method instead of truncating it")
    return _render_with_bounded_review(diagram, format)[0]
