"""Deterministic equations, algorithm records, proof records and result figures.

Rendering a stated equation does not establish its truth. Seed plots describe
training randomness on the frozen split, never population uncertainty.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import re
import statistics
from pathlib import Path

from researchclaw.literature.evidence import write_json
from researchclaw.pipeline.evidence_store import content_hash, file_hash


class AssetError(ValueError):
    pass


_RENDER_DIGESTS: dict[tuple[str, str, str], str] = {}


def _backend() -> dict:
    import matplotlib
    from researchclaw.pipeline.diagram_spec import renderer_identity
    return {"name": "matplotlib", "version": matplotlib.__version__, "renderer_sha256": file_hash(Path(__file__)),
            "diagram_renderer": renderer_identity()}


def _render_digest(figure: dict, format: str) -> str:
    import matplotlib
    key = content_hash(figure), format, content_hash(_backend())
    if key not in _RENDER_DIGESTS:
        from researchclaw.pipeline.diagram_spec import diagram_bytes
        render = diagram_bytes if "nodes" in figure and "edges" in figure else chart_bytes
        _RENDER_DIGESTS[key] = hashlib.sha256(render(figure, format)).hexdigest()
    return _RENDER_DIGESTS[key]


# Deliberately bounded math syntax, not an arbitrary LaTeX program interpreter.
MATH_COMMANDS = set((
    "alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu nu xi pi rho sigma "
    "tau upsilon phi varphi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega "
    "frac dfrac tfrac sqrt sum prod int iint lim max min argmax argmin log ln exp sin cos tan "
    "mathbb mathbf mathcal mathrm mathit text operatorname boldsymbol vec hat bar tilde dot ddot "
    "left right big Big bigg Bigg langle rangle lvert rvert lVert rVert vert Vert "
    "cdot times div pm mp le ge leq geq neq approx sim equiv propto in notin subset subseteq "
    "cup cap emptyset forall exists neg land lor to mapsto rightarrow leftarrow iff implies "
    "partial nabla infty ell ldots cdots vdots ddots top bot underbrace overbrace"
).split())


def math_latex(text: str) -> str:
    if not isinstance(text, str) or not text.strip() or len(text) > 12000:
        raise AssetError("Equation must contain 1–12000 characters")
    if "^^" in text or any(char in text for char in ("%", "#", "$", "&", "\x00")):
        raise AssetError("Equation contains unsupported TeX control syntax")
    depth = 0
    for token in re.findall(r"\\[A-Za-z]+|\\.|[{}]|.", text, re.S):
        if token.startswith("\\"):
            command = token[1:]
            if command not in MATH_COMMANDS and command not in {",", ";", ":", "!", " ", "{", "}", "|", "_"}:
                raise AssetError(f"Unsupported math command: {token}")
        elif token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
            if depth < 0:
                raise AssetError("Unbalanced equation braces")
    if depth:
        raise AssetError("Unbalanced equation braces")
    return text.strip()


def polynomial_latex(expression: str) -> str:
    """Pretty-print the already validated polynomial AST without evaluation."""
    def render(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return str(node.value)
        if isinstance(node, ast.Name):
            return r"\mathrm{" + node.id.replace("_", r"\_") + "}"
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return ("+" if isinstance(node.op, ast.UAdd) else "-") + "(" + render(node.operand) + ")"
        if isinstance(node, ast.BinOp):
            left, right = render(node.left), render(node.right)
            if isinstance(node.op, ast.Div):
                return r"\frac{" + left + "}{" + right + "}"
            if isinstance(node.op, ast.Pow):
                return "(" + left + ")^{" + right + "}"
            operator = {ast.Add: "+", ast.Sub: "-", ast.Mult: r"\cdot"}.get(type(node.op))
            if operator:
                return "(" + left + " " + operator + " " + right + ")"
        raise AssetError("Unsupported polynomial syntax")
    return math_latex(render(ast.parse(expression, mode="eval").body))


def _read(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AssetError("Expected asset source object")
    return data


def content_spec(root: Path) -> dict:
    from researchclaw.pipeline.research_workbench import compile_method, compile_theory
    from researchclaw.pipeline.experiment_protocol import load_protocol
    source_versions, equations, algorithm, proofs, figures, diagrams = {}, [], [], [], [], []
    contract = None
    method_path, theory_path = root / "method_spec.json", root / "theory_bundle.json"
    if method_path.is_file():
        method = _read(method_path)
        if compile_method(method["spec"]) != method:
            raise AssetError("MethodSpec changed")
        source_versions["method_spec.json"] = file_hash(method_path)
        spec = method["spec"]
        from researchclaw.pipeline.diagram_spec import method_diagrams
        diagrams = method_diagrams(method)
        equations = [{"id": "eq-" + equation["id"], "latex": math_latex(equation["latex"]),
                      "inputs": equation["inputs"], "outputs": equation["outputs"],
                      "source_version": method["version"]} for equation in spec["equations"]]
        steps = {step["id"]: step for step in spec["steps"]}
        from researchclaw.pipeline.method_semantics import reported_equivalence
        semantics = reported_equivalence(root, method)
        algorithm = {"method_id": spec["method_id"], "source_version": method["version"],
            "variables": spec["variables"], "steps": [steps[sid] for sid in method["step_order"]],
            "losses": spec["losses"], "stopping_rule": spec["stopping_rule"], "complexity": spec["complexity"],
            "semantic_equivalence": semantics["status"], "semantic_reviewer": semantics["checker"]}
    if theory_path.is_file():
        theory = _read(theory_path)
        if compile_theory(theory["spec"]) != theory:
            raise AssetError("TheoryBundle changed")
        source_versions["theory_bundle.json"] = file_hash(theory_path)
        originals = {item["id"]: item for item in theory["spec"]["obligations"]}
        for item in theory["spec"]["obligations"]:
            if item["statement"].get("kind") == "conjunction":
                for index, part in enumerate(item["statement"]["parts"], start=1):
                    originals[f"{item['id']}_part{index}"] = part
            elif item["statement"].get("kind") == "informal_decomposition":
                for part in item["statement"]["parts"]:
                    originals[f"{item['id']}__{part['id']}"] = part
        for obligation in theory["obligations"]:
            statement = obligation["statement"]
            equation = (polynomial_latex(statement["left"]) + " = " + polynomial_latex(statement["right"])) if statement["kind"] == "polynomial_identity" else ""
            proofs.append({**obligation, "latex": equation, "proof_text": originals.get(obligation["id"], {}).get("proof_text", ""),
                           "definitions": theory["spec"]["definitions"], "source_version": theory["version"]})
    if (root / "research_contract.json").is_file():
        from researchclaw.research_inputs import verify_bundle_contract
        from researchclaw.pipeline.diagram_spec import experiment_split_diagrams
        contract = verify_bundle_contract(root)
        source_versions["research_contract.json"] = file_hash(root / "research_contract.json")
        diagrams.extend(experiment_split_diagrams(contract))
    protocol = load_protocol(root)
    if protocol is not None:
        if contract is None:
            raise AssetError("Experiment protocol requires its frozen ResearchContract")
        from researchclaw.pipeline.diagram_spec import research_overview_diagram
        diagrams.append(research_overview_diagram(contract, protocol))
        from researchclaw.pipeline.analysis_spec import build_analysis, analysis_figures, AnalysisError
        try:
            analysis = build_analysis(root)
        except AnalysisError as exc:
            raise AssetError(str(exc)) from exc
        source_versions.update({name: file_hash(root / name) for name in ("experiment_protocol.json", "evidence_store.json")})
        if (root / "analysis_spec.json").is_file():
            if _read(root / "analysis_spec.json") != analysis:
                raise AssetError("Stale AnalysisSpec cannot drive publication figures")
            source_versions["analysis_spec.json"] = file_hash(root / "analysis_spec.json")
        figures = analysis_figures(analysis)
    return {"schema_version": 1, "renderer": "publication-assets/v1", "sources": source_versions,
            "equations": equations, "algorithm": algorithm, "proofs": proofs, "figures": figures, "diagrams": diagrams}


def chart_csv(figure: dict) -> str:
    output = io.StringIO(newline="")
    kind = figure.get("kind")
    rows = figure.get("rows", [])
    if kind == "effect_summary":
        fields = ["seed", "baseline", "candidate", "difference", "baseline_result", "candidate_result"]
        fields = ["analysis_id", "baseline_method", "candidate_method", *fields]
        rows = [{"analysis_id": item["analysis_id"], "baseline_method": item["baseline"], "candidate_method": item["candidate"], **row}
                for item in figure["comparisons"] for row in item["pairs"]]
    elif kind == "calibration":
        fields = ["seed", "baseline_ece", "candidate_ece", "difference", "test_samples"]
    elif kind == "efficiency_pareto":
        fields = ["seed", "baseline", "candidate", "baseline_seconds", "candidate_seconds"]
    elif kind == "learning_curve":
        fields = ["seed", "step", "baseline", "candidate", "difference"]
    else:
        fields = ["seed", "baseline", "candidate", "difference", "baseline_result", "candidate_result"]
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def chart_bytes(figure: dict, format: str) -> bytes:
    if figure.get("kind") == "effect_summary":
        return effect_chart_bytes(figure, format)
    if figure.get("kind") == "calibration":
        return calibration_chart_bytes(figure, format)
    if figure.get("kind") == "efficiency_pareto":
        return pareto_chart_bytes(figure, format)
    if figure.get("kind") == "learning_curve":
        return learning_curve_chart_bytes(figure, format)
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from textwrap import fill
    settings = {key: value for key, value in matplotlib.rcParamsDefault.items() if key != "backend"}
    settings.update({"font.family": "DejaVu Sans", "font.size": 11, "text.usetex": False})
    with matplotlib.rc_context(settings):
        plot = Figure(figsize=(6, 4.8), dpi=160)
        FigureCanvasAgg(plot)
        left, right = plot.subplots(1, 2)
        rows = figure["rows"]
        for row in rows:
            left.plot([0, 1], [row["baseline"], row["candidate"]], color="#a6adb5", alpha=0.7, linewidth=1)
        left.scatter([0] * len(rows), [r["baseline"] for r in rows], color="#345b9a", label="Baseline", zorder=3)
        left.scatter([1] * len(rows), [r["candidate"] for r in rows], color="#c67520", label="Candidate", zorder=3)
        left.set_xticks([0, 1], [fill(figure["baseline"], 18), fill(figure["candidate"], 18)])
        left.set_ylabel(f"{figure['metric']} ({figure['unit']})")
        left.set_title("Matched seed observations")
        if figure["unit"] == "fraction":
            left.set_ylim(-0.02, 1.02)
            left.set_yticks([i / 5 for i in range(6)])
        elif figure["unit"] == "percent":
            left.set_ylim(-2, 102)
            left.set_yticks([i * 20 for i in range(6)])
        differences = [row["difference"] for row in rows]
        right.axhline(0, color="#555555", linewidth=0.8)
        right.scatter(range(len(rows)), differences, color="#345b9a", zorder=3)
        right.axhline(figure["mean_difference"], color="#c67520", linestyle="--", label="Observed mean")
        # Avoid suggesting a large effect by automatically magnifying zero/noise.
        extent = max([abs(v) for v in differences] + [0.01 if figure["unit"] == "fraction" else 0.1]) * 1.2
        right.set_ylim(-extent, extent)
        right.set_xticks(range(len(rows)), [r["seed"] for r in rows], rotation=45 if len(rows) > 8 else 0)
        right.set_xlabel("Training seed")
        right.set_ylabel(f"Candidate minus baseline ({figure['unit']})")
        right.set_title("Paired differences")
        right.legend(loc="best", fontsize=10)
        for ax in (left, right):
            ax.grid(axis="y", color="#e5e7eb", linewidth=0.5)
            ax.spines[["top", "right"]].set_visible(False)
        plot.suptitle(fill(f"{figure['question']} / {figure['dataset']} / {figure['metric']}", 65), fontsize=12)
        plot.text(0.5, 0.015, "Training-seed variability on one frozen test split;\nno population inference.", ha="center", fontsize=10)
        plot.tight_layout(rect=(0, 0.09, 1, 0.94))
        buffer = io.BytesIO()
        metadata = {"Creator": "AutoResearchClaw", "CreationDate": None, "ModDate": None} if format == "pdf" else {"Software": "AutoResearchClaw"}
        plot.savefig(buffer, format=format, metadata=metadata)
        return buffer.getvalue()


def calibration_chart_bytes(figure: dict, format: str) -> bytes:
    """Matched per-seed expected calibration error; ECE lies in [0, 1]."""
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from textwrap import fill
    settings = {key: value for key, value in matplotlib.rcParamsDefault.items() if key != "backend"}
    settings.update({"font.family": "DejaVu Sans", "font.size": 11, "text.usetex": False})
    with matplotlib.rc_context(settings):
        plot = Figure(figsize=(6, 4.8), dpi=160)
        FigureCanvasAgg(plot)
        left, right = plot.subplots(1, 2)
        rows = figure["rows"]
        for row in rows:
            left.plot([0, 1], [row["baseline_ece"], row["candidate_ece"]], color="#a6adb5", alpha=0.7, linewidth=1)
        left.scatter([0] * len(rows), [r["baseline_ece"] for r in rows], color="#345b9a", label="Baseline", zorder=3)
        left.scatter([1] * len(rows), [r["candidate_ece"] for r in rows], color="#c67520", label="Candidate", zorder=3)
        left.set_xticks([0, 1], [fill(figure["baseline"], 18), fill(figure["candidate"], 18)])
        left.set_ylabel("Expected calibration error")
        left.set_ylim(-0.02, 1.02)
        left.set_yticks([i / 5 for i in range(6)])
        left.set_title("Matched seed ECE")
        differences = [row["difference"] for row in rows]
        right.axhline(0, color="#555555", linewidth=0.8)
        right.scatter(range(len(rows)), differences, color="#345b9a", zorder=3)
        right.axhline(figure["mean_difference"], color="#c67520", linestyle="--", label="Observed mean")
        extent = max([abs(v) for v in differences] + [0.01]) * 1.2
        right.set_ylim(-extent, extent)
        right.set_xticks(range(len(rows)), [r["seed"] for r in rows], rotation=45 if len(rows) > 8 else 0)
        right.set_xlabel("Training seed")
        right.set_ylabel("Candidate minus baseline ECE")
        right.set_title("Paired ECE differences")
        right.legend(loc="best", fontsize=10)
        for ax in (left, right):
            ax.grid(axis="y", color="#e5e7eb", linewidth=0.5)
            ax.spines[["top", "right"]].set_visible(False)
        plot.suptitle(fill(f"{figure['question']} / {figure['dataset']} / {figure['metric']}: calibration", 60), fontsize=12)
        plot.text(0.5, 0.015, f"ECE over {figure['bins']} equal-width score bins per training seed;\n"
                             "lower is better. One frozen test split; no population claim.", ha="center", fontsize=10)
        plot.tight_layout(rect=(0, 0.09, 1, 0.94))
        buffer = io.BytesIO()
        metadata = {"Creator": "AutoResearchClaw", "CreationDate": None, "ModDate": None} if format == "pdf" else {"Software": "AutoResearchClaw"}
        plot.savefig(buffer, format=format, metadata=metadata)
        return buffer.getvalue()


def learning_curve_chart_bytes(figure: dict, format: str) -> bytes:
    """Hash-bound experiment telemetry, with all seed traces retained."""
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from textwrap import fill
    settings = {key: value for key, value in matplotlib.rcParamsDefault.items() if key != "backend"}
    settings.update({"font.family": "DejaVu Sans", "font.size": 11, "text.usetex": False})
    with matplotlib.rc_context(settings):
        plot = Figure(figsize=(6.4, 4.8), dpi=160)
        FigureCanvasAgg(plot)
        axis = plot.subplots()
        seeds = list(dict.fromkeys(row["seed"] for row in figure["rows"]))
        steps = sorted({row["step"] for row in figure["rows"]})
        by_seed = {seed: [row for row in figure["rows"] if row["seed"] == seed] for seed in seeds}
        colors = {"baseline": "#345b9a", "candidate": "#c67520"}
        for seed in seeds:
            for method in ("baseline", "candidate"):
                axis.plot([row["step"] for row in by_seed[seed]], [row[method] for row in by_seed[seed]],
                          color=colors[method], alpha=0.18, linewidth=0.9)
        for method, label in (("baseline", figure["baseline"]), ("candidate", figure["candidate"])):
            means = [statistics.fmean(row[method] for row in figure["rows"] if row["step"] == step)
                     for step in steps]
            axis.plot(steps, means, color=colors[method], linewidth=2.2, marker="o", markersize=3,
                      label=fill(label, 28))
        axis.set_xlabel("Declared training step")
        axis.set_ylabel(f"{figure['curve_metric']} ({figure['curve_split']})")
        axis.set_title(fill(f"{figure['question']} / {figure['dataset']}", 68))
        axis.grid(color="#e5e7eb", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="best", fontsize=9)
        plot.text(0.5, 0.015, "Thin lines are frozen per-seed telemetry; bold lines are observed seed means.\n"
                             "Host schema-validated, not independently recomputed; no convergence claim.",
                  ha="center", fontsize=9)
        plot.tight_layout(rect=(0, 0.1, 1, 0.98))
        buffer = io.BytesIO()
        metadata = {"Creator": "AutoResearchClaw", "CreationDate": None, "ModDate": None} \
            if format == "pdf" else {"Software": "AutoResearchClaw"}
        plot.savefig(buffer, format=format, metadata=metadata)
        return buffer.getvalue()


def pareto_chart_bytes(figure: dict, format: str) -> bytes:
    """Efficiency positions of the two declared methods; wall-clock on one host."""
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from textwrap import fill
    settings = {key: value for key, value in matplotlib.rcParamsDefault.items() if key != "backend"}
    settings.update({"font.family": "DejaVu Sans", "font.size": 11, "text.usetex": False})
    with matplotlib.rc_context(settings):
        plot = Figure(figsize=(6, 4.8), dpi=160)
        FigureCanvasAgg(plot)
        axis = plot.subplots()
        rows, means = figure["rows"], figure["means"]
        colors = {"baseline": "#345b9a", "candidate": "#c67520"}
        # Faint per-seed observations show spread; no aggregation is invented.
        for method in ("baseline", "candidate"):
            axis.scatter([r[f"{method}_seconds"] for r in rows], [r[method] for r in rows],
                         s=18, color=colors[method], alpha=0.45, zorder=2)
        for method in ("baseline", "candidate"):
            axis.scatter([means[method]["seconds"]], [means[method]["metric"]], s=90, marker="D",
                         color=colors[method], zorder=4, edgecolors="black", linewidths=0.6,
                         label=f"{figure[method]} (mean)")
        # Declared direction, not an observed ranking: with two methods the
        # frontier is their non-dominated subset, never a fitted curve.
        low, high = (min(m["seconds"] for m in means.values()), max(m["seconds"] for m in means.values()))
        metric_values = [m["metric"] for m in means.values()] + [v for r in rows for v in (r["baseline"], r["candidate"])]
        axis.set_xlim(0, low + (high - low) * 2 + (high or 1) * 0.2)
        pad = max(abs(v) for v in metric_values) * 0.15 if metric_values else 0.1
        axis.set_ylim(min(metric_values) - pad, max(metric_values) + pad)
        axis.annotate("", xy=(0.32, 0.96), xytext=(0.45, 0.96), xycoords="axes fraction",
                      arrowprops=dict(arrowstyle="->", color="#555555"))
        axis.text(0.46, 0.955, f"better: fewer seconds, {figure['metric_direction']} metric",
                  transform=axis.transAxes, fontsize=9, va="center", color="#555555")
        axis.set_xlabel("Wall-clock seconds per cell (frozen host ledger)")
        axis.set_ylabel(fill(f"{figure['metric']} ({figure['unit']})", 30))
        axis.legend(loc="best", fontsize=10)
        axis.grid(color="#e5e7eb", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        plot.suptitle(fill(f"{figure['question']} / {figure['dataset']}: efficiency", 60), fontsize=12)
        plot.text(0.5, 0.015, "Single-host wall-clock seconds from the frozen execution ledger;\n"
                             "no cross-hardware, cost or statistical ranking claim.", ha="center", fontsize=10)
        plot.tight_layout(rect=(0, 0.09, 1, 0.94))
        buffer = io.BytesIO()
        metadata = {"Creator": "AutoResearchClaw", "CreationDate": None, "ModDate": None} if format == "pdf" else {"Software": "AutoResearchClaw"}
        plot.savefig(buffer, format=format, metadata=metadata)
        return buffer.getvalue()


def effect_chart_bytes(figure: dict, format: str) -> bytes:
    """A forest-style comparison of declared effects, never sorted by outcome."""
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from textwrap import fill
    from researchclaw.pipeline.analysis_spec import effect_label_weight
    settings = {key: value for key, value in matplotlib.rcParamsDefault.items() if key != "backend"}
    settings.update({"font.family": "DejaVu Sans", "font.size": 11, "text.usetex": False})
    comparisons = figure["comparisons"]
    weights = [effect_label_weight(item) for item in comparisons]
    centers = [sum(weights[:i]) + weight / 2 for i, weight in enumerate(weights)]
    with matplotlib.rc_context(settings):
        height = max(4.8, 1.8 + 0.22 * sum(weights))
        plot = Figure(figsize=(6, height), dpi=160)
        FigureCanvasAgg(plot)
        axis = plot.subplots()
        axis.axvline(0, color="#555555", linewidth=0.8)
        endpoints, labels, interval_drawn = [0], [], False
        for index, item in enumerate(comparisons):
            summary, pairs = item["statistics"], item["pairs"]
            values = [row["difference"] for row in pairs]
            # The tiny vertical offsets separate identical marks without
            # changing any observed effect or pretending to add observations.
            center = centers[index]
            positions = [center + (i - (len(pairs) - 1) / 2) * 0.16 / max(1, len(pairs) - 1) for i in range(len(pairs))]
            axis.scatter(values, positions, s=15, color="#8e969e", alpha=0.65,
                         label="Matched seed differences" if index == 0 else None)
            interval = summary["interval"]
            if interval["status"] == "computed_conditional":
                axis.plot([interval["low"], interval["high"]], [center, center], color="#c67520", linewidth=2,
                          label="Conditional seed-mean interval" if not interval_drawn else None)
                endpoints += [interval["low"], interval["high"]]
                interval_drawn = True
                qualifier = f"n={summary['n_pairs']}; {interval['confidence']:.0%} interval"
            else:
                qualifier = f"n={summary['n_pairs']}; interval unavailable" if interval["method"] != "none" else f"n={summary['n_pairs']}; descriptive"
            axis.scatter([summary["mean_difference"]], [center], marker="D", s=38, color="#345b9a", zorder=4,
                         label="Observed paired mean" if index == 0 else None)
            endpoints += [*values, summary["mean_difference"]]
            labels.append(fill(f"{item['candidate']} - {item['baseline']}", 23) + "\n" + qualifier)
        extent = max(max(map(abs, endpoints)), 0.01 if figure["unit"] == "fraction" else 0.1) * 1.2
        axis.set_xlim(-extent, extent)
        axis.set_ylim(sum(weights), 0)
        axis.set_yticks(centers, labels)
        axis.set_xlabel(fill(f"Candidate minus baseline: {figure['metric']} ({figure['unit']})", 30))
        axis.grid(axis="x", color="#e5e7eb", linewidth=0.5)
        axis.spines[["top", "right", "left"]].set_visible(False)
        # Figure coordinates reserve separate physical space for legend, axis
        # labels and scope text; an axes-relative legend overlaps short plots.
        handles, legend_labels = axis.get_legend_handles_labels()
        legend = plot.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, 0.48 / height), fontsize=10)
        plot.suptitle(fill(f"{figure['question']} / {figure['dataset']}: paired effects", 60), fontsize=12)
        scope_text = plot.text(0.5, 0.08 / height, "One frozen test split; training-seed uncertainty only.\nNo population or simultaneous-coverage claim.", ha="center", fontsize=10)
        plot.subplots_adjust(left=0.36, right=0.97, bottom=2.2 / height, top=0.87)
        plot.canvas.draw()
        renderer = plot.canvas.get_renderer()
        label_bounds = [label.get_window_extent(renderer) for label in axis.get_yticklabels()]
        left = max(0.36, (max(b.width for b in label_bounds) + 26) / plot.bbox.width)
        if left > 0.64:
            raise AssetError("Effect plot labels leave insufficient plotting width")
        required_axis_pixels = max((b.height + 16) / weight for b, weight in zip(label_bounds, weights)) * sum(weights)
        height = max(height, (required_axis_pixels / plot.dpi + 2.2) / 0.87)
        plot.set_size_inches(6, height)
        legend.set_bbox_to_anchor((0.5, 0.48 / height), transform=plot.transFigure)
        scope_text.set_position((0.5, 0.08 / height))
        plot.subplots_adjust(left=left, right=0.97, bottom=2.2 / height, top=0.87)
        plot.canvas.draw()
        renderer = plot.canvas.get_renderer()
        legend_bounds = legend.get_window_extent(renderer)
        if (legend_bounds.overlaps(axis.xaxis.label.get_window_extent(renderer))
                or legend_bounds.overlaps(scope_text.get_window_extent(renderer))):
            raise AssetError("Effect plot legend overlaps labels or inference scope")
        label_bounds = [label.get_window_extent(renderer) for label in axis.get_yticklabels()]
        if any(a.overlaps(b) for i, a in enumerate(label_bounds) for b in label_bounds[i + 1:]):
            raise AssetError("Effect plot comparison labels overlap")
        for label in [*axis.get_yticklabels(), axis.xaxis.label, scope_text]:
            bounds = label.get_window_extent(renderer)
            if bounds.x0 < 0 or bounds.y0 < 0 or bounds.x1 > plot.bbox.width or bounds.y1 > plot.bbox.height:
                raise AssetError("Effect plot text exceeds the publication canvas; split or shorten the declared labels")
        buffer = io.BytesIO()
        metadata = {"Creator": "AutoResearchClaw", "CreationDate": None, "ModDate": None} if format == "pdf" else {"Software": "AutoResearchClaw"}
        plot.savefig(buffer, format=format, metadata=metadata)
        return buffer.getvalue()


REPRODUCE_SCRIPT = '''"""Rebuild and verify publication plots from the bundled authoritative evidence.
Run from an environment with researchclaw and matplotlib installed.
"""
from pathlib import Path
from researchclaw.pipeline.publication_assets import prepare_assets

if __name__ == "__main__":
    prepare_assets(Path(__file__).resolve().parent.parent)
'''


def prepare_assets(root: Path) -> dict:
    import matplotlib
    from researchclaw.pipeline.analysis_spec import prepare_analysis, AnalysisError
    try:
        prepare_analysis(root)
    except AnalysisError as exc:
        raise AssetError(str(exc)) from exc
    spec = content_spec(root)
    directory = root / "publication_assets"
    directory.mkdir(parents=True, exist_ok=True)
    path = root / "publication_assets.json"
    previous = _read(path) if path.is_file() else {}
    if previous:
        try:
            verify_assets(root)
            if previous.get("backend") == _backend():
                return previous
        except (OSError, ValueError, KeyError, TypeError):
            pass
        write_json(root / "evidence_artifacts/publication_history" / f"{content_hash(previous)}.json", previous)
    write_json(path, {"status": "building"})
    outputs = {}
    for figure in spec["figures"]:
        for extension, data in (("csv", chart_csv(figure).encode("utf-8")),
                                ("png", chart_bytes(figure, "png")), ("pdf", chart_bytes(figure, "pdf"))):
            target = directory / (figure["id"] + "." + extension)
            target.write_bytes(data)
            outputs[target.relative_to(root).as_posix()] = file_hash(target)
            if extension in {"png", "pdf"}:
                _RENDER_DIGESTS[content_hash(figure), extension, content_hash(_backend())] = hashlib.sha256(data).hexdigest()
    from researchclaw.pipeline.diagram_spec import diagram_bytes, diagram_visual_review
    visual_reviews = {}
    for diagram in spec["diagrams"]:
        visual_reviews[diagram["id"]] = diagram_visual_review(diagram)
        for extension in ("svg", "png", "pdf"):
            data = diagram_bytes(diagram, extension)
            target = directory / (diagram["id"] + "." + extension)
            target.write_bytes(data)
            outputs[target.relative_to(root).as_posix()] = file_hash(target)
            _RENDER_DIGESTS[content_hash(diagram), extension, content_hash(_backend())] = hashlib.sha256(data).hexdigest()
    script = directory / "reproduce.py"
    script.write_text(REPRODUCE_SCRIPT, encoding="utf-8", newline="")
    outputs[script.relative_to(root).as_posix()] = file_hash(script)
    report = {"spec": spec, "outputs": outputs, "backend": _backend(),
              "status": "rendered", "visual_review": "machine_checked_geometry",
              "visual_reviews": visual_reviews,
              "semantic_scope": "deterministic_values_labels_and_declared_topology"}
    report["version"] = content_hash(report)
    write_json(path, report)
    return report


def verify_assets(root: Path) -> dict:
    report = _read(root / "publication_assets.json")
    if report.get("spec", {}).get("figures") and not (root / "analysis_spec.json").is_file():
        raise AssetError("Publication figures require their bound AnalysisSpec")
    payload = dict(report)
    if payload.pop("version", None) != content_hash(payload) or report.get("spec") != content_spec(root):
        raise AssetError("Publication source or specification changed")
    if report.get("backend") != _backend():
        raise AssetError("Plot renderer environment changed; regenerate assets and dependent manuscript")
    expected = {"publication_assets/reproduce.py"}
    for figure in report["spec"]["figures"]:
        expected.update(f"publication_assets/{figure['id']}.{ext}" for ext in ("png", "pdf", "csv"))
        if (root / f"publication_assets/{figure['id']}.csv").read_bytes() != chart_csv(figure).encode("utf-8"):
            raise AssetError("Plot data changed independently of evaluated results")
        for format in ("png", "pdf"):
            name = f"publication_assets/{figure['id']}.{format}"
            if file_hash(root / name) != _render_digest(figure, format):
                raise AssetError("Figure differs from deterministic rendering of its evaluated source")
    from researchclaw.pipeline.diagram_spec import diagram_visual_review
    expected_reviews = {}
    for diagram in report["spec"]["diagrams"]:
        expected_reviews[diagram["id"]] = diagram_visual_review(diagram)
        for format in ("svg", "png", "pdf"):
            name = f"publication_assets/{diagram['id']}.{format}"
            expected.add(name)
            if file_hash(root / name) != _render_digest(diagram, format):
                raise AssetError("Diagram differs from deterministic rendering of MethodSpec")
    if (set(report["outputs"]) != expected or report.get("status") != "rendered"
            or report.get("visual_review") != "machine_checked_geometry"
            or report.get("visual_reviews") != expected_reviews
            or report.get("semantic_scope") != "deterministic_values_labels_and_declared_topology"):
        raise AssetError("Missing publication outputs")
    for name, digest in report["outputs"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or file_hash(path) != digest:
            raise AssetError("Publication output changed or missing: " + name)
    if (root / "publication_assets/reproduce.py").read_text(encoding="utf-8") != REPRODUCE_SCRIPT:
        raise AssetError("Reproduction script changed")
    return report


def render_asset_sections(root: Path) -> dict[str, tuple[str, str]]:
    """Return paired fragments from the same checked specification."""
    from researchclaw.pipeline.manuscript import _md, _tex
    spec = verify_assets(root)["spec"]
    fragments = {}
    md, tex = "", ""
    if spec["algorithm"]:
        algorithm = spec["algorithm"]
        md += "### Declared variables and algorithm\n\n"
        tex += "\\subsection{Declared variables and algorithm}\n"
        for name, variable in algorithm["variables"].items():
            text = f"{name}: {variable['description']}; shape {variable['shape']}."
            md += _md(text) + "\n\n"
            tex += _tex(text) + "\n\n"
        for equation in spec["equations"]:
            md += f"Equation {equation['id']}\n\n$$\n{equation['latex']}\n$$\n\n"
            tex += "\\begin{equation}\\label{" + equation["id"] + "}\n" + equation["latex"] + "\n\\end{equation}\n"
        tex += "\\begin{enumerate}\n"
        for index, step in enumerate(algorithm["steps"], 1):
            text = (f"[{step['phase']}] {step['id']}: {step['description']}. Inputs: {', '.join(step['inputs'])}; "
                    f"outputs: {', '.join(step['outputs'])}; depends on: {', '.join(step['depends_on']) or 'none'}. "
                    f"Equations: {', '.join('eq-' + eid for eid in step['equations']) or 'none'}. "
                    f"Implementation: {step['code_file']}::{step['code_symbol']}.")
            md += f"{index}. " + _md(text) + "\n\n"
            # A leading [phase] is prose, not LaTeX's optional item label.
            tex += "\\item{} " + _tex(text) + "\n"
        tex += "\\end{enumerate}\n"
        text = (f"Loss equations: {', '.join('eq-' + eid for eid in algorithm['losses']) or 'none'}. "
                f"Stopping rule: {algorithm['stopping_rule']}. Declared complexity: time {algorithm['complexity']['time']}; "
                f"space {algorithm['complexity']['space']}. Code/method semantic equivalence: "
                f"{algorithm['semantic_equivalence']}"
                + (f" (reviewer: {algorithm['semantic_reviewer']})." if algorithm.get("semantic_reviewer") else ".")
                + " An informal review is not a machine proof.")
        md += _md(text) + "\n\n"
        tex += _tex(text) + "\n\n"
        for diagram in (item for item in spec["diagrams"] if "source_method_version" in item):
            page = (f" Page {diagram['page']} of {diagram['page_count']}."
                    if diagram.get("page_count", 1) > 1 else "")
            semantics = ("Arrows are dependencies, not tensor flow or execution order. " if diagram["kind"] == "architecture"
                         else "Variable-to-step arrows are declared reads and step-to-variable arrows are declared writes; "
                              "they are not observed runtime tensors. " if diagram["kind"] == "data_flow"
                         else "True/false arrows are declared decisions; dashed edges are loops. ")
            caption = (f"Declared {diagram['kind']} of {diagram['method_id']}.{page} " + semantics
                       + "Continuation annotations preserve cross-page edges. Node colors distinguish training and inference. "
                         "Code equivalence and termination are unverified.")
            md += f"![{_md(caption)}](publication_assets/{diagram['id']}.png)\n\n"
            tex += ("\\begin{figure}[!htbp]\n\\centering\n\\includegraphics[width=\\linewidth,height=0.8\\textheight,keepaspectratio]{publication_assets/"
                    + diagram["id"] + ".pdf}\n\\caption{" + _tex(caption) + "}\\label{fig:" + diagram["id"] + "}\n\\end{figure}\n")
        fragments["methods"] = md, tex
    split_diagrams = [item for item in spec["diagrams"] if item.get("kind") == "dataset_split"]
    if split_diagrams:
        md, tex = "", ""
        for diagram in split_diagrams:
            caption = (f"Frozen split for {diagram['dataset']}. Counts and split strategy come from the verified "
                       "ResearchContract. Test labels remain in a host-only evidence artifact; this file separation "
                       "does not establish operating-system isolation.")
            md += f"![{_md(caption)}](publication_assets/{diagram['id']}.png)\n\n"
            tex += ("\\begin{figure}[!htbp]\n\\centering\n\\includegraphics[width=\\linewidth,height=0.8\\textheight,keepaspectratio]{publication_assets/"
                    + diagram["id"] + ".pdf}\n\\caption{" + _tex(caption) + "}\\label{fig:" + diagram["id"] + "}\n\\end{figure}\n")
        fragments["experiments"] = md, tex
    overview = next((item for item in spec["diagrams"] if item.get("kind") == "research_overview"), None)
    if overview is not None:
        caption = ("Frozen research overview derived from the ResearchContract and ExperimentProtocol. "
                   "Arrows show declared inputs, comparisons and host evaluation flow; they do not establish "
                   "causality, implementation fidelity or scientific validity.")
        md = f"![{_md(caption)}](publication_assets/{overview['id']}.png)\n\n"
        tex = ("\\begin{figure}[!htbp]\n\\centering\n\\includegraphics[width=\\linewidth,height=0.8\\textheight,keepaspectratio]{publication_assets/"
               + overview["id"] + ".pdf}\n\\caption{" + _tex(caption) + "}\\label{fig:" + overview["id"] + "}\n\\end{figure}\n")
        fragments["introduction"] = md, tex
    md, tex = "", ""
    for proof in spec["proofs"]:
        heading = "Obligation " + proof["id"] + " - " + proof["status"]
        md += "### " + _md(heading) + "\n\n"
        tex += "\\subsection{" + _tex(heading) + "}\n"
        text = (f"Definitions: {json.dumps(proof['definitions'], ensure_ascii=False)}. "
                f"Assumptions: {'; '.join(proof['assumptions']) or 'none declared'}. "
                f"Depends on: {', '.join(proof['depends_on']) or 'none'}. Checker: {proof['checker'] or 'unavailable'}.")
        md += _md(text) + "\n\n"
        tex += _tex(text) + "\n\n"
        if proof["latex"]:
            md += "$$\n" + proof["latex"] + "\n$$\n\n"
            tex += "\\begin{equation}\n" + proof["latex"] + "\n\\end{equation}\n"
        else:
            typed = proof["statement"]
            if "text" in typed:
                statement = typed["text"]
            elif typed.get("kind") in {"rational_identity", "symbolic_equality"}:
                statement = f"Typed {typed['kind']}: {typed['left']} = {typed['right']}."
            elif typed.get("kind") == "linear_arithmetic":
                statement = ("Typed linear arithmetic implication: "
                             + json.dumps(typed, ensure_ascii=False, sort_keys=True) + ".")
            elif typed.get("kind") == "conjunction":
                statement = f"Typed conjunction of {len(typed['parts'])} generated proof obligations."
            else:
                statement = json.dumps(typed, ensure_ascii=False, sort_keys=True)
            md += _md(statement) + "\n\n"
            tex += _tex(statement) + "\n\n"
        detail = proof["proof_text"] or json.dumps(proof["evidence"], ensure_ascii=False, sort_keys=True)
        md += _md(detail) + "\n\n"
        tex += "\\begingroup\\raggedright " + _tex(detail) + "\\par\\endgroup\n\n"
    if md:
        fragments["theory"] = md, tex
    md, tex = "", ""
    for figure in spec["figures"]:
        if figure.get("kind") == "effect_summary":
            caption = (f"{figure['question']} on {figure['dataset']}: paired effects in declared comparison order, "
                       f"metric {figure['metric']} in {figure['unit']}. Gray points are matched training seeds; diamonds are observed means. "
                       "Intervals, when available, are predeclared paired-seed bootstrap intervals conditional on exchangeability. "
                       "Positive differences need not mean improvement. No population, significance or simultaneous-coverage claim is implied.")
        elif figure.get("kind") == "learning_curve":
            caption = (f"{figure['question']} on {figure['dataset']}: declared {figure['curve_split']} telemetry "
                       f"for {figure['curve_metric']}, comparing {figure['candidate']} with {figure['baseline']}. "
                       "Thin lines retain every training seed and bold lines are observed seed means. The frozen experiment "
                       "process emitted these values; the host validated and hash-bound them but did not independently recompute "
                       "the training metric. No test-performance or convergence claim is implied.")
        else:
            caption = (f"{figure['question']} on {figure['dataset']}: {figure['candidate']} versus {figure['baseline']}, "
                       f"metric {figure['metric']} in {figure['unit']}. Each point is a training seed; "
                       "the difference is candidate minus baseline. No population uncertainty or significance is implied.")
        md += f"![{_md(caption)}](publication_assets/{figure['id']}.png)\n\n"
        tex += ("\\begin{figure}[!htbp]\n\\centering\n\\includegraphics[width=\\linewidth,height=0.8\\textheight,keepaspectratio]{publication_assets/"
                + figure["id"] + ".pdf}\n\\caption{" + _tex(caption) + "}\\label{fig:" + figure["id"] + "}\n\\end{figure}\n")
    if md:
        fragments["results"] = md, tex
    return fragments
