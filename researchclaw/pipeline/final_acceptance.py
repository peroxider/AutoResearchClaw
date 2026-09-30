"""Fail-closed acceptance of the exact delivered manuscript version.

Unknown dimensions cannot be compensated by a high writing score. Reviews
must name their checker and bind the complete input inventory. This module
does not pretend that compilation is visual PDF review, or that a metadata
lookup establishes citation support.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from researchclaw.pipeline.evidence_store import EvidenceStore, content_hash, file_hash
from researchclaw.pipeline.resource_ledger import build_resource_ledger, ledger_issues

RESEARCH_DIMENSIONS = ("data", "experiments", "numeric", "citations", "theory")
PRESENTATION_DIMENSIONS = ("quality", "consistency", "figures", "layout")
FINAL_ACCEPTANCE_CHECKER = "final_acceptance/v2"
_GENERATED = {"manifest.json", "final_acceptance.json", "final_reviews.json", "resource_ledger.json"}


def inventory(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): file_hash(p)
            for p in sorted(root.rglob("*")) if p.is_file()
            and p.relative_to(root).as_posix() not in _GENERATED}


def compilation_inputs(root: Path) -> dict[str, str]:
    """All locally bundled inputs that may affect the PDF."""
    return {name: digest for name, digest in inventory(root).items()
            if Path(name).suffix.lower() in {
                ".tex", ".bib", ".sty", ".cls", ".bst", ".bbx", ".cbx", ".lbx", ".png", ".jpg", ".jpeg",
                ".svg", ".eps", ".pdf", ".bbl", ".def", ".clo", ".cfg",
            } and name != "paper.pdf"}


def _read(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        return result if isinstance(result, dict) else {}
    except (OSError, ValueError):
        return {}


def _citations(text: str) -> set[str]:
    keys = set()
    for match in re.finditer(r"\\cite\w*\*?(?:\[[^\]]*\]){0,2}\{([^}]+)\}", text):
        keys.update(k.strip() for k in match[1].split(","))
    for match in re.finditer(r"\[@([^\]]+)\]", text):
        keys.update(k.strip().lstrip("@") for k in match[1].split(";"))
    # Legacy Markdown citations use [author2024key].
    keys.update(re.findall(r"\[([A-Za-z]+\d{4}[A-Za-z0-9_-]*)\]", text))
    return keys


def _has_placeholder(text: str) -> bool:
    if re.search(r"\b(?:TODO|TBD|PLACEHOLDER)\b|\[CITATION NEEDED\]", text):
        return True
    for line in text.splitlines():
        # Markdown table delimiter rows contain dashes by definition. A dash
        # cell in a data row (or a TeX row) still represents missing evidence.
        cells = line.strip().strip("|").split("|")
        if len(cells) > 1 and all(re.fullmatch(r"\s*:?-{3,}:?\s*", cell) for cell in cells):
            continue
        if re.search(r"[&|]\s*---\s*[&|]", line):
            return True
    return False


def _image_ledger_problem(root: Path, manifest_path: Path) -> tuple[str, str] | None:
    """Every provider attempt frozen in the diagram manifest must be covered
    by the run-level image call ledger. Stage retries leave earlier attempts
    as extra ledger records, so the check is containment, not equality.
    Returns (reason, artifact) for an acceptance issue, or None."""
    from researchclaw.llm.image_call_ledger import validate_image_call_ledger

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        # The figures dimension already reports the unreadable manifest.
        return None
    attempts = manifest.get("generation_attempts")
    if not isinstance(attempts, list) or not attempts:
        return None
    ledger_path = root / "image_call_ledger.json"
    if not ledger_path.is_file():
        return ("image_model_calls_missing_run_ledger", "image_call_ledger.json")
    try:
        document = json.loads(ledger_path.read_text(encoding="utf-8"))
        validate_image_call_ledger(document)
        available = [(record.get("provider") or "", record.get("status") or "")
                     for record in document["calls"]]
        for entry in attempts:
            if not isinstance(entry, dict):
                raise ValueError("Image call ledger cross-check is malformed")
            attempt = (str(entry.get("provider", "")), str(entry.get("status", "")))
            if attempt not in available:
                return ("image_model_calls_not_fully_ledgered", "image_call_ledger.json")
            available.remove(attempt)
    except (OSError, ValueError, TypeError):
        return ("invalid_image_call_ledger", "image_call_ledger.json")
    return None


def _nano_banana_ledger_problem(root: Path) -> tuple[str, str] | None:
    """Every figure the Nano Banana orchestrator froze must be covered by a
    run-level image call ledger attempt with the same figure id and outcome
    (SDK fallback to REST leaves both attempts in the ledger; the figure
    needs one matching its final status). Returns (reason, artifact) or None."""
    from researchclaw.llm.image_call_ledger import validate_image_call_ledger

    results_files = sorted(root.glob("stage-*/nano_banana_results.json"))
    if not results_files:
        return None
    ledger_path = root / "image_call_ledger.json"
    if not ledger_path.is_file():
        return ("nano_banana_calls_missing_run_ledger", "image_call_ledger.json")
    try:
        document = json.loads(ledger_path.read_text(encoding="utf-8"))
        validate_image_call_ledger(document)
        available = [(record.get("figure_id"), record.get("status"))
                     for record in document["calls"]
                     if record.get("figure_id") is not None]
        for results_path in results_files:
            results = json.loads(results_path.read_text(encoding="utf-8"))
            generated = results.get("generated") if isinstance(results, dict) else None
            if not isinstance(generated, list):
                return ("invalid_nano_banana_results", results_path.name)
            for entry in generated:
                if not isinstance(entry, dict):
                    return ("invalid_nano_banana_results", results_path.name)
                status = "succeeded" if entry.get("success") else "failed"
                attempt = (entry.get("figure_id"), status)
                if attempt not in available:
                    return ("nano_banana_calls_not_fully_ledgered",
                            "image_call_ledger.json")
                available.remove(attempt)
    except (OSError, ValueError, TypeError):
        return ("invalid_image_call_ledger", "image_call_ledger.json")
    return None


def _assess_delivery(root: Path, *, target_status: str = "exploratory",
                     quality_threshold: float = 7.0) -> dict[str, Any]:
    hashes = inventory(root)
    version = content_hash(hashes)
    reviews = _read(root / "final_reviews.json")
    dimensions: dict[str, str] = {}
    issues: list[dict[str, str]] = []

    def issue(dimension: str, reason: str, artifact: str = "") -> None:
        dimensions[dimension] = "failed"
        issues.append({"dimension": dimension, "reason": reason, "artifact": artifact,
                       "repair_owner": {"numeric": "analysis", "citations": "literature",
                           "layout": "export", "consistency": "writing",
                           "figures": "figures", "quality": "writing"}.get(dimension, dimension)})

    review_bound = reviews.get("input_version") == version
    for dimension in (*RESEARCH_DIMENSIONS, *PRESENTATION_DIMENSIONS):
        review = reviews.get("dimensions", {}).get(dimension, {}) if review_bound else {}
        status = review.get("status")
        valid = bool(review.get("checker") and review.get("evidence"))
        # Only explicitly justified absence of theoretical claims can be N/A.
        dimensions[dimension] = (status if valid and status in {"passed", "failed"}
                                 else "not_applicable" if valid and status == "not_applicable"
                                 and dimension == "theory" else "unknown")
        if dimensions[dimension] not in {"passed", "not_applicable"}:
            issues.append({"dimension": dimension, "reason": "missing_failed_or_stale_review",
                           "artifact": "final_reviews.json", "repair_owner": dimension})

    input_contract = None
    if (root / "research_contract.json").is_file() or (root / "data_preflight.json").is_file():
        from researchclaw.research_inputs import verify_bundle_contract
        try:
            input_contract = verify_bundle_contract(root)
            preflight = _read(root / "data_preflight.json")
            if (preflight.get("status") != "verified"
                    or preflight.get("contract_version") != input_contract["version"]):
                issue("data", "missing_failed_or_stale_data_preflight", "data_preflight.json")
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            issue("data", "invalid_or_changed_data_contract", "research_contract.json")
    else:
        # Status is earned independently of target_status. Legacy bundles remain
        # exportable, but an expert's positive review cannot substitute for the
        # frozen data/protocol and exhaustive shared-source manuscript checks.
        issue("data", "structured_data_contract_required", "research_contract.json")

    text = {}
    for name in ("paper.tex", "paper_final.md"):
        path = root / name
        # Spans are offsets in the exact export, including any CRLF inherited
        # from an imported template. Universal-newline translation shifts them.
        text[name] = path.read_bytes().decode("utf-8") if path.is_file() else ""
        if not text[name].strip():
            issue("consistency", "missing_manuscript", name)
        if _has_placeholder(text[name]):
            issue("consistency", "unresolved_placeholder", name)
    bib = (root / "references.bib")
    bib_text = bib.read_text(encoding="utf-8") if bib.is_file() else ""
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib_text))
    if not bib_keys:
        issue("citations", "missing_bibliography", "references.bib")
    for name, body in text.items():
        for key in sorted(_citations(body) - bib_keys):
            issue("citations", f"unresolved_citation:{key}", name)
    for match in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text["paper.tex"]):
        name = match[1]
        paths = [root / name] + [root / (name + suffix) for suffix in (".pdf", ".png", ".jpg", ".eps")]
        if not any(p.resolve().is_relative_to(root.resolve()) and p.is_file() for p in paths):
            issue("figures", f"missing_figure:{name}", "paper.tex")
    for name in re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", text["paper_final.md"]):
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            issue("figures", f"missing_figure:{name}", "paper_final.md")
    labels = set(re.findall(r"\\label\{([^}]+)\}", text["paper.tex"]))
    for ref in re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", text["paper.tex"]):
        if ref not in labels:
            issue("consistency", f"unresolved_reference:{ref}", "paper.tex")

    compilation = _read(root / "compilation.json")
    pdf = root / "paper.pdf"
    if (compilation.get("success") is not True or not pdf.is_file()
            or compilation.get("inputs") != compilation_inputs(root)
            or compilation.get("pdf_sha256") != (file_hash(pdf) if pdf.is_file() else None)):
        issue("layout", "missing_failed_or_stale_compilation", "compilation.json")
    elif not pdf.read_bytes().startswith(b"%PDF-"):
        issue("layout", "invalid_pdf", "paper.pdf")
    else:
        try:
            import fitz
            with fitz.open(pdf) as document:
                if not document.is_pdf or document.needs_pass or document.page_count < 1:
                    raise ValueError("Unreadable PDF")
                for page in document:
                    page.get_text()
        except ImportError:
            issue("layout", "pdf_parser_unavailable", "paper.pdf")
        except (RuntimeError, ValueError, OSError):
            issue("layout", "invalid_pdf", "paper.pdf")

    gate = _read(root / "quality_report.json")
    if (root / "analysis_spec.json").is_file() or ((root / "manuscript_ir.json").is_file()
                                                  and (root / "experiment_protocol.json").is_file()):
        from researchclaw.pipeline.analysis_spec import verify_analysis
        try:
            verify_analysis(root)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
            issue("numeric", "invalid_missing_or_stale_analysis_spec", "analysis_spec.json")
    if (root / "publication_template.json").is_file():
        from researchclaw.templates.bundle import inspect_constraints, verify_template_resources
        try:
            verify_template_resources(root)
            constraints = inspect_constraints(root)
            if constraints["status"] != "passed" or _read(root / "template_constraints.json") != constraints:
                issue("layout", "template_constraints_failed_or_stale", "template_constraints.json")
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            issue("layout", "invalid_or_changed_template_bundle", "publication_template.json")
    if (root / "publication_assets.json").is_file():
        from researchclaw.pipeline.publication_assets import verify_assets
        try:
            verify_assets(root)
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            issue("figures", "invalid_or_stale_publication_assets", "publication_assets.json")
    framework_dir = root / "charts"
    framework_image = framework_dir / "framework_diagram.png"
    framework_manifest = framework_dir / "framework_diagram_generation.json"
    if framework_image.is_file() or framework_manifest.is_file():
        from researchclaw.agents.figure_agent.framework_diagram import verify_framework_diagram_artifacts
        try:
            framework = verify_framework_diagram_artifacts(framework_dir)
            if (framework.get("provider") not in {None, "matplotlib"}
                    and not (framework_dir / "framework_diagram_semantic_review.json").is_file()):
                issue("figures", "framework_diagram_independent_visual_review_missing",
                      "charts/framework_diagram_semantic_review.json")
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            issue("figures", "invalid_or_stale_framework_diagram_evidence",
                  "charts/framework_diagram_generation.json")
    if (root / "manuscript_ir.json").is_file():
        from researchclaw.pipeline.manuscript import (
            quality_report, validate_peer_review, verify_exports, verify_manuscript_revision,
        )
        try:
            verify_exports(root)
            if gate != quality_report(root, quality_threshold):
                issue("quality", "stale_section_contract_reviews", "quality_report.json")
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            issue("numeric", "unverified_manuscript_claim_coverage", "manuscript_ir.json")
            issue("consistency", "invalid_or_stale_manuscript_ir", "manuscript_ir.json")
        peer_path, revision_path = root / "manuscript_peer_review.json", root / "manuscript_revision.json"
        try:
            if revision_path.is_file():
                if not peer_path.is_file():
                    raise ValueError("revision lacks peer review")
                verify_manuscript_revision(root)
            elif peer_path.is_file():
                peer = _read(peer_path)
                validate_peer_review(root, peer)
                if peer.get("issues"):
                    issue("quality", "peer_review_issues_not_closed", "manuscript_peer_review.json")
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            issue("quality", "invalid_or_stale_issue_directed_revision", "manuscript_revision.json")
        from researchclaw.pipeline.submission_bundle import verify_submission
        try:
            verify_submission(root)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError):
            issue("consistency", "missing_failed_or_stale_submission_archive", "submission_bundle.json")
        from researchclaw.templates.bundle import verify_template
        try:
            if verify_template(root)["compiled"]["policy"]["anonymous"]:
                review = reviews.get("dimensions", {}).get("anonymity", {}) if review_bound else {}
                dimensions["anonymity"] = (review.get("status") if review.get("checker") and review.get("evidence")
                                          and review.get("status") in {"passed", "failed"} else "unknown")
                if dimensions["anonymity"] != "passed":
                    issues.append({"dimension": "anonymity", "reason": "missing_failed_or_stale_content_anonymity_review",
                                   "artifact": "submission.zip", "repair_owner": "export"})
        except (OSError, ValueError, KeyError, TypeError):
            issue("anonymity", "invalid_anonymous_template_contract", "publication_template.json")
    else:
        issue("numeric", "structured_manuscript_required", "manuscript_ir.json")
    score = gate.get("score_1_to_10", gate.get("score", gate.get("overall_score")))
    if (type(score) not in (int, float) or not math.isfinite(score)
            or score < quality_threshold):
        issue("quality", "missing_or_low_quality_score", "quality_report.json")
    citation_report = _read(root / "verification_report.json")
    if citation_report.get("status") != "verified":
        issue("citations", "citation_metadata_not_verified", "verification_report.json")
    if citation_report.get("requires_claim_review") and dimensions["citations"] != "passed":
        issue("citations", "removed_citations_require_claim_review", "verification_report.json")

    paper_hashes = {name: hashes.get(name) for name in text}
    support = _read(root / "citation_support.json")
    from researchclaw.literature.evidence import citation_support_issues, validate_evidence, coverage_report
    for reason in citation_support_issues(root, support, text):
        issue("citations", reason, "citation_support.json")
    if (root / "literature_evidence.json").is_file():
        literature_valid = False
        try:
            literature = _read(root / "literature_evidence.json")
            validate_evidence(root, literature)
            if coverage_report(literature)["status"] != "review_ready":
                issue("citations", "literature_coverage_unresolved", "literature_evidence.json")
            literature_valid = True
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            issue("citations", "invalid_literature_evidence", "literature_evidence.json")
        if literature_valid:
            from researchclaw.literature.positioning import validate_novelty, contribution_ledger
            try:
                novelty = _read(root / "novelty_matrix.json")
                validate_novelty(root, novelty, literature)
                if novelty["status"] != "reviewed":
                    issue("citations", "novelty_comparison_unresolved", "novelty_matrix.json")
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                issue("citations", "invalid_or_stale_novelty_matrix", "novelty_matrix.json")
            try:
                if contribution_ledger(root, write=False) != _read(root / "contribution_ledger.json"):
                    issue("numeric", "stale_contribution_bindings", "contribution_ledger.json")
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                issue("numeric", "invalid_contribution_evidence", "contribution_ledger.json")
    for key in citation_report.get("removed_citation_keys", []):
        resolution = support.get("removed_claim_resolutions", {}).get(key, {})
        if (support.get("manuscript_hashes") != paper_hashes or not resolution.get("checker")
                or not resolution.get("evidence") or resolution.get("status") not in
                {"claim_removed", "rewritten_with_evidence"}):
            issue("citations", f"removed_claim_unresolved:{key}", "citation_support.json")

    theory = _read(root / "theory_bundle.json")
    if theory.get("schema_version") == 2:
        from researchclaw.pipeline.research_workbench import compile_theory
        try:
            if compile_theory(theory.get("spec")) != theory:
                issue("theory", "changed_or_invalid_proof_check", "theory_bundle.json")
        except (ValueError, TypeError, KeyError):
            issue("theory", "invalid_typed_proof_bundle", "theory_bundle.json")
    if (root / "method_spec.json").is_file():
        from researchclaw.pipeline.research_workbench import compile_method, check_implementation
        try:
            method = _read(root / "method_spec.json")
            if compile_method(method.get("spec")) != method:
                issue("experiments", "invalid_method_spec", "method_spec.json")
            elif check_implementation(method, root / "evidence_artifacts" / "protocol_source")["status"] != "mapped":
                issue("experiments", "method_implementation_mapping_failed", "method_spec.json")
            if "validation" in method.get("spec", {}):
                from researchclaw.pipeline.method_validation import verify_validation
                try:
                    verify_validation(root, method, _read(root / "protocol_code.json"))
                except (ValueError, TypeError, KeyError, OSError):
                    issue("experiments", "method_runtime_validation_failed", "method_validation.json")
            if (root / "method_semantic_review.json").is_file():
                from researchclaw.pipeline.method_semantics import verify_semantic_review
                try:
                    reviewed = verify_semantic_review(root, method, _read(root / "protocol_code.json"))
                    if reviewed["semantic_equivalence"] == "contradicted":
                        issue("experiments", "method_semantic_review_contradicted", "method_semantic_review.json")
                except (ValueError, TypeError, KeyError, OSError):
                    issue("experiments", "invalid_or_stale_method_semantic_review", "method_semantic_review.json")
        except (ValueError, TypeError, KeyError):
            issue("experiments", "invalid_method_spec", "method_spec.json")
    obligations = theory.get("obligations", [])
    if dimensions["theory"] == "passed" and not obligations:
        issue("theory", "missing_proof_obligations", "theory_bundle.json")
    if dimensions["theory"] == "not_applicable" and (
        obligations or re.search(r"\\begin\{(?:theorem|lemma|proposition)\}", text["paper.tex"])
    ):
        issue("theory", "theoretical_claims_cannot_be_exempted", "theory_bundle.json")
    for obligation in obligations:
        if obligation.get("required", True) and (
            obligation.get("status") not in {"machine_checked", "reviewed_informal"}
            or not obligation.get("checker") or not obligation.get("evidence")
        ):
            issue("theory", "unresolved_proof_obligation", "theory_bundle.json")

    evidence_version = None
    try:
        store = EvidenceStore.from_dict(_read(root / "evidence_store.json"))
        evidence_version = store.version
        if input_contract is not None:
            from researchclaw.research_inputs import check_evaluation_binding
            for record in store.records.values():
                dataset = next((d for d in input_contract["datasets"]
                                if d["manifest"]["dataset"] == record.key.dataset), None)
                labels = dataset["labels"] if dataset else ""
                try:
                    check_evaluation_binding(input_contract, record.key, labels,
                                             dict(record.artifacts).get(labels, ""))
                except ValueError as exc:
                    issue("data", str(exc), "evidence_store.json")
        protocol = _read(root / "experiment_protocol.json")
        try:
            from researchclaw.pipeline.experiment_protocol import load_protocol, audit_coverage
            frozen_protocol = load_protocol(root, input_contract)
            if frozen_protocol is None:
                issue("experiments", "frozen_experiment_protocol_required", "experiment_protocol.json")
            else:
                coverage = audit_coverage(root, frozen_protocol, store)
                if coverage["status"] != "complete":
                    issue("experiments", "frozen_experiment_matrix_incomplete", "experiment_protocol.json")
        except (OSError, ValueError, TypeError, KeyError):
            issue("experiments", "invalid_or_changed_experiment_protocol", "experiment_protocol.json")
        required = protocol.get("required_keys", [])
        if not required:
            issue("experiments", "missing_required_experiment_matrix", "experiment_protocol.json")
        available = {content_hash(record.key.__dict__): result_id
                     for result_id, record in store.records.items()}
        for key in required:
            result_id = available.get(content_hash(key))
            if result_id is None or store.validate_record(result_id, root):
                issue("experiments", "required_experiment_missing_or_invalid", "experiment_protocol.json")
        claims = _read(root / "numeric_claims.json")
        if claims.get("evidence_version") != store.version:
            issue("numeric", "stale_evidence_version", "numeric_claims.json")
        expected_papers = paper_hashes
        if claims.get("manuscript_hashes") != expected_papers:
            issue("numeric", "stale_manuscript_bindings", "numeric_claims.json")
        if not claims.get("claims"):
            issue("numeric", "no_bound_numeric_claims", "numeric_claims.json")
        for claim in claims.get("claims", []):
            for reason in store.verify_claim(claim, root):
                issue("numeric", reason, "numeric_claims.json")
            # Bind the rendered number to an exact span in both formats.
            for name, body in text.items():
                span = claim.get("spans", {}).get(name, {})
                start, end = span.get("start"), span.get("end")
                if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(body)
                        or body[start:end] != str(claim.get("rendered"))):
                    issue("numeric", "missing_or_changed_numeric_span", name)
            try:
                if float(claim["rendered"]) != claim["value"]:
                    issue("numeric", "rendered_value_mismatch", "numeric_claims.json")
            except (ValueError, TypeError, KeyError):
                issue("numeric", "invalid_rendered_value", "numeric_claims.json")
    except (ValueError, KeyError, TypeError, AttributeError):
        issue("numeric", "missing_or_invalid_evidence_store", "evidence_store.json")

    # Run-level resource accounting is part of the audited bundle: unreadable
    # sources and exceeded budgets are acceptance issues, not notes. The
    # ledger is derived data, so it is excluded from the input inventory and
    # cannot shift the review binding.
    ledger = build_resource_ledger(root)
    (root / "resource_ledger.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    dimensions["resources"] = "passed"
    for reason in ledger_issues(ledger):
        issue("resources", reason, "resource_ledger.json")
    # Checked here so an image-ledger issue cannot be clobbered by the
    # unconditional "passed" reset above; issue() re-fails the dimension.
    if framework_manifest.is_file():
        problem = _image_ledger_problem(root, framework_manifest)
        if problem is not None:
            issue("resources", problem[0], problem[1])
    problem = _nano_banana_ledger_problem(root)
    if problem is not None:
        issue("resources", problem[0], problem[1])

    # Retry exhaustion or a failed stage is never an earned quality state.
    blockers = _read(root / "pipeline_blockers.json")
    if blockers.get("issues"):
        issue("experiments", "pipeline_has_unresolved_blockers", "pipeline_blockers.json")
    research_ok = (all(dimensions[d] in {"passed", "not_applicable"} for d in RESEARCH_DIMENSIONS)
                   and dimensions.get("resources") == "passed")
    submission_ok = (research_ok and all(dimensions[d] == "passed" for d in PRESENTATION_DIMENSIONS)
                     and dimensions.get("anonymity", "passed") == "passed")
    status = "submission_candidate" if submission_ok else "research_complete" if research_ok else "exploratory"
    rank = {"exploratory": 0, "research_complete": 1, "submission_candidate": 2}
    return {"schema_version": 1, "checker": FINAL_ACCEPTANCE_CHECKER, "input_version": version,
            "evidence_version": evidence_version, "review_version": content_hash(reviews),
            "artifact_status": status, "target_status": target_status,
            "target_met": rank[status] >= rank[target_status], "dimensions": dimensions,
            "issues": issues, "file_hashes": hashes}


def assess_delivery(root: Path, *, target_status: str = "exploratory",
                    quality_threshold: float = 7.0) -> dict[str, Any]:
    if target_status not in {"exploratory", "research_complete", "submission_candidate"}:
        raise ValueError("Invalid target status")
    if (type(quality_threshold) not in (int, float) or not math.isfinite(quality_threshold)
            or not 3 <= quality_threshold <= 10):
        raise ValueError("Quality threshold must be finite and in [3, 10]")
    try:
        return _assess_delivery(root, target_status=target_status, quality_threshold=quality_threshold)
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
        # Malformed evidence is a failed check, never a reason to retain an old PASS.
        return {"schema_version": 1, "checker": FINAL_ACCEPTANCE_CHECKER, "input_version": None,
                "evidence_version": None, "review_version": None,
                "artifact_status": "exploratory", "target_status": target_status,
                "target_met": False, "dimensions": {d: "unknown" for d in
                    (*RESEARCH_DIMENSIONS, *PRESENTATION_DIMENSIONS)},
                "issues": [{"dimension": "integrity", "reason": f"invalid_bundle:{type(exc).__name__}",
                            "artifact": str(root), "repair_owner": "evidence"}], "file_hashes": {}}


def seal_delivery(root: Path, **kwargs: Any) -> dict[str, Any]:
    report = assess_delivery(root, **kwargs)
    (root / "final_acceptance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest = _read(root / "manifest.json")
    manifest.update(artifact_status=report["artifact_status"], target_met=report["target_met"],
                    evidence_version=report["evidence_version"], review_version=report["review_version"])
    manifest["file_hashes"] = {**report["file_hashes"],
                              "final_acceptance.json": file_hash(root / "final_acceptance.json")}
    if (root / "final_reviews.json").is_file():
        manifest["file_hashes"]["final_reviews.json"] = file_hash(root / "final_reviews.json")
    if (root / "resource_ledger.json").is_file():
        manifest["file_hashes"]["resource_ledger.json"] = file_hash(root / "resource_ledger.json")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return report


def validate_seal(root: Path) -> bool:
    manifest = _read(root / "manifest.json")
    actual = {p.relative_to(root).as_posix(): file_hash(p) for p in root.rglob("*")
              if p.is_file() and p != root / "manifest.json"}
    return bool(manifest.get("file_hashes")) and manifest["file_hashes"] == actual


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Audit and seal the exact final manuscript bundle")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--target-status", choices=("exploratory", "research_complete", "submission_candidate"),
                        default="submission_candidate")
    parser.add_argument("--quality-threshold", type=float, default=7.0)
    parser.add_argument("--check-seal", action="store_true")
    parser.add_argument("--print-input-version", action="store_true")
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error("directory does not exist")
    if args.check_seal:
        valid = validate_seal(args.directory)
        print(json.dumps({"seal_valid": valid}))
        return 0 if valid else 2
    if args.print_input_version:
        print(content_hash(inventory(args.directory)))
        return 0
    report = seal_delivery(args.directory, target_status=args.target_status,
                           quality_threshold=args.quality_threshold)
    print(json.dumps({"artifact_status": report["artifact_status"], "target_met": report["target_met"],
                      "issues": report["issues"]}, indent=2))
    return 0 if report["target_met"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
