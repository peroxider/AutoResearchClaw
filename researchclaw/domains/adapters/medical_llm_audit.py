"""Stage-specific guardrails for auditable clinical LLM research."""
from __future__ import annotations
from typing import Any
from researchclaw.domains.prompt_adapter import PromptAdapter, PromptBlocks


class MedicalLLMAuditPromptAdapter(PromptAdapter):
    def get_experiment_design_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        return PromptBlocks(experiment_design_context=(
            "This is an auditable clinical-LLM/agent methods study. BEFORE proposing methods, "
            "freeze a data_contract.json with index time, outcome, allowed/prohibited fields, "
            "identifier-removal rule, external-validation label, and privacy/ethics status. "
            "The supplied DATA CONTRACT is the sole source of truth: do not invent records, notes, "
            "time splits, gold-standard cohorts, external datasets, labels, clinical-score definitions, "
            "or training/fine-tuning. If a detail is absent, represent it as unavailable rather than "
            "filling it in. The plan must declare a small executable file layout and every condition's "
            "actual API call, input, output score semantics, and stopping rule. "
            "Proposed methods MUST include direct prompt, retrieval-constrained prompt, no-critic, "
            "the study-specific proposed protocol. Every condition must save patient-level scores and "
            "write-ahead request/response hashes before the outcome join. A validation dataset with a "
            "different task, population, setting, endpoint, or horizon must be labelled accordingly.\n"
        ), statistical_test_guidance=(
            "Use only metrics justified by the target and score semantics. For prediction tasks use stratified "
            "bootstrap confidence intervals and paired patient-level comparisons; report calibration or decision "
            "utility only when probability/threshold assumptions are satisfied. Always report coverage, safety failures, cost, and latency."
        ))

    def get_code_generation_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        return PromptBlocks(compute_budget=self.domain.compute_budget_guidance,
            dataset_guidance=self.domain.dataset_guidance,
            hp_reporting=self.domain.hp_reporting_guidance,
            code_generation_hints=self.domain.code_generation_hints,
            output_format_guidance=(
                "Read stage-09/data_contract.json before coding; it is binding. Implement only its source "
                "paths, sample scope, feature allow-list, condition list, and API provider. Use a compact, "
                "runnable implementation (normally main.py plus a small helper module), not a speculative "
                "multi-file platform. Do not import FAISS, torch, transformers, XGBoost, or create synthetic "
                "patients unless the frozen contract explicitly authorizes them. Do not fine-tune or train a "
                "local model when the contract specifies an API evaluation. results.json MUST reference "
                "data_contract.json, patient_results.jsonl, ledger files, metrics.json, calibration.csv only "
                "when valid probabilities exist, and decision_curve.csv only when its stated prerequisites hold. "
                "Refuse completion if any required real-record artifact is absent."
            ))

    def get_result_analysis_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        return PromptBlocks(result_analysis_hints=self.domain.result_analysis_hints,
            statistical_test_guidance="Reject numbers without patient-level ledger provenance; evaluate calibration and decision utility separately from discrimination.")
