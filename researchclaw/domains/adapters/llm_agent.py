"""Prompt adapter for large-language-model and agentic-system research."""

from __future__ import annotations

from typing import Any

from researchclaw.domains.prompt_adapter import PromptAdapter, PromptBlocks


class LLMAgentPromptAdapter(PromptAdapter):
    """Makes LLM/agent methods primary methods throughout stages 9 and 10."""

    def get_experiment_design_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        baselines = ", ".join(self.domain.standard_baselines)
        return PromptBlocks(
            experiment_design_context=(
                f"This is a **{self.domain.display_name}** experiment.\n\n"
                "METHOD PRIORITY (MANDATORY): Proposed methods MUST be LLM- or "
                "Agent-based. Classical ML may be included only as a clearly "
                "labelled lightweight control, never as the proposed contribution.\n"
                "- Specify the model, prompt/template, planner, memory, retrieval, "
                "and typed tool interface used by each proposed agent.\n"
                "- Include an explicit bounded agent loop, termination/failure policy, "
                "and trajectory logging in each implementation_spec.\n"
                "- Compare against current, literature-grounded LLM/agent baselines: "
                f"{baselines}.\n"
                "- Evaluate task quality/success, groundedness where relevant, tool "
                "success, token cost, latency, and robustness to retrieval/tool failure.\n"
                "- Ablate the actual claimed mechanisms (e.g. planning, memory, "
                "retrieval, reflection, routing), not arbitrary hyperparameters.\n"
            ),
            statistical_test_guidance=(
                "Use paired task-level comparisons and bootstrap confidence intervals. "
                "Report confidence intervals for quality, cost, latency, and tool success."
            ),
        )

    def get_code_generation_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        return PromptBlocks(
            compute_budget=self.domain.compute_budget_guidance,
            dataset_guidance=self.domain.dataset_guidance,
            hp_reporting=self.domain.hp_reporting_guidance,
            code_generation_hints=self.domain.code_generation_hints,
            output_format_guidance=(
                "Write results.json with per-condition task metrics, token/cost and "
                "latency summaries, tool-call success/failure counts, and sampled "
                "trajectories or deterministic replay identifiers."
            ),
        )

    def get_result_analysis_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        return PromptBlocks(
            result_analysis_hints=self.domain.result_analysis_hints,
            statistical_test_guidance=(
                "Use paired task-level tests and bootstrap confidence intervals; "
                "analyze quality-cost, quality-latency, and failure-rate trade-offs."
            ),
        )
