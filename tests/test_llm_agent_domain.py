"""Regression tests for the LLM/Agent research profile."""

from researchclaw.domains.detector import _keyword_detect, get_profile
from researchclaw.domains.prompt_adapter import get_adapter


def test_llm_agent_profile_is_available_and_uses_llm_baselines():
    profile = get_profile("ml_llm_agent")
    assert profile is not None
    assert profile.gpu_required is True
    assert "ReAct" in profile.standard_baselines
    assert "transformers" in profile.core_libraries


def test_llm_agent_keywords_precede_generic_nlp_detection():
    assert _keyword_detect("tool-using LLM agent with retrieval augmented generation") == "ml_llm_agent"
    assert _keyword_detect("multi-agent planning for a language model") == "ml_llm_agent"


def test_llm_agent_adapter_makes_agent_methods_mandatory():
    profile = get_profile("ml_llm_agent")
    assert profile is not None
    adapter = get_adapter(profile)
    design = adapter.get_experiment_design_blocks({})
    code = adapter.get_code_generation_blocks({})
    assert "MUST be LLM- or Agent-based" in design.experiment_design_context
    assert "MUST be an LLM- or Agent-based" in code.code_generation_hints
