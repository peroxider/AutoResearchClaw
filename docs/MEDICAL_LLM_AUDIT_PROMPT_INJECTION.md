# Configuring a clinical LLM/agent study

`medical_llm_audit` is intentionally algorithm- and dataset-agnostic. It
provides the 23-stage evidence contract; the current study supplies its
contents through `prompts.extra_prompts`.

Start with `config.medical_llm_audit.example.yaml`. Keep each study's data
contract and method contract in version-controlled Markdown/YAML files and
reference those files from `extra_prompts` when they are long. This ensures
Stages 9, 10, 12, 14, 17, and 20 receive the same frozen specification.

Do not put an API key, a direct identifier, raw patient record, or an outcome
value in prompts. Pass only paths to approved local data and the declared
field-level contract. Stage 12 code must construct de-identified state cards
and lock them before outcome linkage.

The profile does not mandate an algorithm. A run may specify retrieval,
planning, tools, memory, multi-agent review, calibration, or another LLM
protocol, provided each proposed component has runnable code and a named
ablation. Likewise, a dataset with another endpoint or setting may be used,
but its validation role must be described accurately in the prompt contract.
