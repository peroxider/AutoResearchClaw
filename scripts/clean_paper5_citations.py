"""Remove unresolved citation keys from the Paper-5 revised manuscript."""
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "artifacts/paper5_trace_guard/full_run/stage-19/paper_revised.md"
text = path.read_text(encoding="utf-8")
replacements = {
    ", mccullagh2025actionable": "",
    "liu2022structgpt, ": "",
    ", alipour2025evaluation": "",
    "lee2022ehrsql, alipour2025evaluation": "singhal2023large, gallifant2025tripodllm",
    "liu2023lost, ": "",
    "albright2024medfak, ": "",
    ", albright2024medfak": "",
    ", chen2025llm_rubric": "",
    "wei2025systematic, schippling2025judges": "li2024llmsasjudges, gu2024survey",
    ", schippling2025judges": "",
    "lee2022ehrsql": "singhal2023large",
}
for old, new in replacements.items():
    text = text.replace(old, new)
path.write_text(text, encoding="utf-8")
print(path)
