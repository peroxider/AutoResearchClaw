"""Build the evidence-corrected Paper-5 Markdown/LaTeX/PDF deliverables."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from researchclaw.templates import get_template, markdown_to_latex


ROOT = Path(__file__).resolve().parents[1]
DELIV = ROOT / "artifacts/paper5_trace_guard/full_run/deliverables"
STAGE23 = ROOT / "artifacts/paper5_trace_guard/full_run/stage-23"
md = DELIV / "paper_final_corrected.md"
bib = DELIV / "references_corrected.bib"

# Keep the first entry for each key; Stage 23 verified all cited sources but
# emitted one duplicate key from two candidate records.
source = (STAGE23 / "references_verified.bib").read_text(encoding="utf-8")
entries = re.split(r"(?=^@)", source, flags=re.MULTILINE)
seen: set[str] = set()
kept: list[str] = []
for entry in entries:
    match = re.match(r"@\w+\s*\{\s*([^,]+),", entry.strip())
    if not match:
        continue
    key = match.group(1).strip()
    if key not in seen:
        seen.add(key)
        kept.append(entry.strip())
bib.write_text("\n\n".join(kept) + "\n", encoding="utf-8")

paper = md.read_text(encoding="utf-8")
tex = markdown_to_latex(
    paper, get_template("generic"), authors="Anonymous",
    bib_file="references_corrected",
)
# Generic template variants may load geometry twice with different options.
if "\\usepackage[margin=1in]{geometry}" in tex:
    tex = tex.replace("\\usepackage{geometry}\n", "", 1)
tex_path = DELIV / "paper_final_corrected.tex"
tex_path.write_text(tex, encoding="utf-8")

for command in (
    ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
    ["bibtex", tex_path.stem],
    ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
    ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
):
    result = subprocess.run(command, cwd=DELIV, text=True, capture_output=True)
    if result.returncode:
        raise SystemExit(result.stdout[-3000:] + result.stderr[-1000:])

pdf = DELIV / "paper_final_corrected.pdf"
assert pdf.exists() and pdf.stat().st_size > 10_000
print(md)
print(tex_path)
print(pdf)
