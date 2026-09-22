"""CLI for the medical-informatics AI-methods manuscript quality gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from researchclaw.pipeline.medical_paper_quality import audit_medical_ai_manuscript

parser = argparse.ArgumentParser()
parser.add_argument("paper", type=Path)
parser.add_argument("--assets", type=Path)
parser.add_argument("--report", type=Path)
args = parser.parse_args()
report = audit_medical_ai_manuscript(args.paper.read_text(encoding="utf-8"), args.assets)
payload = report.to_dict()
if args.report:
    args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
raise SystemExit(0 if report.passed else 2)
