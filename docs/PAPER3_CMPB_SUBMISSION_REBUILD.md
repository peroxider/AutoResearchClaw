# Paper 3: CMPB submission-ready rebuild protocol

## Decision

The current `paper3_arc23` run is an exploratory, evidence-corrected audit
report, not a submission-ready clinical prediction paper. Its 86 encounters
and 16 recorded events are retained only as an internal feasibility cohort.
No claim of clinical superiority, calibration, or deployability may be made
from that cohort.

The rebuilt manuscript will target a methods-and-software contribution:

> **SCAFFOLD-R: a retrieval-constrained, outcome-blind and auditable
> foundation-model protocol for clinical risk-assessment research.**

The contribution is a reproducible software method for making foundation-model
evaluations falsifiable and auditable. It is not a patient-facing decision
support system and it must not issue treatment recommendations.

## Algorithm: SCAFFOLD-R

Each assessment follows this fixed state machine:

1. **Schema compiler** converts the pre-specified, time-indexed baseline
   variables into a minimal state card. It rejects direct identifiers,
   post-index variables, endpoint fields, free-text leakage, and unrecognised
   units.
2. **Guideline retrieval gate** retrieves only versioned, study-approved
   reference passages. Retrieval is logged and no retrieved document may
   contain the cohort outcome.
3. **Constrained clinical-reasoner agent** returns a typed JSON assessment:
   risk rank/probability, cited state-card fields, cited retrieval identifiers,
   uncertainty flags, and abstention decision.
4. **Independent audit-critic agent** checks schema conformance, unsupported
   evidence references, uncertainty consistency, and prompt-injection/tool
   errors. A failed check is an abstention, not a repaired prediction.
5. **Write-ahead commitment ledger** hashes the canonical input, retrieval
   set, model/version, prompt version, response, and critic verdict before
   the held-out outcome can be linked.
6. **Outcome linker and evaluator** performs the sole outcome join and emits
   per-patient, per-seed, and per-condition metrics with reproducibility
   manifests.

Novelty is therefore testable: the method couples outcome-blindness, bounded
retrieval, multi-agent critique, abstention, and cryptographic commitment.
Every component has an ablation rather than a narrative-only novelty claim.

## Study design and evidence gates

### Cohorts

| Role | Required data | Permitted model execution |
|---|---|---|
| Internal feasibility | Current 86-encounter cohort | MiniMax only after identifier removal and an approved, outcome-blind field allow-list |
| Temporal/internal validation | A later non-overlapping cohort from the same source | Same as internal feasibility, subject to governance approval |
| Public method-validation cohort | UCI Diabetes 130-US Hospitals (1999-2008), using 30-day readmission as its separately specified outcome | MiniMax after removal of `encounter_id` and `patient_nbr`; retain only the public, protocol-approved feature allow-list |

The present CSV lacks documented index time, follow-up window, endpoint
adjudication, and enough events for a publication-grade development study.
Before modelling, provide a versioned data dictionary that defines all four.
The UCI cohort is not a bleeding-risk external validation set and must never
be described as one. It is a public, cross-task method-validation cohort for
testing whether the SCAFFOLD-R protocol, logging, abstention, and safety gates
remain reproducible at scale on a distinct clinical readmission task.

### Primary endpoint

Use one prospectively specified, clinically interpretable endpoint. For the
local cohort, the currently recorded "post-anticoagulation bleeding" flag is
an exploratory administrative endpoint only; it is not a major-bleeding
endpoint unless timing and adjudication criteria are supplied. The external
cohort must use a pre-mapped compatible endpoint or be reported as a separate
transportability task, never pooled silently.

### Conditions (all genuinely executed)

1. Fixed clinical score(s), such as HAS-BLED, when their definitions are
   available in the cohort.
2. Direct structured prompting.
3. Retrieval-constrained prompting.
4. SCAFFOLD-R without the audit critic.
5. SCAFFOLD-R without abstention.
6. Full SCAFFOLD-R.

Each condition must persist patient-level scores, model response hashes,
retrieval IDs, critic verdicts, abstention reason, token/cost data, and wall
time. Seed variation must alter a documented stochastic mechanism; bootstrap
replicates are confidence-interval calculations, not independent model runs.

### Required analyses

- Cohort flow, missingness, and baseline characteristics by outcome.
- AUROC and AUPRC with stratified bootstrap confidence intervals.
- Calibration intercept, slope, calibration curve, and Brier score only for
  valid probability outputs.
- Decision-curve analysis over a pre-specified threshold range.
- Net benefit, abstention rate, unsafe-output rate, schema-violation rate,
  evidence-grounding rate, token cost, and latency.
- Paired patient-level comparisons between conditions; no invented
  seed-level standard deviations.
- Subgroup and missing-data sensitivity analyses only when event counts make
  them estimable; otherwise label them exploratory.
- Internal feasibility and UCI public method validation reported separately,
  with confidence intervals and explicit cross-task transportability limits.

## Software package required for CMPB submission

Create a new `researchclaw/paper3_scaffold_r/` package with:

```text
config/                 frozen protocol, model, prompt, and data contracts
schema/                 field allow-list, leakage rules, unit mappings
retrieval/              versioned guideline corpus and retrieval index
agents/                 reasoner, critic, typed tool interfaces
ledger/                 canonicalisation, hashes, write-ahead commitment
evaluation/             discrimination, calibration, DCA, paired inference
reports/                tables, figures, TRIPOD+AI and software appendices
tests/                  leakage, lock-before-link, determinism, metric tests
```

The release must include runnable code, a synthetic fixture that exercises all
conditions, a dependency lockfile, model/prompt hashes, and a reproduction
command. Restricted cohort extracts and raw model responses must remain out of
the public repository; publish only approved aggregate outputs and the code
needed by authorised users to regenerate them.

## Manuscript package

Target a CMPB-style methods paper rather than a generic conference manuscript:

1. Structured abstract (Objective, Methods, Results, Conclusion).
2. Introduction: clinical-information leakage and reproducibility problem.
3. Method: formal protocol, state machine, threat model, implementation.
4. Data and evaluation: cohorts, index time, endpoint, governance, metrics.
5. Results: all conditions, calibration, utility, audit failures, efficiency.
6. Discussion: what the method demonstrates, failure modes, limitations, and
   non-deployment boundary.
7. Software availability, data-access statement, model/API statement,
   conflicts/funding, ethics, and TRIPOD+AI checklist.

Required figures: system architecture, cohort flow, calibration/DCA, and
performance-versus-cost/abstention trade-off. Required tables: cohort
characteristics, main results, ablations/safety failures, and external
validation.

## Acceptance gates before submission

- No direct identifier or endpoint is present in a model request.
- Every reported result can be recomputed from a saved patient-level record.
- All conditions and figures have real execution artifacts.
- Endpoint and time window are documented before scoring.
- At least one independent validation cohort is complete.
- Reporting checklist, code archive, and data-access statement are complete.
- Claims are limited to the evidence; no clinical superiority or deployment
  claim without corresponding external validation and utility evidence.

## Public-data decision: UCI Diabetes 130-US Hospitals

Use the UCI Diabetes 130-US Hospitals for Years 1999-2008 dataset as the
public method-validation cohort. It contains 101,766 inpatient diabetes
encounters from 130 US hospitals, with a 30-day readmission task and a CC BY
4.0 licence. Download through the versioned UCI endpoint, record the dataset
DOI and file checksum, and remove both ID columns (`encounter_id` and
`patient_nbr`) before creating any model request. The frozen protocol must
declare the feature cut-off (only data available at the chosen prediction
time), target mapping, missing-value treatment, and any rows excluded before
scoring.

UCI data support reproducible, public method validation; they do not create a
same-endpoint external validation for the local anticoagulation/bleeding
cohort. The manuscript must use that distinction consistently in the title,
abstract, results, and limitations.
