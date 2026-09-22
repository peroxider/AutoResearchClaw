# Medical Informatics AI-Methods Paper Quality Standard

This standard is mandatory when `project.profile: medical_llm_audit`. It records
the quality requirements learned from the Paper-5 revision cycle and is enforced
at Stage 20 by `researchclaw.pipeline.medical_paper_quality`.

## Scientific positioning

- Treat the manuscript as a medical-informatics AI/algorithm methodology paper.
- The central contribution must be a genuinely LLM-based application algorithm,
  not conventional machine learning relabelled as an agent.
- Methods must explain the algorithm, objective, inputs/outputs, component
  composition, release/abstention rule, and computational complexity.
- Include mathematical definitions and at least one proposition/theorem with an
  explicit argument or proof. Governance, threats, audit logs, retries, and
  information-system design may support the method but must not replace theory.
- Include a detailed algorithm framework figure in Methods.

## Two-level architecture-figure requirement

Introduction and Methods must contain two distinct architecture figures. Reusing
the same figure in both sections does not satisfy this requirement.

### Introduction: study-level conceptual overview

- Place an abstract overview figure near the end of Introduction.
- Show only the macro flow: **input data → LLM application method/model → output
  and evaluation**.
- Make the paper's innovations visually prominent with numbered nodes, accent
  borders, callouts, or a dedicated innovation wrapper.
- Prefer shapes, arrows, grouping, and short labels over explanatory paragraphs.
- Do not use emoji or decorative clip art. Avoid equations, retry logic, field
  lists, and patient-level metric definitions in this overview.

### Methods: detailed algorithm framework

- Place a separate detailed framework figure in Materials and Methods.
- Group text blocks into functional modules using large dashed frames or clearly
  separated regions, following high-quality journal figure organization.
- Use one color family within each large module and visibly different color
  families between modules.
- Show component order, branching, information-flow boundaries, and the relation
  between generator, critic, selective release, commitment, and evaluation.
- Text must fill its block at readable journal size without overflow; inspect both
  the rendered image and final PDF.
- The two architecture figures must have different files, captions, visual
  density, and abstraction levels.

## Dataset characterization

- Add a dedicated dataset statistical-profile subsection before model results.
- Report cohort size, event/class composition, missingness, and outcome-stratified
  summaries. State variable coding and unknown units rather than guessing them.
- Use distribution-appropriate summaries and tests; for small/non-normal groups,
  prefer median [IQR], Mann–Whitney U, and Fisher exact tests.
- Label unadjusted tests as exploratory and avoid interpreting non-significance as
  absence of association. Do not use these tests for post-hoc feature selection.
- Provide both a descriptive table and a multi-panel dataset figure.

## Results

- A single aggregate table is insufficient.
- Analyze at least four distinct dimensions: discrimination/ranking, calibration,
  score and release coverage/failures, and efficiency/critic behavior. Add
  patient-level distribution or common-case sensitivity analysis when supported.
- Report denominators and event counts for every condition when failures differ.
- Distinguish numeric-score coverage from operational release coverage.
- Provide at least two result figures in addition to the dataset and algorithm
  figures. Confidence intervals, calibration-bin sizes, and individual-patient
  points should be visible where applicable.
- Tie every number to executed patient-level evidence; never invent or impute a
  failed API result merely to complete a comparison.

## Figure style contract

- One manuscript, one shared plotting-style module or manifest.
- Use a colorblind-safe palette consistently: the same condition/outcome always
  receives the same color in every figure. Do not rely on color alone.
- Use consistent typography, panel labels (`A`, `B`, ...), axis/spine weights,
  legends, and caption terminology.
- Export every scientific figure as 300-dpi PNG plus vector PDF/SVG.
- Captions must be self-contained and must not contain a manual `Figure N` prefix;
  numbering is owned by the document converter.
- Verify figure numbering and all in-text references after inserting or removing a
  figure. Visually inspect the final PDF, not only the source images.

## Final deliverable gate

- Target at least 4,500 words for a full methods paper unless a venue imposes a
  shorter limit. A six-page draft is not considered complete by default.
- Require a readable PDF, sequential figures, no duplicate captions, and no
  missing image assets.
- Preserve Markdown, LaTeX, PDF, source figure files, statistical JSON/CSV, and
  the quality report in the deliverable bundle.
