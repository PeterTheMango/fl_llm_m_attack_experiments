# CLAUDE.md — written_paper/

Guidance for work on the paper in this folder. These rules come from the researcher and override older drafting habits.

## The paper is the final submission, not a status report

`grounded_fl_study.tex` is the **final proposed paper that will be submitted for academic publication**.

- Write it as a finished academic paper ("we", past tense for what was done, present tense for what the paper shows).
- **Never describe anything as "in progress", "pending", "not started" or "interim".** Do not use `\pending{}` marks, status colours, phase timelines, dated status lines or "the researcher chose" narration.
- Keep internal project bookkeeping out of the text: core fingerprints, launch hashes, GPU-hour budgets, branch names, Firestore exports, test counts, worktree or server steps.
- **Results that do not exist yet.** Give no numbers. Say in one sentence that the evaluation is being completed, and then state the expected result and the reason for it ("We expect …, because …"). Label expectations clearly so that they can never be read as measured results. When the frozen analysis output exists, replace the expectation with the measured value from that output only.

## Scope and emphasis

- **The main focus is Phase 4, Stages 0–2** of the grounded federated RAG study (the design in `outputs/retrieval_grounded_fl_proposal_20260930/proposal.md`, the frozen `stage01_protocol.json` and the living record `status.md` on branch `feat/grounded-fl-stage0`). Describe the full implementation of that design.
- Earlier phases (attack screening, Reference/AMIA pilots, Mirabel calibration, causal-attack validation, detector controls) get **brief** treatment only, as the route to the attack battery. The guard/detector line is not part of the paper.
- **Dropped attacks.** Mention briefly that the other screened MIAs were considered and dropped because they were incompatible with this environment, so no useful membership information could be extracted through them. Keep every stated reason consistent with the code (`master_script/core/attacks/`) and the recorded outputs (`outputs/experiment_analysis_20260912/report.md`).
- **Further studies** build on Phase 4 as if Stages 0–2 have been completed. Do not list "finish Stage 0/1/2" as future work.

## References must be documented

The references were outdated. Every entry in `references.bib` must be fully documented, in the style of the author-supplied entries at the top of the file:

- full author list (never "and others"), full venue, volume and pages where they exist, DOI where one exists, `eprint` + `archivePrefix` (+ `primaryClass` for arXiv-only entries);
- a `file = {reference_papers/<Surname>_<Year>_<Short>.pdf}` field;
- metadata checked against the primary record (arXiv, ACL Anthology, OpenReview, publisher, DBLP), with the check recorded in `reference_papers/references_log.md`;
- an arXiv-only entry is updated when the paper appears at a peer-reviewed venue.

Do not cite a paper that has not been verified. Do not change the author-supplied entries without telling the researcher.

## Sources of numbers

Every number in the paper must trace to one of these files (comment block at the top of the .tex lists them):

- `outputs/retrieval_grounded_fl_proposal_20260930/status.md` (Stage 0 build, gate, datastore check, timing, positive control)
- `outputs/causal_attack_validation_20260929/results.md` (causal attack validation)
- `outputs/experiment_analysis_20260912/report.md` (attack screening)
- `outputs/mirabel_calibration_20260929/README.md`, `outputs/reference_light_analysis_20260917/`, `outputs/amia_light_analysis_20260918/` (earlier pilots)

## Build

```bash
cd written_paper && tectonic grounded_fl_study.tex
```

The Mac has no TeX Live; `tectonic` (Homebrew) builds the PDF, including BibTeX.
