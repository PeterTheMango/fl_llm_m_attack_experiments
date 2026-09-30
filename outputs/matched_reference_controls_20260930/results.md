# Matched-reference benign controls — result

**Decision: fails: matched benign flagged**, under the pre-registered rule. The frozen guard-v4 pilot detector D flagged every causal request (20/20). It also flagged **every** step-matched benign request (200/200 across 20 targets) and 3 of 20 norm-matched honest updates. D does not tell the causal request apart from benign requests built to match it.

This is a shadow-mode result about the detector's discrimination only. It makes no privacy claim, and none of it is protection. Attack effectiveness remains the separate result in [`../causal_attack_validation_20260929/results.md`](../causal_attack_validation_20260929/results.md).

## Provenance

- **Protocol and code.** The protocol was approved on 2026-09-30 before any run ([`protocol.md`](protocol.md), [`protocol.json`](protocol.json)). The code is branch `analysis/matched-reference-controls` at `9992a9e` (code from `a828b47`), core fingerprint `5c53eaa5b4a3`. The server tests passed (705) in the worktree `fl_matched_controls`.
- **Runs.** 20 jobs, seeds 7000–7019, with no collision at resolve. Job 1 ran on its own (14 m 18 s). The other 19 ran as one invocation (267 m 39 s, about 14.1 min per job). That makes about **4.7 GPU-hours** in total, inside the 3–6 h budget. All 20 jobs passed every required matching check. The per-job `check` output is in [`status.md`](status.md).
- **Analysis.** The researcher ran `matched_reference_controls analyze` once, on LCALC08 on 2026-09-30, and pasted the full `analysis.json` and its `sha256sum`. Every number here is transcribed from that paste.

| Server file | sha256 (pasted) |
|---|---|
| `$OUT/analysis.json` | `929d766901aaefa49a3aa7da69c50f0e6a4ae9d15f0deba7bebe195b32f95bb1` |
| `$OUT/launch.json` | `eb9e494f5db9281a5d3962eeacd921140d541f9bc55de40bf26336f0adec9a56` |

- **Hash checks.** Inside `analysis.json`:
  - `protocol_sha256` (`9ea7f5e1…`) matches the committed `protocol.json`;
  - `study_tool_sha256` (`84f6df45…`) matches `master_script/tools/matched_reference_controls.py` at `a828b47`;
  - `detector_sha256` is the pinned `2588f09e…1611`.
- **Independent recomputation.** E1–E3 and their intervals, the decision, S1, S2 and the two S4 rows quoted below were recomputed locally from the pasted per-target values with the tool's own functions and RNG seed. They agree exactly.

## Endpoints (20 targets; targets are the unit)

| Endpoint | Value | Bootstrap 95% | Decision interval | Rule |
|---|---:|---|---|---|
| **E1: causal-request detection TPR** | **1.00** (20/20) | [1.00, 1.00] | Clopper–Pearson [0.832, 1.00] | "Causal missed" needs an upper bound < 0.50: not met |
| **E2: rejection of step-matched benign (M1)** | **1.00** (200/200; all 20 targets) | [1.00, 1.00] | [1.00, 1.00] | Lower bound > 0.01: **met** |
| **E3: rejection of norm-matched benign (M2)** | **0.15** (3/20) | [0.00, 0.30] | Clopper–Pearson [0.032, 0.379] | Lower bound > 0.01: **met** |

Both benign-flagged conditions hold independently. The "separates" branch needed zero matched rejections.

## Secondary (descriptive; not part of the decision)

- **S1: ordinary training requests** (initial-LM reference).
  - 72/240 events rejected, a rate of **0.30** [0.20, 0.40].
  - Per distinct request it is 18/60, also **0.30** [0.20, 0.40]. Every distinct request was decided identically for all 4 clients.
  - All rejections are round-2 or round-3 broadcasts. The round-1 request equals the reference and was never rejected.
  - On fresh targets, this is far above both the pilot's single-target 8.33% and the ≤1% gate.
- **S2: M2raw** (the unscaled honest round): 0.10 (2/20) [0.00, 0.25].
- **S3: magnitude-only baseline D0** (relative change > t0): it flagged **nothing**, whether C, M1 or M2 (0/20, 0/200 and 0/20). As in the pilot, the causal request's change is far below a threshold set on training drift.
- **Where D's decisions sit.** Taken from the per-target scores, D's threshold lies between 0.012657 (the highest unflagged score) and 0.012775 (the lowest flagged score).
  - Every C and M1 score lies between 0.020 and 0.039.
  - Every M2 and M2raw score lies between 0.0108 and 0.0134.
  - So D separates "optimizer steps on text from the post-FL model" (C and M1 alike) from "FedAvg-style updates". The M2 decisions sit right at the threshold, and the 3 M2 rejections are knife-edge.
- **S4: within-target separability of C against its 10 M1 requests (exploratory).**
  - **D's score:** AUC 0.725 [0.585, 0.85]. It ranks C above its twins more often than chance, but both lie far above the threshold.
  - **Strongest single feature:** `mean_update_concentration`, AUC 0.915 [0.855, 0.965]. C's per-tensor updates are more concentrated than M1's.
  - **Size features** (`relative_delta`, `median_layer_delta` and others) run the other way (AUC 0.11–0.19). They track the fact that M1's net ‖Δ‖ is mostly larger than C's (see S5), not the request's intent.
  - **Plausible confound.** C ascends the loss of **one** record, while M1 descends the loss of a batch of **four**. A single-text gradient touches fewer embedding rows, so it would be more concentrated regardless of intent. This study cannot separate the two explanations.
  - These values are exploratory. They make this cohort detector-development data only.
- **S5: matching.**
  - The reference was identical across arms in all 20 targets.
  - M2's norm ratio was 1 within 8×10⁻⁷ everywhere.
  - M1's net ‖Δ‖ was 0.76–1.79× C's.
  - The candidate loss rose under C in 20/20 targets. The public-batch loss fell under M1 in 200/200 requests.

## What this means

- **D does not work as a causal-request detector.**
  - Its pilot "4/4 causal detected" reflects the reference confound identified before the study. A small step from the post-FL model looks alike to D whether it is an attack or benign public-data descent.
  - Per the decision rule: record a negative result, stop escalating D as a causal-request control, and **run no enforced comparison with D**. This triggers the fallback for parked item 2 (see below).
- **The ≤1% legitimate-rejection gate is refuted.** It fails on matched requests (E2 1.00, E3 0.15). On ordinary training requests, S1 gives 0.30.
- **Detection is not protection.** Nothing here measures leakage with the guard enforcing.
- **Whether any parameter-feature detector could work is not settled.** One feature separated C from M1 in exploratory analysis. It may reflect the batch-size difference rather than intent. Testing that needs a new pre-registration, with single-record benign twins, on a fresh cohort. It is the researcher's decision whether it is worth the cost. Per the v4 handoff, if matched controls show no stable separability, stop developing a classifier as a privacy measure.

## Limits

- One detector, fitted on 2 pilot targets. One model (Qwen2.5-0.5B-Instruct), one data set, and one harness-supplied reference (post-FL W_R).
- Two constructed benign families. M1 differs from C in both objective sign and batch size (4 records against 1).
- 20 targets, with targets as the unit.
- No adaptive attacker, and no enforcement.

## Deviations

- **Run cadence.** After job 1, the remaining 19 jobs ran as one collector invocation, and `check` ran at the end rather than after every job. The researcher chose this option before running them. The collector would have stopped on any failed job, and all 20 turned out valid. No design, parameter or threshold changed.
- **Prepare and resolve output** was not pasted. The launch hash is recorded above, and the collector-verified results imply a clean resolve.
