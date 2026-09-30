# Matched-reference benign controls — run log and status

Protocol: [`protocol.md`](protocol.md) / [`protocol.json`](protocol.json), approved 2026-09-30. Code: branch `analysis/matched-reference-controls`, core fingerprint **`5c53eaa5b4a3`**. The collector digest is unchanged at `5a33b51ec411`.

## Where things are (2026-09-30, study finished)

| Step | State |
|---|---|
| Protocol | Approved (decisions 1–8 as drafted) |
| Detector D | Pinned: sha256 `2588f09e…1611` (server `outputs/guard-v4-pilot-detector.json`) |
| Implementation | **Done**, with 23 new tests. Full suite: 703 passed, 2 skipped (local, conda env `peter_experiments_fl`) |
| Server CPU checks | **Passed** on LCALC08: worktree `fl_matched_controls` detached at `9992a9e`, 705 passed, fingerprint `5c53eaa5b4a3` confirmed |
| Prepare / resolve | **Done**, no collision: seeds 7000–7019 resolved and job 1 ran. The prepare and resolve output was not pasted |
| Jobs (20) | **20/20 complete**, all valid, weights retired. Job 1 ran alone (14 m 18 s); jobs 2–20 ran as one invocation (267 m 39 s) |
| Analysis | **Done** on 2026-09-30, once. Decision: **fails: matched benign flagged** (E1 20/20, E2 200/200, E3 3/20). See [`results.md`](results.md). `analysis.json` sha256 `929d7669…`, `launch.json` `eb9e494f…` | Stage A, Stage B and the reserved-final cohort have not been touched.

## Changes after approval (recorded, not hidden)

Both are implementation details, made before any launch and before any result. The protocol hash that launches pin is taken at `prepare`, after both.

1. **Job field names.** The draft's single mapping `matched_controls: {public_descent_batches: 10, honest_round: true}` became two scalar `AmiaConfig` fields: `matched_descent_requests: 10` and `matched_honest_round: true`. The meaning is the same. `protocol.json` was updated to match.
2. **`baseline_D0_threshold` added to `protocol.json`.** It is the numeric value of the approved t0 (0.0023743033104398573), so the analysis reads a number rather than parsing prose.

Two further implementation notes:

- **Tied weights.** ‖Δ‖₂ is computed over state tensors, the guard's order. Qwen2.5-0.5B ties its input and output embeddings, so that tensor counts once per state key. This is the same for every arm, so the M2 norm match is unaffected. An optimizer step of length `probe_lr` is measured over the unique parameters.
- **`check` subcommand.** It lets the researcher inspect each job's matching checks and timing between jobs without seeing any detector decision. Decisions only appear in `analyze`, after all 20 jobs.

## Job log

`check` output, as pasted by the researcher. Decision fields are withheld by design.

| Job | Seed | Target (12 hex) | Wall time | `computation_seconds` | Valid | M2 ‖Δ‖ ratio | M1 ‖Δ‖ ratio range | Candidate loss rose (C) | M1 public loss fell | Training events / distinct |
|---|---:|---|---:|---:|---|---:|---|---|---:|---|
| 000 | 7000 | `c27dd1fe0324` | 14 m 18 s | 835 | yes | 1.00000004 | 0.83–1.19 | yes | 10/10 | 12 / 3 |
| 001–019 | 7001–7019 | see `analysis.json` | 267 m 39 s for all 19 | 821–842 | yes (19/19) | 1.00000003–1.00000075 | 0.76–1.79 | yes (19/19) | 190/190 | 12 / 3 each |

- **Result sha256 for job 000:** `b2b12788…2747`.
- **Timing.** The first job took 14.3 min, under the 25-min pause rule. The estimate is now about 20 × 14.3 min ≈ 4.8 GPU-hours, inside the 3–6 h budget.
- **Matching checks.** M1 turned out size-matched as well as step-matched in this job: its net ‖Δ‖₂ is 0.83–1.19× the causal request's.
- **Collector printout.** `adv=0.750` is the run's own attack balanced accuracy from 4 trials. It is not an endpoint and is not analysed.

- **Run cadence (deviation).** Jobs 2–20 ran as one invocation, with `check` at the end instead of after each job. The researcher chose this after job 1 passed. The collector stops on any failure, and all 20 jobs were valid.

## Run sequence (on LCALC08)

Use a separate worktree, so the experiment checkout keeps its code. Outputs go to an absolute path outside the worktree. Clear any leftover `HF_HUB_OFFLINE` before GPU jobs.

Server paths:

- `E=/home/calc08/projects/LLMPrivacy/fl_llm_m_attack_experiments/outputs`
- `OUT=$E/matched-controls-20260930`
- worktree `/home/calc08/projects/LLMPrivacy/fl_matched_controls`, detached at the pushed branch head

The steps are:

1. **CPU:** fetch and add the worktree, then run the full test suite in it.
2. **CPU:** `prepare`, passing every exclusion source. These are the v4 pilot `cohort.json`, `splits.complete.json` and `launch.json`, the causal-validation `stage-a-2` and `stage-b` `splits.complete.json`, and the committed v3 splits copy. Then `collect_guard_traces resolve`. On a collision, prepare a new directory with `--first-seed 7020` (and so on, inside the band) and record it here. Never edit a launch.
3. **GPU, one job:** `collect_guard_traces run --gpu N --max-jobs 1` in tmux. Record the wall time. If it exceeds 25 min, pause and re-estimate.
4. **CPU:** `check` after every job. Every job must be `valid`, and M2's norm ratio must be 1 within 1e-4.
5. **GPU:** repeat step 3 until 20/20 jobs are complete.
6. **CPU:** `analyze "$OUT" "$OUT/analysis.json"` once. Paste the output and `sha256sum` it for the write-up.
