# Matched-reference benign controls — run log and status

Protocol: [`protocol.md`](protocol.md) / [`protocol.json`](protocol.json), approved 2026-09-30. Code: branch `analysis/matched-reference-controls`, core fingerprint **`5c53eaa5b4a3`**. The collector digest is unchanged at `5a33b51ec411`.

## Where things are (2026-09-30)

| Step | State |
|---|---|
| Protocol | Approved (decisions 1–8 as drafted) |
| Detector D | Pinned: sha256 `2588f09e…1611` (server `outputs/guard-v4-pilot-detector.json`) |
| Implementation | **Done**, with 23 new tests. Full suite: 703 passed, 2 skipped (local, conda env `peter_experiments_fl`) |
| Server CPU checks | Not run |
| Prepare / resolve | Not run |
| Jobs (20) | None run |
| Analysis | None |

No result exists. Stage A, Stage B and the reserved-final cohort have not been touched.

## Changes after approval (recorded, not hidden)

Both are implementation details, made before any launch and before any result. The protocol hash that launches pin is taken at `prepare`, after both.

1. **Job field names.** The draft's single mapping `matched_controls: {public_descent_batches: 10, honest_round: true}` became two scalar `AmiaConfig` fields: `matched_descent_requests: 10` and `matched_honest_round: true`. The meaning is the same. `protocol.json` was updated to match.
2. **`baseline_D0_threshold` added to `protocol.json`.** It is the numeric value of the approved t0 (0.0023743033104398573), so the analysis reads a number rather than parsing prose.

Two further implementation notes:

- **Tied weights.** ‖Δ‖₂ is computed over state tensors, the guard's order. Qwen2.5-0.5B ties its input and output embeddings, so that tensor counts once per state key. This is the same for every arm, so the M2 norm match is unaffected. An optimizer step of length `probe_lr` is measured over the unique parameters.
- **`check` subcommand.** It lets the researcher inspect each job's matching checks and timing between jobs without seeing any detector decision. Decisions only appear in `analyze`, after all 20 jobs.

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
