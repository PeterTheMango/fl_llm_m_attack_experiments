# Causal-attack validation — run log and status

Protocol: [`protocol.md`](protocol.md) / [`protocol.json`](protocol.json). Code: branch `analysis/causal-attack-validation` at `5b8b593`, core fingerprint `521a82164f09`. Server outputs are not in git, except the Stage B [`analysis.json`](analysis.json), which is committed as a byte-identical copy.

## Where things are (2026-09-30, study finished)

| Step | State |
|---|---|
| Stage A (4 targets × 2 arms) | **Complete**: 8/8 jobs, verified and weights retired |
| Selection | **Frozen**: `probe_epochs 12`, score **cosine** |
| Stage B (10 targets, seeds 6004–6013) | **Complete**: 10/10 jobs, verified and weights retired (last job finished 2026-09-30) |
| Analysis | **Done** on 2026-09-30. Decision: **effective**. Mean per-target AUC 0.992 [0.981, 0.999]. See [`results.md`](results.md) |
| Calibration | Not flagged as a failure (FPR lower bound 0.07 ≤ 0.10). But the 5% target gave a private FPR of **0.165** [0.07, 0.275] |

The result is attack effectiveness only; no guard was in the loop. Stage A and Stage B cohorts must never tune the detector or its threshold. Next, per the protocol: pre-register matched-reference benign controls.

Server paths (host LCALC08, conda env `LLMPrivacy`, worktree `/home/calc08/projects/LLMPrivacy/fl_causal_validation` detached at `5b8b593`; it can be removed once this commit, which holds `analysis.json`, is pushed):

- `OUT=/home/calc08/projects/LLMPrivacy/fl_llm_m_attack_experiments/outputs/causal-validation-20260929`
- `PILOT=/home/calc08/projects/LLMPrivacy/fl_llm_m_attack_experiments/outputs/guard-v4-pilot`
- `$OUT/stage-a` — first Stage A launch, refused at resolve (see deviations); never run
- `$OUT/stage-a-2` — completed Stage A; `$OUT/selection.json` — frozen selection (sha256 `1c4aeb0c…`)
- `$OUT/stage-b` — completed Stage B (`launch.json` sha256 `b2b23038…`)
- `$OUT/analysis.json` — the analysis (sha256 `f512284d…`, identical to the committed copy)

## Analysis record

- The researcher ran `causal_attack_validation analyze "$OUT/stage-b" "$OUT/selection.json" "$OUT/analysis.json"` on the server.
- They pasted `cat "$OUT/analysis.json"` and `sha256sum` of `analysis.json`, `selection.json` and `stage-b/launch.json` into the writing session. The results are transcribed from that paste only.
- Two lines of the paste had shifted indentation. Re-serialising it with the tool's `indent=2` format reproduced the server file's sha256 exactly.
- An earlier local rebuild from the per-target rows alone, using the same bootstrap, agreed with every pooled value.

## Stage A results (tuning only; consumed)

Mean per-target AUC over 4 targets, 20 pairs each:

| probe_epochs | Projection | Cosine |
|---:|---:|---:|
| 12 | 0.868 [0.58, 0.99, 0.90, 1.00] | **1.000** [1.00, 1.00, 1.00, 1.00] |
| 80 | 0.331 [0.37, 0.20, 0.45, 0.30] | 0.958 [0.87, 1.00, 0.96, 1.00] |

These numbers only chose the arm and the score. They are not effectiveness evidence; that comes from Stage B. The collector's `adv` printouts (for example 0.575 and 0.325) are balanced accuracy of the run's own projection score at its calibrated threshold, not the selected score.

## Deviations and incidents (recorded, not hidden)

1. **Seed collision → new Stage A directory.** At resolve, seed 6000 selected guard-v3 seed 2007's target, and seed 6003 selected a v4 reserved-final target (seed 5000). The collector refused the launch before any GPU work. Target choice is the first record of a 4,096-record pool shuffled by `random.Random("squad_research:<seed>")`. A test over seeds 6000–9999 found 11 hits on earlier targets against about 15 expected, and a uniform number of distinct indices: bad luck, not a bug. Per the protocol's collision rule, Stage A moved to the next unallocated block, seeds 6014–6017 (`stage-a-2`). `stage-a` is kept, unrun. Stage B seeds 6004–6013 were pre-checked and resolved cleanly.
2. **Job 1 stopped by the researcher.** Stage A job 1 (seed 6014, 80 epochs) was stopped with Ctrl-C to restart the remaining 7 jobs as one run. Its partial folder and Ray logs moved to `stage-a-2/interrupted/job-001-attempt-1` and were recorded in `stage-a-2/interruptions.json`. No partial scores were read. `active_job` was cleared and the job was rerun from scratch with the same frozen config. There is no guard ledger in this study. On 2026-09-30 the folder was listed, and it holds no weights. It has only `ray-logs.tar.gz` (58 B), the run's `manifest.json` (684 B) and one `target-progress/.../progress.json` (76 B). The job was stopped before any weights were written, so nothing needs deleting. All three files stay as the inventory.
3. **Misleading collector message (fixed after the study in `26504e2`).** `collect_guard_traces` raised "overlaps the diagnostic smoke target" for any excluded target. In deviation 1 the two collisions were a guard-v3 target and a v4 reserved-final target, not the smoke target. Fixing the message changes the collector hash that launches pin, so it waited until this study ended. Since `26504e2`, the smoke message appears only when the smoke target itself is selected. Any other excluded target is reported with the job seed and the first 12 hex digits of its sha256, and no record text. The refusals themselves are unchanged and still happen before any GPU or private work. This study's launches pin the old collector hash and core fingerprint `521a82164f09`, so newer code refuses to resolve or run them. They are finished and will not be re-run. `select` and `analyze` read neither hash, and `causal_attack_validation.py` is unchanged, so the analysis path is unaffected.
