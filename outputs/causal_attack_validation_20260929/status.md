# Causal-attack validation — run log and status

Protocol: [`protocol.md`](protocol.md) / [`protocol.json`](protocol.json). Code: branch `analysis/causal-attack-validation` at `5b8b593`, core fingerprint `521a82164f09`. Server outputs are not in git.

## Where things are (2026-09-30, after Stage B)

| Step | State |
|---|---|
| Stage A (4 targets × 2 arms) | **Complete**: 8/8 jobs, verified and weights retired |
| Selection | **Frozen**: `probe_epochs 12`, score **cosine** |
| Stage B (10 targets, seeds 6004–6013) | **Complete**: 10/10 jobs, verified and weights retired (last job finished 2026-09-30) |
| Analysis | Not run |

Server paths (host LCALC08, conda env `LLMPrivacy`, worktree `/home/calc08/projects/LLMPrivacy/fl_causal_validation` detached at `5b8b593`; keep it until Stage B is analysed, because the launches pin the fingerprint):

- `OUT=/home/calc08/projects/LLMPrivacy/fl_llm_m_attack_experiments/outputs/causal-validation-20260929`
- `PILOT=/home/calc08/projects/LLMPrivacy/fl_llm_m_attack_experiments/outputs/guard-v4-pilot`
- `$OUT/stage-a` — first Stage A launch, refused at resolve (see deviations); never run
- `$OUT/stage-a-2` — completed Stage A; `$OUT/selection.json` — frozen selection
- `$OUT/stage-b` — resolved Stage B launch

## Next commands (on the server, in the worktree)

```bash
python -m master_script.tools.causal_attack_validation analyze "$OUT/stage-b" "$OUT/selection.json" "$OUT/analysis.json"
```

This runs on CPU and takes seconds. It refuses to overwrite an existing `analysis.json`. Paste its printed summary, and the per-target rows from `analysis.json`, into the session that writes the results up. Keep the worktree until the analysis has been run.

## Stage A results (tuning only; consumed)

Mean per-target AUC over 4 targets, 20 pairs each:

| probe_epochs | Projection | Cosine |
|---:|---:|---:|
| 12 | 0.868 [0.58, 0.99, 0.90, 1.00] | **1.000** [1.00, 1.00, 1.00, 1.00] |
| 80 | 0.331 [0.37, 0.20, 0.45, 0.30] | 0.958 [0.87, 1.00, 0.96, 1.00] |

These numbers only chose the arm and the score. They are not effectiveness evidence; that comes from Stage B. The collector's `adv` printouts (for example 0.575 and 0.325) are balanced accuracy of the run's own projection score at its calibrated threshold, not the selected score.

## Deviations and incidents (recorded, not hidden)

1. **Seed collision → new Stage A directory.** At resolve, seed 6000 selected guard-v3 seed 2007's target, and seed 6003 selected a v4 reserved-final target (seed 5000). The collector refused the launch before any GPU work. Target choice is the first record of a 4,096-record pool shuffled by `random.Random("squad_research:<seed>")`. A test over seeds 6000–9999 found 11 hits on earlier targets against about 15 expected, and a uniform number of distinct indices: bad luck, not a bug. Per the protocol's collision rule, Stage A moved to the next unallocated block, seeds 6014–6017 (`stage-a-2`). `stage-a` is kept, unrun. Stage B seeds 6004–6013 were pre-checked and resolved cleanly.
2. **Job 1 stopped by the researcher.** Stage A job 1 (seed 6014, 80 epochs) was stopped with Ctrl-C to restart the remaining 7 jobs as one run. Its partial folder and Ray logs moved to `stage-a-2/interrupted/job-001-attempt-1` and were recorded in `stage-a-2/interruptions.json`. No partial scores were read. `active_job` was cleared and the job was rerun from scratch with the same frozen config. There is no guard ledger in this study. The partial weights there can be deleted after the analysis; keep the inventory.
3. **Misleading collector message.** `collect_guard_traces` raises "overlaps the diagnostic smoke target" for any excluded target. Fixing it would change the collector hash that launches pin, so it waits until this study ends.
