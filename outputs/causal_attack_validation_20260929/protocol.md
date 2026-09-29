# Causal gradient-alignment attack — validation protocol

Status: **final.** The researcher approved every proposed decision on 2026-09-29: success thresholds, the 4 + 10 target budget, seed band 6000–6099, 100 calibration batches, and the telemetry change. The machine-readable version is [`protocol.json`](protocol.json); each launch copies it and pins its hash. Nothing has run yet.

## Question

Is `causal_gradient_alignment` an effective membership attack on its own, with no guard in the loop? The detector's 4/4 detection of causal requests only matters if the request actually reveals membership.

## What the existing evidence shows

Recomputed from the three guard-v4 pilot causal jobs (`attack_trials` in the exported results):

| Job | Trials | Member − nonmember score per pair | Batches the calibrated threshold labelled as member |
|---|---:|---|---|
| 001 | 4 | −62.8, −50.3 | 4/4 |
| 004 | 4 | +129.1, +203.3 | 2/4 |
| 007 | 4 | +70.1, −26.2 | 2/4 |

- **The data are uninformative.** There are 12 trials in all, one target per job and 2 pairs per target. The member batch scored higher in 3 of 6 pairs. The recorded per-target AUCs (0.25 / 1.00 / 0.50) cannot separate an effective attack from chance. "AUC 0.50" means "not measured", not "ineffective".
- **Calibration may not transfer.** The threshold, fitted on 20 public nonmember batches, labelled all four of job 001's private batches as members. Public and private batch scores may sit on different scales. The nominal 5% false-positive target is unverified.
- **Batch composition likely dominates the score.** Scores vary far more between pairs (for example 1466–1774 within one target) than within a pair. The score is the projection `dot(released, direction) / ‖direction‖`, which grows with the batch's overall gradient size.
- **The attack was weak.** The pilot used `probe_epochs: 12`, while the master config default is 80.

## Mechanism (unchanged)

- The server applies bounded gradient ascent on the public target's loss to the post-FL LM (`optimize_request`: `probe_epochs` normalized steps of size `probe_lr`).
- It sends those weights to the victim, and the victim returns its batch gradient.
- The server scores the projection of that gradient onto the target's gradient at the request.
- Pairs share one batch seed: the member batch is the nonmember batch with one record replaced by the target.

## Design

**Cohort.** Fresh development targets from the seed band 6000–6099. Exclude:

- the smoke target;
- every group in the guard-v3 and guard-v4 pilot `splits.complete.json` manifests;
- the four reserved-final targets frozen in the pilot `progress.json` / `cohort.json`;
- any target consumed in an earlier stage of this protocol.

Resolution and exclusion run on CPU on the server before any GPU job, as `collect_guard_traces resolve` does. The reserved-final targets are only read for exclusion, never evaluated. This cohort is attacker validation only and never enters detector fitting.

**Every job:**

- AMIA, `attack_variant: causal_gradient_alignment`, `attack_targets: 1`;
- `observation_defense: none`, no `client_guard`, no RAG pipeline (RAG costs more GPU time than the attack and is irrelevant here);
- `counterbalance_trials: true`, `threshold_mode: calibrated`, `calibration_fpr: 0.05`;
- `calibration_nonmember_count: 100` (up from 20, to test calibration);
- **40 trials (20 pairs)** per target. Observations are cheap next to FL training.

**Telemetry (small code change, before any run).** Record in each trial row the three numbers the server already sees: `dot(released, direction)`, `‖released‖` and `‖direction‖`. Record the same for calibration batches. This lets the projection score and the cosine score be compared afterwards without rerunning. It changes the core fingerprint, which is acceptable because these are new results. No gradients or private text are stored.

**Stage A — attacker tuning (consumed).**

- 4 targets × 2 arms: `probe_epochs` ∈ {12, 80}, everything else fixed (8 jobs).
- Selection: the arm × score pair (projection or cosine) with the highest mean per-target AUC. Ties go to fewer epochs, then to projection.
- Stage A targets are then consumed and are not reused in Stage B.

**Stage B — evaluation.**

- 10 fresh targets, with the Stage A choice frozen (10 jobs).
- The calibrated threshold uses the selected score.

**Rough cost.** Pilot jobs took about 10–25 minutes each, including RAG and shadow-guard checks. Removing both should shorten them. 18 jobs is therefore very roughly 3–6 GPU-hours. This is untimed and should be recorded on the first job.

## Endpoints (Stage B, fixed before it runs)

- **Primary — operational attack.** Per-target ROC AUC of the selected score, members vs nonmembers. Pooled as the mean of per-target AUCs, with a 95% interval from 10,000 bootstrap resamples over targets. Targets are the unit; trials are never treated as independent across targets.
- **Calibrated operating point.** TPR and observed FPR at the calibrated threshold, per target and pooled with target-cluster bootstrap intervals. Compare the observed FPR on private nonmember batches with the 5% target.
- **Paired effect.** Fraction of pairs with member > nonmember. Report the exact sign-flip permutation p-value within pairs, plus a target-cluster bootstrap interval.
- **Negative control.** Permuting labels within pairs gives each target's exact null AUC distribution; report where the observed AUC falls.

## Decision rule

- **Effective:** the pooled-AUC 95% lower bound is at least 0.60. Detector work on causal requests is justified. Proceed to matched-reference benign controls.
- **Not effective enough:** the pooled-AUC 95% upper bound is below 0.60. Stop escalating the causal-request detector. Record a negative result, and do not present concentration as blocking a validated attack.
- **Inconclusive:** anything else. Extending the study requires a new pre-registration, never adding targets after seeing results.
- **Calibration failure,** reported separately: an observed pooled FPR 95% lower bound above 0.10 means the attacker's public calibration does not transfer. Report operational claims only at FPR measured on private data.

## Not shown by this study

- It measures one attack family, one model (Qwen2.5-0.5B), batch size 4, and no observation noise.
- It says nothing about the detector, enforced guard runs, adaptive evasion, or utility.
- Results on this cohort must not be used to tune the detector or its threshold.

## Implementation (done before any GPU run)

- **Telemetry:** `alignment_terms` in `master_script/core/attacks/causal_probe.py`, recorded per trial and per calibration batch in `amia.py`. Existing projection scores are bit-identical. Core fingerprint `4b82f0dd4eb4` → `521a82164f09`.
- **Tool:** `master_script/tools/causal_attack_validation.py` provides `prepare`, `select` and `analyze`. The unchanged `collect_guard_traces` resolves and runs the launches: it freezes and excludes targets before any GPU job, runs one job at a time, checks each result against its declared target, and retires weights.
- **Tests:** `tests/test_causal_attack_validation.py`.

## Run sequence (on the server)

1. **Prepare Stage A.** Pass `--exclude-manifest` for every earlier cohort: the v4 pilot `cohort.json`, which holds the reserved-final targets, the v4 `splits.complete.json`, and the v3 splits if still available. `prepare` refuses to continue unless at least 4 reserved-final (`final`) targets are excluded.
2. `collect_guard_traces resolve` the launch, which runs on CPU and freezes the targets. On a collision, prepare a new directory with `--first-seed` still inside the band. Never edit a launch.
3. `collect_guard_traces run --gpu N --max-jobs 1`, repeated until all 8 jobs are complete. Inspect each job before starting the next.
4. `select` on the Stage A directory writes `selection.json`. It is frozen and never overwritten.
5. **Prepare Stage B** with `--selection` and Stage A's `splits.complete.json` added to the exclusions, then resolve and run its 10 jobs the same way.
6. `analyze` on the Stage B directory with the selection gives the endpoints and the decision.
