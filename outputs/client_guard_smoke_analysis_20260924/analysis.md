# Guard v2 GPU smoke interpretation — 24 September 2026

The received report supports three-round guard operation and an architecture-preserving AMIA bypass on the tested target. It does not yet support a usable learned privacy defense. Prioritize measured guard overhead and absolute no-context utility before expanding the collection stage unchanged.

## Evidence boundary

Source: the user-supplied `smoke-analysis` JSON for batch `20260922-173328-2d75dea95957`, preserved byte-for-byte as `smoke-analysis.received.json`. All 11 condition, provenance and study entries join on unique run IDs. Reported per-client totals agree with reported training-request totals. All entries report `complete`, implementation fingerprint `1423ad8f55aa`, seed 1000, four clients and three FL rounds. No detector artifact is configured.

The raw 11 result files and remote manifest have not been supplied locally. Input-file hashes in the received report are provenance claims, not locally verified checksums. This analysis does not independently recompute confusion matrices, gradients, calibration, token exposure or timings from raw observations. `analyze_summary.py` regenerates the compact `summary.json` from the received report.

AMIA has one target with four observations per condition (two per membership world). Reference has one paired target, two separate training worlds. The target identity matches across these families. The 11 runs are not 11 independent targets. The 200 RAG document groups are repeated across conditions/checkpoints; their bootstrap intervals do not establish model-world generalization.

## Privacy

| Attack | Condition | BA | Per-observation AUROC | Interpretation |
|---|---|---:|---:|---|
| AMIA head | Baseline | 0.75 | 1.00 | Score ranking and chosen threshold differ |
| AMIA head | Rules | Unavailable | Unavailable | Release coverage zero; prevention, not zero attack success |
| AMIA head | Noise | 0.50 | 0.75 | Chance threshold accuracy does not imply absent ranking signal |
| AMIA head | Shadow | 0.75 | 1.00 | Logging does not prevent the attack |
| Causal gradient alignment | Baseline | 1.00 | 1.00 | Reported success for the distinct full-LM variant on this target |
| Causal gradient alignment | Rules | 1.00 | 1.00 | Release coverage 1.0: rules do not prevent this variant |
| Causal gradient alignment | Noise | 0.50 | 0.75 | Same calibration/interpretation concern |
| Causal gradient alignment | Shadow | 1.00 | 1.00 | Logging does not prevent the attack |
| Reference | Baseline / rules / shadow | 1.00 | 1.00 | No measured training-membership protection |

The two noise arms have world-mean AUROC 1.00 when the two observations in each membership world are averaged. This describes the ordering of just one positive mean and one negative mean. It is a useful warning against relying on BA=0.50, not evidence of a general repeated-query recovery attack. Freeze directions and thresholds on independent defended validation targets before assessing final targets. The earlier noise AUROC reversal does not recur in these reported raw scores, but the changed target/configuration prevents attributing that difference solely to defense changes.

Decision and timing summaries remain descriptive and have one independent target. No fitted held-out joint transcript-attack result was supplied. A constant refusal decision cannot establish that the complete transcript is private.

## Legitimate operation and overhead

All six guarded/shadow arms report zero training rejections and zero policy flags: 96 request events in total, covering 24 completed round instances and zero aborted rounds. Each AMIA guarded arm contributes 12 requests and three rounds; each Reference guarded arm contributes 24 requests and six rounds across two worlds. These repeated related events cannot establish a <=1% population rejection rate. The report also cannot establish convergence or absence of client utility disparities because it omits loss trajectories and per-client utility.

| Guarded arm | Median matched round overhead | End-to-end increase |
|---|---:|---:|
| Head / rules | 91.4% | 25.4% |
| Head / shadow | 94.0% | 29.5% |
| Causal / rules | 83.6% | 34.5% |
| Causal / shadow | 81.4% | 34.1% |
| Reference / rules | 87.0% | 38.8% |
| Reference / shadow | 84.1% | 37.6% |

Every guarded arm descriptively exceeds the provisional 10% round-overhead target. These are dependent, sequential comparisons, not confidence bounds or randomized performance estimates. End-to-end differences include skipped attack work and other phases, and should not replace round comparisons.

Reference rules is the cleanest supplied profile because all its guard events concern training. It reports 306.25 guard seconds across 24 requests, or 12.76 seconds/request. Feature extraction takes 130.01 s (42.5% of summed profile components), reference reads 65.94 s (21.5%), and the two hashes together 96.76 s (31.6%). Copying accounts for about 2.4%, numerical validation 2.0%, ledger setup/reservation 0.02%; detector loading/inference is zero. Audit commit time is outside those component percentages. Thus classifier inference and ledger operations are not the observed bottleneck. Optimizing feature extraction alone cannot remove the remaining several seconds of reference I/O and hashing. The earlier tiny CPU feature benchmark did not establish acceptable actual model overhead.

The causal noise-only arm has only +0.76% median training-round overhead but +54.3% total runtime. That is consistent with added work outside ordinary training; raw phase timings are required before attributing it specifically to full-vector noise/calibration.

## Utility and RAG

No-context F1 is 1.783% for AMIA checkpoints and 2.117% averaged across Reference worlds. Every condition fails the report's provisional 10% absolute F1 screen. All supplied matched F1 and EM differences are zero: the defense preserves these measured answers, while the baseline no-context capability on this particular utility set remains low. This is not evidence that the entire model is unusable or that the guard caused a utility decline. Diagnose the task, prompt, truncation and pre/post-training baseline before changing training hyperparameters.

For AMIA checkpoints, ordinary public RAG F1/EM is 20.94%/10%, private RAG 40.67%/30%. Across the two Reference checkpoints, means are public 17.02%/5% and private 38.18%/25%. Question subsets differ between public/private retrieval, so these are not a controlled comparison of corpus quality. All matched defenses preserve F1/EM on these slices. No semantic answer-support audit was supplied.

Datastore AUROC is unchanged across conditions: AMIA public/private 0.57/0.53; Reference's document scores averaged across its two checkpoints give 0.56535/0.53075. Recognition is 100% and refusal 0% for these ordinary membership queries. Recognition means the parser found an answer, not that the answer is grounded or an informative membership decision. TPR/FPR and raw answers are absent from this report, so the earlier broad positive-response bias cannot be resolved from these aggregate AUC values.

The Reference overlap cells are now [neither=0, training-only=0, datastore-only=1, both=1]. For this target, predictions track datastore presence in both training worlds; the reported interaction is zero. The prior constant-positive pattern is not universal. This does not isolate why the previous pilot differed because the target, checkpoint and implementation changed. Raw token-exposure, retrieval and truncation audits are still needed before claiming the four cells meet every validity check.

## Reporting issues and next work

`excluded_runs=0` and `exclusion=null` do not mean all utility gates pass: the report simultaneously marks every no-context `utility_valid=false`. The current exclusion field only screens missing/zero utility. Future reports should separate operational inclusion from failure of the absolute utility floor. Keep all completed runs, including utility-invalid ones, in the study record.

Next work should be bounded and use existing artifacts first:

1. Obtain the 11 raw result JSONs and `manifest.json`. Inspect raw guard features, per-round losses, processing times, request hashes, memory/communication, calibration scores and overlap validity. Absolute run durations are absent from this derived report, so a credible collection-runtime estimate cannot yet be made.
2. Compare benign later-round feature ranges with the accepted causal malicious requests. Treat this single target as a diagnostic only; do not train/validate/test a detector by splitting these correlated events.
3. Profile and optimize reference I/O, hashing and feature passes while preserving exact approved-state integrity and the checked snapshot. Measure any reduction with a small matched baseline/rules/shadow rerun; never use unverified cached state to meet the overhead target.
4. Audit low no-context task quality using the existing checkpoints and raw local utility answers, including a pretrained checkpoint control. Do not count unchanged low F1 as a successful usable defense.
5. Then collect independent detector targets with the source revision and feature schema frozen. The proposed eight-target collection is a mechanics dataset, not enough to substantiate a 1% FPR gate. Reserve whole targets and at least one attack variant before fitting. Run the seven-arm comparison only with a pinned detector and independent attacker calibration.

No training, remote job, source-code edit or publication was performed as part of this report review. Historical results and configurations were not changed.
