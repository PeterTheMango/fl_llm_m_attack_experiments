# Causal gradient-alignment attack — Stage B result

**Decision: effective** under the pre-registered rule. The pooled mean per-target AUC is 0.992 with a 95% interval of [0.981, 0.999]. The rule needs a lower bound of at least 0.60. **No calibration failure** is flagged: the private FPR lower bound is 0.07, and the rule needs more than 0.10. However, the public calibration aimed at 5% FPR and gave **16.5%** on private batches, and the interval excludes 5% (see below).

This is a result about the attack alone, with no guard in the loop. It makes no privacy claim about the guard or the detector.

## Provenance

- **Inputs.** Stage B is 10 targets, seeds 6004–6013, with the frozen Stage A selection of `probe_epochs 12` and the **cosine** score. The protocol and analysis rules were fixed in [`protocol.json`](protocol.json) before Stage A ran.
- **Code.** Branch `analysis/causal-attack-validation` at `5b8b593`, core fingerprint `521a82164f09`.
- **The analysis** was run by the researcher on LCALC08 with `causal_attack_validation analyze` on 2026-09-30.
- **How the numbers got here.** Every number on this page is transcribed from what the researcher pasted into the session, and nothing else. They pasted the output of `cat "$OUT/analysis.json"` and `sha256sum` of three server files. The paste had two lines with shifted indentation. Re-serialising it the way the tool writes files (`json.dumps(indent=2)` plus a newline) reproduced the server file exactly. The committed [`analysis.json`](analysis.json) therefore has the server's hash.

| Server file | sha256 (pasted `sha256sum`) |
|---|---|
| `$OUT/analysis.json` (= committed [`analysis.json`](analysis.json)) | `f512284dc14bf1e1b501102e731a9571c1af1fd17c55d44686d161fb010a4e35` |
| `$OUT/selection.json` | `1c4aeb0cc383bb8da9b568d10835c915e2d43954a12b9f2af80cdd25bbb1e25e` |
| `$OUT/stage-b/launch.json` | `b2b23038b8c896dc63b7a0ee90ebeb066d744f7144b989f4636b800b1b448ab7` |

- **Hash checks.** Inside `analysis.json`, `protocol_sha256` (`619d8d9f…`) matches the committed `protocol.json`, and `validation_tool_sha256` (`9acd6cd1…`) matches `master_script/tools/causal_attack_validation.py` at `5b8b593`. The `selection_sha256` and `stage_b_launch_sha256` fields match the pasted `sha256sum` lines.
- **Earlier cross-check.** Before the full file arrived, the pooled values were rebuilt locally from the per-target rows with the tool's own bootstrap. The rebuild agreed exactly with the authoritative output for AUC, TPR, FPR and the paired results.

## Endpoints (Stage B, cosine score, 10 targets × 20 pairs)

| Endpoint | Value | 95% interval (target bootstrap, 10,000 resamples) |
|---|---:|---|
| Primary: mean per-target ROC AUC | **0.99175** | [0.98075, 0.99850] |
| Calibrated operating point: TPR | 0.965 (193/200) | [0.92, 1.00] |
| Calibrated operating point: private FPR (target 0.05) | **0.165** (33/200) | [0.07, 0.275] |
| Paired: fraction of pairs with member > nonmember | 1.00 (200 positive, 0 negative, 0 ties) | [1.00, 1.00] |
| Paired: exact one-sided sign test | p = 2⁻²⁰⁰ ≈ 6.2 × 10⁻⁶¹ | — |
| Negative control: null percentile of the mean AUC | 1.0 | 10,000 within-pair label permutations |

Every per-target AUC also sits at null percentile 1.0.

### Per target

Rows are in the order of `per_target` in `analysis.json`. Targets are shortened to the first 12 hex digits of `target_sha256`.

| Target | AUC | Threshold (cosine) | TP / 20 | FP / 20 | Private FPR | Pairs member > nonmember |
|---|---:|---:|---:|---:|---:|---:|
| `57bd68ffe4f5` | 0.9975 | 0.592 | 18 | 0 | 0.00 | 20/20 |
| `ab7d56856fee` | 0.9925 | 0.647 | 20 | 2 | 0.10 | 20/20 |
| `aca057863e33` | 1.0000 | 0.604 | 20 | 11 | **0.55** | 20/20 |
| `37563aeeb058` | 1.0000 | 0.679 | 20 | 4 | 0.20 | 20/20 |
| `8f365cca5621` | 0.9950 | 0.719 | 19 | 2 | 0.10 | 20/20 |
| `5cae99b086e3` | 0.9900 | 0.855 | 20 | 3 | 0.15 | 20/20 |
| `d9cc27232bd7` | 0.9975 | 0.867 | 20 | 1 | 0.05 | 20/20 |
| `70766304186e` | 1.0000 | 0.627 | 20 | 8 | **0.40** | 20/20 |
| `98f84d63a2c5` | 1.0000 | 0.137 | 20 | 0 | 0.00 | 20/20 |
| `e558fb3f494a` | 0.9450 | 0.962 | 16 | 2 | 0.10 | 20/20 |

## Decision against the pre-registered rule

| Rule | Condition | Observed | Outcome |
|---|---|---|---|
| Effective | pooled AUC 95% lower bound ≥ 0.60 | 0.981 | **met** |
| Not effective | pooled AUC 95% upper bound < 0.60 | 0.999 | not met |
| Calibration failure (reported separately) | pooled private FPR 95% lower bound > 0.10 | 0.07 | not flagged |

## The calibration concern

Each target's threshold is the 95th percentile of the cosine score on 100 public nonmember batches, so it aims at 5% FPR. On the private victim batches it gave:

- **Pooled FPR of 16.5% (33/200), with an interval of [0.07, 0.275].** The interval excludes the 5% target. The formal failure flag stays off only because its bar is a lower bound above 10%.
- **Wide spread between targets.** Private FPR runs from 0 to 0.55. Four of the 10 targets exceed 10% (0.55, 0.40, 0.20, 0.15).
- **Scores on different scales.** Thresholds range from 0.137 to 0.962, so the cosine score's scale differs a lot between targets. A single threshold shared across targets would not work.

Private nonmember batches clear the public threshold more often than public ones do. The pilot review already suspected this kind of public-to-private shift, but this study does not test its cause. The practical rule is to quote the operating point as **TPR 0.965 at an observed private FPR of 0.165**, never as "at 5% FPR". With 20 nonmembers per target, low-FPR operating points such as TPR at 1% FPR cannot be estimated from this cohort. They were not an endpoint.

AUC is a ranking within each target, so the primary endpoint does not depend on calibration.

## What this means for the research

- **The attack is validated as effective with the cosine score.** A victim that faithfully returns its gradient for a causal request reveals whether a known candidate record is in its batch, nearly perfectly in this setting. The member batch scored higher in all 200 pairs.
- **The implemented projection score was unreliable.** In Stage A (tuning data, 4 targets) it gave a mean AUC of 0.868 at 12 epochs and **0.331 at 80 epochs**, below chance, while cosine gave 1.000 and 0.958. The projection `dot / ‖direction‖` grows with the batch's overall gradient size. Cosine divides that size out. Stage B evaluated only the frozen cosine score. Any projection numbers computed on Stage B now would be exploratory, not an endpoint.
- **The pilot's "causal AUC 0.50" never showed the attack failing.** Each pilot job had 4 trials (2 pairs) of the projection score at 12 epochs, with a threshold from 20 public batches. That design could not tell an effective attack from chance.
- **Detector work on causal requests is justified.** Per the protocol, the next step is matched-reference benign controls. That study needs its own pre-registration.
- **Attack effectiveness is not a privacy claim about the guard.** The guard was not in the loop. The detector's 4/4 detection of causal requests in the pilot shows detection, not protection. The guard's own gates remain open, for example legitimate rejection of 8.33% against a gate of ≤1%.
- **The Stage A and Stage B cohorts are attacker-validation data only.** They must not tune the detector or its threshold. Later cohorts must exclude both stages' `splits.complete.json`.

## Limits (from the protocol, plus what the data add)

- **Narrow setting.** One attack family, one model (Qwen2.5-0.5B-Instruct), batch size 4 (the target is a quarter of the victim's batch), and no observation noise.
- **No guard, RAG, adaptive evasion or utility.** None of these were in the loop, so the study says nothing about them.
- **Small cohort.** 10 Stage B targets with 20 pairs each. Targets are the unit of the intervals.
- **Membership inference on a candidate the server already knows.** It is not extraction.
- **Calibration covers only 5% FPR.** It was tested at that one target, with 100 public batches per target.

## Follow-ups (not part of this result)

- **Fix the causal attack's defaults.** `amia.py` still scores with the projection (`causal_probe.score_from_terms`) and defaults to `probe_epochs: 80`, the weakest Stage A configuration. Results from the default causal attack therefore understate it. Future causal runs should use the validated configuration (12 epochs, cosine) with thresholds calibrated on private data. Changing the implemented score changes the core fingerprint, so it goes in a separate change.
- **Fix a misleading collector message.** `collect_guard_traces` says "overlaps the diagnostic smoke target" for any excluded target. It can be fixed now that the study's launches are finished.
- **Delete the interrupted job's partial weights.** Remove them from `stage-a-2/interrupted/job-001-attempt-1` on the server, and keep `stage-a-2/interruptions.json` and the folder's inventory.
