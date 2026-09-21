# AMIA light pilot and comparison with Reference

Nine completed experiments; no failed runs to exclude. Total elapsed time: 4.88 hours. Each condition has 20 balanced batch observations across only two unique target records. The same target identities, seeds, batch-pair seeds and labels are used across conditions. All files report Firestore saving; remote persistence was not independently checked.

Confusion counts, balanced accuracy, pooled score AUROC, threshold decisions and RAG utility averages were independently recomputed from saved rows. All checks passed. These checks establish internal consistency, not an independent rerun of the training process.

## Attack results

`adv` means balanced accuracy, not TPR minus FPR. Chance level is 50%.

| Condition | Balanced accuracy | AUROC | TP / TN / FP / FN | Target 0 correct | Target 1 correct |
|---|---:|---:|---|---:|---:|
| baseline | 95% | 1.00 | 10 / 9 / 1 / 0 | 9/10 | 10/10 |
| gradient_clip_only | 95% | 1.00 | 10 / 9 / 1 / 0 | 9/10 | 10/10 |
| gradient_gaussian_sigma1 | 50% | 0.56 | 1 / 9 / 1 / 9 | 5/10 | 5/10 |
| gradient_gaussian_sigma2 | 50% | 0.59 | 0 / 10 / 0 / 10 | 5/10 | 5/10 |
| feature_bitrand_epsilon5 | 85% | 1.00 | 10 / 7 / 3 / 0 | 9/10 | 8/10 |
| feature_bitrand_epsilon20 | 90% | 1.00 | 10 / 8 / 2 / 0 | 9/10 | 9/10 |
| feature_ome_epsilon5 | 95% | 1.00 | 10 / 9 / 1 / 0 | 9/10 | 10/10 |
| feature_ome_epsilon20 | 95% | 1.00 | 10 / 9 / 1 / 0 | 9/10 | 10/10 |
| training_dp_sgd_only | 100% | 1.00 | 10 / 10 / 0 / 0 | 10/10 | 10/10 |

## What the results establish

**AMIA succeeds strongly in the tested setup.** Baseline detects all ten positive batches and falsely flags one of ten negative batches (95% balanced accuracy). Seven of nine settings retain 85–100% accuracy. These repeated batches cover two targets, so this is not evidence of success across twenty independent records.

**Clipping alone does not suppress the attack.** Its accuracy remains 95% and AUROC remains 1.00. This is consistent with magnitude limiting leaving a detectable distinction between target-present and target-absent gradients. The result supports that explanation, rather than proving it universally.

**Feature randomization leaves score separation intact.** BitRand reaches 85% and 90% calibrated accuracy; OME reaches 95%. All four have pooled AUROC 1.00. Thus the lower BitRand accuracy does not establish removal of the membership signal: at least on these saved scores, a threshold can separate positives and negatives perfectly. Selecting that threshold on test data would not be a valid estimate of deployment accuracy. Feature privacy is conditional on the next-token label; the archive explicitly disclaims full-text or training privacy from these mechanisms.

**Gaussian noise on returned probe gradients weakens this attack.** Noise multipliers 1 and 2 both give 50% balanced accuracy, with AUROC 0.56 and 0.59. Sigma 1 gives one true positive and one false positive; sigma 2 predicts every batch negative. There is no evidence here that sigma 2 is decisively better. This concerns the observed probe-gradient release. It is not a demonstrated defense against every active strategy or repeated-query attack.

**Training-only DP-SGD leaves AMIA successful.** At training noise multiplier 2, AMIA reaches 100% accuracy. Reference under training DP-SGD noise 2 previously reached 50%. This does not contradict a training guarantee: the saved AMIA methodology measures membership in a fresh private client batch through a malicious next-token head and its returned gradient. Reference measures membership in model training records through model scores. The observation channel remains separately exposed.

## RAG utility and membership

Means cover two evaluated fine-tuned models, with 200 membership candidates and 10 utility questions per corpus per model. These repeated questions must not be counted as independent samples across conditions.

| Condition | Private ordinary RAG attack accuracy | Private ordinary answer F1 | Private exact match | Public ordinary answer F1 |
|---|---:|---:|---:|---:|
| baseline | 57.00% | 47.86% | 40.0% | 15.27% |
| gradient_clip_only | 57.00% | 47.86% | 40.0% | 15.27% |
| gradient_gaussian_sigma1 | 57.00% | 47.86% | 40.0% | 15.27% |
| gradient_gaussian_sigma2 | 57.00% | 47.86% | 40.0% | 15.27% |
| feature_bitrand_epsilon5 | 57.00% | 47.86% | 40.0% | 15.27% |
| feature_bitrand_epsilon20 | 57.00% | 47.86% | 40.0% | 15.27% |
| feature_ome_epsilon5 | 57.00% | 47.86% | 40.0% | 15.27% |
| feature_ome_epsilon20 | 57.00% | 47.86% | 40.0% | 15.27% |
| training_dp_sgd_only | 71.00% | 14.61% | 0.0% | 22.00% |

Gaussian observation noise preserves these measured answer scores: private ordinary answer F1 stays 47.86% and exact match stays 40%. Unlike the earlier Reference FedAvg results, the low attack score is not accompanied by collapse of measured RAG answer quality. However, the noise protects probe observations, not the generator training process, so these results do not measure the utility cost of adding equivalent noise to every ordinary FL update.

Private ordinary RAG membership accuracy remains 57% under gradient-noise defenses, with TPR 50% and FPR 36%. Blocking AMIA has not demonstrated removal of retrieval membership leakage. The modest above-chance RAG result needs more independent models and questions before claiming reliable discrimination. Training-only DP-SGD yields private RAG attack accuracy 71% while answer F1 falls to 14.61%.

Across the first eight conditions, classification and answer-quality averages coincide. These conditions share the same training mechanism; their interventions concern probe observations or frozen features. Repeated matching outcomes should not be presented as eight independent confirmations of RAG utility.

For AMIA baseline private RAG, Mirabel reduces attack accuracy from 57% to 53%, but answer F1 falls from 47.86% to 18.92%. Instructions alone increase attack accuracy to 59.75% while answer F1 rises to 51.50%. Mirabel plus instructions gives 52.5% attack accuracy and 21.5% answer F1. The retrieval privacy/utility tradeoff remains.

## Comparing the attack families

| Property | AMIA pilot | Reference pilot |
|---|---|---|
| Membership question | Is the target in this private client batch? | Was the target used to train the model? |
| Observable | Returned gradient of malicious probe/head | Model score relative to reference model |
| Baseline balanced accuracy | 95% | 100% |
| Training DP-SGD sigma 2 | 100% | 50% |
| Unique target coverage per condition | 2 | 10 paired targets |

Do not conclude that Reference is superior from 100% versus 95%: the membership questions, attacker access and independent target coverage differ. Both attacks succeed as pilot baselines. The contrast under training DP-SGD motivates separately protecting training outputs, client gradient observations and retrieval responses.

## Formal privacy and reliability limits

Gaussian observation accounting reports composed epsilon approximately 82.92 (sigma 1) and 31.46 (sigma 2), at delta 0.00001, over 20 private-batch gradient releases. These bounds exclude training, feature privacy and retrieval releases; they are not strong end-to-end privacy guarantees. Low measured attack accuracy is not equivalent to a small privacy bound.

There is one sweep seed, one model/dataset setup, two unique AMIA targets and ten negative observations per condition. Test FPR resolution is 10 percentage points. Calibration on 200 negative batches does not add independent test outcomes. No general low-FPR reliability, broad attack ranking or universal defense effectiveness is established. Raw generated answer text is not present for qualitative auditing.

## Study implication

The most promising tested AMIA-specific intervention is protection of the returned probe gradients: it suppresses this attack while leaving measured RAG quality intact. The next validation should broaden independent target and seed coverage and test adaptive/repeated observations. An end-to-end defense still needs a separate treatment of retrieval leakage. Training DP alone should not be described as protecting every fresh client interaction.

Source archive SHA-256: `e480f62a8679583dfdd90ae70184f482cf797839aae0a2d229a3b235fb96cab9`
