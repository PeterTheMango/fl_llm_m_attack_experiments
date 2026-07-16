# zlib-Entropy Ratio Membership Inference on a Federated LLM — Results Report

**Attack:** `zlib` (perplexity / zlib-entropy ratio)
**Primary source:** Carlini, Tramèr, Wallace, et al. *Extracting Training Data from Large Language Models.* USENIX Security 2021 (arXiv:2012.07805).
**Experiment:** run `1a7ec748918d23e0`, artifact `artifacts/zlib_adaptation/1a7ec748918d23e0`, status `complete`.

## What was run

The experiment transfers Carlini et al.'s zlib-entropy ratio attack — originally a pre-training extraction filter for GPT-2 XL — to a federated LLM fine-tuning setting. Four clients participate in a single round of Flower (`flwr`) FedAvg over `synthetic_client_text` on `sshleifer/tiny-gpt2` (`client_lr` 5e-5, `local_epochs` 1, `local_batch_size` 2, `max_length` 64, `seed` 7). Each of the 4 attack trials is a paired world differing only by target-client membership: trials 0 and 2 are member worlds (the target record is in target client 0's data), trials 1 and 3 are non-member worlds. The adversary observes only the final global model and scores the target record as `-(log_perplexity / zlib_entropy_bits)` — the orientation is chosen so that members, which the model finds easy beyond what generic compressibility explains, score *higher*. No reference model is used; zlib is the model-independent calibrator. Membership is predicted when the score exceeds the threshold `-0.0058`.

Per the run's own `methodology.deviation_from_source`, Carlini et al. flag fine-tuning as future work, so this is a transfer of their pre-training signal rather than a replication; and the smoke run uses a deterministic toy causal scorer rather than genuine fine-tuning (`use_hf_models=True` is required for the latter). Both caveats matter for interpreting the numbers below.

## Headline results

| Metric | Value |
|---|---|
| Trials | 4 (2 member / 2 non-member) |
| Member scores (trials 0, 2) | −0.0159292165, −0.0159291983 |
| Non-member scores (trials 1, 3) | −0.0159292417, −0.0159292502 |
| Member–non-member separation | +2.5 × 10⁻⁸ (members above) |
| Threshold | −0.0058 |
| ROC-AUC | 1.00 |
| TP / FP / TN / FN | 0 / 0 / 2 / 2 |
| TPR | 0.00 |
| TNR | 1.00 |
| Precision / Recall / F1 | 0 / 0 / 0 |
| Accuracy | 0.50 |
| Reported "Adv" (½·TPR + ½·TNR) | 0.50 |

## Evaluation against the paper

**The ranking signal points the right way — perfectly, by the metric's own account.** Carlini et al.'s premise is that dividing model perplexity by a model-independent compressibility term isolates sequences the model finds unexpectedly easy, the hallmark of memorization. Here every member outscores every non-member: the two member worlds (−0.0159292165, −0.0159291983) sit above the two non-member worlds (−0.0159292417, −0.0159292502), a clean +2.5 × 10⁻⁸ separation that yields ROC-AUC = 1.00. Taken at face value, the calibrated ratio orders members above non-members exactly as the paper's logic predicts — and does so more cleanly than the sibling `loss` run (AUC 0.944) on the comparable federated setup.

**But AUC = 1.00 here is not evidence of memorization, and the setup makes that explicit.** Two facts undercut any strong reading of the perfect AUC. First, with only 2 members and 2 non-members, perfect ordering is one of just a handful of possible arrangements and is easily produced by an effect unrelated to membership. Second, and decisively, the scorer is a *deterministic toy* and training is a *single* FedAvg round on a minimal randomly-initialized model — conditions under which the global model cannot have meaningfully absorbed, let alone memorized, the target record. A genuine membership signal therefore has nothing to draw on. The separation that does appear is ~2.5 × 10⁻⁸ and the four scores agree to roughly eight significant figures, consistent with the score being dominated by per-record `zlib_entropy` differences across four distinct target texts rather than by any train-vs-held-out effect. In other words, the attack is almost certainly ranking on intrinsic record compressibility that happens to align with the member/non-member labels, not on memorization. The perfect AUC is an artifact of a degenerate scorer and a tiny sample, not a confirmation of the paper's extraction result.

**The operational attack achieves zero membership advantage — a calibration failure identical in shape to the `loss` run.** Every trial predicts "non-member" (TP = 0, FP = 0, TN = 2, FN = 2), giving chance accuracy (0.50). The cause is plainly visible: all four scores cluster near −0.01593, while the decision threshold is −0.0058. Since the rule is *higher score ⇒ member*, and no score comes within 0.0101 of the threshold, the member branch never fires. That gap between the threshold and the top score is about 4 × 10⁵ times larger than the +2.5 × 10⁻⁸ separation the attack is trying to resolve. The threshold is not merely imperfect; it is placed in a region the scorer never reaches. This mirrors the `loss` experiment exactly — a discriminative ranking paired with a threshold transplanted from a different score regime — except that here the miscalibration is even more extreme because the toy scorer's outputs are compressed into a microscopic band.

**zlib's headline benefit — false-positive suppression — is present but vacuous here.** The paper's selling point is that compression-calibration strips boilerplate false positives (67% of zlib-flagged Internet samples were confirmed memorized, versus single digits for raw perplexity). This run does report FP = 0 and TNR = 1.00, i.e. perfect false-positive control. But that is achieved trivially by predicting "non-member" for everything, not by zlib correctly down-ranking repetitive text. The result cannot be credited to the calibration mechanism the paper describes.

**A definitional caveat on the reported advantage.** The config reports `adv = 0.50` under `½·TPR + ½·TNR` (balanced accuracy). With TPR = 0 and TNR = 1 this is exactly chance. Under Carlini et al.'s threshold-free framing the attack would instead be summarized by AUC (1.00 here), and under a TPR−FPR advantage it is 0 − 0 = 0. All readings agree the deployed test extracts no usable membership decision at its operating point; the 0.50 should not be read as a 50% advantage.

## Insights

1. **The decision threshold, not the score, is what fails — again.** As in the `loss` run, a strong-looking ranking (AUC 1.00) coexists with TPR = 0 because the threshold (−0.0058) lies ~0.01 above every score the model produces. The attack never gets the chance to be right or wrong on the member class; it simply never fires.

2. **A perfect AUC on 4 trials from a non-training run is a warning sign, not a success.** With no real fine-tuning and a deterministic toy scorer, there is no memorization for zlib to detect, so the perfect ordering most plausibly reflects record-intrinsic compressibility differences across four texts coinciding with the labels. The number is an artifact of sample size and scorer degeneracy and should not be reported as membership signal.

3. **zlib's claimed advantage is untestable in this configuration.** The whole point of the zlib ratio is to suppress boilerplate false positives relative to raw loss. The FP = 0 / TNR = 1 result here comes from predicting "non-member" everywhere, so it neither validates nor exercises the calibration mechanism. Demonstrating the paper's benefit requires a regime where the model actually fires positives.

4. **Federated, single-round, tiny-model conditions suppress any exploitable gap.** One round of FedAvg over four clients on `tiny-gpt2`, with the target in only one client, leaves the global model essentially untrained — the same near-stability that flattened the `loss` gap to ~10⁻⁴ collapses the zlib score band to ~10⁻⁸ here. Stronger leakage would require genuine fine-tuning (`use_hf_models=True`), more rounds/epochs, a larger model, and fewer clients diluting the target.

5. **Methodological gaps to close.** (a) Run with real HF fine-tuning so the perplexity term carries an actual membership signal rather than a constant. (b) Calibrate the threshold on held-out scores drawn from the same scorer/distribution as the targets so the operating point lands inside the score range. (c) Increase the trial count by orders of magnitude — 2 vs 2 cannot distinguish a real effect from a coincidental ordering, and AUC = 1.00 here carries essentially no statistical weight.

## Bottom line

The zlib transfer reproduces the *orientation* the paper predicts — members rank above non-members — and on paper does so perfectly (AUC 1.00). But that perfect ordering is an artifact of a deterministic toy scorer, a single untrained FedAvg round, and a 2-vs-2 sample, not evidence of the memorization the attack is designed to surface. Operationally the attack is inert: a threshold of −0.0058 sits ~0.01 above a score band of ~−0.0159, so it predicts "non-member" every time and returns chance accuracy. As with the companion `loss` run, the limiting factor is a threshold calibrated against the wrong score regime — compounded here by a scorer and training budget that leave zlib nothing real to calibrate. The configuration needs genuine fine-tuning, distribution-matched threshold calibration, and far more trials before any claim about the zlib ratio's privacy signal in federated LLMs can be supported.
