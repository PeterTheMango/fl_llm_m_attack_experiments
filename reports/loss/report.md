# LOSS Membership Inference on a Federated LLM — Results Report

**Attack:** `loss` (per-token negative-log-likelihood threshold)
**Primary source:** Yeom, Giacomelli, Fredrikson, Jha. *Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting.* CSF 2018 (arXiv:1709.01604).
**Experiment:** `loss_federated_llm_adaptation_v1` (run `loss_federated_llm_adaptation_v1_952b56bc2ecd70aa`), status `complete`.

## What was run

The experiment ports Yeom et al.'s LOSS decision rule to a federated LLM fine-tuning setting. Four clients locally fine-tune `sshleifer/tiny-gpt2` with Flower's FedAvg strategy (2 federated rounds, 1 local epoch, `client_lr` 5e-5, batch size 4, `max_length` 64) over `synthetic_private_client_text`. Each of the 12 attack trials is a paired "world": odd/even `trial_id`s alternate between a member world (the target sequence is included in target client 0's data) and a non-member world (it is held out, replaced by a labelled canary). The adversary observes only the final global model, scores the target sequence by its average per-token NLL, and predicts membership when that loss falls below a threshold calibrated to the 0.1 quantile of 24 calibration non-members. This preserves Yeom's rule — *lower loss ⇒ predict member* — while replacing classical supervised models with FedAvg causal-LM fine-tuning.

## Headline results

| Metric | Value |
|---|---|
| Trials | 12 (6 member / 6 non-member) |
| Member mean loss | 10.829094 |
| Non-member mean loss | 10.829199 |
| Member–non-member gap | 1.06 × 10⁻⁴ |
| ROC-AUC (inverted loss) | 0.944 |
| TPR | 0.00 |
| TNR | 1.00 |
| Accuracy | 0.50 |
| Reported "Adv" (½·TPR + ½·TNR) | 0.50 |

## Evaluation against the paper

**The direction of the signal confirms Yeom's central claim.** The theory says membership leaks through the train-vs-held-out generalization gap: members receive lower loss than non-members, and the membership advantage scales with that gap (Theorem 2, `Adv = R_gen(A)/B`). The results reproduce this ordering — members average a lower loss than non-members (10.829094 vs 10.829199), and the threshold-free ranking is strong: ROC-AUC on inverted loss is 0.944, meaning 34 of 36 member/non-member pairs are ordered correctly. Five of the six members score below *every* non-member; only the highest-loss member overlaps the non-member range. The latent loss signal Yeom predicted is unmistakably present.

**The magnitude confirms the contrapositive just as strongly.** The same theory implies that when a model barely overfits, membership advantage approaches zero. Here the generalization gap is ~10⁻⁴ in absolute terms, because the setup produces almost no learning: `tiny-gpt2` is a minimal randomly-initialized architecture, trained for only 2 rounds of FedAvg with a single local epoch, and the target sequence lives in just one of four clients whose updates are averaged. Per-round client losses hover at 10.82–10.83 throughout (round 0 → round 1 changes are in the third–fourth decimal), so the global model never meaningfully memorizes the target. A near-zero gap is exactly what Yeom predicts for a near-stable, non-overfit model — the loss attack has almost nothing to exploit in absolute terms.

**The operational attack achieves no membership advantage — a calibration failure, not a contradiction of the theory.** Every trial predicts "non-member" (TPR = 0, TNR = 1), giving chance-level accuracy (0.50). The cause is visible in the data: the calibrated threshold sits at ≈10.8195, while every evaluated target sequence scores ≈10.829. All scores lie *above* the threshold, so the *lower-loss ⇒ member* rule never fires. The 0.1-quantile threshold was set from calibration non-members whose intrinsic loss (~10.8195) is ~0.01 lower than the canary target sequences (~10.829) — a systematic offset roughly 100× larger than the member/non-member gap the attack is trying to detect. This is precisely the weakness Yeom's framework and the LOSS summary flag: raw loss is biased by the intrinsic likelihood of the text, so an absolute threshold transferred from one text distribution to another mis-fires. The AUC stays high because it ranks *within* matched canary pairs, where the bias cancels; the hard threshold fails because it compares *across* distributions, where the bias dominates.

**A definitional caveat on the reported advantage.** The config reports `adv = 0.50` under the definition `½·TPR + ½·TNR` (balanced accuracy). With TPR = 0 and TNR = 1 this equals 0.5 — i.e., chance. Under Yeom's own definition (Definition 4, `Adv = TPR − FPR`), the advantage is 0 − 0 = 0. Both readings agree that the deployed test extracts no membership information at its operating point; the 0.50 figure should not be misread as a meaningful 50% advantage.

## Insights

1. **The loss signal exists but is operationally unusable here.** A 0.944 AUC alongside 0 TPR is the signature of a discriminative score paired with a broken decision threshold. The experiment demonstrates Yeom's *signal* (low-loss members rank above non-members) while failing to convert it into a *decision*.

2. **Raw-loss bias, not absence of leakage, breaks the threshold.** The ~0.01 offset between calibration text and target canaries swamps the ~10⁻⁴ membership gap. This is the canonical motivation for calibrated successors (`reference`, `zlib`, `min_k`): they normalize out intrinsic text difficulty, exactly the term that defeated the absolute threshold here.

3. **Federated, under-trained settings suppress the absolute gap.** Averaging one client's update across four participants over two rounds on a tiny model yields near-stability and thus near-zero overfitting — consistent with Yeom's prediction that advantage vanishes as the generalization gap closes. Stronger leakage would require more rounds/epochs, a larger model, or fewer clients diluting the target.

4. **Threshold-free and threshold-based metrics tell different stories; report both.** AUC alone would overstate the attack (0.944); accuracy/TPR alone would understate the underlying signal. The honest conclusion is that the signal is present and rankable but not calibrated for a usable operating point.

5. **Methodological gap to close.** Calibrate the threshold on held-out scores drawn from the *same* distribution as the target sequences (or switch to a difficulty-calibrated score) so the operating point lands inside the score range rather than ~0.01 below it. Twelve trials is also too few to treat the 0.944 AUC as stable, given how small the effect size is.

## Bottom line

The experiment faithfully reproduces the qualitative prediction of Yeom et al. — members carry lower loss and rank above non-members (AUC 0.944) — and equally reproduces the quantitative one — a near-stable, barely-trained federated model yields a vanishingly small generalization gap and therefore essentially zero exploitable advantage. The deployed thresholded attack returns chance-level performance not because the signal is absent but because the absolute loss threshold is miscalibrated against intrinsic text bias, the exact limitation that motivates the calibrated attacks built on top of LOSS.
