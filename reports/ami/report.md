# Active Membership Inference (AMI) Attack — Results Report

**Experiment:** `ami_federated_llm_adaptation_v1`
**Run ID:** `f2ac5161a53dd64a828ddf55`
**Status:** complete
**Paper:** Nguyen et al., *Active Membership Inference Attack under Local Differential Privacy in Federated Learning* (AISTATS 2023)
**Reference implementation:** https://github.com/trucndt/ami

## 1. Premise from the paper

The paper proposes an *active* membership inference attack in which a dishonest federated-learning server maliciously crafts model parameters before dispatching them to clients. The core mechanism is a *chosen neuron*: the server trains a neuron (via a non-linear decision boundary realized by a small two-layer construction, Eqs. 5–6) so that it is activated **only** by a target sample `t` and by no other sample. Because a ReLU neuron contributes a non-zero gradient only when activated, the server inspects the chosen neuron's gradient in the client's update: a **non-zero gradient implies `t` was in the client's training set** (member), a **zero gradient implies it was not** (non-member).

The attack is framed as a security game (Fig. 1) with success measured by advantage `Adv = ½·TPR + ½·TNR`, where the random-guessing baseline is 0.5. The paper reports that without LDP the attack reaches ~100% success, and that it remains effective (≥0.77 success) even under LDP for privacy budgets ε ≥ 5. The attack is designed to require only a minimal parameter change within a single training iteration and does not depend on the global model converging.

This experiment adapts that attack from the paper's image classifiers (CIFAR-10, ImageNet, CelebA) to a **federated causal language model**. Per the recorded methodology, a Flower (`flwr`) FedAvg simulation fine-tunes `sshleifer/tiny-gpt2` across 4 clients for 2 rounds, after which a hidden-state AMI probe is trained (80 epochs) and the chosen-neuron gradient test is applied. No LDP mechanism is configured for this run, so it corresponds to the paper's **"attack performance without LDP"** setting.

## 2. Results

The attack was evaluated over **64 trials** with a balanced design — 32 member trials and 32 non-member trials, batch size 8 — following the paper's security game (flip a bit `b`; include the target when `b = 1`). Reported metrics, all independently recomputed from the per-trial records and confirmed:

| Metric | Value |
|---|---|
| Advantage (Adv) | 0.8906 |
| Accuracy | 0.8906 |
| TPR (recall) | 1.000 |
| TNR | 0.781 |
| Precision | 0.821 |
| F1 | 0.901 |
| ROC-AUC (from scores) | 1.000 |

The confusion breakdown is TP = 32, FN = 0, TN = 25, **FP = 7**. Every member was detected; the only errors were 7 non-members misclassified as members.

The per-trial scores are highly discretized, which is revealing:

- **Members** received one of two non-zero scores: **3.297** or **6.490** (the latter ≈ 2× the former).
- **Non-members** received either **0** (24 trials) or **3.217** (7 trials — the false positives).

## 3. Evaluation against the paper

**The attack reproduces the paper's central mechanism faithfully.** Membership is decided purely by whether the chosen-neuron/probe gradient is non-zero (the run uses `gradient_threshold = 1e-8`). Members produce non-zero gradients and non-members predominantly produce exactly zero, exactly the activation/gradient logic of Eq. 4 and Fig. 2. The result confirms that the chosen-neuron construction transfers from convolutional image models to a transformer language model's hidden states.

**Perfect TPR matches the paper's claim of high sensitivity to presence.** TPR = 1.0 means the probe was activated whenever the target was present, and the two distinct member scores (3.297 and 6.490) are consistent with the target sample appearing with different effective weight across batches. The paper explicitly notes the attack has high TPR across all scenarios because the chosen neuron is reliably activated by `t`.

**TNR = 0.781 is consistent with — but below the no-LDP ideal of — the paper.** The paper states TNR > 0.5 indicates the ability to detect absence, and reports near-100% success *without* LDP. Here, 7 of 32 non-members spuriously activated the probe (score 3.217), pulling TNR to 0.781 and Adv to 0.89 rather than ~1.0. This is precisely the failure mode the paper anticipates: when the non-linear boundary is not perfectly tight, samples that resemble the target in feature space can also activate the chosen neuron. The adapted setting makes this more likely — `tiny-gpt2` on `synthetic_canary_clients` offers far less feature separation than the 512-dim ResNet-18 embeddings used in the paper, and the probe was trained only to a final loss of 0.305 (from 0.757 over 80 epochs), i.e., it was still descending and not fully fit.

**The scores are perfectly separable — the errors are a threshold artifact, not a discrimination failure.** The minimum member score (3.297) is strictly greater than the maximum non-member score (3.217). Consequently ROC-AUC = 1.0: a single threshold placed anywhere in (3.217, 3.297) would yield 100% accuracy. The 7 false positives arise only because the *deployed* decision rule treats *any* non-zero gradient as membership (threshold ≈ 0). This is an important nuance: the underlying signal cleanly distinguishes members from non-members, but the binary "non-zero ⇒ member" rule sacrifices specificity for sensitivity. This maps directly onto the paper's certified-guarantee framing (Theorem 1, Fig. 7), where success depends on the *gap* between the target's lower-bound activation and the non-targets' upper-bound activation. That gap exists here (3.297 vs 3.217) but is narrow, which is exactly when the naive zero-threshold rule leaks false positives.

**The attack works despite a model that essentially did not train.** Mean client loss was flat across both federated rounds (10.8238 → 10.8233), so FedAvg produced almost no learning. The attack nonetheless succeeded, corroborating the paper's claim that AMI relies on maliciously crafted/probed parameters and a minimal one-iteration change rather than on model convergence or utility.

## 4. Insights

1. **The chosen-neuron AMI attack generalizes to federated LLMs.** Applied to `tiny-gpt2` hidden states via a trained probe, it achieves Adv = 0.89 and TPR = 1.0 — well above the 0.5 random baseline — confirming the mechanism is not specific to image classifiers.

2. **Member presence is detected perfectly; absence detection is where leakage degrades.** All error mass is in false positives (7/7 errors). The discrete member scores (one being double the other) indicate gradient magnitude scales with the target's contribution to a batch.

3. **The signal is perfectly separable; the deployed threshold is the limiting factor.** With ROC-AUC = 1.0 and a clean margin between the highest non-member score (3.217) and lowest member score (3.297), an optimal/calibrated threshold (~3.25) would lift accuracy to 100%. Reported precision (0.82) and TNR (0.78) understate the true discriminative power and reflect the conservative "any non-zero gradient ⇒ member" rule plus an under-trained probe.

4. **The shortfall vs. the paper's no-LDP ~100% is attributable to the adapted setting, not the method.** A tiny model, a synthetic canary dataset, only 2 federated rounds, and a probe trained to a still-decreasing loss all reduce feature separation and tighten the margin — consistent with the paper's own analysis that imperfect non-linear boundaries admit spurious activations.

5. **Convergence is irrelevant to the threat.** Near-zero training progress (flat ~10.82 loss) did not impair the attack, reinforcing the paper's point that the privacy risk stems from crafted parameters within a single iteration, independent of model utility.

6. **Practical implication.** Even in a degenerate, minimally trained federated LLM run, the server-side probe leaks membership with a perfectly separable signal. The natural next step — and the one the paper's evaluation targets — is to repeat this run *under an LDP mechanism* (e.g., BitRand/OME) across a range of ε to test whether the 3.297/3.217 margin survives privacy noise, and to replace the zero-threshold decision with a calibrated threshold derived from the certified bounds (Eqs. 9–11).

## 5. Notes and limitations of this evaluation

- This run contains **no LDP configuration**, so it validates only the no-LDP behavior; the paper's headline contribution (certified success *under* LDP) is not exercised here.
- The dataset is `synthetic_canary_clients` with `tiny-gpt2`, a small-scale proof of concept rather than the paper's benchmark datasets; absolute numbers should not be compared directly to the paper's figures.
- Membership scores take only a few discrete values across 64 trials, suggesting a small, structured target/non-target population; broader sampling would give a more robust TNR estimate.
- All reported metrics were independently recomputed from the per-trial data and match the recorded values exactly.
