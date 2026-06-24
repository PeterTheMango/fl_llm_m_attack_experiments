# `loss` — The Loss / Negative-Log-Likelihood Threshold Attack

**Primary source:** Samuel Yeom, Irene Giacomelli, Matt Fredrikson, Somesh Jha. *Privacy Risk in
Machine Learning: Analyzing the Connection to Overfitting.* 2018 IEEE 31st Computer Security
Foundations Symposium (CSF). arXiv:1709.01604.

## Brief overview

The LOSS attack is the canonical, simplest MIA: it thresholds the model's per-example loss to
decide membership, predicting that a record was in the training set when the model's loss on it is
low. Yeom et al. formalised membership inference (their "membership advantage" definition) and
proved that an adversary using the loss signal gains advantage proportional to the model's
generalization error. The loss/error gap between training and held-out data is identified as the
direct cause of membership leakage, and is tied to overfitting. It is the reference-free, no-extra-
model baseline against which essentially every later LLM MIA is compared.

## Detailed attack mechanism

Yeom et al. give two concrete adversaries:

- **Adversary 1 (bounded loss).** Assume the loss is bounded, `ℓ(A_S, z) ≤ B`. On input
  `z = (x, y)`: query the model for `A_S(x)`, then output "non-member" with probability
  `ℓ(A_S, z)/B`, else "member". **Theorem 2:** this adversary's membership advantage equals
  `R_gen(A) / B` — the generalization error divided by the loss bound. For 0–1-loss classification
  with `B = 1`, membership advantage equals the generalization error exactly.
- **Adversary 2 (threshold / known error distribution).** Assume the conditional error densities
  `f(ε | member)` and `f(ε | non-member)` are known. Compute `ε = y − A_S(x)` and output the more
  likely class. For Gaussian errors `N(0, σ_S²)` (members) vs `N(0, σ_D²)` (non-members) with
  `σ_S < σ_D`, the optimal decision reduces to a threshold `|ε| < ε_eq`, and advantage grows with
  the ratio `σ_D / σ_S` (0 when the variances match, → 1 as the ratio → ∞).

The unifying decision rule is therefore: **lower loss ⇒ predict member**, with the optimal test
being a threshold on the loss (equivalently a likelihood ratio over the two error distributions).
The attack requires only black-box query access plus knowledge of the data distribution.

## Attack implementation

No public code repository is stated in the paper. The empirical evaluation used linear-regression,
decision-tree, and deep-CNN models on Eyedata, IWPC, Adult, MNIST, and CIFAR-10/100. When the error
variances `σ_S, σ_D` are unknown, the paper notes they can be estimated by repeatedly sampling
training sets `S ∼ Dⁿ`, retraining, and measuring the resulting error spread. In modern LLM MIA
toolkits the loss attack is implemented trivially as a single forward pass: score = the model's
average per-token negative log-likelihood (cross-entropy) over the target sequence, then threshold
or report AUC.

## Adaptation for LLM fine-tuning + data extraction

The loss-threshold rule maps directly onto language models by replacing the regression/classification
loss with the LM's per-example negative log-likelihood (equivalently log-perplexity). For a
fine-tuned LLM, compute the per-token average loss `−(1/n) Σ log f_θ(xᵢ | x_{<i})` on a candidate
sequence and threshold it: sequences seen during fine-tuning have lower loss/perplexity (the
train-vs-held-out generalization gap Yeom formalised) and are flagged as members. Because the
membership advantage scales with generalization error, fine-tuning on small, task-specific datasets
— which tend to overfit and are run for multiple epochs — amplifies the signal relative to
single-pass pre-training.

For data extraction, the same low-loss signal is the membership classifier applied to candidate
generations: high-confidence (low-perplexity) generations from the fine-tuned model are the ones
most likely to be memorized fine-tuning records, so an extraction-then-membership pipeline (later
formalised by Carlini et al. 2021) ranks generations by loss/perplexity to surface likely training
data. The loss attack is the weakest member of this family precisely because raw loss is biased by
the intrinsic likelihood of text, which motivates the calibrated attacks (`reference`, `zlib`,
`min_k`, etc.) that follow.
