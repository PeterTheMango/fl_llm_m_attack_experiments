# `min_k_plus_plus` — Min-K%++

**Primary source:** Jingyang Zhang, Jingwei Sun, Eric Yeats, Yang Ouyang, Martin Kuo, Jianyi Zhang,
Hao Frank Yang, Hai Li. *Min-K%++: Improved Baseline for Detecting Pre-Training Data from Large
Language Models.* ICLR 2025 (OpenReview ZGkfoufDaU). arXiv:2404.02936.

## Brief overview

Min-K%++ is a reference-free MIA that improves on Min-K%. Its insight, derived from the theory of
maximum-likelihood training via score matching, is that training samples tend to form **local maxima
(modes) of the model's likelihood along each input dimension (token)**. So instead of using the raw
token probability, Min-K%++ tests whether the observed next token is a *mode* of the conditional
next-token distribution by z-score-normalising its log-probability against the mean and standard
deviation of log-probs over the whole vocabulary. It is state-of-the-art among reference-free
methods on WikiMIA, beating Min-K% by roughly 6–12% AUROC across length settings.

## Detailed attack mechanism

For each token position with context `x_{<t}` and observed token `x_t`, the per-token statistic is a
vocabulary-calibrated z-score:

`Min-K%++_token(x_{<t}, x_t) = ( log p(x_t | x_{<t}) − μ_{x<t} ) / σ_{x<t}`

where, over the full conditional categorical distribution `p(· | x_{<t})` given by the model's output
logits:

- `μ_{x<t} = E_{z ∼ p(·|x_{<t})}[ log p(z | x_{<t}) ]` — the expected next-token log-prob over the
  vocabulary.
- `σ_{x<t} = sqrt( E_{z ∼ p(·|x_{<t})}[ (log p(z|x_{<t}) − μ_{x<t})² ] )` — the std of the next-token
  log-prob over the vocabulary.

Both are computed analytically from the softmax over the logits (in code:
`probs = softmax(logits); mu = Σ probs·log_probs; sigma = sqrt(Σ probs·log_probs² − mu²)`).

The sequence score selects the **k% of token positions with the minimum per-token z-score** and
averages them:

`Min-K%++(x) = (1/|min-k%|) · Σ_{(x_{<t}, x_t) ∈ min-k%} Min-K%++_token(x_{<t}, x_t)`

A **higher score ⇒ member** (the token is a mode). Interpretively, `μ` provides a per-position
calibration of the likelihood and `σ` acts as an adaptive (per-input) temperature, versus the
constant temperature implicit in raw probability. `k` is swept over {10, 20, …, 100} and shown to be
robust.

## Attack implementation

- **Code repository (verified):** https://github.com/zjysteven/mink-plus-plus (adapted from the
  official Min-K%/WikiMIA repo). MIMIR experiments use the fork https://github.com/zjysteven/mimir.
- **Benchmarks:** WikiMIA (`swj0419/WikiMIA`, plus the authors' paraphrased+perturbed split and a
  new "detect-while-generating" concatenated split) and **MIMIR** (Pile train vs. test, a much harder
  minimal-distribution-shift setting).
- **Models evaluated:** Mamba (1.4B/2.8B), Pythia (2.8B/6.9B/12B), GPT-NeoX-20B, LLaMA (13B/30B/65B),
  OPT-66B. **Baselines:** Loss, Ref, Lowercase, Zlib, Neighbor, Min-K%. **Metrics:** AUROC,
  TPR@5%FPR, FPR95.
- **Access assumption:** grey-box — it needs the **full output logits / vocabulary distribution** to
  compute `μ` and `σ`. The paper notes its one limitation: it cannot run on pure black-box APIs that
  return only top-k probabilities.

## Adaptation for LLM fine-tuning + data extraction

The Min-K%++ paper targets *pre-training* detection and does not run a dedicated fine-tuning MIA
experiment, so the following is the natural adaptation given the method's definition. The method is
model-agnostic and requires only output logits: to detect fine-tuning members, run candidate text
through the fine-tuned model, obtain per-position logits, compute `μ_{x<t}` and `σ_{x<t}` from the
vocabulary distribution, form the per-token z-score, average over the K% smallest, and threshold.
Because fine-tuning over multiple epochs sharply increases memorization, fine-tuned members should
more strongly form modes of the next-token distribution (large positive `log p(x_t|x_{<t}) − μ`),
making the z-score signal larger and detection easier than in the single-pass pre-training setting it
was benchmarked on.

For surfacing/extracting memorized fine-tuning data, rank candidate sequences by `Min-K%++(x)`; high-
scoring sequences are those whose tokens the fine-tuned model treats as modes (i.e., memorized).
These flagged passages can then be elicited by prompting the model with their prefixes and sampling
continuations — a high per-token z-score indicates the model will reproduce the memorized token at
each position, enabling reconstruction. The paper's own "detect-while-generating" (WikiMIA_concat)
setting demonstrates scoring during generation, which is the primitive such an extraction pipeline
would build on. The grey-box logit requirement is the practical constraint: it suits open-weight or
logit-exposing fine-tuned deployments but not opaque generation-only APIs.
