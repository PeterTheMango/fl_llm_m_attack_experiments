# `reference` — Reference- / Comparison-Model Calibrated Attack

**Primary sources:** Nicholas Carlini, Florian Tramèr, Eric Wallace, et al. *Extracting Training Data
from Large Language Models.* USENIX Security 2021. arXiv:2012.07805. (Comparison/reference-model
membership signal.) The reference-calibration framing for language models is also developed by
Mireshghallah et al. (Likelihood Ratio Attacks / LiRA adapted to MLMs and CLMs) and by Carlini et
al., *Membership Inference Attacks From First Principles*, IEEE S&P 2022 (arXiv 2112.03570), both
cited as the reference-based baselines in the SPV-MIA paper (arXiv 2311.06062) in this folder.

## Brief overview

The reference attack calibrates the target model's likelihood against a **second "reference" model**,
flagging samples whose likelihood is *unexpectedly high* on the target model relative to the
reference. This corrects the central failure of the raw-loss attack: intrinsically high-likelihood
text (common phrases, boilerplate, repeated substrings) produces false positives. Because a good
reference model *also* assigns high likelihood to such generic content, taking the ratio/difference
of the two models' scores cancels the intrinsic difficulty and isolates text that the target model
"knows" specifically — i.e., likely training members.

## Detailed attack mechanism

Base perplexity of a sequence is `P = exp(−(1/n) Σ log f_θ(xᵢ | x_{<i}))`. The reference attack
scores by comparing this against a reference model `φ`:

- **Ratio / difference of (log-)perplexities.** In Carlini et al. 2021, the "Small" and "Medium"
  metrics rank a sample by the ratio of the large target model's log-perplexity to that of a smaller
  reference GPT-2 (117M / 345M). A sample is ranked most likely to be a member when the target
  model's likelihood is high but the reference model's is comparatively low — an abnormally high
  likelihood ratio between the two models.
- **Difficulty-calibration view (Watson et al.; LiRA).** Generalising, the calibrated membership
  metric is `Δm(x) = m_θ(x) − E_{φ ← T(D_ref)}[m_φ(x)]`, the discrepancy between the metric measured
  on the target model `θ` and the expected metric on reference models `φ` trained on a reference
  dataset `D_ref` drawn from (ideally) the same distribution as the training set. The decision is
  `A(x) = 1[Δp_θ(x) ≥ τ]`, i.e. `1[p_θ(x) − p_φ(x) ≥ τ]`.

The reference model can be a smaller model (less memorization capacity) or a model trained on
disjoint data (unlikely to memorize the same examples). The attack's strength depends heavily on how
well the reference distribution matches the target's training distribution — a dependence that the
later SPV-MIA work shows degrades exponentially as the reference dataset diverges, and which it
addresses with a self-prompted reference model.

## Attack implementation

No public code repository is stated in the Carlini et al. 2021 paper (it deliberately attacks the
already-public GPT-2 to minimize harm). Target: GPT-2 XL (1.5B); reference models: GPT-2 Small
(117M) and Medium (345M). In modern LLM-MIA toolkits the reference attack is implemented as: load a
second model, compute per-token log-likelihood on the candidate under both, and use the difference
or ratio as the membership score. The LiRA variants (LiRA-Base using a pre-trained reference,
LiRA-Candidate using a domain-matched reference) are the standard reference baselines benchmarked by
SPV-MIA.

## Adaptation for LLM fine-tuning + data extraction

For a fine-tuned LLM the natural reference is the **un-fine-tuned base / pre-trained model** (or a
model fine-tuned on a disjoint dataset of the same task). The target score is the fine-tuned model's
per-token perplexity on a candidate sequence; the reference is the base model's perplexity on the
same sequence. The ratio cancels the intrinsic likelihood of generic text, so a high ratio indicates
text the fine-tuned model knows *specifically* — i.e., from the fine-tuning set rather than text any
LM finds easy. This is one of the strongest practical signals for fine-tuning MIA when a good
reference is available, because fine-tuning concentrates new memorization in exactly the directions
the base model does not share.

For data extraction, the reference metric is the core of the extraction-then-membership pipeline:
Carlini et al. state that "the problem of training-data extraction reduces to one of membership
inference" — generate many candidate samples, then rank them by a membership metric. The reference
(comparison) metric dramatically improves extraction precision over raw perplexity by suppressing
false positives from common boilerplate, so applied to a fine-tuned model it surfaces memorized
fine-tuning records (PII, proprietary text) while filtering out generic content. The principal
limitation — needing a reference dataset close to the private fine-tuning distribution — is what
SPV-MIA later removes.
