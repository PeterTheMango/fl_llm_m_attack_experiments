# `neighborhood` — Neighbourhood Comparison Attack

**Primary source:** Justus Mattern, Fatemehsadat Mireshghallah, Zhijing Jin, Bernhard Schölkopf,
Mrinmaya Sachan, Taylor Berg-Kirkpatrick. *Membership Inference Attacks against Language Models via
Neighbourhood Comparison.* Findings of the ACL 2023. arXiv:2305.18462.

## Brief overview

The neighbourhood attack is a **reference-model-free** MIA. For a target text `x`, the attacker
generates a set of semantically/syntactically near-identical "neighbour" texts via masked-LM word
substitutions, then compares `x`'s loss under the target model to the *average* loss of its
neighbours. Because the neighbours are practically interchangeable with `x` under any plausible text
distribution, a target loss substantially below its neighbours' average can only arise from
overfitting — i.e., membership. This replaces the reference model used in Likelihood Ratio Attacks
(LiRA) with synthetic neighbours as the difficulty-calibration signal, removing any need for
in-domain reference data.

## Detailed attack mechanism

Grey-box access (loss/confidence scores only). Decision rule:

`A(x) = 1[ ( L(f_θ, x) − (1/n) Σ_{i=1}^{n} L(f_θ, x̃_i) ) < γ ]`

where `{x̃₁ … x̃_n}` are `n` neighbours and `γ` is a threshold. Membership is predicted when the
target's loss minus the mean neighbour loss falls below `γ` (the target is much lower than its
neighbours).

**Neighbour generation** (adapting Zhou et al. 2019 BERT lexical substitution):

- For text `x = (w¹ … w^L)`, use a masked LM (BERT) to score candidate replacement tokens at
  position `i`. The suitability score normalises out the original token's own probability:
  `p_swap(ŵ^i, w̃^i) = p_θ(w̃ = w^i | x) / (1 − p_θ(ŵ = w^i | x))`.
- Crucially, the original token is **not** masked out; instead strong dropout (`p = 0.7`) is applied
  to the input embedding at position `i`, so the model still respects the original word's meaning
  when proposing replacements (preventing semantic flips such as "great" → "bad").
- Over all `m`-word swap combinations, compute the joint suitability and return the `n` highest-
  scoring neighbours.
- Best configuration from ablations: `n = 100` neighbours, `m = 1` word replacement (single
  replacement clearly beat 2–3; more neighbours helped monotonically). Neighbours come from an
  off-the-shelf BERT (110M) with no domain adaptation.

## Attack implementation

- **Code repository (paper footnote):** https://github.com/mireshghallah/neighborhood-curvature-mia
  (the authors note in their ethics statement that code is provided on request rather than freely
  publicized).
- **Target model:** GPT-2 base (117M), fine-tuned. **Neighbour generator:** BERT 110M
  (HuggingFace transformers + PyTorch).
- **Datasets** (each split into disjoint train/non-train halves for member/non-member labels):
  AG News summaries, Sentiment140 tweets, Wikitext-103 excerpts — i.e., a **fine-tuning / multi-epoch
  MIA setting**, not WikiMIA pretraining.
- **Baselines:** LOSS attack; LiRA with Base (pretrained GPT-2), Candidate (domain-matched), and
  Oracle reference models. **Metrics:** TPR at low FPR (1%, 0.1%, 0.01%) and AUC. Results
  (TPR@1%FPR): 8.29% (News) / 7.35% (Twitter) / 2.32% (Wiki); AUC 0.79 / 0.77 / 0.62 — beats LOSS and
  realistic LiRAs and is competitive with the Oracle LiRA. DP-SGD (`ε = 5, 10`) defends effectively.

## Adaptation for LLM fine-tuning + data extraction

This attack was **designed in the fine-tuning setting**: the target is a model fine-tuned on a
corpus, and the attack detects which samples were in the fine-tuning set. It applies directly to a
fine-tuned LLM — feed a candidate sample, generate BERT neighbours, and threshold the loss-vs-
neighbour-mean difference. Its decisive practical advantage is that **no reference model and no
knowledge of the fine-tuning distribution are required**, which matters in privacy-sensitive domains
(e.g., medical or proprietary text) where in-domain reference data is unavailable — exactly the
regime where reference-based attacks collapse.

For data extraction, the paper frames MIA as a core component of extraction attacks (Carlini et al.
2021): once neighbourhood comparison flags a candidate string as a member, it confirms the string was
memorized, helping surface and verify memorized fine-tuning data. The authors note the close
relationship to DetectGPT (which uses an analogous perturbation-likelihood/curvature idea) and
explicitly list improving extraction attacks as future work. Notably, SPV-MIA (arXiv 2311.06062)
later "rediscovers" and reformulates this neighbourhood signal as a probabilistic-variation /
local-maximum (memorization) test.
