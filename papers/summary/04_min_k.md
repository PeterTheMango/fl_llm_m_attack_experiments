# `min_k` — Min-K% Prob

**Primary source:** Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra
Blevins, Danqi Chen, Luke Zettlemoyer. *Detecting Pretraining Data from Large Language Models.* ICLR
2024. arXiv:2310.16789.

## Brief overview

Min-K% Prob is a reference-free pretraining-data detection method (a membership inference attack)
for LLMs. It rests on the hypothesis that a **non-member** (unseen) text is more likely to contain a
few outlier tokens with very low probability (high negative log-likelihood), whereas a **member**
(seen) text rarely contains such low-probability tokens. It needs no knowledge of the pretraining
corpus and no auxiliary reference model, departing from prior reference-based MIAs, and reported a
7.4% AUC improvement over prior methods on the authors' WikiMIA benchmark.

## Detailed attack mechanism

For a token sequence `x = x₁ … x_N`, the per-token statistic is the conditional log-likelihood
`log p(xᵢ | x₁ … x_{i−1})`. The procedure:

1. Compute `log p(xᵢ | x_{<i})` for every token.
2. Select the **k% of tokens with the minimum token probability**, forming the set `Min-K%(x)`.
3. Score = the average log-likelihood over that set:
   `MIN-K%-PROB(x) = (1/E) · Σ_{xᵢ ∈ Min-K%(x)} log p(xᵢ | x_{<i})`, where `E = |Min-K%(x)|`.
4. Threshold against `ε`: a **high average log-likelihood over the worst tokens ⇒ member**
   (equivalently, the negative-sum form exceeds `ε` ⇒ non-member, per the paper's Algorithm 1).

The key hyperparameter `k` was swept over {10, 20, 30, 40, 50}; `k = 20` worked best and was used
throughout. Because AUC is the reported metric, the absolute threshold `ε` need not be fixed.
Intuitively, averaging only over the lowest-probability tokens concentrates the signal where members
and non-members differ most, rather than diluting it across the whole (often easy) sequence.

## Attack implementation

- **Code repository (verified from paper/repo):** https://github.com/swj0419/detect-pretrain-code
  (project page: https://swj0419.github.io/detect-pretrain.github.io/).
- **Benchmark introduced:** WikiMIA — Wikipedia event pages, members = pages created before 2017,
  non-members = events created after the model cutoff (post-2023); 394 + 394 examples; *original* and
  *paraphrase* settings; length buckets 32/64/128/256. Also BookMIA (Books3/copyright). Hosted on
  HuggingFace as `swj0419/WikiMIA` and `swj0419/BookMIA`.
- **Models evaluated:** LLaMA (1/2), GPT-Neo, GPT-NeoX-20B, OPT, Pythia, and OpenAI
  text-davinci-001/002/003. The released code supports HuggingFace causal LMs and OpenAI logit-
  returning models. Metrics: AUC (ROC) and TPR@5%FPR. Demonstrated applications: copyrighted-book
  detection (Books3 in GPT-3, AUC 0.88), benchmark-contamination detection, and machine-unlearning
  auditing.

## Adaptation for LLM fine-tuning + data extraction

The same scoring transfers directly to a fine-tuned model: feed candidate fine-tuning text into the
fine-tuned LLM, compute per-token `log p` under that model, average the K% lowest, and threshold.
Members (fine-tuning data) score higher because the fine-tuned model assigns them few low-probability
tokens. The paper itself situates the method against prior *fine-tuning* data-detection work,
noting that fine-tuning runs many epochs over a smaller set (a stronger memorization signal) than
single-pass pre-training, which makes the signal generally easier to exploit in fine-tuning regimes.

The closest empirical analogue in the paper (Section 6) continually fine-tunes **LLaMA-7B for one
epoch** on a corpus deliberately contaminated with held-out benchmark examples (0.1% of 27M tokens,
LR 1e-4), then uses Min-K% Prob to detect which examples were inserted (AUC 0.91; +12.2% TPR@5%FPR
over the best baseline). The appendix shows detectability grows with learning rate and with the
number of times an example occurs — both directly relevant to fine-tuning. For **data extraction**,
the unlearning audit (Section 7) shows the score can surface memorized content: texts are ranked by
the difference in Min-K% Prob between the original and unlearned models, and high-scoring (memorized)
passages are then elicited by prompting the model to regenerate them. The same procedure on a
fine-tuned model flags high-scoring fine-tuning passages that can be reconstructed by prefix-prompted
generation.
