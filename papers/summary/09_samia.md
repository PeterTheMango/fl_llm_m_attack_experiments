# `samia` — SaMIA (Sampling-based Pseudo-Likelihood)

**Primary source:** Masahiro Kaneko, Youmi Ma, Yuki Wata, Naoaki Okazaki. *Sampling-based
Pseudo-Likelihood for Membership Inference Attacks.* arXiv:2404.11262 [cs.CL], 2024.

## Brief overview

SaMIA is a **likelihood-independent, fully black-box** MIA. Because proprietary LLMs (ChatGPT,
Gemini, Claude) expose no token likelihoods, SaMIA replaces the likelihood signal with a "Sampling-
based Pseudo-Likelihood" (SPL): it prompts the model with a prefix of the target text, samples
multiple continuations, and measures n-gram (ROUGE-N) overlap between those continuations and the
held-out remainder of the target text. High overlap implies membership. It performs on par with
likelihood-based methods despite using only generated text.

## Detailed attack mechanism

For a target text `x = (w₁ … w_T)`:

1. Split into a prefix `x_prefix = (w₁ … w_{⌊T/2⌋})` and a reference suffix
   `x_ref = (w_{⌊T/2⌋+1} … w_T)`.
2. Generate `m` candidate continuations `x_cand^j` (`j = 1…m`) by sampling from `f_θ(x_prefix)`. The
   paper uses `m = 10`, with temperature = 1.0, top_k = 50, top_p = 1.0, max_length = 1024.
3. Score each candidate with **ROUGE-N recall**:
   `ROUGE-N(x_cand, x_ref) = (Σ_{gram_n ∈ x_ref} Count_match(gram_n)) / (Σ_{gram_n ∈ x_ref} Count(gram_n))`.
   Default `N = 1` (unigram, best vs. ROUGE-2; recall beats precision, which is near-random).
4. **Decision:** `A(x) = 1[ (1/m) Σ_j ROUGE-N(x_cand^j, x_ref) > τ ]`. Members yield higher average
   overlap.

A **SaMIA×zlib** variant weights each candidate by its zlib-compressed bit length to penalise
repetitive (low-information) generations:
`1[ (1/m) Σ_j ROUGE-N(x_cand^j, x_ref) · zlib(x_cand^j) > τ ]`, which consistently improves results.

Theoretical justification: by the law of large numbers, the sampling frequency of a sequence
approximates `P(W = x)`, so overlap statistics over many samples reflect the model's output
distribution — a *pseudo-likelihood* obtainable without any logit access.

## Attack implementation

- **Code repository:** https://github.com/nlp-titech/samia (`src/sampling.py` generates candidates;
  `src/eval_samia.py` computes ROUGE/AUC, with `--zlib` and `--save` flags).
- **Models:** GPT-J-6B, OPT-6.7B, Pythia-6.9B, LLaMA-2-7B.
- **Benchmark:** WikiMIA (Shi et al. 2023) — Wikipedia event pages, pre-2017 = member, post-2023 =
  non-member; length groups 32/64/128/256. **Metrics:** AUC and TPR@10%FPR. SaMIA×zlib reaches SOTA
  on OPT and LLaMA-2; SaMIA wins outright at length 256 (AUC up to 0.80).

## Adaptation for LLM fine-tuning + data extraction

SaMIA's only requirement is **black-box text generation**, so it transfers directly to fine-tuned and
proprietary LLMs where logits are hidden — arguably the most deployment-realistic of the ten attacks
for closed fine-tuned APIs. Performance grows with target-text length and with prefix length up to
roughly 50% of the sequence, so longer fine-tuning records are easier to detect.

The membership-to-extraction link is unusually direct: because the mechanism literally prompts the
model to regenerate the suffix and rewards reproduction of the true continuation, a high SaMIA score
is itself evidence of **verbatim memorization** — the sampled candidates are reconstructions of the
held-out portion of the target. Applied to a fine-tuned model, prompting with prefixes of suspected
fine-tuning records and inspecting the high-overlap generations surfaces memorized fine-tuning
content. The authors note (as an ethical safeguard) that the tool outputs only a membership score,
not the generated text, to limit misuse — but the generation step is exactly what an extraction
pipeline would retain.
