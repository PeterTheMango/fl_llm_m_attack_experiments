# `wbc` — Window-Based Comparison (WBC) Attack

**Primary source:** Yuetian Chen, Yuntao Du, Kaiyuan Zhang, Ashish Kundu, Charles Fleming, Bruno
Ribeiro, Ninghui Li. *Window-based Membership Inference Attacks Against Fine-tuned Large Language
Models.* arXiv:2601.02751 (Purdue University / Cisco). Code: https://github.com/Stry233/WBC.

**Source-identification note.** Unlike the other nine entries, `wbc` is not a widely known attack
acronym. It was identified by reading both the repository README
(github.com/Stry233/WBC, whose configuration key is literally `wbc`) and the paper's arXiv HTML.
The identification is high-confidence, but if `wbc` in your evaluation harness denotes a different
method, please flag it so this summary can be revised. (Expansions such as "white-box calibration"
were considered and ruled out.)

## Brief overview

WBC is a reference-based, score-based **black-box** MIA designed specifically for **fine-tuned LLMs**.
It rejects the standard "global average loss" paradigm, arguing that the memorization signal from
fine-tuning is not spread evenly across a sequence but appears as **sparse, localized, extremal
events** that global averaging dilutes. WBC slides windows of multiple sizes over the per-token
loss-difference sequence (reference minus target), lets each window cast a binary membership vote,
and ensembles the votes across geometrically spaced window sizes. The paper reports an average AUC of
0.839 vs. 0.754 for the best baseline, and 2–3× higher TPR at low FPR, across 11 datasets against 13
baselines.

## Detailed attack mechanism

- **Per-token loss difference:** `Δ_j(x) = ℓ_j^R − ℓ_j^T`, the reference model's per-token NLL minus
  the target model's per-token NLL at position `j`.
- **Generative model of the signal:**
  `Δ_j(x) = 1[x ∈ D_train]·δ_j(x) + ξ_j + ε_j`, where `δ_j` is the (sparse) membership signal, `ξ_j`
  is heavy-tailed rare-domain-token noise that dominates the right tail, and `ε_j` is baseline noise.
- **Empirical structure motivating the design:** the signal is sparse and extremal (excess kurtosis
  > 18; ~1.77% of member tokens beyond 3σ), spatially scattered (Poisson-random, not clustered), and
  the strongest membership signal sits in the **left tail** — positions where the target model has
  *higher* loss than the reference. The global average `Δ̄` is shown to be masked because rare-token
  variance can be effectively infinite, so a single outlier dominates it.
- **Windowed sum:** `S_i(w) = Σ_{j=i}^{i+w−1} Δ_j`, computed for every start position `i` and window
  size `w`.
- **Sign-based aggregation:** each window contributes a *binary* vote (the sign of its comparison);
  the membership score is the fraction of windows favouring membership. This is justified via Pitman
  asymptotic relative efficiency — the sign test beats the mean test under long-tailed contamination.
- **Geometric ensemble** over window sizes, default `[2, 3, 4, 6, 9, 13, 18, 25, 32, 40]`, requiring
  no per-dataset tuning.

## Attack implementation

- **Code repository:** https://github.com/Stry233/WBC (MIT licence). Structure includes
  `attacks/wbc.py`, `run.py`, `dataset/prep.py`, `trainer/get_target.py`, and
  `configs/config_all.yaml`. Output: AUC plus TPR@FPR (0.1 / 0.01 / 0.001) with bootstrap confidence
  intervals.
- **Reference model default:** `EleutherAI/pythia-2.8b`. The empirical analysis fine-tunes Pythia-2.8B
  on the Khan Academy subset of `HuggingFaceTB/Cosmopedia` (10k member / 10k non-member).
- **Evaluation:** 11 datasets, 13 baselines, multiple model scales and architectures. Defenses tested:
  differential privacy, LoRA, selective data obfuscation, and temperature scaling.

## Adaptation for LLM fine-tuning + data extraction

WBC is purpose-built for the fine-tuning threat model: a score-based black-box attacker queries the
fine-tuned target plus a pre-trained reference and obtains per-token loss sequences (the paper argues
this is realistic given open-weight fine-tuning and APIs that expose log-probabilities, e.g. vLLM's
`prompt_logprobs`). It directly quantifies fine-tuning training-data leakage and remains robust even
when the reference model is misaligned with the target — a key practical advantage over classic
reference attacks that collapse under reference mismatch.

For data extraction, WBC's central contribution is **localisation**: by identifying *where* in a
sequence the fine-tuned model is unusually confident relative to the reference, it pinpoints which
spans were memorized during fine-tuning, exposing privacy leakage in proprietary fine-tuning data.
The authors conjecture the localized-signal insight extends beyond text LLMs to pre-trained models
and to vision/diffusion models. Because the attack already operates on per-token loss differences,
its flagged high-signal spans are the natural seeds for a prefix-prompted regeneration/extraction
step.
