# `zlib` — zlib-Entropy Ratio Attack

**Primary source:** Nicholas Carlini, Florian Tramèr, Eric Wallace, et al. *Extracting Training Data
from Large Language Models.* USENIX Security 2021. arXiv:2012.07805.

## Brief overview

The zlib attack is a non-neural instance of the calibration idea: instead of a second language model
acting as the reference, it uses **zlib compression entropy** as a cheap, model-independent notion of
how "surprising" a text is. The membership score is the ratio of the target model's perplexity to the
text's zlib compression size. It is essentially free to compute (no extra model) and is especially
good at filtering out repeated substrings and trivially common text that otherwise receive
spuriously high LM likelihood.

## Detailed attack mechanism

1. Compute the **zlib entropy** of the candidate text = the number of bits when the sequence is
   compressed with the zlib library.
2. Compute the target model's perplexity (or log-perplexity) on the same text.
3. The membership metric is the **ratio of the (log-)perplexity to the zlib entropy**. Samples are
   ranked by this ratio; a high ratio (the model finds the text easy beyond what a generic compressor
   would predict) indicates likely memorization.

Because general-purpose compressors model repeated substrings extremely well, repetitive/boilerplate
text gets a large zlib-entropy reduction and is therefore down-ranked, removing exactly the false
positives that plague the raw-perplexity (`loss`) attack. The reported effect is large: 67% of
"Internet"-conditioned samples flagged by the zlib metric were confirmed memorized, versus a single-
digit percentage for raw perplexity.

## Attack implementation

No public code repository is stated in the paper; the only external dependency is the standard zlib
compression library (Gailly & Adler), which is available in essentially every language's standard
library (e.g., Python's `zlib.compress`). Target: GPT-2 XL, evaluated over 200,000 generated samples
across three generation strategies. In modern LLM-MIA toolkits the implementation is two lines:
`score = model_logppl(x) / len(zlib.compress(x.encode()))` (sign/orientation chosen so that members
score higher), then threshold or report AUC.

## Adaptation for LLM fine-tuning + data extraction

For a fine-tuned LLM, compute the fine-tuned model's per-token (log-)perplexity on a candidate
sequence and divide by that sequence's zlib entropy. The zlib term is model-independent, so it acts
as a cheap reference that flags sequences the fine-tuned model finds easy *beyond* what their raw
compressibility explains — i.e., likely fine-tuning-set members rather than generically simple text.
This is attractive in the fine-tuning setting because it needs no second model and no access to the
private training distribution.

In the extraction-then-membership pipeline, candidate generations from the fine-tuned model are
ranked by the perplexity/zlib ratio; high-ranked generations are inspected as candidate verbatim
fine-tuning data. zlib is particularly useful as a **zero-extra-model first-pass filter** to strip
repeated/boilerplate generations before applying more expensive reference-model or
neighbourhood-based checks. Carlini et al. explicitly flag fine-tuning as future work, so the
fine-tuning adaptation is the natural transfer of their pre-training-era zlib-ratio signal to a
fine-tuned model rather than an experiment performed in the original paper.
