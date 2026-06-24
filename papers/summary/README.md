# Membership Inference Attacks — Background Research Summaries

This folder contains background-research summaries for ten membership inference attacks (MIAs),
each grounded in its primary source paper (retrieved from arXiv) and, where available, its public
code repository. Each summary contains four sections: **Brief overview**, **Detailed attack
mechanism**, **Attack implementation** (with code repositories), and **Adaptation for LLM
fine-tuning + data extraction**.

## Scope note / grounding

The reference framing for this task is the main project paper, *Active Membership Inference Attack
under Local Differential Privacy in Federated Learning* (IEEE, 2023), which studies an **active**
attacker in **federated vision** models protected by local differential privacy (LDP). The ten
attacks summarised here are a complementary family: **passive, score-based MIAs against (mostly
fine-tuned or pre-trained) large language models**. The related literature already in the `papers`
folder (FL/LLM surveys and MIA-defense papers) references the broad MIA threat but does **not**
contain the primary papers for these ten methods; those primary papers were therefore retrieved
from arXiv to ground the summaries (per the user's instruction). Every claim below is attributed to
a specific source.

## The ten attacks at a glance

| # | Attack | Key idea | Primary source | Code repo |
|---|--------|----------|----------------|-----------|
| 1 | `loss` | Threshold the per-example loss/perplexity; members have lower loss | Yeom et al., CSF 2018 (arXiv 1709.01604) | none stated |
| 2 | `reference` | Calibrate target-model likelihood against a reference model (ratio/difference) | Carlini et al., USENIX Sec. 2021 (arXiv 2012.07805); LiRA-for-LMs | none stated in source |
| 3 | `zlib` | Ratio of model perplexity to zlib compression entropy | Carlini et al., USENIX Sec. 2021 (arXiv 2012.07805) | none stated |
| 4 | `min_k` | Average log-prob over the K% lowest-probability tokens | Shi et al., ICLR 2024 (arXiv 2310.16789) | github.com/swj0419/detect-pretrain-code |
| 5 | `neighborhood` | Compare target loss to mean loss of MLM-generated neighbours (reference-free) | Mattern et al., ACL Findings 2023 (arXiv 2305.18462) | github.com/mireshghallah/neighborhood-curvature-mia |
| 6 | `min_k_plus_plus` | Vocabulary-calibrated z-score of token log-prob, averaged over K% smallest | Zhang et al., ICLR 2025 (arXiv 2404.02936) | github.com/zjysteven/mink-plus-plus |
| 7 | `wbc` | Sliding-window sign votes over per-token reference-minus-target loss differences | Chen et al., 2026 (arXiv 2601.02751) | github.com/Stry233/WBC |
| 8 | `recall` | Relative change in conditional log-likelihood when prepending a non-member prefix | Xie et al., EMNLP 2024 (arXiv 2406.15968) | github.com/ruoyuxie/recall |
| 9 | `samia` | Sample continuations from a prefix; ROUGE overlap with the true suffix (black-box) | Kaneko et al., 2024 (arXiv 2404.11262) | github.com/nlp-titech/samia |
| 10 | `spv_mia` | Self-prompt reference model + probabilistic-variation (memorization) signal | Fu et al., NeurIPS 2024 (arXiv 2311.06062) | github.com/tsinghua-fib-lab/NeurIPS2024_SPV-MIA |

## Source-identification caveat

`wbc` is not a household-name attack like the others. It was identified (with high confidence) as
the **Window-Based Comparison (WBC)** attack by reading both the repository README
(github.com/Stry233/WBC, whose config key is literally `wbc`) and the arXiv HTML of the paper
(arXiv 2601.02751). The exact 10-key evaluation harness that bundles all ten names together could
not be located as a public repository; it is most likely an aggregated/private fine-tuning-MIA
benchmark (the first seven names match the MIMIR-style attack set, plus SaMIA, SPV-MIA, and WBC).
If `wbc` in your codebase refers to something else, flag it and the summary can be revised.

## Files

- `01_loss.md`
- `02_reference.md`
- `03_zlib.md`
- `04_min_k.md`
- `05_neighborhood.md`
- `06_min_k_plus_plus.md`
- `07_wbc.md`
- `08_recall.md`
- `09_samia.md`
- `10_spv_mia.md`
