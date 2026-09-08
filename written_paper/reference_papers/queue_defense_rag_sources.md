# Sources for the queue, FL defenses, and RAG implementation

Recorded 2026-09-07 (America/Toronto), as used for design refinement.
The user accepted the recommended design and consolidation into the existing
`written_paper/reference_papers` library. This log retains the candidate literature
review as well as adopted mechanisms; candidate inclusion does not claim implementation.
Linked primary records were verified online. PDFs are not embedded in this log.

| ID | Academic source | Design use and boundary |
| --- | --- | --- |
| abadi2016 | Martín Abadi et al. **Deep Learning with Differential Privacy.** ACM CCS 2016, pp. 308–318. [Primary record](https://arxiv.org/abs/1607.00133v2), [DOI](https://doi.org/10.1145/2976749.2978318). | Algorithm 1: individually clip example gradients, add Gaussian noise, account across training. Candidate for sequence-level private client optimization. Clipping a minibatch mean is insufficient. |
| mcmahan2018 | H. Brendan McMahan, Daniel Ramage, Kunal Talwar, Li Zhang. **Learning Differentially Private Recurrent Language Models.** ICLR 2018. [Primary record](https://arxiv.org/abs/1710.06963v3). | User-level private FedAvg via bounded updates and noise. Candidate for client-level aggregation baseline; evidence is recurrent language models and large populations, not a guarantee of utility for a tiny Transformer federation. Central aggregation privacy does not hide raw updates from its server. |
| andrew2021 | Galen Andrew, Om Thakkar, H. Brendan McMahan, Swaroop Ramaswamy. **Differentially Private Learning with Adaptive Clipping.** NeurIPS 2021. [Publisher record](https://proceedings.neurips.cc/paper_files/paper/2021/hash/91cff01af640a24e7f9f7a5ab407889f-Abstract.html). | Private quantile estimation adapts the clipping norm, with its own privacy cost. Candidate established adaptive baseline, not a novel attack-aware defense. |
| li2022 | Xuechen Li, Florian Tramèr, Percy Liang, Tatsunori Hashimoto. **Large Language Models Can Be Strong Differentially Private Learners.** ICLR 2022. [Primary record, revised 2022](https://arxiv.org/abs/2110.05679v6). | Full-model DP fine-tuning, including GPT-2 generation and private Adam updates; ghost clipping addresses memory. Supports LLM-specific feasibility, not guaranteed privacy/utility for this dataset. Pretraining exposures are outside a fine-tuning guarantee. |
| mironov2019 | Ilya Mironov, Kunal Talwar, Li Zhang. **Rényi Differential Privacy of the Sampled Gaussian Mechanism.** 2019. [Primary record](https://arxiv.org/abs/1908.10530). | Sampled-Gaussian accounting requires a matching sampling model. Do not claim Poisson amplification for an unchanged deterministic or shuffled schedule. Compose repeated accesses and declare adjacency and delta. |
| lewis2020 | Patrick Lewis et al. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.** NeurIPS 2020. [Primary record](https://arxiv.org/abs/2005.11401), [full text](https://arxiv.org/html/2005.11401v4). | Retrieval-conditioned learning/generation foundation. Original seq2seq model marginalizes retrieved passages; causal-LM context concatenation would be an adaptation. Ordinary RAG provides no DP guarantee. |
| huang2023 | **Privacy Implications of Retrieval-Based Language Models.** Huang et al., EMNLP 2023. [Publisher record](https://aclanthology.org/2023.emnlp-main.921/), [DOI](https://doi.org/10.18653/v1/2023.emnlp-main.921). | Retrieval datastores can leak; study covers kNN-LMs and empirical mitigation. Motivates separate training and datastore membership evaluation. Transfer to prompt-based RAG is a hypothesis to test. |
| anderson2024 | Maya Anderson, Guy Amit, Abigail Goldsteen. **Is My Data in Your Retrieval Database? Membership Inference Attacks Against Retrieval Augmented Generation.** 2024. [Primary record](https://arxiv.org/abs/2405.20446), [full text](https://arxiv.org/html/2405.20446v2). | Black-box answer and gray-box probability attacks target retrieval membership. Candidate datastore evaluation alongside existing model attacks. Prompt defenses show model-dependent behavior, not formal privacy. |
| koga2025 | Tatsuki Koga, Ruihan Wu, Zhiyuan Zhang, Kamalika Chaudhuri. **Privacy-Preserving Retrieval-Augmented Generation with Differential Privacy.** arXiv preprint, first 2024, revised November 2025. [v3 primary record](https://arxiv.org/abs/2412.04697v3), [v3 full text](https://arxiv.org/html/2412.04697v3). | DPVoteRAG and DPSparseVoteRAG privately aggregate token votes; sparse-vector decisions economize budget. Candidate formal retrieval defense. Training must be outside the private-corpus barrier, or separately protected/composed. The guarantee covers released answers, not internal raw logits. Earlier v2 was also inspected during discovery; use v3 author list and algorithm for implementation. |
| choi2025 | Choi et al. **Safeguarding Privacy of Retrieval Data against Membership Inference Attacks: Is This Query Too Close to Home?** Findings of EMNLP 2025. [Publisher record](https://aclanthology.org/2025.findings-emnlp.438/), [DOI](https://doi.org/10.18653/v1/2025.findings-emnlp.438). | Mirabel detects unusually close query/document similarity with a Gumbel-based threshold and hides suspect documents. Candidate compact adaptive retrieval defense. Empirical protection depends on retrieval embeddings and attacks; not DP or training-membership protection. |
| carlini2022 | Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, Florian Tramèr. **Membership Inference Attacks From First Principles.** IEEE Security and Privacy 2022. [Primary record](https://arxiv.org/abs/2112.03570). | Motivates attack evaluation at low false-positive rates. Small smoke batches cannot substantiate low-FPR privacy claims. This does not select LiRA as an additional implemented attack. |

## Research synthesis and unresolved evidence

The strongest fit for training-record membership is private local optimization
with per-record clipping and an accountant that persists across rounds. A
client-level private aggregate answers a different privacy question. The
existing shared trainer uses uniform averaging; the final implementation must
inspect AMIA/LOSS weighting independently before calibrating noise.

RAG phase, corpus ownership/disjointness, attacker access, and protected
membership must be explicit experimental variables. Training-MIA improvements
cannot establish privacy of a private retrieval datastore. Public retrieval,
private retrieval, and ordinary FL need separate controls. Utility should be
measured on generated answers and/or held-out likelihood with disjoint data.

Discovery used primary-source searches for FL/private language modeling,
DP fine-tuning/accounting, RAG/datastore MIA, and private RAG. Follow-up checked
algorithms, protected units, assumptions, costs, and the latest DP-RAG version.
The design-stage search stopped when each viable option had a verified primary
source and its main limitations were identified. Exact algorithm/API review
will follow the user's mechanism choices; no proposed combination is asserted
to inherit a guarantee without its required assumptions and composition.

## Added at accounting implementation

**bun2016** — Mark Bun and Thomas Steinke. *Concentrated Differential Privacy:
Simplifications, Extensions, and Lower Bounds* (2016).
[Verified primary record](https://arxiv.org/abs/1605.02065).
Gaussian sensitivity Δ and standard deviation s yield ρ = Δ²/(2s²);
adaptive composition adds ρ, and conversion gives ε = ρ +
2√(ρ log(1/δ)). We use this conservative accounting without sampling
amplification under fixed-cardinality replacement adjacency. Each record
gradient or client update is clipped to C, giving replacement sensitivity 2C
before public normalization. This permits the existing fixed-size shuffled
batches/client selection without pretending they are Poisson samples.

Implementation documentation consulted: [Flower differential privacy](https://flower.ai/docs/framework/how-to-use-differential-privacy.html),
[Opacus PrivacyEngine](https://opacus.ai/api/privacy_engine.html), and
[PyTorch autograd.grad](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html).
The Context7 CLI was unavailable due package-registry DNS failure.

**reimers2019** — Nils Reimers and Iryna Gurevych. *Sentence-BERT:
Sentence Embeddings using Siamese BERT-Networks*. EMNLP-IJCNLP 2019,
pp. 3982–3992. [Publisher record](https://aclanthology.org/D19-1410/),
[DOI](https://doi.org/10.18653/v1/D19-1410). Frozen sentence embeddings
and cosine retrieval; the default MiniLM encoder also matches the model
family used by Anderson et al. (2024). No privacy claim is attributed to
the embedding model.

## Final adoption map

Implemented: per-sequence private Adam gradients (`abadi2016`, `li2022`),
fixed-clipping client DP-FedAvg (`mcmahan2018`), conservative no-amplification
zCDP accounting (`bun2016`, with sampling caution from `mironov2019`),
frozen cosine retrieval (`reimers2019`) and causal-LM inference RAG adapted
from `lewis2020`, black-box datastore MIA (`anderson2024`), Mirabel detection
and hiding (`choi2025`), and separate/low-FPR membership reporting motivated
by `huang2023` and `carlini2022`. `andrew2021` and `koga2025` informed the
choice discussion but adaptive private clipping and DPVoteRAG were not selected.
The implementation report is `master_script/docs/queue_defense_rag_implementation.md`.

## Larger research configuration (September 8, 2026)

**rajpurkar2016** — Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, Percy Liang.
*SQuAD: 100,000+ Questions for Machine Comprehension*. EMNLP 2016, pp. 2383–2392.
[Publisher record](https://aclanthology.org/D16-1264/),
[official dataset and distribution](https://rajpurkar.github.io/SQuAD-explorer/).
The new `squad_research` profile uses training QA records; the RAG preparation
command samples contexts and answered questions from the validation distribution.
Public/private corpus assignment is a simulated access boundary on public data,
not a claim that SQuAD is sensitive or absent from model pretraining.

Implementation model source: [Qwen2.5-0.5B-Instruct official model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
verified 0.49B total parameters and the documented native chat-template format.
This size is selected for a single 20GB allocation with the existing full-model
DP optimizer, not because the user's approved ceiling is only 0.5B. The
[1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct),
[3B](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), and
[7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) cards were inspected as
alternatives; the last reports 7.61B total parameters. No larger model is
silently selected by the research config.
