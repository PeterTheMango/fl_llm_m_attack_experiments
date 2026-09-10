# Attack implementation theory audit

2026-09-08 · Source commit `94d4016` · Historical pre-change audit

Implementation was subsequently approved. Current dispositions and validation limits are in the [implementation record](../../master_script/docs/theory_corrections.md). Code references below identify the audited baseline, not current line numbers.

The implementations are not yet demonstrably faithful to their papers. The most consequential defects are AMIA's replacement of the observed client update with an attacker-computed gradient, SPV-MIA's bypassed self-prompt calibration, and WBC's incorrect aggregation. Correct scalar formulas in other attacks do not establish the validity of their surrounding experiments.

This report follows the supplied audit prompt directly; the improve skill was set aside. Only audit documents and reading aids were created. No implementation, notebook, test, configuration, or stored result was changed. No project code, model training/inference, tests, installation, or remote job was executed. Runtime verification belongs on the remote server after approval. Findings establish static differences, not measured effect sizes.

## Inventory

There are **12 attack families**: 11 in the registry and a datastore-membership attack in the optional RAG pipeline. Mirabel is a supporting defense. The review covers the registered modules, their adaptation notebooks, nine score recreation notebooks, legacy AMIA experiment/evaluation variants, and shared execution/evaluation boundaries. Notebook cell numbers are one-based and count markdown; `notebook-source/` contains searchable extracts.

| Attack | Runtime under `master_script/core/` | Other implementations | Source | Verdict |
|---|---|---|---|---|
| AMIA | [attacks/amia.py](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/amia.py>) | AMIA adaptation; legacy `recreations/ami/` | Nguyen 2023 | T01–T03, T13–T15 |
| LOSS | [attacks/loss.py](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/loss.py>) | LOSS adaptation | Yeom 2018 | T04–T05 |
| Reference | [attacks/reference.py](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/reference.py>) | adaptation + recreation | Carlini 2021 | T06 |
| Zlib | [attacks/zlib.py](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/zlib.py>) | adaptation + recreation | Carlini 2021 | Scalar matches §6.1; D02 |
| Min-K% | [attacks/min_k.py](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/min_k.py>) | adaptation + recreation | Shi 2024 | Scalar matches; T12/D08 |
| Min-K%++ | [attacks/min_k_plus_plus.py](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/min_k_plus_plus.py>) | adaptation + recreation | Zhang 2025 | Scalar matches; T12/D08 |
| Neighborhood | [attacks/neighborhood.py](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/neighborhood.py>) | adaptation + recreation | Mattern 2023 | Loss difference matches; T10 |
| ReCaLL | [attacks/recall.py](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/recall.py>) | adaptation + recreation | Xie 2024 | Ratio matches; D01 |
| SaMIA | [attacks/samia.py](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/samia.py>) | adaptation + recreation | Kaneko 2024 | Recall/weighting match; T11/D06 |
| SPV-MIA | [attacks/spv_mia.py](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/spv_mia.py>) | adaptation + recreation | Fu 2024 | T07–T09 |
| WBC | [attacks/wbc.py](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/wbc.py>) | adaptation + recreation | Chen 2026 v2 | T16, B01, D05 |
| Datastore MIA | `rag.py:evaluate_pipeline` | optional pipeline | Anderson 2024 | Primitive matches; T17 |

The nine score notebook stems are the nine rows from Reference through WBC. They live in `code_experiments/adaptations/<stem>_adaptations.ipynb` and `code_experiments/recreations/<stem>_recreation.ipynb`. AMIA and LOSS use `AMIA_adaptation.ipynb` and `LOSS_adaptation.ipynb`. Legacy AMIA contains CIFAR, CelebA and ImageNet, each with LDP/non-LDP training/evaluation scripts: twelve files. Preprocessing/image encoders support these, rather than defining other attacks.

The [initial inventory](initial-inventory.md) preserves all dataclass field lists and the original WBC pause. Its incomplete-status wording is historical. Entry points are `perform_experiments.py`, registry, `SPEC`, and runner; dependencies include config/yaml_config, datasets, federation, scoring, metrics and pipeline/RAG. The RAG configuration adds document studies, retrieval/context budgets, embedding model, prompt format and evaluation count.

Authorities: attack PDFs in `papers/summary/papers/`, the AMIA PDF directly under `papers/`, and corresponding `written_paper/reference_papers/` copies. No literal `@reference paper` directory exists. The RAG reference folder contains primary-paper links in `queue_defense_rag_sources.md`; those full texts were consulted. Broad FL surveys, related-literature PDFs and Nguyen 2025 vision have no separate registered implementations. The phrase “LiRA-style” in Reference does not establish a LiRA implementation; remove that unsupported attribution unless a distinct LiRA claim is intended.

## Theory baseline

| Source | Definition, access, assumptions and procedure | Evaluation |
|---|---|---|
| Nguyen 2023 §§2.2, 3.2, 4; Figs. 1–4 | Malicious server sends a chosen-neuron model and infers target inclusion in a private client batch from its returned loss gradient. Distribution access does not reveal the private batch. Train target/non-target discrimination. LDP uses independent privatizations and robust target training; Eq. (11) supplies confidence bounds. Eq. (3) Adv is balanced accuracy. | Frozen ResNet features, CIFAR/CelebA/ImageNet, non-LDP and BitRand/OME batch experiments, thousands of trials. The IEEE-prefixed filename contains an AISTATS 2023 paper. |
| Yeom 2018 §3 Exp. 1, Def. 4/Eq. (2), §§3.2–3.3 | Member sampled from S, fresh record from the underlying distribution; the latter is not explicitly D minus S. Adversary 1 randomizes using bounded loss; Adversary 2 uses a Gaussian-error density decision. Adv=TPR−FPR. | §6 regression, repeated train/test splits. An arbitrary nonmember-loss quantile is not these adversaries. |
| Carlini 2021 §§5.2, 6.1 | Compare generated candidate likelihood with reference LM or compression. §6.1 explicitly uses ratios of log perplexities for reference models, and log perplexity/compressed bits for zlib; smaller is more promising. | Three 200k-generation corpora, rank/sample/deduplicate and verify extraction. §6.1 resolves the looser perplexity wording in §5.2 for this audit. |
| Shi 2024 §§2.1, 3 Eq. (1), 4 | Token-probability access; average lowest k% token log probabilities, high=member. No pretraining-distribution access. | WikiMIA original/paraphrased lengths, AUROC and TPR at 5% FPR; k=20 selected on held-out data. §6 also studies fine-tuning contamination. Tiny-sequence rounding is unspecified. |
| Zhang 2025 §§3, 4.2 Eqs. (3)–(4) | Full vocabulary logits; probability-weighted vocabulary mean/std of log probability, normalize observed token, average lowest k%. | §5 WikiMIA/MIMIR, distribution-shift controls, AUROC and low-FPR analysis. Zero-variance policy is not explicitly prescribed. |
| Mattern 2023 §2.1 Eq. (3), §2.2 Alg. 1 | Target/neighbor loss comparison. Dropout on original MLM embeddings, rank replacement suitability p(replacement)/(1−p(original)), change distinct positions. Negating score and comparator together is valid. | §3 GPT-2, AG News/Twitter/WikiText, disjoint splits, n=100, m=1, BERT-base-uncased, dropout .7, AUROC/low-FPR TPR. |
| Xie 2024 §3 Eqs. (1)–(2) | Fixed known-nonmember prefix P; LL(x given P)/LL(x) for identical x, high=member. Synthetic prefixes permitted. | WikiMIA/MIMIR, AUROC and TPR at 1% FPR. One shot is valid; §5.2 ensemble is optional. |
| Kaneko 2024 §2.2 Eqs. (5), (7), Alg. 1 | Generation-only access; word midpoint split; m suffix samples; average ROUGE-N recall. Optional per-generation compressed-bit weighting. | §3 m=10, ROUGE-1, temperature 1/top-k 50/top-p 1, total max length 1024; WikiMIA, AUROC/TPR at 10% FPR. |
| Fu 2024 §§3.2, 4.2–4.3 | Query generation/likelihood and fine-tune fresh base on self-prompt target generations. Eq. (1) joint sequence probability. Eqs. (8)–(10) symmetric perturbation pairs around original. Eq. (5) explicitly uses target variation minus reference variation ≥ threshold. Appendix A.3 reconstructs semantic/opposite perturbations. | §5 several real LMs, WikiText-103/AG News/XSum; distinct target/reference schedules, LoRA, early stopping, AUROC/TPR at 1% FPR. Local-max intuition and printed score sign require clarification, not choosing by test AUC. |
| Chen 2026 §§4.1.3–4.1.4, Eqs. (10), (12), (13), Alg. 1 | Aligned reference-minus-target token NLLs. Per size: fraction of strictly positive window sums. Uniform mean of size fractions. | Fine-tuned LMs/reference models, member/nonmember corpora, AUROC and low-FPR/bootstraps. Eq. (12) conflicts with Appendix C.1.5's schedule: B01. |

Anderson §§2.3–3 targets retrieval-database membership using a candidate-containing question and generated Yes/No only. §3.1 assigns unrecognized answers to nonmember; evaluates 2,000 members/2,000 nonmembers using specified models/corpora/retrieval settings. [Full text](https://arxiv.org/html/2405.20446v2). Mirabel's supporting implementation follows Choi Alg. 1: remove one maximum when estimating cosine mean/std, calculate the Gumbel cutoff, hide the suspect document and refill retrieval. Small-corpus statistical calibration remains empirical. [Choi paper](https://aclanthology.org/2025.findings-emnlp.438.pdf).

## Findings: severity and effort

Critical invalidates the advertised mechanism; High changes scores, labels, assumptions or the estimated experiment; Medium weakens validity/reproducibility; Low is removable complexity. Effort S=localized, M=interacting paths, L=integration/scientific revalidation. **All T/D corrections can change results, validity or failure behavior and are not simplifications.** O proposals must preserve behavior and be verified independently. No corrections were applied.

## A. Theoretical misalignment

### T01 · Critical · AMIA does not observe the client loss update

[amia.py:374](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/amia.py:374>) trains a probe after FL. `:413` embeds raw batches locally and differentiates `relu(probe(embeddings)).sum()`. The adaptation's `probe_gradient_score` is the same. Nguyen §2.2/Fig. 1 and §3.2/Fig. 2 require sending the malicious model and observing the induced client update. Put the intended chosen-neuron component on the actual client loss path and score only its returned authorized update; verify the derivative under that objective. **L.**

### T02 · High · AMIA receives private negatives

[amia.py:388](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/amia.py:388>) trains against records from all supplied client partitions; `:422` pools them for batch construction. Nguyen §2.2 grants distribution access, not those private batches. Train with separate adversary-accessible distribution samples; let the victim client form its private batch. Retain provenance and enforce the attacker/client boundary. **M.**

### T03 · High · AMIA LDP/certification component is absent

`AmiaConfig` ([amia.py:25](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/amia.py:25>)) lacks input-LDP mechanism/budget fields; probe training repeats raw positives instead of independent privatizations. Nguyen §4/Figs. 3–4 require privatized records/robust target training; Eq. (11) is needed for certified claims. Implement that existing-paper component before claiming LDP/certification. Gradient DP in the optional pipeline is a different mechanism. Identify current results as non-LDP empirical probe results meanwhile. **L.**

### T04 · High · LOSS decision differs from its source adversaries

[loss.py:434](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/loss.py:434>) calibrates a held-out-loss quantile and `:463` applies loss≤threshold. Yeom §§3.2–3.3 specify bounded-loss randomization and Gaussian-error decisions. Select the applicable cited adversary and implement its assumptions/procedure. If retaining quantile calibration for FL, state this exact deviation; do not claim source equivalence. **M.**

### T05 · Medium · LOSS advantage has the wrong source meaning

[loss.py:555](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/loss.py:555>) computes .5(TPR+TNR), matching Nguyen Eq. (3), whereas Yeom Def. 4/Eq. (2) gives TPR−FPR=2·balanced_accuracy−1. Keep balanced accuracy separately and report Yeom advantage for LOSS. Do not globally replace Nguyen's metric. **S.**

### T06 · High · Reference subtracts instead of dividing

[reference.py:53](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/reference.py:53>) returns reference NLL−target NLL; both notebooks' score functions do likewise. Carlini §6.1 uses ratios of log perplexities. Difference is not a monotone transformation of ratio across candidates. Implement the source ratio with explicit orientation/denominator handling, and expose the source comparison-model choice instead of forcing the same model ID. **S–M.**

### T07 · Critical · Runner bypasses SPV self-prompt calibration

[runner.py:80](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/runner.py:80>) supplies a generic pretrained reference because SPV's `SPEC` sets `needs_reference=True`. [spv_mia.py:271](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/spv_mia.py:271>) builds a self-prompt reference only when none is supplied. The toy runner ([runner.py:87](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/runner.py:87>)) instead trains a negative-world FL reference. The adaptation explicitly calls the builder. Fu §4.2 requires target-generated reference training. Let this scorer build its own reference with a small attack-specific wiring correction; verify its call and training provenance. No new provider abstraction is needed. **M.**

### T08 · High · SPV probability and orientation change its algebra

[spv_mia.py:182](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/spv_mia.py:182>) uses exp(−mean NLL), not Eq. (1)'s joint probability. `:139` negates Eq. (10), retains target-minus-reference, and runner retains ≥. Recreation cells 2–5 repeat these choices. Monotone nonlinear transforms do not commute with subtraction/averaging; negating fixed-label scores reverses ranking. The claim “ranking/AUC unchanged” is false with unchanged direction. Use the printed Eqs. (1), (5), (10) algebra, with a numerically stable equivalent. Record the paper's local-max/sign tension and seek an erratum if necessary; do not decide from favorable AUC. **M.**

### T09 · High · SPV perturbations are not symmetric reconstructions

[spv_mia.py:187](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/spv_mia.py:187>) independently replaces words with literal `<mask>` strings, without reconstruction or opposite pairs. Recreation cell 5 decodes T5 fills without inserting them back into the original text and lacks pairing. Fu §4.3/Eqs. (8)–(10), Appendix A.3 require symmetric perturbations around x. Reconstruct complete texts, produce source-defined opposite pairs and score the same pairs under both models. The four public prompts and reused FL epochs also differ from the source reference-training schedule. **L.**

### T10 · High · Neighborhood search/reconstruction differs

[neighborhood.py:148](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/neighborhood.py:148>) retains only ten candidates per position before global suitability ranking. `:158` decodes the whole uncased record, potentially changing untouched text. HF scoring ignores `neighbour_swaps`; default n=25 differs from n=100. Compare Mattern §2.2 Alg. 1, §3.3. Restore candidate-selection coverage and reconstruct only intended substitutions using original offsets; support or reject the exposed m setting explicitly. Recreation cell 5 shares the top-ten shortcut; its additional mask-fill helper is a different generator. **M.**

### T11 · Medium · SaMIA generation limit differs

[samia.py:147](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/samia.py:147>) uses training `max_length` as new-token budget, default 64; Kaneko §3 uses total length 1024. Temperature/top-k/top-p and mean ROUGE recall otherwise match. Record and implement the effective source generation budget separately from training truncation. **S.**

### T12 · High · Smoke/FL evaluations do not reproduce source protocols

Modern configs default to tiny-GPT2, synthetic client text, one FL round/four trials; [runner.py:103](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/runner.py:103>) alternates membership worlds. Recreation cell 9 constructs favorable synthetic signals. These do not reproduce the baseline table's WikiMIA/MIMIR, extraction verification or real fine-tuned benchmarks. Some metric hooks omit requested low-FPR metrics; the pipeline's common 1% metric is not a substitute for every source metric. Keep smoke/FL labels explicit. Specify event, data, target/reference training, counts and metrics; state remaining source-protocol deviations. Do not silently replace the authorized FL adaptation with centralized training. **L.**

### T13 · High · Legacy AMIA discriminator objective differs

[recreations/ami/noldp/cifar-exp-noldp.py:124](</Users/pyeshuajs23/Documents/MITACS/Research Documents/code_experiments/recreations/ami/noldp/cifar-exp-noldp.py:124>) returns two sigmoid outputs and `:175` trains with weighted CrossEntropyLoss; analogous blocks appear across the six legacy training scripts. Compare Nguyen §3.2/Eq. (6)'s binary discriminator. Restore the source objective and chosen-neuron mapping, then revalidate. This is not cleanup. **M.**

### T14 · High · Legacy BitRand broadcasts shared batch noise

[recreations/ami/ldp/cifar-eval.py:95](</Users/pyeshuajs23/Documents/MITACS/Research Documents/code_experiments/recreations/ami/ldp/cifar-eval.py:95>) tiles to `(1,r)`, samples one noise row and broadcasts it across records; CelebA/ImageNet copies agree. Training BitRand uses per-record dimensions. Nguyen §4 requires independent privatization. Generate independent per-record randomness and verify it remotely. OME has per-record dimensions and is not implicated in this defect. **S–M.**

### T15 · High · Legacy selection/evaluation reuse and activation-only proxy

[noldp/cifar-exp-noldp.py:212](</Users/pyeshuajs23/Documents/MITACS/Research Documents/code_experiments/recreations/ami/noldp/cifar-exp-noldp.py:212>) uses `x_test` for checkpoint selection at `:230`; `noldp/cifar-eval-noldp.py:144,177` evaluates batches from that same population. Corresponding dataset variants share this pattern. Tests inspect positive `fc2` values, not returned client loss gradients; targets are fixed per dataset. Compare Nguyen §§3.2, 4 and experiments. Use a separate selection split; distinguish activation-condition checks from end-to-end update inference and fixed-target results from population claims. **M–L.**

### T16 · High · WBC pools votes instead of uniformly averaging sizes

[wbc.py:99](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/wbc.py:99>), adaptation cell 14 and recreation cell 3 accumulate all positive windows/all windows. Chen Eq. (13)/Alg. 1 uniformly average per-size fractions. Deltas `[2,-1,-1]`, sizes `[1,3]`: paper=1/6, pooling=1/4. Correct the aggregation independently of schedule B01. **S.**

### T17 · Medium · RAG study differs from its benchmark

`rag.py:107,176` uses custom templates/budgets/study corpora and FL generators. The membership question and answer-only boundary match; gray-box features are not used. Record the experiment as an adaptation and preserve prompt/response provenance. Anderson §3.1's benchmark differs. [Source](https://arxiv.org/html/2405.20446v2). **M.**

## B. Design gaps/flaws

### D01 · High · ReCaLL compares different candidate spans

[recall.py:129](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/recall.py:129>) truncates prefix+text to the budget, while its denominator truncates text alone; long prefixes remove numerator tokens or all evidence. Separate tokenizer calls can introduce internal special tokens. Without BOS, the initial candidate token is unscored in the denominator but scored after the prefix. Xie Eq. (2) requires identical x. Establish matching candidate positions and an explicit initial-token convention; reserve prefix space and reject impossible budgets. `build_prefix:78` also silently clamps/truncates requested shots to the four available. **M.**

### D02 · High · Effective record changes across scoring steps

[zlib.py:79](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/zlib.py:79>) combines truncated NLL with compression of full text, so unused tails change scores. [samia.py:162](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/samia.py:162>) splits full text while FL truncates it; the evaluated suffix may never be exposed in training, and generation accepts unbounded prefixes. Carlini §6.1/Kaneko §2.2 operate on the evaluated record. Define effective text once for matching training, compression and split; retain exact exposure if whole-document membership is intended. Check model context limits. **M.**

### D03 · High · Padding contributes to training targets

[federation.py:223](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/federation.py:223>) and [spv_mia.py:237](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/spv_mia.py:237>) pass padded IDs unchanged as labels. Attention masking does not replace loss targets with the ignore value. This differs from the masking collators in AMIA/LOSS and changes the trained model for every downstream attack. Clone labels and ignore mask-zero positions, preserving genuine EOS tokens. **S.**

### D04 · High · Calibration split disjointness is not guaranteed

[datasets.py:259](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/datasets.py:259>) varies pool/shuffle size with requested budget. Training (`:293`) and calibration (`:347`) request different budgets. When pool sizes diverge, the offset slices different shuffled lists and no longer excludes training records; the streaming shuffle buffer also changes. Actual overlap was not measured. Build one full-budget ordered pool and slice it once; assert ID/effective-token disjointness. **M.**

### D05 · High · WBC silently aligns incompatible tokenizer positions

[wbc.py:139](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/wbc.py:139>) uses separate tokenizers, then `:113` zips NLL arrays, truncating unequal lengths. Config allows different reference models. Chen's per-token delta requires matching events, not equal list lengths. Reject incompatible tokenization and empty evidence. `:91` skips oversized windows; upstream instead clamps. Until short-input policy is fixed, use sufficiently long candidates for the declared schedule. **S–M.**

### D06 · High · Notebook/runtime parity is broken

[runner.py:87](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/runner.py:87>) trains negative-world toy references; Reference/WBC adaptations use untrained references. [samia.py:117](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/samia.py:117>) replaces notebook bigram generation with unigram ranking excluding prefix words; tie direction also differs. [tests/test_toy_parity.py](</Users/pyeshuajs23/Documents/MITACS/Research Documents/tests/test_toy_parity.py>) checks presence/separation, not exact parity. Align implementations and compare fixed-input scores/continuations. SPV's separate wiring defect is T07. **M.**

### D07 · High · Batch membership and model-training membership are conflated

AMIA always performs positive-world FL ([amia.py:261](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/amia.py:261>)) but later varies batch inclusion; [runner.py:179](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/runner.py:179>) labels those results `training_membership_metrics`. Ordinary trial labels mean assignment to a partition. Toy selection always takes first clients ([federation.py:130](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/federation.py:130>)), so an assigned target can never train. Real partial participation also requires exposure evidence. Distinguish batch inclusion, dataset assignment and actual exposure, and retain participant/record evidence. **M.**

### D08 · Medium · Invalid inputs/nonfinite scores can become results

[yaml_config.py:128](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/yaml_config.py:128>) validates dataset/pipeline, but pipeline validation immediately returns for ordinary runs ([pipeline.py:116](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/pipeline.py:116>)). Numeric attack ranges lack equivalent checking. Empty-evidence paths sometimes return zero; [metrics.py:20](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/metrics.py:20>) treats NaN comparisons as losses and [runner.py:95](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/runner.py:95>) predicts negative. Missing classes get zero rates. Validate counts/ranges and finite scores before compute/persistence; represent undefined metrics explicitly. **S–M.**

### D09 · High · Missing aggregation can return an untrained model as success

[federation.py:282](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/federation.py:282>), [amia.py:320](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/amia.py:320>), [loss.py:399](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/loss.py:399>) load captured weights only if present, otherwise leave the base model and continue. Require successful expected-round evidence and an explicit failed-client policy. Missing training cannot count as a completed attack experiment. **M.**

### D10 · Medium · Paired randomness and FL weighting differ across paths

[federation.py:194](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/federation.py:194>) seeds torch inside clients only with a pipeline. Pairing host seeds ([runner.py:74](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/runner.py:74>)) does not guarantee worker RNG/participant equality. Shared FL reports `num_examples=1` ([federation.py:232](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/federation.py:232>)); LOSS uses real counts ([loss.py:310](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/loss.py:310>)), although partition sizes can differ. Record/replay participant and worker seeds, and use the same declared aggregation for comparative studies. No governing black-box paper mandates FedAvg; this is a control defect, not an invented requirement. **M.**

### D11 · Medium · Cache/results omit scientific identity

[config.py:32](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/config.py:32>) hashes config without code/model/dataset revisions; [runner.py:125](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/runner.py:125>) accepts completed cached results. Dataset/model loaders do not pin revisions. [runner.py:164](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/runner.py:164>) retains minimal trial fields; LOSS compact payload drops calibration samples. Corrected code could retrieve old results. Introduce an explicit new method/result version, preserve legacy records, and retain resolved revisions, split/token-span hashes, seeds, reference and perturbation/calibration provenance. **M.**

### D12 · Medium · Default trials cannot substantiate low-FPR performance

Four trials give two negatives, hence empirical FPR resolution .5. Threshold scanning is valid at that resolution but does not establish population FPR .01/.001. Pipeline resolution metadata is useful. Prespecify independent threshold selection, sufficient counts, uncertainty and each paper's operating points. **M.**

### D13 · Medium · Legacy multiprocessing has uncontrolled trial randomness

LDP `task_tpr(i)`/`task_tnr(i)` ignore i for RNG setup ([ldp/cifar-eval.py:264](</Users/pyeshuajs23/Documents/MITACS/Research Documents/code_experiments/recreations/ami/ldp/cifar-eval.py:264>)); module-level pool at `:275`, likewise other variants. Fork can inherit NumPy RNG state; spawn requires guarded execution. Add per-trial streams and a safe entry point, verified on the remote runtime. This is platform-dependent, not an observed duplicate-draw result. **S–M.**

## C. Overengineering

### O01 · Medium · Hidden global run metadata

[amia.py:528](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/amia.py:528>) stores `_AMIA_RUN_CONTEXT` between custom-trial and payload adapters (`:531,561`). Return trials and their metadata explicitly through the existing run boundary. Keep ownership local; compare payloads to preserve behavior. **M.**

### O02 · Low · Unused helper/argument

[wbc.py:147](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/wbc.py:147>) defines `config_to_storage` without callers; shared JSON serialization handles the tuple. [federation.py:115](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/federation.py:115>) accepts unused `global_model`. Remove unused helper/import/argument after checking external caller contracts. **S.**

### O03 · Low · Misleading cleanup indirection

[amia.py:514](</Users/pyeshuajs23/Documents/MITACS/Research Documents/master_script/core/attacks/amia.py:514>) deletes a loop-local name in `clear_experiment_objects`, not caller references. Remove the helper. Any actual lifecycle correction belongs separately at the owner and requires memory verification. **S.**

### O04 · Medium · Duplicated training and metric code has drifted

Shared federation, AMIA/LOSS loops and notebooks differ in padding, seeds and weighting. Repeated `_extra_metrics` functions emit identical AUROC logic. After correctness is agreed, reuse existing canonical helpers for operational notebooks; retain compact pedagogical formula examples. Do not add another framework or unify genuinely different mechanisms. Parity is a prerequisite for calling this simplification. **M.**

### O05 · Low · Inactive legacy experiment blocks

Commented evaluation loops in [ldp/celeba-exp.py:489](</Users/pyeshuajs23/Documents/MITACS/Research Documents/code_experiments/recreations/ami/ldp/celeba-exp.py:489>) onward and [noldp/cifar-eval-noldp.py:217](</Users/pyeshuajs23/Documents/MITACS/Research Documents/code_experiments/recreations/ami/noldp/cifar-eval-noldp.py:217>) onward duplicate active paths. Remove inactive blocks from maintained code, retaining upstream provenance/version history. **S.**

Not flagged: lazy ML imports support optional dependencies; legacy hash variants preserve stored lookups; ScoreContext/AttackSpec serve multiple current attacks; limited RAG embedding caching avoids repeat loads. Removing these without a demonstrated simpler equivalent would not satisfy KISS.

## Remediation order and limits

1. Correct actual mechanism/reference wiring (T01/T02/T07) and membership-event reporting (D07).
2. Correct score/generator algebra (T06/T08/T09/T10/T16), record boundaries (D01–D05), and missing-training handling (D09).
3. Establish reproducible splits, seeds, scientific identities and source metrics before rerunning comparisons.
4. Require T03/T14 before any LDP/certified AMIA claim; revalidate legacy evaluations independently.
5. Apply behavior-preserving O changes separately, with parity checks.

[WBC resolution](wbc-resolution.md) supplies a concrete source-backed schedule policy and unambiguous aggregation correction. [Remote verification](remote-verification.md) supplies discriminating checks. Existing tests passing would not prove paper fidelity. No runtime results or exhaustive runtime-correctness certification are claimed.

**Approval boundary:** the supplied prompt explicitly requires presenting findings and receiving confirmation before code changes. These proposed corrections are ready for review; implementation remains unchanged. WBC's source contradiction and SPV's source-sign interpretation must remain explicit scientific decisions, not choices optimized on test results.

## Local source documents

- [Nguyen_2023_AMIA_FL_LDP.pdf](</Users/pyeshuajs23/Documents/MITACS/Research Documents/written_paper/reference_papers/Nguyen_2023_AMIA_FL_LDP.pdf>)
- [Yeom_2018_Loss_MIA.pdf](</Users/pyeshuajs23/Documents/MITACS/Research Documents/written_paper/reference_papers/Yeom_2018_Loss_MIA.pdf>)
- [Carlini_2021_Extraction_Reference_Zlib.pdf](</Users/pyeshuajs23/Documents/MITACS/Research Documents/written_paper/reference_papers/Carlini_2021_Extraction_Reference_Zlib.pdf>)
- [Shi_2024_MinK.pdf](</Users/pyeshuajs23/Documents/MITACS/Research Documents/written_paper/reference_papers/Shi_2024_MinK.pdf>)
- [Zhang_2025_MinKPP.pdf](</Users/pyeshuajs23/Documents/MITACS/Research Documents/written_paper/reference_papers/Zhang_2025_MinKPP.pdf>)
- [Mattern_2023_Neighborhood.pdf](</Users/pyeshuajs23/Documents/MITACS/Research Documents/written_paper/reference_papers/Mattern_2023_Neighborhood.pdf>)
- [Xie_2024_ReCaLL.pdf](</Users/pyeshuajs23/Documents/MITACS/Research Documents/written_paper/reference_papers/Xie_2024_ReCaLL.pdf>)
- [Kaneko_2024_SaMIA.pdf](</Users/pyeshuajs23/Documents/MITACS/Research Documents/written_paper/reference_papers/Kaneko_2024_SaMIA.pdf>)
- [Fu_2024_SPV_MIA.pdf](</Users/pyeshuajs23/Documents/MITACS/Research Documents/written_paper/reference_papers/Fu_2024_SPV_MIA.pdf>)
- [Chen_2026_WBC.pdf](</Users/pyeshuajs23/Documents/MITACS/Research Documents/written_paper/reference_papers/Chen_2026_WBC.pdf>)
