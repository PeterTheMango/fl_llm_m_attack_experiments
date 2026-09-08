# Primary-Paper Mapping Review for the FL-LLM Membership Attacks

## Scope and interpretation

This review maps the ten named attack families in `papers/summary/` to the eleven implementation paths represented by the adaptation notebooks and `master_script/core/attacks/`. It also checks the corresponding recreation material where present. The eleven implementation names collapse to ten reporting families because **AMIA and reference are grouped under one project-level family name**, as requested. They are not the same scientific attack: AMIA observes a chosen-neuron gradient signal, while reference calibration observes a target-versus-reference likelihood gap. Both equations and both citations therefore remain explicit below.

Unless stated otherwise, the FL implementation evaluates a finalized global causal language model in matched member and nonmember FL worlds. Here

\[
\ell_{\theta,t}(x)=-\log p_\theta(x_t\mid x_{<t}),\qquad
\bar\ell_\theta(x)=\frac{1}{T-1}\sum_{t=2}^{T}\ell_{\theta,t}(x),
\]

and larger attack scores are written as stronger membership evidence. “Direct transfer” means the source score is retained algebraically; it does not mean the source experimental protocol was reproduced. Several smoke-test paths default to a deliberately simple toy model, so successful execution is evidence of integration, not evidence that the published attack efficacy was replicated.

## Implementation coverage

| Reported family | Summary/PDF source | Adaptation implementation | Recreation | Master implementation |
|---|---|---|---|---|
| AMIA/reference | `02_reference.md`; `reference.pdf`; companion AMI paper | `AMIA_adaptation.ipynb`; `reference_adaptations.ipynb` | `recreations/ami/`; `reference_recreation.ipynb` | `amia.py`; `reference.py` |
| LOSS | `01_loss.md`; `loss.pdf` | `LOSS_adaptation.ipynb` | None | `loss.py` |
| zlib | `03_zlib.md`; `reference.pdf` | `zlib_adaptations.ipynb` | `zlib_recreation.ipynb` | `zlib.py` |
| Min-K% | `04_min_k.md`; `min_k.pdf` | `min_k_adaptations.ipynb` | `min_k_recreation.ipynb` | `min_k.py` |
| Neighborhood | `05_neighborhood.md`; `neighborhood.pdf` | `neighborhood_adaptations.ipynb` | `neighborhood_recreation.ipynb` | `neighborhood.py` |
| Min-K%++ | `06_min_k_plus_plus.md`; `min_k_plus_plus.pdf` | `min_k_plus_plus_adaptations.ipynb` | `min_k_plus_plus_recreation.ipynb` | `min_k_plus_plus.py` |
| WBC | `07_wbc.md`; `wbc.pdf` | `wbc_adaptations.ipynb` | `wbc_recreation.ipynb` | `wbc.py` |
| ReCaLL | `08_recall.md`; `recall.pdf` | `recall_adaptations.ipynb` | `recall_recreation.ipynb` | `recall.py` |
| SaMIA | `09_samia.md`; `samia.pdf` | `samia_adaptations.ipynb` | `samia_recreation.ipynb` | `samia.py` |
| SPV-MIA | `10_spv_mia.md`; `spv_mia.pdf` | `spv_mia_adaptations.ipynb` | `spv_mia_recreation.ipynb` | `spv_mia.py` |

AMIA has no numbered summary/PDF among the ten; its companion paper is needed to map the AMIA implementation accurately. The table retains the combined family only to show all eleven implementation paths.

## 1. AMIA/reference

### Primary PDF and exact citations

- Primary PDF for the numbered summary set: **`reference.pdf`**.
- Reference-calibration citation: Nicholas Carlini, Florian Tramèr, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Úlfar Erlingsson, Alina Oprea, and Colin Raffel, “Extracting Training Data from Large Language Models,” *30th USENIX Security Symposium (USENIX Security 21)*, 2021, pp. 2633–2650; arXiv:2012.07805.
- Companion primary source for the AMIA path: Truc Nguyen, Phung Lai, Khang Tran, NhatHai Phan, and My T. Thai, “Active Membership Inference Attack under Local Differential Privacy in Federated Learning,” *Proceedings of the 26th International Conference on Artificial Intelligence and Statistics (AISTATS 2023)*, PMLR 206, pp. 5714–5730; arXiv:2302.12685. The local filename `IEEE-2023-Active Membership Inference Attack under Local Differential Privacy in Federated Learning.pdf` is misleading: the publication venue is AISTATS/PMLR, not IEEE.

### Implementation paths

- Reference branch: `code_experiments/adaptations/reference_adaptations.ipynb`, `code_experiments/recreations/reference_recreation.ipynb`, and `master_script/core/attacks/reference.py`.
- AMIA branch: `code_experiments/adaptations/AMIA_adaptation.ipynb`, the upstream recreation under `code_experiments/recreations/ami/`, and `master_script/core/attacks/amia.py`.

### Source threat model and access level

The Carlini reference method is a passive, example-level attack. It requires likelihood or perplexity access to the target language model and to a smaller or otherwise independently trained reference model. It was presented as a ranking method for extracting rare memorized sequences from a pretrained GPT-2, not as an FL client-attribution game.

Nguyen et al.'s AMI is an active, malicious-server attack in federated learning. The server deliberately introduces a chosen neuron into the global model and observes a target client's update. The source study assumes knowledge of the relevant distribution and, in the LDP setting, the local privacy mechanism and budget. Its distinguishing observable is whether the chosen neuron's gradient is nonzero in the client update.

### Source scores/equations

For reference calibration, Carlini et al. rank sequences by a ratio of target and reference perplexity (equivalently, a ratio of log-perplexities when written using mean NLL):

\[
r_{\mathrm{ref}}(x)=\frac{\bar\ell_\theta(x)}{\bar\ell_\phi(x)},
\]

where lower values indicate unexpectedly high target-model confidence relative to the reference. A commonly used later log-space difficulty-calibration form is

\[
s_{\mathrm{ref}}^{\mathrm{gap}}(x)=\bar\ell_\phi(x)-\bar\ell_\theta(x),
\]

where larger values indicate membership. The ratio and difference rank similarly only under additional conditions; they should not be called algebraically identical.

For AMI, the source chosen neuron is of the form

\[
v_{h,W}(x)=h\,\operatorname{ReLU}(Wx),
\]

and is trained through a sigmoid classifier. At the selected FL round the source adversary decides membership from the chosen-neuron update,

\[
\hat m=\mathbf 1[\,g_t\ne0\,].
\]

The source paper defines attack success as \(\tfrac12\mathrm{TPR}+\tfrac12\mathrm{TNR}\), rather than the more common \(\mathrm{TPR}-\mathrm{FPR}\) advantage.

### Implemented FL-LLM scores/equations

The reference path implements the log-space NLL gap directly:

\[
s_{\mathrm{ref}}^{\mathrm{NLL}}(x)=\bar\ell_{\phi}(x)-\bar\ell_{\theta}(x),
\qquad \hat m=\mathbf 1[s_{\mathrm{ref}}^{\mathrm{NLL}}\ge\tau_{\mathrm{ref}}].
\]

Here \(\theta\) is the final FL model and \(\phi\) is the unfine-tuned base model in the Hugging Face path; the toy path uses an empty/untrained toy reference. The configured default threshold is 0.25.

The AMIA path instead freezes the final FL language model, maps each example to a sentence embedding \(e_\theta(x)\), trains a two-layer ReLU probe \(q\), and scores a batch \(B\) using the norm of the gradient on the second probe layer:

\[
s_{\mathrm{AMI}}^{\mathrm{grad}}(B)
=\left\|\nabla_{w_{\mathrm{fc2}}}
\sum_{x\in B}\operatorname{ReLU}(q(e_\theta(x)))\right\|_2,
\qquad
\hat m=\mathbf 1[s_{\mathrm{AMI}}^{\mathrm{grad}}>10^{-8}].
\]

Thus the two paths grouped here do not observe the same signal and do not share a score equation.

### Direct transfer versus adaptation

- Reference: **score adaptation**. The target/reference difficulty idea transfers, but the code uses an NLL difference rather than Carlini's reported ratio and changes the task from pretraining-data extraction to post-FL fine-tuning membership.
- AMIA: **substantial adaptation**. It keeps the broad “member example activates a specially trained neuron, creating a nonzero gradient” rationale, but changes the model family, probe placement, training schedule, and observable.

### Deviations and caveats

- The AMIA implementation is not an end-to-end reproduction of the source active-server protocol. Its probe is trained post hoc after FL rather than inserted into the global model before the target client update.
- The implementation computes the gradient locally on a constructed raw-text batch; it does not recover the signal from the actual Flower update sent by the target client. Its negative pool spans clients rather than isolating a single target client's matched update.
- The source AMI evaluates image models and explicitly studies LDP. The LLM adaptation uses hidden-state embeddings and does not apply LDP.
- `reports/ami/report.md` shows a discrete gradient-norm signal and AUC 1.0 in its small experiment, but the nearly zero deployed threshold still creates false positives; this should not be read as replication of source efficacy.
- There is no parallel reference-path experiment report in `reports/`. Its code is principally a score implementation and smoke-test path.
- Combining the names is a project taxonomy only. A paper must cite Carlini et al. for the NLL-gap/reference path and Nguyen et al. for the chosen-gradient path.

### Recommended notation

Use \(\phi\) for the reference model, \(s_{\mathrm{ref}}^{\mathrm{NLL}}\) for the likelihood gap, \(q\) for the learned probe, and \(s_{\mathrm{AMI}}^{\mathrm{grad}}\) for the probe-gradient norm. Do not use one unqualified \(s_{\mathrm{AMIA}}\) for both paths.

## 2. LOSS

### Primary PDF and exact citation

- Primary PDF: **`loss.pdf`**.
- Samuel Yeom, Irene Giacomelli, Matt Fredrikson, and Somesh Jha, “Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting,” *2018 IEEE 31st Computer Security Foundations Symposium (CSF)*, 2018; arXiv:1709.01604.

### Implementation paths

`code_experiments/adaptations/LOSS_adaptation.ipynb` and `master_script/core/attacks/loss.py`. There is no LOSS recreation notebook under `code_experiments/recreations/`.

### Source threat model and access level

The source is a passive membership attack on a supervised predictor. Its simple form needs the true label, the target model's per-example loss, and a bound on that loss. The stronger Bayes form assumes knowledge or estimation of the conditional loss/error distributions for member and nonmember records. It does not require a reference model.

### Source score/equation

Yeom et al.'s randomized adversary predicts membership with probability

\[
\Pr[\hat m=1\mid z]=1-\frac{\ell(A_S,z)}{B},
\]

for loss bounded by \(B\). Its source advantage connects to expected generalization error through \(R_{\mathrm{gen}}/B\). The deterministic intuition is simply that lower loss gives stronger membership evidence; the Bayes adversary thresholds a member/nonmember loss-density likelihood ratio.

### Implemented FL-LLM score/equation

The implementation uses the final global causal LM's mean next-token NLL and a calibrated lower-tail threshold:

\[
s_{\mathrm{LOSS}}(x)=-\bar\ell_\theta(x),
\qquad
\tau_{\mathrm{loss}}=Q_{0.10}\{\bar\ell_\theta(x):x\in D_{\mathrm{cal}}^{\mathrm{non}}\},
\qquad
\hat m=\mathbf 1[\bar\ell_\theta(x)\le\tau_{\mathrm{loss}}].
\]

The adaptation constructs matched FL worlds and evaluates the final global model.

### Direct transfer versus adaptation

**Conceptual transfer with a material protocol adaptation.** Lower loss remains the membership signal, but the source randomized/bounded-loss adversary is replaced by an empirical nonmember-quantile rule for sequence NLL in an FL-trained causal LM.

### Deviations and caveats

- The configured quantile rule is neither the source randomized adversary nor its conditional-density Bayes-optimal adversary.
- The reported project “advantage” is balanced accuracy, \(\tfrac12(\mathrm{TPR}+\mathrm{TNR})\), rather than Yeom et al.'s conventional \(\mathrm{TPR}-\mathrm{FPR}\). Label it explicitly.
- This is passive final-model membership, not client-update attribution.
- `reports/loss/report.md` records high score separability by ROC AUC but all-negative predictions at the deployed threshold, exposing calibration-distribution mismatch. AUC and thresholded advantage must be reported separately.

### Recommended notation

Use \(\bar\ell_\theta(x)\) for mean token NLL, \(s_{\mathrm{LOSS}}=-\bar\ell_\theta\) for a consistently higher-is-member score, and \(\tau_{\mathrm{loss}}\) for the calibrated loss cutoff. Denote balanced accuracy as \(\mathrm{BA}\), not \(\mathrm{Adv}\), unless the project definition is stated.

## 3. zlib

### Primary PDF and exact citation

- Primary PDF: **`reference.pdf`**; zlib is an extraction metric from the same paper, not a separate PDF.
- Nicholas Carlini, Florian Tramèr, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Úlfar Erlingsson, Alina Oprea, and Colin Raffel, “Extracting Training Data from Large Language Models,” *30th USENIX Security Symposium (USENIX Security 21)*, 2021, pp. 2633–2650; arXiv:2012.07805.

### Implementation paths

`code_experiments/adaptations/zlib_adaptations.ipynb`, `code_experiments/recreations/zlib_recreation.ipynb`, and `master_script/core/attacks/zlib.py`.

### Source threat model and access level

The source is a passive pretraining-data extraction/ranking method. It needs target-model sequence likelihood or perplexity plus the raw candidate text so its generic compressibility can be measured. It does not require a learned reference model; zlib acts as a non-neural complexity baseline.

### Source score/equation

The paper ranks examples using the ratio of log perplexity to zlib entropy/compressed size:

\[
r_{\mathrm{zlib}}(x)=\frac{\log\operatorname{PPL}_\theta(x)}{H_{\mathrm{zlib}}(x)}
=\frac{\bar\ell_\theta(x)}{H_{\mathrm{zlib}}(x)},
\]

where lower values are more suspicious.

### Implemented FL-LLM score/equation

The code reverses the orientation so larger values indicate membership:

\[
H_{\mathrm{zlib}}(x)=8\,|\operatorname{zlib.compress}(x)|,
\qquad
s_{\mathrm{zlib}}(x)=-\frac{\bar\ell_\theta(x)}{H_{\mathrm{zlib}}(x)}.
\]

This is the same core statistic because mean NLL equals log perplexity.

### Direct transfer versus adaptation

**Direct score transfer with an FL task adaptation.** The algebra is retained up to sign; the target changes from a pretrained LM and extraction ranking to the final model of an FL fine-tuning world and binary membership evaluation.

### Deviations and caveats

- The default smoke path uses a toy token model, so the likelihood component is not a contextual causal-LM likelihood.
- The configured fixed threshold and four-trial experiment are not calibrated to reproduce the paper's ranking setup.
- `reports/zlib/report.md` notes deterministic toy behavior and one-round FL. Its perfect AUC is an artifact-scale result, while the fixed threshold lies outside the observed score range and yields no positive decisions.
- Compressed byte length includes serialization and short-string overhead; for small snippets, this can dominate linguistic complexity.

### Recommended notation

Use \(H_z(x)\) for compressed length in bits, \(r_{\mathrm{zlib}}=\bar\ell/H_z\) for the paper's lower-is-member ratio, and \(s_{\mathrm{zlib}}=-r_{\mathrm{zlib}}\) for the implementation-oriented score.

## 4. Min-K%

### Primary PDF and exact citation

- Primary PDF: **`min_k.pdf`**.
- Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra Blevins, Danqi Chen, and Luke Zettlemoyer, “Detecting Pretraining Data from Large Language Models,” *International Conference on Learning Representations (ICLR)*, 2024; arXiv:2310.16789.

### Implementation paths

`code_experiments/adaptations/min_k_adaptations.ipynb`, `code_experiments/recreations/min_k_recreation.ipynb`, and `master_script/core/attacks/min_k.py`.

### Source threat model and access level

The source is a passive, reference-free attack for detecting pretraining data. It needs per-token target-model log probabilities. This is score-based black-box access if the API exposes token logprobs; otherwise it is naturally grey-box or open-weight access.

### Source score/equation

Let \(I_k(x)\) be the indices of the lowest-\(k\%\) token log probabilities. The score is

\[
s_{\mathrm{MinK}}(x)=\frac{1}{|I_k(x)|}
\sum_{t\in I_k(x)}\log p_\theta(x_t\mid x_{<t}),
\]

with larger, less-negative values indicating membership.

### Implemented FL-LLM score/equation

The FL code retains the equation and evaluates it on the final global model. It selects

\[
K=\max\left(1,\operatorname{round}\left((T-1)\frac{k}{100}\right)\right)
\]

positions with the smallest log probabilities and averages them. The default is \(k=20\) with a fixed threshold near \(-4.08\).

### Direct transfer versus adaptation

**Direct score transfer plus an FL wrapper.** The source paper primarily studies pretraining-data detection but also reports contaminated one-epoch LLaMA fine-tuning, making the fine-tuning application related. The distributed-client membership game and matched FL worlds remain project-specific adaptations.

### Deviations and caveats

- The toy path uses unigram-like pseudo-probabilities and does not model the conditional context that defines the source score.
- Rounding and the forced minimum of one token matter substantially for short examples.
- The recreation demonstrates score behavior on synthetic/Hugging Face inputs; it is not a replication of the source datasets or target checkpoints.
- Small trial counts and a fixed threshold do not establish source-level efficacy.

### Recommended notation

Use \(a_t=\log p_\theta(x_t\mid x_{<t})\), \(I_k(x)=\operatorname{BottomK}_k\{a_t\}\), and \(s_{\mathrm{MinK}}\) for the average. Reserve \(\ell_t=-a_t\) for loss-based derivations to prevent sign errors.

## 5. Neighborhood

### Primary PDF and exact citation

- Primary PDF: **`neighborhood.pdf`**.
- Justus Mattern, Fatemehsadat Mireshghallah, Zhijing Jin, Bernhard Schölkopf, Mrinmaya Sachan, and Taylor Berg-Kirkpatrick, “Membership Inference Attacks against Language Models via Neighbourhood Comparison,” *Findings of the Association for Computational Linguistics: ACL 2023*, Toronto, Canada, July 2023, pp. 11330–11343, doi:10.18653/v1/2023.findings-acl.719; arXiv:2305.18462.

### Implementation paths

`code_experiments/adaptations/neighborhood_adaptations.ipynb`, `code_experiments/recreations/neighborhood_recreation.ipynb`, and `master_script/core/attacks/neighborhood.py`.

### Source threat model and access level

The source attacks fine-tuned language models. It needs the target model's sequence loss/confidence and a separate masked language model to construct plausible semantic neighbors, but it does not need a member-trained reference model. This is best described as target score access plus auxiliary-model access.

### Source score/equation

For a generated neighborhood \(N(x)=\{\tilde x_i\}_{i=1}^{n}\), the source decides

\[
\hat m=\mathbf 1\!\left[
\bar\ell_\theta(x)-\frac1n\sum_{i=1}^{n}\bar\ell_\theta(\tilde x_i)<\gamma
\right].
\]

Equivalently, a higher-is-member score is

\[
s_{\mathrm{NB}}(x)=\frac1n\sum_i\bar\ell_\theta(\tilde x_i)-\bar\ell_\theta(x).
\]

### Implemented FL-LLM score/equation

The implementation uses exactly this higher-is-member form on the final FL model and thresholds \(s_{\mathrm{NB}}\). The Hugging Face path masks and replaces tokens with BERT-generated candidates; the default configuration asks for 25 neighbors.

### Direct transfer versus adaptation

**Direct core-score transfer plus an FL wrapper.** The source already targets fine-tuned language models, so the main adaptation is from centralized fine-tuning membership to federated final-model membership.

### Deviations and caveats

- The paper's strongest settings use a larger neighborhood (commonly 100); the code default of 25 changes estimator variance and cost.
- The Hugging Face neighbor generator uses token masking, embedding dropout, and a suitability normalization specific to this implementation.
- The toy path substitutes benign filler tokens, not masked-LM semantic neighbors, so it tests the subtraction logic rather than the source neighborhood construction.
- The auxiliary BERT model makes “reference-free” mean no membership reference model, not no auxiliary model.

### Recommended notation

Use \(N(x)\) for the neighbor multiset, \(\tilde x_i\) for an individual neighbor, and \(s_{\mathrm{NB}}=\mathbb E_{\tilde x\in N(x)}[\bar\ell_\theta(\tilde x)]-\bar\ell_\theta(x)\). If quoting the paper's \(\gamma\), state that the implementation's threshold has the opposite-sign orientation.

## 6. Min-K%++

### Primary PDF and exact citation

- Primary PDF: **`min_k_plus_plus.pdf`**.
- Jingyang Zhang, Jingwei Sun, Eric Yeats, Yang Ouyang, Martin Kuo, Jianyi Zhang, Hao Frank Yang, and Hai Li, “Min-K%++: Improved Baseline for Detecting Pre-Training Data from Large Language Models,” *International Conference on Learning Representations (ICLR)*, 2025, OpenReview forum ZGkfoufDaU; arXiv:2404.02936.

### Implementation paths

`code_experiments/adaptations/min_k_plus_plus_adaptations.ipynb`, `code_experiments/recreations/min_k_plus_plus_recreation.ipynb`, and `master_script/core/attacks/min_k_plus_plus.py`.

### Source threat model and access level

The source is a passive, reference-free pretraining-data detector. Unlike Min-K%, it needs the entire next-token output distribution at every scored position so that the observed token log probability can be standardized. This is full-logit API access or grey-box/open-weight access.

### Source score/equation

At token position \(t\), define

\[
\mu_t=\sum_{v}p_t(v)\log p_t(v),
\qquad
\sigma_t=\sqrt{\sum_v p_t(v)(\log p_t(v))^2-\mu_t^2},
\]

\[
z_t=\frac{\log p_t(x_t)-\mu_t}{\sigma_t}.
\]

The source score averages the lowest-\(k\%\) standardized values:

\[
s_{\mathrm{MinK++}}(x)=\frac1{|I_k^z(x)|}\sum_{t\in I_k^z(x)}z_t.
\]

Larger scores indicate membership.

### Implemented FL-LLM score/equation

The Hugging Face implementation retains the same \(\mu_t\), \(\sigma_t\), \(z_t\), and lower-tail averaging on the final FL model. It uses

\[
K=\max\left(1,\left\lfloor (T-1)\frac{k}{100}\right\rfloor\right)
\]

and maps zero-variance positions to \(z_t=0\).

### Direct transfer versus adaptation

**Direct score transfer with an FL task adaptation.** The statistical normalization is preserved, while the evaluated membership event changes from pretraining inclusion to client-local FL fine-tuning inclusion.

### Deviations and caveats

- The source requires a meaningful contextual vocabulary distribution. The toy path reuses a unigram pseudo-distribution at every position, so it cannot reproduce the intended conditional standardization.
- The toy membership signal is introduced through a fixed temperature manipulation, making smoke separability partly constructed.
- Flooring versus rounding the selected fraction affects short sequences. The zero-variance fallback also creates artificial zero scores in degenerate cases.
- The source paper does not establish this method as an FL attack; that claim must be attributed to the adaptation experiment, not the citation.

### Recommended notation

Use \(\mu_t\) and \(\sigma_t\) for the model-distribution moments, \(z_t\) for the standardized observed-token log probability, \(I_k^z\) for its lower tail, and \(s_{\mathrm{MinK++}}\) for the average.

## 7. WBC

### Primary PDF and exact citation

- Primary PDF: **`wbc.pdf`**.
- Yuetian Chen, Yuntao Du, Kaiyuan Zhang, Ashish Kundu, Charles Fleming, Bruno Ribeiro, and Ninghui Li, “Window-based Membership Inference Attacks Against Fine-tuned Large Language Models,” arXiv:2601.02751v2, 5 March 2026. The local primary source is an arXiv preprint; no peer-reviewed venue should be claimed from the supplied material.

### Implementation paths

`code_experiments/adaptations/wbc_adaptations.ipynb`, `code_experiments/recreations/wbc_recreation.ipynb`, and `master_script/core/attacks/wbc.py`.

### Source threat model and access level

The source is a passive attack against fine-tuned LLMs. It needs per-token target-model NLLs and per-token NLLs from the corresponding pretrained/reference model. It therefore assumes score access to two models, or open weights for both.

### Source score/equation

Define the tokenwise target improvement and a window sum:

\[
\Delta_t(x)=\ell_{\phi,t}(x)-\ell_{\theta,t}(x),
\qquad
S_i(w)=\sum_{j=i}^{i+w-1}\Delta_j(x).
\]

For each window size, the source normalizes the number of positive windows,

\[
T_{\mathrm{sign}}(w)=\frac{1}{T-w+1}\sum_{i=1}^{T-w+1}\mathbf 1[S_i(w)>0],
\]

then gives every window size equal weight:

\[
s_{\mathrm{WBC}}^{\mathrm{paper}}(x)=\frac1{|\mathcal W|}\sum_{w\in\mathcal W}T_{\mathrm{sign}}(w).
\]

### Implemented FL-LLM score/equation

The implementation forms the same tokenwise NLL differences and positive-window indicators, but pools all windows before normalizing:

\[
s_{\mathrm{WBC}}^{\mathrm{pool}}(x)=
\frac{\sum_{w\in\mathcal W}\sum_{i=1}^{T-w+1}\mathbf 1[S_i(w)>0]}
{\sum_{w\in\mathcal W}(T-w+1)}.
\]

The configured window-size set is

\[
\mathcal W=(2,3,4,6,9,13,18,25,32,40).
\]

The final FL model is \(\theta\); the unfine-tuned base model is \(\phi\).

### Direct transfer versus adaptation

**Direct token/window construction with a material aggregation deviation and an FL wrapper.** The source already studies fine-tuned LLMs, but the project changes centralized fine-tuning to federated fine-tuning and changes how window sizes are weighted.

### Deviations and caveats

- The implementation's pooled normalization weights smaller window sizes more because they contribute more windows. The paper normalizes within each size first and then averages sizes equally. These scores should not share an unqualified equation label.
- The literal window set used in the implementation and source text is not exactly the set obtained from the geometric formula reproduced in the recreation notes, which yields approximately `(2,3,4,5,8,11,15,21,29,40)`. State whether the literal list or formula is used.
- The default threshold 0.75 is configuration-specific and is not a universal source-paper cutoff.
- The toy path uses an empty reference model and an artificial background vocabulary; it tests the mechanics, not calibrated LLM tokenwise difficulty.
- Because the source is a March 2026 preprint, citation and equation numbering should be versioned as v2.

### Recommended notation

Use \(\phi\) for the pretrained reference, \(\Delta_t\) for tokenwise NLL improvement, \(S_i(w)\) for the window sum, and \(T_{\mathrm{sign}}(w)\) for the within-size positive fraction. Distinguish \(s_{\mathrm{WBC}}^{\mathrm{paper}}\) from \(s_{\mathrm{WBC}}^{\mathrm{pool}}\).

## 8. ReCaLL

### Primary PDF and exact citation

- Primary PDF: **`recall.pdf`**.
- Roy Xie, Junlin Wang, Ruomin Huang, Minxing Zhang, Rong Ge, Jian Pei, Neil Zhenqiang Gong, and Bhuwan Dhingra, “ReCaLL: Membership Inference via Relative Conditional Log-Likelihoods,” *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, Miami, Florida, USA, November 2024, pp. 8671–8689, doi:10.18653/v1/2024.emnlp-main.493; arXiv:2406.15968.

### Implementation paths

`code_experiments/adaptations/recall_adaptations.ipynb`, `code_experiments/recreations/recall_recreation.ipynb`, and `master_script/core/attacks/recall.py`.

### Source threat model and access level

The source is a passive, reference-model-free pretraining-membership attack. It needs target-model likelihood access and one or more fixed nonmember passages used as prefixes. It does not train or query a second language model.

### Source score/equation

Let \(P\) be a known nonmember prefix and \(\operatorname{LL}_\theta(x)=\sum_t\log p_\theta(x_t\mid x_{<t})\). ReCaLL uses

\[
s_{\mathrm{ReCaLL}}(x;P)=
\frac{\operatorname{LL}_\theta(x\mid P)}{\operatorname{LL}_\theta(x)}.
\]

Because both log-likelihoods are negative, members whose likelihood is relatively destabilized by the nonmember context tend to obtain a larger ratio, often above one.

### Implemented FL-LLM score/equation

The Hugging Face path concatenates \(P\) and \(x\), masks the prefix labels, sums only the target-token conditional log-likelihood, and computes the same ratio against the unprefixed target log-likelihood. The adaptation applies it to the final global FL model with one default prefix shot.

### Direct transfer versus adaptation

**Direct score transfer with an FL task adaptation.** The prefix-conditioning equation is preserved; pretraining membership is replaced by membership in a target client's FL fine-tuning data.

### Deviations and caveats

- The source gains stability by aggregating across selected nonmember prefixes. The default code uses one shot and does not reproduce the paper's complete prompt-selection/ensemble protocol.
- The toy path explicitly depresses memorized-token probabilities under the prefix by a hand-designed count-dependent factor. This validates score direction but bakes in the expected signal.
- The configured 64-token limit may truncate the target after adding the prefix, changing which target tokens contribute.
- A “known nonmember prefix” is an auxiliary-data assumption even though no reference model is required.

### Recommended notation

Use \(P\) for the nonmember prefix, \(\operatorname{LL}_\theta(x\mid P)\) for target-only conditional log-likelihood, and \(s_{\mathrm{ReCaLL}}(x;P)\) for the ratio. If multiple prefixes are averaged, add an explicit prefix index rather than hiding the ensemble in \(P\).

## 9. SaMIA

### Primary PDF and exact citation

- Primary PDF: **`samia.pdf`**.
- Masahiro Kaneko, Youmi Ma, Yuki Wata, and Naoaki Okazaki, “Sampling-based Pseudo-Likelihood for Membership Inference Attacks,” arXiv:2404.11262 [cs.CL], 2024.

### Implementation paths

`code_experiments/adaptations/samia_adaptations.ipynb`, `code_experiments/recreations/samia_recreation.ipynb`, and `master_script/core/attacks/samia.py`.

### Source threat model and access level

The source is a generation-only black-box attack. It does not need token probabilities or model weights. It splits a candidate into a visible prefix and held-out suffix, samples multiple continuations, and measures their textual overlap with the true suffix.

### Source score/equation

Let \(p(x)\) and \(r(x)\) be the first and second portions of the candidate, and sample \(c_j\sim p_\theta(\cdot\mid p(x))\) for \(j=1,\ldots,m\). The core score is

\[
s_{\mathrm{SaMIA}}(x)=\frac1m\sum_{j=1}^{m}
\operatorname{ROUGE\!\!-N}(c_j,r(x)).
\]

The source also studies a zlib-weighted variant that multiplies or normalizes sample similarity using compressed-length information. Larger overlap indicates stronger membership evidence.

### Implemented FL-LLM score/equation

The Hugging Face FL path retains the prefix/suffix split and mean ROUGE score, with default \(m=10\), ROUGE-1, temperature 1, top-k 50, and top-p 1.0. The optional zlib weighting is disabled by default. It scores generations from the final global model.

### Direct transfer versus adaptation

**Direct black-box score transfer with an FL task adaptation.** The source generation-and-overlap observable is preserved, while pretraining membership is replaced by client-local FL fine-tuning membership.

### Deviations and caveats

- `max_new_tokens=64` is far shorter than the source's long generation budget, so it changes both recall and computational cost.
- The toy path returns the same deterministic count-ranked candidate repeatedly; it is not sampling and cannot estimate variation across continuations.
- The implementation does not reproduce the source prompt, decoding, dataset, or model-scale study. The recreation is formula-level validation.
- Generation-only access is a genuine black-box advantage, but score reproducibility depends heavily on decoding seeds and sampling parameters.

### Recommended notation

Use \(p(x)\) for the revealed prefix, \(r(x)\) for the held-out suffix, \(c_j\) for sampled continuations, \(R_N(c_j,r)\) for ROUGE-N, \(s_{\mathrm{SaMIA}}\) for the plain mean, and \(s_{\mathrm{SaMIA}\times z}\) for the compression-weighted variant.

## 10. SPV-MIA

### Primary PDF and exact citation

- Primary PDF: **`spv_mia.pdf`**.
- Wenjie Fu, Huandong Wang, Chen Gao, Guanghua Liu, Yong Li, and Tao Jiang, “Membership Inference Attacks against Fine-tuned Large Language Models via Self-prompt Calibration,” *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*, 2024, doi:10.52202/079017-4290; arXiv:2311.06062.

### Implementation paths

`code_experiments/adaptations/spv_mia_adaptations.ipynb`, `code_experiments/recreations/spv_mia_recreation.ipynb`, and `master_script/core/attacks/spv_mia.py`.

### Source threat model and access level

The source attacks fine-tuned LLMs using an API setting that exposes generation and probability/loss information and permits fine-tuning a base model on self-prompted data. It is consequently not a pure text-only black-box attack: it requires both scoring access and the capability to construct a calibrated auxiliary model \(\dot\theta\).

### Source score/equation

The source generates a self-prompt dataset \(D_{\mathrm{self}}\), fine-tunes a base model to obtain \(\dot\theta\), and approximates a probability-variation statistic using positive and negative paraphrases:

\[
\widetilde p_\theta(x)\approx
\frac{1}{2N}\sum_{n=1}^{N}
\left[p_\theta(\tilde x_n^+)+p_\theta(\tilde x_n^-)\right]-p_\theta(x).
\]

Its calibrated score is

\[
s_{\mathrm{SPV}}^{\mathrm{paper}}(x)
=\widetilde p_\theta(x)-\widetilde p_{\dot\theta}(x),
\]

followed by a threshold decision.

### Implemented FL-LLM score/equation

The implementation defines the sign-reversed variation proxy

\[
\operatorname{PV}_\theta(x)=p_\theta(x)-\frac1J\sum_{j=1}^{J}p_\theta(\tilde x_j),
\qquad
p_\theta(x)\approx\exp[-\bar\ell_\theta(x)],
\]

and uses

\[
s_{\mathrm{SPV}}^{\mathrm{impl}}(x)=
\operatorname{PV}_\theta(x)-\operatorname{PV}_{\dot\theta}(x).
\]

If the same paraphrase set were used, this is the negative of the source difference. The code consistently treats larger implementation scores as stronger membership evidence, so the sign convention must be stated whenever comparing numbers with the paper.

### Direct transfer versus adaptation

**Family-level transfer with material approximation and FL adaptations.** The self-prompt calibration architecture transfers, and the source already targets fine-tuned LLMs. However, the implemented probability proxy, perturbation construction, sign convention, and reference-training budget differ.

### Deviations and caveats

- The code's \(\exp(-\bar\ell)\) is a length-normalized geometric-mean token probability, not a joint sequence probability.
- The adaptation's fallback paraphrases are independently masked/replaced strings, not paired positive/negative perturbations around \(x\). They therefore do not implement the source symmetric finite-difference approximation exactly.
- The recreation contains T5 masking/reconstruction helpers closer to the paper, but its smoke path is synthetic and does not establish end-to-end equivalence.
- The Hugging Face adaptation trains \(\dot\theta\) on only four default public prompts for one local epoch, far below the source self-prompt corpus and reported target/reference training budgets.
- The source assumes a fine-tuning facility for constructing \(\dot\theta\); papers should not describe the resulting attack as needing only a single target-model query interface.

### Recommended notation

Use \(\theta\) for the attacked final FL model, \(\dot\theta\) for the self-prompt reference, \(D_{\mathrm{self}}\) for reference-training prompts, and \(\tilde x_n^+,\tilde x_n^-\) for paired perturbations. Distinguish \(\widetilde p_\theta\) (paper orientation), \(\operatorname{PV}_\theta\) (implementation orientation), \(s_{\mathrm{SPV}}^{\mathrm{paper}}\), and \(s_{\mathrm{SPV}}^{\mathrm{impl}}\).

## Cross-cutting conclusions for paper use

1. **Only the reference and AMIA names are combined; their scientific attribution must remain split.** Reference calibration is a passive example-level target/reference NLL comparison. AMIA is intended to be an active malicious-server gradient attack, but the present LLM path is a post-hoc local probe-gradient experiment rather than the source FL intervention.
2. **The most direct score transfers are zlib, Min-K%, Neighborhood, Min-K%++, ReCaLL, and SaMIA.** Each still changes the membership event to FL fine-tuning and may default to a toy model.
3. **WBC and SPV-MIA need equation-level qualification.** WBC pools windows instead of averaging normalized scores by window size. SPV-MIA reverses the paper variation convention and approximates sequence probability and perturbations differently.
4. **LOSS retains the loss intuition but not the source decision rule.** Its nonmember-quantile calibration is an implementation choice.
5. **Recreation notebooks are not source-paper replications.** They primarily check formulas and interfaces on synthetic or small Hugging Face examples. The absence of a LOSS recreation should be disclosed rather than silently treated as coverage.
6. **Metric naming must be harmonized.** The project reports \(\tfrac12(\mathrm{TPR}+\mathrm{TNR})\), which is balanced accuracy. If the final paper calls it “advantage,” it should state that definition and distinguish it from \(\mathrm{TPR}-\mathrm{FPR}\).
