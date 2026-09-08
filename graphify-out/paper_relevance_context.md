# Paper relevance review for the FL–LLM MIA manuscript

## Bottom line

Copy six PDFs into `written_paper/reference_papers`: the Nguyen et al. AMI paper, its 2025 theoretical follow-up, and four FedLLM papers that directly justify the federated fine-tuning architecture and its system/privacy context. Do not copy the two defense papers, the gradient-reconstruction paper, the generic hardware chapter, or the obsolete v1 duplicate of the FedLLM survey.

The citation boundary is important: the recommended FedLLM surveys support background claims about federated fine-tuning, PEFT, communication, heterogeneity, and privacy threats. They do **not** support the definitions or results of the ten non-AMIA attacks implemented in `master_script`; those claims should cite the corresponding primary papers in `papers/summary/papers/`.

## 1. PDFs to copy and cite

| PDF | Recommendation | Claims it can support |
|---|---|---|
| `papers/IEEE-2023-Active Membership Inference Attack under Local Differential Privacy in Federated Learning.pdf` | **Copy; primary method source.** Despite the filename, this is an AISTATS/PMLR paper, not an IEEE paper. | A dishonest FL server can craft a target-selective chosen neuron; a non-zero returned gradient reveals membership; the nonlinear boundary is trained on target versus non-target examples; the attack is adapted to known LDP noise; success is certified with Monte Carlo estimates and Hoeffding bounds; privacy protection and model utility conflict; `Adv = 0.5(TPR + TNR)` is the paper's success metric. |
| `papers/related litertures/2506.17292v2.pdf` | **Copy; closest follow-up.** | Extends the active-server/LDP research line with theoretical attack-success bounds for fully connected and self-attention layers; demonstrates the LDP privacy–utility trade-off on ResNet and ViT federated vision models; explicitly cites Nguyen et al. (2023). Use it to motivate why transformer internals may expose membership, not as evidence that the repository's LLM probe has the same theorem. |
| `papers/related litertures/2409.15723v3.pdf` | **Copy; current FedLLM background survey.** | Defines the FedLLM setting; motivates federated fine-tuning and prompt learning; documents communication/computation, heterogeneity, convergence, privacy, and security challenges; surveys PEFT and reports that attacks can recover private text from federated language-model training. |
| `papers/related litertures/2501.04436v1.pdf` | **Copy; architecture comparison.** | Distinguishes direct-update FedLLMs, KD-FedLLMs using logits, and Split-FedLLMs using intermediate activations; compares accuracy, communication overhead, and client computation; explains why PEFT is normally needed in resource-constrained FedLLM deployments. This is the best source for positioning the repository's direct model-update pipeline as only one of several designs. |
| `papers/related litertures/2503.12016v3.pdf` | **Copy; systematic federated fine-tuning survey.** | Supports the four principal deployment constraints—communication overhead, non-IID client data, memory wall, and computation burden—plus the role of PEFT, datasets, and benchmarks in practical FedLLM research. |
| `papers/related litertures/COMST-AUG25.pdf` | **Copy; peer-reviewed lifecycle/privacy survey.** | Supports the full FedLLM lifecycle, the use of FedAvg, PEFT/LoRA for practical tuning, privacy threats from shared gradients and model inversion, and the roles/trade-offs of DP, secure multiparty computation, and homomorphic encryption. |

### Exact bibliographic identities

1. **Truc Nguyen, Phung Lai, Khang Tran, NhatHai Phan, and My T. Thai.** “Active Membership Inference Attack under Local Differential Privacy in Federated Learning.” *Proceedings of the 26th International Conference on Artificial Intelligence and Statistics (AISTATS 2023)*, PMLR 206, pp. 5714–5730, 2023. arXiv:2302.12685v2. The vendored recreation provides the same BibTeX identity in `code_experiments/recreations/ami/README.md:37-48`.

2. **Quan Nguyen, Minh N. Vu, Truc Nguyen, and My T. Thai.** “Theoretically Unmasking Inference Attacks Against LDP-Protected Clients in Federated Vision Models.” *Proceedings of the 42nd International Conference on Machine Learning (ICML 2025)*, PMLR 267, 2025. arXiv:2506.17292v2.

3. **Yuhang Yao, Jianyi Zhang, Junda Wu, Chengkai Huang, Yu Xia, Tong Yu, Ruiyi Zhang, Sungchul Kim, Ryan Rossi, Ang Li, Lina Yao, Julian McAuley, Yiran Chen, and Carlee Joe-Wong.** “Federated Large Language Models: Current Progress and Future Directions.” arXiv:2409.15723v3, revised 8 June 2026. No archival venue is stated in the PDF.

4. **Na Yan, Yang Su, Yansha Deng, and Robert Schober.** “Federated Fine-Tuning of LLMs: Framework Comparison and Research Directions.” arXiv:2501.04436v1, 8 January 2025. No archival venue is stated in the PDF.

5. **Yebo Wu, Chunlin Tian, Jingguang Li, He Sun, Kahou Tam, Zhanting Zhou, Haicheng Liao, Jing Xiong, Zhijiang Guo, Li Li, and Chengzhong Xu.** “A Survey on Federated Fine-Tuning of Large Language Models.” *Transactions on Machine Learning Research*, February 2026. arXiv:2503.12016v3.

6. **Yujun Cheng, Weiting Zhang, Zhewei Zhang, Chuan Zhang, Shengjin Wang, and Shiwen Mao.** “Toward Federated Large Language Models: Motivations, Methods, and Future Directions.” *IEEE Communications Surveys & Tutorials*, vol. 27, no. 4, August 2025, pp. 2733 onward. DOI: 10.1109/COMST.2024.3503680 (online publication 21 November 2024).

## 2. PDFs not to copy or cite for the present manuscript

| PDF | Classification and reason |
|---|---|
| `papers/related litertures/Federated_Large_Language_Models_Current_Progress_a.pdf` | **Do not cite/copy: obsolete duplicate.** It is arXiv:2409.15723v1 (24 September 2024) of the same Yao et al. survey. Use `2409.15723v3.pdf` exclusively. |
| `papers/related litertures/2026-s413-paper.pdf` | **Tangential defense work.** “A Unified Defense Framework Against Membership Inference in Federated Learning via Distillation and Contribution-Aware Aggregation” (NDSS 2026) combines modified-entropy teacher training, CVAE synthetic-data distillation, and contribution-aware aggregation. None of these mechanisms is implemented in the notebooks or `master_script`. Cite only if the manuscript later adds a dedicated defenses/future-work comparison. |
| `papers/related litertures/United_We_Defend_Collaborative_Membership_Inferenc.pdf` | **Tangential defense work.** “United We Defend: Collaborative Membership Inference Defenses in Federated Learning” (arXiv:2601.06866v1, 2026) proposes CoFedMID, defender coalitions, selective data use, utility-aware compensation, and aggregation-neutral perturbation against trajectory MIAs. The current pipeline has no coalition, trajectory defense, or noise-cancellation mechanism. |
| `papers/related litertures/2510.23931v1.pdf` | **Different attack objective.** “Differential Privacy: Gradient Leakage Attacks in Federated Learning Environments” (arXiv:2510.23931v1, 2025) evaluates input reconstruction from gradients and compares DP-SGD with PDP-SGD. Gradient reconstruction is not membership inference, and neither defense is implemented here. It cannot substantiate the repository's MIA results. |
| `papers/related litertures/RP_9788770041010C3.pdf` | **Too generic and hardware-focused.** The book chapter “Federated Learning: Privacy, Security and Hardware Perspectives” reviews edge hardware, energy, privacy, and integrity broadly. The experiments do not study hardware optimization or edge deployment, and the recommended FedLLM surveys provide stronger, more direct support for the relevant background claims. |

## 3. What the implementation actually supports

- The repository consolidates 11 attack adaptations—AMIA, LOSS, zlib, Reference, Min-K%, Min-K%++, Neighbourhood, ReCaLL, SaMIA, SPV-MIA, and WBC—into one runner (`master_script/README.md:1-22`).
- For the nine shared-path attacks, each trial constructs a target-member or target-absent client partition, performs Flower FedAvg (or a toy count-model analogue), and scores the **final global model** (`master_script/core/runner.py:26-53`; `master_script/core/federation.py:25-32,74-95,114-166`).
- The real shared path loads a Hugging Face causal LM, optimizes **all** `model.parameters()`, exchanges the full state dictionary, and deliberately reports `num_examples=1`, making aggregation an unweighted client mean (`master_script/core/federation.py:117-166`).
- The AMIA path separately performs ordinary Flower FedAvg, trains a two-layer ReLU probe on pooled final-model hidden states, and thresholds the probe's `fc2` gradient norm (`master_script/core/attacks/amia.py:224-288,310-386`).
- The faithful Nguyen recreation is a vendored upstream baseline at commit `989bc54`; it covers CIFAR-10, ImageNet, and CelebA embeddings, with no-LDP and BitRand/OME code paths (`code_experiments/recreations/ami/UPSTREAM_SOURCE.txt`; `code_experiments/recreations/ami/README.md:23-35`). It does not implement an LLM experiment.

## 4. Material paper-to-code mismatches that the manuscript must disclose

1. **The LLM AMIA is not the paper's malicious-server update attack.** Nguyen et al. modify global-model parameters before a targeted client computes and returns a gradient. Here normal FedAvg finishes first; then a separate probe is trained on frozen, mean-pooled hidden states using the target plus pooled examples from all clients (`amia.py:243-266,342-374`). The measured gradient is produced locally by backpropagating probe activations on a synthetic batch (`amia.py:377-407`), not extracted from a target client's returned FL update.

2. **AMIA positive/negative trials are not matched FL worlds.** The AMIA fine-tuning function always calls `build_client_texts(..., include_target=True)` (`amia.py:243`). Trial labels only control whether the already-known target string is inserted into a post-hoc probe batch (`amia.py:389-407`). In contrast, the shared runner does rebuild the FL world with `truth_member=True/False` (`runner.py:26-39`). Therefore the AMIA `TPR/TNR/Adv` numbers are probe-batch detection metrics, not a faithful execution of Nguyen et al.'s security game.

3. **No LDP or DP experiment exists in the master runner.** The README explicitly states there are no `epsilon` or `ldp_mechanism` fields (`master_script/README.md:433-440`). The AMIA notebook lists hidden/token-embedding perturbation with BitRand/OME only as future scaling work. Consequently, the manuscript may cite Nguyen et al. and Quan Nguyen et al. for prior evidence, but must not say the repository tested LDP, reproduced their privacy–utility curve, or verified their certified bounds.

4. **The modality and attacked layers differ.** Nguyen et al. (2023) use 512-dimensional ResNet-18 image embeddings and a crafted fully connected ReLU neuron; the 2025 follow-up evaluates ResNet and ViT FC/self-attention vulnerabilities. The adaptation uses a post-hoc MLP over mean-pooled causal-LM hidden states. It does not poison the transformer's own fully connected or attention parameters, and it does not implement the 2025 paper's low-polynomial self-attention attack.

5. **The FedLLM system is full-parameter, not PEFT/KD/split.** The recommended FedLLM papers emphasize LoRA/adapters/prompts, KD-logit exchange, or split learning as practical responses to LLM scale. The code optimizes every parameter and sends every state-dict tensor (`federation.py:117-166`; AMIA has the same behavior at `amia.py:200-216`). Claims about PEFT efficiency, heterogeneous LoRA, KD-FedLLM, or Split-FedLLM must be framed as literature context or future work—not as implemented capabilities.

6. **Scale and heterogeneity evidence is limited.** Defaults are `sshleifer/tiny-gpt2`, four synthetic clients, short synthetic records, one or two rounds, and mostly equal-size partitions (`amia.py:23-48`; `federation.py:15-32`). The setup validates orchestration and scoring, but cannot by itself support claims about production LLM scale, realistic non-IID corpora, cross-device resource constraints, or deployment-level communication costs.

7. **Most attacks observe a final global model, not client updates or trajectories.** The common runner fine-tunes a separate member/non-member global model and then scores the target record (`runner.py:26-39`). It does not expose per-client gradients to the attack, model historical trajectories, secure aggregation, or coalition behavior. Accordingly, the two 2026 defense PDFs and the gradient-leakage paper do not describe the implemented threat model.

## Citation recommendation for the manuscript

- Cite **Nguyen et al. (2023)** for the AMI threat model, chosen-neuron mechanism, LDP analysis, and TPR/TNR/Adv definition.
- Cite **Quan Nguyen et al. (2025)** as related work showing theoretical LDP vulnerability in FC and self-attention layers, while explicitly noting its vision-model setting and different attack implementation.
- Cite **Yao et al. (v3), Yan et al., Wu et al., and Cheng et al.** only for FedLLM architecture, PEFT/system challenges, and privacy/security background.
- Cite the attack-specific primary sources under `papers/summary/papers/` for LOSS, Reference/zlib, Min-K%, Min-K%++, Neighbourhood, ReCaLL, SaMIA, SPV-MIA, and WBC methodology or empirical claims.
- Do not cite the five excluded PDFs unless the manuscript's scope expands to implement or systematically compare DP-SGD, reconstruction attacks, collaborative defenses, knowledge-distillation defenses, or edge-hardware optimization.
