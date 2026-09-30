# Novelty and research-value review

**Proposal:** Keeping Private Facts Out of the Model: Retrieval-Grounded Federated Fine-Tuning for Privacy-Preserving RAG  
**Review date:** 30 September 2026  
**Recommendation:** Revise the contribution and experimental design, then run an informative pilot.

## 1. Overall assessment

The proposal asks a worthwhile question, but its central training mechanism has substantial prior art. Training a language model to predict answers from supplied documents, including computing loss only on output tokens, is established. Federated fine-tuning of RAG systems also exists. The proposed verbatim gate is a modest variation within an existing family of retrieval filtering defenses.

The best opportunity is an **empirical contribution about how grounded federated fine-tuning changes privacy across model parameters, server-visible updates, and the retrieval datastore**. An interaction with differential privacy that improves answer quality at a matched privacy budget would strengthen that contribution.

I did not identify a paper in this targeted search that reproduces the complete proposed experiment: the same grounded-versus-closed-book training comparison, adaptive active-server membership attack, training-record membership audit, datastore audit, and DP comparison in a single federated system. That supports investigating a specific gap. It does not establish priority or justify a “first” claim.

| Question | Assessment |
|---|---|
| Has the core training recipe been done? | Yes, closely: especially RA-DIT's language-model fine-tuning objective. |
| Has federated RAG fine-tuning been done? | Yes, including the FedRAG framework. |
| Has retrieval been studied as a privacy trade-off? | Yes; both reduced exposure of memorized training data and new datastore leakage are documented. |
| Is there a clearly new defense mechanism here? | Not yet established. |
| Could the complete study add useful knowledge? | Yes, if it isolates the training effect, uses credible adaptive attacks, and explains cross-channel differences. |
| Is the current study enough for broad privacy-preserving-LLM claims? | No. Its model, data, adversary coverage, and privacy accounting are too narrow. |

These are research judgments, not predictions of acceptance at a particular venue.

## 2. Closest academic precedents

The table distinguishes direct method overlap from related privacy work. Dates denote initial public versions unless a venue is given. Descriptions below concern the papers' methods and scope; their reported effectiveness is not independently reproduced here.

| Paper | Relevant prior contribution | Consequence for this proposal |
|---|---|---|
| **Lin et al., RA-DIT: Retrieval-Augmented Dual Instruction Tuning** (2023; ICLR 2024) | Section 2.3 conditions on background documents and instructions, with next-token loss restricted to the output segment. Some reading-comprehension examples use their supplied gold context. [Paper, §2.3](https://arxiv.org/html/2310.01352v3#S2.SS3) | The closest objective-level precedent. Context conditioning and answer-only loss cannot carry the novelty claim. RA-DIT also tunes a retriever; the proposal does not need that extra component to overlap with its generator-training method. |
| **Zhang et al., RAFT: Adapting Language Model to Domain Specific RAG** (2024) | Fine-tunes models on questions, relevant documents, distractors, and answers with reasoning and citations. [Paper](https://arxiv.org/abs/2403.10131) | “Teach the model to read documents” is established. The proposal's short extractive answers and lack of RAFT's reasoning targets are differences, but do not by themselves establish a new privacy mechanism. |
| **Fajardo et al., FedRAG: A Framework for Fine-Tuning Retrieval-Augmented Generation Systems** (ICML CODEML workshop, 2025) | Supports generator/retriever fine-tuning, including RALT, and conversion to federated training using client datasets and FedAvg. [Paper, §§2–4](https://arxiv.org/html/2506.09200v2) | Federating grounded training is already supported. Its framework contribution leaves room for a focused privacy evaluation of that training. |
| **Zeng et al., The Good and The Bad: Exploring Privacy Issues in RAG** (Findings of ACL 2024) | Studies retrieval-data extraction and shows that adding retrieval at inference can reduce the output of memorized training data. [Paper](https://aclanthology.org/2024.findings-acl.267/) | The broad privacy trade-off is known. An inference-time effect is different from demonstrating that a training objective changes stored information or update leakage. |
| **Fu et al., Ensemble Privacy Defense for Knowledge-Intensive LLMs against Membership Inference Attacks** (December 2025 preprint) | Compares membership attacks against SFT and RAG and proposes inference-time ensembling with a base model and judge. [Paper, §§3–4](https://arxiv.org/html/2512.03100v1) | RAG-versus-SFT privacy comparison and defenses spanning both settings are prior work. This paper does not supply the proposal's active-server FL experiment. |
| **Choi et al., Safeguarding Privacy of Retrieval Data against Membership Inference Attacks: Is This Query Too Close to Home?** (Findings of EMNLP 2025; Mirabel) | Uses unusually high query–document similarity to detect suspicious queries and hide a retrieved document. [Paper](https://aclanthology.org/2025.findings-emnlp.438/) | The gate belongs to an existing detect-and-hide family. Exact-span overlap and benign-loss calibration could be an incremental variant; no priority for that exact variant is established here. |
| **Naseh et al., Riddle Me This! Stealthy Membership Inference for RAG** (CCS 2025) | Infers datastore membership using natural questions specific to a document, without requiring verbatim document requests. [Paper](https://arxiv.org/abs/2502.00306) | A narrow verbatim-only audit is insufficient for a general datastore-protection claim. |
| **Nguyen et al., Five Queries Are Enough: Query-Efficient and Surrogate-Free Membership Inference Attacks on RAG via Entailment** (USENIX Security 2026; MEntA) | Uses ordinary information-seeking questions and entailment scoring, with five-query attacks and evaluation against RAG defenses. [Conference paper page](https://www.usenix.org/conference/usenixsecurity26/presentation/nguyen-nguyen) | A recent, relevant attack to add or use as a design reference for the library evaluation. |
| **Deng et al., Toward Efficient Membership Inference Attacks against Federated Large Language Models: A Projection Residual Approach** (2026; ProjRes) | A passive attack using candidate representations and server-visible gradient subspaces. [Paper](https://arxiv.org/abs/2604.21197) | Different gradient attacks use different information. One adapted cosine attack is not comprehensive evidence of update privacy. Check compatibility with the actual trainable layers and released tensors. |
| **Jiang and Shen, FISGuard: Defending Against Membership Inference via Fixed Input Subspaces** (August 2026 preprint) | Fixes a representation subspace derived from public data to reduce the gradient signal exploited by ProjRes; evaluates Adapter and LoRA tuning. [Paper](https://arxiv.org/abs/2608.27836) | A direct contemporary defense comparison for compatible update settings. Its mechanism differs from grounding. Do not equate its threat model with the proposal's active-server setting. |
| **Mao et al., Privacy-Preserving Federated Embedding Learning for Localized RAG** (2025; FedE4RAG) | Federates retrieval-model learning using knowledge distillation and homomorphic encryption. [Paper](https://arxiv.org/html/2504.19101v1) | Privacy-motivated federated RAG already exists. Its focus is retrieval embeddings and protected communication, rather than this generator-objective comparison. |
| **pFedRAG: A Personalized Federated RAG System with Depth-Adaptive Tiered Embedding Tuning** (Findings of EMNLP 2025) | Studies personalized federated embedding adaptation for retrieval. [Paper](https://aclanthology.org/2025.findings-emnlp.769/) | Additional architecture-level prior art; distinguish retriever adaptation from generator memorization. |
| **Grislain, RAG with Differential Privacy** (December 2024; revised 2025) | Develops differentially private token generation for RAG. [Paper](https://arxiv.org/abs/2412.19291) | Separate formal protection for the retrieval corpus is established research. |
| **Mori et al., Differentially Private Synthetic Text Generation for RAG** (2025; accepted to Findings of ACL 2026; DP-SynRAG) | Creates a reusable DP synthetic retrieval database, avoiding an additional privacy cost for every subsequent query to that released database. [Paper](https://arxiv.org/abs/2510.06719) | A stronger formal datastore-protection comparator than an overlap heuristic, although answering unique private facts may present a different utility trade-off. |
| **Jiang et al., RAPID: Retrieval Augmented Training of Differentially Private Diffusion Models** (ICLR 2025) | Uses public retrieval during private diffusion-model training to improve utility under DP. [Paper](https://arxiv.org/abs/2502.12794) | An adjacent precedent for retrieval improving private training. It concerns image diffusion, not an exact LLM/FL duplicate. |

## 3. The main conceptual correction

The proposal says that private facts remain in the library and the model learns only a reading skill. Its specified training procedure does not enforce this separation.

For a passage c, question q, and answer a, the loss is:

`L(theta; c, q, a) = -log p_theta(a | c, q)`.

Masking context labels removes direct prediction targets for the passage. The answer prediction still depends on passage representations, and the parameter gradient still depends on c and q. Private answers are also explicit prediction targets. Consequently, the model can retain information about the passage, question, or answer, and the update can reveal information about them.

This is an analysis of the proposed objective, not a claim that grounded training must leak more. The possible reduction is plausible but empirical. The claim to investigate is: **grounding may reduce the incentive to memorize answers and change which privacy attacks work**.

Also distinguish:

- **Membership inference:** whether a known candidate record was used.
- **Extraction:** whether an attacker can recover previously unknown content.
- **Context reliance:** whether changing or removing retrieved evidence changes the answer.

Lower membership AUC on the chosen attack establishes neither absence of stored facts nor resistance to extraction. Add a controlled extraction or synthetic-secret test if retaining the title's claim about keeping facts out of weights.

## 4. Experimental changes needed before confirmation

### 4.1 Isolate grounding from loss masking and prompt changes

CB currently uses the old question–answer format; RG introduces context, a chat-format prompt, and answer-only loss. If CB trains on question tokens as well, a reduction in membership leakage could be caused by masking rather than grounding.

At minimum compare CB and RG with identical answer-only supervision and a matched prompt structure. Preserve the old CB as a diagnostic baseline. A context-present/absent × full/answer-only-loss ablation is informative, but its primary controlled contrast should change only the presence of context. Report record exposures, answer truncation, update counts, clipping behavior, and token/computation cost.

RG's description supplies the gold SQuAD paragraph. That is an oracle-context condition. It is not yet evidence that the method works with an imperfect retriever. Include retrieved top-k context with distractors or explicitly limit the claim to training on supplied passages.

### 4.2 Make the private-training decision meaningful

Keep P0: pretrained model plus RAG. Add a model trained to read using **public-only** QA/context examples and evaluated on the same private library. This directly tests whether private examples are necessary to learn the useful reading behavior.

If public training or P0 achieves comparable quality with lower exposure from private fine-tuning, that is a useful deployment result. RG failing to beat P0 should stop unnecessary expensive confirmation of a utility-improvement claim; it need not make the whole research question “moot.”

### 4.3 Align the DP claim with the actual experiment

H2 currently tests utility and attack AUC at one selected noise multiplier. It does not test the headline claim that RG “needs less noise.”

For standard DP-SGD accounting, privacy depends on the protected unit, sampling process, noise multiplier, number of steps, and delta. Changing the language-model objective alone does not automatically improve the formal bound when these quantities are fixed. This follows from the mechanism/accounting framework of [Abadi et al., Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133).

Choose one clear claim:

1. **Better utility at a matched formal privacy budget:** compare F1 at the same accounted epsilon and delta.
2. **Less noise against specified attacks:** compare complete noise–utility–attack curves, describing the result as empirical resistance.

AUC near 0.5 for a limited attack battery is not a DP guarantee. Publish epsilon/delta and the accounting assumptions for every claimed DP channel. Include all releases visible to the server, including active queries and repeated rounds. The client must enforce the mechanism and its release budget under the claimed malicious-server threat model.

### 4.4 Strengthen the library audit and test the full gate behavior

Run at least one natural-question attack, such as IA or MEntA, in addition to the verbatim yes/no probe. Calibrate against realistic benign queries, including legitimate quotations. Attack the defended system itself: document hiding can change responses or refusals in a way that reveals membership.

Measure answer F1/EM with the gate enabled. Gold-document retention is a retrieval diagnostic, not a substitute for final answer quality. A gate calibrated to 5% removal can still lose more than 5% utility on important or difficult questions.

### 4.5 Measure interactions if claiming a shift of privacy risk

With T and L disjoint, the study measures privacy on separate collections. It cannot directly show where the same fact's risk goes. If the central claim concerns shifting risk between weights and a datastore, make a dedicated four-cell experiment part of the core design:

| | Absent from retrieval store | Present in retrieval store |
|---|---|---|
| Absent from training | Neither source | Retrieval only |
| Present in training | Training only | Both sources |

Use controlled target placement and consistent questions. Evaluate model-only attacks with retrieval disabled to attribute signals to the weights, then enable retrieval to measure its added effect. Otherwise, describe the result as separate channel measurements.

### 4.6 Resolve the data-split specification

Section 4 requires no shared articles or passages across T, L, V, and F, but F contains questions about library passages. Their supporting documents must be available in the evaluation library if the goal is answerable RAG questions.

Define document partitions and question-use partitions separately. A final question's supporting document may reside in the frozen evaluation library while the question/answer is withheld from training and tuning. State this intended relationship explicitly. Article separation reduces sibling-question leakage; it does not prove that no fact is repeated across different articles.

SQuAD's public content also cannot establish that a fact was absent from pretraining. Treat it as a simulated private benchmark, and use synthetic private facts or a suitable additional corpus for stronger memorization claims.

### 4.7 Repair overlapping decision rules

H1 can currently be both supported and refuted. An AUC-reduction interval of [0.005, 0.015], with utility passing, has a lower bound above zero and an upper bound below 0.02. H4 has the same issue. H3 can also meet both rules with an interval such as [0.01, 0.04].

Separate statistical improvement from practically meaningful improvement, and make decision categories mutually exclusive. Freeze one coherent threshold scheme before confirmation. If making an overall claim across four hypotheses, specify how multiplicity is handled.

### 4.8 Keep attack calibration and sampling units explicit

Scoring members against held-out nonmembers in one trained world is a useful conventional audit, but it estimates a different quantity from changing the same candidate's inclusion across paired worlds. Randomize membership placement, match distributions and record characteristics, and retain a smaller paired-world check if making a causal inclusion claim.

AUC should be accompanied by operating-point results with enough nonmembers to estimate the chosen false-positive rate. Low-FPR evaluation is emphasized by [Carlini et al., Membership Inference Attacks From First Principles](https://arxiv.org/abs/2112.03570). Do not report 0.1% or 1% FPR performance from a cohort too small to resolve it.

Bootstrap units should follow the desired generalization claim. A world-level bootstrap estimates variation across worlds; 300 questions do not create 300 independent training runs. If questions or documents are fixed across worlds, state whether uncertainty is conditional on that test set or also covers new documents.

## 5. What the existing local evidence supports

The proposal's linked reports help justify a study, but do not establish the new defense's effectiveness:

- The [causal-attack result](../causal_attack_validation_20260929/results.md) reports AUC 0.992 in a narrow setting with candidate-known batch membership. Its nominal 5% calibration produced 16.5% private FPR. Carry the observed calibration limitation into the new protocol.
- The [utility control](../guard_v2_utility_review_20260924/analysis.md) explicitly notes that the empty-context prompt still asks the model to use context and that some questions depend on an absent paragraph. Its low scores do not isolate model capacity or prove inability to memorize SQuAD facts.
- The [Mirabel calibration](../mirabel_calibration_20260929/README.md) reports a severe retrieval/utility trade-off in this setup. That motivates a comparison, but does not establish that Mirabel fails generally or that the replacement gate protects natural-question traffic.

These are findings reported by the local documents. I read those reports; I did not rerun the experiments or independently validate their underlying model artifacts.

## 6. Where the study can add value

Three possible findings could matter:

1. **An improved privacy–utility trade-off.** Grounded training preserves more answer quality at matched formal DP budgets, after controlling for loss masking and training exposure.
2. **Different outcomes across channels.** Model membership falls while gradient or datastore leakage persists or increases. This would explain why apparent privacy gains at the output are insufficient for a federated deployment.
3. **Private fine-tuning is unnecessary in the tested regime.** Public grounded training or an unchanged pretrained model matches useful performance. That gives practitioners a reason to avoid an unnecessary source of private-data exposure.

The second finding is especially well aligned with the proposal's existing attack infrastructure. A well-supported negative result can be a contribution. Running several familiar components together without establishing an interaction or useful conclusion is less compelling.

The current Qwen2.5-0.5B/SQuAD/four-client configuration is a pilot. For a broader empirical paper, prioritize a second model size, another data setting with controlled private facts, and at least one meaningful variation in client heterogeneity or training exposure. Choose these after the pilot identifies which effect needs explanation.

## 7. Suggested positioning

**Suggested title:** *Privacy Trade-offs in Retrieval-Grounded Federated Fine-Tuning: Model, Update, and Datastore Leakage.*

**Suggested contribution statement:**

We investigate how conditioning federated language-model fine-tuning on supporting documents changes training-record membership leakage, server-visible update leakage, and retrieval-datastore membership leakage. Using controlled supervision, adaptive attacks, and explicit differential-privacy accounting, we test whether grounded training improves answer quality at matched privacy budgets and whether privacy gains persist across these channels.

That statement describes a testable study without assuming that facts remain outside model weights. A claim of a new mechanism would require an additional substantive method contribution and its own prior-art review.

## 8. Search scope and confidence

This was a targeted novelty review, not an exhaustive systematic review. Searches covered retrieval-augmented fine-tuning and memorization, federated RAG training, membership inference, private retrieval, and DP training, including 2025–2026 work. Primary sources were checked on arXiv, ACL Anthology, OpenReview, and USENIX. Relevant method sections were examined for RA-DIT, RAFT, FedRAG, EPD, Mirabel, IA, FISGuard, and FedE4RAG; some additional entries were screened at the abstract or official conference-page level.

Confidence is high that the core objective and federated architecture have prior art. Confidence is moderate that the exact combined experiment is an open opportunity: failing to find an exact duplicate cannot establish that none exists. Indexing delays, terminology differences, and inaccessible literature remain limitations.

**Final recommendation:** Continue as a focused privacy-evaluation study, after revising the mechanism claims and core comparisons. Use the pilot to decide which result deserves the larger confirmation budget.
