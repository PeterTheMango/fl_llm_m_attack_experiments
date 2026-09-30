# Does Retrieval Grounding Keep Private Facts Out of the Model? Leakage Through Weights, Client Updates and the Datastore in Federated RAG

*Short title: Read, Don't Memorize? — A Controlled Study of Grounded Federated RAG*

Status: **draft proposal, revision 2 (after review), awaiting the researcher's approval.** Nothing is implemented or run.

- **Decided on 2026-09-30:** the training/library overlap set is included, and is now the central design (decision 3). There are two trained models per target (decision 4). The final stage has 20 targets (decision 8), which this revision's budget options revisit.
- **Still open:** the rest, including the title.
- **Review response:** [§0](#0-changes-in-this-revision) lists every review point and how it was addressed.

---

## Summary in plain terms

Today the federated clients teach the model to memorize private question–answer pairs, and the document library (RAG) is only added when the model answers. We want to know what happens to privacy if clients instead teach the model to **answer from a document**. This is "grounded" training, and it is not a new training method: RA-DIT and FedRAG already do versions of it.

What nobody has measured, as far as our search found, is **where the private information goes** when training is grounded. There are three places it can leak:

1. **The model's weights**, which can reveal whether a record was trained on.
2. **The updates clients send to the server**, which a malicious server can attack.
3. **The document library**, which can reveal whether a document is stored in it.

Grounding might reduce leakage in one place and increase it in another. For example, the model might memorize less from its training answers but become better at revealing what is in the library. It might also still soak up paragraph content during training, even though it is never graded on the paragraph.

We test this with careful comparisons. Every target record is placed in training or not, and in the library or not, giving four combinations. We also compare three training setups, all graded the same way:

- training without a paragraph;
- training with a private paragraph;
- training on public data only.

Then we compare usefulness at **equal, stated privacy budgets**.

---

## 0. Changes in this revision

| Review point | Change |
|---|---|
| The novelty claim overlaps RA-DIT (document-conditioned training, output-only loss) and FedRAG (federated RAG fine-tuning) | **The method is not claimed as new.** The contribution is a **controlled study of how grounding changes leakage across three channels**: weights, client updates and datastore. Our search found no exact duplicate of that combined experiment. It does not establish "first", and no such claim is made (§2). |
| Reduced memorization must be a hypothesis | **Now stated as hypotheses only.** Masking passage tokens from the loss does not stop their information from shaping gradients and weights, so this is tested directly (H4). |
| A closed-book baseline with identical answer-only supervision and prompt structure is needed | **CB-AO added** as the primary comparison. It uses the same prompt template and answer-only loss, with an empty context slot. The only difference from RG is whether the passage is present. Today's full-loss closed-book training (CB-LM) is kept as a pilot bridge only. |
| Keep pretrained + RAG | **P0 retained.** |
| Test whether private training is necessary | **RG-public added:** grounded fine-tuning on public data only (H5). |
| A single noise level cannot establish "needs less noise" | **Replaced by comparisons at matched, explicitly accounted DP budgets,** ε ∈ {∞, 64, 16} at δ = 10⁻⁵, with ε reported per arm (H6). The claim is reworded to "better utility at equal ε". |
| The datastore evaluation needs natural-question attacks | **Two natural-question attacks added,** adapted from Riddle Me This and MEntA. The verbatim yes/no attack is kept as a weak baseline (§6). |
| The overlap experiment should be central if claiming a channel shift | **The four-cell design (training × library) is now the core design.** Every primary endpoint is defined in its cells (§5). |
| Ambiguity in the final-question/library split | **Resolved** with explicit groups and one-question-per-passage rules (§4). |
| Overlapping hypothesis decision rules | **Replaced** with mutually exclusive, exhaustive outcome rules and a multiplicity adjustment (§9). |
| Revise controls before committing the full budget | **The Stage 2 budget is approved separately,** after Stage 1 timings and variances (§10). |

## 1. Research question

**Main question.** When federated clients fine-tune with retrieved context (grounded training) rather than closed-book, how does private-information leakage change through each channel (the model weights, the client updates and the retrieval datastore), and at what cost in answer quality under matched, accounted privacy budgets?

| Sub-question | Hypothesis (a claim only if supported) | Primary endpoint |
|---|---|---|
| **Q1. Weights (training records)** | **H1.** RG leaks less training-record membership than CB-AO, with non-inferior RAG answer quality | Reference-attack AUC; RAG F1 |
| **Q2. Client updates** | **H2.** RG changes leakage through client updates compared with CB-AO. Two-sided: any direction, or no change, is a result | Causal gradient attack AUC, strongest attacker score |
| **Q3. Datastore** | **H3.** RG makes library documents *more* detectable than CB-AO, because a model trained to use context reveals it more faithfully | Natural-question RAG membership AUC, library-only cell |
| **Q4. Passage information in the weights** | **H4.** Although passage tokens are masked from the loss, RG still stores passage information in the weights. Questions about a trained-on passage become answerable without retrieving it | Natural-question membership AUC in the training-only cell |
| **Q5. Is private training needed?** | **H5.** Grounded fine-tuning on public data only (RG-public) gives RAG answer quality non-inferior to RG on private data | RAG F1 |
| **Q6. Utility at equal privacy budget** | **H6.** At matched ε, RG answers better than CB-AO, with no worse training-record leakage | RAG F1 and Reference AUC at ε = 16 |

**"Channel shift"** is a pattern claim: leakage falls in one channel and rises in another. It is only claimed if H1 is supported and at least one of H3 or H4 is supported. Nothing is averaged across channels.

## 2. Related work and novelty

- **The method already exists.**
  - [RA-DIT](https://arxiv.org/abs/2310.01352) fine-tunes the LM on instructions prefixed with retrieved chunks and minimizes the loss on the output segment only.
  - [FedRAG](https://arxiv.org/abs/2506.09200) supports federated fine-tuning of RAG components.
  - This study uses grounded training as the treatment. It does not propose it.
- **Relevant privacy findings.** [Zeng et al. (ACL Findings 2024)](https://aclanthology.org/2024.findings-acl.267.pdf) found that retrieval reduces an LLM's output of memorized training data at inference. RAG membership attacks include:
  - verbatim and templated probes ([Mirabel](https://arxiv.org/abs/2505.22061) defends against these);
  - natural-question attacks: [Riddle Me This](https://arxiv.org/abs/2502.00306) and [MEntA](https://arxiv.org/abs/2605.24312).
- **What may be new is the combined experiment.** It crosses grounding against training membership against library membership, under active FL attacks and RAG membership attacks, with matched, accounted DP budgets. Our search found no exact duplicate, but it was not systematic. A structured related-work search is a Stage 0 deliverable, and no "first" claim is made.

## 3. Arms

All arms share the same model, FL setting, questions, prompt template and answer-only loss wherever training happens.

| Arm | Training data | Prompt at training | Loss | Role |
|---|---|---|---|---|
| **RG** (grounded, private) | Private client triples (passage, question, answer) | RAG inference template (`rag._prepare_prompt`, chat format) with the passage in the context slot | Answer tokens only | Treatment |
| **CB-AO** (closed-book, answer-only) | The same questions and answers, no passage | The **same template** with an **empty** context slot | Answer tokens only | Primary comparison: differs from RG only by the passage |
| **RG-public** | Public triples only, from separate articles; no private data | As RG | Answer tokens only | Tests whether private training is needed. No private client updates exist; trained once per seed, not per target |
| **P0** | None (pretrained) | — | — | Floor and reference point; evaluation only |
| CB-LM (pilot bridge) | Today's closed-book records | Today's `"Question: …\nAnswer: …"` | Full LM loss | Links to earlier results. Stage 1 only |

**DP variants.** RG and CB-AO are each trained at ε ∈ {∞, 64, 16} (§7). RG-public and P0 contain no private training data.

## 4. Data design (resolving the split)

The source is the SQuAD train split, partitioned **by article**. Each passage contributes **at most one question** to any training set. That way a passage's other ("sibling") questions stay unused and can serve as natural-question probes.

| Group | Articles | Contents | Used for |
|---|---|---|---|
| **T** (private training) | Set A | One (passage, question, answer) per passage, spread over 4 clients × 32 records | RG and CB-AO client data; the **target records** for training-side attacks |
| **T-hold** | Set B | Same form as T | The non-member record that replaces the target in each non-member world |
| **U** (public training) | Set C | Same form | RG-public training only |
| **L** (private library) | Set D | 256 passages | The private datastore. Its passages' questions become **final utility questions (F)** |
| **N** (non-member documents) | Set E | Passages never placed in any library | Non-member documents for RAG membership attacks, with their SQuAD questions as probes |
| **P** (public library) | Set F | 256 passages, with their questions | The public datastore and public-library utility questions |
| **V** (tuning) | Set G | A smaller copy of all the above, plus a separate public slice for attacker calibration | Every tuning choice and every pilot |

**Rules that resolve the old ambiguity:**

- **F consists of questions about L passages.** L and F therefore share articles on purpose, and no F question is ever used for training or tuning. T, T-hold, U, N, P and V are article-disjoint from L and from each other.
- **Probes come from siblings.** The natural-question probes for a document are its unused sibling SQuAD questions. They never overlap F: probe documents are the target passages (from T) and N passages, never L passages.
- **The overlap set is the one planned exception to disjointness.** For each target, its **own passage** is either inserted into the private library or left out, which is the library switch. That is the only way a T passage enters L.
- **Checks before any GPU job:**
  - no shared article across groups, except that planned exception;
  - one question per passage in each training set;
  - `validate_partition_tokens` passes;
  - every earlier target question is excluded (guard v3/v4, causal Stage A/B, matched controls and the reserved finals, matched by recomputed old record hashes).
- **The reserved-final cohort stays closed.**

## 5. Core design: the four-cell overlap experiment

For each target record *t* from T, and for each training arm (RG and CB-AO, at each ε):

|  | *t*'s passage **in** library | *t*'s passage **not** in library |
|---|---|---|
| ***t* in training** (member world) | **both** | **training-only** |
| ***t* not in training** (non-member world: *t* replaced by a T-hold record) | **library-only** | **neither** |

- **Two FL worlds are trained per target** (decision 4). The library switch needs no retraining. This is the existing `membership_overlap` mechanism.
- **Weights (H1):** the Reference attack, training-only against neither.
- **Client updates (H2):** the causal attack in the member world.
- **Datastore (H3):** the natural-question attack, library-only against neither.
- **Passage information in the weights (H4):** the natural-question attack, training-only against neither. For CB-AO the model never saw the passage. For RG it saw the passage as context, masked from the loss.
- **Compounding (secondary):** the "both" cell against the stronger of the two single-channel cells.

## 6. Attacks

| Channel | Attack | Standard score | Adaptive score (attacker knows the training format) |
|---|---|---|---|
| Weights | **Reference** (Carlini log-perplexity ratio against the pretrained model) | Loss of the record text | Answer loss given the template and, for RG, the passage |
| Client updates | **Causal gradient alignment** (validated: cosine score, `probe_epochs: 12`) | The record's LM-loss direction | The answer-only loss direction given the template and passage |
| Datastore | **Natural-question attack, adapted from Riddle Me This.** Ask the document's sibling questions; the membership score is answer correctness (token F1 against the gold answer) | — | — |
| Datastore | **Entailment attack, adapted from MEntA.** About 5 queries per document; score whether the responses are entailed by the document, using a pinned NLI model | — | — |
| Datastore (baseline) | Verbatim yes/no probe (existing) | — | — |

- **Labels.** The two natural-question attacks are **adaptations**, not reproductions, and are labelled that way. Each must pass an adaptation check on V before use: members must beat non-members with P0 and the library on, following the repo's `adapt-attack` workflow.
- **Choosing the attacker's score.** For each arm, the attacker's score (standard or adaptive, and for the datastore the stronger of the two natural-question attacks) is fixed on the **tuning** cohort by a pre-set rule: highest mean AUC, ties to the simpler score. It is never chosen on final data.
- **Release noise (secondary).** The causal attack runs a second time on the same member world, with σ_obs = 1.0 noise on the released update. This costs attack trials only, and is the first test of release noise on the validated attack.
- **Library gate (secondary).** The verbatim-overlap gate is evaluated against every datastore attack. It is expected to stop verbatim probes but not natural questions.

## 7. Matched, explicitly accounted DP budgets

- **Mechanism.** DP-SGD with per-example clipping and Gaussian noise at each local step (`defenses.private_train`). The answer-only mask applies in the RG and CB-AO paths.
- **Accountant, as implemented** (`defenses.privacy_bound`): Gaussian zCDP, **no subsampling amplification**, replace-one adjacency, counting every local step of the busiest client. The target's client holds 33 records (32 plus the target). With batch 2, 1 local epoch and 3 rounds, that is 51 steps. Where ρ = 2 · 51 / σ²:
  - ε = ρ + 2√(ρ ln(1/δ))

| Target ε (δ = 10⁻⁵) | Required σ | Note |
|---|---:|---|
| ∞ | 0 | No DP |
| 64 | 1.91 | About the pilot's σ = 2 (ε ≈ 60) |
| 16 | 5.45 | The primary matched budget for H6 |

- **Same budget, same noise.** RG and CB-AO have identical record counts, batch size and steps, so matched ε means the same σ. Each run records its accounted ε, and the analysis refuses a comparison if the two differ.
- **The accountant is conservative.** A record takes part in only 3 of the 51 steps. A participation-based accountant would give much smaller ε for the same σ. It is **not** used unless the researcher approves it separately (decision 5).
- **Scope.** DP-SGD covers only the weights and the training updates, per training record. It does **not** protect the datastore, or updates released to a malicious server's crafted requests; release noise and the gate cover those. There is no end-to-end ε.

## 8. Stages

**Stage 0. Build and check (mostly CPU, 3–5 GPU smoke jobs).**

- **Structured related-work search**, recorded as a document.
- **Data:** build the split and its audit (§4).
- **Training code:** the RG, CB-AO and RG-public formatters with the answer-only mask, in both the normal and the DP-SGD paths. Prompt-parity tests against `rag._prepare_prompt`.
- **Attacks:** the adaptive attack scores; the natural-question and entailment attack adaptations, with their checks on V.
- **Library:** gate calibration (CPU).
- **Go/no-go:**
  - On V, RG must beat P0 on RAG F1. If grounded training teaches nothing, stop and report that.
  - The natural-question attacks must pass their adaptation check.
- **Timings:** training at sequence length 384 (median passage 105 words, 90th percentile 145), DP-SGD overhead, and a full job.

**Stage 1. Pilot (tuning seed band 8000–8099; V data).**

- **Scope:** 4 targets × {RG, CB-AO} × ε ∈ {∞, 64, 16}, plus a CB-LM bridge on 4 targets, RG-public (3 seeds) and P0.
- **Fix the attacker scores** by the §6 rule.
- **Estimate variances** and re-estimate the Stage 2 cost.
- **Pilot results** are tuning data and are never reported as evidence.
- **Gate:** the researcher approves the Stage 2 budget from the pilot's timings.

**Stage 2. Confirmation (final seed band 9000–9099; frozen `protocol.json`).**

- The approved design option (§10), paired: every target runs under every arm it includes.

## 9. Endpoints and decision rules (Stage 2)

**Setup.**

- The unit is the target, with paired bootstrap over targets (10,000 resamples).
- **Multiplicity:** six primary hypotheses. Decisions use Bonferroni-adjusted **99.2%** intervals (1 − 0.05/6). 95% intervals are also reported.
- Every difference *D* is oriented so that a positive value is the hypothesis's direction. The smallest effect of interest is **m = 0.05** for AUC and **0.05** for F1.

**Outcome rules** are mutually exclusive and exhaustive. Here [lo, hi] is the decision interval.

- **Directional hypotheses (H1 privacy part, H3, H4, H6 utility part):**
  - *supported* if lo > 0;
  - *not supported, negligible* if lo ≤ 0 and hi < m;
  - *inconclusive* if lo ≤ 0 and hi ≥ m.
  - Within "not supported", the case hi < 0 is additionally flagged as *reversed*.
- **Two-sided hypothesis (H2),** with *D* = AUC(CB-AO) − AUC(RG):
  - *RG reduces* if lo > 0;
  - *RG increases* if hi < 0;
  - *no meaningful change* if −m < lo ≤ 0 ≤ hi < m;
  - *inconclusive* otherwise.
- **Non-inferiority (utility parts of H1 and H6, and H5),** margin 0.05 F1:
  - *non-inferior* if lo > −0.05;
  - *inferior* if hi < −0.05;
  - *inconclusive* otherwise.

| Hypothesis | *D* | Claimed when |
|---|---|---|
| **H1** | AUC_Ref(CB-AO) − AUC_Ref(RG), training-only vs neither, ε = ∞; plus F1(RG) − F1(CB-AO) on F | Privacy part supported **and** utility part non-inferior |
| **H2** | AUC_causal(CB-AO) − AUC_causal(RG), member world, ε = ∞ | Any of the four outcomes is reported as the result |
| **H3** | AUC_NQ(RG) − AUC_NQ(CB-AO), library-only vs neither, ε = ∞ | Supported |
| **H4** | AUC_NQ(RG) − AUC_NQ(CB-AO), training-only vs neither, ε = ∞ | Supported |
| **H5** | F1(RG-public) − F1(RG), on F | Non-inferior |
| **H6** | F1(RG) − F1(CB-AO) at ε = 16; plus AUC_Ref(CB-AO) − AUC_Ref(RG) at ε = 16 | Utility part supported **and** privacy part ≥ −m at its lower bound (no worse) |

- **Secondary, descriptive:**
  - ε = 64, and the full frontier (F1 and each attack's AUC against ε);
  - the causal attack under release noise;
  - the gate;
  - the "both" cell;
  - public-library F1;
  - no-context F1 and answer NLL (utility protocol v4);
  - P0 against every arm;
  - the CB-LM bridge;
  - per-client results.
- **Extending the study** needs a new pre-registration. Targets are never added after seeing results.

## 10. Budget and rough cost

These are estimates, replaced by Stage 0 timings. Assumptions:

- A paired job trains **two** FL worlds at sequence length 384. It also runs the RAG evaluation once (utility and membership probes, with the library on and off) and the causal attack twice (raw and noised). Estimate: **30–60 min** per job without DP; DP-SGD adds an unmeasured overhead.
- RG-public (3 seeds) and P0 need a few GPU-hours of training and evaluation in total.

| Stage | Jobs | Est. GPU-hours |
|---|---|---|
| 0: smoke and timing | 3–5 | 3–5 |
| 1: pilot (4 targets × 6 arms, plus 4 CB-LM) | about 28 | 14–28 |
| 2, option (a): 20 targets × 6 arms (RG and CB-AO at 3 budgets) | 120 | 60–120 |
| 2, option (b): 20 targets for the ε = ∞ arms (H1–H4) and 10 targets for the DP arms (H6) | 80 | 40–80 |
| RG-public and P0 | — | about 5 |
| **Total** | | **(a) about 80–160 · (b) about 60–120** |

Option (b) keeps 20 targets (decision 8) for the central channel study and halves the DP arms, at the cost of a wider H6 interval. **The Stage 2 budget is approved only after Stage 1.**

## 11. What this study cannot show

- **One model** (Qwen2.5-0.5B-Instruct), **one data set** (SQuAD), **one FL setting** (4 clients, 3 rounds). There are no generalization claims.
- **The natural-question attacks are adaptations.** Stronger or adaptive datastore attackers may exist.
- **The ε values are large,** because the accountant has no amplification. They are comparable *between arms*, but they are not strong absolute guarantees. There is no end-to-end ε.
- **No new defense mechanism is claimed.** The contribution is measurement. A defense built on its findings (for example, library-side protection sized by H3/H4) would be a follow-up study.
- **The guard/detector line is closed** and is not part of this study.

## 12. Implementation work (after approval)

1. **Data:** split builder and audit (§4), plus the old-target exclusion by SQuAD question.
2. **Training:** the RG, CB-AO, RG-public and CB-LM formatters; the answer-only mask in the AdamW path and in `private_train`; prompt-parity tests.
3. **RAG study data:** new study data from L, N and P: the libraries, the F questions, and the probe sets.
4. **Attacks:**
   - adaptive Reference and causal scores;
   - the natural-question and entailment attack adaptations, with an NLI model pinned by revision;
   - the second causal pass with release noise.
5. **Accounting:** a per-run ε record, and a refusal in the analysis when the ε of paired arms differs.
6. **Library gate:** the gate and its calibration.
7. **Study tool** (`prepare`, `check`, `analyze`) and tests, following the earlier studies. `check` never shows endpoint values before `analyze`.
8. **Fingerprint.** The core fingerprint will change. Earlier results keep theirs.

## Decisions to approve

1. **Title and main question** (§1). The title is now phrased as a question.
2. **Arms** (§3): RG, CB-AO, RG-public, P0, and CB-LM as a Stage 1 bridge only.
3. **Data design** (§4): the groups, one question per passage, siblings as probes, F drawn from L, and the overlap exception. *(Overlap set already decided: included.)*
4. **Core design** (§5): the four cells with two worlds per target. *(Already decided.)*
5. **DP budgets** (§7):
   - ε ∈ {∞, 64, 16} at δ = 10⁻⁵, with the existing conservative accountant;
   - ε = 16 as the primary budget for H6;
   - whether to also approve a participation-based accountant (recommended: not now).
6. **Attacks** (§6): the two natural-question adaptations and their checks, the attacker-score rule, release noise σ_obs = 1.0, and the gate as secondary.
7. **Endpoints and rules** (§9): the six primaries, the 99.2% decision intervals, m = 0.05, and the exclusive outcome rules.
8. **Budget** (§10): option (a) or (b), with the Stage 2 budget approved after the Stage 1 pilot.
