# Does Retrieval Grounding Keep Private Facts Out of the Model? Leakage Through Weights, Client Updates and the Datastore in Federated RAG

*Short title: Read, Don't Memorize? — A Controlled Study of Grounded Federated RAG*

Status: **draft proposal, revision 3 (after the second review), awaiting the researcher's approval.** Nothing is implemented or run.

- **Decided on 2026-09-30:**
  - the training/library overlap set is included and central (decision 3);
  - two trained models per target (decision 4);
  - 20 final targets (decision 8), with the options in §10.
- **Still open:** everything else, including the title.
- **Review responses:** §0.

---

## Summary in plain terms

Today the federated clients teach the model to memorize private question–answer pairs, and the document library (RAG) is only added when the model answers. We study what happens to privacy if clients instead train the model to **answer from a document** ("grounded" training). This training method is not new: RA-DIT and FedRAG already use it. What we measure is **where private information ends up**:

1. **In the model's weights.** Does the model reveal which records it trained on? Does it keep facts from the training paragraphs, even ones it was never asked about?
2. **In the updates clients send to the server.** Can a malicious server tell what was in a client's batch?
3. **In the document library.** Can someone tell whether a document is stored there?

Every target record goes into training or not, and into the library or not. This gives four combinations, and every main result is measured within them. The grounded training arm is compared with closed-book training graded the same way, with a public-data-only version, and with no fine-tuning at all. Usefulness is compared at equal, stated privacy budgets.

---

## 0. Review responses

### Revision 3 (second review)

| Review point | Change |
|---|---|
| H4 needs a direct test of passage-fact retention; a membership AUC cannot establish factual memorization | **H4 is now a closed-book fact-retention endpoint.** Without retrieval, the model answers held-out sibling questions about facts **absent** from the training question–answer pair. We compare the paired member and non-member worlds, with CB-AO as the control: a difference-in-differences in F1 (§6.4). The membership AUC is secondary. |
| "Channel shift" must not rest on H1 and H4, since both concern weights | **Revised.** A cross-channel claim needs H1 (less weight leakage) **and** more leakage through updates (H2 "RG increases") **or** through the datastore (H3). H4 is reported as a separate weights finding (§1). |
| Specify how document scores, paired worlds, non-member documents, pooled AUC and bootstrap samples are built; siblings are not independent documents | **Specified exactly** (§6, §9.1). Sibling probes are averaged into **one score per document**. The bootstrap resamples **targets**, never probes. |
| Separate statistical improvement, meaningful improvement, equivalence and uncertainty; "negligible" could include substantial harm | **Replaced** by a first-match classification with six outcomes (§9.2). Harm is its own outcome and can never be labelled negligible. |
| The natural-question adaptations need numerical validation criteria, fixed query budgets, refusal handling and sibling-count rules, validated separately for datastore and training-only membership | **Added** (§6.3, §8): 3 queries per document per condition; refusals score 0 in the primary analysis; eligibility needs 3 qualifying siblings; separate numerical validation criteria for the datastore (P0) and for training-only membership (a deliberate-memorization positive control). |
| H5 must specify the RG privacy budget and account for shared public-training seeds | **Specified.** The primary comparator is RG at ε = ∞, the case most favourable to private training, with ε = 64 and 16 as secondary. RG-public has 5 seeds, and the seed is its unit in a crossed bootstrap (§9.1). |
| Replace "RG must beat P0" with a predefined branch | **Replaced** by Branch A / Branch B (§8). Branch B reports "private fine-tuning is unnecessary for utility in this setting" as a finding and still measures leakage. |
| Limit the matched DP guarantee to the accounted training releases | **Stated explicitly** (§7). |

### Revision 2 (first review), kept for the record

- **Novelty.** The method-novelty claim was withdrawn (RA-DIT, FedRAG). The contribution is a controlled three-channel leakage study.
- **Arms.**
  - Added: CB-AO (identical answer-only loss and prompt) and RG-public.
  - Retained: P0.
- **Other changes.**
  - Matched, accounted DP budgets.
  - Natural-question datastore attacks.
  - The overlap design made central.
  - The split made explicit.
  - The Stage 2 budget approved only after the pilot.

## 1. Research question and hypotheses

**Main question.** When federated clients fine-tune with retrieved context (grounded training) rather than closed-book, how does private-information leakage change through each channel (the model weights, the client updates and the retrieval datastore), and at what cost in answer quality under matched, accounted privacy budgets?

| | Hypothesis (claimed only if supported, §9.2) | Primary endpoint (built as in §6) |
|---|---|---|
| **H1. Weights: record membership** | RG leaks less training-record membership than CB-AO, with acceptable RAG answer quality | Reference pooled AUC (training-only vs neither); RAG F1 on F |
| **H2. Client updates** | RG changes update leakage compared with CB-AO. Two-sided | Causal attack mean per-target AUC (member world) |
| **H3. Datastore** | RG makes library documents more detectable than CB-AO | Natural-question pooled AUC (library-only vs neither) |
| **H4. Weights: passage-fact retention** | RG stores facts from training passages that the training question–answer pair did not contain | Closed-book retention difference-in-differences in F1 (§6.4) |
| **H5. Is private training needed?** | RG-public gives RAG answer quality acceptable relative to RG at ε = ∞ | RAG F1 on F |
| **H6. Utility at an equal budget** | At matched ε = 16, RG answers better than CB-AO, with training-record leakage no worse | RAG F1 on F; Reference pooled AUC at ε = 16 |

**Cross-channel ("channel shift") claim.** This is claimed only if both of these hold:

- H1 is supported: weight leakage falls;
- **and** either H2's outcome is "RG increases" (update leakage rises) or H3 is supported (datastore leakage rises).

H4 is a second *weights* finding. It is reported separately and never counts toward a cross-channel claim. Channels are never averaged.

## 2. Related work and novelty

- **The method is not claimed as new.**
  - [RA-DIT](https://arxiv.org/abs/2310.01352) fine-tunes the LM on instructions prefixed with retrieved chunks, minimizing the loss on the output segment only.
  - [FedRAG](https://arxiv.org/abs/2506.09200) supports federated RAG fine-tuning.
- **Privacy background.** [Zeng et al. (ACL Findings 2024)](https://aclanthology.org/2024.findings-acl.267.pdf) found that retrieval reduces the output of memorized training data. RAG membership attacks include:
  - verbatim probes ([Mirabel](https://arxiv.org/abs/2505.22061) defends against these);
  - natural-question attacks: [Riddle Me This](https://arxiv.org/abs/2502.00306) and [MEntA](https://arxiv.org/abs/2605.24312).
- **Possible contribution:** the controlled, combined experiment, with grounding × training membership × library membership, under active FL and RAG attacks at matched, accounted budgets.
  - Our search found no exact duplicate, but it was not systematic, so there is no "first" claim.
  - A structured related-work search is a Stage 0 deliverable.

## 3. Arms

All training arms share the model, FL setting, questions, prompt template (`rag._prepare_prompt`, chat format) and answer-only loss.

| Arm | Training data | Context slot at training | Loss | Unit and replication |
|---|---|---|---|---|
| **RG** | Private client triples | The passage | Answer tokens only | Per target (paired worlds) |
| **CB-AO** | The same questions and answers | **Empty** | Answer tokens only | Per target (paired worlds) |
| **RG-public** | Public triples (group U); no private data | The passage | Answer tokens only | **5 seeds**, shared by all targets; the seed is the unit |
| **P0** | None (pretrained) | — | — | One model |
| CB-LM (bridge) | Today's closed-book records | Today's format | Full LM loss | Stage 1 only |

**DP variants.** RG and CB-AO are each trained at ε ∈ {∞, 64, 16} (§7).

## 4. Data design

The source is the SQuAD train split, partitioned **by article** into groups T, T-hold, U, L, N, P and V:

| Group | Contents | Used for |
|---|---|---|
| **T** | One (passage, question, answer) per passage, over 4 clients × 32 records | Client data; target records |
| **T-hold** | Same form | The record that replaces the target in the non-member world |
| **U** | Same form | RG-public training |
| **L** | 256 passages | Private library. Its passages' questions form the **final utility set F**, at least 300 questions |
| **N** | Passages never placed in any library | Non-member documents for validating the datastore attack and for a secondary standard AUC |
| **P** | 256 passages, with their questions | Public library and public-library utility |
| **V** | A smaller copy of everything above, plus an attacker-calibration slice | All tuning, validation and pilots |

**Target and probe eligibility.** This is fixed in advance, applied on CPU before target resolution, and blind to outcomes.

- **Candidate target passages** need at least 4 SQuAD questions.
- **Training question.** In a seeded order, the first question becomes the **training question–answer pair**.
- **Qualifying probes.** The next **3 qualifying siblings** become the passage's **probes**. A sibling qualifies only if all of these hold:
  - its normalized gold answer does not occur in the training question or the training answer;
  - the training answer does not occur in its gold answer;
  - its SQuAD answer span does not overlap the training answer's span in the passage.
- **Passages with fewer than 3 qualifying siblings** cannot be targets. They are skipped by rule, and the skip counts are recorded.
- **N documents** need at least 3 questions; their first 3 in seeded order are their probes.
- **The overlap exception.** The only way a T passage enters L is the target's own passage, inserted in the library-on condition (§5). Probe documents are never L passages, so probes never overlap F.
- **Checks before any GPU job:**
  - article-disjointness, apart from that exception;
  - one question per passage in every training set;
  - `validate_partition_tokens` passes;
  - every earlier target question is excluded (guard v3/v4, causal Stage A/B, matched controls and the reserved finals, via recomputed old record hashes).
- **The reserved-final cohort stays closed.**

## 5. Core design: four cells per target

For target *t* (with training pair (q_t, a_t) and passage p_t), and for each arm and budget:

| | *p_t* in the library (L+) | *p_t* not in the library (L−) |
|---|---|---|
| **Member world W1** (*t* in client data) | both | training-only |
| **Non-member world W0** (*t* replaced by a T-hold record *h_t*) | library-only | neither |

- W1 and W0 are separate FL trainings, with the same seed and the same other records.
- The library switch needs no retraining (the existing `membership_overlap` mechanism).

## 6. Measurements (exact construction)

### 6.1 H1: Reference attack (weights, record membership)

- **Score.** s(*t*, W) is the Carlini log-perplexity ratio of record *t* under world W against the pretrained model. The attacker's scoring form (the standard record text, or the answer given the template and, for RG, the passage) is fixed per arm on the tuning cohort (§6.5).
- **Per target:** the member score s₁ = s(*t*, W1) and the non-member score s₀ = s(*t*, W0). The library is irrelevant to this weights-only attack.
- **Pooled AUC per arm:** the Mann–Whitney AUC of {s₁} against {s₀} over the *n* targets, with ties counted as ½.

### 6.2 H2: causal gradient attack (client updates)

- **Setting.** The validated configuration: cosine score, `probe_epochs: 12`, `probe_lr: 0.005`.
- **Trials.** In W1 only, **40 trials** (20 counterbalanced member/non-member batch pairs).
- **Per-target AUC** over those trials, as in the causal validation. The pooled value is the **mean of the per-target AUCs**.
- **Attacker direction** (the record's LM loss, or the answer-only loss given the template and passage) is fixed per arm on tuning.
- **Release noise (secondary).** A second 40-trial pass on the same W1, with σ_obs = 1.0 release noise.

### 6.3 H3: natural-question attack (datastore)

- **Queries.** A fixed budget of **3 queries per document per condition**: the document's 3 probes, asked through the RAG system (retrieval on).
- **Per-probe scores,** both computed from the same 3 responses. No extra queries are made.
  - *F1 scorer* (adapted from Riddle Me This): token F1 of the response against the probe's gold answer.
  - *Entailment scorer* (adapted from MEntA): the probability, from a pinned NLI model, that the passage entails the response.
- **Refusals.** A response flagged by `rag.is_refusal` scores **0** in the primary analysis. Refusal rates are reported per arm and cell. A sensitivity analysis excludes refusals. The scorer is chosen on tuning (§6.5).
- **Document score:** the **mean of its 3 probe scores**. The document is the unit; probes are never counted as independent observations.
- **Per target,** both in world W0: the member score d₊ is the score with *p_t* in the library, and the non-member score d₋ is the score with *p_t* out of the library. The **same passage** in and out is the primary non-member comparator, so it is matched on difficulty.
- **Pooled AUC per arm:** the Mann–Whitney AUC of {d₊} against {d₋} over targets.
- **Secondary:** the standard-attack AUC of the inserted target passages against N documents (different documents).

### 6.4 H4: closed-book passage-fact retention (weights)

- **Queries.** The target's 3 probes, which ask about facts absent from (q_t, a_t) by construction (§4). They are asked **without retrieval**: the same template with an empty context slot.
- **Per-world retention score:** the mean token F1 over the 3 probes. Refusals score 0.
- **Per target:** retention r_t = F1(W1) − F1(W0). This is the gain from having trained on *t*, whose passage appeared in RG's context and not in CB-AO's.
- **Endpoint:** D_H4 = mean_t r_t(RG) − mean_t r_t(CB-AO), a difference-in-differences.
  - It measures factual retention directly. It is not a membership score.
  - CB-AO controls for gains from the training question–answer pair itself.
- **Secondary:**
  - mean r_t per arm;
  - the natural-question membership AUC in the training-only cell (W1 L− against W0 L−).

### 6.5 Fixing the attacker's choices on tuning

For each arm, the Reference scoring form, the causal direction and the H3 scorer are fixed on the **tuning cohort (V, seeds 8000–8099)**:

- the choice with the highest pooled tuning AUC wins;
- ties go to the standard form or the F1 scorer.

These choices are frozen in `protocol.json` before Stage 2.

### 6.6 Utility

- **Primary:** RAG token F1 on **F** (private library, retrieval on, top-k = 4), measured on each member world W1 with L+. The non-member world is also scored and reported.
- **Secondary:** EM, public-library F1, no-context F1 and answer NLL (utility protocol v4).

## 7. Matched DP budgets, and what they cover

- **Mechanism.** DP-SGD with per-example clipping and Gaussian noise at each local step (`defenses.private_train`), using the answer-only mask.
- **Accountant, as implemented** (`defenses.privacy_bound`): Gaussian zCDP, no subsampling amplification, replace-one adjacency, over all local steps of the busiest client. The target's client holds 33 records; with batch 2, 1 local epoch and 3 rounds, that is 51 steps. ρ = 2 · 51 / σ², and ε = ρ + 2√(ρ ln(1/δ)), with δ = 10⁻⁵.

| ε | σ |
|---:|---:|
| ∞ | 0 |
| 64 | 1.91 |
| 16 | 5.45 (primary for H6) |

- **Same budget, same noise.** RG and CB-AO have the same record counts, batch and steps, so matched ε means the same σ. Each run records its accounted ε, and the analysis refuses any pair whose ε differ.
- **What ε covers.** The guarantee applies **only to the accounted training releases**: the DP-SGD client updates during the 3 FL training rounds, and anything computed from them (the aggregated models, the final weights and the Reference scores). The privacy unit is one training record.
- **What ε does not cover:**
  - responses to the malicious server's crafted requests (the causal attack), which are separate releases. The optional release noise has its own per-release accounting and is never merged with the training ε;
  - RAG outputs, and the library;
  - the non-member world's own training data.
- **The accountant is conservative.** A record takes part in only 3 of the 51 steps. A participation-based accountant is not used unless approved separately (decision 5).

## 8. Stages, validation and branches

### Stage 0: build and validate (mostly CPU, plus short GPU jobs on V)

1. **Structured related-work search** (a document).
2. **Split builder and audit** (§4), including the skip counts for passage eligibility.
3. **Formatters** (RG, CB-AO, RG-public, CB-LM), the answer-only mask in both training paths, and prompt-parity tests.
4. **Datastore-sensitivity validation (H3).** Run on V with P0, at least 60 V target passages, library in against out, and the §6.3 construction.
   - **Pass** if the pooled AUC's 95% bootstrap lower bound is **≥ 0.70** for at least one scorer. Only scorers that pass can be chosen in §6.5.
   - **If none passes,** H3 is recorded as *not evaluable with these adaptations*, and only the verbatim baseline is reported.
5. **Training-only sensitivity validation (H4).** A **deliberate-memorization positive control** on V: paired worlds on at least 6 V targets, where each member world is additionally fine-tuned on the target passage's full text (full LM loss, 3 epochs).
   - **Pass** if the retention r_t (§6.4) has a mean 95% lower bound **> 0.05 F1**, **and** the training-only natural-question AUC has a 95% lower bound **≥ 0.60**.
   - **If the retention test fails,** H4 is recorded as *not evaluable: the probe cannot detect retention even under deliberate memorization*.
   - **If only the AUC test fails,** the H4 membership AUC is reported as insensitive.
6. **Gate calibration** on V (CPU): at most 5% benign gold-document loss.
7. **Timings:** sequence length 384 (median passage 105 words, 90th percentile 145), DP-SGD overhead, and a full paired job.

### Stage 1: pilot (V; seed band 8000–8099)

- **Scope:** 4 targets × {RG, CB-AO} × ε ∈ {∞, 64, 16}, plus 4 CB-LM bridge targets, RG-public (5 seeds) and P0.
- **Fix** the attacker choices (§6.5).
- **Estimate** variances and re-estimate the Stage 2 cost.
- **Branch decision** (predefined; replaces "RG must beat P0"). Take the difference in pilot F1 on V, D₀ = F1(RG, ε = ∞) − F1(P0), and classify it with §9.2 using 95% intervals.
  - **Branch A** (D₀ is a statistical improvement, outcome 1 or 2): the full Stage 2.
  - **Branch B** (any other outcome). The study reports **"private fine-tuning is unnecessary for utility in this setting"** as a primary finding. Stage 2 still measures leakage (H1–H5 at ε = ∞), because deployments may fine-tune anyway. The DP arms and H6 are dropped, since utility at a matched budget is moot. Stage 2 re-tests RG against P0 on F as a secondary endpoint.
- **Pilot data are tuning data** and are never reported as evidence.
- **Gate:** the researcher approves the Stage 2 budget.

### Stage 2: confirmation (seed band 9000–9099; frozen `protocol.json`)

The approved branch and budget option, paired: every target runs under every per-target arm.

## 9. Statistics

### 9.1 Samples and intervals

- **The unit is the target.** Each target contributes:
  - one Reference score pair (s₁, s₀);
  - one per-target causal AUC;
  - one document score pair (d₊, d₋);
  - one retention value r_t;
  - one utility value per world (the mean F1 over F).
- **Per-target endpoints** (H1–H4, H6, and D₀). A **paired target bootstrap**, 10,000 resamples with `default_rng(20260930)`.
  - Each resample draws target indices with replacement.
  - It carries all of a target's quantities together, for **every arm at once**, so arm differences are paired.
  - Pooled AUCs and means are recomputed on each resample.
- **Utility endpoints** use a **crossed bootstrap**, because F questions are shared across all models. Each resample draws targets (or seeds) **and** F questions independently, then recomputes the mean F1.
- **H5 compares RG-public (5 seeds) with RG (20 targets).** Its resample draws RG-public **seeds**, RG **targets** and F **questions** independently. RG-public's variance therefore rests on 5 seeds, and its interval is wider for it.
- **Multiplicity:** six primary hypotheses. **Decision intervals are 99.2%** (Bonferroni, 1 − 0.05/6); 95% intervals are also reported.

### 9.2 Outcome classification (first match wins)

For a difference D, oriented so that positive is the hypothesis's direction, with decision interval [lo, hi] and margin *m*:

| # | Condition | Outcome |
|---|---|---|
| 1 | lo ≥ m | **Meaningful improvement** |
| 2 | lo > 0 | **Statistical improvement**, not shown to be meaningful |
| 3 | hi ≤ −m | **Meaningful harm** |
| 4 | hi < 0 | **Statistical harm**, not shown to be meaningful |
| 5 | −m < lo and hi < m | **Equivalent** within ±m |
| 6 | otherwise | **Inconclusive** |

- **Margins.** m = 0.05 AUC for the attack endpoints, and m = 0.05 F1 for utility and for H4.
- **"Supported"** means outcome 1 or 2, always reported with which one it is.
- **"Acceptable utility"** means lo > −0.05: any harm is bounded below the margin.
- **H2 is two-sided,** with D = AUC(CB-AO) − AUC(RG). Outcomes 1–2 read "RG reduces", outcomes 3–4 "RG increases", 5 "no meaningful change" and 6 "inconclusive".

| Hypothesis | Oriented D | Claimed when |
|---|---|---|
| H1 | AUC_Ref(CB-AO) − AUC_Ref(RG), at ε = ∞ | D supported **and** RG − CB-AO utility acceptable |
| H2 | AUC_causal(CB-AO) − AUC_causal(RG), at ε = ∞ | Its outcome is the result |
| H3 | AUC_NQ(RG) − AUC_NQ(CB-AO), at ε = ∞ | Supported |
| H4 | D_H4 (§6.4), at ε = ∞ | Supported |
| H5 | F1(RG-public) − F1(RG, ε = ∞); secondary against ε = 64 and 16 | Acceptable (lo > −0.05) |
| H6 | F1(RG) − F1(CB-AO) at ε = 16; plus AUC_Ref(CB-AO) − AUC_Ref(RG) at ε = 16 | Utility supported **and** the AUC difference has lo > −0.05 (no worse beyond the margin) |

**Secondary (descriptive):**

- ε = 64 and the full frontier;
- release noise;
- the gate;
- the "both" cell;
- the standard-attack AUC against N;
- refusal-excluded sensitivity;
- public-library F1;
- no-context F1 and NLL;
- P0 against every arm;
- CB-LM;
- per-client results.

Extending the study needs a new pre-registration.

## 10. Budget and rough cost

These are estimates, replaced by Stage 0 timings. A paired job trains two FL worlds at sequence length 384 and runs the utility, probe and attack passes; estimate 30–60 min without DP, with an unmeasured DP-SGD overhead.

| Stage | Jobs | Est. GPU-hours |
|---|---|---|
| 0: smoke, validation (P0 datastore run, plus a positive control on 6 targets) and timing | about 10 | 6–12 |
| 1: pilot (4 targets × 6 arms, plus 4 CB-LM), RG-public (5 seeds) and P0 | about 30 | 16–32 |
| 2, option (a): 20 targets × 6 arms | 120 | 60–120 |
| 2, option (b): 20 targets for ε = ∞; 10 targets for the DP arms | 80 | 40–80 |
| 2, Branch B: 20 targets, ε = ∞ only | 40 | 20–40 |
| **Total** | | **(a) about 80–165 · (b) about 60–125 · Branch B about 40–85** |

The Stage 2 budget is approved only after Stage 1.

## 11. What this study cannot show

- **Narrow setting.** One model (Qwen2.5-0.5B-Instruct), one data set (SQuAD) and one FL setting. There are no generalization claims.
- **The datastore attacks are adaptations,** validated only for sensitivity. Stronger attackers may exist.
- **The ε values are large,** because of the conservative accountant. They are comparable between arms, and they cover **only the accounted training releases** (§7). There is no end-to-end ε.
- **No new defense mechanism is claimed.** A defense sized by these findings would be a follow-up study.
- **The guard/detector line is closed.**

## 12. Implementation work (after approval)

1. **Data:** the split builder, eligibility rules and audit; old-target exclusion.
2. **Training:** the formatters and answer-only mask in both training paths; parity tests.
3. **RAG study data:** new study data from L, N, P and V, with the F questions and probe sets.
4. **Attacks:**
   - adaptive Reference and causal forms;
   - the 3-query natural-question pass with both scorers (NLI model pinned by revision);
   - the closed-book retention pass;
   - the second causal pass with release noise.
5. **Validation:** the positive-control training path (full-text fine-tuning of member worlds) and the validation scripts.
6. **Accounting:** a per-run ε record, and a refusal on mismatched ε.
7. **Library gate:** the gate and its calibration.
8. **Study tool** (`prepare`, `check`, `analyze`) and tests, including every §9 construction and the classification. `check` never shows endpoint values before `analyze`.
9. **Fingerprint.** The core fingerprint changes. Earlier results keep theirs.

## Decisions to approve

1. **Title and main question; the cross-channel rule** (§1).
2. **Arms** (§3), including RG-public with 5 seeds.
3. **Data design and eligibility** (§4): 3 qualifying siblings, and the absent-fact rules. *(Overlap: decided.)*
4. **Core design** (§5). *(Two worlds per target: decided.)*
5. **DP budgets and scope** (§7): ε ∈ {∞, 64, 16}, the conservative accountant, and the coverage statement.
6. **Measurements and validation** (§6, §8):
   - query budget 3, refusals scored 0;
   - the datastore pass criterion (lower bound ≥ 0.70);
   - the positive-control criteria (retention lower bound > 0.05 F1, AUC lower bound ≥ 0.60);
   - the branch rule.
7. **Statistics** (§9): paired and crossed bootstraps, 99.2% decision intervals, the six-outcome classification, m = 0.05.
8. **Budget** (§10): option (a) or (b). The Stage 2 budget is approved after Stage 1.
