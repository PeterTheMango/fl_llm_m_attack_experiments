# Keeping Private Facts Out of the Model: Retrieval-Grounded Federated Fine-Tuning for Privacy-Preserving RAG

*Short title: Read, Don't Memorize — Privacy-Preserving Federated RAG*

Status: **draft proposal.** On 2026-09-30 the researcher decided three items: decision 3 (include the overlap set), decision 4 (keep two trained models per target) and decision 8 (20 final targets). Decisions 1, 2, 5, 6 and 7 are still open; the researcher wants to change something in them. Nothing is implemented or run. Once approved, the confirmation stage's endpoints are frozen in a `protocol.json` before any of its results exist, as in the earlier studies.

---

## Summary in plain terms

Today, the federated clients teach the model to **memorize** their private question–answer pairs. The documents the model looks things up in (the RAG library) are only added afterwards, when it answers.

This is bad for privacy, because the private facts end up stored inside the model, where attacks can find them. It is also bad for usefulness: a small model cannot remember SQuAD answers without the paragraph in front of it.

**Our defense: teach the model to *read*, not to memorize.**

- Each training example becomes "here is a paragraph, here is a question, give the answer". The model is marked only on the answer.
- The private facts stay in the RAG library, which gets its own protection. The model only learns the skill of answering from a document.

We test whether this approach:

- leaks less, in three places: the model, the updates sent to the server, and the library;
- needs less added noise for the same protection;
- still answers questions as well.

---

## 1. Research question

**Main question.** If federated clients train the model to answer from retrieved documents, instead of memorizing their private question–answer data, does it leak less private information and need less noise, while still answering questions as well?

| Sub-question | Hypothesis | What we measure |
|---|---|---|
| **Q1. Model (training data)** | **H1.** Retrieval-grounded training lowers training-record membership leakage, at no more than a small loss of RAG answer quality | Reference attack AUC; RAG F1 |
| **Q2. Updates sent to the server** | **H3.** Retrieval-grounded training weakens the gradient attack, including an attacker who knows the new training method. *"No reduction" is a valid outcome.* | Causal gradient attack AUC, standard and adaptive; also with release noise |
| **Q3. Noise** | **H2.** At the same noise level, retrieval-grounded training keeps better answer quality with no worse protection | RAG F1 and Reference AUC at a fixed DP-SGD noise σ* |
| **Q4. RAG library** | **H4.** A verbatim-overlap gate lowers document-membership leakage from the library while keeping normal questions working | RAG membership-attack AUC; benign gold-document loss |

**Scope.** The contribution is the combination: an FL training objective designed around the retriever, so that private knowledge moves out of the weights and into a separately protected library. It is evaluated against active FL attacks and RAG membership attacks together. The claim "new" still needs a systematic related-work search; the quick scan is in [`../defense_directions_20260930/README.md`](../defense_directions_20260930/README.md).

## 2. Why this direction (evidence so far)

| Finding | Source | What it means here |
|---|---|---|
| The causal gradient attack on client updates is **effective**: AUC 0.992 [0.981, 0.999] | [causal validation](../causal_attack_validation_20260929/results.md) | The update channel leaks and needs a defense we can test against |
| The request detector flags benign requests as readily as attacks (200/200) | [matched controls](../matched_reference_controls_20260930/results.md) | Detecting attacks does not work. A defense must protect every request, not pick out bad ones |
| The Reference attack reaches 100% in pilots. DP-SGD σ=2 brings it to chance. DP-FedAvg reduced RAG F1 to 0 | [research plan](../../master_script/docs/client_guard_privacy_research_plan.md) §2 | Noise works but can destroy usefulness. **Where** the noise goes matters |
| Release noise (σ=1–2) stopped the old probe attack with RAG quality unchanged | same | Never tested on the validated causal attack. This is a cheap gap to close |
| Mirabel removes the gold document for 81% of benign questions | [Mirabel calibration](../mirabel_calibration_20260929/README.md) | The library needs a better-calibrated protection. Mirabel stays a *retrieval-removal control* |
| No-context F1 is 0.01–0.06, even for the **pretrained** model. RAG F1 is 0.2–0.5 | [utility control](../guard_v2_utility_review_20260924/analysis.md) | Usefulness comes from retrieval, not from memorized facts |
| FL records are closed-book `"Question: …\nAnswer: …"`; the SQuAD passage is dropped (`datasets._format_squad`) | code | FL and RAG are never connected during training. This is the gap the defense fills |

## 3. The defense: retrieval-grounded federated fine-tuning (RG)

- **Training example.** It is a SQuAD triple (passage, question, answer), laid out with the **same prompt the RAG system uses at inference** (`rag._prepare_prompt`, chat format): passage as context, then the question, then the answer.
- **Loss.** Only on the answer tokens. The prompt, passage and question are masked out (label −100). The model is never trained to reproduce the passage or the question.
- **Clients.** Each FL client holds its own triples. FedAvg, rounds, clients and learning rate stay as in the current setting, except for the sequence length below.
- **Inference.** Unchanged RAG: retrieve the top-k passages from the library and answer.
- **Comparison arm (CB, closed-book).** Today's training on `"Question: …\nAnswer: …"` records drawn from the same questions, so both arms see the same facts.
- **Reference point (P0).** The pretrained model with RAG and no FL training. It is cheap (evaluation only) and answers an honest question: is training worth it at all? If P0 answers as well as RG, the weights need no private data.

## 4. Data design: four article-disjoint groups

SQuAD asks several questions about each paragraph, and many paragraphs share an article. Every group is therefore split **by article**, so no fact reaches two groups through sibling questions. SQuAD's own train and dev splits are already article-disjoint. This design adds article-disjointness *inside* the train split, which is what the study uses.

| Group | Contents | Used for |
|---|---|---|
| **T. Training** | Client triples; a held-out slice of other articles provides the *non-member* records for training-side attacks | FL training; member and non-member records for the Reference and causal attacks |
| **L. RAG library** | A private library (the sensitive datastore) and a public library of passages; a held-out slice of other articles provides non-member documents | Retrieval at inference; member and non-member documents for the RAG membership attack |
| **V. Tuning** | A small copy of the whole setup: its own training records, library, questions and targets, plus a separate public slice for the attacker's threshold calibration | Every choice: the noise level, the gating threshold, the attacker's score, pilots |
| **F. Final test** | Questions about library passages that were never used anywhere else, plus fresh target seeds | Grading only, once, in the confirmation stage |

- **Sizes (proposed).** The private and public libraries stay at 256 passages each, like today's study. The final test grows from 20 to **at least 300 questions**, because 20 gives far too wide an F1 interval. Each client keeps 32 records; there are 4 clients.
- **Checks before any GPU job.** No shared article, passage or question across groups. The existing token-level disjointness check (`validate_partition_tokens`) must pass.
- **Earlier targets.** Every earlier target is excluded by SQuAD question: guard v3/v4, causal Stage A/B, matched controls and the reserved final targets. They are matched by recomputing the old record hashes. **The reserved-final cohort stays closed.**
- **Overlap (decision 3: included).** Each target's passage is placed in the private library in one condition and left out in the other. Crossed with the two trained models per target (decision 4), this gives the four cells already in the code (`membership_overlap`): in training only, in the library only, in both, in neither.
  - The library switch needs no retraining.
  - The cells measure how training and library leakage interact, as a secondary analysis.
  - The main comparisons use the disjoint records.

## 5. Where noise and protection go

Noise only protects the thing it is added to, so each channel gets its own mechanism and its own accounting. The channels are **never averaged** into one privacy score.

| Channel | Mechanism (existing code unless noted) | Privacy unit |
|---|---|---|
| Model and training updates | **DP-SGD** (`defenses.private_train`): per-example clipping plus Gaussian noise at each local step. It works for both CB and RG, with the answer-only mask for RG | Training record |
| Updates released to the server | **Release noise** (`defenses.protect_observation`): clip the whole released gradient and add noise, σ_obs = 1.0 | Client batch, per release |
| RAG library | **Verbatim-overlap gate** (*new*): if the query shares a long exact span with a retrieved passage, beyond a threshold calibrated on benign questions in V, that passage is withheld | Library document |

DP-FedAvg is not an arm. It trusts the server to add the noise, which does not fit a malicious server, and it reduced RAG F1 to 0 in the pilot.

## 6. Attacks (fixed battery)

| Attack | Standard version | Adaptive version (attacker knows RG) |
|---|---|---|
| **Reference** (training-record membership, Carlini log-perplexity ratio against the pretrained model) | Loss on the record text | Loss on the answer given the passage and question |
| **Causal gradient** (validated: cosine score, `probe_epochs: 12`) | Candidate direction from the record's LM loss | Candidate direction from the answer-only loss given the passage and question |
| **RAG membership** (yes/no on the private library) | Verbatim yes/no probe | — (paraphrase attacks are a known limit; see §11) |

- **Choosing the attacker's score.** For each arm, the stronger of the standard and adaptive score is fixed on the **tuning** cohort, by a pre-set rule: highest mean AUC, ties to the standard score. It is never chosen on final data.
- **Reference design (decision 4: two models per target, as now).**
  - For each target, one FL world is trained with the target record (member world) and one with a held-out record in its place (non-member world).
  - The Reference attack compares the target's score across the two worlds; the target is the unit.
  - The causal attack runs on the member world.
  - Scoring all client records against held-out records in each world is reported as secondary only; it is free once the worlds exist.
- **Release noise.** The causal attack runs twice on the same world: once on raw releases and once on releases with σ_obs noise. This costs attack trials only.

## 7. Arms (paired: every target runs under every arm)

|  | No training noise | DP-SGD at σ* |
|---|---|---|
| **CB (closed-book)** | CB | CB+DP |
| **RG (retrieval-grounded)** | RG | RG+DP |

Plus P0 (pretrained model with RAG, evaluation only). The gate is evaluated inside every arm, as library "gate on" against "gate off".

## 8. Stages

**Stage 0. Build and check (mostly CPU, 2–3 GPU smoke jobs).**

- Build the data split and its disjointness audit.
- Implement the RG format with the answer-only mask, in both the normal and the DP-SGD training paths. Check prompt parity with `rag._prepare_prompt`.
- Implement the adaptive attack directions and the one-world Reference evaluation.
- Calibrate the gate on V, CPU only, in the style of the Mirabel study: the largest-overlap threshold with ≤5% benign gold-document loss.
- **Go/no-go gate.** On V, RG must improve RAG F1 over P0. If training teaches nothing, the privacy comparison is moot, and we stop and report that.
- Time the DP-SGD overhead and the new sequence length.

**Stage 1. Pilot on the tuning cohort (fresh seed band 8000–8099).**

- Estimate variances and job times.
- Fix σ* by rule: the smallest σ in {0.5, 1, 2} at which CB+DP's Reference AUC upper bound is ≤ 0.60 on tuning worlds. If none qualifies, use 2 and record it.
- Fix each arm's attacker score, using the rule in §6.
- Pilot results are tuning data and are never reported as evidence.

**Stage 2. Confirmation (fresh seed band 9000–9099, frozen protocol).**

- 20 targets × 4 arms, run paired, plus P0.
- Endpoints and decision rules are as in §9, frozen in `protocol.json` before Stage 2 starts.

## 9. Endpoints and decision rules (Stage 2)

The unit is the trained world (target). Intervals are 95%, from a paired bootstrap over targets with 10,000 resamples. Each hypothesis has one primary endpoint and is claimed separately. All results are reported whatever the outcome.

| Hypothesis | Primary endpoint | Supported if | Refuted if |
|---|---|---|---|
| **H1** | ΔAUC_Ref = AUC(CB) − AUC(RG), no training noise; plus ΔF1 = F1(RG) − F1(CB) on the private library | ΔAUC_Ref lower bound > 0 **and** ΔF1 lower bound > −0.05 | ΔAUC_Ref upper bound < 0.02 (no meaningful reduction), **or** ΔF1 upper bound < −0.05 (clearly worse answers) |
| **H2** | At σ*: ΔF1 = F1(RG+DP) − F1(CB+DP), with AUC_Ref(RG+DP) − AUC_Ref(CB+DP) | ΔF1 lower bound > 0 **and** the AUC difference upper bound < 0.02 | ΔF1 upper bound < 0 |
| **H3** | ΔAUC_causal = AUC(CB) − AUC(RG), strongest attacker score | Lower bound > 0 | The interval lies inside ±0.05: "no reduction", a valid result |
| **H4** | Reduction in RAG membership AUC, gate on against gate off (RG arm) | Reduction lower bound > 0 with benign gold-document loss ≤ 5% on F | Reduction upper bound < 0.02 |

Anything between the "supported" and "refuted" bounds is **inconclusive**. Extending the study needs a new pre-registration, never targets added after seeing results.

**Secondary (descriptive):**

- causal attack AUC under release noise σ_obs, in every arm. This is the first test of release noise on the validated attack;
- no-context F1 and answer NLL (utility protocol v4);
- public-library RAG F1 and EM;
- the P0 comparison;
- the overlap cells, if included;
- per-client results.

## 10. Budget and rough cost

These are estimates, to be replaced by Stage 0 timings. Each job trains **two** FL worlds (decision 4). RG examples need a sequence length of about 384 tokens instead of 128 (passages have a median of 105 words and a 90th percentile of 145). That makes training several times more expensive per step. DP-SGD adds roughly 1.5–2× at batch size 2, which is unmeasured.

| Stage | Jobs | Est. per job | Est. GPU-hours |
|---|---|---|---|
| 0: smoke and timing | 2–3 | 30–55 min | 1–3 |
| 1: pilot (4 arms × 4 targets, plus a σ grid of 3 × 3 CB+DP) | about 25 | 30–55 min | 13–23 |
| 2: confirmation (4 arms × 20 targets; decision 8) | 80 | 30–55 min | 40–73 |
| **Total** | | | **about 55–100** | Disk and retention follow the existing collector: one job at a time, weights retired after verification.

## 11. What this study cannot show

- **One model** (Qwen2.5-0.5B-Instruct), **one data set** (SQuAD) and **one FL setting** (4 clients, 3 rounds). Generalization is untested.
- **Library attacks by paraphrase** (for example "Riddle Me This") can bypass a verbatim gate. H4 covers verbatim attackers only.
- **Weights versus library.** Moving facts into the library shifts risk rather than removing it. The library's protection is only as strong as the gate and its access control.
- **The DP guarantees are per channel.** There is no single end-to-end ε, and none is claimed.
- **No end-to-end privacy claim.** RG's effect on release leakage may be zero (H3). The guard/detector line is closed and is not part of this study.

## 12. Implementation work (after approval)

1. **Split builder and audit** (article-disjoint), plus the old-target exclusion by SQuAD question.
2. **RG formatter and answer-only loss mask**, in the AdamW client path and in `private_train`, with a prompt-parity test against `rag._prepare_prompt`.
3. **New RAG study file built from L:** 256 private and 256 public passages, at least 300 final questions, and member and non-member documents.
4. **Attacks:**
   - adaptive Reference score and the one-world Reference evaluation;
   - adaptive causal direction;
   - a second causal pass on the same world with release noise.
5. **The gate**, plus its CPU calibration script.
6. **Study tool** (`prepare`, `check`, `analyze`) and tests, following the earlier studies. `check` never shows endpoint values before `analyze`.
7. **Fingerprint.** The core fingerprint will change. Earlier results keep theirs.

## Decisions to approve

1. **Title and research question**, as written in §1.
2. **The defense (§3):** RG with answer-only loss and the inference prompt layout; CB as the comparison arm; P0 as the reference point.
3. **Data design (§4):** four article-disjoint groups; library sizes 256/256; at least 300 final questions; exclusions by SQuAD question. **Decided: include the overlap set.**
4. **Reference evaluation (§6):** **Decided: keep two trained models per target** (member and non-member worlds).
5. **Noise and protection (§5):**
   - DP-SGD with σ* chosen by the Stage 1 rule from {0.5, 1, 2};
   - release noise σ_obs = 1.0;
   - the verbatim-overlap gate with ≤5% benign loss;
   - no DP-FedAvg arm.
6. **Stages (§8),** including the Stage 0 go/no-go rule and the seed bands 8000–8099 (tuning) and 9000–9099 (final).
7. **Endpoints and decision rules (§9),** including the margins (0.05 on F1, 0.02 on AUC, ±0.05 for "no reduction").
8. **Budget (§10):** **Decided: 20 targets in Stage 2**, about 55–100 GPU-hours in total with two worlds per target.
