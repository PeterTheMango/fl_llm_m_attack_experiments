# Do the Grounded Federated RAG Findings Generalize? Model Size, Architecture, Data, Non-IID Clients and a Stronger Weight Attack

*Short title: Grounded FL-RAG, generalization study*

Status: **draft 1 (2026-10-12); every recommended setting in §12 approved by the researcher on 2026-10-14.** The Stage 0/1 procedures are frozen in [`stage0_protocol.json`](stage0_protocol.json); Stage 0 code is implemented ([`status.md`](status.md)). The "Nothing is implemented or run" wording below is historical. This is a new study with its own pre-registration. It builds on the grounded federated RAG study (`outputs/retrieval_grounded_fl_proposal_20260930/`, "the parent study"), whose Stage 2 confirmation finished on 2026-10-10.

---

## Summary in plain terms

The parent study found four things in one narrow setting: Qwen2.5-0.5B-Instruct, SQuAD, 4 clients with random (IID) data, full fine-tuning, and a fairly weak weight attack.

1. **Grounded training (RG) leaks less about its training records through the weights** than closed-book training (CB-AO). The Reference attack AUC was 0.28 lower, and RG also answered 0.29 F1 better (H1).
2. **RG makes documents in the library easier to detect** with natural questions (H3). Together with (1), leakage **moves from the weights to the datastore** (the cross-channel claim).
3. **RG does not store the passage's other facts in the weights** beyond ±0.05 F1 (H4, equivalent).
4. **Fine-tuning helps**: RG beats the untuned model by 0.21 F1 (H7).

This study asks whether those findings hold when we change **one thing at a time**:

- a bigger model;
- a model from another family;
- different data (including synthetic "private" records the model cannot have seen before);
- clients whose data differ from one another;
- a much stronger weight attack (LiRA with shadow models).

Two practical facts shape the design:

- **Bigger models do not fit full fine-tuning on our 20 GB GPU slice.** So the new models train with **LoRA**, a small add-on to a frozen model. Switching to LoRA is itself a change, so the study includes a **LoRA copy of the parent setting** and measures the switch directly.
- **GPU time is limited.** The plan therefore runs one factor at a time against that LoRA baseline, uses 30 fresh targets per setting, and offers a core tier (about 190 GPU-hours) and an extended tier (about 280).

Nothing here is a privacy guarantee. Every number is the effectiveness of a stated attack.

---

## 1. What carries over, and what does not

| Parent result (Stage 2) | In this study |
|---|---|
| H1 claimed (Reference AUC, CB-AO − RG 0.28 [0.18, 0.37]; utility +0.29) | **Primary in every setting**, with the Reference attack (as before) and with LiRA (new) |
| H3 claimed (NQ AUC, RG − CB-AO 0.35 [0.23, 0.46]) | **Primary in every setting** |
| Cross-channel claimed (H1 and H3) | **Derived in every setting**: it replicates when H1 and H3 both replicate |
| H4 equivalent (no passage-fact retention beyond 0.05 F1) | **Primary where a positive control validates the probes**; otherwise secondary (§8) |
| H2 inconclusive (release-noise causal pass) | **Not run.** Under LoRA, client updates are adapter gradients, so the causal attack would need a new design and validation. That is a separate follow-up (§11) |
| H5 not acceptable (RG-public) | Not run |
| H6 not claimed (DP-SGD at ε = 16 erases fine-tuning) | **Not run.** DP is dropped (§11) |
| H7: RG better than P0 | **Descriptive** in every setting (one cheap P0 job per setting) |

Everything not named here keeps the parent's definitions, refined only by tightening:

- the four cells per target;
- answer-only loss;
- the chat-template prompt;
- the absent-fact probes;
- 3 queries per document;
- the six-outcome classification;
- the 0.05 margins;
- the paired target bootstrap;
- the crossed article-cluster utility bootstrap.

---

## 2. Research questions and hypotheses

### 2.1 Research questions

| | Question | Settings ("cells") that answer it |
|---|---|---|
| **RQ0** | Does the parent result replicate on fresh targets, and does it survive the switch from full fine-tuning to LoRA? | **A-full** (parent setting, fresh targets); **A-LoRA** (the same with LoRA) |
| **RQ5** | Does H1 survive a much stronger weight attack? | LiRA in **A-full** (decisive, no LoRA confound); LiRA reported in every cell |
| **RQ1** | Does it hold at larger model sizes? | **Q-1.5B**, **Q-3B** (Qwen2.5, same tokenizer and template as A-LoRA) |
| **RQ2** | Does it hold in other model families? | **Llama-1B**, **Gemma-2B**; a hybrid Mamba/attention model, **Granite-H-1B**, descriptive only |
| **RQ3** | Does it hold on other data? | **Synth** (synthetic private records); **DuoRC** (movie plots), optional |
| **RQ4** | Does it hold when clients' data differ? | **Non-IID** (Dirichlet α = 0.1 over articles) |

**One factor at a time.**
- Every new cell changes exactly one thing relative to **A-LoRA**: Qwen2.5-0.5B-Instruct, LoRA, SQuAD, IID.
- The exception is the architecture cells. These are compared with the size-matched Qwen cell (**Q-1.5B**), so the model family is not confounded with size.
- A-full differs from A-LoRA only in the training method.

### 2.2 Endpoints in every cell

All endpoints are measured at ε = ∞, as in the parent study.

| | Endpoint (oriented so that positive is the parent's finding) | "Claimed" in the cell when |
|---|---|---|
| **E1: H1-Ref** | D = AUC_Ref(CB-AO) − AUC_Ref(RG); utility RG − CB-AO | D supported (outcome 1 or 2) **and** utility acceptable (lo > −0.05), as in the parent |
| **E2: H1-LiRA** | D = AUC_LiRA(CB-AO) − AUC_LiRA(RG); same utility condition | the same rule |
| **E3: H3** | D = AUC_NQ(RG) − AUC_NQ(CB-AO) | D supported |
| **E4: H4** | R_RG and D_H4 = R_RG − R_CB-AO | both supported (the parent's rule). Primary only in validated cells (§8) |
| *derived* | Cross-channel | E1 and E3 both claimed (H2 is not run) |

### 2.3 What counts as "generalizes" (fixed before any data)

Each endpoint in each cell gets one **replication reading**:

| Endpoint | **Replicates** | **Fails to replicate** | **Inconclusive** | **Not informative** |
|---|---|---|---|---|
| E1, E2, E3 | claimed | D is equivalent (no meaningful difference) or harm (reversed) | otherwise | **ceiling or floor rule** below |
| E4 (parent: no retention) | R_RG equivalent within ±0.05 ("no retention, confirmed") | **H4 claimed**: retention appears | otherwise (reported as "not contradicted") | — |
| Cross-channel | E1 and E3 both replicate | E1 or E3 fails | otherwise | either is not informative |

**Ceiling and floor rule (new; a lesson from the parent's H2).** An AUC endpoint is **not informative** in a cell when either holds:
- **ceiling:** both arms' pooled AUC 95% lower bounds are ≥ 0.95;
- **floor:** both arms' 95% upper bounds are ≤ 0.55.

The reading is then "the attack cannot separate the arms here", never "fails to replicate".

**Factor-level summary words, used verbatim in the write-up:**
- **"Generalizes across [factor]":** the endpoint replicates in every cell of that factor.
- **"Partly generalizes":** it replicates in some cells, and in no cell does it fail.
- **"Does not generalize":** it fails in at least one cell.
- **"Undetermined":** none of the above.

**RQ5 reading.** Run in A-full, the parent's own setting.
- If **E2 replicates**, H1 survives a strong attack.
- If **E2 fails**, H1 depends on the attack used: grounded training would look safer only against a weak attacker.
- **Pre-registered check that LiRA is in fact stronger.** Pooling both arms, AUC_LiRA − AUC_Ref has a 95% lower bound > 0. If not, RQ5 reads "LiRA was not stronger in this setting", whatever E2 shows.

**What the design can and cannot show.** With 30 targets it can show that a large effect is still present, or that it has reversed (§9.3). It can almost never show that an effect is **absent** (equivalent within ±0.05), as in the parent study. So a finding that fades will usually read **inconclusive**, not "fails". This is stated in every report.

---

## 3. Design and arms

### 3.1 Cells

| Cell | Model | Training | Data | Clients | Role |
|---|---|---|---|---|---|
| **A-full** | Qwen2.5-0.5B-Instruct | **full fine-tuning, fp32, AdamW** (the parent's code path) | SQuAD (parent build) | IID | RQ0 replication; RQ5 |
| **A-LoRA** | Qwen2.5-0.5B-Instruct | **LoRA** (§5) | SQuAD | IID | **baseline for every other cell**; RQ0 method bridge |
| **Q-1.5B** | Qwen2.5-1.5B-Instruct | LoRA | SQuAD | IID | RQ1; reference for RQ2 |
| **Q-3B** | Qwen2.5-3B-Instruct | LoRA | SQuAD | IID | RQ1 ("large" on this GPU) |
| **Llama-1B** | Llama-3.2-1B-Instruct | LoRA | SQuAD | IID | RQ2 |
| **Gemma-2B** *(tier 2)* | gemma-2-2b-it | LoRA | SQuAD | IID | RQ2 |
| **Granite-H-1B** *(tier 2, descriptive)* | granite-4.0-h-1b (Mamba-2 plus attention hybrid) | LoRA | SQuAD | IID | RQ2, non-transformer probe |
| **Synth** | Qwen2.5-0.5B-Instruct | LoRA | synthetic private records | IID | RQ3 |
| **DuoRC** *(tier 2)* | Qwen2.5-0.5B-Instruct | LoRA | DuoRC SelfRC (windowed) | IID | RQ3 |
| **Non-IID** | Qwen2.5-0.5B-Instruct | LoRA | SQuAD | **Dirichlet α = 0.1 over articles** | RQ4 |

### 3.2 Arms and jobs within each cell

- **RG and CB-AO paired jobs.**
  - Per target, member world W1 and non-member world W0.
  - The four cells and the library switch work exactly as in the parent study.
  - The FL setting is the same: 4 clients × 32 records plus the target or its hold record in client 0, 3 FedAvg rounds, 1 local epoch, batch 2.
- **Shadow worlds for LiRA** (§6.2): 16 per arm per cell, shared by all of that cell's targets.
- **P0:** one pretrained model per cell, for descriptive utility (H7-like).
- **Not run:** RG-public, CB-LM, DP, the causal attack.

### 3.3 Targets, seeds and cohorts

**SQuAD cells** (A-full, A-LoRA, Q-1.5B, Q-3B, Llama-1B, Gemma-2B, Granite-H-1B, Non-IID):
- **Data.** They reuse the parent build unchanged (study sha256 `d7eac161…`). The library, F, N and P are therefore identical to Stage 2, and utility is directly comparable.
- **Targets.** They take **fresh final targets in eligible order, starting at index 60**. The parent's Stage 2 used indices 0–59; it had 124 eligible targets.
  - With 30 targets this is indices 60–89. **Every SQuAD cell uses the same 30 targets**, so contrasts between cells are paired.
  - Indices 90–123 stay unused, in reserve.
- **Cohort guard.** The tool refuses any target whose hash is in the parent's Stage 2 cohort (read from that launch's `cohort.json`), in addition to every earlier cohort that the parent build already excludes.
- **Seed band (new): 10000–10199.** Seed = 10000 + target index. Shadow worlds use 10150 + shadow number (10100 + s would collide with reserve target seeds; fixed at implementation, 2026-10-14). These never overlap the parent's 8000–8099 (V) or 9000–9099 (final).
- **Pilot and smoke runs** use V targets not used by the parent's pilot, bridge, control or timing slices, with seed band 8100–8199: a target's seed is 8100 + its slot in the new V slices, and V shadow worlds use 8150 + shadow number. Everything on V is tuning data, never evidence.

**New-data cells** (Synth, DuoRC):
- Each has its own build, with V and final cohorts, the same layout rules and article-disjoint groups (§4.2).
- Each uses 30 final targets.
- Their contrasts with A-LoRA are unpaired: different targets.

**Non-IID** uses A-LoRA's 30 targets and seeds. Only the client assignment of the 128 base records changes (§4.3).

---

## 4. Data

### 4.1 Datasets checked (verified 2026-10-12 against the Hugging Face Hub)

The absent-fact probes need passages with **at least 4 questions whose answers are different facts in that passage**, plus an article id for article-disjoint splits.

| Dataset | Licence | Questions per passage | Verdict |
|---|---|---|---|
| SQuAD v1.1 (`rajpurkar/squad` @ `7b6d24c4…`) | CC BY-SA 4.0 | 4.6 (76% have ≥ 4) | the parent's data; reused |
| Natural Questions | CC BY-SA 3.0 | about 1 (one question per page) | **unsuitable**: no sibling questions |
| TriviaQA | unclear tag | about 1; answers not located in the text | **unsuitable** |
| HotpotQA | CC BY-SA 4.0 | about 1 per paragraph pair; no offsets | **unsuitable** |
| NewsQA (CNN) | no clear redistribution licence; unofficial re-hosts | 7.1, but stories are about 490 words | **excluded on licence** |
| Quoref (`allenai/quoref`) | CC BY 4.0 | about 5, on 300–400-word paragraphs | possible, but its questions need coreference across the whole paragraph, so windowing breaks many of them |
| DROP | CC BY-SA 4.0 | 17, but answers are often numbers or dates, with no offsets | weak |
| **DuoRC SelfRC** (`ibm-research/duorc` @ `2de12436b894…`) | **MIT** | 13.6 per plot (80% have ≥ 4); plots about 420 words; text answers without offsets | **usable after windowing** (§4.2) |
| COVID-QA | Apache-2.0 | 2,019 questions over 147 long papers | unsuitable: too little data once windowed |
| BioASQ, PubMedQA | registration / MIT | about 1, or yes/no answers | unsuitable |
| emrQA, RadQA (clinical) | n2c2 DUA / PhysioNet credentialing | many | unsuitable: no redistribution |
| TOFU | MIT | 20 per fictitious author, but no passage text and sentence answers | unsuitable as is |
| **Self-generated synthetic records** | ours | as many as we generate | **recommended** (§4.2) |

**Contamination.** SQuAD, Quoref, DROP and DuoRC-SelfRC all come from Wikipedia, which every candidate model was pretrained on, and the QA sets themselves are widely crawled. **Only freshly generated synthetic records are guaranteed to be outside pretraining.** That is the main scientific reason for Synth: on SQuAD, a model may answer a probe from pretraining knowledge, which could hide retention (H4) and blur the Reference attack.

### 4.2 The new datasets

**Synth: synthetic private records** (recommended; the main new dataset).

- **Generator.** A deterministic CPU generator in the repository, versioned and seeded (study seed 20261012). Its output is under our own licence. No language model writes the text, so the data cannot leak from any model's training.
- **Records.** Fictitious people. Each record is a short "file" of 60–130 words, built from sentence templates with several paraphrases per attribute. It holds 8–10 attributes, for example:
  - name, made from syllable combinations so it matches no real person;
  - birth date, home city, employer (a generated company name), job title, manager;
  - diagnosis, medication, allergy, blood type;
  - an insurance-style code.
- **Questions.** Each attribute has 2–3 templated phrasings, and every answer is a span in the passage by construction. So every passage has 8–10 questions, and the parent's sibling rule (`qualifies`) is met by design. The generator refuses any record in which one answer string contains another.
- **Articles.** "Articles" are fictitious organizations of 8 people. They give article-disjoint groups and utility clusters.
- **Layout.** The parent's layout: T, T_hold, U, L (255 + 1 slot), N, P, V and C, F of 300 questions, and N documents with 3 probes. Passages stay within 24–160 words, and every record must fit 512 tokens.
- **Expected regime, stated in advance.** The facts are arbitrary, so:
  - closed-book answers should be near 0 F1;
  - CB-AO must memorize answer strings;
  - the natural-question attack may reach the **ceiling** in both arms (library on: correct answers; library off: no chance). The ceiling rule (§2.3) handles that case.

**DuoRC SelfRC** (optional; tier 2).

- **Windows.** Each plot is cut into consecutive windows of whole sentences, 24–160 words each.
- **Answers.** A question is kept for a window only if its answer occurs **exactly once** in that window (case-sensitive), which supplies the character offset. Questions marked unanswerable, or whose answer is not a verbatim span, are dropped.
- **Articles.** The article is the plot (`plot_id`), so all windows of one film stay in one group.
- **Go/no-go at build (CPU, before any GPU work):** at least 90 eligible V targets and 40 eligible final targets, which are the parent's minimums. Otherwise DuoRC is dropped and recorded; Quoref under the same rules is the fallback.
- **Pinning.** The full revision sha is pinned at the build.

### 4.3 The non-IID scheme

- **Today (IID).** Each world draws 128 off-target-article T records at random and splits them 32 per client.
- **Non-IID.** Over the T articles other than the target's (about 15–20 SQuAD articles), each client c draws article proportions p_c ~ Dirichlet(α = 0.1). This is the standard label-skew construction of Hsu et al. (2019), with articles as labels.
  - Records are then drawn **without replacement**: 32 per client, client by client, from that client's proportions, with one question per passage.
  - At α = 0.1, each client's data concentrate on a few articles. A simulation with 16 articles of 40 records gives about 3.3 effective articles per client, against about 12 near IID (measured 2026-10-14; earlier drafts said "one or two").
- **Fixed by the seed** (`grounded_noniid:{cohort}:{seed}`), so W1 and W0 share it exactly.
- **Unchanged.** The target (or its hold record) still joins client 0, and client sizes stay 32 + 1. Client sizes and DP accounting are therefore untouched.
- **Recorded.** The effective number of articles per client, and the Jensen–Shannon divergence between client article distributions. `check` reports these as structure, never as an endpoint.
- **Alternative (decision 6).** "One topic per client": cluster the T articles into 4 topic groups with the pinned MiniLM encoder and give each client one group. With only about 17 T articles, the clusters would be very uneven, so Dirichlet is recommended.

---

## 5. Models and training method

### 5.1 Models (Hugging Face revisions verified on 2026-10-12)

| Cell | Repo @ revision | Licence | Gated | Params | Turn-closing token vs EOS | Notes |
|---|---|---|---|---|---|---|
| A-full, A-LoRA, Synth, DuoRC, Non-IID | `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae557604adf67be50417f59c2c2f167def9a775` | Apache-2.0 | no | 0.49B | `<\|im_end\|>` = EOS | the parent's model; default system prompt inserted |
| Q-1.5B | `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | Apache-2.0 | no | 1.54B | same | same template and tokenizer |
| Q-3B | `Qwen/Qwen2.5-3B-Instruct` @ `aa8e72537993ba99e69dfaafa59ed015b17504d1` | **Qwen research licence** (non-commercial) | no | 3.09B | same | fine for research |
| Llama-1B | `meta-llama/Llama-3.2-1B-Instruct` @ `9213176726f574b556790deb65791e0c5aa438b6` | Llama 3.2 Community | **yes** | 1.24B | `<\|eot_id\|>` = EOS | template inserts **today's date**; must be pinned (§5.3) |
| Gemma-2B | `google/gemma-2-2b-it` @ `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8` | Gemma terms | **yes** | 2.61B | `<end_of_turn>` **≠** EOS `<eos>` | no system role (our prompt uses the user turn only, so this is fine); eager attention recommended |
| Granite-H-1B | `ibm-granite/granite-4.0-h-1b` @ `d18cca4c121edb87d022116d281ce212c9136f57` | Apache-2.0 | no | 1.46B | `<\|end_of_text\|>` = EOS | 36 Mamba-2 + 4 attention layers; transformers ≥ 4.56; fast kernels needed |

**Alternatives for RQ2** if the gated licences are a problem: `HuggingFaceTB/SmolLM2-1.7B-Instruct` @ `31b70e2e869a7173562077fd711b654946d38674` (Apache-2.0, ungated, ChatML, turn-closer = EOS) and `microsoft/Phi-3.5-mini-instruct` @ `2fe192450127e6a83f7441aef6e3ca586c338b77` (MIT, 3.8B, `<|end|>` ≠ EOS).
- SmolLM2 uses the Llama block design, so it varies the lab, data and tokenizer more than the architecture.
- Gemma-2 is the most architecturally different transformer here: alternating local and global attention, logit soft-capping and a 256k vocabulary.

**Is a non-transformer model feasible with the current code?** Partly. Pure-SSM chat models exist only at 7B (`tiiuae/falcon-mamba-7b-instruct`), which is too costly here. **Hybrids** with a chat template load natively through `AutoModelForCausalLM`, which is what our code uses. Of those, Granite-4.0-H-1B is the best fit: mostly Mamba-2, Apache-2.0, and a turn-closer equal to EOS. Three conditions:
1. The server needs **transformers ≥ 4.56**. In a separate environment, so the parent's path is untouched (§10).
2. It needs the `mamba-ssm` and `causal-conv1d` CUDA kernels, built against the server's torch. Without them transformers falls back to a pure-torch path, which is far slower and uses more memory.
3. A Stage 0 smoke test must pass: training loss falls, gradients are finite, and generation stops.

If any of these fails, the cell is dropped and recorded. Even when it runs, it is **descriptive only**, with 10 targets. An RWKV-7 option needs `trust_remote_code` and is not recommended.

**"Large".** Qwen2.5-3B is the largest model that fits comfortably (§7).
- Qwen2.5-7B-Instruct (`a09a35458c702b33eeacc393d103063234e8bc28`, Apache-2.0) needs about 15 GB for bf16 weights alone, which is borderline on 20 GB.
- 4-bit QLoRA would fit, but quantizing the base model is a further confound.
- It would cost about 8× A-LoRA (about 110 GPU-h for one cell).
- **Not recommended**; listed as decision 2.

### 5.2 Training method: LoRA, and how the switch is controlled

**Why LoRA.**
- Full fp32 fine-tuning with AdamW needs about 16 bytes per parameter. The 0.5B model uses about 9 GB; 1.5B would need about 25 GB; 3B about 50 GB. Only the 0.5B model fits the 20 GB slice.
- LoRA is also the usual way federated LLM fine-tuning is done in practice (for example FedIT), because clients then send small adapters.

**The LoRA recipe, the same in every LoRA cell:**
- the base model frozen in **bf16**;
- LoRA (Hu et al., 2022), rank 16, α 32, dropout 0, on every attention and MLP projection;
- adapters in fp32, AdamW at lr **2e-4**;
- otherwise the parent's settings: 3 rounds, 1 local epoch, batch 2, answer-only loss, max_length 512.
- FedAvg averages the **adapter tensors only**, weighted by record counts, as Flower's FedAvg does.

**Controlling the confound.**
1. **A-LoRA vs A-full, on the same 30 targets** (RQ0). This pair measures what the switch does to every endpoint as a paired contrast. A-full is the parent's exact code path: fp32, full fine-tuning, Flower.
2. **Every other cell is compared with a LoRA cell**, never with A-full. A difference between, say, Q-3B and A-LoRA is then a size difference, not a method difference.
3. **Learning check (Stage 0, V, pre-set).**
   - Each LoRA cell's smoke RG job must show training loss falling over the 3 rounds, and V RAG F1 above P0's on V F by at least 0.05.
   - If it fails, there is **one pre-set step**: lr 5e-4 for that cell, recorded.
   - If it still fails, the cell is dropped and recorded. Learning must not be tuned further on outcomes.

**Reference model under LoRA.** It is the **same base with the adapters switched off**, so no second copy of the model is needed.

### 5.3 Model-specific prompt and stop rules (opt-in; the parent's path is unchanged)

- **Turn-closing token.** The parent requires the assistant turn to end with EOS (`grounded.end_of_turn_id`).
  - For Gemma and Phi, the token that closes the assistant turn is not EOS.
  - The generalization tool reads the closing token from the template; it must be a single special token. Training labels end with that token, and generation stops on either it or EOS.
  - Qwen, Llama, SmolLM2 and Granite are unaffected, because their closing token is EOS.
- **Llama's date.** Llama 3.2's template inserts "Today Date: …" from the wall clock. The tool passes a fixed `date_string`, so the prompt is identical on every day.
- **Default system prompts** (Qwen, SmolLM2, Granite) are kept as each template inserts them, exactly as the parent study did with Qwen.
- **Token fit.** The parent's build checked fit with Qwen's tokenizer only. `resolve` re-checks every T, T_hold and U record under each cell's own tokenizer and template. Any refusal stops that cell; raising max_length needs the researcher's decision.

### 5.4 Faster federated training for LoRA cells (recommended; decision 9)

In the parent study, a world's training took about 300 s. Most of that was Flower/Ray transport and reloading the model for each client in each round.

**Proposal.** For LoRA cells, run FedAvg **in the job's process**:
- the base model is loaded once;
- each client trains its adapter copy with the same seeds, client schedule and batches;
- the adapters are averaged with the same weights.

This mirrors the observation speed-up the parent study already uses (`observe_in_process`).

**Test.** On a tiny GPT-2, the in-process loop run on **full parameters** must give bit-identical aggregated weights to the Flower path. That validates the loop itself; the LoRA cells then apply it to adapters. A-full keeps the Flower path.

---

## 6. Attacks

### 6.1 The Reference attack (E1), as in the parent study

- **Score:** the Carlini log-perplexity ratio against the pretrained model, in the record form and the template form. Both are recorded.
- **Raw NLLs.** This study **also records the raw NLLs** (model and reference). LiRA needs them.
- **Attacker choices are inherited from the parent's frozen Stage 2 choices**: RG record form, CB-AO template form, F1 scorer for H3. Using one attacker everywhere means the attack is not re-tuned to each setting, which makes "generalizes" a stronger statement (decision 7).

### 6.2 LiRA with shadow models (E2)

**Attack.** Online LiRA (Carlini et al., IEEE S&P 2022), with the fixed-variance estimator recommended for few shadow models.

**Shadow worlds**, per cell and arm, S = 16:
- Each is a full FL training with the cell's own arm, recipe and model: 4 clients × 32 records, 3 rounds.
- Its records are a balanced random **half of the cell's 30 targets** (15), plus 113 other T records. The fill records exclude every target's passage and hold record and are spread at random over the clients, one question per passage. Non-IID shadows use the same Dirichlet construction.
- **Balanced design (fixed by seed):** each target is IN in exactly 8 shadows and OUT in 8.
- Shadows are shared by all of the cell's targets, so 16 trainings serve 30 targets.

**Per-target score.**
- φ(t, M) = −(sum of answer-token NLL) of the target under model M, in the LiRA form.
- The form is fixed on the pilot (§8): record or template, highest pooled tuning AUC, ties to template.
- Gaussians are fitted to φ over IN shadows and over OUT shadows. The means are per target; the variance is pooled over targets (fixed variance).
- **LiRA statistic:** Λ(t, M) = log N(φ; μ_in, σ_in²) − log N(φ; μ_out, σ_out²).

**Endpoint.** For each arm, the pooled Mann–Whitney AUC of {Λ(t, W1)} against {Λ(t, W0)} over the cell's targets, exactly as for the Reference attack. D = AUC(CB-AO) − AUC(RG).

**Secondary, from the same shadows at no extra GPU cost:**
- offline LiRA (OUT shadows only, one-sided);
- RMIA (Zarifzadeh et al., ICML 2024), with the OUT shadows as references and population records from T;
- TPR at 10% FPR.

TPR at 1% FPR cannot be estimated with 30 non-members, and this is stated.

**Threat model.** A strong attacker: it knows the training recipe and data distribution and can train models that contain each candidate record. That is the point of RQ5. It still sees only the final global model, as the Reference attack does.

**Storage.** Under LoRA, a shadow is only its adapter (about 35 MB at 0.5B), so every shadow is **kept**, and shadows could be re-scored later if needed. A-full's fp32 shadows are scored at once and retired.

### 6.3 Datastore (E3) and retention (E4), as in the parent study

- **Datastore:** 3 probe queries per document per condition, the F1 scorer (frozen), refusals scored 0, and the same passage in and out.
- **Retention:** closed-book probes, with r_t = F1(W1) − F1(W0).

---

## 7. GPU memory and cost

### 7.1 Memory on the 20 GB slice (arithmetic; Stage 0 measures)

| Cell | Weights | Training peak (estimate) | Fits? |
|---|---|---|---|
| A-full (fp32, AdamW) | 2.0 GB | about 9 GB (measured) + 2 GB reference | yes (as in the parent) |
| A-LoRA, Synth, DuoRC, Non-IID (0.5B bf16) | 1.0 GB | about 3 GB | yes |
| Q-1.5B | 3.1 GB | about 5–6 GB | yes |
| Llama-1B | 2.5 GB | about 5 GB (128k vocabulary logits) | yes |
| Granite-H-1B | 2.9 GB | about 5–7 GB with kernels | yes, if the kernels build |
| Gemma-2B | 5.2 GB | about 8–9 GB (256k vocabulary logits) | yes |
| Q-3B | 6.2 GB | about 9–10 GB | yes |
| Qwen2.5-7B (not recommended) | 15.2 GB | about 18 GB | borderline |

The NLI scorer (about 0.7 GB) shares the GPU during measurement, as now. The reference model needs no extra memory under LoRA.

### 7.2 Cost per job (rough; replaced by Stage 0 timing, as the parent did)

**Starting point:** the parent's measured Stage 2 jobs (core `8b031bbfdcc1`, ε = ∞). They took 0.45 h (RG) and 0.38 h (CB-AO), including the causal passes.

**This study's lean job:**
- **no causal pass** (saves about 0.11 h);
- **utility on W1 only**, without the no-context F1 and the public-library F1, both of which were secondary (decision 10);
- this saves about half of the measurement generations;
- the result is about **0.20 h** per 0.5B job, averaged over the two arms.

**Scaling to larger models.** Generation dominates measurement. It is limited by memory bandwidth, so it scales roughly with the size of the weights in bytes. The multipliers relative to A-LoRA are:

| | m |
|---|---:|
| Q-1.5B | 2.0 |
| Q-3B | 3.5 |
| Llama-1B | 1.6 |
| Gemma-2B | 3.2 |
| Granite-H-1B (with kernels) | 2.5 |
| Synth (shorter passages) | 0.9 |

These are **rough**; the Stage 0 timing launch replaces them. In-process FedAvg (§5.4) could lower the LoRA cells' training time further. That is not assumed here.

### 7.3 Cost per cell and per tier

Assumptions: 30 targets, S = 16 shadows per arm, and one P0 job.

| Cell | Paired jobs (60) | Shadows (32) | Total GPU-h |
|---|---:|---:|---:|
| A-full | 12.0 | 2.9 | **15** |
| A-LoRA | 12.0 | 2.2 | **14** |
| Q-1.5B | 24.0 | 4.4 | **29** |
| Q-3B | 42.0 | 7.7 | **50** |
| Llama-1B | 19.2 | 3.5 | **23** |
| Synth | 10.8 | 2.0 | **13** |
| Non-IID | 12.0 | 2.2 | **14** |
| *Gemma-2B (tier 2)* | 38.4 | 7.0 | *46* |
| *DuoRC (tier 2)* | 12.0 | 2.2 | *14* |
| *Granite-H-1B (tier 2, 10 targets, S = 8)* | 10.0 | 2.8 | *13* |

| Stage | Tier 1 (core) | Tier 2 (adds Gemma-2B, DuoRC, Granite-H-1B) |
|---|---:|---:|
| Stage 0: P0 datastore checks, positive controls, smoke and timing pairs | about 26 | about 40 |
| Stage 1: pilot (A-full and A-LoRA) | about 6 | about 6 |
| Stage 2: confirmation | about 158 | about 231 |
| **Total** | **about 190 GPU-h** (about 8 days of continuous GPU time) | **about 277 GPU-h** (about 12 days) |

- **40 targets per cell instead of 30:** about +27% (tier 1 about 240).
- **For comparison:** the parent study used about 150 GPU-h in total (Stage 2: 67).

---

## 8. Stages

### Stage 0: build and validate (CPU, plus short GPU jobs on V)

1. **Code** (opt-in; the parent's tool and paths are unchanged):
   - model profiles (§5.1, §5.3);
   - the LoRA path and in-process FedAvg (§5.2, §5.4);
   - the Dirichlet partition (§4.3);
   - the shadow-world builder and LiRA/RMIA statistics (§6.2);
   - the synthetic generator and the DuoRC windowing builder (§4.2);
   - a new tool, `tools/grounded_generalization_study.py`, with its own frozen protocol.
   - Tests run on CPU with tiny models.
2. **Builds (CPU):** Synth and DuoRC, with skip counts, the audit and DuoRC's go/no-go. Token fit is checked for every cell's tokenizer.
3. **P0 datastore check** for every new model or dataset. This is the parent's rule: 60 V targets, library in vs out; the F1 scorer passes if its 95% lower bound is ≥ 0.70. If it fails, E3 is "not evaluable" in that cell.
4. **Positive control** (the parent's 12-target procedure) for **every cell with a new model, dataset or training method**: A-LoRA, Q-1.5B, Q-3B, Llama-1B, Synth, and in tier 2 Gemma-2B and DuoRC.
   - **Under LoRA,** the memorization step trains the adapter: 6 steps at lr 1e-3, central fine-tuning on the passage text. The one pre-set strengthening step is 12 steps.
   - **Validated:** at least 10 of 12 targets learn, and the retention lower bound is > 0.05 F1. E4 is then primary in that cell.
   - **Otherwise:** E4 is secondary there.
   - **Inheritance.** A-full inherits the parent's passed 6-epoch control (same code and setting). Non-IID inherits A-LoRA's (same model, data and method).
   - **Option:** skip the model-cell controls (saves about 16 GPU-h), with E4 then secondary in those cells (decision 8).
5. **Smoke and timing pair** per cell: one V target × {RG, CB-AO} + 2 shadows. It includes the LoRA learning check (§5.2) and Granite's kernel check. The cost table is then re-done from these timings before Stage 2 is frozen.

### Stage 1: pilot (V; tuning data only)

- **Where:** in A-full and A-LoRA. 4 V targets × 2 arms per cell, with S = 8 shadows per arm.
- **It fixes the LiRA form** (record or template) per arm, pooled over the two pilot cells, and applies it everywhere.
- **It projects the precision** of each endpoint at 30 targets.
- **Other cells** run only their smoke pair. They inherit the attacker choices: the "same attacker everywhere" rule.
- **Gate.** The researcher freezes `protocol.json` (cells, targets, budget) after the pilot. Definitions may be tightened, never loosened.

### Stage 2: confirmation (fresh targets; seeds 10000–10199)

- **Order.** Cells run in a fixed order: A-full, A-LoRA, Non-IID, Synth, Q-1.5B, Llama-1B, Q-3B, then tier 2.
  - A cell is analysed only when all its jobs are complete.
  - A family is analysed once, when all its cells are complete. Each RQ is therefore final as soon as it finishes.
- **Never re-run or re-analyse a family.**

---

## 9. Statistics

### 9.1 Samples and intervals (as in the parent study, unless stated)

- **Unit and bootstrap.** The target is the unit, with a paired target bootstrap: 10,000 resamples, `default_rng(20261012)`.
- **Shared resamples across the SQuAD cells.** All SQuAD cells share the same targets, so **one index matrix is drawn for all of them**. Contrasts between those cells are then paired.
- **New-data cells** draw their own matrices.
- **LiRA inside the bootstrap.** The shadow Gaussians are re-fitted on every resample, so the interval carries the shadow-fit uncertainty. The resampled targets bring their shadow scores, and the pooled variance is recomputed.
- **Utility.** The crossed article-cluster bootstrap, as in the parent study.

### 9.2 Multiplicity: one family per research question (decision 11)

Each research question is its own pre-registered family. The 0.05 error rate is Bonferroni-split over that family's cells × primary endpoints.

| Family | Cells | Primary tests | Decision interval |
|---|---|---:|---:|
| RQ0 + RQ5 | A-full, A-LoRA | 8 | 99.375% |
| RQ1 | Q-1.5B, Q-3B | 8 | 99.375% |
| RQ2 | Llama-1B (+ Gemma-2B in tier 2) | 4 (8) | 98.75% (99.375%) |
| RQ3 | Synth (+ DuoRC in tier 2) | 4 (8) | 98.75% (99.375%) |
| RQ4 | Non-IID | 4 | 98.75% |

- A cell whose E4 is secondary (no validated control) has 3 primary tests, and its family's level is computed from the actual count before Stage 2 is frozen.
- **The error rate is controlled within each question, not across all five.** That is usual for separately stated questions, and the write-up will say so.
- **The stricter alternative:** one study-wide family of about 28 tests (99.82% intervals), with intervals about 15% wider.

### 9.3 Expected precision at 30 targets

This scales the parent's Stage 2 spread to 30 targets at 99.375%.

| Endpoint | Parent estimate | Projected half-width | Smallest effect whose lower bound clears 0 |
|---|---:|---:|---:|
| E1 Reference AUC D | 0.28 | about 0.14 | about 0.14 |
| E3 NQ AUC D | 0.35 | about 0.17 | about 0.17 |
| E4 R_RG | 0.006 | about 0.042 | — (equivalence needs \|estimate\| < 0.008) |
| E4 D_H4 | 0.001 | about 0.067 | — (equivalence out of reach) |
| E2 LiRA AUC D | unknown | from the pilot | — |

**Plain reading.**
- **If the parent's effects hold at full size,** 30 targets will show them as "meaningful improvement" in each cell.
- **If an effect halves,** it will usually read "inconclusive".
- **"No retention" (E4)** can be confirmed only if the estimate is almost exactly 0. Otherwise it reads "not contradicted", unless retention clearly appears.

### 9.4 Secondary (descriptive, 95%)

- **Contrasts between cells:**
  - D(cell) − D(reference cell) for E1–E3: paired for the SQuAD cells, unpaired for the new-data cells;
  - a size trend: D against log(parameters) over 0.5B, 1.5B and 3B, paired.
- **Per arm:** LiRA vs Reference AUC; offline LiRA; RMIA; TPR at 10% FPR.
- **Per cell:** P0 vs RG utility (H7-like); refusal rates; the training-only natural-question AUC; the "both" cell; the standard AUC against N.
- **A-full vs the parent's Stage 2**, descriptively: different targets, same setting.

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| LoRA learns less than full fine-tuning, which shrinks the utility gain and the H1 difference | A-full vs A-LoRA measures this (RQ0); the learning check plus one pre-set lr step |
| Gated models: Llama and Gemma need their licences accepted on Hugging Face and a token on the server | The researcher's action before Stage 0; ungated substitutes named (§5.1) |
| A newer transformers (≥ 4.56 for Granite) or peft could change the parent's environment | Tier 1 needs neither: LoRA is implemented in the repository (`core/lora.py`, no peft) and Granite is not run. The study runs in its own server worktree; the parent's worktree stays on its fingerprint |
| Model-specific template behaviour (date, turn-closer, system prompt) | §5.3 rules, with a CPU test per model that renders and encodes a record twice and checks they are identical |
| Granite's Mamba kernels fail to build | Descriptive and optional; dropped and recorded if the smoke test fails |
| Ceiling effects (likely for Synth's datastore attack and for LiRA on CB-AO) | The pre-registered ceiling and floor rule (§2.3) |
| DuoRC windowing gives too few targets | Go/no-go at build; Quoref fallback; otherwise RQ3 rests on Synth |
| Some records exceed 512 tokens under another tokenizer | `resolve` refuses; raising max_length is the researcher's decision |
| The LiRA assumptions (the attacker trains shadows with the targets) are strong | Intended (RQ5); offline LiRA (no target needed in shadows) is reported alongside |
| Cost overrun (estimates are rough) | Timing in Stage 0, then the budget is frozen; cells run in a fixed order, so a stop leaves complete families |
| GPU training non-determinism | Accepted, as in the parent (2026-10-04); each world is one random draw |
| Many cells invite cherry-picking | Fixed replication words (§2.3), fixed cell order, every cell reported, one analysis per family |

## 11. What this study cannot show, and follow-ups

- **No update channel (H2) and no DP.** Under LoRA the release is an adapter gradient, and the causal attack would need a new design and validation.
  - DP was erased by noise in the parent study. **DP-LoRA** (far fewer noised parameters, so the model may still learn) is the natural follow-up, and it could re-open H6.
- **Weight leakage** is measured by two attacks only, and **datastore leakage** by one adapted attack.
- **SQuAD and DuoRC facts are in pretraining.** Only Synth avoids this.
- **"Large" means 3B.** Conclusions about 7B and beyond would be extrapolation.
- **Equivalence is out of reach at 30 targets** (§9.3).
- **No defense is claimed.**

## 12. Decisions for the researcher (plain language; recommendation first)

1. **Training method.**
   - **Recommended:** LoRA for every new setting, **plus** a copy of the original full fine-tuning setting (A-full) on fresh targets. A-full both replicates the parent study and lets us measure what LoRA changes.
   - The alternative is LoRA only, which is cheaper by about 15 GPU-h but cannot separate "LoRA" from "the new factor".
2. **How much to run.**
   - **Recommended:** tier 1, about 190 GPU-h, about 8 days of GPU: A-full, A-LoRA, Qwen 1.5B and 3B, Llama-1B, synthetic records and non-IID.
   - Tier 2 adds Gemma-2B, DuoRC and the Granite hybrid, about 280 GPU-h in total.
   - **Qwen 7B: not recommended.** About 110 GPU-h for one cell, borderline memory, or a quantized base.
3. **Targets per setting.**
   - **Recommended:** 30, the same 30 fresh SQuAD targets (indices 60–89) in every SQuAD setting. That is enough to see the parent-sized effects.
   - 40 costs about 27% more and narrows intervals by about 13%.
4. **Other model families.**
   - **Recommended:** Llama-3.2-1B and Gemma-2-2B. You would need to accept both licences on Hugging Face and put a token on the server.
   - If you prefer no gated models: SmolLM2-1.7B and Phi-3.5-mini.
   - The non-transformer option (Granite-4.0-H-1B, a mostly-Mamba hybrid) is feasible with our code only with a newer transformers and compiled kernels. It would be a small, descriptive extra.
5. **New data.**
   - **Recommended:** our own synthetic private records (fictitious people with medical and work details). They are the only data guaranteed to be new to every model.
   - DuoRC (movie plots, MIT licence) as an optional second real dataset.
   - Natural Questions, TriviaQA and HotpotQA cannot be used: they have about one question per passage, so the absent-fact probes are impossible.
6. **Non-IID clients.**
   - **Recommended:** a Dirichlet split over articles with α = 0.1 (strongly skewed: each client's data concentrate on about three articles).
   - The alternative is one topic cluster per client.
7. **The attacker.**
   - **Recommended:** keep the parent's frozen Reference-attack choices everywhere, and add online LiRA with 16 shared shadow models per arm in every setting. Its form is chosen once on the pilot.
   - Offline LiRA and RMIA are reported from the same shadows at no extra cost.
8. **Positive controls for the new models.**
   - **Recommended:** run them (about 16 GPU-h within the tier 1 Stage 0), so "no passage-fact retention" can be tested in the bigger models. That is where retention is most likely to appear.
   - Skipping them makes H4 secondary in those settings.
9. **Faster federated training for LoRA.**
   - **Recommended:** run FedAvg inside the job process, tested to give the same aggregate as Flower on a tiny model.
   - The alternative keeps Flower: slower, but no new training loop.
10. **Leaner measurements.**
    - **Recommended:** drop the no-context F1, the public-library F1 and W0's utility, all secondary in the parent study. This roughly halves the answer generation.
11. **Error control.**
    - **Recommended:** one family per research question (about 99.4% intervals).
    - The alternative is one study-wide family (about 99.8%, intervals about 15% wider).
12. **Not repeated.**
    - **Recommended:** drop the causal attack (H2), DP (H6) and RG-public (H5) from this study. A DP-LoRA study can follow.

**Next, after your decisions:** I write `stage0_protocol.json` and implement Stage 0 (§8, step 1) with tests. The parent's tool, the frozen guard files and `causal_attack_validation.py` stay unchanged, and every core-fingerprint change is recorded in this study's `status.md`.

## References

- Carlini et al., *Membership Inference Attacks From First Principles*, IEEE S&P 2022 (arXiv:2112.03570): LiRA.
- Zarifzadeh, Liu, Rehman, Shokri, *Low-Cost High-Power Membership Inference Attacks*, ICML 2024 (arXiv:2312.03262): RMIA.
- Duan et al., *Do Membership Inference Attacks Work on Large Language Models?*, COLM 2024 (arXiv:2402.07841).
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022 (arXiv:2106.09685).
- Zhang et al., *Towards Building the Federated GPT: Federated Instruction Tuning* (FedIT), ICASSP 2024 (arXiv:2305.05644).
- Hsu, Qi, Brown, *Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification*, 2019 (arXiv:1909.06335): the Dirichlet split.
- Allen-Zhu and Li, *Physics of Language Models: Part 3.1*, 2023 (arXiv:2309.14316): synthetic biographies (bioS), the model for Synth.
- Saha et al., *DuoRC*, ACL 2018 (arXiv:1804.07927); Dasigi et al., *Quoref*, EMNLP 2019 (arXiv:1908.05803).
- The parent study: `outputs/retrieval_grounded_fl_proposal_20260930/` (proposal revision 4, `stage2_protocol.json`, `status.md`).
