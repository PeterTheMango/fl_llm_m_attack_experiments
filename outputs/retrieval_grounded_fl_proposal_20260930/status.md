# Grounded federated RAG study: status

**2026-09-30.** Stage 0 tooling is built and tested on the Mac (branch `feat/grounded-fl-stage0`, stacked on `proposal/retrieval-grounded-fl` at ce4b286). **No GPU job has run and no result exists.** The GPU code paths are covered by CPU tests with tiny models and fakes, but they have never run against Qwen on the server. The first Stage 0 jobs therefore double as their smoke test.

- **Core fingerprint:** `714ec988b38b` (it was `5c53eaa5b4a3`; `86eacb02a66f` at 62ebf0f, before the 512 change). Earlier results keep theirs.
- **Tests:** 777 passed, 2 skipped (previously 703 passed). The frozen guard files and `causal_attack_validation.py` are byte-identical, and a test pins each.

## Stage 0 results so far (validation data on V; not evidence)

**Build (server, fingerprint `714ec988b38b`).** Study sha256 `d7eac161d178b6e855979fc879752a5d511516674e913788e590194d4dbecd92`.

- **Exclusions.** All 50 earlier target hashes matched SQuAD rows; the smoke target is the only allowed unmatched one. The cohorts are:

  | Cohort | Targets |
  |---|---|
  | Reserved finals | 4 |
  | Causal Stage A | 4 |
  | Causal Stage B | 10 |
  | Matched controls | 20 |
  | Guard v3/v4 development | 11 |

  52 questions were removed, because SQuAD repeats some identical question–answer pairs.
- **Skips.**
  - 2,972 passages fell outside 24–160 words.
  - 174 duplicate questions within a passage.
  - No article was skipped.
- **Eligible targets.**

  | | Targets | Limit | Fewer than 4 questions | Fewer than 3 qualifying probes | Eligible but no hold record |
  |---|---:|---|---:|---:|---:|
  | V | 186 | hold records | 63 | 55 | 311 |
  | final | 124 | hold records | 123 | 35 | 362 |

**Gate calibration (CPU, V).**
- The threshold is τ = 0.45 (3-gram overlap with the top document).
- Benign gold-document loss is 4.4% (13 of the 297 V utility questions whose gold document is retrieved), within the 5% limit.
- Verbatim membership probes are withheld 100% of the time.

**P0 datastore-sensitivity check (60 V targets, the same passage in vs out; launch sha256 `972362d8…`). It passed for both scorers, so H3 is evaluable and both scorers are eligible for §6.5.**

| Scorer | Pooled AUC | 95% CI (target bootstrap) | Passes (lower bound ≥ 0.70) | Secondary AUC vs 64 N documents |
|---|---:|---|---|---|
| F1 | 0.902 | [0.854, 0.947] | yes | 0.890 [0.826, 0.945] |
| Entailment | 0.887 | [0.837, 0.934] | yes | 0.907 [0.852, 0.953] |

- The verbatim yes/no baseline has AUC 0.75.
- No refusals occurred, so the refusal-excluded AUCs equal the primary ones.
- **Timing.** The P0 job took 31 min:
  - 14 min for the datastore pass;
  - 16.5 min for utility (300 F questions with and without context, plus 100 F_P).
- **Scope.** This validates the attack's sensitivity on the pretrained model only. It is not a privacy result.

**Timing launch (V timing target, seed 8081; two full paired RG jobs; all checks verified).**

| | ε = ∞ | ε = 16 (accounted 16.0096 per world, 51 steps) |
|---|---:|---:|
| FL training, per world | 299 s / 303 s | 319 s / 323 s |
| Measurements, per world (W1 / W0) | 271 s / 304 s | 1009 s / 1084 s |
| Causal: 2 directions × (plain + release noise), 160 trials | 5796 s (97 min) | 5802 s (97 min) |
| **Total** | **1.94 h** | **2.37 h** |

- **Sequence length.** The longest training record was 329 tokens (mean 199), so the 512 cap leaves room.
- **DP-SGD overhead.** About 7% on training time.
- **DP measurements are 3.7× slower.** The probable cause, not yet checked, is that the noisier model's answers run to the 64-token cap.
- **The causal attack dominates.** It takes about 36 s per trial: each trial reloads the model in a Ray client.
- **Budget projection.** At these rates the pilot as planned costs about 59 GPU-hours (budget 16–32). Stage 2 option (b) costs about 110–130 GPU-hours (budget 40–80), even with only the chosen direction. **Cost decision (2026-10-03).** The researcher chose one measure only: the victim client keeps its loaded model across observation trials (`amia.cached_victim_model`, opt-in through `run_attack_trials(reuse_victim_model=True)`). Each trial still overwrites every tensor with the request, and a test shows the gradients are bit-identical to a fresh load. Nothing else changes: the pilot keeps both directions and the release-noise pass, and no-context F1 stays. The guard studies' path is unchanged. Core fingerprint `714ec988b38b` → `4da0821d7111`. The 3- and 6-epoch control runs stay on `714ec988b38b`. A new timing launch must measure the saving before the pilot is costed.

**Second timing launch `timing-2` (2026-10-03).**
- **Ran on the old code.** `d171bed` had not been pushed, so the server worktree was still at `b7a185d`. This is inferred from the branch state; the launch's fingerprint has not been confirmed. **It does not test the fix.**
- **Repeatability check.** It does show that timings repeat. Totals: 6958 s vs 6984 s (ε = ∞) and 8577 s vs 8548 s (ε = 16). Causal: 5767 s and 5863 s.

**Third timing launch `timing-3` (fingerprint `4da0821d7111`, with the reuse fix).**
- **Saving.** The causal attack took 5567 s and 5558 s, only about 4% less. Model reloading was not the bottleneck.
- **Per-trial split (ε = ∞ job).**

  | Pass | Wall time per trial | Victim client compute |
  |---|---:|---:|
  | Plain | 27–28 s | 3 s |
  | Release noise | 41–43 s | 17–18 s |

- **Cause.** About 25 s per trial is simulation overhead. The likely cause, unconfirmed, is that the 2 GB request goes to all four clients and the three bystanders return it. Drawing release noise in float64 costs about 15 s.
- **Totals.** 1.87 h (ε = ∞) and 2.30 h (ε = 16).

**Second cost measure (2026-10-03, researcher's choice).** Observation rounds now simulate only the attacked client (`run_attack_trials(observe_target_only=True)`, opt-in; the guard studies' path is unchanged). It receives the same partition, seeds and request, so observations are unchanged; the three bystanders no longer receive and return the 2 GB request every trial. Core fingerprint `4da0821d7111` → `4a001c5eb678`. The pilot keeps both directions, the release-noise pass and no-context F1. A fourth timing launch measures the saving. The 6-epoch control rerun runs from a worktree pinned to `b7a185d` (fingerprint `714ec988b38b`), the same code as the 3-epoch run.

**Fourth timing launch `timing-4` (fingerprint `4a001c5eb678`, victim-only observation).**
- **Saving.** The causal attack took 3074 s and 3037 s, 45% less than `timing-3`.
- **Totals.** 1.18 h (ε = ∞) and 1.60 h (ε = 16). Training and measurement times are unchanged.
- **Re-costed pilot.** About 37 GPU-hours (budget 16–32):

  | Jobs | GPU-h |
  |---|---:|
  | 8 × ε = ∞ | 9.5 |
  | 8 × ε = 64 (unmeasured; assumed between the two measured budgets) | ≈ 11 |
  | 8 × ε = 16 | 12.8 |
  | 4 CB-LM, record direction only | ≈ 3 |
  | 5 RG-public | ≈ 1 |

- **Re-costed Stage 2 option (b).** About 75–80 GPU-hours (budget 40–80), with the chosen direction only:
  - 40 ε = ∞ jobs at about 0.76 h;
  - 40 DP jobs at up to about 1.18 h;
  - P0 and RG-public.
- **Remaining large cost.** The DP worlds' measurements take about 1000 s per world, against 275 s at ε = ∞.

**Positive control, 3 epochs (12 V targets; launch sha256 `9d203eb0…`).**
- **Learning check.** 4 of 12 targets learned their passage; at least 10 are needed.
- **Rule.** By the pre-registered rule, the control is rerun once at 6 epochs, on the same 12 targets, lr 1e-4 and code fingerprint `714ec988b38b`.
- **Not computed.** Retention and AUC sensitivity are not computed until the rerun decides.

## What was built

| Stage 0 item (proposal §8, §12) | Where |
|---|---|
| Article-disjoint split builder, eligibility rules, skip counts, old-target exclusion, audit | `master_script/core/grounded.py` (`build_study`, `audit_study`); tool `build` |
| RG / CB-AO / RG-public encoders (the `rag._prepare_prompt` chat template, answer-only labels) and CB-LM (today's closed-book text) | `grounded.encode_record`, `grounded.training_record` |
| Answer-only loss in the AdamW client path and in `private_train` | `scoring.EncodedExample`, `amia.make_loader`, `defenses.private_train` |
| The same mask in the causal attack (victim gradients and the optional template direction) | `causal_probe._loss_inputs`, `optimize_request` |
| Supplied worlds for FL training; attack batches with a declared target | `amia.federated_fine_tune(world=…)`, `amia.run_attack_trials(target=…)` |
| New RAG study data from L, N, P and V, with F (300 questions, article ids kept), F_P and the probe sets | `study.json` written by `build` |
| Reference (record and template forms), causal (both directions, 40 trials, release-noise pass), the 3-query natural-question pass (F1 and pinned NLI), retention, utility | `master_script/core/grounded_job.py` |
| Per-run ε record; refusal to compare runs whose ε differs | `grounded.epsilon_record`, `same_epsilon`; tool `require_matched_epsilon` |
| P0 datastore-sensitivity check; positive control with learning check and the one 6-epoch rerun; retention sensitivity (decisive); AUC ≥ 0.60 (secondary) | tool `prepare datastore|control`, `analyze` |
| Verbatim-overlap library gate and its CPU calibration on V | `grounded.verbatim_overlap`, `gated_ranking`, `calibrate_gate`; tool `calibrate-gate` |
| Study tool: `build`, `calibrate-gate`, `prepare`, `resolve`, `run`, `job`, `check`, `analyze` | `master_script/tools/grounded_fl_study.py` |
| §9 constructions: paired target bootstrap, crossed article-cluster utility bootstrap, first-match six outcomes, H1–H7, H4 needing both endpoints, H7 non-inferiority, the cross-channel rule | tool `hypotheses` and helpers; tests in `tests/test_grounded_fl_study.py` |

The executor keeps the collector's guarantees without changing `collect_guard_traces.py`:
- frozen launches, with the core fingerprint, tool hash, protocol and study data all checked;
- CPU resolution of every target, world and token check before any GPU work;
- one job per process, one job per invocation by default;
- weights retired after verification.

`check` shows only integrity, accounted ε, timings and row counts, never a score, AUC, F1, refusal rate or learning value.

## Decisions made with the researcher (2026-09-30)

1. **Answer tokens.** The answer followed by the end-of-turn token (`<|im_end|>`, the tokenizer's EOS). The template must end the assistant turn with EOS, or encoding refuses.
2. **Library slot.** L = 255 fixed passages plus 1 slot. The slot holds a never-queried filler in L− and the target passage in L+, so the size is always 256. F comes only from the fixed passages.
3. **NLI.** `cross-encoder/nli-deberta-v3-base` at revision `6c749ce3425cd33b46d187e45b92bbf96ee12ec7` (label 1 is entailment; the label is read from the config). The premise is the passage. The hypothesis is `Question: <probe>\nAnswer: <response>`.
4. **Positive control.** The base arm is RG at ε = ∞. After FL, W1 gets a central AdamW fine-tune on the plain passage text: full LM loss, 1 step per epoch, lr 1e-4, 3 epochs, or 6 in the one rerun. W0 is not extra-tuned.
5. **Passage length.** 24–160 words in every group. Every T, T_hold and U record must fit `max_length` tokens untruncated for RG, CB-AO and CB-LM, or `build` refuses. `max_length` is **512** (see deviations). Inference uses top-4 retrieval, `max_context_tokens` 768 and `max_new_tokens` 64, as in the earlier configs.
6. **F and libraries.** L takes 4 passages from each of 64 articles (64 clusters). F is exactly 300 questions in seeded order, at most 2 per fixed L passage. P takes 8 passages from each of 32 articles. V has the same library layout.
7. **Base records.** For each target seed, 128 T records (one question per passage) are sampled from articles other than the target's. W1 and W0 share them exactly.
8. **Exclusion.** Question-level with a strict audit. Every manifest hash must match a SQuAD train row under the old closed-book formatting, or `build` refuses. The only allowed exception is the synthetic smoke target.

## Implementation choices (not fixed by the proposal; change any before Stage 0 runs)

- **Old record identity.** `_clean_record(_format_squad(row))` is recomputed at every `_char_budget` that `max_length` 32–512 could produce. `_clean_record` collapses whitespace, so today's records are single-line (`Question: … Answer: …`). CB-LM trains on exactly that string.
- **Sibling rules.** SQuAD v1.1 answer normalization, then containment of normalized token sequences, plus character-span overlap.
- **Seeded orders.** Study seed 20260930, with string-seeded `random.Random` for articles, groups, questions per passage and worlds. Target *k* is paired with T_hold record *k*.
- **Group sizes.**
  - Record groups take whole articles until a passage minimum is met:
    - T: 600 passages;
    - T_hold: 150 (V) or 100 (final);
    - U: 160;
    - C: 100, V only.
  - Library-like groups take a fixed number of articles:
    - L: 64 × 4;
    - N: 16 × 4;
    - P: 32 × 8.
  - A synthetic SQuAD-scale run of the builder used 270 of 442 articles.
  - The builder refuses fewer than 90 V or 40 final eligible targets.
  - **V is not smaller than final** (the proposal calls V "a smaller copy"). The library layout was kept equal so the datastore check transfers.
- **The C slice.** V's attacker-calibration slice (C) is built but not yet used by any Stage 0/1 measurement.
- **V target slices.** These are fixed from the study alone, in eligible order:
  - datastore: the first 60;
  - control: the next 12 whose passage has ≥ 64 tokens (skips counted);
  - pilot: 4;
  - CB-LM bridge: 4;
  - timing: 1.

  The seed is 8000 + target index; RG-public uses seeds 8095–8099.
- **Scorers.** The F1 scorer and utility F1 both use `rag.answer_utility` token F1, the existing utility metric, not SQuAD's normalized F1. The secondary answer NLL is the unchanged `rag.answer_nll`.
- **Verbatim baseline.** One Anderson yes/no query per document per condition, outside the 3-query budget.
- **Standard-attack AUC against N.** All 64 N documents in the P0 check; the first 16 in paired jobs (W0, library off).
- **Utility library.**
  - RG, CB-AO and CB-LM: both worlds use L+ (primary: W1).
  - P0 and RG-public: they have no target, so they use L with the filler.
- **ε per world.** 51 accounted steps, recorded for W1 and W0 and required to match. The two worlds are not composed: the pair is a counterfactual device, not two deployments.
- **DP-SGD clip norm.** 1.0, as in the earlier DP configs.
- **Gate.**
  - Rule: the share of the query's word 3-grams that occur verbatim in the top-ranked document; withhold that document at τ or above.
  - τ grid: 0.05–1.00. The smallest τ with ≤ 5% benign gold-document loss wins.
  - Benign set: V's F questions.
  - In jobs, the gate is recorded at retrieval level only (whether it would withhold, and the overlap); it triggers no extra generation.
- **Pilot attacker choices.** Made per arm, pooling that arm's pilot jobs over the three budgets. **Needs your OK before the Stage 1 analysis:** the alternative is ε = ∞ jobs only.
- **Learning-check continuation.** A positional token match over 32 greedy tokens (with `min_new_tokens` 32), given the first 32 passage tokens. This follows Biderman et al.'s 32/32 memorization score.

## Deviations and gaps

- **Sequence length 512, not 384 (2026-09-30).** The first server `build` refused because an RG training record (SQuAD question `57063fcb52bb8914006899b6`) needs 405 tokens. The researcher chose to raise `max_length` to 512 rather than skip long passages or tighten the word cap. The word range and the no-truncation rule are unchanged. The protocol's timing item "sequence length 384" is now measured at 512.
- **Related-work search.** The structured related-work search (Stage 0 item 1 in §8) is not part of this change.
- **Pilot projections.** Precision projections exist (`projection`) but are not yet wired into the pilot `analyze`. This is Stage 1 work.
- **Stage 2.** No Stage 2 `prepare` exists, because its protocol is finalized after the pilot. `hypotheses()` is implemented and tested on synthetic data.
- **Timing.** Timing is a separate launch: one full paired RG job at ε = ∞ and one at ε = 16, on the V timing target.

## Server steps (Stage 0; the researcher runs these)

Variables: `E=/home/calc08/projects/LLMPrivacy/fl_llm_m_attack_experiments/outputs`, `G=$E/grounded-fl-20260930`. Work in a separate worktree, `../fl_grounded_stage0`, in tmux.

The order:
1. `build`, with every exclusion source: the pilot's `cohort.json`, `splits.complete.json` and `launch.json`, the causal `stage-a-2` and `stage-b` splits, the matched-controls splits and the committed v3 copy. Paste back the printed report.
2. `calibrate-gate`.
3. `prepare datastore`, `resolve`, `run` and `analyze`.
4. `prepare timing`, `resolve`, `run --max-jobs 2` and `analyze`.
5. `prepare control`, `resolve`, `run --max-jobs 12` and `analyze --datastore-analysis`. If the analysis says `rerun_at_6_epochs`, prepare the one rerun with `--previous-analysis`.

Never edit a launch; on any refusal, prepare a new directory and record it here.
