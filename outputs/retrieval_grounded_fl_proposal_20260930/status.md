# Grounded federated RAG study: status

**2026-09-30.** Stage 0 tooling is built and tested on the Mac (branch `feat/grounded-fl-stage0`, stacked on `proposal/retrieval-grounded-fl` at ce4b286). **No GPU job has run and no result exists.** The GPU code paths are covered by CPU tests with tiny models and fakes, but they have never run against Qwen on the server. The first Stage 0 jobs therefore double as their smoke test.

- **Core fingerprint:** `86eacb02a66f` (it was `5c53eaa5b4a3`). Earlier results keep theirs.
- **Tests:** 777 passed, 2 skipped (previously 703 passed). The frozen guard files and `causal_attack_validation.py` are byte-identical, and a test pins each.

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
5. **Passage length.** 24–160 words in every group. Every T, T_hold and U record must fit 384 tokens untruncated for RG, CB-AO and CB-LM, or `build` refuses. Inference uses top-4 retrieval, `max_context_tokens` 768 and `max_new_tokens` 64, as in the earlier configs.
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
