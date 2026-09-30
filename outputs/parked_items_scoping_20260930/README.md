# Parked items — scoping proposals (2026-09-30)

**Decisions (researcher, 2026-09-30): all three as recommended.**

1. **Utility:** a prospective utility protocol v4 with a matched no-retrieval NLL non-inferiority endpoint. The v3 no-context failure stays recorded. It will be implemented with the next study that evaluates utility, which is the enforced comparison, and never applied retroactively.
2. **Enforced comparison:** deferred. It gets pre-registered only if the matched-reference controls return "separates at this resolution".
3. **Refusal-instruction RAG defense:** dropped from the privacy study, recorded as failed at smoke (no refusals, a utility cost).

Nothing here has been implemented or run. Mirabel stays labelled a **retrieval-removal control**, not a defense ([calibration](../mirabel_calibration_20260929/README.md)).

## 1. No-context utility floor fails (F1 ≈ 0.01–0.06)

**What the evidence says.**

- The **pretrained** checkpoint already scores F1 0.026 and EM 0/20 on the 20-question no-context set, against 0.018 after fine-tuning ([utility control](../guard_v2_utility_review_20260924/analysis.md)).
- The questions are extractive SQuAD questions that depend on a passage, and the empty-context prompt still says "use the context".
- As built, the F1 ≥ 0.10 floor is therefore likely unreachable by any checkpoint of this model, so it cannot tell defenses apart. The relative-F1 criterion is meaningless near zero, as the research plan already notes.

**Proposal (recommended): a prospective utility protocol v4. The failed v3 result stays recorded as failed.**

- **Keep the three v3 endpoints**, reported unchanged, including the no-context failure.
- **Add a declared no-retrieval endpoint that measures what FL fine-tuning should preserve:** teacher-forced answer NLL and next-token NLL on held-out client-distribution text. Use a non-inferiority margin against a matched baseline arm, with target-cluster intervals.
- **Make the pass criterion for no-retrieval utility the new endpoint**, from v4 on, never retroactively.
- **Cost:** evaluation only, about 1 GPU-minute per checkpoint. It adds nothing if run inside the jobs of the next enforced study.

**Alternatives:**

- (b) Keep F1, but replace the absolute floor with non-inferiority against the matched baseline.
- (c) Pin a new closed-book question set the model can answer from pretrained knowledge. This needs a new data pin and an overlap audit.

## 2. No enforced, matched privacy comparison

**Why it is parked.**

- Attack effectiveness (causal AUC 0.992) is not guard protection.
- An enforced guard whose detector flags legitimate requests aborts FL rounds, because `reject_training_round` gives no partial aggregates. It also blocks benign requests, so any privacy gain would be bought with availability.
- The comparison is only meaningful once the matched-reference controls show that the detector's false positives are under control.

**Proposal (recommended): pre-register only if the matched-controls study returns "separates at this resolution".**

- **Arms, on the same fresh targets:**
  - no guard;
  - enforced `rules_classifier` with the frozen detector;
  - noise-only at one fixed σ;
  - guard plus the same noise.
- **An adaptive attacker is required.** It sends the largest interpolation of the causal request that the detector accepts (`adaptive_public_requests` bisection), because a detector that blocks the fixed request invites a smaller one.
- **Endpoints:**
  - attack AUC on the full observable transcript, where the refusal and timing channel counts as an observation and an all-rejected AUC is undefined, not 0.5;
  - legitimate round-abort rate under enforcement;
  - the utility endpoints from item 1.
- **Separate cohorts:** a defended attacker-calibration cohort, kept apart from evaluation. The reserved-final cohort stays closed until a shortlist is frozen.
- **Rough cost:** 4 arms × 20 targets is about 80 jobs, roughly 12–20 GPU-hours, plus an attacker-calibration cohort.
- **If matched controls fail:** do not run it with this detector. Consider a release-protection comparison (noise and accounting only) instead.

## 3. Refusal-instruction RAG defense

**What the evidence says** (smoke run, one checkpoint, 10 utility questions per corpus; [context-exposure README](../context_exposure_repair_20260927/README.md)):

- **No refusals at all.** Attack AUC was 0.675 against 0.595 for ordinary RAG on public, and 0.695 against 0.575 on private. Public F1 fell from 0.316 to 0.154.
- **The AUC rise is small evidence.** Each AUC has a standard error of about 0.04, so the rise is 2–3 SE on a single seed. It is not evidence that the instruction increases leakage.
- **The mechanism did not engage.** The instruction is prepended to the user turn. Qwen2.5-0.5B ignored it, and `is_refusal` only matches a few literal phrases.

**Proposal (recommended): drop it from the privacy study** and record it as failed at smoke: no refusals and a utility cost.

**Alternative, if the researcher wants one more try: a cheap pre-registered mechanics gate before any privacy run.**

- Move the instruction into the chat template's system turn and add two refusal examples.
- On 50 membership-probing and 50 benign queries on the pretrained model, require:
  - at least 80% refusal on membership probes;
  - at most 5% refusal on benign queries;
  - a public F1 loss of at most 0.05.
- Continue only if all three hold. The cost is a few GPU-minutes.
