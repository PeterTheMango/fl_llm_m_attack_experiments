# Client guard continuation — implementation v2, 22 September 2026

Implemented code and local mechanics checks; **no new GPU experiment or effectiveness claim**. The previous six-run pilot and its configuration are preserved. Source fingerprints change for this implementation: do not compare its eventual scores to the old pilot as a defense-only intervention. The existing uncommitted CLI changes were retained.

## What changed

- `GuardRuntime.authorize` returns an owned parameter snapshot. Training and AMIA execute that snapshot after validation, before loading the victim model or accessing its private batch. Each approved checkpoint array is read once per request and checked against a content digest pinned by the trusted harness. No timestamp-based or stale model cache is used. Guard settings and detector pins never come from server request metadata.
- Parameter features use bounded NumPy reductions instead of repeated BLAS dot calls. Timings distinguish copying, reference reads, reference hashing, numerical validation, feature extraction, detector loading/inference, ledger setup/reservation, and audit commits. Ordinary training also records model loading, per-client loss, communication bytes and round duration. Guard events include process RSS samples; these are not peak CUDA-memory measurements.
- Training continues to require every scheduled update. Completed and aborted rounds are durably recorded in the client ledger. The recovery policy is **stop the model world, preserve reservations and artifacts, and require a new independently approved run**. There is no automatic resumption from a partial aggregate, refund after a crash, or override to finish a rejected round. Worker reconstruction within the same run preserves the budget. Separate experimental counterfactual worlds have separate, explicitly scoped ledgers; this is not a cross-experiment identity or Sybil defense.
- `causal_gradient_alignment` is a separate architecture-preserving variant. It performs bounded candidate-loss ascent in existing LM weights, then uses only the released full LM gradient projected onto a public candidate gradient. It receives no private examples, activations or losses. `probe_head` remains the original attack. No AMIA theorem or certificate transfers to the new variant. Its undefended LLM success remains **unverified**.
- Optional `adaptive_public_steps` performs a bounded white-box detector evasion search over public parameter interpolation. These are public detector evaluations, not free private releases. Subsequent private observations still use durable per-world reservations and the configured fixed noise mechanism. This is not an implemented black-box adaptive timing oracle.
- New stage runs counterbalance AMIA member/nonmember order within matched pairs. Baseline, noise, and guarded requests record processing time. There is no transport-time or GPU warmup correction; calibration and matched timing comparisons are still required.
- RAG refuses to turn ambiguous/unrecognized/refusal answers into nonmember predictions. Its leading yes/no parser rejects buried words and Yes/No echoes. All-unrecognized metrics remain unavailable. A separate `continuation` attack asks for the omitted half of a public candidate and scores answer overlap; each document uses one query per condition/checkpoint. Its threshold and direction must be frozen using independent document validation before claiming attack accuracy.
- Raw membership, overlap and utility answers are written to `private-audit/rag-answers.jsonl` with mode 0600, outside result JSON/Firestore payloads. Utility rows include latency and an explicitly limited lexical support screen. Human support auditing is still necessary.
- Overlap rows include a same-checkpoint no-context answer, datastore replacement at fixed cardinality, tokenized training exposure evidence, and candidate visibility after context truncation. Reference supplies the separate training-present/absent checkpoints. Reports retain all four cells and estimate within-target response differences; constant yes in every cell is response bias, not evidence of overlap leakage.
- Detector trace assembly verifies source hashes, explicit target roles and implementation revision, reserves an attack variant from fitting, and keeps identities/labels/paths out of the feature vector. Detector fitting remains logistic regression with a relative-delta threshold baseline. Reports include held-out variant outcomes and precision at a declared prevalence.
- Attack calibration and analysis are separate from detector training. The calibration tool freezes direction and threshold on independent defended validation worlds, rejects final-target overlap, binds artifacts to code/config/defense identities, and collapses repeated requests before estimating independent sample resolution. It supports score, refusal, timing and a joint statistic/decision/timing attacker. The joint attacker splits validation target groups between coefficient fitting and threshold selection. This is a tested attacker, not an upper bound on full transcript leakage.

## Local evidence and limits

The final regression suite passed **498 tests; 11 skipped** in the available Python environment. Skips include unavailable Torch/Flower tests and a new tiny-Torch architecture test. NumPy mechanics tests exercise reference tampering, check/use snapshots, one-read-per-reference behavior, large later-round deltas, concurrent/persistent budgets, all-rejected metrics, reversed validation ranking, grouped splits, held-out variants, transcript leakage, parser ambiguity, stage counts, and four-cell response bias. See `outputs/client_guard_v2_verified/verification.json` and `pytest.txt` for the final counts.

`outputs/client_guard_v2_verified/report.json` records four rounds of actual linear SGD over four heterogeneous synthetic clients, followed by a deliberate aborted round. Loss decreases, later rounds produce nonzero parameter deltas, and restarting the guard cannot reset duplicate/exhausted reservations. **This is CPU linear-model mechanics, not federated LLM or Flower evidence.** The toy dataset/detector generated beside it must not be used as a research detector.

One local 16-MiB synthetic-array profiling comparison measured feature extraction at approximately 0.30 s before and 0.02 s after replacing the reductions. This single CPU comparison is not a GPU speedup estimate. The unit test also verifies that each reference array is read exactly once. Fresh snapshot copying and hashing still cost time/memory, which must be measured on the actual model.

No suitable local CUDA runtime is available: the current Python has neither Torch, Transformers nor Flower. No remote jobs were submitted, packages installed, or CPU substitution made for the proposed GPU study.

## Reproducible commands

Run commands from the repository root in the established research environment. Files written by dataset assembly, calibration, detector training and stage generation are exclusive outputs; choose new names rather than overwriting prior results.

```bash
python -m pytest tests
python -m master_script.tools.guard_local_smoke outputs/guard-v2-mechanics-new
```

Generate the next GPU smoke stage **on the remote host**, so artifact paths resolve there:

```bash
python -m master_script.tools.build_guard_stage smoke outputs/guard-v2-smoke-remote --gpu 0
python -m master_script.perform_experiments \
  --queue outputs/guard-v2-smoke-remote/experiments.yaml \
  --attack amia --attack reference --dry-run --no-firestore --no-charts
```

Expected: **11 runs**, three FL rounds each, seed 1000, four clients, one AMIA target per condition, four AMIA observations per target, four probe epochs and 20 public nonmember calibration batches. Eight AMIA runs cover both variants × baseline/rules/noise/shadow; three Reference runs cover baseline/rules/shadow with one paired target (two separate training worlds). Ordinary RAG and overlap are enabled. GPU 0 is the explicit default in this example; select the allocated physical index. Inspect `launch.json` before execution.

The command below is the next GPU job to execute **only after authorization for its measured/estimated duration**:

```bash
EXPERIMENT_GPU=0 python -m master_script.perform_experiments \
  --queue outputs/guard-v2-smoke-remote/experiments.yaml \
  --attack amia --attack reference \
  --queue-output outputs/guard-v2-smoke-remote/results \
  --no-firestore --no-charts
```

Outputs: a queue manifest and one JSON per resolved run; per-run `result.json`, attempt-specific trial/observation and target progress JSONs, `client-guard/release-ledger.sqlite`, checkpoint snapshots, private audit JSONL, model artifacts, round histories and stage timings. Keep these together when recovering a run. AMIA observation and completed-target checkpoints use fresh attempt directories, preserving prior accepted observations across retries. The runner does not claim automatic continuation of partially completed private worlds; interrupted reservations remain burned.

The old one-round pilot measured about 329–371 s per AMIA target and 734–996 s per two Reference worlds, with ordinary training rounds about 75–81 s without guard. Eleven three-round runs imply a substantial job; a rough head-only extrapolation is about 1.7–2.5 hours, **excluding unknown additional cost of full-LM causal gradients and revised checks**. Do not treat this as a scheduling guarantee. Use stage timings from the first completed model world to revise the estimate or split the stage before authorizing a longer sweep. No checkpoint reuse is enabled across runs: every AMIA target and Reference world is freshly trained. Therefore there is no unverified shared-checkpoint shortcut or false independent sample count.

A shorter single-condition preflight is supported through a copied stage definition, keeping the main stage unchanged:

```bash
python - <<'PY'
import json
from pathlib import Path
p = json.loads(Path('master_script/configs/guard/stages/smoke_v2.json').read_text())
p['conditions'] = ['baseline']
p['variants'] = ['causal_gradient_alignment']
p['reference_conditions'] = ['baseline']
with Path('outputs/guard-v2-preflight-plan.json').open('x') as f:
    json.dump(p, f, indent=2)
PY
python -m master_script.tools.build_guard_stage smoke outputs/guard-v2-preflight \
  --plan outputs/guard-v2-preflight-plan.json --gpu 0
```

An already generated local copy is `outputs/client_guard_v2_preflight/launch.json` (regenerate on the remote host to resolve its paths). This prepares **two runs**: one new causal-probe baseline and one paired Reference baseline, with RAG. Establish finite gradients and actual undefended discrimination before treating the new attack as a useful adversary. The absence of signal is a valid negative result; it does not prove the guard effective. A full-LM gradient request is much larger than the head probe; provision host RAM and GPU memory accordingly and inspect recorded RSS/communication costs.

## Real detector collection, splits and fitting

```bash
python -m master_script.tools.build_guard_stage collection outputs/guard-v2-collection --gpu 0
```

This prepares three shadow runs: two AMIA variants across eight targets and a Reference run across eight paired targets. It does not launch them. Collect more disjoint targets, seeds, clients or permitted training configurations by making new frozen stage definitions. Include a predeclared high-learning-rate/local-step benign stress set; never label a legitimate large update malicious merely because it is large. Run those as separate matched cohorts and retain their configuration hashes.

Create an explicit split manifest from the completed real trace sources. Format:

```json
{
  "schema": "guard_splits_v2",
  "sources": [{"path": "relative/path/to/result.json", "sha256": "FULL_FILE_SHA256"}],
  "groups": {
    "ACTUAL_TARGET_SHA256_A": "train",
    "ACTUAL_TARGET_SHA256_B": "validation",
    "ACTUAL_TARGET_SHA256_C": "test",
    "ACTUAL_TARGET_SHA256_D": "attacker_validation",
    "ACTUAL_TARGET_SHA256_E": "final"
  },
  "held_out_variants": ["causal_gradient_alignment"]
}
```

Use actual target hashes from `attack_trials`, including all related seeds, clients, member/nonmember worlds and conditions. One target has one role. Do not assign the same target to a new role under a new seed. Assembly rejects missing source/identity metadata; non-test occurrences of held-out variants are excluded with denominators. Attacker validation/final targets are excluded from detector fitting. Small collection stages cannot establish 1% benign FPR.

```bash
python -m master_script.tools.assemble_guard_dataset outputs/guard-splits-v2.json outputs/guard-dataset-v2.json
python -m master_script.tools.train_guard_detector \
  outputs/guard-dataset-v2.json outputs/guard-detector-v2.json --max-fpr 0.01 --prevalence 0.01
python -m master_script.tools.build_guard_stage validation outputs/guard-v2-validation \
  --detector outputs/guard-detector-v2.json --gpu 0
```

Expected validation plan: **20 runs**: 14 AMIA (two variants × seven conditions) and six Reference conditions. It uses 20 targets, four private queries per world, three FL rounds, 12 probe epochs, ordinary/instruction retrieval, and the continuation datastore attack. It is a large explicit plan, not authorized execution. `--adaptive` adds one causal combined-noise evasion condition. Reference `training_dp` is a separate training-defense arm. Observation protection does not protect Reference or retrieval by implication. Guard-only arms and the observation combined arm deliberately keep training unprotected as a **diagnostic ablation**; train-plus-observation DP requires an additional explicitly specified combination, not relabeling this arm.

The shipped confirmation shortlist prepares 11 runs on seed band 6000 and 50 targets. Freeze a selected shortlist based on validation before final execution. Current validation and confirmation settings match except independent sample selection/counts. If training, detector, noise, retrieval or probe settings change, regenerate compatible validation instead of reusing an old calibration artifact.

## Frozen attacker calibration and joined reporting

The runtime's legacy nonmember threshold is retained as a diagnostic. Its score direction is fixed higher-is-member. **Do not interpret its below-chance score as successful privacy.** Use independent validation and the following frozen artifact workflow for a defended comparison. Substitute the actual queue batch directories containing the numeric result JSON files (not artifact-directory copies):

```bash
python -m master_script.tools.calibrate_guard_attacks fit \
  outputs/guard-v2-validation/results/BATCH_ID/000*.json \
  --attack amia --condition noise_only --variant probe_head \
  --channel score --query-budget 1 --fpr 0.05 --output outputs/amia-noise-calibration-v2.json
python -m master_script.tools.calibrate_guard_attacks evaluate \
  outputs/guard-v2-confirmation/results/BATCH_ID/000*.json \
  --attack amia --condition noise_only --variant probe_head \
  --calibration outputs/amia-noise-calibration-v2.json --output outputs/amia-noise-final-v2.json
```

Repeat separately per attack, condition, variant, and declared query budget. For Reference use `--attack reference` without `--variant`. For datastore calibration add `--rag-condition private_ordinary` and `--group-manifest outputs/datastore-roles-v2.json`. This latter file maps **every candidate document SHA256** to `attacker_validation` or `final`; using the same questions at different training checkpoints is not a new independent document. Historical RAG rows without document hashes cannot be used as independent calibration data.

`--channel decision`, `--channel response_seconds`, and `--channel transcript` fit distinct observable attacks. Transcript coefficient fitting and threshold selection reserve separate validation target groups, so a 5% threshold needs at least 20 nonmember groups **in its threshold half**, not 20 repeated batches. Insufficient calibration fails explicitly. Fit separate budgets (1, 2, 4, …); do not tune the budget on final labels. All-rejected gradient calibration is unavailable; evaluate decision/timing/transcript channels instead, while preserving successful prevention.

```bash
python -m master_script.tools.analyze_guard_study \
  outputs/guard-v2-confirmation/results/BATCH_ID/000*.json \
  --output outputs/guard-v2-joined.json
```

The joined report separates batch, training-record and datastore privacy; preserves null metrics, per-target estimates and target bootstrap intervals; reports per-client guard outcomes, profile components, paired round/end-to-end costs, availability, no-context/RAG utility and four-cell differences. Baseline matching requires code revision plus all non-defense training/attack settings, including LR and probe settings. The absolute no-context F1 screening floor is 0.10; this is a provisional study choice, not a validated usability guarantee. Relative F1 ≥90%, EM loss ≤0.05, rejection ≤1%, and median round overhead ≤10% remain study targets. Dependent small samples do not pass a population rejection gate; 299 independent error-free Bernoulli observations would be needed even for a one-sided 95% upper limit below 1%.

## Research questions: what can currently be answered

| Question | Current answer / next evidence |
|---|---|
| Architecture-preserving bypass | Implemented variant passes shape rules by construction; meaningful LLM attack success and learned detection remain unverified. |
| Generalization | Grouped splits and held-out variant enforcement implemented. Needs real detector data across clients/configurations and unseen targets. |
| Repeated querying | Persistent budgets and independent-world averaging tested. Fit and compare frozen 1/2/4-query attackers on GPU results. |
| Decision/timing leakage | Independent and joint transcript attackers implemented; local synthetic timing signal detected even with all gradients refused. No real side-channel conclusion. |
| Noise-only reversed ranking | Validation learns negative direction in regression tests; pilot's four final labels were not used to refit. Independent GPU validation needed. |
| Absent-document positives | Historical raw answers were not saved. New parser, no-context control, local raw audit and continuation attack distinguish parsing/echo/refusal from response bias; exact historical cause remains unknown. |
| Training/datastore overlap | Four-cell matched differences and exposure/truncation audits implemented. Historical constant-positive cells yield zero interaction. Needs more targets and calibrated numeric scores. |
| Client disparity | Per-client losses, flags, decisions and timings recorded. No population fairness claim from the mechanics smoke. |
| Training completion/convergence | Four synthetic SGD rounds improve loss; deliberate rejection halts aggregation and survives restart. Genuine multi-round LLM result is pending. |
| Classifier value | All seven arms and a simple threshold comparator supported. No held-out real-data advantage established. |

Remaining limits include lack of GPU verification for the new attack, absence of a real trained detector artifact, no cross-run checkpoint reuse, no automatically approved recovery/resumption from aborted worlds, no public behavioral-text feature set beyond parameter features, no semantic answer-support audit, no peak CUDA-memory profiler, and no end-to-end formal privacy bound covering training plus retrieval plus transcript outputs. These are explicit limits, not implied guarantees.
