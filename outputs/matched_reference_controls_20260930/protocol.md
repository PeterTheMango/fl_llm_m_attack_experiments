# Matched-reference benign controls for the causal-request detector — protocol

Status: **final.** The researcher approved decisions 1–8 as drafted on 2026-09-30. Nothing has run yet, and no result exists. The machine-readable version is [`protocol.json`](protocol.json). Once approved, each launch copies it and pins its hash, as in the causal-attack validation.

## Question

Can the guard's detector flag causal gradient-alignment requests **without** flagging benign requests matched to them? A benign request is matched when it:

- is checked against the same reference (approved) model as the causal request;
- is built on the same architecture and code path;
- has a comparable perturbation size or the same step count.

This is a question about the detector's discrimination only. It is **not** a privacy question. Blocking a request in this study protects nothing, because the guard runs in shadow mode and releases everything it would block.

## Why now

- The pre-registered validation found the causal attack **effective**: Stage B mean per-target AUC 0.992 [0.981, 0.999] with the cosine score at 12 probe epochs and no guard in the loop ([results](../causal_attack_validation_20260929/results.md)). Its decision rule says to pre-register matched-reference benign controls next.
- The detector's pilot "4/4 causal detection" came from a comparison that confounds request type with reference model. The next section shows how.

## What the code shows about benign and causal requests

Every guard check computes features of `incoming − reference`, where the reference is a harness-pinned NPZ snapshot (`guard_runtime.prepare_guard`). The reference differs by request type:

| Request (pilot label) | Built by | Reference the guard compares against | What the delta measures |
|---|---|---|---|
| Legitimate FL training (benign) | FedAvg broadcast of round *r*'s global model to each client (`federated_fine_tune`, scope `training:<seed>:True`) | **Initial pretrained LM**, fixed for all rounds | Cumulative training drift from initialization |
| Causal observation (malicious) | `optimize_request`: 12 normalized ascent steps of size 0.005 on the candidate's loss, from the post-FL model W_R | **Post-FL model W_R** (`approved_parameters` in `_run_target`) | Only the candidate-directed perturbation, with summed step length ≤ 0.06 |
| Probe head (malicious) | Trained AMIA head | Fixed seed-0 head template | Trained head vs a different template |

Consequences for any detector fitted on these rows:

- **Benign rows never share the causal request's reference.** Any detector fitted on them may separate "small step from a post-FL reference" from "cumulative drift from initialization", rather than malicious from benign. The v4 handoff lists this as the unresolved confound (§5).
- **Benign requests are heavily duplicated.** All clients in a round receive the same global model, so they produce identical requests and identical features. I checked this on the retained v3 evidence (`outputs/guard_v4_review_20260926/evidence/dataset.json`). The 384 benign rows fall into 72 (source, round) groups. Each AMIA group is 4 identical events. Each Reference group is 8 events with 2 distinct vectors, one per world. So there are 88 distinct benign requests, not 384.
- **Round 1 is trivial.** In round 1 the incoming model *is* the reference, so the delta is exactly zero. All 128 zero-delta benign rows in the v3 evidence are round-1 rows.
- **The causal request is one request per target.** It is deterministic, and every attack trial of a target checks the same parameters against the same reference. The pilot's "4/4 detected" is therefore **one** decision, on one test target (seed 4002).
- **The 8.33% is 4 of 48 events on one test target.** The researcher pasted the pilot detector report (`outputs/guard-v4-pilot-detector.json.report.json`) on 2026-09-30. It is transcribed here, not committed, and its file hash was not taken.

  | Split | Benign rejected | Probe head flagged | Causal flagged | Independent groups |
  |---|---:|---:|---:|---:|
  | Validation | 0/48 | 4/4 | — | 1 |
  | Test | **4/48 (0.0833)** | 4/4 | 4/4 | 1 |

  - **Unit.** Every count is events on one target, so neither 8.33% nor 4/4 is a population rate. The 4 rejected benign events are **one request**: the round-2 broadcast of one source (`72c321e2…`) to clients 0–3, with detector score 0.012846 and relative change 0.00167. The researcher ran a read-only listing of the rejected test events on the server on 2026-09-30. The pilot test target was already consumed, so this tunes nothing.
  - **Magnitude-only baseline.** At threshold t0 = 0.0023743033104398573, relative change alone rejected 0/48 benign events on test and flagged 4/8 malicious ones. That is the probe head only: the causal request's tiny change stays under t0.
  - **D beyond magnitude.** D flagged the causal request, which t0 did not, but it also rejected 4 benign events that t0 accepted.

## Detector under test (frozen)

- **D** is the guard-v4 pilot detector: `parameter_structure_v2` logistic regression from `train_guard_detector`.
  - It was fitted on the pilot's train and validation targets (seeds 4000 and 4001), with `probe_head` as the malicious class and causal held out.
  - Its threshold is the maximum validation benign score (`--max-fpr 0.01` allows floor(0.01·n) = 0 exceedances).
  - **Pinned file:** `outputs/guard-v4-pilot-detector.json` on LCALC08 (1,552 B), sha256 `2588f09e544574e81685b00edd6bf6b9381fef930594f8f42884c66de8af1611`, pasted `sha256sum`. The report's `detector_sha256` field agrees.
  - **Fitted from** `outputs/guard-v4-pilot-dataset.json`, with the report's `dataset_sha256` `38a58a44de185e949fdc4148cb8ab3554111543b1fafb345dbcc3841cbfcd81e`.
  - `prepare` refuses any other detector hash.
- **No refit and no threshold change**, on this cohort or any other.
  - `guard_features.py`, `guard_detector.py`, `guard_runtime.py` and `guard_replay.py` are unchanged since the pilot (`git diff 25cca4d..f551ccf` is empty for all four), so D's features and decisions mean the same thing.
  - `optimize_request` is also unchanged. `causal_probe.py` changed only in its scoring telemetry.
- **Magnitude-only baseline D0**, for a secondary comparison only: `relative_delta > t0`, with t0 = 0.0023743033104398573, the `relative_delta_baseline.threshold` in D's pilot report.
- The causal-validation Stage A and Stage B cohorts are never used by this study, for tuning or for anything else.

## Requests per target

Each job trains one fresh FL world (3 rounds, 4 clients). W_R denotes the round-3 global model. Every check below runs in **shadow** mode with D pinned. Every benign control uses `check(..., reserve=False)`, so it takes no ledger debit, and no private gradient is computed or released for it.

| Arm | Construction | Reference | Distinct requests per target |
|---|---|---|---:|
| **C** (causal, validated configuration) | `optimize_request` from W_R: `probe_epochs: 12`, `probe_lr: 0.005`, `request_interpolation: 1.0` | W_R | 1 |
| **M1** (step-matched public descent twin) | Same optimizer and code path as C: 12 normalized steps of size 0.005 from W_R, but **descending** the mean loss of a public batch of 4 records. Batches b₀…b₉ are seeded and drawn from the public pool, token-prefix disjoint from the target, all client records and the adversary records. | W_R | 10 |
| **M2** (norm-matched honest continuation) | One honest FL round R+1 from W_R: the same 4 clients and training recipe (AdamW, lr 2e-5, 1 local epoch, batch 2, per-client training seed as in round R+1), FedAvg by record count, giving Δ_H. Then M2 = W_R + s·Δ_H with s = ‖Δ_C‖₂ / ‖Δ_H‖₂, where the model-wide L2 is taken over all state tensors in float64 and cast back to dtype. This is a legitimate server learning-rate choice (FedOpt-style server step size). | W_R | 1 |
| M2raw (secondary) | W_R + Δ_H, unscaled: the natural next-round broadcast | W_R | 1 |
| T (secondary) | The job's ordinary training requests: 3 rounds × 4 clients = 12 events, 3 distinct, round 1 equal to the reference | Initial LM | 3 |

What "matched" means concretely:

- **Same reference.** C, M1, M2 and M2raw must carry the same `reference_sha256` in their guard events. If they do not, the job is invalid.
- **Same architecture, dtype and code path.**
- **M1 is step-matched.** It has the same step count, step size and normalization as C, so its summed step length is identical (≤ 0.06). Its net ‖Δ‖₂ is comparable but not forced equal.
- **M2 is norm-matched.** ‖Δ_M2‖₂ = ‖Δ_C‖₂ to within float32 rounding. Its direction comes from real benign training.
- **Neither is claimed to be the population of legitimate requests.** They are the two hardest benign cases this code can build without new trust assumptions.

## Cohort

- **20 fresh targets** from the new seed band **7000–7099**, first block 7000–7019, one job per target.
- **Excluded**, all resolved on CPU before any GPU job by `collect_guard_traces resolve`:
  - the smoke target;
  - every group in the guard-v3 `splits.complete.json`. The committed copy is `outputs/guard_v4_review_20260926/evidence/splits.complete.json`; the pilot `launch.json` also carries these targets;
  - the guard-v4 pilot `splits.complete.json`, `cohort.json` and `launch.json`. `cohort.json` holds the 4 reserved-final targets, and `launch.json` carries the v3 and smoke exclusions;
  - the causal-validation `stage-a-2` and `stage-b` `splits.complete.json`. The unrun `stage-a` directory froze no targets, and its seeds lie in the 6000–6099 band, which this study avoids.
- **Reserved-final cohort.** It is read only for exclusion and never evaluated. `prepare` refuses unless at least 4 `final` groups are excluded and both causal-validation splits are supplied.
- **Collisions.** If any seed selects an excluded or repeated target, prepare a new directory with the next unallocated block inside the band, record the deviation, and never edit a launch. The causal validation hit 2 collisions in 4 seeds by chance. About 30 targets are excluded out of a 4,096-record pool, so roughly 0.7% per seed.
- **Role label** `matched_control_evaluation`. `assemble_guard_dataset` accepts only train, validation, test, attacker_validation and final, so it fails closed if this cohort is ever fed to detector fitting.
- **After analysis the cohort is consumed.** It may be used as development data for a future detector only under a new pre-registration, and never as test or final data. Later cohorts must exclude its `splits.complete.json`.

## Every job

- **Attack:** AMIA, `attack_variant: causal_gradient_alignment`, `attack_targets: 1`, `probe_epochs: 12`, `causal_score: cosine`, `probe_lr: 0.005`, `attack_batch_size: 4`, `counterbalance_trials: true`, `observation_defense: none`.
- **Attack trials:** `attack_trials: 4` (2 pairs), only so that the observation path and the collector's target check run. **Attack outcomes are not analysed and are not an endpoint.** Attack effectiveness is already established separately.
- **Threshold:** `threshold_mode: fixed`, so no public calibration batches are needed. The attack threshold is irrelevant here.
- **No RAG pipeline. No training defense.**
- **Guard:** `client_guard: {mode: shadow, policy_file: approved_training_policy.json, release_budget: 3, diagnostic: true, validation_workers: 4, feature_schema: parameter_structure_v2, detector_file, detector_sha256}`.
- **Training:** the same pinned model and dataset revisions and training recipe as the causal validation: Qwen2.5-0.5B-Instruct @ 7ae5576, squad_research @ 7b6d24c, 4 clients, 3 rounds, 1 local epoch, batch 2, lr 2e-5, max_length 128.
- **New job fields:** `matched_descent_requests: 10`, `matched_honest_round: true`. The draft's single `matched_controls` mapping was renamed at implementation; see status.md.

## Endpoints (fixed before any run)

A request is **rejected** when its guard event has `would_accept = false` for any reason. Reasons are reported. Structural reasons are not expected for any arm. If one appears, it is recorded as a deviation and still counts as a rejection.

Per target *t*:

- c_t ∈ {0, 1}: C rejected.
- m1_t = (M1 requests rejected) / 10.
- m2_t ∈ {0, 1}: M2 rejected.

Targets are the unit throughout. Duplicated events are never counted as independent.

| Endpoint | Pooled value | Interval |
|---|---|---|
| **E1: causal-request detection TPR** | mean of c_t | Exact Clopper–Pearson on targets (decision); target bootstrap also reported |
| **E2: legitimate rejection on step-matched benign (M1)** | mean of m1_t | Lower bound: target bootstrap. Upper bound: the larger of the target-bootstrap upper bound and the Clopper–Pearson upper bound on the number of targets with m1_t > 0. The second covers the degenerate all-zero case. |
| **E3: legitimate rejection on norm-matched benign (M2)** | mean of m2_t | Exact Clopper–Pearson on targets (decision); target bootstrap also reported |

The target bootstrap uses 95% percentile intervals from 10,000 resamples, with numpy `default_rng(20260930)`.

**Secondary (descriptive; they do not enter the decision):**

- **S1: ordinary training-request rejection (T).**
  - Per event, comparable to the historical 8.33%.
  - Per distinct request (`request_sha256`), by round. Round 1 is the reference itself.
  - Target-bootstrap intervals.
- **S2: M2raw rejection.**
- **S3: magnitude-only baseline D0** on C, M1 and M2. Does D add anything beyond `relative_delta`?
- **S4: within-target separability (exploratory).**
  - For each target, the AUC of D's score for C against its 10 M1 requests, and the same for each of the 10 v2 features. Each feature is reported two-sided, with its direction.
  - Mean over targets, with target-bootstrap intervals.
  - This makes the cohort detector-development data (see Cohort).
- **S5: matching checks.**
  - The same `reference_sha256` for C, M1, M2 and M2raw within each target. **Required**, or the job is invalid.
  - ‖Δ_M1‖₂/‖Δ_C‖₂ and ‖Δ_M2‖₂/‖Δ_C‖₂, the latter required to be 1 within 1e-4.
  - The candidate loss rises under C, and the public-batch loss falls under M1.
  - Failures are recorded as deviations, never silently dropped.

## Decision rule (n = 20 targets)

- **Fails: matched benign flagged.** This holds if the E2 lower bound is above 0.01, or the E3 Clopper–Pearson lower bound is above 0.01. At n = 20, the E3 condition means at least 2 targets have M2 rejected.
  - The ≤1% legitimate-rejection gate is then refuted on matched benign requests. D does not tell the causal request from benign requests built to match it.
  - Record a negative result. Stop escalating D as a causal-request control, and run no enforced comparison with D.
  - Consult S4 only to decide whether a *new*, separately pre-registered detector study is worth its cost. Per the v4 handoff, if no feature separates, stop developing a parameter classifier as a privacy measure.
- **Fails: causal requests missed.** This holds if the E1 Clopper–Pearson upper bound is below 0.50, meaning at most 5 of 20 detected. Same consequence.
- **Separates at this resolution.** This needs all three of:
  - E1 Clopper–Pearson lower bound ≥ 0.70, meaning at least 19 of 20 detected;
  - E2 upper bound ≤ 0.20;
  - E3 Clopper–Pearson upper bound ≤ 0.20.

  At n = 20, the E2 and E3 conditions mean that **no** target has any matched benign request rejected.
  - This justifies pre-registering two studies. One is an enforced, matched privacy comparison that includes adaptive requests (interpolation and bisection against D, as in `adaptive_public_requests`). The other is a larger benign-rejection study sized for the ≤1% gate.
  - **The ≤1% gate stays open.** With 20 targets and zero rejections, the upper bound is still 16.8%.
  - Detection is not protection.
- **Inconclusive:** anything else. Extending the study needs a new pre-registration, never targets added after seeing results.

The outcomes are mutually exclusive: "separates" requires zero matched rejections, and "fails: benign" requires at least one. If both failure conditions hold, both are reported.

## Not shown by this study

- **No privacy claim.** Shadow mode enforces nothing. Detecting or blocking a request is not privacy protection, and attack effectiveness stays a separate result.
- **No adaptive attacker.** An attacker who knows D can shrink or interpolate the request. That belongs to the enforced comparison.
- **Two constructed benign families**, not the population of legitimate requests. The study does not certify ≤1% legitimate rejection.
- **One model, one data set, one reference policy.** W_R is the post-FL model that the harness supplies. Whether a real client would hold W_R as an approved reference is a trust question this study does not answer; the v4 handoff warns against promoting accepted requests to references.
- **D was fitted on 2 pilot targets.** A "separates" result would not show that it generalizes beyond this setting.

## Implementation (done before any GPU run; see [`status.md`](status.md))

- **`master_script/core/attacks/matched_controls.py`** (new) builds and checks the controls.
  - M1 uses `normalized_steps`, the update rule of `optimize_request`. A test shows that ascending the candidate's loss with it reproduces `optimize_request` exactly.
  - M2 and M2raw come from `honest_round`, which runs the unguarded training client for round R+1 and applies FedAvg.
  - Each control is checked with `dc_replace(guard_runtime, scope="benign_control:<seed>:<arm>:<i>")` and `reserve=False`. The result records ‖Δ‖₂, the losses and public-batch hashes, never text.
- **`amia.py`:**
  - two opt-in fields, `matched_descent_requests` (0–100) and `matched_honest_round`. They are valid only for the causal variant with one target and no adaptive steps;
  - `_run_target` refuses them **before any training** unless a detector is pinned;
  - the M1 public records join the existing token disjointness check;
  - both LMs are parked on the CPU while the controls run.
- **Unchanged:** `guard_features`, `guard_detector`, `guard_runtime` and `guard_replay` are byte-identical to the pilot commit, and a test pins their hashes. `collect_guard_traces` and `build_guard_stage` are also unchanged (collector digest `5a33b51ec411`).
- **Fingerprint:** the core fingerprint changes from `c9f0701f97b6` to `5c53eaa5b4a3`.
- **`master_script/tools/matched_reference_controls.py`:**
  - `prepare` writes the jobs and a `guard_collection_v4` launch. It pins D and this protocol, and enforces the exclusions and the seed band.
  - `check` reports each completed job's matching checks and timing, and withholds every detector decision.
  - `analyze` computes E1–E3, S1–S5 and the decision.
- **Tests (`tests/test_matched_reference_controls.py`, 23):**
  - optimizer parity, M1 step length and M2 norm match;
  - FedAvg, and honest-round scheduling;
  - identical reference hashes and no ledger debit;
  - the assembler refusing the `benign_control` scope and the `matched_control_evaluation` role;
  - refusal of exclusions, the seed band and other detectors;
  - endpoint arithmetic and every decision branch, including the degenerate-interval rule;
  - a decision withheld on a matching failure;
  - `check` hiding decisions.
- **Full suite:** 702 passed, 2 skipped.
- The collector resolves and runs the launches unchanged. It freezes and excludes targets before any GPU work, runs one job at a time, verifies each result against its target, and retires weights.

## Budget and rough cost

- **Budget:** 20 jobs, one target each, fixed in advance. No additional targets.
- **GPU per job, estimated and not measured:**

  | Step | Time |
  |---|---|
  | FL training, 3 rounds, with shadow checks of 12 training events | ~4–6 min |
  | Causal request | < 1 min |
  | M1: 10 × 12 steps plus 10 checks | ~2 min |
  | Honest round R+1 plus 2 checks | ~2 min |
  | 4 attack trials | ~1–2 min |

  That is roughly **10–15 min per job**, so about **3–6 GPU-hours** for 20 jobs.
- **Timing check.** Record the first job's wall time. If it exceeds 25 min, pause and re-estimate. Timing never changes the design.
- **Disk:** transiently about 8–10 GB per job (federated model, causal request, and two guard NPZ references of about 2.5 GB each). The collector's 20 GiB output and 4 GiB scratch preflight checks apply, and weights are retired after verification.
- **CPU only:** `resolve` takes minutes, and `analyze` takes seconds.

## Approved decisions (2026-09-30)

1. **Detector:** evaluate the frozen v4 pilot detector D (sha256 `2588f09e…1611`), with no refit and D0 (t0 = 0.00237) as a secondary baseline.
2. **Benign controls:**
   - M1: 10 step-matched public-descent twins, batch 4.
   - M2: 1 honest next round, rescaled to ‖Δ_C‖₂.
   - M2raw and T are secondary.
3. **Cohort:**
   - 20 targets from seed band 7000–7099, first block 7000–7019;
   - the exclusion list above;
   - the collision rule;
   - the role `matched_control_evaluation`, consumed after analysis.
4. **Job settings:** 4 unanalysed attack trials, `threshold_mode: fixed`, no RAG, shadow guard.
5. **Endpoints E1–E3** and their interval rules, including the degenerate-case rule. **Secondary S1–S5**, with S4 marked exploratory.
6. **Decision thresholds:**
   - fails if the E2 or E3 lower bound is above 0.01, or the E1 upper bound is below 0.50;
   - separates if the E1 lower bound is at least 0.70 and the E2 and E3 upper bounds are at most 0.20.
7. **Implementation:** the code changes and tests listed above, with the fingerprint change accepted.
8. **Budget:** 20 jobs, about 3–6 GPU-hours.
