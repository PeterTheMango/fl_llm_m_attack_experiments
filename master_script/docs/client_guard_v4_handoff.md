# Guard v4: negative-result audit and public update geometry pilot

## Outcome

The frozen v3 detector failed to generalize to the architecture-preserving attack. This change adds an **opt-in diagnostic feature schema**, `parameter_structure_v2`, and a bounded fresh-target pilot. It does not deliver a demonstrated privacy defense or a fitted v2 detector. Defaults and historical v1 artifacts remain supported.

Repository inspection started at clean commit `e2b812ad8807d97543ee54bea2483ebe1019f08b`. The historical experiment fingerprint is `7d9f9493f7c3`. The implemented core fingerprint is `489525a1da38`; never resume the historical launch under it.

Review evidence: `outputs/guard_v4_review_20260926/reproduced_evidence/review.json`. The accompanying archive, dataset, detector, manifests, worker benchmark and first raw result are byte-preserved copies of the supplied inputs. The final reproduction also reconstructs the historical core fingerprint from Git blobs and confirms `7d9f9493f7c3`. `review_guard_v3.py` independently recomputes scores with scalar arithmetic and AUC with pair comparisons; it does not fit or change the detector.

## 1. What the evidence establishes

| Frozen test result | Reproduced value |
|---|---:|
| Legitimate requests rejected | 0/96 |
| Probe-head requests rejected | 8/8 |
| Causal-gradient requests rejected | 0/8 |
| Attack recall | 0.5 |
| ROC AUC | 0.6666667 |
| Independent test targets | 2 |
| Decisions equal validation-calibrated relative-delta baseline | All 112 |

The dataset has 424 rows: train 192 legitimate + 16 probe, validation 96 + 8, test 96 + 8 probe + 8 causal. Source mappings, target roles, artifact hashes, training means/scales and validation threshold all reconcile. The threshold is `0.011303227541612549`, the maximum of 96 validation legitimate scores: with a 1% empirical allowance, `floor(.01*96)=0`. The strict `score > threshold` rule accepts ties. No inspected test row selects normalization, weights, or threshold in the fitting code.

The review can directly reconstruct **16 rows from one raw result**. The other 23 raw results, original launch and configurations were not supplied locally. Their hashes and mappings reconcile across provenance/progress/splits, but their contents and the chronology of prospective declaration cannot be independently verified here. Retired weights prevent recomputing historical features. These are limits of the local audit, not a reason to silently treat metadata as full raw evidence.

### Extraction, labels and supported explanations

* `relative_delta` is the **unweighted mean of per-tensor relative L2 changes**, not the model-wide relative norm. `maximum_layer_delta` is their maximum. Here “layer” means an array in state-dict order, including biases/buffers/tied entries; it does not mean a transformer block. Ratios use the historical scale-dependent denominator floor and cap at 1e6.
* `parameter_concentration` is the maximum **incoming weight** L-infinity/L2 ratio, not update concentration. AMIProbe includes a scalar `fc2.bias`: any nonzero scalar has ratio 1. Probe test concentration is exactly 1; LM concentration is about .990979. This is an architectural shortcut available to the classifier, not evidence of membership protection.
* Observation scopes are labeled malicious by experiment construction. Training scopes are labeled legitimate. These are **request-origin labels**, not actual membership, successful leakage, or a semantic maliciousness oracle. Public calibration is excluded; causal malicious rows in train/validation are excluded. Group/seed/path/client/round/revisions remain provenance only.
* Normalization uses 208 training rows, with large probe changes dominating scale. All three learned coefficients are positive. Causal relative changes are only `3.54e-5–9.27e-5`, versus legitimate `0–.002331` and probe `1.637–2.465`. Causal detector scores `.0111873–.0112375` lie inside legitimate `.0111724–.0112961`.
* **Score overlap is not feature-vector equality.** Test concentration has a tiny opposite-direction gap (causal maximum .9909791423, legitimate minimum .9909792371). This is a post-inspection diagnostic, not a valid threshold selection. No new rule is fitted to that gap.

### Reference and attack-construction audit

| Request | Pinned harness reference | What the delta measures |
|---|---|---|
| AMIA/Reference legitimate FL training | Initial LM for that world | Cumulative training movement from initialization |
| Causal-gradient observation | Post-FL LM before attack optimization | Candidate-directed perturbation only |
| Probe-head observation | Fixed seed-0 head template | Difference between a trained head and a different template |

The causal attack performs 12 normalized candidate-loss-ascent steps at LR .005 in the declared configuration, retains the LM parameter structure and scores released gradients against a public candidate gradient. Its summed ideal step-length bound is .06 before interpolation/rounding. It has no inherited chosen-neuron AMIA guarantee. The head is trained with a different procedure and reference. Thus request type, architecture, reference stage and attack construction are confounded in the existing positive/negative comparison. The client runtime pins the harness-provided reference; this is not remote attestation or a solution to malicious-server trust establishment.

**Supported:** the frozen classifier learned the large probe/reference difference, failed on small structure-preserving changes, and adds no test decisions beyond relative delta. **Hypotheses requiring fresh data:** a per-tensor distribution, signed direction, update sparsity, or a comparable reference can separate legitimate optimization from candidate-directed requests. Neither score overlap nor this small test proves that all public-parameter features are useless.

## 2. Implemented candidate and its limits

The v2 schema retains all three v1 values and adds seven summaries:

| Feature | Definition across parameter tensors |
|---|---|
| `median_layer_delta`, `p90_layer_delta` | Median and linear-interpolated .9 quantile of the existing relative deltas |
| `mean_update_reference_cosine`, `minimum_update_reference_cosine`, `maximum_update_reference_cosine` | Mean/min/max of cosine(incoming − reference, reference) |
| `mean_update_concentration`, `maximum_update_concentration` | Mean/max of L-infinity(update)/L2(update) |

Undefined zero-update/reference cosines are 0; zero-update concentration is 0. Tensors are equally weighted, consistently with v1; localization and transformer-block identity are still discarded. Synthetic tests construct matching v1 summaries with opposite v2 directions and norm-matched sparse/dense changes. They establish additional information, not detection of the real attack.

For float16/32, new reductions share the existing pass using float64 accumulators. Wider floats/integers use scaled chunk passes to avoid overflow and underflow in ordinary extreme cases. Chunk size is 65,536 elements; strided arrays use bounded slice copies. V1 arithmetic and feature names remain supported. Scratch is O(chunk size), plus O(tensor count) scalar summaries; the full guard still owns a request snapshot and up to one reference tensor per worker. Extreme mixed dynamic ranges and quantization can erase information; no arithmetic recovers a perturbation rounded away in incoming parameters.

The client legitimately has the incoming request and a locally pinned approved reference. It does not need private records, membership labels, target identities, filenames, seeds, split roles, a candidate text or private gradients. The reference can be derived from earlier private training but is already distributed to the client; “public” here means observable input to the client, not a claim that training data became public.

The runtime now records a combined `reference_sha256` beside its request digest. Every reference tensor is freshly loaded and hashed before use. Snapshot ownership, exact dtype/shape/byte hashes, structural refusals in shadow mode, reservation ordering, duplicate-release checks, budgets and durable ledgers remain in place. Schema mismatch fails before release reservation. Schema is client-owned configuration, included in new pipeline identities; it is not supplied by server fit configuration.

Detector JSON, strict vector allowlists, assembler, explicit-label exporter, trainer, stage builder and public replay all dispatch by schema. The default remains `parameter_behavior_v1`; a v2 detector cannot silently run on v1 features. No historical detector/result was overwritten or retuned. No v2 coefficients exist yet. New extraction requires **fresh requests** because the retained v3 summaries cannot reconstruct directions or update concentrations.

Alternatives considered:

* A full fixed per-block vector retains localization but requires a pinned topology/name map and more samples. Arbitrary array indices must not become incidental model/task identifiers. Defer until the small pilot supports pursuing it.
* Direction relative to a fixed public loss gradient is closer to optimization behavior, but requires approved public text/tokenizer/loss, model forward/backward work, more memory and fresh collection. It was not implemented under the current cost constraints.
* A previous accepted server request is not automatically a trusted reference. Promoting it permits gradual drift and poisoning. A new reference policy needs independent client trust, content pins, schedule/lineage, and matched collection. Existing references are preserved and their mismatch is disclosed.

Even exact parameter inspection cannot universally distinguish legitimate and malicious uses of **identical requests**. A server may learn from an accepted unmodified-model gradient. Parameter detection cannot by itself bound batch-membership leakage, training-record leakage, or retrieval-document leakage. If fresh matched controls show no stable separability, stop developing a classifier as a privacy solution and investigate separately justified release protection/accounting.

## 3. Local verification and cost

See `tests_final_verified.txt`, `benchmark_optimized.json` and `verification.json` in the review directory. Tests cover dense numerical oracles, old-feature projection, opposite directions, benign rescaling, sparse perturbations, zeros/subnormals/extrema, strided arrays, schema/label isolation, test-data independence, owned snapshots, tampered references, duplicate accounting, reserved cohorts and no-GPU target resolution. The maintained suite passes (554 passed, 11 skipped); archived vision tests outside `tests/` could not be collected because this Python environment lacks PyTorch. GPU-dependent tests remain skipped.

Synthetic float32 CPU measurements on this Mac, 16 tensors, three repeats, separate processes per configuration:

| Request bytes | Workers | v1 full guard median | v2 full guard median |
|---|---:|---:|---:|
| 64 MiB | 1 | .208 s | .233 s |
| 64 MiB | 4 | .146 s | .172 s |
| 256 MiB | 1 | .819 s | .857 s |
| 256 MiB | 4 | .658 s | .642 s |

At 256 MiB/one worker, extraction was .560 s v1 versus .589 s v2; full-check increment was approximately 4.7%. Three repeats cannot support a speed advantage from the slightly lower four-worker number. Extraction-only `tracemalloc` peak was about **2.36 MB** at both sizes. Whole-process RSS peaks at 256 MiB were approximately 914–1,008 MB, including input arrays and snapshot preparation; these are not incremental feature memory measurements. RSS and tracemalloc cover different scopes. The full guard timings exclude the separately recorded primary audit commit, consistent with previous reporting.

`benchmark.json` retains the initial slower multi-pass implementation's measurement and fingerprint: its 256 MiB v2 medians were 2.159 s (one worker) and 2.115 s (four). The optimized report supersedes it for current code. Neither report measures a full-model FL round, GPU memory, malicious-attack success, utility or production concurrency. Historical 11.13→5.01 s worker improvement and first-job 5.067 s checks remain **v3** measurements.

## 4. Exact next remote actions — none launched here

The next remote action is CPU verification and cohort resolution. Apply the supplied patch only after reviewing the checkout; `git apply --check` must succeed. Transfer the patch and exclusion manifest from this Mac:

```bash
scp 'outputs/guard_v4_review_20260926/implementation.patch' calc08:/tmp/guard-v4.patch
scp 'outputs/guard_v4_review_20260926/evidence/splits.complete.json' calc08:/tmp/guard-v3-splits.complete.json
```

On calc08, in the existing experiment environment:

```bash
cd /home/calc08/projects/LLMPrivacy/fl_llm_m_attack_experiments
git status --short
git rev-parse HEAD
git apply --check /tmp/guard-v4.patch
git apply /tmp/guard-v4.patch
python -m pytest -q tests/test_guard_structure.py tests/test_guard_v3.py tests/test_guard_detector.py tests/test_guard_release_ledger.py tests/test_guard_integration.py
python -m master_script.tools.benchmark_guard_structure \
  --mib 64 256 --workers 1 4 --repeats 3 \
  --output outputs/guard-v4-cpu-cost.json
python -m master_script.tools.collect_guard_traces prepare \
  outputs/guard-v4-pilot --workers 4 --targets 3 --seed 4000 \
  --feature-schema parameter_structure_v2 \
  --exclude-manifest /tmp/guard-v3-splits.complete.json \
  --reserve-final-targets 4
python -m master_script.tools.collect_guard_traces resolve \
  outputs/guard-v4-pilot/launch.json
```

Use the actual SSH alias if `calc08` is not configured locally. The patch excludes local outputs; the manifest transfer is required. Preserve any unrelated remote changes. A failed patch check requires reconciliation, not a checkout reset. Use new output names if these exist. Local `pilot-preview-final/` demonstrates nine validated job configs plus four reservation configs; recreate on remote so absolute policy paths are correct.

`resolve` loads the pinned dataset on CPU and hashes target selections. It launches no subprocess training. Version-4 launches also pin the collector and stage-builder source hashes, since the core fingerprint excludes tools. It freezes three distinct development targets and four reserved final targets in `progress.json` and `cohort.json`, rejects historical/smoke overlap, repeated targets across seeds, cross-variant target changes, and final/development collisions. Repeated `--exclude-manifest` flags can include other known prior cohorts: the supplied eight targets plus the built-in smoke target are not a global history registry. If any selected target collides, stop and prepare a new directory with a fresh predeclared seed band; never rewrite an active launch. Final reservations have **no scheduled jobs**. Do not execute their YAML files directly. All four remain outcome-untouched; four is far too small for a population claim.

After CPU checks and the declaration are reviewed, this is the bounded **future GPU command**, provided but not executed:

```bash
python -m master_script.tools.collect_guard_traces run \
  outputs/guard-v4-pilot/launch.json --gpu 0 --max-jobs 1 --scratch-root /tmp
```

Inspect each job before issuing the next one. Nine jobs total: three targets × (probe, causal, paired Reference), three FL rounds/four clients, same utility protocol. The final four targets never enter this queue. Every job still checks 20 GiB free output and 4 GiB scratch; these are headroom estimates, not enforced disk quotas. Successful new jobs retire owned weights only after result/ledger verification. Failed/interrupted jobs retain weights and stop; never reset their ledger or auto-retry a private world. No historical checkpoint is deleted by these instructions.

Resources: one GPU with the same capacity used for completed v3 Qwen-0.5B jobs, one concurrent simulated FL client, four CPU validation workers, and sufficient host RAM for the ~2.52 GB request, its owned copy, live reference tensors and model/Ray processes. GPU/total RAM peaks have not been measured locally; record them on the first job. Allow roughly 10–25 minutes for an initial job as a rough historical scheduling estimate, and several hours for nine jobs; hardware/load/variant can change this. The CPU benchmark itself needs about 1 GB process memory plus filesystem cache at 256 MiB, no model download and no GPU. Dataset resolution may access the pinned dataset cache/network.

Required returned artifacts: launch, frozen protocol, all job/reservation YAMLs, exclusion manifests, `cohort.json`, progress, complete/partial splits, all result JSONs, JSONL answers, SQLite ledgers and audit timing tables, checkpoint-retirement inventories, archived Ray logs, CPU benchmark, code fingerprint/diff and device/RAM/disk observations. Return small traces rather than retired weights. These are private research artifacts; never publish raw audit data as a client response.

After completion, for diagnostic fitting only in new artifact names:

```bash
python -m master_script.tools.assemble_guard_dataset \
  outputs/guard-v4-pilot/splits.complete.json outputs/guard-v4-pilot-dataset.json \
  --feature-schema parameter_structure_v2
python -m master_script.tools.train_guard_detector \
  outputs/guard-v4-pilot-dataset.json outputs/guard-v4-pilot-detector.json \
  --feature-schema parameter_structure_v2 --max-fpr 0.01 --prevalence 0.01
```

The three-target pilot has only one target per train/validation/diagnostic-test role. Its `test` target is consumed development data once inspected. **It is not the reserved final cohort.** A fitted classifier from this pilot is exploratory and must not be deployed as a validated defense.

## 5. Prospective decisions and remaining gaps

**Stop immediately:** hash/schema/target-role mismatch, failed reservation/integrity checks, nonfinite features, unexpected storage growth or interrupted job. Preserve artifacts. Do not trade away integrity to improve timing.

**Continue to larger development only:** the complete pilot is auditable, costs are feasible, and new features show a plausible signal across targets beyond v1 and request-type shortcuts. Any apparent signal must next be checked against legitimate post-FL requests with the same reference policy, ordinary benign update trajectories, varying magnitudes and benign sparse/rescaled controls. The nine-job pilot does not resolve that reference confound by itself. If only probe-head discrimination remains, or matched legitimate requests overlap causal requests, record another negative result and stop escalating this classifier.

Before further fitting, freeze a larger development/validation cohort, a separate attacker-validation cohort, and expanded untouched final cohorts. Exclude every inspected v3 and pilot target and every reserved final target from development. Carry forward `cohort.json` via exclusion manifests. Plan any new attack family for final testing before seeing its outcomes; causal-gradient alignment is now a **known** diagnostic family, even if its examples remain held out from a particular detector fit. An operationally held-out family is not scientifically novel after this review.

Select normalization, feature subset, regularization and rejection threshold using development/validation only. Report v1 projection, magnitude-only baseline and v2 using identical fresh target roles; do not tune a v1 replacement on historical test data. Freeze artifact hashes, reference policy, attack optimization/query budgets and thresholds before final execution. Fit attacker score direction, gradient/decision/timing transcript classifiers and thresholds only on separate defended attacker-validation groups. Include adaptive low-magnitude/interpolated and unchanged-request attacks; successful release is not equivalent to successful inference.

Retain all declared gates unchanged:

* All three utility endpoints required: F1 ≥ .10, F1 ≥ 90% of a useful matched baseline, EM loss ≤ .05. First historical job no-context F1 .0608 fails; public/private RAG .2459/.5086 do not repair it. Shadow collection supplies no matched preservation evidence.
* Legitimate rejection ≤1%. Use independent target-level/cluster-aware uncertainty with explicit request sampling. Repeated checks do not create new independent trials. Even 299 independent error-free Bernoulli observations are needed for a one-sided 95% upper bound below 1%; two or four targets cannot establish this. Final cohort/sample design must be expanded prospectively for that claim.
* Median **matched FL round overhead ≤10%**, measured on fresh interleaved baseline/defended runs with identical source, workload and hardware; include process/host/GPU memory and end-to-end latency. A 4.7% increment in guard cost is not a 4.7% FL overhead result.
* A privacy advance requires improved held-out attack/utility tradeoff with uncertainty, not just request-origin recall. Report fresh-batch gradient/decision/timing leakage, paired training-record Reference leakage, and retrieval-document leakage separately. Missing/all-rejected numeric attack scores remain undefined; refusal/timing channels still matter.

Final outcomes are opened once after freezing the shortlist. A failed final gate becomes a documented failure, never a reason to tune on that cohort. If public-parameter features cannot discriminate the relevant requests without rejecting legitimate ones, they cannot support the intended protection; release mechanisms with an independently specified privacy scope must carry that burden. General privacy, utility preservation, cross-target/model generalization, full-model cost and the trust basis for deployment remain unestablished.

## Reproduce this local review

Run from the repository root, with a Python environment containing NumPy, PyYAML and pytest:

```bash
python -m master_script.tools.review_guard_v3 \
  outputs/guard_v4_review_20260926/evidence outputs/guard-v3-review-reproduced
python -m pytest -q tests
python -m master_script.tools.benchmark_guard_structure \
  --mib 64 256 --workers 1 4 --repeats 3 --output outputs/guard-v4-cost-reproduced.json
```

Choose unused output paths. The reviewer needs the historical commit in Git. Outputs record input hashes and the historical/current source identities; wall times and process RSS vary by host. `verification.json` pins this delivery's source, patch and evidence hashes.
