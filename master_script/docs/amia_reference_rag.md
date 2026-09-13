# AMIA / Reference federated RAG study

Use [`configs/amia_reference_rag_master.yaml`](../configs/amia_reference_rag_master.yaml).
It is the only active saved configuration. The 18 previous YAML files and their
shared demo study are preserved byte for byte in [`configs/archive`](../configs/archive/README.md),
with a SHA-256 manifest. The other attack implementations remain available for
historical reproduction.

## Launch

From the repository root, activate the experiment environment:

```bash
conda activate peter_experiments_fl
python -m master_script.perform_experiments --dry-run --no-firestore
python -m master_script.perform_experiments \
  --queue master_script/configs/amia_reference_rag_master.yaml \
  --attack amia --attack reference --no-charts
```

In the dashboard, select **amia_reference_rag_master.yaml**, select **amia**
and **reference**, then start the sweep or add this file to the queue and start
it. Selecting just one attack expands only that attack's variants. No second
configuration or data-preparation command is required. Copy the complete
`configs/research_data/` directory when transferring the setup to the GPU server.

The queue saves a manifest and full per-run JSON under
`master_script/outputs/queues/<batch-id>/`, including when Firestore is unavailable.
Add `--no-firestore` for local-only runs. Only `status: complete` runs belong in
attack-effectiveness analysis; report failed counts separately. Cached results
use the source fingerprint, pinned model/dataset revisions, all attack and
pipeline settings, and the RAG study hash. These changes create fresh identities.

## Experiment matrix

The model and Reference baseline are `Qwen/Qwen2.5-0.5B-Instruct`, pinned to
`7ae557604adf67be50417f59c2c2f167def9a775`. Training uses SQuAD's training split,
pinned to `7b6d24c440a36b6815f21b70d25016731768db1f`, with 32 records per client,
four participating clients, three FL rounds, and sequence length 128.
Parameters are loaded explicitly in FP32 for full-model AdamW; conversion to
Flower arrays handles bfloat16 tensors without changing integer buffers.

Each condition runs at seeds 7, 1007, and 2007:

| Attack | Conditions | Runs |
| --- | --- | ---: |
| AMIA | Baseline; gradient clipping; Gaussian gradient noise at sigma 1 and 2; BitRand at epsilon 5 and 20; OME at epsilon 5 and 20; training DP-SGD diagnostic | 27 |
| Reference | Baseline; training DP-SGD at sigma 1 and 2; central DP-FedAvg at sigma 1 and 2 | 15 |

AMIA uses five separately trained target-specific probes per run, each with
40 balanced private-batch trials. Trials record target hashes, target seeds,
local trial IDs and batch pairing. `unique_target_count` reports any repeated
record selection; do not treat all 200 batches as independent target evidence.

Reference uses 200 trials: 100 paired target-member/nonmember FL worlds. Each
world trains its own model; the reference is the unchanged base model. Larger
scores mean more likely membership: `-target_NLL / reference_NLL`.

Both attacks calibrate their decision thresholds on 200 held-out public
nonmembers at a target empirical FPR of 5%. AMIA uses public negative batches
separate from its 64 probe-fitting negatives. Reference uses record scores
under each trained model. Effective token overlaps fail validation. Calibration
never optimizes a threshold on the test membership labels. The calibration FPR
is not a population guarantee; report the measured test FPR and TPR as well.
A threshold can legitimately produce one-class predictions if the scores do
not separate; that outcome must remain visible.

## What each defense protects

- **AMIA gradient clipping/noise:** the client computes the actual next-token
  cross-entropy gradient, clips the entire outgoing probe-gradient vector, and
  optionally adds fresh Gaussian noise before the server observes it. Clipping
  alone is a diagnostic ablation. Gaussian releases carry a conservative
  replacement-adjacency composition bound over all observation trials in the
  run. Noise is independent of the public experiment seed.
- **AMIA BitRand / OME:** feature randomization is also used while fitting the
  target probe. This lets the attacker adapt to the mechanism. The implemented
  privacy scope concerns the randomized representation conditional on the
  held-out label; it does not establish privacy of the full text and label.
- **Training DP-SGD:** clips per-sequence training gradients before adding
  Gaussian noise and taking AdamW steps. This is relevant to Reference's
  access to released trained models. Its AMIA condition is explicitly a
  diagnostic: private training does not privatize a later fresh probe response.
- **Central DP-FedAvg:** clips client updates and adds noise to their mean.
  This assumes a trusted aggregator; it is therefore included for Reference,
  not as protection against AMIA's observing server.

Accounting uses conservative Gaussian zCDP composition with no sampling
amplification. Inspect the reported epsilon/delta rather than interpreting
sigma 1 or 2 as a small privacy budget. Training and observation bounds have
different release scopes. They exclude private audit labels, calibration audit
outputs on a private model, probe-fitting outputs, retrieval responses and
composition across separate experiments. The local results and model artifacts
are research audit material, not a protected release to an attacker.

These are FL/LLM adaptations, not reproductions of paper benchmark numbers.
The attack/defense rationale follows [Nguyen et al.'s active membership inference](https://proceedings.mlr.press/v206/nguyen23e.html),
[DP training](https://arxiv.org/abs/1607.00133), and
[evaluation at controlled false-positive rates](https://arxiv.org/abs/2112.03570).

## RAG evaluation and interpretation

The prepared study uses public SQuAD validation data: 256 documents in each of
two disjoint simulated public/private corpora, 200 membership candidates, and
20 utility queries. Each utility query is evaluated against its supporting
corpus. The source file, selection provenance and SHA-256 hashes accompany the
study. “Private” denotes the experimental datastore role; this study contains
public benchmark material.

Every evaluated model has eight RAG conditions: both corpora crossed with
ordinary retrieval, Mirabel retrieval filtering, a refusal instruction, and
Mirabel plus that instruction. A no-retrieval utility baseline is also measured.
The refusal instruction is an experimental prompt defense, not a formal
privacy mechanism. Mirabel follows the existing statistical retrieval-filter
adaptation. The datastore test uses the
[Anderson et al. black-box membership question](https://arxiv.org/html/2405.20446v2).
It is distinct from membership in an FL training partition or an AMIA batch.

For Reference, `evaluation_trials: 2` evaluates RAG on the first paired FL worlds
per run. Training privacy evidence is collected for every world, including
worlds whose RAG generation is skipped. For AMIA, RAG runs once per trained target
model, not once per batch observation. The RAG sample is consequently much
smaller than the Reference attack sample.

Results record answer recognition, member-document retrieval rate before
context truncation, exact match, token F1, answer NLL, and elapsed time. The
source scoring convention counts an unrecognized answer as nonmembership, but
`rag_validity` flags an ordinary baseline with no recognized membership answers
or no positive utility overlap. This flag is only a basic validity screen;
a single recognized or partially correct answer is not proof of useful RAG.
Compare recognition rates and useful-answer performance quantitatively.

Keep three outcomes separate when comparing defenses:

1. AMIA batch-membership inference, clustered by target and seed.
2. Reference training-record inference, clustered by paired world and seed.
3. Datastore inference with RAG utility and answer-recognition diagnostics.

The historical `metrics.adv` field is balanced accuracy, with chance at 0.5;
conventional membership advantage is `TPR - FPR = 2 * adv - 1` for balanced
classes. Also report AUROC, measured FPR, TPR and confusion counts. Pooling raw
scores across differently calibrated probes/models can hide scale differences;
inspect per-target trials and calibrated operating-point results. A low attack
score accompanied by unusable answers is not evidence of a useful defense.
Neither these settings nor the earlier small runs establish which attack is
reliable: that is the empirical question for this sweep.

## Workload and a smaller validation run

The full file expands to 42 runs, with 135 AMIA FL models plus 3,000 Reference
FL models, or 37,620 client training fits. It also fits 135 malicious probes and
performs 8,400 attack trials. There are 165 RAG model evaluations, totaling
approximately 280,500 answer generations with the included study. Full-model
FP32 optimization and retained AMIA checkpoints need substantial GPU, host
memory, disk space and time. GPU capacity and throughput must be verified on
the actual server; no full Qwen/Flower sweep was run during implementation.

For an initial execution check, temporarily edit this same file:

- Change the shared seed list to `[7]`.
- Set `defaults.attack_trials: 2` and AMIA `attack_targets: 1`.
- Set `defaults.federated_rounds: 1` and AMIA `probe_epochs: 2`.
- Retain only each attack's baseline and one defense variant of interest.
- Keep the calibration count at 200; reducing trials does not justify fitting
  thresholds from the test labels. RAG still evaluates the supplied study.

Those settings check execution, not attack reliability. Restore the research
settings for the study. The loader requires balanced AMIA trial counts per
target and rejects invalid calibration resolution before compute.

## Persistence and verification

Local JSON always contains the full trial detail. Large Firestore results keep
summary metrics as normal fields and store detailed arrays losslessly in a
checksummed `detail_storage` zlib/base64 JSON field. The cache and dashboard
restore those arrays automatically. For analysis, use full local JSON or
`core.result_storage.unpack_result` on an exported document; flattened CSV
summaries alone cannot reconstruct clustered trial evidence. A conservative
size check preserves local output if a custom result remains too large after
compression. This addresses Firestore's
[1 MiB document limit](https://firebase.google.com/docs/firestore/quotas).

Verification includes offline tests of configuration expansion and selection,
archive hashes, nonmember calibration, actual PyTorch loss gradients, clipping
and fresh noise, bfloat16 checkpoint loading into FP32, target/trial accounting,
RAG response validity, large-result roundtrips, and CLI/dashboard launch paths.
Final local verification: **428 tests passed** (one dependency deprecation warning);
the queue dry run expanded **42 pending runs**. Model-based efficacy and GPU
resource sufficiency require the server runs.

## Reference run stopped by Ray's host-memory guard (13 September 2026)

The reported node used 59.84 GB of 62.79 GB. Ray killed a 12.36 GB client
worker while the experiment driver used 43.44 GB. That caused the incomplete
private round; Flower then surfaced its generic `Exception in ServerApp thread`.
The log shows the sweep continuing after failed runs. These failures are
infrastructure failures, not evidence that Reference resisted an attack.

The memory correction removes several avoidable model-sized allocations:

- DP-FedAvg reads one serialized tensor at a time, using two passes to compute
  each client's global clipping norm and the uniform noised mean. Float64 delta
  scratch space is chunked. It emits one aggregate instead of encoding an
  identical noised model for every client and averaging those copies again.
- DP-SGD no longer retains an unnecessary previous-model copy. Model capture
  shares the serialized aggregate, and final loading copies tensors individually.
- Each newly owned Ray runtime is shut down after its simulation, including on
  failure. Trial boundaries collect cyclic garbage, release unused CUDA cache,
  and ask glibc to return free heap memory where supported. Logs show the trial
  number and driver RSS at each trial boundary.
- `sim_max_concurrent_clients: 1` bounds the local simulation to one worker.
  The same four clients still contribute to every round. This uses Flower's
  [documented simulation resource controls](https://flower.ai/docs/framework/1.25/en/how-to-run-simulations.html).
  A capped run requires a fresh local Ray runtime; it refuses to silently reuse
  a runtime whose resource limit cannot be changed.

The experiment count, targets, rounds, DP clipping/noise settings and rejection
of incomplete rounds remain unchanged. Noise stays freshly randomized. Numerical
regression tests compare aggregation with the previous mathematical formula.
New failures retain attack/pipeline identity in Firestore; historical failure
records are not rewritten.

Stop the failed queue using its normal Stop control, synchronize these code
changes to the server, and start a fresh experiment process. Do not stop unrelated
Ray jobs on a shared node. Then preview or rerun Reference alone:

```bash
python -m master_script.perform_experiments \
  --config master_script/configs/amia_reference_rag_master.yaml \
  --attack reference --dry-run --no-firestore

python -m master_script.perform_experiments \
  --queue master_script/configs/amia_reference_rag_master.yaml \
  --attack reference --no-charts
```

The preview should show 15 Reference runs. Changes to the implementation and
configuration create new result IDs. Failed trials restart; this is not a
checkpoint resume. Keep the new server RSS logs to verify memory remains bounded
across trials. More RAM may still be needed for the actual server workload.
Do not disable [Ray's memory guard](https://docs.ray.io/en/latest/ray-core/scheduling/ray-oom-prevention.html)
as a substitute for reducing allocations.

A host-only synthetic benchmark (three million parameters, four clients,
separate processes) measured 325.6 MiB peak RSS for the previous aggregation
path and 257.2 MiB for the replacement. This does not predict Qwen's peak RAM.
The full Linux/Ray/CUDA workload has not been rerun locally; Ray itself is not
installed in the local test environment, so runtime ownership tests use a stub.

Memory-fix verification: **445 tests passed**, one dependency deprecation
warning; the Reference-only dry run expanded **15 pending runs**.
