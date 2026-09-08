# Experiment queue, FL defenses, and inference RAG

Implemented and verified September 7–8, 2026. This report describes the new
opt-in pipeline; the pre-change inventory is in
[the survey](queue_defense_rag_survey.md). Academic IDs below resolve in the
[reference log](../../written_paper/reference_papers/queue_defense_rag_sources.md).

## Scope and decisions

After the repository survey and primary-source research, the user answered
“Use all recommended options.” The accepted design adds CLI and dashboard
queues, validates the full batch before execution, runs FIFO with one active
experiment, continues after ordinary failures, and saves full local JSON plus
optional Firestore results. It implements sequence-level private optimization,
fixed-clipping client DP-FedAvg, and inference RAG with public/private corpora,
separate training/datastore membership measurements, and Mirabel filtering.
The reference log was consolidated into the existing
`written_paper/reference_papers` library as accepted by the user.

These are established baselines for the eventual adaptive-defense study. No
novel defense or improvement in real-world privacy/utility is claimed.
Adaptive private clipping and DPVoteRAG were researched but not selected.

## Queue and preservation of existing behavior

`core/queue.py` loads every requested YAML through the existing loader,
expands its attacks/sweeps in order, and snapshots parsed settings, raw RAG
study bytes, source paths, and SHA-256 digests. An invalid later file prevents
all computation. `run_batch` uses `core.runner.run_sweep`, which already handles
sequential execution and Ray cleanup after ordinary failures. There is no new
executor, database, queue daemon, or automatic retry/resume subsystem.

Each invocation allocates a fresh timestamp/UUID directory beneath
`master_script/outputs/queues`. Atomic `manifest.json` updates record each
entry's start/end time and status. Numbered JSON files retain full results or
failure envelopes. Per-run artifact directories are isolated even when input
files repeat. The queue continues after a failed experiment; Ctrl-C stops it
and records pending entries as `not_run`. The dashboard terminates its owned
process tree and then records interrupted state. Uncatchable machine/process
failure can leave a manifest marked running; it is not a resume checkpoint.

Example, from the repository root in an environment with training dependencies:

```sh
python -m master_script.perform_experiments --queue \
  master_script/configs/pipeline_zlib_dp.yaml \
  master_script/configs/pipeline_min_k_dp.yaml \
  master_script/configs/pipeline_min_kpp_dp.yaml \
  --no-firestore --no-charts
```

`--dry-run` validates and lists runs without creating queue output.
`--queue-output` selects another parent directory. Queue execution rejects
`--max-parallel 2`; the existing single-config parallel mode remains available.
On the dashboard, choose **Launch → Existing config**, select files with
**Add to queue**, inspect the numbered list, and select **Start queue**.
The launch status exposes progress and the local result directory. Local JSON
is readable independently of Firestore; the existing Results page remains
Firestore-backed and now displays persisted pipeline metrics.

Without `pipeline`, the frozen attack dataclasses, original run-ID formulas,
cache collections, scoring algorithms, and single-config behavior remain
unchanged. Enabled settings live on `AttackSpec`, outside historical config
serialization. `pipeline_v1_*` identities include the legacy key, defense/RAG
settings and the study content hash; moving identical study bytes does not
change identity. Spawned CLI workers receive the same study snapshot and
remove their temporary files. The original 13 YAML configs and previously
existing results were not edited. New smoke configs and queue outputs are
separate additions.

Queue and pipeline runs retain completed local computation if optional remote
cache reads or writes fail; error metadata makes this visible. The unchanged
legacy standalone path retains its previous remote-error behavior. Full local
results are written before attempting a remote save.

## Training mechanisms and exact privacy scope

`core/defenses.py` supplies two mechanisms to the shared trainer and the
separate AMIA and LOSS trainers:

| YAML mechanism | Implemented operation | Supporting theory |
| --- | --- | --- |
| `dp_sgd` | Compute one causal-LM sequence gradient at a time, L2-clip the full trainable parameter vector to C, sum, add independent Gaussian noise with standard deviation σC to each coordinate, divide by public batch size, then apply AdamW. | Abadi et al. [abadi2016](https://arxiv.org/abs/1607.00133v2), private language-model optimization in Li et al. [li2022](https://arxiv.org/abs/2110.05679v6). |
| `dp_fedavg` | Subtract the preceding global model from each selected client's update, clip the full floating-point delta to C, average uniformly over m clients, and add Gaussian noise with standard deviation σC/m. | User-level private language-model aggregation in McMahan et al. [mcmahan2018](https://arxiv.org/abs/1710.06963v3). |

The `dp_sgd` name denotes the private-gradient mechanism; its optimizer is
AdamW, matching this repository's training family. Microbatching computes
actual per-sequence gradients without a ghost-clipping dependency. It trades
speed for simple, inspectable correctness and is not a high-throughput LLM
implementation. Padding tokens are excluded from private training labels.
Client DP-FedAvg deliberately uses uniform weighting, including in LOSS's
opt-in client defense, to make its sensitivity and noise normalization explicit.

Accounting uses Gaussian zCDP with **fixed-cardinality replace-one adjacency**.
For a clipped sum, replacement sensitivity is 2C. Therefore one release has
ρ = 2/σ². Public averaging scales sensitivity and noise equally. Costs add
across repeated releases, and conversion uses
ε = ρ + 2√(ρ log(1/δ)), following Bun and Steinke
[bun2016](https://arxiv.org/abs/1605.02065). The implementation uses no sampling
amplification; the existing shuffled batches and fixed-size client selection
are not treated as Poisson sampling. This avoids the sampling mismatch
highlighted by sampled-Gaussian accounting
[mironov2019](https://arxiv.org/abs/1908.10530).

Record accounting accumulates optimization steps separately for every client
across rounds, then takes the worst client; fixed disjoint partitions permit
this parallel composition. Client accounting accumulates released rounds.
The result also conservatively composes trained models across the experiment's
trials. Missing accounting evidence fails instead of reporting ε = 0.
Fresh noise seeds are independent of the public experiment seed. Identical
configs therefore do not reproduce identical private model weights.

The record guarantee concerns one training sequence at a fixed public
partition/index, not all records of one person, and not an entire client's
local-DP guarantee. Partition sizes, selection, hyperparameters, record
boundaries and step counts are assumed public and data-independent. Client
DP-FedAvg protects noised global models from downstream observers; it trusts
the server, which sees raw client updates. It does not protect against a
malicious server or implement secure aggregation. Nonfloating public buffers
are retained. Partial/failed aggregation is rejected.

Private training histories suppress unnoised loss diagnostics. However,
**the complete research output is not a DP release**: attack scores, labels,
AMIA's separately trained probe and probe loss, private retrieval outputs,
and data-dependent error/timing behavior are outside the training bound.
The guarantee also excludes pretraining exposures. Repeated experiments on
overlapping data require further composition; the per-experiment ε is not a
study-wide privacy ledger. Normal PyTorch/NumPy floating-point pseudorandom
Gaussian samplers are used, without hardened cryptographic or finite-precision
DP sampling. These are research implementations, not deployment certification.

## RAG and membership evaluation

`core/rag.py` performs inference-only retrieval with a frozen MiniLM encoder,
attention-mask mean pooling, L2 normalization and cosine similarity. This
follows sentence embedding practice
[reimers2019](https://aclanthology.org/D19-1410/). Retrieved text is concatenated
into a causal-LM prompt; this is an adaptation of retrieval-conditioned
generation, not the original jointly trained seq2seq marginalization in
[lewis2020](https://arxiv.org/abs/2005.11401).

The same trained generator is evaluated under four conditions:

1. Public corpus, ordinary retrieval.
2. Public corpus, Mirabel detection/hiding.
3. Private corpus, ordinary retrieval.
4. Private corpus, Mirabel detection/hiding.

A fifth no-context utility control measures the generator without retrieval.
Public/private status is a study declaration, not an automatic privacy
classification. Corpora and utility questions are supplied in one validated
JSON file. Document and candidate rows use `{id, text}`; utility rows use
`{id, question, answer}`. The top-level keys are `public_documents`,
`private_documents`, `membership_candidates`, and `utility_queries`. Each
corpus needs at least three unique documents and both candidate membership
classes. Conflicting IDs/texts and duplicate document text are rejected.
Exact utility overlap with training strings is checked; semantic overlap,
near-duplicates, pretraining overlap and provenance require study curation.

Mirabel follows Choi et al.'s Algorithm 1
[choi2025](https://aclanthology.org/2025.findings-emnlp.438/): exclude one maximum
similarity, estimate the remaining mean/standard deviation, compute the
Gumbel-based cutoff at the configured significance, and remove the suspect
maximum before filling top-k. It is an empirical query-sensitive retrieval
defense; its assumptions can fail for small or non-Gaussian similarity
populations. It does not provide differential privacy.

The datastore attack uses Anderson et al.'s black-box membership question
[anderson2024](https://arxiv.org/abs/2405.20446): present the candidate in a query
and ask whether it appears in the context. The attacker receives generated
text only; neither retrieval IDs nor similarities are attack features. Yes/no
is parsed, with an unrecognized answer treated as nonmember, and recognition
is logged. This binary score supports only a coarse ROC curve. The ordinary
training attacks retain their original model/probe access and score the
trained generator separately. Training membership and retrieval membership
are distinct variables, as motivated by
[huang2023](https://aclanthology.org/2023.emnlp-main.921/).

Datastore trials record candidate indices, corpus membership, exact training
membership, predictions and scores. They do not serialize raw study text or
generated answers. Utility includes normalized exact match, token F1 and
answer-only negative log-likelihood: prompt and context labels are masked.
Prompt construction reserves model context for generated/answer tokens and
fails on an overlong question rather than silently removing the target.
Each condition also records timing and the number of filtered queries.

Training metrics add ROC-AUC, TPR at 1% FPR, class counts and attainable FPR
resolution, following the evaluation concern in
[carlini2022](https://arxiv.org/abs/2112.03570). Adjacent member/nonmember trials
use paired corpus seeds in enabled shared/LOSS runs. AMIA preserves its
single-model/probe protocol and evaluates RAG once on that model. Neither
this addition nor a good datastore score establishes training privacy.

## Configuration

Existing `defaults`, attack `base`, and attack `sweep` mappings continue to
work. Add one shared pipeline, or override `attacks.<name>.pipeline`:

```yaml
pipeline:
  defense:
    mechanism: dp_sgd  # none, dp_sgd, or dp_fedavg
    clip_norm: 1.0
    noise_multiplier: 1.0
    delta: 0.00001
  rag:
    study_file: pipeline_demo_study.json
    embedding_model: sentence-transformers/all-MiniLM-L6-v2
    top_k: 2
    max_context_tokens: 128
    max_new_tokens: 8
    significance: 0.05
```

Study paths are relative to the YAML file, including dashboard editor
validation. Unknown pipeline fields and invalid numbers fail before training.
An omitted/empty pipeline or `defense: {mechanism: none}` without RAG uses the
legacy path. Enabled pipelines require real models; toy defense/RAG claims are
explicitly rejected. Use separate configs for mechanism comparisons; no new
sweep framework or attack-aware adaptive optimizer was added.

## Verification evidence

The full regression suite passed **411 tests**, with one existing FastAPI/
Starlette deprecation warning, before the final optional-cache error test was
added. After the final cache-failure handling change, all **62 targeted runner, queue,
pipeline and editor tests** passed, including the added remote-read/write failure
case. After tightening terminal-state handling, all **15 dashboard worker tests**
passed, including refusal to mark a still-live worker interrupted.
`git diff --check` also passed. Coverage includes all
historical hash/config contracts, queue FIFO and failure/interrupt semantics,
upfront validation and immutable input snapshots, true per-record clipping,
replacement sensitivity, cross-round accounting, Mirabel hide-and-refill,
answer-label masking, editor-relative study paths and spawned-pipeline identity.

A real CLI batch and a real dashboard batch each completed all three distinct
attack configs with defenses and RAG enabled. The dashboard batch is
[20260907-232243-cae61b4e9a65](../outputs/queues/20260907-232243-cae61b4e9a65/manifest.json).
Its recorded end/start timestamps prove sequential, nonoverlapping execution:

| Attack | Defense | Completed duration | Models / RAG conditions per model |
| --- | --- | ---: | --- |
| zlib | sequence DP | 118.25 s | 2 / 4 |
| min_k | client DP-FedAvg | 66.68 s | 2 / 4 |
| min_k_plus_plus | sequence DP | 58.73 s | 2 / 4 |

All used `sshleifer/tiny-gpt2`, two clients, one FL round, two membership trials,
CPU training, the cached MiniLM encoder, six documents per corpus and two
utility questions. Each trained model had the conservative bound ε ≈ 11.597
at δ = 10⁻⁵; composing the two models gave ε ≈ 17.572. These deliberately loose
smoke-test bounds are not suggested privacy settings. Every datastore AUC was
0.5 and answer exact match/token F1 were zero; the tiny generator did not
produce useful answers. Training AUCs of 0 or 1 from one positive and one
negative are uninformative. Mirabel filtering and answer likelihood paths were
exercised, but no defense effectiveness or preserved utility was demonstrated.

The verification environment was an isolated temporary Python 3.11 venv over
`peter_experiments_fl`, with Flower simulation 1.33.0 / Ray 2.55.1, PyTorch 2.13
and Transformers 5.14.1. Public model weights were cached under
`/private/tmp/mitacs-pipeline-hf`; the existing conda environment was not
modified. Flower's existing `run_simulation` API emits a deprecation warning
but completed successfully. Verification used `--no-firestore`; real remote
writes were not tested. Mocked regression tests cover remote failure behavior.
JavaScript syntax and Python compilation were checked; the dashboard queue
was exercised through its actual controls and visually inspected.

## What remains for the research study

The requested infrastructure is a baseline for controlled experiments. The
novel adaptive defense remains research work: select adequately sized models
and curated public/private corpora, independently define training/datastore
membership, calibrate attacks on held-out data, choose meaningful privacy
budgets, and compare defenses at matched utility across seeds. Low-FPR
conclusions need substantially more nonmembers. Consider public retrieval
first when interpreting model DP; private retrieval needs a separate privacy
mechanism if released answers require formal protection. Adaptive private
clipping [andrew2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/91cff01af640a24e7f9f7a5ab407889f-Abstract.html)
and private vote aggregation [koga2025](https://arxiv.org/abs/2412.04697v3) are
logged alternatives, not implemented or claimed as this study's new defense.

## Verification handoff, September 8

At the user's request, further local verification was stopped and moved to the
[remote-server instructions](remote_verification.md). The additional local batch
[20260908-005828-ca79071e2f1c](../outputs/queues/20260908-005828-ca79071e2f1c/manifest.json)
completed AMIA with sequence DP and LOSS with client DP-FedAvg, both with RAG.
AMIA produced one accounted model (three steps, ε ≈ 22.623 at δ = 10⁻⁵);
LOSS produced two model evaluations. Its final no-defense control failed at
Ray startup with a node/GCS timeout. The failed run was retained in the batch
manifest; the process was interrupted and exited. That control has not been
verified locally. No retry was launched after the user requested the handoff.
The earlier two three-attack batches remain successful verification evidence.
