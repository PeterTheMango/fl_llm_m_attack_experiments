# Queue, defenses, and RAG: implementation survey

Status: pre-implementation survey retained as a historical baseline. See the
implementation report in this directory for the delivered behavior.
Survey date: 2026-09-07 (America/Toronto).

## Actual execution path

`perform_experiments.main` accepts one `--config`, loads it through
`core.yaml_config.load_config_file`, optionally prints a dry run, selects a GPU
before torch imports, configures logging, then calls `_run`.
`_run` publishes the optional Firestore manifest and brackets execution with a
`RunStateReporter`. It calls `core.runner.run_sweep` for sequential execution;
`--max-parallel 2` instead starts GPU-pinned CLI subprocesses.

`run_sweep` iterates `(config, spec)` pairs in order, calls observer hooks, and
continues after ordinary exceptions with a failed result and best-effort Ray
shutdown. Interrupts propagate. `run_single_experiment` checks the Firestore
cache, creates the artifact directory, trains and scores trials, builds metrics,
persists, and optionally cleans artifacts after successful persistence.

The dashboard has its own `SweepWorker` process and passes the same pairs to
`run_sweep`; `start_sweep` accepts one config filename. It rejects another launch
while active. The parent owns process-tree cancellation. A queue should reuse
these boundaries rather than introduce another experiment executor.

## Config inventory

| Existing file | Expanded runs |
| --- | ---: |
| all_datasets_master.yaml | 77 |
| baseline_master.yaml | 11 |
| client_lr_sweep_master.yaml | 33 |
| client_participation_sweep_master.yaml | 33 |
| example_sweep.yaml | 18 |
| federated_rounds_sweep_master.yaml | 33 |
| gpt2_master.yaml | 11 |
| high_clients_master.yaml | 11 |
| high_trials_master.yaml | 11 |
| local_epochs_sweep_master.yaml | 33 |
| max_length_sweep_master.yaml | 33 |
| seed_sweep_master.yaml | 55 |
| smoke.yaml | 9 |

All files parsed successfully through the actual loader. The schema is
`defaults` plus `attacks.<name>.base` and `attacks.<name>.sweep`. Shared defaults
silently omit fields absent from a particular attack; base/sweep unknown fields
are errors. Sweep products preserve the mapping order. Dataset names and the
fixed result collection are validated. The smoke config includes nine toy
attacks; AMIA and LOSS require real models and have no toy path.

## Compatibility constraints and integration seams

- Eleven frozen dataclasses contain 21–24 fields each. Their serialized payloads
  determine historical run IDs. The three hash formulas must remain unchanged
  for existing experiments; enabled defense/RAG experiments need distinct IDs.
  Adding fields to the marker base would break existing defaults and hashes.
- Nine attacks share `runner.run_attack_trial` and the federation functions.
  `ScoreContext` passes a target bundle, candidate, config, and optional reference.
  Scorers directly invoke model forwards, token likelihoods, and generation;
  merely concatenating retrieved text would also change what gets scored.
- AMIA and LOSS have separate Flower client training implementations and custom
  trial/payload adapters. AMIA trains once, then trains an attack probe and
  measures probe gradients; it is not interchangeable with datastore MIA.
  LOSS retrains per membership trial and calibrates against held-out losses.
- The shared real path uses full-parameter AdamW client training and unweighted
  FedAvg (`num_examples=1`). Local optimization is the record-defense seam;
  client deltas/aggregation are the client-defense seam.
- Real dataset worlds use paired seeds and differ in target inclusion. Retrieval
  corpus membership, training membership, and held-out utility need explicit
  definitions before adding RAG. Existing dataset formatting flattens QA rows
  into question/answer strings, so a new QA evaluation should retain its split.

## Outputs and state

The authoritative results collection is `ami_federated_llm_results`; LOSS also
reads its former collection for cache compatibility. Session logs and per-run
logs live under `master_script/logs`; charts under `outputs/Charts`; model/probe
artifacts under `artifacts/<attack-root>/<run-id>`. Local-only mode returns full
results in memory but does not persist a full JSON result file.

Between runs, the process can retain global RNG state, cached dataset pools,
Firebase clients, imported model libraries, and Ray state. Trial-level reseeding
and failure cleanup already exist. The dashboard's reporter is transient, not a
durable queue checkpoint. Restart currently means resubmitting a config and
reusing completed Firestore results.

## Verification preflight

The shell's default Python lacks torch, transformers, Flower, Ray, and Opacus.
The existing `peter_experiments_fl` Python 3.11 environment has torch,
transformers, Flower, pytest, and YAML, but lacks Ray and Opacus. A distilgpt2
cache directory exists; completeness has not yet been checked. No experiment
was launched and no existing result or config was modified during this survey.

Meaningful verification must retain the hash-equivalence and toy-parity tests,
exercise failure ordering/cancellation and malformed batch validation, and
separate synthetic orchestration checks from real LLM defense/RAG evidence.

## Design-stage questions (subsequently resolved below)

The survey raised queue scope and failure semantics, persistence, reference
library location, privacy unit, RAG phase/corpus/target, and verification runtime.
The source log now resides at
`written_paper/reference_papers/queue_defense_rag_sources.md`.

Baseline verification: `python -m pytest tests/test_hash_equivalence.py
tests/test_yaml_config.py tests/test_config.py -q -p no:cacheprovider`
passed all 87 tests. This verifies the existing config/hash baseline only;
it is not verification of the requested new features.

## Accepted design

The user answered “Use all recommended options.” This selects CLI and dashboard
queues; upfront batch validation, supplied order and continued execution after
ordinary run failures; full local JSON and optional Firestore; record-level DP
training and fixed-clipping client-level DP-FedAvg baselines; inference-only
RAG with public/private corpus controls and separate training/datastore
membership evaluation; and Mirabel as the empirical retrieval defense. Sources
are consolidated into the existing `written_paper/reference_papers` library.
No new adaptive defense is claimed: these are established comparison mechanisms.
