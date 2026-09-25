# Guard cost, bounded collection and explicit utility endpoints

This continuation implements the three next steps after the September 24 paired run and pretrained utility control. The observed 70.8% round overhead, successful architecture-preserving probe, and failed no-context utility floor remain recorded failures. No GPU effectiveness result or learned detector is fabricated by this change.

## 1. Concurrent checks with unchanged integrity requirements

`client_guard.validation_workers` accepts 1, 2 or 4 and defaults to 1. It is a client-owned harness setting, never a server `fit_config` override. Counts above one appear in pipeline identity and audit events. Request copying remains a fresh owned snapshot. Hashing independent tensors and inspecting independent reference layers can execute concurrently; returned hashes and feature reductions retain tensor order. Each reference layer is still freshly read once and checked against its pinned content digest on every request. There is no checkpoint cache, timestamp-only approval, omitted feature calculation or bypass after an integrity error.

One worker retains the serial behavior. Up to one owned reference layer per worker can be live at once, in addition to the request snapshot. Concurrency trades CPU/RAM pressure for possible lower elapsed time; it is opt-in until measured on the actual machine. NumPy/ZipFile serializes shared-file reads as needed while hashes/reductions can overlap. No claim is made that this alone reaches the 10% FL-round target.

`seconds` is the total guard wall time, excluding the separately recorded audit commit, as before. `reference_validation_wall_seconds` measures the combined reference section. Reference entries in `stages_seconds` **sum task elapsed times and overlap with multiple workers**; do not divide their sum into total wall time to produce a time breakdown. The event explicitly labels that distinction. RSS fields remain samples, not peak memory measurements.

First remote command, using the recent paired run whose snapshots were retained:

```bash
run_dir=outputs/guard-v2-paired-check-20260924-095524-b38414b5/results/20260924-095524-72a408e220e8
python -m master_script.tools.benchmark_guard_workers \
  --run-dir "$run_dir" --workers 1 2 4 --repeats 2 \
  --output "$run_dir/guard-workers-$(date +%Y%m%d-%H%M%S).json"
```

This is six CPU checks on the initial approved checkpoint, in forward/reverse worker order. It verifies snapshot provenance and identical reported request hashes/features, keeps the historical snapshot/ledger read-only, and uses a temporary isolated ledger. It loads no model, accesses no private records, allocates no GPU and saves no checkpoint. Based on the preceding roughly 11-second serial checks, allow approximately one to two minutes plus loading; actual timing depends on hardware/load. Return the small `guard-workers-*.json` before selecting a worker count for GPU collection. Initial weights alone cannot establish malicious/late-round feature equivalence; nonzero updates, shape/numerical failures and reference tampering are separately tested locally. Actual GPU round overhead must still be checked on fresh matched controls after choosing the configuration.

## 2. Storage-bounded independent-target collection

The new collection tool prepares **24 one-target jobs** by default: eight seeds, each with shadow probe-head AMIA, shadow architecture-preserving AMIA, and paired-world Reference. Each AMIA job has four observations; each Reference job has two training worlds. The settings remain three FL rounds, four clients and 12 probe epochs. Model checkpoints are not reused across targets.

Seeds are assigned roles before execution: four train, two validation, two test. All variants/worlds for one target retain one role. The complete target identities are resolved on the remote host before private work. Repeated target selection under different seeds, cross-role collisions, attack variants selecting different targets, overlap with the known smoke target, or changes on later invocations cause an explicit stop. Model/dataset revisions remain pinned. Additional already-used target exclusions should be added to a new launch before its first execution if other cohorts exist; the built-in exclusion is the known September smoke target, not a global experiment registry.

`causal_gradient_alignment` remains the held-out malicious variant for detector fitting: its train/validation malicious rows are excluded by the existing dataset assembler, while legitimate training rows remain usable. This tests whether a detector trained on the known probe generalizes to the architecture-preserving probe; a negative outcome is valid. The small number of groups does not certify a 1% false-positive rate, validate a final privacy bound or replace larger independent confirmation. Detector collection can proceed diagnostically despite failed utility/cost gates, but cannot be reported as a successful defense study.

Prepare on the remote host after selecting a measured worker count. The example keeps the conservative serial setting:

```bash
python -m master_script.tools.collect_guard_traces prepare \
  outputs/guard-v3-collection --workers 1 --targets 8 --seed 2000
```

This writes frozen configurations, the utility protocol and `launch.json`; it starts no training. Choose a new output directory if that one already exists. Inspect the launch before explicitly starting a job:

```bash
python -m master_script.tools.collect_guard_traces run \
  outputs/guard-v3-collection/launch.json --gpu 0 --max-jobs 1
```

Each invocation defaults to **one** job. Repeating the command selects the next unstarted job after verifying the prior results and frozen plan; it does not rerun completed private worlds. The full plan is a multi-hour study. Time a first job before budgeting larger `--max-jobs` invocations. There is no automatic background execution or scheduled continuation.

Before every job, the tool requires an estimated 20 GiB free on the output volume and 4 GiB on scratch. These are headroom estimates, not disk reservations or guarantees. Configurations can live on a larger mounted volume. `--scratch-root` can select an existing scratch volume; keep its path short for Ray sockets. The driver uses a new short, privately created Ray directory for each subprocess and never cleans another job's `/tmp/ray`.

After a successful job, the runner verifies the target, implementation/source identities, both durable copies of the complete result, and the saved SQLite guard events. Only then does it retire model/probe weights and guard NPZ snapshots in that job's owned artifact directory. Result JSON, configuration, features, raw answer JSONL and ledgers remain. A retirement inventory records each removed file's SHA256 and size; the original result JSON is not rewritten, so the dataset source hash stays stable. The retained artifact metadata references retired checkpoints; they cannot be replayed. A tombstone prevents `prepare_guard` from silently recreating those scopes.

Ray text logs are archived before that invocation's own scratch directory is removed. The output accumulates small traces/audits/logs instead of all model weights. Log volume and cache growth are still possible; preflight checks repeat per job. The tool does not purge shared Hugging Face caches, old experiments or unrelated temporary files.

Failed/interrupted jobs retain all artifacts and their active-job marker. A restart stops for inspection; it never resets budgets or automatically retries a partial private world. An exclusive collector lock prevents concurrent invocations from launching the same job twice; a lock left by process termination requires inspection. Interrupted retirement keeps its inventory and blocks automatic recreation. Recovery is deliberately conservative rather than claiming resumable private computation.

After all jobs complete, `splits.complete.json` contains checksummed sources and the declared target roles. A partial manifest is labeled partial and must not be presented as a complete dataset. The existing fitting workflow then applies:

```bash
python -m master_script.tools.assemble_guard_dataset \
  outputs/guard-v3-collection/splits.complete.json outputs/guard-v3-dataset.json
python -m master_script.tools.train_guard_detector \
  outputs/guard-v3-dataset.json outputs/guard-v3-detector.json --max-fpr 0.01 --prevalence 0.01
```

Use new output names for exclusive artifact writers. Independent attacker-calibration and final-evaluation targets remain separate future stages. Do not use these detector train/validation/test targets as fresh final attack evidence.

## 3. Utility protocol and reporting

`configs/guard/study_protocol_v3.json` declares all three endpoints separately: no-context, public ordinary RAG and private ordinary RAG. Every endpoint is required for an overall utility pass. The absolute F1 floor remains 0.10, relative F1 at least 90% of a useful matched baseline, and EM loss at most 0.05. Missing or nonfinite endpoint data are unmeasured, never passes. The 10% round-overhead and 1% benign-rejection targets are preserved in the declaration; utility success alone supplies neither privacy nor population rejection evidence.

The collector snapshots this protocol and its hash in the launch before jobs begin and verifies it on later invocations. The parser/model prompts, question set and scoring are unchanged. No failed historical utility endpoint is silently demoted to an optional diagnostic. A future protocol change must be explicit and prospective.

The analyzer accepts an optional `--protocol` and adds separate endpoint results while retaining the existing no-context `utility_valid`, `exclusion`, raw privacy and matched overhead fields:

```bash
python -m master_script.tools.analyze_guard_study RESULT_BASELINE.json RESULT_DEFENDED.json \
  --protocol master_script/configs/guard/study_protocol_v3.json \
  --output outputs/new-joined-endpoints.json
```

The shadow-only collection has no matched baseline and consequently cannot pass matched utility gates. Adding the new report to an old run is explicitly a reanalysis, not a retroactively predeclared protocol. Historical paired-run reanalysis still marks no-context failed even when both RAG endpoints pass. Privacy units remain separate: fresh batch, training record, and retrieval datastore.

## Local verification and remaining evidence

The full local suite passed **536 tests, with 11 skipped**. New tests exercise concurrent work, exact hashes and feature agreement on nonzero updates, tampered references, invalid values, duplicate reservations, safe retirement, immutable trace evidence, retired-scope refusal, bounded job advancement, exclusive collection, scratch ownership, frozen split roles, and missing/failed utility endpoints. A small synthetic worker benchmark is recorded under `outputs/guard_v3_verified`; it is not a full-model speedup measurement.

The 24-job configuration was generated and validated locally without model downloads or training. No remote GPU experiment, classifier fit, publication or history cleanup was performed in this turn. The next user action is the CPU worker benchmark above. A useful worker count, actual FL-round improvement, collected detector data and any held-out detector benefit remain unmeasured.
