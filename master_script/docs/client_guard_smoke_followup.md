# Smoke follow-up, 24 September 2026

The remote batch `20260922-173328-2d75dea95957` completed 11 runs with implementation fingerprint `1423ad8f55aa`. All source-queue hashes and result hashes match the supplied manifest and analysis. Recomputing attack and RAG metrics from the trial rows reproduces their reported balanced accuracy and AUROC.

## New evidence from the raw files

- Every client's recorded training loss decreases across the three rounds. For AMIA rules with the architecture-preserving probe, mean losses are 3.2041, 2.3160 and 1.6137. The observed guard preserves learning on the training data; this does not establish held-out utility.
- The architecture-preserving observation has relative delta about 0.00003648 and maximum layer delta 0.002236. Legitimate training reaches 0.002333 and 0.012753, respectively. These features use different approved references: initialization for training, the trained checkpoint for the observation. The ranges do not support a simple large-change rejection threshold that accepts every observed legitimate request. They do not establish classifier separability or generalization. No detector is fitted on this single target.
- Each legitimate request contains 2,520,669,696 bytes. Reference rules spends about 12.76 seconds per request: 5.42 on feature extraction, 2.75 loading reference arrays and 4.03 hashing request/reference bytes. All checks still matter after the feature optimization; there is no claim that optimizing features alone meets the 10% round-overhead gate.
- Ordinary RAG's binary yes/no attack has substantial false-positive bias. On the AMIA checkpoint, public TPR/FPR are 99%/85%; private TPR/FPR are 97%/91%. Correct parsing does not make this an effective attack. The separate continuation/calibration study remains necessary.
- The four-cell overlap audit records training exposure once in the member world and zero in the nonmember world. The candidate is visible in context exactly when inserted in the datastore, with no truncation in these cells. No-context membership predictions are positive in both training worlds. The paired datastore effect for this one target is informative about context use, not evidence of a training-membership interaction.
- The no-context F1 floor fails across all arms. The prompt explicitly asks the model to use context and supplies none in this control. This is a retrieval-ablation measurement, not a sufficient general assessment of the language model. Raw no-context answers were not saved by the old implementation, so their wording cannot be diagnosed from these files. Ten utility questions per retrieval corpus and one target are insufficient for general conclusions.
- Total recorded computation across this batch is 8,401 seconds (about 140 minutes). This is historical timing, not a larger-stage estimate.

## Implemented changes

Float16/float32 feature extraction now uses one bounded-memory pass with double-precision reductions. It preserves the original scale-dependent denominator floors and checks finite values in both arrays. Larger numeric types retain the previous two-pass scaled calculation. Features can differ by floating-point rounding; comparisons are tested numerically, not claimed bit-identical. Feature schema and definitions are unchanged. All runtime snapshot copying, request/reference content hashing, structural checks, detector pins and release accounting remain in place. New experiments acquire a changed source fingerprint and must have fresh matched controls.

The analyzer now counts positive but below-floor F1 as `utility_below_absolute_floor`. It retains those rows and their privacy metrics for diagnosis. Newly generated no-context answers are saved only to the existing private audit JSONL, alongside the other raw answers; prompts and scoring are unchanged.

The CPU benchmark compares the previous feature calculation and current calculation inside the complete current guard path. It uses only a recorded initial training snapshot, verifies its content against the old request hash, alternates calculation order, and creates a separate temporary ledger. It does not modify historical snapshots or ledgers, train a model, access private training records, load CUDA, or consume experiment release reservations.

## Next remote command

Transfer/apply the supplied `guard-v2-followup.patch`, or synchronize these changed source files to the remote repository using your normal workflow. When using the patch, run from the repository root:

```bash
git apply --check guard-v2-followup.patch && git apply guard-v2-followup.patch

run_dir=outputs/guard-v2-smoke-remote/results/20260922-173328-2d75dea95957
python -m master_script.tools.benchmark_guard_runtime \
  --run-dir "$run_dir" \
  --repeats 2 \
  --output "$run_dir/guard-cpu-benchmark-$(date +%Y%m%d-%H%M%S).json"
```

Run on the original host, where the `artifacts/*/client-guard/*.npz` snapshots still exist. The review archive contains only JSON/YAML, so the full-model benchmark cannot run on that archive alone. The process uses several GiB of host RAM (source arrays, an owned request copy and one reference layer); no GPU allocation is needed. Four checks at the old approximately 13-second rate suggest roughly a minute plus loading, but hardware/load can change this. The report uses a unique filename and never overwrites earlier results.

Return the new `guard-cpu-benchmark-*.json`. Its `median_seconds`, `current_over_previous`, per-stage profiles and feature equivalence check determine the next optimization. Warm-cache initial-checkpoint CPU measurements do not establish FL overhead or protection against malicious requests. After that measurement, prepare a small matched baseline/guard rerun before expanding the collection stage. Do not reuse the old baseline as a defense-only comparison after a source change.

## Validation

Local suite: **518 passed, 11 skipped**. Coverage includes numeric equivalence at small updates and float32 extremes, chunk boundaries, nonfinite values, existing snapshot tamper checks and accounting, benchmark preservation of all historical files, utility exclusion, and private no-context audit output. Torch/Flower/CUDA execution remains unavailable locally. No remote training job was launched. The full-model runtime improvement is unmeasured until the command above runs.
