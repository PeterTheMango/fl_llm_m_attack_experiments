# Paired runtime and utility check — 24 September 2026

The new pair confirms that the optimized guard preserves the observed training trajectory, but still fails the overhead target and does not stop the tested architecture-preserving attack. Low no-context utility is also present in the unguarded baseline. The next diagnostic should compare the original pretrained model and saved fine-tuned model with identical questions, prompts and scoring, without another training run.

The queue hash matches both manifest entries; both raw result hashes match the attached analyzer report. The two completed runs share implementation `db21a0271c3a` and match on all non-defense settings. Attack balanced accuracy and AUROC were recomputed from the four trial rows per condition. No-context exact match and token F1 were recomputed from the raw answer audit. The audit contains 20 distinct questions per run, with exactly identical questions and answer strings between conditions.

| Measurement | Baseline | Rules-only |
|---|---:|---:|
| Round 1 time | 74.12 s | 117.68 s |
| Round 2 time | 56.16 s | 97.50 s |
| Round 3 time | 55.20 s | 94.28 s |
| End-to-end time | 635.00 s | 806.47 s |
| Mean training loss, rounds 1 / 2 / 3 | 3.2041 / 2.3160 / 1.6137 | 3.2041 / 2.3160 / 1.6137 |
| Attack balanced accuracy / AUROC | 1.00 / 1.00 | 1.00 / 1.00 |
| No-context token F1 / exact match | 0.01783 / 0 | 0.01783 / 0 |
| Public RAG token F1 / exact match | 0.20935 / 0.10 | 0.20935 / 0.10 |
| Private RAG token F1 / exact match | 0.40667 / 0.30 | 0.40667 / 0.30 |

Paired round overheads are 58.8%, 73.6% and 70.8%, with a **70.8% median**, above the 10% target. End-to-end overhead is 27.0%. All four clients' recorded losses decrease each round. All 12 scheduled legitimate checks are accepted and all three guarded rounds complete. All four malicious observations are also accepted. These are repeated checks on one previously used diagnostic target, not independent evidence for a population false-rejection or privacy guarantee.

The mean legitimate guard check takes **10.91 seconds**: 3.61 s for features, 4.03 s for request/reference hashing, 2.71 s reading the reference, 0.31 s copying, and 0.25 s numerical validation. Detector load/inference is zero because no learned detector is installed. Ledger setup/reservation is about 0.0021 s. The separate controlled CPU benchmark established a feature-computation improvement, but the remaining full-array costs still dominate actual training overhead. Simply removing feature calculation would not plausibly meet the 10% target at the observed sequential-client throughput. No integrity checks or accounting were disabled here.

The earlier implementation's causal pair had approximately 83.6% median overhead; this pair has 70.8%. That difference is descriptive across separately timed, changed-source batches. The same-host alternating CPU benchmark provides stronger evidence about the arithmetic optimization itself. This pair supplies the current matched overhead measurement, not a general performance estimate across machines or client workloads.

The no-context audit gives a more specific explanation than the previous aggregate scores. The model produces substantive answers, several of which contradict the study references: for example, it supplies a different name for the War Doctor and a different year for the Siege of Antioch. Some questions require missing passage context, such as a particular building's refurbishment dates. Some partially relevant answers are long: the hymn answer includes the expected concept of baptism but receives low token F1 because the reference is one word. Therefore the low aggregate is not just a parsing failure or an across-the-board refusal; both answer correctness relative to the references and answer length contribute. These observations concern agreement with this study's answer key, not an independently verified factual audit.

The baseline and guarded answers are byte-for-byte identical for all 20 questions. Thus this check supplies no evidence that the guard caused their poor quality. It also cannot tell whether fine-tuning reduced utility relative to the original pretrained checkpoint: that control has not yet been measured. The current prompt explicitly requests context even in the empty-context ablation. Both original and fine-tuned models should first be compared under that unchanged prompt; changing it now would confound a model-weight comparison. Neither the 0.10 floor nor the reported exclusions should be silently changed after seeing these results. Both runs remain below the declared no-context utility floor; RAG utility is reported separately.

The file `guard_v2_utility_control.py` performs that next comparison. It pins the source implementation and study hash, uses the saved baseline tokenizer for both model weights, uses the original model revision from the run, and replays the original deterministic generation and answer scoring. It evaluates 20 questions on each model, saves raw answers and metrics in one mode-0600 JSON, and checks whether the saved model reproduces the previously recorded EM/F1. A reproduction mismatch must be investigated before attributing any difference to training. No detector is fitted, no threshold is selected, and these reused questions are not presented as independent final evaluation.

Copy the script into the original remote repository root and run:

```bash
python guard_v2_utility_control.py --gpu 0 \
  --run-dir outputs/guard-v2-paired-check-20260924-095524-b38414b5/results/20260924-095524-72a408e220e8
```

The script uses the cached pretrained weights and the newly saved baseline checkpoint. It does not download weights, save new checkpoints, train, or launch Ray. Keep the new paired-run baseline checkpoint until this completes. It writes a small `utility-control-*.json` under the supplied results directory; return that file. Forty short answer generations should be far smaller than another training pair, but no runtime measurement for this standalone diagnostic is available yet.

Local verification covered the actual report identities, metrics and audit scores; the new diagnostic also passed its configuration/provenance dry run. GPU inference has not been executed locally because the saved model weights and Torch/Transformers runtime are on the remote machine. The large detector-collection sweep remains premature while utility interpretation and guard cost are unresolved.
