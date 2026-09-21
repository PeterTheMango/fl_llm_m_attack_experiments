# Reference light pilot: results analysis

All 5 experiments completed; none were excluded as failed. Each contains 20 training attack trials (10 member/nonmember pairs). All five files report successful Firestore saving; this is not an independent check of the remote database.

Recomputed confusion matrices, balanced accuracy and AUROC for training and RAG; recomputed RAG utility means; verified training exposure, paired candidates and identical training provenance across conditions. All checks passed.

## Training membership inference

The archive field `adv` is balanced accuracy: (TPR + TNR)/2. It is not TPR minus FPR. Chance-level balanced accuracy is 50%. AUROC below is recomputed from pooled raw scores across separately trained models; per-model calibrated decisions and confusion counts should also be considered.

| Condition | TP / TN / FP / FN | Balanced accuracy | AUROC |
|---|---|---:|---:|
| baseline | 10 / 10 / 0 / 0 | 100.0% | 1.00 |
| training_dp_sgd_sigma1 | 0 / 10 / 0 / 10 | 50.0% | 0.62 |
| training_dp_sgd_sigma2 | 0 / 10 / 0 / 10 | 50.0% | 0.45 |
| training_dp_fedavg_sigma1 | 0 / 9 / 1 / 10 | 45.0% | 0.54 |
| training_dp_fedavg_sigma2 | 0 / 10 / 0 / 10 | 50.0% | 0.60 |

Baseline succeeds on every tested member and nonmember. All defended conditions detect zero members at their calibrated thresholds. This supports suppression of this particular attack in this pilot, not a general privacy guarantee or a reliable ranking between the defenses.

## RAG membership and answer quality

These are datastore-membership tests, distinct from the Reference training-membership attack. Means below cover two fine-tuned models per condition, evaluated on the same study. Each corpus condition has 200 candidate decisions per model and only 10 utility questions. Repeated evaluation of the same questions is not additional independent question coverage. Answer F1 measures utility; higher is better. Attack balanced accuracy measures discrimination; lower towards 50% is desirable only when utility and response recognition remain adequate.

| Training condition | RAG condition | Attack balanced accuracy | TPR | FPR | Answer F1 | Exact match | Recognized answers |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | public_ordinary | 63.50% | 97.0% | 70.0% | 18.41% | 10.0% | 100.00% |
| baseline | public_mirabel | 52.00% | 69.5% | 65.5% | 6.71% | 0.0% | 100.00% |
| baseline | public_instruction | 67.25% | 72.0% | 37.5% | 2.81% | 0.0% | 100.00% |
| baseline | public_mirabel_instruction | 47.75% | 31.5% | 36.0% | 0.00% | 0.0% | 100.00% |
| baseline | private_ordinary | 63.50% | 94.5% | 67.5% | 53.72% | 40.0% | 100.00% |
| baseline | private_mirabel | 56.50% | 73.0% | 60.0% | 17.84% | 10.0% | 100.00% |
| baseline | private_instruction | 69.50% | 71.5% | 32.5% | 49.00% | 30.0% | 100.00% |
| baseline | private_mirabel_instruction | 53.75% | 35.5% | 28.0% | 18.96% | 10.0% | 100.00% |
| training_dp_sgd_sigma1 | public_ordinary | 70.50% | 87.0% | 46.0% | 22.94% | 0.0% | 100.00% |
| training_dp_sgd_sigma1 | public_mirabel | 53.75% | 48.0% | 40.5% | 20.30% | 0.0% | 100.00% |
| training_dp_sgd_sigma1 | public_instruction | 74.50% | 79.5% | 30.5% | 21.62% | 0.0% | 100.00% |
| training_dp_sgd_sigma1 | public_mirabel_instruction | 48.75% | 29.5% | 32.0% | 17.51% | 0.0% | 100.00% |
| training_dp_sgd_sigma1 | private_ordinary | 72.50% | 87.5% | 42.5% | 14.73% | 0.0% | 100.00% |
| training_dp_sgd_sigma1 | private_mirabel | 55.25% | 50.5% | 40.0% | 5.17% | 0.0% | 100.00% |
| training_dp_sgd_sigma1 | private_instruction | 71.75% | 79.0% | 35.5% | 35.45% | 20.0% | 100.00% |
| training_dp_sgd_sigma1 | private_mirabel_instruction | 52.50% | 31.0% | 26.0% | 10.70% | 0.0% | 100.00% |
| training_dp_sgd_sigma2 | public_ordinary | 70.25% | 86.5% | 46.0% | 20.37% | 0.0% | 100.00% |
| training_dp_sgd_sigma2 | public_mirabel | 53.75% | 49.5% | 42.0% | 20.27% | 0.0% | 100.00% |
| training_dp_sgd_sigma2 | public_instruction | 73.25% | 77.0% | 30.5% | 24.67% | 0.0% | 100.00% |
| training_dp_sgd_sigma2 | public_mirabel_instruction | 48.25% | 28.5% | 32.0% | 20.30% | 0.0% | 100.00% |
| training_dp_sgd_sigma2 | private_ordinary | 71.25% | 86.5% | 44.0% | 14.84% | 0.0% | 100.00% |
| training_dp_sgd_sigma2 | private_mirabel | 55.00% | 50.5% | 40.5% | 5.21% | 0.0% | 100.00% |
| training_dp_sgd_sigma2 | private_instruction | 72.00% | 77.5% | 33.5% | 35.30% | 20.0% | 100.00% |
| training_dp_sgd_sigma2 | private_mirabel_instruction | 52.25% | 29.5% | 25.0% | 11.19% | 0.0% | 100.00% |
| training_dp_fedavg_sigma1 | public_ordinary | 50.25% | 0.5% | 0.0% | 0.00% | 0.0% | 0.75% |
| training_dp_fedavg_sigma1 | public_mirabel | 50.00% | 0.0% | 0.0% | 0.00% | 0.0% | 0.25% |
| training_dp_fedavg_sigma1 | public_instruction | 49.25% | 0.0% | 1.5% | 0.00% | 0.0% | 1.50% |
| training_dp_fedavg_sigma1 | public_mirabel_instruction | 49.75% | 0.0% | 0.5% | 0.00% | 0.0% | 1.00% |
| training_dp_fedavg_sigma1 | private_ordinary | 50.00% | 0.0% | 0.0% | 0.00% | 0.0% | 0.00% |
| training_dp_fedavg_sigma1 | private_mirabel | 50.00% | 0.0% | 0.0% | 0.00% | 0.0% | 0.75% |
| training_dp_fedavg_sigma1 | private_instruction | 50.00% | 0.0% | 0.0% | 0.00% | 0.0% | 0.00% |
| training_dp_fedavg_sigma1 | private_mirabel_instruction | 50.25% | 0.5% | 0.0% | 0.00% | 0.0% | 0.75% |
| training_dp_fedavg_sigma2 | public_ordinary | 50.25% | 0.5% | 0.0% | 0.00% | 0.0% | 0.75% |
| training_dp_fedavg_sigma2 | public_mirabel | 50.00% | 0.0% | 0.0% | 0.00% | 0.0% | 0.00% |
| training_dp_fedavg_sigma2 | public_instruction | 50.25% | 0.5% | 0.0% | 0.22% | 0.0% | 0.75% |
| training_dp_fedavg_sigma2 | public_mirabel_instruction | 50.00% | 0.0% | 0.0% | 0.00% | 0.0% | 0.75% |
| training_dp_fedavg_sigma2 | private_ordinary | 50.00% | 0.0% | 0.0% | 0.00% | 0.0% | 0.50% |
| training_dp_fedavg_sigma2 | private_mirabel | 49.75% | 0.5% | 1.0% | 0.00% | 0.0% | 1.25% |
| training_dp_fedavg_sigma2 | private_instruction | 49.75% | 0.0% | 0.5% | 0.00% | 0.0% | 1.50% |
| training_dp_fedavg_sigma2 | private_mirabel_instruction | 50.00% | 0.5% | 0.5% | 0.36% | 0.0% | 1.00% |

## Interpretation

1. **Training privacy and retrieval privacy separate in these results.** SGD sigma 1 reduces Reference balanced accuracy from 100% to 50%, while ordinary private-datastore attack accuracy rises from 63.5% to 72.5%. Protecting training does not establish protection of retrieval outputs. This is an observed association in this pilot, not proof that SGD generally increases leakage.
2. **FedAvg settings fail the utility requirement.** Both have zero answer F1 and exact match in ordinary public/private RAG. Membership responses are almost never recognized, and unrecognized answers are scored as nonmembers. Their near-50% RAG scores therefore do not establish a useful defense. Retain these completed runs as utility failures, but exclude them from claims of successful usable RAG protection.
3. **A refusal instruction can improve attack discrimination.** With baseline training, private RAG balanced accuracy increases from 63.5% to 69.5% with the instruction. True-positive rate falls from 94.5% to 71.5%, but false-positive rate falls more, from 67.5% to 32.5%. Reporting only fewer positive answers would hide this effect.
4. **Mirabel has a substantial utility cost.** For baseline private RAG it reduces attack balanced accuracy from 63.5% to 56.5%, while answer F1 drops from 53.72% to 17.84%. Combining Mirabel and instructions reaches 53.75% attack accuracy with 18.96% answer F1. This is a tradeoff, not an unqualified success.
5. **Ordinary RAG attack sensitivity is not reliability at low false-positive rates.** Baseline private RAG detects 94.5% of members but falsely flags 67.5% of nonmembers. Its discrimination is much weaker than the baseline Reference result.

## Limits on study claims

There is one sweep seed (7), with ten paired trial seeds (7–16), one model and one dataset setup. These are ten paired targets, not twenty independent target samples. RAG covers just the first two trained models. No AMIA results are included. This archive cannot establish cross-model reliability, superiority over AMIA, or the general best defense.

Only ten training nonmembers were tested, giving 10 percentage point empirical FPR resolution. Zero observed false positives is not proof of population FPR below 5% or 1%. The 200 calibration nonmembers are threshold-selection data, not additional held-out test outcomes.

Saved training provenance and exposure flags agree with the intended labels and are matched across defenses. This is an internal consistency check, not an independent reconstruction of the entire training process. The archive does not contain generated answer text for qualitative error auditing.

## Reported privacy accounting

The following are composed bounds over training releases within each experiment at delta = 0.00001, not a bound for a single deployed model or the entire RAG system. They exclude private retrieval outputs and other experiments. SGD and FedAvg protect different units, so their epsilon values should not be used to rank equivalent protection. These very large bounds do not demonstrate strong formal privacy.

| Condition | Composed epsilon |
|---|---:|
| training_dp_sgd_sigma1 | 2346.51 |
| training_dp_sgd_sigma2 | 663.25 |
| training_dp_fedavg_sigma1 | 194.34 |
| training_dp_fedavg_sigma2 | 67.17 |

## Study conclusion

Reference is a promising successful attack baseline in this tested setting. The useful research question is now whether a defense can reduce both training and datastore membership inference while retaining answer quality. These results establish a pilot signal and expose unsuccessful defense settings; they do not yet prove broad attack reliability.

Before a larger sweep, investigate the FedAvg utility collapse. For confirmation, repeat matched baseline/SGD comparisons across independent sweep seeds and include AMIA. Retain balanced accuracy, AUROC, TPR/FPR, answer F1, exact match, and response recognition together.

Archive SHA-256: `05ae70bcc24cde2e1676d1c068bc0744fd1efa3070d494e36457adfc43724e1e`
