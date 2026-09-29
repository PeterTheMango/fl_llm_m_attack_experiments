# Mirabel significance calibration — result

**Verdict: not viable under the pre-registered criterion.** No significance level both preserves benign retrieval and removes member passages reliably. At the level selected for ≤5% benign gold loss, Mirabel removes only 57.5% of member passages from the verbatim yes/no attack. The criterion required at least 90%.

The protocol ([`protocol.json`](protocol.json)) and script ([`calibrate.py`](calibrate.py)) were committed in `d8000cd` before any output existed. Everything ran on CPU with frozen MiniLM embeddings through the maintained `rag.mirabel` and `rag.retrieve`. No LLM, GPU, training data, guard targets or reserved-final cohort were used. Full curves and per-query values are in [`results.json`](results.json).

## Checks

- The closed-form detection threshold disagreed with `rag.mirabel` on 0 of the grid points checked.
- At α = 0.05, the calculation reproduces the server smoke run exactly: 181/210 public and 172/210 private queries had a document hidden, and 100/100 member passages were removed in each corpus.
- Benign set: 2,597 SQuAD dev questions on the 512 study documents (1,336 calibration, 1,261 validation), with the 20 study utility questions held out. Without the defense, the gold document is in the top 4 for 95.6% of them.

## Trade-off (validation benign questions; 200 member queries per attack format)

| α | Benign gold loss [95% CI] | yes/no member passage removed [95% CI] | Continuation removed |
|---:|---|---|---:|
| 0.05 (configured) | 0.81 [0.79, 0.84] | 1.00 [0.98, 1.00] | 1.00 |
| 1e-3 | 0.59 | 1.00 | 0.98 |
| 1e-5 | 0.24 | 0.95 [0.90, 0.97] | 0.75 |
| 1e-6 | 0.14 | 0.87 [0.81, 0.91] | 0.59 |
| 1e-7 | 0.08 | 0.70 | 0.43 |
| **1.78e-8 (selected)** | **0.048 [0.037, 0.061]** | **0.575 [0.51, 0.64]** | **0.33** |
| 1e-9 | 0.018 | 0.40 | 0.16 |

- Reaching 90% removal of yes/no member passages takes α ≈ 5.6e-6, where 21% of benign questions lose their gold document.
- At the selected α, nonmember queries were almost never flagged (0.5%). Every benign detection removed the gold document, because Mirabel fires precisely when one document is an outlier in similarity to the query.
- Detection AUC against validation benign questions is 0.93 for the yes/no attack and 0.84 for continuation. Even the verbatim attack, which quotes the whole passage, is not cleanly separable from ordinary questions.

## Interpretation

- **The configured α = 0.05 is badly miscalibrated for real retrieval.** The Gumbel null model assumes no document stands out. A good factual question has a standout gold document, so Mirabel flags 89–91% of benign questions and removes the gold for 81%. This explains the smoke-run utility collapse. Mirabel utility and attack results from any run at α = 0.05 should be read as "retrieval mostly removed", not as a calibrated defense.
- **Separation is limited even against the weakest attacker.** A realistic setting (≤5% benign loss) still leaves 42% of verbatim yes/no member passages in context and 67% for continuation queries. Adaptive attackers who paraphrase or quote partially would fare better; they are not measured here.
- **Limits.** SQuAD dev questions were written about these paragraphs, so they are easier retrieval queries than real traffic. The results cover one embedding model and one study. Gold-document loss stands in for answer quality.

## Reproduce

```bash
HF_HUB_OFFLINE=1 /opt/anaconda3/envs/peter_experiments_fl/bin/python outputs/mirabel_calibration_20260929/calibrate.py /tmp/mirabel-recheck.json
```
