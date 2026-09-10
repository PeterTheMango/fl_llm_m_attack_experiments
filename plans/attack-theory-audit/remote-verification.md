# Remote-only verification after approval

No commands in this plan were executed. Run on the user's remote research server, in a new worktree/result namespace after implementation approval. Do not run training, notebooks, inference or tests on the local Mac. The server/path were not provided, so no SSH destination is invented and no remote job was started.

## First capture reproducibility evidence

Record the reviewed commit and correction commit, environment/library/CUDA versions, model/tokenizer/dataset revisions, method version, resolved configuration, seeds, participant schedule, split hashes and effective candidate-token spans. Retain old results as legacy results; prevent old complete-cache entries from satisfying corrected runs. Disable external persistence for small verification fixtures or use a separately authorized verification namespace.

## Acceptance checks that distinguish current defects from corrected behavior

| Finding | Remote check | Required evidence |
|---|---|---|
| T01/T02 | Trace one malicious model through a real selected client loss/backward/update step. Keep raw private batches inaccessible to the attacker. Compare returned chosen-neuron signal with an instrumented client oracle available only to the test harness. | Inference depends on the returned client update; no local attacker recomputation on private examples. |
| T03/T14 | Verify input-LDP per-record randomness and train on independently privatized target variants. Repeat with distinct batch sizes. | Different records receive independent draws; mechanism/budget and any confidence-bound assumptions are recorded. Gradient DP is not substituted. |
| T04/T05 | Fixed losses/errors with hand-derived predictions for the selected Yeom adversary; confusion fixture TP=3,FN=1,TN=2,FP=2. | Balanced accuracy=.625; Yeom advantage=.25. Selected source decision agrees algebraically. |
| T06 | Use candidates A: target NLL=1, reference=2; B: target=8, reference=10. | Source ratio ranks A as more member-like (.5 vs .8); current difference ranks B higher (2 vs 1). |
| T07 | Instrument self-prompt builder and generic reference loader during the real main-runner SPV path. | Self-prompt builder runs and trains fresh base on target generations. Generic preloaded reference does not bypass it. |
| T08 | Fixed joint probabilities and symmetric-pair probabilities; derive both raw variation and final comparator by the printed equations. Separately use sequences of different lengths. | Exact source algebra, correct sign/comparator, and no mean-probability substitution. Report source-sign ambiguity rather than choosing favorable AUC. |
| T09 | Inspect reconstructed full texts and opposite-pair provenance before any likelihood calls. | No literal placeholder remains; unmodified spans retained; pair construction follows source; both models score identical pairs. |
| T10 | A mocked MLM distribution where a candidate outside per-position top ten belongs in global top n; a mixed-case/punctuation record with one intended replacement. | Global suitability selection and unchanged original spans; unsupported swap settings rejected. |
| T11/D02 | Long candidate crossing training/generation budgets, plus suffix exposure report. | Effective record/suffix is explicit; total/new-token budgets agree with declared protocol; model capacity respected. |
| T16/B01 | WBC `[2,-1,-1]`, sizes `[1,3]`, plus tie/short/alignment cases. | Score=1/6, not 1/4; metadata identifies schedule and short-input policy. |
| D01 | ReCaLL candidate near capacity, with zero/short/long prefix and tokenizer special tokens. | Identical scored candidate token events in numerator and denominator; impossible contexts rejected before loss call. |
| D02 | Append a tail beyond the declared effective zlib record. | The effective-prefix score stays unchanged, or the whole record is consistently scored/compressed. No mixed policy. |
| D03 | Same real record in batches with different amounts of right padding, dropout controlled. | Loss/gradient for the record unchanged; mask-zero labels ignored, genuine EOS retained. |
| D04 | Request training/calibration budgets below and above the pool-size transition. | One stable ordered pool; no overlap among training/target/replacement/calibration IDs and effective token sequences. |
| D05 | Different-tokenizer reference and unequal/equal-length-but-different-event arrays. | Reject unsupported alignment; no silent zip truncation. |
| D06 | Replay fixed notebook/runtime toy inputs, including SaMIA ties. | Scores/continuations and result payloads match the approved canonical behavior exactly. |
| D07 | Target client excluded/included by a controlled schedule; AMIA batch positive/negative trials on one model. | Assignment, exposure and batch events are distinct and correctly named. |
| D08/D09 | NaN/Inf/empty score, invalid k/counts, zero/failed aggregations. | No completed scientific result; meaningful validation/failure state. |
| D10/D13 | Replay paired worlds/legacy trials with explicit worker streams and participant schedules. | Same controlled nuisance randomness across paired worlds; distinct independent trial streams where required. |
| D11 | Reuse old config with a corrected method version. | Corrected run cannot return the old cached result. |
| T17 | Candidate members/nonmembers, generated Yes/No/unrecognized answers. | Black-box decision uses generated answer only; unrecognized becomes nonmember; retrieval IDs/similarities are not attack features. |
| O01–O05 | Compare pre/post cleanup on already-corrected fixtures. | Identical scores, ordering, predictions and serialized scientific payloads. |

## Regression and scientific evaluation

After targeted fixes, run relevant existing tests remotely. A useful starting selection, subject to updated tests for the defects above:

```bash
python -m pytest tests/test_scoring.py tests/test_metrics.py tests/test_datasets.py tests/test_federation.py tests/test_runner.py tests/test_toy_parity.py tests/test_legacy_runner_wiring.py tests/test_hash_equivalence.py
```

Historical hash tests may need to preserve old lookup behavior while validating a new explicit result version; do not rewrite them merely to accept changed scores. Notebook parity tests must compare actual outputs rather than only metric keys or AUC=1 on constructed favorable data.

Then perform the smallest real-model end-to-end run for each corrected path. Only after mechanism, labels, spans and persistence pass should source-scale evaluation begin. Compare attacks on controlled target training conditions, not different weighting/splits hidden behind identical model names.

Use each paper's dataset/model/metric protocol when claiming replication. For the project's FL adaptation, record all remaining departures and preserve the FL flow. Source low-FPR reports require sufficient independent negative examples and uncertainty estimates; four trials cannot substantiate .01/.001 population FPR. Select/calibrate independently of the final test; retain trial counts and empirical resolution.

Passing these checks supports the tested implementation/configuration. It does not resolve the WBC textual contradiction or SPV sign ambiguity, and does not by itself establish a certified privacy claim.
