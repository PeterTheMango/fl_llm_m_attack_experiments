# Client-guard pilot analysis — 22 September 2026

## Verdict

The pilot validates architecture-based AMIA prevention with unchanged measured answer quality. It does not yet validate the proposed learned detector, a detector-plus-noise advantage, or improved privacy for both the LLM and RAG. Treat it as Stage A / early Stage B evidence from the research plan.

All confusion matrices, score AUROCs, balanced accuracies, threshold predictions, null rejection semantics, RAG utility means, training event counts/times and reported timing AUROCs checked successfully.

All six runs and the manifest are complete; none excluded as failed. Queue duration: 51.70 minutes. All results report firestore_saved=true; Firestore was not independently queried.

## Scope and comparison limits

Four AMIA conditions: baseline, rules_only, noise_only and shadow. Two Reference conditions: baseline and shadow. Every attack condition has four attempted trials (two members, two nonmembers). AMIA uses one target; Reference covers two paired targets. Training uses one federated round; probe fitting uses four epochs and calibration uses twenty nonmember batches. There are four clients and one sweep seed. Only ordinary RAG is enabled.

This differs from the earlier three-round, twenty-trial light pilots, and the implementation fingerprint changed from c05016e0f007 to f48ee1ad4062. Do not interpret cross-pilot score differences as effects of the new defense. Within this queue, compare matching conditions. No detector file or detector score is present; diagnostic mode is enabled. No combined rules/classifier/noise arm or training-DP arm is included.

## Attack results

| Attack | Condition | Released scores / attempts | Balanced accuracy | AUROC | Interpretation |
|---|---|---:|---:|---:|---|
| amia | baseline | 4/4 | 100% | 1.00 | Baseline detects every tested case |
| amia | rules_only | 0/4 | Unavailable | Unavailable | All four requests prevented by architecture_mismatch |
| amia | noise_only | 4/4 | 25% | 0.00 | Reversed ordering; not evidence of reliable privacy |
| amia | shadow | 4/4 | 100% | 1.00 | Monitoring only; malicious requests still released |
| reference | baseline | 4/4 | 100% | 1.00 | Training membership remains detectable |
| reference | shadow | 4/4 | 100% | 1.00 | Monitoring only; no training privacy defense |

The noise-only scores give AUROC 0.00: all member scores lie below all nonmember scores. Reversing score direction gives AUROC 1.00 on these same four observations; reversing binary predictions gives 75% accuracy. These are post-hoc diagnostics, not held-out attack estimates. They show why below-chance accuracy must not be treated as stronger privacy. Calibrate direction and thresholds on independent defended validation data, then test many more independent targets.

Rules-only legitimately has undefined gradient metrics, because nothing was released. It is incorrect to call this 0% attack success, 50% attack accuracy, or a privacy guarantee. All four refusals are identical at the decision level (refusal AUROC 0.50). Descriptive processing-time AUROC is 0.75, versus 1.00 in shadow mode. With only four chronologically ordered trials, this cannot establish a timing attack; randomize/counterbalance labels, warm up execution and test timing on held-out requests.

## Q1. LLM utility and usability

No-retrieval F1 is identical across all four AMIA arms: 6.43%, with exact match 0%. Reference baseline and shadow average 6.64% F1 and 0% exact match. NLL changes are tiny. Measured answer quality is preserved, but the low no-context score is not proof of acceptable general LLM capability. A broader held-out utility task and an absolute quality floor are still needed.

The enforced rules arm accepts all four legitimate training requests and rejects all four malicious observations. This is a useful functioning baseline. AMIA shadow accepts four legitimate requests; Reference shadow accepts sixteen. Shadow acceptance alone is not protection, though its policy flags are also zero for these benign requests. These observations are too few and too dependent to validate a 1% population false-rejection bound.

All recorded benign parameter changes are zero in this one-round setup. Consequently, later-round legitimate changes, distribution drift, convergence and detector tolerance remain untested. The classifier is absent, so no learned discrimination is demonstrated.

### Runtime

| Condition | Computation seconds | Difference from matching baseline |
|---|---:|---:|
| amia / baseline | 328.69 | +0.0% |
| amia / rules_only | 355.12 | +8.0% |
| amia / noise_only | 305.39 | -7.1% |
| amia / shadow | 371.38 | +13.0% |
| reference / baseline | 733.82 | +0.0% |
| reference / shadow | 996.20 | +35.8% |

Training guard processing totals 56.10 seconds for AMIA rules, 55.83 for AMIA shadow and 220.65 for Reference shadow: approximately 14 seconds per legitimate request. Total computation rises about 8.0% for AMIA rules, 13.0% for AMIA shadow and 35.8% for Reference shadow. These are single-run end-to-end comparisons, not controlled median round overhead. AMIA rules also skips work by rejecting observations, obscuring its cost. The plan’s 10% median round-overhead gate is therefore not established; measure isolated and end-to-end times over matched repeats.

## Q2. What changes with RAG?

Every run has identical ordinary-RAG answer F1/EM and membership rates (Reference averages two evaluated models; AMIA evaluates one).

| Evaluation | Answer F1 | Exact match | Datastore attack balanced accuracy | TPR | FPR |
|---|---:|---:|---:|---:|---:|
| No retrieval, AMIA checkpoint | 6.43% | 0% | Not applicable | — | — |
| Public ordinary RAG | 31.57% | 20% | 59.5% | 96% | 77% |
| Private ordinary RAG | 47.50% | 30% | 57.5% | 94% | 79% |

RAG provides useful answer information on this question set, and the guard has no observed effect on those scores. Public and private question subsets differ, so do not infer that private retrieval intrinsically outperforms public retrieval. The no-context score is across the combined utility set. Each corpus has ten utility questions; repeated answers do not create independent question coverage.

RAG membership predictions have very high false-positive rates: 77% and 79%. High member detection alone is not reliable membership inference. There is no measured privacy improvement from the guard on either corpus. Only ordinary retrieval was tested; no retrieval-defense comparison is present.

## Q3. Are privacy and utility improving together?

| Plan criterion | Evidence and assessment |
|---|---|
| Preserve answer quality | Descriptively met versus matched baseline on this small set; no significance or general usability claim |
| Block current AMIA probe | Met for 4/4 architecture-mismatched requests |
| Improve training-membership privacy | Not shown: Reference is 100% in baseline and shadow |
| Improve datastore privacy | Not shown: RAG metrics unchanged |
| Classifier adds value | Not tested |
| Guard plus noise outperforms noise alone | Not tested |
| Benign rejection ≤1% | Insufficient independent benign requests |
| Median round overhead ≤10% | Not measured; runtime costs require attention |
| Robustness to adaptive/repeated attacks | Not established; curve spans only 1–2 queries per world for one target |

Noise-only reports composed observation epsilon 27.19 at delta 0.00001 across four releases. Its scope excludes training and retrieval. Rules-only does not provide an end-to-end epsilon bound. Do not combine these into a global privacy claim.

## Membership-overlap experiment

AMIA evaluates training-only and both cells for one target; both return a positive datastore-membership prediction. Reference evaluates all four cells for the first paired target, both in baseline and shadow. Every cell returns positive, including neither (training=false, datastore=false). Datastore inclusion toggles retrieval as expected, with document count fixed at 256.

This establishes that the overlap instrumentation produced the four cells in Reference. It does not demonstrate that overlap increases leakage: predictions are constant, and training-absent/datastore-absent also produces yes. On these four binary predictions, datastore TPR and FPR are both 100%, yielding chance balanced accuracy. Expand the independent target set and use a calibrated attack less dominated by yes responses. Keep repeated conditions from being counted as independent evidence.

## What is supported about novelty?

The current result supports client-side policy enforcement and explicit prevention metrics. The logged rejection reason is architecture_mismatch, so it is the simple rule baseline anticipated in the plan. It does not establish behavioral detection of hidden malicious parameters, learned classifier generalization or a superior privacy–utility frontier. Shadow mode correctly logs violations while allowing release; its lack of protection is expected.

## Next experiments, in priority order

1. Run several legitimate training rounds with matched seeds, recording changing parameter deltas, per-round guard cost and false rejections. Profile the roughly 14-second request validation cost before scaling.
2. Broaden independent AMIA targets and Reference paired worlds. Keep member/nonmember pairs and target/run groups together; calibrate defended attack score direction and thresholds separately from final test data.
3. Add the planned classifier-only, rules-plus-classifier and combined-noise arms after constructing disjoint detector train/validation/test datasets. Include architecture-preserving attacks so rejection is not based solely on the unexpected probe shape.
4. Compare gradient noise levels and detector/noise combinations on the same validation targets. Select a shortlist before final evaluation; report undefined all-rejected gradient metrics and full transcript results.
5. Expand the four overlap cells beyond one target. Add matched no-context controls and a more discriminative RAG attack; report false positives and recognition.
6. Evaluate training and retrieval defenses separately and together with the guard. A global improvement requires relevant evidence at all three privacy layers.
7. Run adaptive and repeated-request attacks, counterbalanced timing tests, ledger restart checks and a benign heterogeneous-client stress test.

## Additional questions raised by this queue

- Will architecture-preserving probes bypass rules while appearing normal on public inputs?
- Why are ordinary RAG responses so frequently positive even for absent documents?
- Is validation cost dominated by parameter hashing/copying, feature extraction or model loading?
- Can later legitimate FL updates trigger false alarms that round-one validation misses?
- Does the noise-only reversed ranking persist on independent targets, or disappear as small-sample variation?
- Does rejecting a malicious request cause client dropout or loss of legitimate training opportunities over multiple rounds?

## Reproducibility

`analyze.py` reads only the supplied recovered files and writes this report plus `summary.json`. The summary includes file SHA-256 hashes, recomputed-check status, observed metrics, timings and overlap rows. No source code or experiment configuration was changed and no new training was launched.
