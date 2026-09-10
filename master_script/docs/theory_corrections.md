# Theory corrections: implementation record

The approved audit corrections are implemented in the working tree. The maintained entry points are `master_script/core` and the 11 adaptation notebooks, which now import the same configs and runner. The nine scalar recreation notebooks import the canonical formula functions. Legacy vision AMIA scripts remain separate, explicitly labeled activation-condition experiments.

No experiments, project imports, tests, package installations or remote jobs were executed during implementation. Local verification was limited to source inspection, Python AST parsing, notebook JSON/code parsing, Ruff undefined-name analysis and `git diff --check`. The tests below must be run on the research server before treating these changes as validated.

## Scientific decisions

**WBC:** the default is the local v2 Appendix C.1.5 / author-config schedule `[2,3,4,6,9,13,18,25,32,40]`. Equation 12 produces `[2,3,4,5,8,11,15,21,29,40]`; it remains an explicit sensitivity condition. Both use Equation 13's uniform mean of per-size positive-window fractions. Windows are strictly positive; a zero sum is negative. Each requested size retains its weight when clamped to a short record. The WBC notebook compares the schedules on the same supplied deltas. See the [source-backed resolution](../../plans/attack-theory-audit/wbc-resolution.md).

**SPV-MIA:** the maintained HF path uses the paper's Appendix A.3 embedding-domain variant: the same Gaussian direction and its negative are evaluated in both models. It requires compatible vocabularies and embedding shapes. The reference is a fresh base model trained on target-generated public prompts. The default 32 generated reference records is a resource setting, not the paper's 10,000-record benchmark. Joint probabilities are combined in signed log space; only the final score receives an increasing transform to preserve rank and zero while avoiding probability underflow. Nonzero thresholds are in transformed units. Printed Equations 5 and 10 are retained; their tension with the local-maximum explanation is recorded rather than resolved by choosing whichever orientation improves test AUC. Toy masking remains an explicitly artificial illustration.

**AMIA:** membership concerns the private batch observed by the malicious server. A frozen LM prefix feature feeds a malicious head; the selected client computes next-token cross entropy and returns probe gradients through Flower. The server decides from the returned chosen-neuron gradient. Probe negatives come from an independent public/calibration partition. The feature LDP options are `none`, `BitRand`, and `OME`; private perturbations use independent draws with an unrecorded random seed. `epsilon`, `ldp_target_samples`, `certificate_samples`, and `certificate_delta` configure the mechanism and finite-set diagnostic. Labels are not privatized: the scope is conditional feature privacy, not complete-text privacy. The diagnostic uses analytic network-range bounds and a union bound over the finite public audit set. It explicitly reports `certified_attack: false`; unseen negatives and downstream derivative cancellation are not certified. Legacy vision scripts remain activation proxies, not end-to-end client-update demonstrations.

**LOSS:** the default decision is Yeom Section 3.2 Adversary 1 with bounded loss `min(NLL, B)` and membership probability `1 - loss/B`. The bound and random draw are stored. The previous nonmember quantile decision remains available as `decision_rule: nonmember_quantile`, explicitly labeled an FL variant. LOSS `adv` is TPR−FPR; `balanced_accuracy` is separate. Other attacks retain their documented balanced-accuracy convention for `adv`.

## Finding disposition

IDs refer to the [pre-change audit](../../plans/attack-theory-audit/AUDIT.md). “Implemented” describes code changes, not successful runtime verification.

| ID | Disposition and implementation |
| --- | --- |
| T01 | Implemented: AMIA observes the selected client's downstream loss gradients through Flower observation rounds. |
| T02 | Implemented: AMIA probe training no longer accepts private client partitions; public negatives are checked for effective-token overlap. |
| T03 | Implemented with limited claim: BitRand/OME feature mechanisms and finite-set confidence diagnostics; no universal certification or full-text privacy claim. |
| T04 | Implemented: randomized bounded-loss adversary; quantile rule explicitly identified as a variant. |
| T05 | Implemented: Yeom advantage and balanced accuracy are separate fields. |
| T06 | Implemented: reference score is negative target/reference mean-NLL ratio, following Carlini Section 6.1. |
| T07 | Implemented: SPV builds its own self-prompt reference; the runner no longer supplies an unrelated reference. |
| T08 | Implemented with source ambiguity recorded: joint probability arithmetic, printed sign, final increasing numerical transform. |
| T09 | Implemented: source embedding-domain symmetric perturbations replace unrelated independent masks in the HF path. |
| T10 | Implemented: full-vocabulary neighborhood suitability search, original-text offset replacements and deduplication. |
| T11 | Implemented: SaMIA total generation budget defaults to 1024, with explicit context-capacity rejection. |
| T12 | Scope corrected: all new results identify FL adaptation; toy/scalar notebooks do not claim benchmark reproduction. Published-scale experiments remain server work. |
| T13 | Implemented: legacy AMIA uses a single binary logit with BCEWithLogits, with target class mapped consistently. New checkpoint suffix prevents reuse of incompatible weights. |
| T14 | Implemented: legacy BitRand evaluation samples separate record/bit randomness instead of broadcasting one mask. |
| T15 | Implemented with limited claim: checkpoint selection uses public training evidence and post-update weights; legacy activation evaluation is explicitly a proxy. |
| T16 | Implemented: WBC uniform per-size aggregation and explicit schedule metadata. |
| T17 | Scope corrected: RAG evaluations remain study-specific adaptations, not a replication of the source benchmark. Existing answer-only likelihood and decision mechanics were retained. |
| D01 | Implemented: ReCaLL scores exactly the same candidate tokens in both conditions; prefix consumes extra context, with capacity checks. |
| D02 | Implemented: effective candidate boundaries for compression, generation and neighborhood; reference truncation cannot silently change the candidate. |
| D03 | Implemented: all maintained training paths mask attention padding; genuine EOS remains a target when PAD equals EOS. |
| D04 | Implemented: one deterministic run-level dataset ordering reserves calibration records; effective-token overlap checks reject hidden leakage. |
| D05 | Implemented: WBC requires matching vocabularies, candidate token IDs and loss lengths. |
| D06 | Implemented: 11 operational notebooks delegate to the canonical runner; 9 scalar recreation notebooks delegate formulas and clear stale outputs. |
| D07 | Implemented: batch-membership and assigned-training-membership results are distinct; training trials record actual target exposure. |
| D08 | Implemented: shared direct/YAML validation, finite-score rejection, nonempty/aligned evidence checks. |
| D09 | Implemented: every expected aggregate must arrive; failed clients or missing final parameters fail the run. |
| D10 | Implemented: paired trial seeds, deterministic partition scheduling, actual-record weighting, explicit generation/perturbation seeds. LDP private draws are intentionally independent. |
| D11 | Implemented: version/source fingerprint, resolved model/dataset/auxiliary-model revisions, local checkpoint content hashes, token/generation provenance and resolved queue IDs. |
| D12 | Implemented: AUROC and low-FPR metrics report class counts and empirical resolution; insufficient-resolution fields are null, including SaMIA's legacy alias and pipeline metrics. |
| D13 | Implemented: legacy evaluation uses explicit per-trial randomness and sequential execution rather than inherited multiprocessing streams. |
| O01 | Implemented: AMIA trial context is returned explicitly, replacing the hidden global run map. |
| O02 | Implemented: unused WBC serializer and unused toy-FedAvg argument removed. |
| O03 | Implemented: misleading cleanup helper removed; model references are released by their owner. Memory behavior still requires server observation. |
| O04 | Implemented: operational notebooks use the same implementation; common metrics/padding/validation helpers replace drifting behavior. Distinct AMIA observation and LOSS decision loops remain explicit. |
| O05 | Implemented: inactive trailing legacy experiment blocks and unused conversion helper removed; upstream provenance retained. |

## Result identity and migration

New IDs have the form `theory_v2_<core-source-fingerprint>_<attack-config-digest>`. The digest retains each attack's historical formatting convention, but corrected methods deliberately do not reuse historical result IDs. The resolved model, reference and dataset revisions are hashed; neighborhood also pins its masked-LM revision. Local checkpoint directories are content-hashed, which can be expensive for large models. Core-source changes invalidate corrected cached results too.

Historical Firestore documents are not deleted, rewritten or claimed to be comparable. Do not remove the method prefix to recover cache hits. Archive/export historical results separately when comparing methods. Queue entries are provisional until revision resolution; execution updates their IDs to match the result. Keep the resolved config, dependency versions, hardware details, source commit/diff and model artifacts with research results. Seeding does not guarantee bitwise CUDA reproducibility across environments.

`max_length` is the candidate training/scoring budget. Prefix/generation budgets and model capacity are separate constraints. If different raw records become identical after truncation, the run raises an error: increase the budget or choose a different dataset record under a prespecified protocol.

## Server verification

Use the existing allocated server environment. No dependency changes or jobs have been initiated for you. From the repository root:

```bash
python -m pytest -q tests/test_theory_corrections.py tests/test_hash_equivalence.py tests/test_config.py tests/test_metrics.py tests/test_scoring.py tests/test_toy_parity.py tests/test_legacy_runner_wiring.py tests/test_runner.py
python -m pytest -q tests
```

The first command checks corrected formulas, token overlap, version isolation, padding, the AMIA loss-gradient boundary, LDP row independence and runner wiring. The second checks integration regressions. Optional Torch tests skip when Torch is unavailable; such skips are not AMIA validation.

Then use the [remote discriminating checks](../../plans/attack-theory-audit/remote-verification.md) for real Flower training, generation, perturbation, tokenizer alignment, failure injection and both WBC schedules. Start with the tiny model and a small authorized trial count; keep artifacts and resolved configs. The notebooks' `RUN_REMOTE` / `RUN_SWEEP_REMOTE` flags default to false and must be explicitly enabled on the server. Four-trial smoke results cannot measure 1% FPR: at least 100 nonmembers are required even for that empirical resolution, and substantially more independent data is needed for a reliable population estimate.

Source benchmark reproduction, empirical attack effectiveness, universal AMIA certification and end-to-end privacy claims are not established by these changes or by passing the regression suite.
