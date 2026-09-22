# Client-side request validation and gradient protection: research and implementation plan

Status: historical proposal prepared 18 September 2026. Parts were implemented before the 21 September pilot and extended on 22 September. For current executable workflows, implemented boundaries, test evidence and remaining GPU work, see [the v2 handoff](client_guard_v2_handoff.md). The sections below preserve the original proposal; they are not an inventory of missing code or authorization to start experiments.

## 1. Objective and proposed contribution

Investigate whether a client-controlled request detector, combined with clipping and a fixed minimum level of gradient noise, can reduce active membership leakage without materially impairing legitimate federated LLM training or downstream RAG answers.

The central hypothesis is that preventing suspicious requests before a private gradient is released can improve the privacy–utility tradeoff compared with relying on noise alone. This is a hypothesis to test, not a demonstrated result or formal privacy guarantee.

The study should distinguish three membership questions:

1. **Client-batch membership:** AMIA observes a fresh client response to a malicious probe.
2. **Training-record membership:** Reference compares model scores to infer training inclusion.
3. **Retrieval-document membership:** a black-box attacker queries a RAG system to infer datastore inclusion.

A defense can improve one and leave the others unchanged. The proposed guard directly targets the first. Training and retrieval defenses must be tested separately and in combination before claiming system-wide improvement.

### What could be novel

The contribution should be a demonstrated result, not merely adding a classifier or combining FL with RAG:

- Client-side behavioral detection of active membership probes in federated causal-LM fine-tuning, including attacks with the expected architecture.
- Measurement of whether detection permits lower observation noise at comparable residual attack risk.
- Joint auditing of the three membership questions with matched model checkpoints and explicit utility constraints.
- Evaluation of adaptive attacks, repeated requests, and leakage from accept/reject decisions.
- A controlled training-membership × datastore-membership experiment separating memorization from retrieval disclosure.

Do not claim “first” without a fuller literature review. A negative result is useful if it shows that a detector adds false alarms or overhead without outperforming rules and noise.

### Closest related work and boundaries

- [Hiding in Plain Sight / SEER, ICLR 2024](https://www.sri.inf.ethz.ch/publications/garov2024seer): studies client-side detectability of malicious server models and attacks that evade checks. It concerns reconstruction; it motivates testing evasive attacks rather than only obvious added heads.
- [Poison to Detect, 2025 preprint](https://arxiv.org/abs/2509.11974): client-side integrity tests for orchestrator-induced targeted overfitting. Related detection objective, different attack mechanism.
- [Analysis of Privacy Leakage in Federated Large Language Models, AISTATS 2024](https://proceedings.mlr.press/v238/vu24a.html): active membership inference in federated LLMs. Essential attack-related prior work.
- [FedRAG, 2025](https://arxiv.org/abs/2506.09200): federated fine-tuning of RAG components; architectural precedent.
- [FedMosaic, 2026 preprint](https://arxiv.org/abs/2602.05235): federated parametric RAG with extraction-attack evaluation. Establishes that federated RAG privacy evaluation is already studied.
- [Eluding Secure Aggregation via Model Inconsistency](https://arxiv.org/abs/2111.07380): supports consistency validation but does not establish that identical models are benign.

These sources motivate the proposal; none is evidence that the proposed classifier will succeed in this implementation.

## 2. Current status and evidence

The light configuration uses Qwen2.5-0.5B-Instruct, SQuAD-derived records, four clients, three federated rounds, one local epoch, and one concurrent simulated client. The LLM is fine-tuned first; retrieved context is supplied during evaluation. The retriever is not jointly trained in this setup.

| Pilot finding | Implication |
|---|---|
| Reference baseline: 100% balanced accuracy, 20 trials | Strong pilot training-membership signal, not broad reliability proof |
| Reference DP-SGD σ=2: 50% balanced accuracy | Suppresses this calibrated attack in the pilot |
| AMIA baseline: 95%; training-only DP-SGD σ=2: 100% | Training protection does not automatically protect fresh probe responses |
| AMIA gradient Gaussian σ=1 and σ=2: 50%, AUROC 0.56/0.59 | Promising observation defense; no clear winner between noise levels |
| AMIA clipping only: 95%; feature defenses: AUROC 1.00 | Clipping and tested feature perturbations leave substantial signal |
| AMIA private ordinary RAG answer F1: 47.86%, unchanged under observation noise | Preserved measured inference quality; does not measure normal-training cost of the guard |
| AMIA private ordinary RAG membership accuracy: 57%, unchanged under observation noise | Retrieval protection is a separate issue |
| Reference DP-FedAvg ordinary RAG F1: zero | Retain as completed utility failures; do not label usable privacy successes |

AMIA has only two unique targets per condition and 20 repeated batch observations. Reference has ten paired target worlds. RAG covers two evaluated models per condition with a small repeated question set. Existing pilots cannot train or validate a general detector. The saved `adv` field is balanced accuracy, not TPR−FPR. Ten test nonmembers provide only 10-percentage-point empirical FPR resolution.

Local analysis artifacts are in `outputs/amia_light_analysis_20260918/` and `outputs/reference_light_analysis_20260917/`. These may be local-only; keep the original archives and checksums with the study records.

## 3. Research questions and decision criteria

| Question | Required measurement |
|---|---|
| Q1. Does protection affect LLM utility and usability? | Held-out no-retrieval task F1/EM and NLL, training convergence, legitimate requests blocked, completed rounds, latency, memory, communication |
| Q2. What changes after RAG is added? | Evaluate the same checkpoint with no context, public RAG and private RAG; measure QA F1/EM, support for answers, retrieval recall, refusal and latency |
| Q3. Do LLM and RAG privacy improve while utility remains acceptable? | Separate AMIA, Reference and datastore attack results for each usable condition; show tradeoff plots instead of a single averaged privacy score |
| Q4. Does the classifier outperform simpler checks? | Rules-only and noise-only baselines at matched legitimate-request false-positive rate and comparable residual risk |
| Q5. Does detection generalize? | Unseen targets, runs, client distributions, attack implementations and model revisions |
| Q6. Can an adaptive or repeated-query attacker bypass protection? | Residual risk versus query budget; policy decisions, timing and rejection observables included |
| Q7. Does overlap between training and retrieval increase leakage? | Four membership cells described below, with other factors matched |
| Q8. Are some clients harmed more than others? | Per-client utility, false rejection and privacy under heterogeneous data distributions |
| Q9. Does rejection itself reveal membership? | Attacker using only decision/timing transcripts, particularly across target-present/absent worlds |
| Q10. What is the computational price? | Detector loading and inference cost, per-round overhead, GPU memory and total sweep time |

Provisional engineering gates, to be agreed before final testing: retain at least 90% of matched baseline QA F1; reduce EM by no more than 5 percentage points; legitimate-request rejection no more than 1%; median round overhead no more than 10%. These are study design choices, not universal thresholds. Relative F1 criteria are meaningless when baseline F1 is near zero: require an absolute task-quality floor on such slices. Report confidence intervals and sample counts; a small pilot cannot validate a 1% false-rejection target.

A defense advances only if it improves the relevant privacy–utility tradeoff on held-out data. If RAG privacy is unchanged, describe the result as AMIA-specific improvement. Low attack accuracy caused by broken answers is not success.

## 4. Threat model and trust boundary

The malicious server may choose model/probe parameters, send requests repeatedly and adapt to observable responses. The client guard, its policy, detector artifact and release budget are controlled locally and cannot be overwritten through server `fit_config`. The trusted study harness owns ground truth but must not provide it to the guard or attacker.

Begin with a single malicious server without client compromise. Treat collusion and multiple identities as separately declared extensions. A local counter must survive worker restarts; otherwise restarting a Ray actor bypasses the budget.

The first detector should use incoming parameters, public test examples and protocol metadata only. No private records, membership labels, target IDs, condition names, file paths, seeds or outcome metrics may be detector features. Any future private-data-dependent detector requires separate analysis of privacy leakage through its decision.

Model consistency is useful only when expected commitments are established outside unilateral server control. Same-model checks cannot detect a malicious model sent identically to all clients. Do not represent a simulation flag as cryptographic verification or remote attestation.

## 5. Proposed client guard

1. Parse a declarative request using client-owned model/training code; do not execute server-provided code.
2. Validate operation, architecture, parameter shapes, finite values, trainable-layer policy, round identity and release allowance.
3. Measure bounded behavioral features on a fixed, versioned public test set. Examples: layerwise relative parameter changes, activation concentration, output drift and loss change relative to an approved prior checkpoint.
4. Normalize features using training-split statistics and apply a small frozen classifier. Start with logistic regression and a simple threshold baseline; add a tree model only if justified.
5. Reject policy violations or excessive risk without calculating or releasing a private gradient. Return a generic typed refusal; retain detailed reasons locally.
6. For accepted requests, apply a fixed client-owned clipping/noise mechanism before release. Detector-based selection is not permission to release unprotected gradients.
7. Record a private audit event and atomically debit the release ledger. Network failures and retries must not reset accounting or inadvertently produce unlimited fresh releases.

Public-input detection helps avoid making refusal directly depend on the target's private membership, but does not replace a complete transcript analysis. Noise strength should initially be fixed per condition, not selected from private detector signals. A classifier does not itself provide DP. Keep observation, training and retrieval accounting scopes explicit.

### Prevent a trivial experiment

The present malicious head is structurally different from legitimate full-model training. An architecture rule may reject every such request. Report this as a valid prevention baseline, not evidence of a sophisticated learned detector.

To test behavioral novelty, add attacks constrained to the same permitted architecture and request interface as benign training. Treat that as a new attack variant requiring its own correctness check. Do not silently redefine the existing AMIA implementation or assume its attack guarantees transfer.

## 6. How to integrate into the current project

Paths below are repository-relative. New paths and interfaces are proposals.

| Existing component | Integration work |
|---|---|
| `master_script/core/attacks/amia.py` | In `run_attack_trials` / `VictimClient.fit`, run the guard before private batch access and `client_loss_gradients`. Change `ObserveGradient.aggregate_fit` to handle a structured rejection separately from a gradient response. Preserve the undefended path. |
| `master_script/core/attacks/amia.py` calibration | `calibrate_probe` currently expects gradient scores. Calibrate the attacker against the actual defended interface, including refusals. If no gradients are accepted, numeric gradient AUROC is unavailable; evaluate transcript leakage separately. |
| `master_script/core/defenses.py` | Reuse `protect_observation` for accepted responses. Keep bounds and clipping scoped to the actual release. Do not count rejected requests as noised gradient releases. |
| `master_script/core/federation.py` and AMIA `_ami_flower_client_cls` | Route legitimate training through the same guard policy so false positives can affect actual convergence. Do not relax the private-round completeness check to hide rejected training updates. |
| `master_script/core/pipeline.py` | Add a strictly validated, optional client-guard schema; include policy/detector/test-set hashes in new run identities. Preserve old identities when guard is absent. |
| `master_script/core/yaml_config.py`, `config.py`, `queue.py` | Validate new fields and supported combinations; expose explicit named variants without automatically multiplying every factor. Preserve manifests and old configurations. |
| `master_script/core/rag.py` | Extend `evaluate_pipeline` to support matched training/retrieval membership cells and a larger held-out utility set. Keep no-retrieval controls. Store recognition/refusal separately from privacy decisions. |
| `master_script/core/metrics.py` | Add detector metrics, release coverage, transcript metrics and cluster-aware summaries. Keep historical `adv` behavior for compatibility and label it correctly in new reports. |
| `master_script/core/runner.py`, `result_storage.py`, `runstate.py` | Persist guarded observations incrementally; distinguish successful prevention, operational error, invalid evaluation and interrupted work. Include provenance and immutable detector identity. |
| Web UI configuration/results modules | Display defense status, rejection count, detector false positives, utility validity and progress. Guard on/off and detector identity must be visible in summaries. |

Proposed new modules:

- `master_script/core/client_guard.py`: immutable policy, typed request/response, rule evaluation and guard execution.
- `master_script/core/guard_features.py`: public-data feature extraction with strict feature allowlist.
- `master_script/core/guard_detector.py`: fit/load/score with versioned preprocessing and threshold; client uses a pinned local artifact.
- `master_script/core/release_ledger.py`: persistent per-client budget and request identity handling.
- `master_script/tools/build_guard_dataset.py`: export labeled synthetic/public experimental request traces.
- `master_script/tools/train_guard_detector.py`: grouped splitting, fitting and validation-only threshold selection.
- `master_script/tools/analyze_guard_study.py`: joined privacy/utility report and validity checks.

Use safe detector serialization suitable for the chosen model and verify its content hash before loading. Keep attack labels and detailed audit features in research artifacts, not server-visible response metadata.

### Response semantics

Use an explicit `accepted` / `rejected` observation result. For rejected requests, `gradient`, `score` and gradient-based `pred_member` are null. Do not send all-zero arrays: that creates an artificial attack statistic. Do not set the run to failed merely because an attack was blocked.

The current aggregator requires exactly one active victim update and scores every returned array. Both this logic and the trial schema must change together. Keep infrastructure exceptions as failures. For legitimate training rejected by policy, record a policy-aborted round and lost availability; retain the existing rule against silently aggregating incomplete private rounds.

Record: request/run/target grouping IDs for private analysis, decision, local reason, detector score, policy and detector hashes, feature schema version, release count, gradient availability, attack prediction if defined, runtime, privacy-accounting scope and evaluation validity. Never feed grouping IDs or labels back into the detector.

### Configuration proposal

Create a new `master_script/configs/client_guard_privacy_pilot.yaml`; retain both existing AMIA/Reference configurations unchanged. The following is illustrative schema only and will be rejected by today's parser until implemented:

```yaml
pipeline:
  condition: guard_classifier_noise
  defense: {mechanism: none}
  client_guard:
    mode: rules_classifier
    policy_file: guard/approved_training_policy.json
    detector_file: guard/detector.json
    detector_sha256: REPLACE_WITH_VERIFIED_ARTIFACT_HASH
    public_probe_file: guard/public_probe_set.json
    release_budget: 10
    rejection_action: refuse
# Reuse the existing AMIA base fields for accepted observations:
# observation_defense: gaussian
# observation_clip_norm: 1.0
# observation_noise_multiplier: 0.5
```

Do not maintain conflicting copies of the noise parameters. A shared guard for normal FL training must specify its training-update mechanism separately; observation noise is not automatically equivalent to DP-SGD or DP-FedAvg. Resolve referenced files and hash contents using the same reproducibility principles as the current RAG study loader.

## 7. Detector dataset and leakage-resistant splitting

Generate benign requests from real permitted training across multiple seeds, rounds and client distributions. Include legitimate large updates, warmup behavior and different allowed heads/adapters so the detector does not equate unfamiliarity with malice.

Generate malicious requests from independently trained probes and architecture-preserving variants. Vary attack optimization and repeated-request strategies. Label the request's origin as benign/malicious; do not confuse this label with member/nonmember truth.

Split complete target/run groups into detector training, detector validation and final test sets before fitting. Keep paired member/nonmember observations together. Feature normalization, selection and thresholds use only training/validation. Hold out at least one attack variant for generalization testing. Reserve separate public data for attacker calibration, detector behavior tests and task utility evaluation; verify token-prefix/document overlap as appropriate.

The existing two-target pilot is insufficient. Begin with a small mechanics pilot, then use pilot variance and compute measurements to plan an independent-target study. Avoid reporting hundreds of repeated batches as hundreds of independent targets. Sample size for low false-positive claims must be planned explicitly; even hundreds of error-free observations have nonzero uncertainty.

## 8. Experimental design

### Stage A: correctness and replay

Implement response semantics and rule checks using tiny synthetic requests. Reproduce current undefended metrics from saved rows. Confirm the classifier sees only allowed inputs. Run the guard in shadow mode on benign training: decisions are logged but not enforced. This collects evidence about false alarms without claiming protection.

### Stage B: AMIA defense comparison

Use matched target identities and training seeds for:

1. No defense.
2. Rules only.
3. Classifier only (diagnostic; no formal privacy claim).
4. Clipping plus noise only.
5. Rules plus classifier.
6. Rules plus classifier plus the same clipping/noise as arm 4.

Start with one noise level and a small target set to validate operation. On validation data, explore σ ∈ {0.25, 0.5, 0.75, 1.0}; preserve σ=2 as a comparison when needed. Select a shortlist before final testing. Do not launch the complete Cartesian product of targets, seeds, thresholds, noise and RAG variants.

### Stage C: genuine training utility

Enforce the guard during legitimate federated fine-tuning. Compare accepted rounds, convergence, no-retrieval utility and per-client completion. Include rounds with malicious requests interleaved under a predeclared schedule. Define recovery after rejection, such as pausing for an independently approved checkpoint; do not silently accept the next malicious checkpoint or drop clients to make training finish.

Evaluate whether legitimate training uses observation-like releases or a different update interface. If different, state the boundary and separately measure the mechanism protecting ordinary updates. Preserved utility of an unchanged checkpoint alone is not evidence that guarded training works well.

### Stage D: RAG and Reference on matched checkpoints

For every shortlisted checkpoint, run no-retrieval, ordinary public RAG and ordinary private RAG evaluation. Add Mirabel/instruction conditions only after the core comparisons are stable. Evaluate Reference on correctly constructed paired training-inclusion worlds; evaluating only checkpoints where the target is always trained cannot establish training-membership discrimination.

Do not compare AMIA and Reference percentages as interchangeable attack rankings: their access and membership questions differ. Where model worlds match, reuse immutable checkpoints, verifying full training/data/config hashes. Shared checkpoints reduce compute but are not independent repetitions.

### Stage E: training/retrieval overlap

Construct matched records in four cells:

| Cell | In FL training? | In retrieval datastore? |
|---|---|---|
| Neither | No | No |
| Training only | Yes | No |
| Retrieval only | No | Yes |
| Both | Yes | Yes |

Keep prompts, document difficulty and corpus sizes balanced; document substitutions when maintaining fixed cardinality. For each checkpoint, toggle datastore inclusion without retraining. Training inclusion still requires separate model worlds. Query all cells with no context and with RAG. Audit actual tokenized exposure and retrieval, including context truncation.

Estimate whether training membership changes attack success within each retrieval stratum, and vice versa. Do not attribute every RAG-positive answer to training memorization. Add an authorized/unauthorized access scenario if access control is introduced; define what information authorized users may learn.

### Stage F: adaptive attacks and combined protections

Test architecture-preserving attacks, alternate target/probe fitting, validation-aware evasion and increasing query budgets. Include an attacker using refusal and timing transcripts. Compare the guard with and without existing training and retrieval protections. Report precisely which layer improves and whether all utility gates remain met.

## 9. Metrics, statistical analysis and validity

- **Detector:** malicious-request recall at fixed benign-request FPR, precision at declared attack prevalence, PR-AUC, per-client false rejection and added latency. Avoid accuracy alone on artificially balanced request datasets.
- **AMIA:** accepted-release coverage; gradient-based BA/AUROC conditional on acceptance; overall attack results using the full observable transcript; queries needed; release-accounting totals. Zero accepted gradients means gradient AUROC is undefined, not 0.5 or perfect privacy.
- **Reference:** BA, AUROC, TPR/FPR and low-FPR TPR only where sample resolution supports it. Include threshold calibration provenance.
- **RAG privacy:** datastore BA/AUROC or applicable attack statistic, TPR/FPR, recognition and refusal rates, query budget and corpus-specific results. Unrecognized answers must not manufacture apparent privacy.
- **LLM utility:** held-out next-token NLL and task F1/EM without retrieval; training loss trajectories and convergence.
- **RAG utility:** F1/EM, answer support/faithfulness on an audited subset, retrieval recall, false refusal, answer latency and authorized task completion.
- **Usability:** fraction of successful legitimate rounds, queue completion, resource cost and recovery burden.
- **Formal accounting:** epsilon/delta, privacy unit, adjacency, accountant and covered releases. Do not combine different-unit epsilon values into a ranking or claim the detector itself supplies DP.

Report paired differences with intervals clustered by independent target/run; use a nested design where appropriate. Do not bootstrap repeated batches or the same RAG questions as independent records. Keep tuning and final-test analyses separate. Report per-target results as well as pooled scores, since target-specific score scales may differ.

Failed infrastructure runs are excluded from attack-effectiveness estimates but counted in operational failure summaries. Successful refusals remain in the experiment. Utility-invalid completed runs remain in the report and are excluded from claims of usable protection. Publish denominators and reasons for every exclusion; report sensitivity to incomplete conditions.

## 10. Testing instructions and acceptance checks

Proposed automated tests:

- `tests/test_client_guard.py`: permitted/rejected architecture, nonfinite values, policy overrides, missing artifacts and schema errors; verify rejected requests never access private data or return gradients.
- `tests/test_guard_detector.py`: stable preprocessing, pinned artifact loading, grouped split disjointness and prohibited-feature rejection.
- `tests/test_guard_release_ledger.py`: restart persistence, concurrent debits, exhausted budgets and retry behavior.
- `tests/test_guard_integration.py`: accepted/rejected Flower responses, one ledger debit per new release, aggregation behavior and explicit policy-aborted training.
- Extend `test_metrics.py`: mixed accepted/rejected trials, all-rejected case, transcript attacks and invalid RAG responses.
- Extend `test_pipeline.py`, `test_hash_equivalence.py`, `test_queue.py`: optional-schema compatibility, content-based identities and strict validation.
- Extend `test_trial_checkpointing.py`, `test_result_storage.py`: AMIA guard progress, interruption persistence and result round-trip with null scores.
- Extend `test_webui_configs.py` and result-page tests: guard controls, detection/rejection display and no false completed-score claims.

After implementation, run targeted CPU tests first:

```bash
python -m pytest tests/test_client_guard.py tests/test_guard_detector.py tests/test_guard_release_ledger.py tests/test_guard_integration.py tests/test_metrics.py tests/test_pipeline.py tests/test_hash_equivalence.py tests/test_queue.py tests/test_trial_checkpointing.py tests/test_result_storage.py
```

Then run the existing regression suite in the project's supported environment:

```bash
python -m pytest tests
```

Run one GPU smoke experiment with minimal rounds/targets solely for correctness, followed by the small guarded pilot. Verify GPU selection before launch, incremental observations, expected rejection counts and valid ordinary answers. Tiny smoke results are not scientific evidence.

Once the new configuration and parser are implemented, the intended project-root command is:

```bash
python -m master_script.perform_experiments --queue master_script/configs/client_guard_privacy_pilot.yaml --attack amia --attack reference --no-charts
```

The proposed configuration must explicitly include both attack sections. No new command above is executable until its proposed files exist. Reusing `--attack reference` alone will again exclude AMIA. Record the resolved queue and planned run count before launching.

## 11. Implementation order and deliverables

1. Add typed guard responses, local policy and release ledger; maintain legacy defaults.
2. Add rules and shadow-mode instrumentation to AMIA and legitimate training; test persistence and data boundaries.
3. Build disjoint benign/malicious request datasets; train a small detector and freeze its artifact and threshold.
4. Run the six-arm AMIA validation pilot; stop classifier expansion if simple rules/noise dominate.
5. Test guarded legitimate training, then evaluate matched LLM/RAG utility and Reference risk.
6. Add the four membership-overlap cells and adaptive-attacker evaluation.
7. Produce a final reproducible report with source revision, config/data/artifact hashes, counts, exclusions, accounting scopes, paired uncertainty, per-client outcomes and privacy–utility plots.

The study succeeds scientifically if it determines when detection adds value, when it does not, and which leakage remains. A system-wide privacy improvement claim requires evidence for training, fresh client releases and retrieval outputs while the complete system remains useful.
