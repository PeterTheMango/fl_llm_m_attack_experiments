# Client guard: implementation and research workflow

The guard is now connected to AMIA observations and to AMIA/Reference federated
training. Existing guard-free configurations still load; absent guard settings
are omitted from the pipeline identity. The normal source fingerprint changes
when scientific code changes, as it does for other implementation changes.

## Implemented behavior

- Client-owned rules validate the approved observation architecture, round range,
  parameter count/shapes and finite values. Model construction, operation and
  trainable layers remain client-owned code; server fit config cannot replace
  them. Public parameter behavior is measured relative to a pinned initial
  checkpoint. An AMIA probe is compared to a fixed public random template, never
  to itself. These simulation checks are not cryptographic attestation.
- `rules`, `classifier`, `rules_classifier` and `shadow` modes are supported.
  Classifier-only mode omits the semantic architecture rule but still requires
  valid array shapes/values. Shadow logs proposed refusals while allowing valid
  requests; malformed tensors and exhausted budgets still fail closed. Shadow
  is instrumentation, not protection.
- A SQLite ledger reserves allowances before private computation and survives
  worker restarts. Duplicate accepted requests are refused, without recomputing
  noise. Errors consume allowances. Local audit records are durable, separate
  from Flower response metadata. Reservations bound possible releases; they do
  not assert delivery. A client-private directory and a shared persistent path
  for restarted workers are required.
- Paired AMIA member/nonmember worlds have separate, matched request histories.
  This prevents alternating trial order from turning an odd budget limit into
  apparent membership leakage. Ledger identities/grouping are never features.
- Rejected Flower observations carry an empty payload and an explicit refusal,
  not a zero-valued gradient. Stored score/prediction are null. Accepted AMIA
  observations retain the existing protection function, applied once, using the
  existing observation configuration as the single noise authority.
- Guarded training uses the same decision runtime. A refusal aborts the entire
  scheduled round, including under the private strategy wrapper. It is recorded
  as `policy_aborted`, distinct from infrastructure failure. Enforced runs require
  client-side DP-SGD and Gaussian observation protection; noiseless comparison
  arms require `diagnostic: true`. Central DP-FedAvg is not substituted for a
  client-side release mechanism against a malicious server.
- AMIA calibration checks the same frozen public policy/detector before scoring
  public batches. All-refused calibration has no numeric threshold. Public
  calibration does not consume private allowances. Runtime budget exhaustion is
  reported through refusal coverage/transcripts, not a made-up gradient score.
- AMIA observations checkpoint incrementally under each target artifact directory.
  The local ledger/audit also survives interruption. Completed results include
  guard summaries and private audit events; large details round-trip through the
  existing compressed storage. Results UI handles null scores and displays guard
  mode, diagnostic status, detector identity, refusals and release coverage.
  The YAML editor validates guard settings through the ordinary config loader.

`client_guard.py` retains the standalone typed boundary used in synthetic tests.
The Flower runtime uses its existing private-gradient path and does not wrap that
path in the standalone noising callback, which would add noise twice.

## Detector workflow

`guard_features.py` deliberately starts with three bounded parameter features:
mean relative layer change, maximum relative layer change and parameter
concentration. Chunked calculations bound scratch memory. Feature extraction has
no text, labels, target IDs, seeds, paths or outcomes as inputs. This version does
not add public forward-pass probes or an extra public-probe-file setting.

`guard_detector.py` implements logistic regression with fixed full-batch training,
a small L2 penalty, training-only normalization, and validation-only threshold
selection. JSON artifacts contain coefficients, preprocessing, threshold, feature
schema and group provenance; the client verifies their SHA256 before loading.
There is no pickle/executable artifact. Loading or numerical errors fail closed.

For actual research, collect benign shadow traces across independent seeds,
rounds and client distributions, plus independently optimized malicious requests.
Export result `guard_events` (or a JSON list of events). Assign complete related
run/target groups, including paired worlds, to splits **before fitting**. The
label manifest is a list containing `event_index`, `group`, `split`, `malicious`
and `variant`. Structural refusals without usable features cannot train a
behavioral detector. Group labels are research metadata only.

```sh
python -m master_script.tools.build_guard_dataset events.json labels.json traces.json
python -m master_script.tools.train_guard_detector traces.json detector.json --max-fpr 0.01
python -m master_script.tools.build_guard_pilot detector.json guarded-comparison.yaml
```

The fitter requires both classes in every split, disjoint complete groups, and an
attack variant seen only in final testing. The fitting report includes a simple
relative-delta threshold baseline at the same validation FPR target. These checks
cannot prove that an incorrectly prepared label manifest or correlated source
runs are independent. Test results must not be reused to tune the detector.
No scientific detector artifact has been fabricated or fitted from the two-target
pilot. Synthetic unit-test examples demonstrate mechanics only.

## Configurations and execution

`configs/client_guard_privacy_pilot.yaml` validates as **six mechanics runs**:
four AMIA arms (baseline, rules-only, noise-only, shadow) and two Reference arms
(baseline, shadow). It retains the existing model/dataset pins, uses one FL round
and four trials, and enables paired datastore audits. Four probe epochs and tiny
counts are solely for a smoke test, not an attack-effectiveness conclusion.

After a real detector is frozen, `build_guard_pilot` creates **eight runs**:
the requested six AMIA arms and two matched Reference arms. Its diagnostic probe
policy permits the existing AMIA head to avoid every classifier request being
trivially rejected by the architecture rule. This is a declared permissive
ablation, not proof of evasion against legitimate full-model training.

On the supported GPU environment, review the resolved queue before executing:

```sh
python -m master_script.perform_experiments --queue master_script/configs/client_guard_privacy_pilot.yaml --attack amia --attack reference --no-charts
```

Do not delete/reassign ledger scopes to resume a partially spent run unnoticed.
A fresh queue directory creates a new experimental accounting scope, which must
be composed separately when releasing results from overlapping private records.
These local simulation identities are not a defense against compromised clients,
multiple identities, or an attacker who can delete the local ledger.

## Evaluation and reporting

Gradient metrics are conditional on accepted responses, with release coverage
reported separately. All-refused gradient AUC is null. Refusal-only and client
processing-time score AUCs are separate descriptive transcript diagnostics; they
are not a trained adaptive transcript adversary. Repeated-query curves aggregate
within each target/world before reporting independent target counts.

`guard_replay.py` provides bounded public decision-oracle interpolation with
unchanged parameter shapes. This is replay/evasion mechanics, not a validated
same-architecture causal-LM membership attack. It does not inherit AMIA's theorem,
is not automatically substituted into existing AMIA experiments, and requires
an independent effectiveness/correctness study before scientific use.

Optional `rag.membership_overlap: true` evaluates two datastore worlds on each
checkpoint, replacing one document at fixed corpus size. Reference's paired
training worlds then provide the four training-membership × datastore-membership
cells. AMIA alone supplies only training-present cells. Answers, recognition and
lexical refusal diagnostics remain separate. Raw generated answers/corpus text
are not saved. Existing public/private RAG and no-context controls remain.

```sh
python -m master_script.tools.analyze_guard_study result-a.json result-b.json --output comparison.json --plot comparison.png
```

The report retains failed, aborted, unmeasured and utility-invalid conditions with
exclusion reasons. It matches baseline runs by attack, model/dataset revisions,
seed and training shape; reports F1/EM differences; and bootstraps paired target
means rather than repeated questions. The provisional gates are 90% F1 retention,
no more than a .05 EM drop and a .1 absolute F1 floor. These are engineering
choices, not validated scientific thresholds. Single-group intervals are null.
Round time and received parameter bytes are recorded for pipeline training.
Parameter snapshot loading and detector overhead still need measurement on the
actual GPU workload; the 10% round-overhead and 1% rejection gates are unproven.
Plots keep AMIA and Reference separate and omit unavailable numeric AUCs explicitly.
No aggregate system-wide privacy score is computed.

## Research basis and remaining evidence

[Vu et al., AISTATS 2024](https://proceedings.mlr.press/v238/vu24a.html)
motivates measuring active membership probes separately from training protection.
[SEER, ICLR 2024](https://www.sri.inf.ethz.ch/publications/garov2024seer)
shows why passing structural checks does not establish safety.
[Pasquini et al., CCS 2022](https://arxiv.org/abs/2111.07380)
motivates validation without implying that identical models are benign.
[Flower's NumPyClient contract](https://flower.ai/docs/framework/ref-api/flwr.client.NumPyClient.html)
provides the array/count/metrics transport used by the explicit refusal handling.

Still required for research conclusions: GPU smoke validation, representative
trace collection, a frozen independently tested detector, a validated attack
using the legitimate full-model interface, adaptive private-gradient experiments,
resource profiling, larger utility sets, audited answer support/faithfulness,
and the final powered study/report. No classifier effectiveness, full-system
privacy improvement, or new DP guarantee is established by CPU tests. This local
environment has no PyTorch, Flower or Transformers; no GPU experiment or external
result publication was started.
