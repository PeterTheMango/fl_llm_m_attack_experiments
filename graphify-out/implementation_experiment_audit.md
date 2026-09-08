# Implementation and Experiment Parity Audit

## Audit conclusion

The repository contains **11 executable attack paths but 10 manuscript-level named families**. Per the requested terminology, the **AMIA/reference family** contains two distinct implementations: the active hidden-state probe path (`amia`) and the comparison-model calibration path (`reference`). They share neither a decision statistic nor a trial protocol and must remain separate rows in implementation tables.

There is **not one exact protocol shared by all 11 paths**. Nine “modern” paths share one runner/federation scaffold; `amia` and `loss` use legacy custom pipelines. The strongest notebook-to-master evidence is byte-exact configuration and run-ID parity. Runtime parity is partial: the generic runner changes notebook-specific reference construction for some methods, toy SaMIA uses a different generator, and no Hugging Face path is exercised end-to-end by the relevant tests. The manuscript may describe these as implemented adaptations, but it must not describe all 11 as empirically validated under one identical protocol.

## 1. Exact implemented protocol

### 1.1 Nine-path modern protocol

This protocol applies to `zlib`, `min_k`, `min_k_plus_plus`, `neighborhood`, `recall`, `reference`, `samia`, `spv_mia`, and `wbc`.

1. A frozen attack config is expanded over a Cartesian sweep. The run ID is the first 16 hexadecimal characters of SHA-256 over compact, sorted JSON of `asdict(config)` (`master_script/core/config.py:19-67`).
2. `run_single_experiment` checks Firestore for a complete document before compute. On a miss it creates a run-specific artifact directory (`master_script/core/runner.py:57-68`).
3. Trials alternate membership labels exactly as `True, False, True, False, ...`. Trial `i` uses a copied config with `seed = base_seed + i`; the original config, not the trial copy, is hashed (`master_script/core/runner.py:26-54`).
4. Positive and negative worlds use the same fixed synthetic client corpus. Client 0 receives either one fixed target record or one fixed held-out replacement; the other records are unchanged (`master_script/core/federation.py:15-32`).
5. The default **toy** backend trains a unigram token-count model. In every round it selects client IDs from zero upward, locally adds token counts for `local_epochs`, and averages all selected client token counts (`master_script/core/federation.py:35-86`).
6. With `use_hf_models=true`, each client reloads the configured causal LM, receives the current global parameters, trains with AdamW, and returns its parameters to Flower FedAvg. Every modern client reports `num_examples=1`, so aggregation is an **unweighted** mean, regardless of actual record count. The default selects all four clients (`master_script/core/federation.py:89-212`).
7. The final model scores the same fixed target record. The generic decision rule is `score >= threshold` (`master_script/core/runner.py:30-46`). All modern scores are oriented so higher means more likely member.
8. The base metric set is computed, optional attack metrics are added, and compact trials plus per-trial federation history are assembled (`master_script/core/metrics.py:40-70`; `master_script/core/runner.py:86-104`).
9. The result is written with Firestore merge semantics. Artifacts are deleted only after a successful write unless `keep_artifacts=true`; failed runs retain artifacts (`master_script/core/runner.py:68-111`).

The phrase “matched positive and negative worlds” needs qualification. The partition builder differs only in the target payload **when called with the same config**, which is unit-tested (`tests/test_federation.py:5-24`), but actual alternating trials use different seeds (`seed+i`). Thus an adjacent positive/negative trial pair is not a strict same-seed matched pair, particularly on stochastic Hugging Face/Flower paths.

### 1.2 AMIA/reference family: active AMIA path

`amia` is a custom Hugging Face-only pipeline, not a `ScoreContext` trial:

1. It builds 12 records per client; the target text replaces client 0’s first record.
2. It performs one two-round Flower FedAvg fine-tune with the target present. Clients report `num_examples=1`, hence unweighted aggregation.
3. It mean-pools final hidden states and trains a two-layer width-128 probe for 80 epochs. Probe positives are repeated copies of the target text; negatives are all non-target client records.
4. It then samples attack **batches** from that already-trained world. Even-numbered trials inject the target into the batch; odd-numbered trials do not. The score is the L2 norm of the gradient of the probe’s output weight induced by the ReLU-summed probe activation.
5. Membership is predicted only when `gradient_norm > 1e-8` (strict `>`, not `>=`).

Sources: `master_script/core/attacks/amia.py:23-48`, `:64-108`, `:224-300`, `:323-416`.

This path’s `truth_member` denotes membership of the target in the **attack batch**, while the federated model and probe were both trained with the target present. It is therefore an active/batch-membership game, not the independently trained positive-versus-negative record-membership game used by the modern paths and `loss`.

### 1.3 LOSS path

`loss` is also custom and Hugging Face-only:

1. Every trial independently constructs a positive or negative client-data world. Client 0 gets either the fixed target or a trial-specific held-out replacement.
2. A two-round Flower FedAvg model is trained from the base checkpoint. Unlike the other paths, clients report their actual dataset sizes, so FedAvg is example-weighted.
3. The final model computes mean token NLL on 24 fixed calibration non-members. The trial-specific decision threshold is their 10th percentile.
4. The same model computes target NLL and predicts membership when `target_nll <= calibrated_threshold`.
5. Trials alternate positive and negative worlds, but the FL seed is reset to the same default seed (13) for every trial rather than `seed+i`.

Sources: `master_script/core/attacks/loss.py:28-53`, `:80-131`, `:225-358`, `:361-449`.

## 2. Exact defaults and attack-specific parameters

### Shared defaults for the nine modern paths

Unless a row below says otherwise: `model_id=sshleifer/tiny-gpt2`; `dataset_name=synthetic_client_text`; `num_clients=4`; `clients_per_round=4`; `federated_rounds=1`; `local_epochs=1`; `local_batch_size=2`; `client_lr=5e-5`; `target_client_id=0`; `attack_trials=4`; `max_length=64`; `seed=7`; `firestore_collection=ami_federated_llm_results`; `fl_framework=flower`; `sim_num_gpus=0.0`; `keep_artifacts=false`; `use_hf_models=false`. Artifact roots follow `artifacts/<attack>_adaptation`. These defaults are defined in each module and mechanically compared with the notebook dataclasses by `tests/test_hash_equivalence.py:37-48`.

| Manuscript family | Executable path | Defaults differing from the modern block | Attack-specific defaults | Implemented score and decision |
|---|---|---|---|---|
| zlib | `zlib` | None | `threshold=-0.0058` | `-(mean NLL / zlib-compressed bits)`; member iff `score >= -0.0058` (`zlib.py:61-83`) |
| Min-K% | `min_k` | None | `min_k_percent=20`; `threshold=-4.08` | Mean of lowest 20% next-token log-probabilities; member iff `score >= -4.08` (`min_k.py:80-119`) |
| Min-K%++ | `min_k_plus_plus` | None | `min_k_percent=20`; `threshold=-3.25` | Mean of lowest 20% vocabulary-calibrated token z-scores; member iff `score >= -3.25` (`min_k_plus_plus.py:100-163`) |
| Neighbourhood | `neighborhood` | None | `num_neighbours=25`; `neighbour_swaps=1`; `threshold=0.02` | Mean neighbour NLL minus target NLL; member iff `score >= 0.02` (`neighborhood.py:69-77`, `:112-201`) |
| ReCaLL | `recall` | None | `num_shots=1`; `threshold=1.05` | `LL(x | fixed non-member prefix) / LL(x)`; member iff `score >= 1.05` (`recall.py:67-81`, `:116-172`) |
| **AMIA/reference** | `reference` comparison-model path | None | `threshold=0.25`; reference checkpoint defaults to `model_id` | Reference NLL minus target NLL; member iff `score >= 0.25` (`reference.py:53-96`) |
| **AMIA/reference** | `amia` active-probe path | `dataset=synthetic_canary_clients`; `rounds=2`; `batch=4`; `trials=64`; no toy/HF switch; artifact root `artifacts/adaptation` | `probe_lr=5e-3`; `probe_epochs=80`; `attack_batch_size=8`; `gradient_threshold=1e-8`; probe width 128 | Probe-gradient L2 norm; member iff `score > 1e-8` (strict) (`amia.py:23-48`, `:323-408`) |
| SaMIA | `samia` | None | `num_samples=10`; `rouge_n=1`; `use_zlib_weighting=false`; `threshold=0.5`; HF generation `temperature=1`, `top_k=50`, `top_p=1` | Mean ROUGE-N recall against the target suffix, optionally multiplied by compressed bits; member iff `score >= 0.5` (`samia.py:80-165`) |
| SPV-MIA | `spv_mia` | None | `num_paraphrases=4`; `mask_ratio=0.2`; `self_prompt_tokens=8`; `threshold=0.0`; HF self-generation `top_k=50` | `[p(x)-mean p(paraphrases)]_target - [...]_reference`; member iff `score >= 0` (`spv_mia.py:111-168`, `:187-272`) |
| WBC | `wbc` | Adds `reference_model_id=sshleifer/tiny-gpt2` | `window_sizes=(2,3,4,6,9,13,18,25,32,40)`; `threshold=0.75` | Fraction of windows with `sum(reference NLL - target NLL) > 0`; member iff `score >= 0.75` (`wbc.py:16-42`, `:91-166`) |
| LOSS | `loss` | `dataset=synthetic_private_client_text`; `seed=13`; `rounds=2`; `batch=4`; `trials=12`; collection `loss_federated_llm_results`; no toy/HF switch; artifact root `artifacts/adapted_loss` | `threshold_quantile=0.10`; `calibration_nonmember_count=24` | Raw mean target NLL; member iff `loss <= trial-specific 10th-percentile calibration threshold` (`loss.py:28-53`, `:380-419`) |

The shipped `smoke.yaml` overrides only `use_hf_models=false` and `attack_trials=4` and lists the nine modern paths. It does not run `amia` or `loss` (`master_script/configs/smoke.yaml:1-16`). `example_sweep.yaml` is not an all-method real protocol: it runs only `zlib` and `min_k`, changes the model to `distilgpt2`, sets HF/GPU mode, uses 12 trials, and sweeps rounds and seed (`master_script/configs/example_sweep.yaml:1-17`).

## 3. Toy and Hugging Face paths

| Executable path(s) | Toy path | Hugging Face / Flower path | Verification status |
|---|---:|---:|---|
| `zlib`, `min_k`, `min_k_plus_plus`, `neighborhood`, `recall`, `reference`, `samia`, `spv_mia`, `wbc` | Yes | Yes | Toy scoring and runner smoke contracts tested; HF scoring/training not run end-to-end by the relevant tests |
| `amia` | No | Yes, mandatory custom pipeline | Threshold logic and mocked runner wiring tested; heavy FL/probe path not executed |
| `loss` | No | Yes, mandatory custom pipeline | Calibration logic and mocked runner wiring tested; heavy FL path not executed |

The existence of a `score_hf` function is implementation support, not evidence that the path has produced a valid paper result.

## 4. Score direction and thresholds

- The nine modern paths use a normalized **higher-is-more-member** convention and `score >= constant_threshold`.
- `amia` is higher-is-more-member but uniquely uses strict `score > gradient_threshold` (`tests/test_legacy_attacks.py:26-32`).
- `loss` stores raw NLL, so **lower is more member** and uses `loss <= calibrated_threshold`. Its AUC explicitly negates loss before evaluation (`loss.py:452-480`).
- Modern thresholds are fixed config constants and are not calibrated by the runner. Notebook comments recommend held-out calibration or threshold-free AUC for real runs. They should not be described as universal paper thresholds.
- A direct current-code diagnostic of the four-trial toy defaults found perfect ranking (`roc_auc=1`) for every modern path that records AUC, while the fixed threshold gave `adv=0.5` for `min_k_plus_plus`, `reference`, `samia`, and `spv_mia`. The smoke/test contract generally checks ordering and key presence, not reliable fixed-threshold classification (`tests/test_scoring.py:15-61`; `tests/test_toy_parity.py:19-34`). This is a software diagnostic on synthetic data, not an empirical attack result.

## 5. Metrics and stored records

### Modern paths

The common metric object contains exactly:

`tp`, `tn`, `fp`, `fn`, `tpr`, `tnr`, `adv`, `accuracy`, `precision`, `recall`, `f1`, `num_trials`.

`adv = 0.5 * TPR + 0.5 * TNR`, i.e. balanced accuracy (`master_script/core/metrics.py:40-62`). Its random-guess baseline is 0.5. It is **not** the centered advantage `TPR-FPR`; the conversion is `centered_advantage = 2*adv - 1`. Main text must define the repository’s convention explicitly.

Extra metric keys are:

- `roc_auc`: zlib, Min-K%, Min-K%++, neighbourhood, ReCaLL, SaMIA, SPV-MIA, and WBC.
- No AUC: comparison-model `reference` path, intentionally matching its adaptation notebook.
- SaMIA additionally records `tpr_at_10fpr`, computed at FPR 0.1 (`samia.py:176-197`).

Modern compact trials store `trial_id`, `truth_member`, `score`, and `pred_member`; federation history is stored separately as one map per trial (`runner.py:93-103`).

### Legacy paths

- AMIA: `tpr`, `tnr`, `adv`, `accuracy`, `precision`, `recall`, `f1`, `roc_auc_from_scores`, `num_trials`, `positive_trials`, `negative_trials`; its trials also retain `batch_size` (`amia.py:419-477`).
- LOSS: `tpr`, `tnr`, `adv`, `accuracy`, `roc_auc_loss_inverted`, `num_trials`, `member_mean_loss`, `nonmember_mean_loss`; compact trials retain their calibrated threshold and negative-world replacement text (`loss.py:452-517`).

Only `tpr`, `tnr`, `adv`, `accuracy`, and `num_trials` are consistently present across all 11 paths. AUC is neither universally present nor universally named. Modern federation history records selected clients but no training loss; AMIA and LOSS record loss-bearing histories with different schemas. The current result documents do not provide a uniform model-utility metric.

## 6. Notebook-to-master equivalence evidence

### Strong, mechanical evidence

1. `tests/notebook_configs.py` reads the `ExperimentConfig` class directly from every adaptation notebook rather than from a hand-maintained copy.
2. For all 11 paths, tests compare dataclass field order and defaults, default run IDs, run IDs after non-default seed/round sweeps, and frozen golden IDs. Three notebook-specific key formulas are preserved: 16-char SHA for the nine modern paths, 24-char SHA for AMIA, and an experiment-name-prefixed 16-char SHA for LOSS (`tests/test_hash_equivalence.py:1-115`).
3. During this audit, the relevant parity suite passed: **121 tests passed** across `test_hash_equivalence`, `test_federation`, `test_scoring`, `test_toy_parity`, `test_legacy_attacks`, `test_legacy_runner_wiring`, `test_metrics`, and `test_runner`.
4. The modern federation layer was ported from the zlib notebook’s shared cells and is stated to be identical across the nine notebooks (`master_script/core/federation.py:1-6`). World construction, deterministic toy behavior, and membership-induced NLL ordering are unit-tested (`tests/test_federation.py`).

### Partial behavioral evidence

1. All nine toy scorers are tested for deterministic output and member-score-above-nonmember ordering (`tests/test_scoring.py:1-37`).
2. All nine run through the consolidated toy runner and emit both worlds plus TPR/TNR/Adv. Only zlib has an explicit consolidated-run `roc_auc == 1` assertion (`tests/test_toy_parity.py`).
3. AMIA/LOSS threshold semantics are unit-tested, and their runner handshake is tested with the expensive FL/torch work monkeypatched (`tests/test_legacy_attacks.py`; `tests/test_legacy_runner_wiring.py`).

### Evidence that does **not** exist

1. No test numerically compares notebook scorer outputs with master-script scorer outputs over identical Hugging Face models and inputs.
2. No relevant test executes Flower plus Hugging Face end-to-end for any method. AMIA and LOSS heavy paths are mocked.
3. The adaptation and recreation notebooks have no saved execution counts or outputs; they are source artifacts, not retained run evidence.
4. The nine recreation notebooks are stand-alone synthetic correctness demonstrations with optional real-model hooks, generally using percentile-derived smoke thresholds and often `max_length=256`. They are not the FL adaptation protocol and are not called by `master_script`.
5. The AMIA recreation is a vendored, unmodified image-classification FL/LDP repository at upstream commit `989bc54`, using CelebA, CIFAR-10, and ImageNet and supporting BitRand/OME (`code_experiments/recreations/ami/README.md`; `UPSTREAM_SOURCE.txt`). It is provenance for the original attack, not end-to-end validation of the LLM hidden-state-probe adaptation. There is no corresponding LOSS recreation notebook.

The defensible parity claim is therefore: **configuration, cache-key, core metric formulas, and much of the modern toy/scoring structure are preserved; full numerical runtime equivalence for all attack backends is not established.**

## 7. Material deviations and limitations that `main.tex` must disclose

### Cross-cutting experimental limitations

1. **Synthetic data only in the implemented default protocol.** `dataset_name` is metadata; the modern federation code always uses the fixed three-sentence-per-target-client synthetic corpus. AMIA and LOSS likewise use hard-coded synthetic canaries and client text. No configured real dataset loader is used by these attack runners.
2. **Tiny/default scale.** Nine paths default to tiny GPT-2, four clients, one FL round, one local epoch, and four trials (two positive/two negative). AMIA defaults to 64 batch-membership trials; LOSS to 12 independently trained worlds. These counts are not statistically comparable.
3. **Toy results are not LLM/FL results.** The default smoke configuration uses a deterministic token counter, no Hugging Face model, no Flower simulation, no GPU, and no Firestore. Toy-specific constructions deliberately amplify ordering and cannot support quantitative privacy claims.
4. **No all-method HF experiment config or retained output.** The only shipped real sweep covers zlib and Min-K%. Notebook cells are unexecuted and no all-11 HF results are present in the audited artifacts.
5. **No uncertainty estimates.** The pipeline reports point metrics only: no confidence intervals, bootstrap intervals, or repeated-run dispersion.
6. **No uniform utility measurement.** Attack metrics are recorded, but model utility (held-out perplexity/task accuracy) is not uniformly measured. Modern histories do not include client training losses.
7. **`adv` terminology.** Repository `adv` is balanced accuracy in `[0,1]`, not centered MIA advantage. A random baseline is 0.5, not zero.
8. **Thresholds are not cross-method calibrated.** Modern fixed thresholds are attack-specific smoke constants. Several yield perfect toy ranking but chance-level `adv` at the stored cutoff. Comparisons should prioritize consistently computed threshold-free metrics or use a documented common calibration split.
9. **Trial matching is imperfect.** Modern trials increment the seed with trial ID, so positive and negative worlds are not identical-randomness pairs. HF model initialization, dataloader shuffling, and Flower client sampling are not globally seeded in the modern federation function.
10. **FedAvg semantics differ.** Modern and AMIA clients force `num_examples=1` (unweighted mean); LOSS uses actual example counts. This is not one aggregation protocol.
11. **Metrics/result schemas differ.** Reference omits AUC; AUC keys differ for AMIA and LOSS; history schemas and retained diagnostics differ. Direct averaging across raw Firestore fields is unsafe without normalization.

### Implementation-parity gaps

12. **Reference construction is changed by the generic runner.** For every modern `needs_reference` attack, toy mode passes a separately fine-tuned negative-world model; the notebooks use an untrained reference for `reference` and WBC and a self-prompt-trained reference for SPV-MIA. In HF mode, the runner always loads a plain pretrained reference bundle. That is correct for `reference`/WBC but bypasses SPV-MIA’s self-prompt reference construction. The scorer fallbacks implement the notebook behavior only when `ctx.reference is None`, which the runner never allows for these specs (`runner.py:30-39`; `reference.py:77-86`; `spv_mia.py:260-286`; `wbc.py:154-180`). **Master-script SPV-MIA runs are therefore not notebook-equivalent as wired.**
13. **SaMIA toy generation is a surrogate.** The notebook toy model tracks bigram transitions; the consolidated model only has unigram counts, so `samia.py` ranks frequent tokens deterministically instead. ROUGE scoring is preserved, but generation behavior is not (`samia.py:117-127`; `.superpowers/sdd/task-8-report.md:64-75`).
14. **Notebook-specific result metadata is dropped.** The generic runner stores only `artifact_dir` and `federated_model_path`, whereas notebooks record such fields as `reference_model` and ReCaLL prefix metadata. WBC’s `config_to_storage` helper is defined but not used by the generic runner. This prevents a strict claim of byte-identical Firestore document shape even when hashes match (`runner.py:86-104`; `wbc.py:147-151`).
15. **Test coverage overstates neither threshold quality nor HF parity.** Toy tests primarily assert ordering/determinism and field presence; AMIA/LOSS FL work is mocked. Passing tests establish software contracts, not paper-scale attack efficacy.

### Method-specific scientific deviations

16. **AMIA/reference family — AMIA path:** the vendored original is an active FL attack on image classifiers and includes LDP experiments (BitRand/OME). The LLM adaptation replaces the image/chosen-neuron mechanism with a learned hidden-state probe, omits LDP mechanisms/epsilon entirely, trains the model and probe with the target present, and labels membership by target inclusion in an attack batch. It is an adapted active-probe experiment, not a reproduction of the original AMIA security game.
17. **AMIA/reference family — comparison-model path:** it is a final-model likelihood comparison against a base checkpoint, not the active-gradient AMIA path. Grouping them is nomenclature only; results and threat models must remain separate.
18. **LOSS:** Yeom et al.’s rule is moved from classical supervised learning to causal-LM FedAvg. The implemented threshold is a per-trained-model 10th percentile of synthetic calibration nonmembers, not a single common threshold.
19. **zlib:** the Carlini zlib ratio is transferred from pretraining-data extraction to a fine-tuned FL model; the score is additionally sign-flipped and uses compressed **bits** (8× bytes), which changes threshold scale but not ranking.
20. **Min-K%:** the statistic transfers reasonably to fine-tuning, but the default synthetic tiny-model setting does not reproduce the paper’s models, datasets, or sample sizes.
21. **Min-K%++:** the pretraining-data detector is transferred to FL fine-tuning. It requires full vocabulary logits and is not a pure black-box method.
22. **Neighbourhood:** the default is 25 neighbours, although the methodology text describes the paper-faithful HF setting as `n=100`; the code honors the config value 25. Toy neighbours are filler-word substitutions, while HF uses BERT dropout/substitution.
23. **ReCaLL:** a pretraining-membership method is applied to FL fine-tuning with one fixed synthetic nonmember prefix shot. The implementation uses total conditional/unconditional log-likelihood ratios and no external reference model.
24. **SaMIA:** a WikiMIA/pretrained-model black-box method is transferred to FL fine-tuning. The toy generator is not notebook-equivalent; HF sampling parameters are hard-coded, and no paper-scale sampling evaluation is retained.
25. **SPV-MIA:** the fine-tuning attack is transferred to federated fine-tuning. Its HF paraphraser is literal `<mask>` token replacement, not the paper-default T5 mask-and-reconstruct model. More importantly, the current shared runner bypasses its self-prompt reference model, as noted above.
26. **WBC:** the fine-tuned-LLM reference attack is transferred to FL. The chosen literal window list follows the stated/repository set despite a discrepancy with the paper’s printed generating formula (documented in `wbc_recreation.ipynb`). The toy model also introduces a hard-coded background vocabulary of 50,000.

## Manuscript-safe summary

A precise methods statement would say that the project implements 11 executable adaptations (reported as 10 named families by grouping AMIA and reference), with nine sharing a synthetic matched-world scaffold and two using custom protocols. It should identify toy versus HF results, define score direction and `adv`, state all calibration rules, and report AMIA/reference as two distinct threat models. It should claim verified config/hash parity and tested toy ordering, not universal notebook runtime equivalence or a completed, common, paper-scale evaluation across all methods.
