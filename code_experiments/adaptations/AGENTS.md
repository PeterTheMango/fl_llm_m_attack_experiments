# AGENTS.md

Guidance for adapting membership inference attacks to federated LLM fine-tuning experiments in this folder.

## Core Rule

Adapt attacks to the federated learning setting first. Do not turn an FL membership inference attack into a centralized fine-tuning or black-box-only experiment unless the user explicitly asks for that comparison.

The first adaptation notebook, `../AMIA_adaptation.ipynb`, is the reference implementation pattern:

1. Federated fine-tune an open-source LLM.
2. Run the active membership inference attack against a client update path.
3. Measure attack performance with the paper's metrics.
4. Persist and retrieve all results through Firebase Firestore.

## Attack Adaptation Pattern

When adapting a paper attack to FL-based LLM fine-tuning:

- Read the source paper and identify the exact threat model, attacker capability, target record definition, and metrics before writing code.
- Preserve the original security game when possible: construct positive worlds where the target record is included in a client dataset and negative worlds where it is absent.
- Keep the server/adversary role explicit. In FL experiments, the server may initialize parameters, select clients, observe client updates, and aggregate updates depending on the paper's assumptions.
- Fine-tune the LLM through FL, not centralized training. Use FedAvg or a clearly documented FL strategy: copy global weights to clients, train locally on client partitions, collect updates, and aggregate.
- Partition data by client. A target record belongs to a specific target client, and non-target examples should come from held-out or other-client partitions consistent with the paper.
- Adapt the attack mechanism to LLM internals only after preserving the FL control flow. For the first notebook, the image chosen-neuron AMI attack becomes a malicious probe over frozen LLM hidden states.
- Infer membership from the client update signal the paper relies on. In the first notebook, the server observes whether the malicious probe has a non-zero gradient induced by positive activations.
- Run attack trials in matched positive and negative worlds. The only intended difference should be target membership.
- Report at minimum `TPR`, `TNR`, and `Adv = 0.5 * TPR + 0.5 * TNR`, matching Nguyen et al. (2023), Eq. 3.
- Include extra metrics such as accuracy, precision, recall, F1, or ROC-AUC only as secondary diagnostics.

## Notebook Structure

Use this order for new adaptation notebooks:

1. **Configuration**
   - Define model id, dataset, client count, FL rounds, local epochs, target client id, attack trials, random seed, and Firestore collection.
   - Compute a stable experiment id by hashing the complete config.
   - Support independent factor sweeps by declaring a grid of variable lists, then expanding the grid into one immutable config per run.
   - Keep each run isolated: one config hash, one Firestore document, one artifact directory, and one cleanup step.

2. **Firestore Cache Check**
   - Initialize Firebase Admin.
   - Look up the experiment document by config hash.
   - If a complete result exists, load it and skip expensive fine-tuning and attack execution.

3. **Federated Fine-Tuning**
   - Load an open-source model with `AutoModelForCausalLM`.
   - Load or construct client-partitioned text data.
   - Locally fine-tune each selected client model.
   - Aggregate updates with FedAvg or the stated FL algorithm.
   - Save local model artifacts under an experiment-specific artifact folder.

4. **Attack Construction**
   - Translate the paper's attack primitive to LLM-compatible signals.
   - For chosen-neuron AMI, train a probe or adapter that activates for the target text and stays inactive for non-target text.
   - Keep this attack component separate from normal FL model utility code.

5. **Attack Execution**
   - Sample positive and negative client batches.
   - Observe the client update, gradient, activation, or loss signal authorized by the threat model.
   - Convert the signal to a membership prediction using a documented threshold or classifier.

6. **Measurement**
   - Compute TPR, TNR, and Adv.
   - Store per-trial truth labels, scores, and predictions so results are auditable.

7. **Firestore Write**
   - Persist config, method summary, FL history, attack metrics, per-trial records, artifact paths, and status.
   - Write with merge semantics so partial metadata can be updated without destroying prior results.

8. **Artifact Cleanup**
   - After a run is persisted successfully, delete local model and probe artifacts for that run unless `keep_artifacts=True`.
   - Store compact metadata and metrics in Firestore before cleanup.
   - Never delete artifacts for a run whose Firestore write failed.

## Independent Factor Sweeps

Experiments should make it easy to test one or more factors independently. Use a parameter grid where every variable is listed explicitly, then expand the grid into complete configs. Each expanded config is a separate experiment.

Good sweep variables:

- `model_id`: open-source LLM checkpoint.
- `num_clients`: number of simulated FL clients.
- `clients_per_round`: client participation per FL round.
- `federated_rounds`: number of server aggregation rounds.
- `local_epochs`: local fine-tuning epochs per selected client.
- `local_batch_size`: client update batch size.
- `client_lr`: local fine-tuning learning rate.
- `target_client_id`: client that may contain the target record.
- `attack_trials`: number of positive/negative membership worlds to evaluate.
- `gradient_threshold`: attack decision threshold.
- `probe_epochs`: malicious probe training epochs.
- `probe_lr`: malicious probe learning rate.
- `ldp_mechanism`: optional LDP mechanism such as `none`, `BitRand`, or `OME`.
- `epsilon`: optional LDP privacy budget.
- `seed`: random seed.

Recommended grid expansion pattern:

```python
from dataclasses import replace
from itertools import product

SWEEP = {
    "model_id": ["sshleifer/tiny-gpt2", "distilgpt2"],
    "federated_rounds": [1, 2, 4],
    "num_clients": [4, 8],
    "local_epochs": [1],
    "client_lr": [5e-5, 1e-4],
    "seed": [7, 11, 23],
}

def expand_sweep(base_config, sweep):
    keys = list(sweep.keys())
    for values in product(*(sweep[key] for key in keys)):
        updates = dict(zip(keys, values))
        yield replace(base_config, **updates)
```

Recommended sweep runner pattern:

```python
def run_sweep(base_config, sweep, keep_artifacts=False):
    results = []
    for config in expand_sweep(base_config, sweep):
        run_id = experiment_key(config)
        cached = load_cached_result(config)
        if cached and cached.get("status") == "complete":
            results.append(cached)
            continue

        artifact_dir = artifact_dir_for(config)
        try:
            result = run_single_experiment(config, artifact_dir=artifact_dir)
            save_result(config, result)
            results.append(result)
            if not keep_artifacts:
                cleanup_artifacts(artifact_dir)
        except Exception:
            mark_result_failed(config)
            raise

    return results
```

Rules for factor sweeps:

- Treat every expanded config as immutable after hashing.
- Do not reuse model objects across configs unless the experiment explicitly studies warm-starting.
- Reset random seeds at the start of every run.
- Reinitialize the tokenizer, model, optimizer, FL clients, probe, and attack state for every run.
- Use Firestore cache checks per config, not once for the whole sweep.
- Compare runs by reading Firestore records into a dataframe after the sweep completes.
- Keep full local artifacts only for debugging, publication checkpoints, or runs marked as best candidates.

Recommended cleanup helper:

```python
import shutil
from pathlib import Path

def cleanup_artifacts(artifact_dir):
    artifact_dir = Path(artifact_dir)
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
```

Cleanup safety rules:

- Cleanup happens only after metrics and trial records are stored in Firestore.
- Cleanup removes generated model/probe artifacts, optimizer states, and temporary tokenized datasets.
- Cleanup must not remove source notebooks, `AGENTS.md`, cloned paper repositories, credentials, or shared datasets.
- If model artifacts are uploaded to Cloud Storage, store the `gs://` URI in Firestore before deleting local files.

## Firebase Firestore Setup

Use Firebase Firestore specifically for experiment persistence.

Required Python package:

```python
%pip install -U firebase-admin
```

Authentication options:

- Preferred local option: set `GOOGLE_APPLICATION_CREDENTIALS` to a Firebase service account JSON path.
- CI or notebook secret option: set `FIREBASE_SERVICE_ACCOUNT_JSON` to the full service account JSON string.
- Optional: set `FIREBASE_PROJECT_ID` if the project id is not inferred from credentials.

Expected initialization pattern:

```python
import json
import os
import firebase_admin
from firebase_admin import credentials, firestore

def get_firestore_client(project_id=None):
    if not firebase_admin._apps:
        raw_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
        cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if raw_json:
            cred = credentials.Certificate(json.loads(raw_json))
        elif cred_path:
            cred = credentials.Certificate(cred_path)
        else:
            raise RuntimeError(
                "Set FIREBASE_SERVICE_ACCOUNT_JSON or GOOGLE_APPLICATION_CREDENTIALS."
            )

        options = {"projectId": project_id} if project_id else None
        firebase_admin.initialize_app(cred, options=options)

    return firestore.client()
```

Expected cache pattern:

```python
doc = db.collection("ami_federated_llm_results").document(run_id)
snapshot = doc.get()
if snapshot.exists and snapshot.to_dict().get("status") == "complete":
    result = snapshot.to_dict()
else:
    result = run_experiment()
    doc.set(result, merge=True)
```

Recommended Firestore document schema:

```python
{
    "run_id": "stable_config_hash",
    "status": "complete",
    "updated_at_unix": 0,
    "config": {},
    "methodology": {
        "paper_attack": "summary of original paper attack",
        "llm_adaptation": "summary of adapted FL LLM attack",
        "metric_definition": "Adv = 0.5 * TPR + 0.5 * TNR",
    },
    "federated_history": [],
    "metrics": {
        "tpr": 0.0,
        "tnr": 0.0,
        "adv": 0.0,
        "num_trials": 0,
    },
    "attack_trials": [
        {
            "trial_id": 0,
            "truth_member": True,
            "score": 0.0,
            "pred_member": True,
        }
    ],
    "artifacts": {
        "federated_model_path": "local/or/remote/path",
        "probe_path": "local/or/remote/path",
    },
}
```

Firestore should store metrics, configs, and compact trial records. Do not store large model weights directly in Firestore. Save model/probe artifacts locally or in Cloud Storage, then store paths or `gs://` URIs in the document.

## Quality Checks

Before considering an adaptation complete:

- Verify the notebook fine-tunes through FL before running the attack.
- Verify target membership is represented at the client-data level.
- Verify positive and negative attack trials are both present.
- Verify `TPR`, `TNR`, and `Adv` are computed from trial records.
- Verify Firestore lookup happens before compute and Firestore write happens after measurement.
- Verify the notebook can be rerun without recomputing completed experiments when Firestore has a complete document.
- Document any deviation from the source paper's threat model.
