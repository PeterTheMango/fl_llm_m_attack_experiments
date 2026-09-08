# Verify the queue, FL defenses, and RAG on the remote server

Run these commands on the server from the repository root. Transfer the current
working-tree changes first: this implementation has not been committed or
pushed. Include the new `master_script/core` modules, `pipeline_*` configs,
tests and docs. Existing result directories need not be transferred.

## 1. Activate the server's working CUDA environment

```bash
cd /path/to/Research-Documents
conda activate peter_experiments_fl
python -m venv --system-site-packages .venv-pipeline-verify
source .venv-pipeline-verify/bin/activate
python -m pip install 'flwr[simulation]==1.33.0' pytest pyyaml
```

The overlay venv keeps the existing CUDA PyTorch installation available without
modifying the shared conda environment. The base environment should already
contain the dependencies in `requirements.txt`. If using a fresh machine,
follow the repository's environment setup first and verify its PyTorch build
matches the server GPU. Do not replace a working CUDA PyTorch merely to run
these checks.

Select an allocated GPU by its physical index. The example uses GPU 0; change
both variables together to the GPU allocated to you. The runner uses
`EXPERIMENT_GPU`, so setting only `CUDA_VISIBLE_DEVICES` is insufficient here.

```bash
export EXPERIMENT_GPU=0
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
python - <<'PY'
import torch, transformers, flwr, ray
print('torch', torch.__version__, 'transformers', transformers.__version__)
print('flwr', flwr.__version__, 'ray', ray.__version__)
assert torch.cuda.is_available(), 'CUDA unavailable: check the environment and GPU allocation'
print(torch.cuda.get_device_name(0))
PY
```

The first real run downloads public `sshleifer/tiny-gpt2` and
`sentence-transformers/all-MiniLM-L6-v2` weights unless already cached. The
server needs outbound model-download access or a populated Hugging Face cache.
Use the server's normal job allocation/session mechanism for the long command.
Do not run multiple verification batches simultaneously on the same allocation.

## 2. Run the regression suite

This command prevents the test process from using actual Firestore credentials.
It does not change your credential files or saved configuration.

```bash
PYTHON_DOTENV_DISABLED=1 FIREBASE_SERVICE_ACCOUNT_JSON='{}' \
GOOGLE_APPLICATION_CREDENTIALS=/nonexistent/verification-credentials \
python -m pytest tests -q -p no:cacheprovider
```

Expect all tests to pass. The full suite previously passed 411 tests locally;
two later added regression cases increase the current collected total to 413.
The relevant final subsets passed locally. A FastAPI/Starlette deprecation
warning may appear and is not a failed test.

## 3. Create GPU verification copies without editing existing configs

The included demo settings use CPU clients. This creates separate copies with
half a GPU allocated per Flower client (two clients can fit on one GPU for
this tiny-model smoke test). It also creates AMIA, LOSS and no-defense controls.

```bash
python - <<'PY'
from pathlib import Path
from copy import deepcopy
import yaml
source = Path('master_script/configs')
target = source / 'remote-verification'
target.mkdir(exist_ok=False)
for name in ('pipeline_zlib_dp', 'pipeline_min_k_dp', 'pipeline_min_kpp_dp', 'pipeline_baseline'):
    doc = yaml.safe_load((source / f'{name}.yaml').read_text())
    doc['defaults']['sim_num_gpus'] = 0.5
    doc['pipeline']['rag']['study_file'] = str((source / 'pipeline_demo_study.json').resolve())
    (target / f'{name}.yaml').write_text(yaml.safe_dump(doc, sort_keys=False))
base = yaml.safe_load((target / 'pipeline_zlib_dp.yaml').read_text())
for attack, mechanism, fields in (
    ('amia', 'dp_sgd', {'probe_epochs': 2, 'attack_batch_size': 2}),
    ('loss', 'dp_fedavg', {'calibration_nonmember_count': 2}),
):
    doc = deepcopy(base)
    doc['pipeline']['defense']['mechanism'] = mechanism
    doc['attacks'] = {attack: {'base': fields}}
    (target / f'pipeline_{attack}_dp.yaml').write_text(yaml.safe_dump(doc, sort_keys=False))
print(target)
PY
```

Run this creation step once. If the directory already exists, inspect/reuse it
or choose a new directory name; the command deliberately refuses to overwrite.

## 4. Validate, then execute the required three-attack batch

```bash
python -m master_script.perform_experiments --queue \
  master_script/configs/remote-verification/pipeline_zlib_dp.yaml \
  master_script/configs/remote-verification/pipeline_min_k_dp.yaml \
  master_script/configs/remote-verification/pipeline_min_kpp_dp.yaml \
  --dry-run --no-firestore

python -m master_script.perform_experiments --queue \
  master_script/configs/remote-verification/pipeline_zlib_dp.yaml \
  master_script/configs/remote-verification/pipeline_min_k_dp.yaml \
  master_script/configs/remote-verification/pipeline_min_kpp_dp.yaml \
  --no-firestore --no-charts
```

Expected: `3/3 run(s) complete` and exit code 0. Copy the printed `queue results:`
path; each invocation creates a new directory. Do not add `--max-parallel 2`:
this check must prove sequential scheduling.

## 5. Inspect durable results and prove execution order

Replace the argument with the directory printed by the preceding command:

```bash
python - /path/from/queue-results <<'PY'
from pathlib import Path
import json, sys
root = Path(sys.argv[1])
manifest = json.loads((root / 'manifest.json').read_text())
entries = manifest['entries']
assert manifest['status'] == 'complete'
assert [e['attack'] for e in entries] == ['zlib', 'min_k', 'min_k_plus_plus']
for i, entry in enumerate(entries):
    assert entry['status'] == 'complete'
    if i:
        assert entries[i-1]['ended_unix'] <= entry['started_unix'], 'Overlapping runs'
    result = json.loads((root / entry['result_file']).read_text())
    assert result['status'] == 'complete'
    assert len(result['pipeline_evaluations']) == 2
    assert result['training_privacy_composed']['steps'] > 0
    for evaluation in result['pipeline_evaluations']:
        conditions = evaluation['rag_conditions']
        assert set(conditions) == {'public_ordinary', 'public_mirabel',
                                   'private_ordinary', 'private_mirabel'}
        assert 'no_retrieval_utility' in evaluation
        for condition in conditions.values():
            assert condition['membership_trials'] and condition['utility_trials']
            assert set(condition['utility']) == {'exact_match', 'token_f1', 'answer_nll'}
    print(entry['attack'], result['pipeline']['defense']['mechanism'],
          'epsilon:', result['training_privacy_composed']['epsilon'])
print('PASS: three complete, nonoverlapping runs with privacy and RAG outputs')
PY
```

## 6. Exercise the custom training paths and control

```bash
python -m master_script.perform_experiments --queue \
  master_script/configs/remote-verification/pipeline_amia_dp.yaml \
  master_script/configs/remote-verification/pipeline_loss_dp.yaml \
  master_script/configs/remote-verification/pipeline_baseline.yaml \
  --no-firestore --no-charts
```

Expect `3/3 run(s) complete`. AMIA has one pipeline evaluation because it trains
one model and then its probe. LOSS has two. The baseline has two RAG evaluations
and no `training_privacy_composed`, since training is not private. Inspect its
manifest and per-run JSON; the step-5 script is specifically for the first batch.

These tiny-model checks establish integration only. Zero answer utility or
uninformative membership scores are plausible. Scientific effectiveness needs
larger models, curated held-out questions, sufficient membership trials and
matched-utility comparisons. Keep Firestore off for initial checks so cached
results cannot skip the requested computation. See the
[implementation report](queue_defense_rag_implementation.md) for privacy scope
and source citations.
