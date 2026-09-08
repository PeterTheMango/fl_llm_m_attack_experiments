# One larger research configuration for the 20GB server allocation

Use `master_script/configs/pipeline_research_master.yaml`. It defines **99
sequential experiments**: 11 attacks × three training conditions × three seeds.
There is one experiment YAML; the RAG dataset is a separate generated data file.
The preparation command downloads public text only and does not load a model.

| Setting | Value |
| --- | --- |
| Target and causal reference model | Qwen2.5-0.5B-Instruct, 0.49B parameters |
| Training conditions | None, sequence DP (C=1, σ=2), client DP-FedAvg (C=0.1, σ=2) |
| Seeds | 7, 107, 207 |
| Training data | SQuAD train; deterministic 4,096-record pool |
| Federation | Four clients; all participate; three rounds; one local epoch |
| Client data | 32 base QA records each; one member/replacement record on target client |
| Batch / sequence | Two records / 128 tokens |
| Membership trials | 200 per run; shared/LOSS: 100 positive and 100 negative |
| GPU allocation | `sim_num_gpus: 1.0`, so one client trains at a time |
| RAG corpora | 256 public + 256 simulated-private SQuAD validation paragraphs |
| Datastore probes | 200 candidates: 100 members and 100 nonmembers for each corpus |
| Utility | 20 questions; evidence documents separate from membership probes |
| Retrieval | Top 4; context budget 768 tokens; maximum answer 64 tokens |
| RAG model sampling | First two trials per run; once for AMIA's single trained model |

The user's approval allows models up to 7B; it does not imply that full-model
private training of a 7B model fits 20GB. The current implementation has no
LoRA, quantization, mixed-precision training, sharding, or optimizer offload.
The 0.49B model is a materially larger research baseline than tiny GPT-2 while
leaving room for private-gradient buffers and optimizer state. Fit is an
engineering estimate, not a measured guarantee for this GPU allocation.
The official [model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
provides the parameter count and native chat-template usage.

## Run on the remote server

Sync the implementation changes as well as the YAML: this file needs the new
pipeline-list loader, research dataset profile, chat-prompt option and bounded
RAG evaluation option. Copying just the YAML into the previous code is not enough.
From the repository root, in your existing `LLMPrivacy` environment:

```bash
python -m master_script.prepare_research_study
python -m master_script.perform_experiments --queue \
  master_script/configs/pipeline_research_master.yaml \
  --dry-run --no-firestore
```

The preparation command writes `configs/research_data/squad_rag_study.json`,
the original source JSON and a provenance sidecar with source/study SHA-256s.
It reuses a valid existing preparation and refuses to overwrite conflicting
files. An offline server can use `--source /actual/path/dev-v1.1.json` with the
official SQuAD validation JSON. Prepare once and reuse the same bytes for all
comparisons. Until preparation, a missing-study-file error is expected.

After the dry run reports 99 planned runs, run within your GPU allocation:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
python -m master_script.perform_experiments --queue \
  master_script/configs/pipeline_research_master.yaml \
  --no-firestore --no-charts
```

Keep the same GPU environment selection that worked for the remote verification.
For a MIG allocation, preserve its assigned UUID/device visibility; do not
replace it with a physical GPU index belonging to another allocation. The
runner's `EXPERIMENT_GPU` setting takes precedence over device visibility, so
it must identify your allocated device too. The config reserves one logical
GPU, which is the whole 20GB allocation, not a second client sharing it.

For a narrower first submission, add `--attack zlib`; that still runs nine
research experiments with 200 trials each. This is not a quick smoke test.
`--no-firestore` prevents cache skips and keeps results in a fresh queue directory.
If you use Firestore later, large result documents can exceed its document limit;
the queue retains full local JSON and records remote-write errors.

## Cost and interpretation

The full file requests **18,009 FL model trainings** (the ten per-trial attacks
train 18,000 models; AMIA trains nine models and separate probes), or **216,108
client fits** at three rounds and four clients per model. There is no model
sharing between attacks. Runtime is not benchmarked; this is a long research
batch, not an overnight promise. RAG is evaluated on 189 trained models rather
than all 18,009: 170,100 generated answers including no-context controls.
That sampling means RAG results describe the first paired worlds of each seed,
not every training-membership trial. Remaining trials retain training privacy
accounting and explicitly record `rag_evaluation_skipped`.

The seeds separate the 100 paired-world seed ranges, but samples still overlap
within the same training pool. A hundred nonmembers per run provides 1% FPR
resolution, not precise confidence at low FPR. AMIA keeps its existing one-model
probe protocol; its 200 trials are not 200 independent model trainings. Existing
attack thresholds and adaptations are retained (including SPV-MIA's fallback
paraphraser); the file does not claim exact paper reproduction or recalibration.
Use continuous scores and uncertainty estimates when comparing attacks, rather
than interpreting the printed `adv` alone.

DP noise/clip settings are declared research baselines, not calibrated utility
optima or a promised epsilon. With full participation and no amplification,
51 record-DP steps per model at σ=2 give a loose bound; releasing 200 models
composes their costs. A four-client DP-FedAvg population can lose substantial
utility under noise. The entire audit output, AMIA probe, private RAG answers,
and pretraining exposure remain outside the training privacy guarantee.

SQuAD train supplies the FL QA strings; validation supplies RAG paragraphs and
utility questions. The “private” datastore is simulated on public data and
not a bank/healthcare dataset. Articles may occur in both retrieval partitions;
pretraining and semantic overlap have not been ruled out. The preparer samples
complete short paragraphs and never trims away utility answers. It records
these selection limits, which must be accounted for in research claims.
Native chat formatting is used for RAG generation only; the original training
attacks retain their established scoring input format.

## Implementation and verification status

The loader now accepts a top-level list of pipeline mappings and expands it in
condition → attack → seed order. Existing single-mapping configs still work.
The new `squad_research` dataset key controls 32-record partitions and a larger
pool consistently in shared training, AMIA, LOSS and LOSS calibration. Existing
profiles retain their four-record partitions. Optional `prompt_format: chat`
and `evaluation_trials: 2` are part of new run identities; omitted defaults
preserve earlier pipeline identities.

Only syntax and static expansion were checked locally (99 unique run IDs using
the existing small fixture to stand in for the unprepared study path). No
training, downloads of research data/models, or pytest suite was run for this
change, following the user's request to keep verification on the server.
Regression cases were added for remote execution:

```bash
PYTHON_DOTENV_DISABLED=1 FIREBASE_SERVICE_ACCOUNT_JSON='{}' \
GOOGLE_APPLICATION_CREDENTIALS=/nonexistent/verification-credentials \
python -m pytest tests/test_research_config.py tests/test_pipeline.py \
  tests/test_datasets.py tests/test_yaml_config.py tests/test_hash_equivalence.py -q
```

Sources: [SQuAD, Rajpurkar et al. 2016](https://aclanthology.org/D16-1264/),
[official SQuAD distribution](https://rajpurkar.github.io/SQuAD-explorer/), and
the [reference log](../../written_paper/reference_papers/queue_defense_rag_sources.md).
Training-defense and retrieval-defense theory/limits remain documented in the
[implementation report](queue_defense_rag_implementation.md).
