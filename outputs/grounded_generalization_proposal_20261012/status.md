# Grounded FL-RAG generalization study: status

**2026-10-14.** The researcher approved every recommended setting of proposal draft 1 (§12): "Use all the recommended settings". The Stage 0/1 procedures are frozen in [`stage0_protocol.json`](stage0_protocol.json), and the **Stage 0 code is implemented and tested**. Nothing has run on the server yet.

- **Core fingerprint:** `a6cd6438f84b` (was `8b031bbfdcc1`). Three new core modules were added; no existing core file changed.
- **Tests:** 841 passed, 2 skipped (809 before; 32 new in `tests/test_grounded_generalization.py`).
- **Branch:** `claude/grounded-generalization-5cddaf`, cut from `feat/grounded-fl-stage0` at `d6a9cf0`.
- **The parent study is unaffected.** Its server worktree must stay on `feat/grounded-fl-stage0` (fingerprint `8b031bbfdcc1`), because its launches refuse any other fingerprint. Its tool, the frozen guard files and `causal_attack_validation.py` are byte-identical.

## Decisions (the researcher, 2026-10-14: every recommendation)

| # | Decision |
|---|---|
| 1 | LoRA for every new cell, plus the full fine-tuning replication cell A-full |
| 2 | Tier 1: A-full, A-LoRA, Q-1.5B, Q-3B, Llama-1B, Synth, Non-IID (about 190 GPU-h); Qwen 7B not run |
| 3 | 30 targets per cell; the same 30 fresh final SQuAD targets (eligible indices 60–89) in every SQuAD cell |
| 4 | Llama-3.2-1B-Instruct for the family question (gated). Gemma-2-2B and the Granite hybrid are tier 2 and are not run |
| 5 | Self-generated synthetic private records. DuoRC is tier 2 and is not run |
| 6 | Dirichlet α = 0.1 over T articles |
| 7 | The parent's frozen Reference choices everywhere, plus online LiRA (16 shared shadows per arm); offline LiRA and RMIA secondary |
| 8 | Positive controls in every cell with a new model, dataset or training method |
| 9 | In-process FedAvg for the LoRA cells, tested bit-identical to the Flower path on full parameters |
| 10 | Lean measurements: utility on W1 only, without no-context and public-library F1 |
| 11 | One Bonferroni family per research question |
| 12 | No causal attack (H2), no DP (H6), no RG-public (H5); P0 descriptive in each cell |

## What was built (Stage 0, proposal §8 item 1)

| Item | Where |
|---|---|
| Model profiles (pinned revisions), the tier 1 cells, the LoRA recipe | `core/generalization.py` (`MODELS`, `CELLS`, `LORA`) |
| Llama's date pinned in every chat-template call; a template check (identical renderings, assistant turn ends with EOS) | `generalization.PinnedTemplate`, `pinned`, `template_check` |
| Dirichlet-over-articles client partitions (W1 and W0 share base records; target or hold record joins client 0); a skew summary for `check` | `generalization.dirichlet_partitions`, `world_records`, `skew_summary` |
| Balanced shadow design (each target IN in exactly half the shadows) and shadow worlds (fill excludes every target passage) | `generalization.shadow_design`, `shadow_records`, `check_shadow_world` |
| Synthetic private records: fictitious people in fictitious clinics, answers as exact spans by construction, consumed by `grounded.build_study` unchanged | `generalization.synthetic_rows` |
| LoRA adapters (own implementation, no peft): frozen base, rank 16, α 32, every Linear/Conv1D except the LM head; reference = adapters off | `core/lora.py` (`attach`, `adapters_disabled`, adapter arrays) |
| In-process FedAvg with Flower's own aggregation (`FedAvg.aggregate_fit`), the same client schedule and per-client seeds | `lora.fedavg_rounds`, `lora.federated_fine_tune` |
| LoRA positive control (adapters trained on the passage text) | `lora.memorize` |
| The GPU job: paired worlds (lean), shadow worlds, the LoRA control, P0 (datastore and utility); weight scores with raw NLLs for the target and 32 U population records | `core/generalization_job.py` |
| The study tool: `build-synth`, `calibrate-gate`, `prepare {datastore\|control\|smoke}`, `resolve`, `run`, `job`, `check`, `analyze` | `tools/grounded_generalization_study.py` |
| LiRA (online, fixed variance; offline) and RMIA statistics | the tool (`lira_fit`, `lira_scores`, `lira_offline_scores`, `rmia_scores`, `phi`) |

**Tests (`tests/test_grounded_generalization.py`).**
- Pinned revisions; each cell changes exactly one factor against A-LoRA; the frozen protocol matches the code.
- Llama's date is pinned, and a template that renders differently is refused.
- Dirichlet worlds are deterministic and pass the parent's world checks. At α = 0.1, clients hold about 3.3 effective articles against about 12 near IID.
- Shadow designs are balanced, and shadow worlds hold exactly their IN targets.
- The full-size synthetic build meets the parent's layout (at least 90 V and 40 final targets, every passage within 24–160 words).
- LoRA starts as the base model, trains only adapters, switches off exactly, and repeats bit for bit.
- **In-process FedAvg gives bit-identical aggregated weights to `amia.federated_fine_tune`**, driven through Flower's serialization and strategy without Ray, on a tiny GPT-2 with full parameters.
- Weight scores reproduce the parent's `score_candidate_hf` and template ratios, with the adapters-off reference.
- LiRA separates IN from OUT where the uncalibrated score cannot; RMIA is checked.
- The tool's slices, seeds, prepare refusals, resolve, the control escalation and the smoke learning check behave as frozen.
- **`check` withholds AUC-, F1- and score-like values** smuggled into a result.

## Specifications fixed at implementation (tightening or detail; none loosens the proposal)

- **V slices** come after every slice the parent used:
  - **datastore:** the parent's first 60 V targets, P0 only, so the check is comparable with the parent's;
  - **then** control 12 (study token count ≥ 80, which leaves margin for other tokenizers' 64-token minimum), pilot 4 and smoke 1.
- **Seeds:**
  - V target seed = 8100 + slot (control 8100–8111, pilot 8112–8115, smoke 8116);
  - V shadows 8150 + shadow number;
  - final targets 10000 + index;
  - final shadows **10150** + shadow number. The proposal said 10100, which would collide with reserve target seeds.
- **Population records:** the first 32 U records are scored in both forms in every world and shadow. U is never trained in this study. RMIA uses them.
- **LiRA's statistic:** φ = −(mean NLL × scored tokens) in the frozen form.
- **Smoke:** one V target × {RG, CB-AO} lean paired jobs, plus one shadow world per arm containing that target.
- **Non-IID inherits A-LoRA's datastore check and control.** The tool refuses a datastore analysis from any other cell.
- **The learning check (LoRA cells):**
  - **Pass:** RG's mean client loss falls from round 1 to round 3, and RG's V RAG F1 is at least P0's V F1 + 0.05.
  - **First failure:** one rerun at lr 5e-4.
  - **Second failure:** the cell is dropped.

## Deviations and gaps

- **The proposal described the Dirichlet skew as "mostly one or two articles"; the measured figure is about 3.3 effective articles.** Corrected in proposal §4.3 and §12. The scheme itself is unchanged.
- **No separate conda environment is needed for tier 1.** LoRA is in the repository, and the Granite hybrid is not run.
  - Llama 3.2 needs transformers ≥ 4.45 on the server. Check the version before Stage 0.
- **Pilot and confirmation tooling are not built yet:** `prepare pilot`/`final`, the per-cell analysis, families and the ceiling rule. They follow the Stage 0 results, as in the parent study.
- **The structured related-work search** is still open (parent study).

## Server steps for Stage 0 (the researcher runs these, in tmux)

**Variables:**
- `E=/home/calc08/projects/LLMPrivacy/fl_llm_m_attack_experiments/outputs`
- `G=$E/grounded-fl-20260930` (the parent's build and gate)
- `N=$E/grounded-generalization-20261014`
- `T="python -m master_script.tools.grounded_generalization_study"`

**Order:**
1. **Before anything else:**
   - accept the Llama 3.2 licence on Hugging Face (`meta-llama/Llama-3.2-1B-Instruct`);
   - make a token available on the server (`huggingface-cli login`, or `HF_TOKEN` in the tmux environment);
   - check `python -c "import transformers; print(transformers.__version__)"`, which must be ≥ 4.45.
2. **Worktree.** After the branch is pushed, create a new worktree. Leave the parent's `fl_grounded_stage0` untouched.
   ```bash
   git -C /home/calc08/projects/LLMPrivacy/fl_llm_m_attack_experiments fetch origin
   git -C /home/calc08/projects/LLMPrivacy/fl_llm_m_attack_experiments worktree add ../fl_grounded_generalization origin/claude/grounded-generalization-5cddaf
   cd /home/calc08/projects/LLMPrivacy/fl_grounded_generalization
   python -c "from master_script.core.config import implementation_fingerprint as f; print(f())"
   ```
   The last command must print `a6cd6438f84b`.
3. **Synthetic build and gate (CPU).** Paste back both printed reports.
   ```bash
   $T build-synth "$N/synth-build"
   $T calibrate-gate "$N/synth-build/study.json" "$N/synth-gate.json"
   ```
4. **Datastore checks** for A-LoRA, Q-1.5B, Q-3B, Llama-1B and Synth. Synth uses `--study "$N/synth-build/study.json" --gate "$N/synth-gate.json"`. For a SQuAD cell, for example:
   ```bash
   $T prepare datastore "$N/datastore-Q-3B" --cell Q-3B --study "$G/build/study.json" --gate "$G/gate.json"
   $T resolve "$N/datastore-Q-3B/launch.json"
   $T run "$N/datastore-Q-3B/launch.json" --gpu 0
   $T analyze "$N/datastore-Q-3B" "$N/datastore-Q-3B-analysis.json"
   ```
5. **Smoke pairs** for all seven cells (4 jobs each). They give the timings and the LoRA learning check.
   - LoRA cells pass their own datastore analysis; Non-IID passes A-LoRA's; A-full passes none.
   - On `"status": "escalate"`, prepare the one rerun into a new folder with `--previous-analysis "$N/smoke-<cell>-analysis.json"`.
   ```bash
   $T prepare smoke "$N/smoke-Q-3B" --cell Q-3B --study "$G/build/study.json" --gate "$G/gate.json"
   $T resolve "$N/smoke-Q-3B/launch.json"
   $T run "$N/smoke-Q-3B/launch.json" --gpu 0 --max-jobs 4
   $T analyze "$N/smoke-Q-3B" "$N/smoke-Q-3B-analysis.json" --datastore-analysis "$N/datastore-Q-3B-analysis.json"
   ```
6. **Positive controls** for A-LoRA, Q-1.5B, Q-3B, Llama-1B and Synth (12 jobs each). Use `prepare control`, `resolve`, `run --max-jobs 12`, then `analyze` with that cell's `--datastore-analysis`.
   - On `"status": "rerun_at_12_steps"`, prepare the one rerun with `--previous-analysis`.
7. **`check <folder>`** can run at any time. It shows integrity, timings and structure only.

**Rules.** Never edit a launch: on any refusal, prepare a new folder and record it here. Paste back each `analyze` output and the smoke timings. The cost table (proposal §7) is re-done from them before the pilot and Stage 2 are prepared.
