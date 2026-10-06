# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Research code for privacy in federated-learning (FL) fine-tuned LLMs that answer with RAG. It covers membership-inference attacks, DP defenses and RAG audits. Everything runs through the `master_script` package. Experiments run on the researcher's GPU server; the Mac is used to write code and run CPU tests.

**The current focus is the grounded federated RAG study.** Its working title is "Does Retrieval Grounding Keep Private Facts Out of the Model?" (revision 4, approved with corrections for Stage 0/1 on 2026-09-30). These three files in `outputs/retrieval_grounded_fl_proposal_20260930/` are authoritative:

- `proposal.md` is the design: arms RG / CB-AO / RG-public / P0 / CB-LM, H1–H7, the four-cell design, §9 statistics. It is a frozen review record, so its "Nothing is implemented or run" line is historical.
- `stage01_protocol.json` freezes the Stage 0/1 procedures. The tool refuses to run unless its `status` starts with `frozen`. Stage 2 gets its own protocol after the pilot. Definitions may be tightened, never loosened.
- `status.md` is **the living record of the repo's state**: what is built, every server result, decisions, deviations and the next server steps.

## Keep the status documented (required on every change)

Every code or protocol change for this study must be recorded in the same commit:

1. **Update `status.md`.** Say what changed and why, give the new core fingerprint if `master_script/core/` changed, give the test count, and record any deviation from `proposal.md` / `stage01_protocol.json` under "Deviations and gaps".
2. **Update the "Current state" section below** so the next session starts from the right place.
3. **Ask the researcher first** about any change to a frozen definition, threshold or procedure. Record their decision in `status.md`.

## Current state (2026-10-05)

- **Branch.** Stage 0 code lives on `feat/grounded-fl-stage0`, which is stacked on `proposal/retrieval-grounded-fl`, which is stacked on `analysis/matched-reference-controls`. `main` does **not** contain the grounded code yet.
- **Core fingerprint.** `4a001c5eb678` (pilot); the 3- and 6-epoch controls ran on `714ec988b38b`.
- **Stage 0 complete on the server:**
  - build (study sha `d7eac161…`);
  - gate calibration (τ 0.45);
  - P0 datastore check passes for both scorers;
  - the 6-epoch positive control passes (10 of 12 learned, retention r 0.142 [0.066, 0.225]), so **H4 is primary**;
  - timings: a paired job takes about 1.2 h at ε = ∞ and 1.6 h at ε = 16 after the victim-only observation fix.
- **Stage 1 pilot done (2026-10-05).**
  - Attacker choices are fixed: RG uses record/template/F1, CB-AO template/template/F1, CB-LM record/record/F1.
  - The projections show the AUC endpoints (H1, H3, H6's leakage) far too imprecise at option (b): about 0.15–0.24 against a 0.05 margin. H2 is at the causal-AUC ceiling.
  - **Descriptive:** RG beats CB-AO by about 0.27 F1 at ε = ∞, and RG-public is about RG. At ε = 64 and 16, the DP models land at about P0's utility (DP erases the fine-tuning effect), so H6 compares near-pretrained models.
  - **Next:** the researcher reviews `stage2_design_memo.md` and decides the Stage 2 design. The memo recommends S2: 40 targets at ε = ∞, H6 at ε = 16 with 10 targets, no ε = 64 and no release-noise pass, about 56 GPU-h. Pilot release-noise causal AUC at ε = ∞: RG 0.744, CB-AO 0.621 (off the ceiling). **Decision C = C2** (2026-10-05): the release-noise pass becomes H2's primary ✱, so the noise pass is kept (D1). A (target count) and B (DP arms) are still open; speed-ups are under discussion. Nothing in Stage 2 is prepared yet, and margins and rules may only be tightened.
- **Decisions already made:**
  - accept GPU training non-determinism;
  - pool attacker choices over budgets;
  - max_length 512;
  - pilot budget about 37 GPU-h accepted.
- **Open questions:**
  - the Stage 2 design (target counts or restructuring);
  - the structured related-work search (Stage 0 item 1) is not done.

## Commands

The environment is `conda env create -f environment.yml` (env `peter_experiments_fl`). On a shared server env, use `pip install --user -r requirements.txt` instead, and drop the `torch` line if a CUDA torch is already installed. torch, transformers and `flwr[simulation]` are imported function-locally, so the toy path runs without them.

Run the tests on the Mac with the project env. The base anaconda `python` lacks torch, and the grounded tests need it. On 2026-10-04 the suite gave 781 passed and 2 skipped.

```bash
/opt/anaconda3/envs/peter_experiments_fl/bin/python -m pytest tests -q
```

```bash
/opt/anaconda3/envs/peter_experiments_fl/bin/python -m pytest tests/test_grounded_fl_study.py::test_name -q
```

The grounded study tool runs one subcommand per stage step:

```bash
python -m master_script.tools.grounded_fl_study --help
```

Its subcommands are `build`, `calibrate-gate`, `prepare {datastore|control|timing|pilot}`, `resolve`, `run --gpu N [--max-jobs N]`, `job`, `check` and `analyze`.

This is the toy smoke run of the older attack runner: no GPU, model download or credentials.

```bash
python -m master_script.perform_experiments --config master_script/configs/archive/smoke.yaml --no-firestore
```

Use `python -m master_script.webui.app` for the dashboard. `master_script/README.md` documents every runner flag, and `tests/test_docs.py` fails if a `--flag` of `perform_experiments` is missing from it.

## Architecture

**Scientific identity.**
- `core/config.implementation_fingerprint()` hashes every `.py` under `master_script/core/`, so any edit there changes the core fingerprint. The fingerprint is part of run IDs and of every frozen launch. A launch prepared on one fingerprint refuses to run on another.
- That is why runs in flight must stay on their fingerprint: the server uses a separate git worktree per study.
- Note fingerprint changes in `status.md`. Earlier results keep their old fingerprints.

**Two execution paths.**
1. **The legacy sweep runner:** `perform_experiments.py` → `core/yaml_config.py` → `core/runner.py`.
   - Order: cache check → FL fine-tune → attack → measure → persist → cleanup.
   - The 11 attacks in `core/attacks/` are registered in `core/registry.py`. Results persist to Firestore (`core/firestore.py`) under versioned `run_id`s.
   - The config dataclasses are byte-frozen, because their serialized form is the Firestore key.
2. **The pre-registered study tools** in `master_script/tools/`: `grounded_fl_study.py`, `causal_attack_validation.py`, `matched_reference_controls.py` and `collect_guard_traces.py`.
   - **`prepare`** writes a frozen launch: protocol, study data, tool hash, core fingerprint.
   - **`resolve`** runs every CPU check before any GPU work.
   - **`run`** runs one job per process and retires the weights after verification (`core/checkpoint_retention.py`).
   - **`check`** reports integrity and timing **only**, never an endpoint value.
   - **`analyze`** applies the fixed rules once.
   - Never edit a launch. On any refusal, prepare a new directory and record it.

**The grounded study.**
- `core/grounded.py` is CPU-only and deterministic: article-disjoint SQuAD splits (T, T-hold, U, L, N, P, V), probe eligibility, old-target exclusion by recomputed record hashes, RG / CB-AO / CB-LM encoders with answer-only labels, the verbatim-overlap gate, scorers and the ε record.
- `core/grounded_job.py` is the GPU job: paired worlds W1/W0, Reference, causal (both directions, plus release noise), the 3-query natural-question pass (F1 and the pinned NLI model), closed-book retention, utility, the positive control, RG-public and P0.
- The §9 statistics (paired target bootstrap, crossed article-cluster utility bootstrap, six-outcome classification, H1–H7, the cross-channel rule) live in the tool and are tested in `tests/test_grounded_fl_study.py`.

**Shared machinery.**
- `core/attacks/amia.py` holds FL fine-tuning (`federated_fine_tune(world=…)`) and attack trials. Flower FedAvg runs through `core/federation.py`.
- `core/attacks/causal_probe.py` is the causal gradient-alignment request. New runs use `causal_score: cosine`, `probe_epochs: 12`.
- `core/defenses.py` provides DP-SGD (`private_train`) and the conservative zCDP accountant (`privacy_bound`).
- `core/rag.py` handles prompts (`_prepare_prompt`), retrieval, refusals and RAG utility.

## Research rules that the code and history enforce

- **Byte-identical files.** Tests pin the frozen guard files (`guard_features.py`, `guard_detector.py`, `guard_runtime.py`, `guard_replay.py`) and `tools/causal_attack_validation.py` by sha256. Do not edit them.
- **Cohorts.**
  - The reserved-final cohort stays closed.
  - Earlier target cohorts (guard v3/v4, causal Stage A/B, matched controls) must stay excluded from new studies.
  - Pilot and V data are tuning data, never evidence.
- **Closed lines.** The guard/detector line is closed: the matched controls failed. Mirabel is a "retrieval-removal control", not a defense. Do not present attack effectiveness or detector blocking as privacy protection.
- **Server work.** The researcher runs every server command (LCALC08, in tmux, in a separate worktree, with outputs at absolute paths outside it) and pushes from the Mac. Give exact commands. `status.md` holds the current server steps.
