# Grounded federated RAG study: status

**2026-10-10.** **Stage 2 (confirmation) is complete and analysed once** (146 jobs on fresh final targets, core `8b031bbfdcc1`, frozen `stage2_protocol.json`). Results are below. Stage 0 and the Stage 1 pilot are complete; everything on V is tuning or validation data, never evidence.

- **Core fingerprint before the diagnostic:** `8b031bbfdcc1` (2026-10-06, the two observation speed-ups; Stage 2). The pilot ran on `4a001c5eb678`; the 3- and 6-epoch controls on `714ec988b38b`. Earlier results keep theirs.
- **Tests:** 822 passed, 2 skipped (821 before the replicate; 820 before the order-independent aggregation; 816 before the deterministic repeat; 812 before the world-order diagnostic; 809 before `describe`). `tests/test_charts.py::test_chart_filenames_are_timestamped_so_runs_do_not_clobber` fails now and then in a full run (two renders within one timestamp); it passes alone and is unrelated to this study.
- **Core fingerprint `2dc5fd6ca6e6` (2026-10-18):** in deterministic mode only, the server sums client updates in client-id order (`amia.aggregation_order`), for the repeat's rerun (below). Before it, `8a1aa7717e8c` (2026-10-18): opt-in deterministic client training and the `W1-S2-W1` order, for the deterministic repeat (below). Before it, `0832ce3dbe5e` (2026-10-16): an opt-in `diagnostic` job kind for the world-order diagnostic (below). The paired, control, public and P0 paths are unchanged. Stage 2 ran on `8b031bbfdcc1`; `describe` still reads it, because `frozen_results` checks each result against its own launch's fingerprint.
- **2026-10-12:** a read-only `describe` subcommand for the descriptive secondaries (tool only; below). **Run on the server 2026-10-16:** results below. A read-only check then confirmed a systematic W1–W0 world difference at ε = ∞, which bears on H1's validity (see "Deviations and gaps").
- **2026-10-12:** the structured related-work search is written ([`related_work.md`](related_work.md)); see "Deviations and gaps".

## Stage 2 confirmation results (2026-10-10)

**Provenance.** Launch `$G/final` (launch sha256 `c2a1b6f1e9b638cdf1cd42c40ccd369113f2429b52e58a81d72583141b3f3b91`), protocol sha256 `967ff26f…` (the committed `stage2_protocol.json`), study tool sha256 `3921ed6a…` (commit `748378d`), core fingerprint `8b031bbfdcc1`, study sha256 `d7eac161…`. 146 of 146 jobs complete and verified; `analyze` was run once and written to `$G/final-analysis.json` (sha256 `39bbd7803d59d9656e6ad5f05ccc3b7a1ce8a122da7433cdef048f963df7e8a6`). Accounted ε: ∞ and 16.0096 (matched in every pair). About 67 GPU-h (estimate 76): about 0.45 h per RG job and 0.38 h per CB-AO job at ε = ∞, about 0.87 h and 0.81 h at ε = 16.

**Primary results** (60 targets at ε = ∞, 10 at ε = 16; 99.3% decision intervals, seven primaries; margins 0.05).

| | Difference (oriented as the hypothesis) | Estimate | 99.3% interval | Outcome | Claimed |
|---|---|---:|---|---|---|
| **H1** weights, record membership | Reference AUC, CB-AO − RG | 0.280 | [0.176, 0.368] | meaningful improvement | **yes** |
| | utility, RG − CB-AO (needs lo > −0.05) | +0.292 F1 | [0.237, 0.350] | acceptable | |
| **H2** client updates (primary: release-noise pass, C2) | causal AUC, CB-AO − RG | −0.052 | [−0.110, 0.006] | inconclusive | reading: inconclusive |
| | secondary: no-noise pass | +0.002 | [−0.007, 0.017] | equivalent | (never decides H2) |
| **H3** datastore | NQ AUC, RG − CB-AO | 0.350 | [0.230, 0.461] | meaningful improvement | **yes** |
| **H4** passage-fact retention | R_RG | +0.006 F1 | [−0.022, 0.036] | equivalent | no |
| | D_H4 = R_RG − R_CB-AO | +0.001 F1 | [−0.046, 0.047] | equivalent | |
| **H5** is private training needed? | F1(RG-public) − F1(RG, ε = ∞), needs lo > −0.05 | −0.096 | [−0.227, 0.025] | not acceptable | no |
| | secondary vs RG at ε = 16 (95%) | +0.110 | [0.014, 0.209] | | |
| **H6** utility at ε = 16 | F1, RG − CB-AO | +0.001 | [−0.006, 0.012] | equivalent (not supported) | no |
| | Reference AUC, CB-AO − RG (needs lo > −0.05) | −0.10 | [−0.56, 0.33] | not shown no-worse | |
| **H7** is fine-tuning needed? | F1(P0) − F1(RG, ε = ∞) | −0.211 | [−0.274, −0.144] | RG is meaningfully better | |
| **Cross-channel** | H1 and (H2 "RG increases" or H3) | | | H1 and H3 | **yes** |

**Readings (plain terms; measurement results under these attacks, not privacy guarantees).**
- **H1, weights.** Under each arm's pre-registered Reference attack, grounded training (RG) reveals much less about which records were trained on than closed-book answer-only training (CB-AO), and answers about 0.29 F1 better.
- **H3, datastore.** RG makes library documents meaningfully *easier* to detect with natural questions: the RG model uses retrieved passages more, so whether a passage is in the library shows more in its answers.
- **Cross-channel claim holds (H1 and H3):** with grounded training, record-membership leakage moves out of the weights and into the datastore channel.
- **H2, client updates: inconclusive.** The point estimate says RG leaks somewhat more through noise-protected releases (the 95% interval [−0.094, −0.010] excludes 0, the 99.3% decision interval does not). With no noise both arms sit at the ceiling and are equivalent. The primary endpoint was chosen after the pilot (C2), which the protocol states.
- **H4, retention: equivalent, not claimed.** Neither the within-RG retention nor the difference against CB-AO exceeds the ±0.05 F1 margin. The probes are sensitive (the 6-epoch positive control detected r = 0.142), so this is evidence that grounded training does not store the passage's other facts in the weights by more than 0.05 F1 in this setting.
- **H5: not acceptable.** Public-only grounded training is about 0.10 F1 worse than private RG (lower end −0.23), so acceptability cannot be shown; private data helps answer quality here. (The pilot's −0.02 was tuning data.)
- **H6: not claimed.** At ε = 16 both arms land at the same utility, as the pilot predicted (DP-SGD at σ 5.45 erases fine-tuning); the Reference AUC interval at 10 targets is very wide.
- **H7: fine-tuning helps.** RG at ε = ∞ beats the pretrained model by about 0.21 F1.

**Caveats to carry into any write-up.**
- **H1 compares each arm's own frozen attack form:** RG's Reference attack uses the record form (a pilot tie at 0.611, broken to the standard form by rule), CB-AO's the template form. This is the pre-registered rule; both forms were recorded in every job, so the other forms can be reported as a labelled descriptive sensitivity.
- **H2's primary was chosen after seeing the pilot (C2).** It measures leakage through releases protected by σ_obs 1.0 noise, not raw updates.
- **ε covers only the accounted training releases** (§7), not the crafted-request releases, the RAG outputs or the library.
- **Narrow setting:** one model (Qwen2.5-0.5B-Instruct), SQuAD, one FL setting; the datastore attacks are adaptations validated for sensitivity only.
- **The other descriptive secondaries** are computed by the read-only `describe` (run 2026-10-16, below). Two of them qualify the primary readings: H1's reduction disappears when the attacker uses the template form against RG, and H3's difference comes mostly from CB-AO ignoring retrieved text (its RAG F1 is below P0's). A tool change makes `check` and `analyze` refuse this launch (tool hash), so descriptive extras read the results through `frozen_results` and never re-run `analyze`.

## Replicate: Stage 2 paired jobs vs no-causal pairs on V (2026-10-18; tool only, core unchanged at `2dc5fd6ca6e6`; run 2026-10-22: inconclusive)

**Why.** The deterministic repeat shows no effect of the between-world work under deterministic kernels, which is inconclusive for Stage 2's normal-mode training. The researcher's rule (2026-10-18) is paired jobs next. This test asks directly whether Stage 2's world difference reproduces on fresh V targets, and whether it needs the between-world work.

**Design (`prepare replicate [--targets N]`, default 60, range 20–60).** For each of the first N V datastore-slice targets, with one seed (8000 + index), in normal (non-deterministic) training:
- **A Stage 2 RG paired job at ε = ∞:** the existing `paired` kind, with every measurement. Its between-world work is every W1 measurement, then the causal passes in the frozen template direction with release noise. W0 gets the natural questions and the 16 N documents, as in Stage 2.
- **A no-causal pair:** the existing `diagnostic` kind, order `W1-W0`, not deterministic. Only W1's utility measurement runs between the two trainings.
- **`analyze`** (`grounded_replicate_analysis_v1`) reports, for each set:
  - W1 − W0 in RAG F1, with the count of targets where W1 is better;
  - W0 − W1 in last-round client loss, with the count where W0 is higher;
  - and "with minus without" for both, paired by target, with 95% target-bootstrap intervals.
  It decides nothing.
- **Test (1 new, 822 passed, 2 skipped):** `prepare` writes one paired job and one no-causal pair per target with the Stage 2 measurements and seeds; `analyze` recovers a built-in procedural effect (+0.06 with, 0 without).

**Readings fixed before the run (V tuning data; never evidence; 95% intervals).**
1. **Procedural:** the paired set's W1 − W0 RAG F1 is above 0, and "with minus without" is above 0. Then the between-world work of a paired job makes W0 worse, and Stage 2's world difference is a procedural artifact.
2. **Real (membership):** both sets' W1 − W0 are above 0, and "with minus without" includes 0. Then training on the target record makes the world better, whatever runs between.
3. **Not reproduced:** the paired set's W1 − W0 interval includes 0. Then Stage 2's world difference does not reproduce on V; it is most likely chance or specific to that cohort, and the write-up reports it as an unexplained, unreplicated difference.
4. **Otherwise:** inconclusive.
The loss differences are reported alongside, descriptively.

**Cost and precision.** About 0.45 h per paired job (Stage 2 at ε = ∞) and about 0.28 h per no-causal pair, so 60 targets is 120 jobs, about 44 GPU-h (40 targets: about 29 GPU-h). Using Stage 2's per-target spread (about 0.11 F1), 60 targets give about ±0.03 on each set's W1 − W0, which is enough to see a 0.057 gap, and about ±0.04 on "with minus without".

**Server steps** (the researcher runs these in tmux, in `/home/calc08/projects/LLMPrivacy/fl_grounded_stage0`, env `LLMPrivacy`, with `G` as above), after this commit is pushed:
1. `git fetch origin` and `git checkout --detach origin/feat/grounded-fl-stage0`; confirm the core fingerprint still prints `2dc5fd6ca6e6`.
2. `prepare replicate "$G/replicate-1" --study "$G/build/study.json" --gate "$G/gate.json"` (or `--targets 40`), then `resolve`.
3. `run --gpu 0 --max-jobs 120`, repeated until complete; `check` at any time.
4. `analyze "$G/replicate-1" "$G/replicate-1-analysis.json"`, and paste the output back.

**Result (server, 2026-10-22; launch `$G/replicate-1`, sha256 `6cee5816…`, tool `83e5559c…`, core `2dc5fd6ca6e6`; 60 V targets, 120 jobs; analysis `$G/replicate-1-analysis.json`).**

| Set (normal training) | W1 − W0 RAG F1 | W1 better | W0 − W1 last-round loss | W0 higher |
|---|---|---:|---|---:|
| With the between-world work (Stage 2 paired jobs) | **0.037 [0.001, 0.074]** | 33 of 60 | 0.047 [−0.011, 0.105] | 30 of 60 |
| Without it (no-causal `W1-W0` pairs) | 0.007 [−0.037, 0.050] | 33 of 60 | 0.031 [−0.035, 0.097] | 31 of 60 |
| With minus without (paired by target) | 0.031 [−0.025, 0.086] | | 0.016 [−0.046, 0.077] | |

**Against the fixed readings: inconclusive (reading 4).**
- Reading 1 (procedural) needs the paired set's W1 − W0 above 0 (it is, barely: lower end 0.001) **and** "with minus without" above 0. That interval includes 0. Not met.
- Reading 2 (real) needs both sets above 0. The no-causal set is not. Not met.
- Reading 3 (not reproduced) needs the paired set's interval to include 0. It does not, by 0.001. Not met.

**Descriptive reading (plain terms; tuning data).**
1. **On fresh V targets, Stage 2's world difference reappears in paired jobs, smaller:** W1 leads by 0.037 F1, against 0.057 in Stage 2. Without the causal passes and W1's other measurements between the trainings, it nearly vanishes (0.007).
2. **All the evidence points the same way, but no single test decides it.**
   - With the between-world work, W1 leads in Stage 2 (+0.057, 60 final targets) and in the replicate (+0.037, 60 V targets).
   - Without it, W1 does not lead: the replicate's no-causal pairs (+0.007), the 12-target diagnostic's W1 → W0 (−0.070) and W0 → W1 (−0.001).
   - The direct comparison, 0.031 [−0.025, 0.086], is about the gap expected if the work causes it, but it is too noisy to show.
3. **The lead comes from a minority of targets.** W1 is better in only 33 of 60 targets in both sets. The mean gap is driven by a few large differences, as expected when one trained world is a noisy draw.
4. **Under deterministic kernels the between-world work changes nothing** (the repeat). If the effect is real, it acts only through non-deterministic training; no mechanism is known.
5. **Several tests have been run on this question,** so a single nominal result should not be over-read.

**What this means for H1.** The question that matters for the paper is not the cause but whether the world difference moves the Reference scores that H1 compares. A read-only check of both launches answers that without GPU work:
- **If the world difference contaminates the Reference attack,** a target whose W1 answers better than its W0 (or whose W0 ends at a higher loss) should also have a larger s₁ − s₀.

**Proposed read-only check (server; it reads result files only).** Per launch, arm and Reference form at ε = ∞, it prints:
- the mean s₁ − s₀ and the share of targets with s₁ > s₀;
- the correlation of s₁ − s₀ with the W1 − W0 RAG F1 gap and with the W0 − W1 last-round loss gap, with 95% target-bootstrap intervals.

The replicate holds RG only. **Reading fixed before it runs:**
- In the final launch, a positive correlation with an interval above 0 for an arm's frozen form (RG record, CB-AO template) means that arm's H1 Reference AUC carries a world component.
- Intervals that include 0 mean there is no evidence that the world difference moves the Reference scores (it cannot be ruled out).

```bash
python - <<'PY'
import json, os, glob
import numpy as np
G = os.environ['G']
rng = np.random.default_rng(20260930)

def rows(launch):
    out = {}
    for f in sorted(glob.glob(f"{G}/{launch}/results/*/result.json")):
        r = json.load(open(f))
        if r['kind'] != 'paired' or str(r['epsilon_budget']) != 'inf':
            continue
        w1, w0 = r['worlds']['W1'], r['worlds']['W0']
        f1 = np.mean([q['f1'] for q in w1['utility']['F']]) - np.mean([q['f1'] for q in w0['utility']['F']])
        loss = r['federated_history_W0'][-1]['mean_client_loss'] - r['federated_history_W1'][-1]['mean_client_loss']
        for form in ('record', 'template'):
            out.setdefault((r['arm'], form), []).append((w1['reference'][form] - w0['reference'][form], f1, loss))
    return out

def corr(x, y):
    idx = rng.integers(0, len(x), (10000, len(x)))
    boot = [np.corrcoef(x[i], y[i])[0, 1] for i in idx]
    return round(float(np.corrcoef(x, y)[0, 1]), 3), [round(float(q), 3) for q in np.nanquantile(boot, [0.025, 0.975])]

for launch in ('final', 'replicate-1'):
    for key, values in sorted(rows(launch).items()):
        d, f1, loss = (np.array(v, float) for v in zip(*values))
        print(launch, key, 'targets', len(d), 'mean s1-s0', round(d.mean(), 3), 's1>s0 share', round(float((d > 0).mean()), 3),
              '| corr with W1-W0 F1', *corr(d, f1), '| corr with W0-W1 loss', *corr(d, loss))
PY
```

## Deterministic same-world repeat (2026-10-18; core `0832ce3dbe5e` → `8a1aa7717e8c` → `2dc5fd6ca6e6`; rerun 2026-10-18: control valid, no effect)

**Why.** The world-order diagnostic left one suspect for Stage 2's world difference: the work a paired job does between its two trainings (every W1 measurement, then the causal passes). One trained world varies by about ±0.1–0.17 F1 per target from GPU non-determinism, so a statistical test would need about 60 targets. **The researcher's decision (2026-10-18):** first a deterministic repeat, then paired jobs (about 60 V targets) if that is inconclusive.

**Design.** The same member world W1 is trained twice in one job, with deterministic client kernels, so two trainings of one world should give bit-identical weights. Two versions per target:
- **`W1-W1` (the determinism control):** nothing between the two trainings.
- **`W1-S2-W1`:** between the two trainings, exactly what a Stage 2 RG member world does:
  - every W1 measurement: Reference, retention, the natural-question cells "both" and "training-only" with the NLI model, and utility;
  - then the causal passes (frozen template direction, no-noise and release-noise passes) on the first W1 model.
  After that, W1 is trained again and its utility is measured.
- Each world's final weights get a digest (`grounded_job.parameters_sha256`), and the per-round client losses are kept.
- Default 4 V targets (`--targets`, 2–12), from the P0 datastore slice, with the same seeds as the diagnostic (8000 + index).

**What was built (opt-in; every existing path is unchanged unless the variable is set).**
- **`amia.deterministic_training()`** (shared machinery, opt-in). When the job process sets `GROUNDED_DETERMINISTIC_TRAINING=1`, each training client turns on:
  - strict `torch.use_deterministic_algorithms(True)`;
  - deterministic cuDNN;
  - the math attention kernel (flash and memory-efficient attention backward passes are not deterministic).
  It refuses unless `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set. Each client reports `deterministic: true` in its round outcome, so the analysis can confirm the clients really ran deterministically. Unset, nothing changes.
- **`grounded_job._diagnostic`:**
  - the new order `W1-S2-W1`, whose job carries a Stage 2 RG member world's measurements and the validated causal configuration;
  - the `deterministic` flag, which sets both variables before the first simulation;
  - the weight digest of every trained world.
- **`prepare repeat [--targets N]`** writes 2N deterministic diagnostic jobs. `analyze` adds, per order: deterministic jobs and clients, identical weights, identical per-round client losses, and per target the last-round loss difference.
- **Tests (4 new, 820 passed, 2 skipped):**
  - the deterministic switch is opt-in and strict (it refuses without the cuBLAS workspace);
  - the weight digest changes with any weight;
  - the between order runs exactly train → full W1 measurements → causal passes → train → utility, with both variables set before each training;
  - `prepare`/`resolve`/`run`/`analyze` of a repeat launch report exact matches for the control and a built-in drift for the between order.

**Readings fixed before the run (V tuning data; never evidence).**
1. **Validity:** the control (`W1-W1`) is valid only if every one of its jobs ran deterministic clients and gave identical weights for every target. If not, determinism was not achieved, the repeat cannot decide, and the next step is paired jobs.
2. **Not shown:** the control is valid and `W1-S2-W1` also gives identical weights for every target. Then the between-world work does not change the next training through deterministic kernels. In normal (non-deterministic) training it could still act through kernel choices, so this is inconclusive for Stage 2, and the next step is paired jobs.
3. **Shown:** the control is valid and `W1-S2-W1` gives different weights for any target. Then the between-world work changes the next training, a procedural cause of Stage 2's world difference. The direction and size (loss and F1, second − first) are reported descriptively.

**Result (server, 2026-10-18; launch `$G/repeat-1`, sha256 `edfdbac2…`, tool `d5018040…`, core `8a1aa7717e8c`; 4 V targets; analysis `$G/repeat-1-analysis.json`).**

| Order | Deterministic jobs / clients | Identical weights | Round-by-round mean client loss, run 1 → run 2 | Last-round loss, second − first | RAG F1, second − first |
|---|---|---|---|---|---|
| `W1-W1` (control) | 4 / yes | **0 of 4** | 1.4594 → 1.4594; 1.014 → 1.045; 0.722 → 0.712 | −0.010 [−0.032, 0.022] (higher in 1 of 4) | +0.022 [0.001, 0.059] |
| `W1-S2-W1` | 4 / yes | 0 of 4 | 1.4594 → 1.4594; 1.048 → 1.045; 0.609 → 0.637 | +0.028 [−0.006, 0.077] (higher in 3 of 4) | +0.007 [−0.006, 0.020] |

**Against the fixed readings: reading 1, the control is not valid, so the repeat cannot decide.** Every client confirmed deterministic kernels, yet no control target reproduced its weights.

**Why (found 2026-10-18).**
- The round-1 mean client loss is identical to every printed digit in every run (1.459396449769188), so client training from the same start is deterministic.
- The runs part from round 2 on, so the difference enters between rounds. Flower's FedAvg (`aggregate_fit`, used unchanged by `SaveModelFedAvg`) sums the client updates in the order they arrive. The order depends on which client finishes first, and floating-point sums depend on order, so the aggregated model differs in its last bits; chaotic training then amplifies that.
- This is a source of Stage 2's accepted training non-determinism too. Nothing makes it directional by itself.
- **A flaw in the new metric:** "identical client losses" compares each round's client losses in arrival order, so it can report a difference where only the order differs. It needs to compare by client id.

**Descriptive (4 targets, tuning data).** The between order's second run ends at a higher loss in 3 of 4 targets (+0.028), while the control's does in 1 of 4 (−0.010). That is Stage 2's direction, but 4 targets with an invalid control show nothing.

**Next (the researcher's decision).** The fixed plan says paired jobs on about 60 V targets (about 27 GPU-h plus a matching no-causal set). The alternative is to make the aggregation order-independent in deterministic mode (sum the client updates in client-id order; opt-in, a small core change), fix the loss comparison, and rerun the repeat (about 3–4 GPU-h). That could give the exact answer the repeat was designed for.

**Researcher's decision (2026-10-18): fix the order and rerun the repeat** (instead of moving to paired jobs).
- **`amia.aggregation_order(results)`.** In deterministic mode only (`GROUNDED_DETERMINISTIC_TRAINING=1`, which the job process sets, so the in-process server sees it), the server sorts client updates by client id before FedAvg sums them. Unset, the arrival order is kept, so every other path is unchanged.
- **The analysis's "identical client losses"** now compares each round's losses by client id, not by arrival order.
- **Test:** four float32 updates averaged by Flower's `aggregate` in three arrival orders give different bits, and after `aggregation_order` they give one identical result. The repeat test now permutes arrival order and still counts identical losses. 821 passed, 2 skipped.
- **Core fingerprint** `8a1aa7717e8c` → `2dc5fd6ca6e6`.
- **The rerun** is a new launch, `$G/repeat-2`, with the same 4 targets and seeds and the same readings as above. The control must give identical weights for every target, or the repeat again cannot decide.

**Server steps for the rerun:**
1. `git fetch origin` and `git checkout --detach origin/feat/grounded-fl-stage0`; confirm the core fingerprint prints `2dc5fd6ca6e6`.
2. `prepare repeat "$G/repeat-2" --study "$G/build/study.json" --gate "$G/gate.json"`, then `resolve`.
3. `run --gpu 0 --max-jobs 8`, `check`, then `analyze "$G/repeat-2" "$G/repeat-2-analysis.json"`.

**Result of the rerun (server, 2026-10-18; launch `$G/repeat-2`, sha256 `a145e1f3…`, tool `f1647d9f…`, core `2dc5fd6ca6e6`; 4 V targets; analysis `$G/repeat-2-analysis.json`).**

| Order | Deterministic jobs / clients | Identical weights | Identical client losses (by id) | RAG F1 run 1 / run 2 | Round losses run 1 = run 2 |
|---|---|---|---|---|---|
| `W1-W1` (control) | 4 / yes | **4 of 4** | 4 of 4 | 0.31518630308161333 / same | 1.4594, 1.0154, 0.6953 (identical) |
| `W1-S2-W1` | 4 / yes | **4 of 4** | 4 of 4 | 0.31518630308161333 / same | identical |

**Against the fixed readings: reading 2.** The control is valid: every target reproduced its weights, its per-round losses and its answers exactly. The between order also reproduced every target exactly, and its first W1 is bit-identical to the control's.

**So, under deterministic kernels, a Stage 2 member world's work between two trainings (every W1 measurement and the causal passes, in the same process) does not change the next training at all.** That rules out a direct carry-over of process state, for example a changed global setting, a mutated object or reused data.

It does not settle Stage 2, which ran in normal mode. There the between-world work could still act through non-deterministic kernel choices, although no mechanism is known. A by-product: deterministic mode with ordered aggregation makes a whole paired training reproducible bit for bit.

**What is now ruled out, and what remains.**
- Ruled out:
  - the target record making W1 better (world-order diagnostic, 12 targets; W1 led in neither mixed order);
  - the causal attack altering W0's training data (the encoded records are separate per world and immutable);
  - the between-world work changing the next training through process state (this repeat).
- Remaining:
  - an effect of the between-world work that exists only in non-deterministic training;
  - a real effect of the target record too small for 12 targets;
  - chance in Stage 2 (unlikely: 60 targets, both arms' losses agree).
- **Next, per the researcher's rule (2026-10-18: paired jobs if the repeat is inconclusive):** real paired jobs on V in normal mode, against matching no-causal pairs on the same targets (below).

**Cost.** About 17 min per control job and 25 min per between job at normal speed (from `timing-5`). Deterministic kernels and math attention may slow training, perhaps up to 2×. At 4 targets, 8 jobs take about 3–4 GPU-h. A deterministic-kernel refusal (an operation with no deterministic version) would stop the first job within minutes, with the error in its log.

**Server steps** (the researcher runs these in tmux, in `/home/calc08/projects/LLMPrivacy/fl_grounded_stage0`, env `LLMPrivacy`, with `G` as above), after this commit is pushed:
1. `git fetch origin` and `git checkout --detach origin/feat/grounded-fl-stage0`; confirm the core fingerprint prints `8a1aa7717e8c`.
2. `prepare repeat "$G/repeat-1" --study "$G/build/study.json" --gate "$G/gate.json"`, then `resolve`.
3. `run --gpu 0 --max-jobs 8`, then `check` and `analyze "$G/repeat-1" "$G/repeat-1-analysis.json"`; paste the output back.

## World-order diagnostic (2026-10-16; core `8b031bbfdcc1` → `0832ce3dbe5e`; nothing run yet)

**Why.** The read-only check confirmed a systematic difference between the paired worlds at ε = ∞ (below, "Result of the W1–W0 check"): W0 ends training at a higher loss in both arms, and RG's W1 answers 0.057 F1 better, although the worlds differ in one record. No code cause was found. The leading suspect is the order of work in a paired job: W1 trains first in a fresh process; W0 trains after the W1 measurements and the causal passes in the same process. The researcher chose to diagnose it before the write-up (2026-10-16).

**What was built (opt-in; every existing job path is unchanged).**
- **`grounded_job._diagnostic`,** a new job kind `diagnostic` (V only, utility only, RG or CB-AO, no DP).
  - It trains two worlds one after the other in one process, in a declared order, and measures each world's utility (F, F_P, no-context F1, NLL) right after its training.
  - As in a paired job, the pretrained reference model is loaded after the first training and parked on the CPU during the second.
  - Unlike a paired job, no causal pass and no other measurement runs between the two trainings.
  - Each world's per-round training losses are kept (`federated_history_run1/run2`).
- **Three orders:** `W1-W1` (the same world twice: any gap is pure order), `W1-W0` (Stage 2's order without the causal passes) and `W0-W1` (reversed).
- **`prepare diagnostic [--targets N]`** (default 12, range 6–60).
  - It uses the first N targets of V's P0 datastore slice, which no paired job has trained on, in every order (3N jobs).
  - Each target keeps the same seed (8000 + index) in every order, so the first-trained W1 of `W1-W1` and of `W1-W0` is the same setup.
  - `--targets` is refused for every other launch kind.
- **`analyze` on a diagnostic launch** reports, per order:
  - RAG F1 and public-library F1 of each run, second − first with a 95% target-bootstrap interval, and W1 − W0 where the order has both worlds;
  - each round's mean client loss per run, and the last-round loss difference (second − first) with its interval.
  It decides nothing. `check` shows timings only.
- **Tests (4 new):**
  - the job rules: V only, utility only, a declared order, no causal attack;
  - the runner trains in the declared order, loads the reference once after the first training, parks it before the second, and never calls a causal pass;
  - `prepare`/`resolve`/`run`/`check`/`analyze` on synthetic results recover a built-in pure order effect (W1 looks better first and worse second) and a built-in membership effect (W1 better in both orders).

**Readings fixed before the run (tuning data on V; never evidence; no margins, 95% intervals).**
1. **Order effect:** in `W1-W1`, the second training ends at a higher last-round loss (interval above 0) or answers worse (RAG F1 interval below 0). Then the second training in a job process trains worse, and Stage 2's W1–W0 difference is at least partly an artifact of training W0 second. `W0-W1` should then show W1 − W0 at or below 0.
2. **Membership effect:** `W1-W1` shows no difference (intervals include 0) and both `W1-W0` and `W0-W1` show W1 better. Then the target record itself makes W1 train better, a real data effect that would need its own explanation.
3. **Causal passes involved:** `W1-W1` and `W1-W0` show no difference here, although Stage 2 did. Then the causal passes between the worlds are the likely cause; the next step would add the causal passes to the diagnostic.
4. **Otherwise:** inconclusive at this size.

**Cost.** About 17–18 min per job (2 × about 300 s training, from `timing-5`, plus utility after each training, an estimate). At 12 targets, 36 jobs ≈ 10–11 GPU-h. Precision: in Stage 2 the per-target W1–W0 F1 difference has a standard deviation of about 0.11, so 12 targets give a 95% half-width of about ±0.06 F1, borderline for a 0.057 gap. The loss difference is probably steadier, but that is not known. 20 targets (60 jobs, about 18 GPU-h) would give about ±0.05.

**Result (server, 2026-10-18; launch `$G/diagnostic-1`, sha256 `40d7f93c…`, tool `258ba17b…`, core `0832ce3dbe5e`; analysis `$G/diagnostic-1-analysis.json`).** It ran at the default 12 targets (36 jobs): `--targets 20` was not passed to `prepare`, because the first command list given to the researcher omitted it. V tuning data; 95% target-bootstrap intervals.

| Order (run 1 → run 2) | RAG F1 run 1 / run 2 | Second − first, RAG F1 | W1 − W0, RAG F1 | Round-3 client loss run 1 / run 2 | Second − first, round-3 loss |
|---|---|---|---|---|---|
| W1 → W1 | 0.363 / 0.322 | −0.041 [−0.135, 0.056] (second better in 5 of 12) | n/a | 0.542 / 0.631 | +0.089 [−0.057, 0.234] (higher in 5) |
| W1 → W0 (no causal pass) | 0.332 / 0.402 | +0.070 [0.025, 0.119] (9 of 12) | **−0.070 [−0.119, −0.025]** | 0.606 / 0.511 | −0.095 [−0.220, 0.036] (higher in 3) |
| W0 → W1 | 0.334 / 0.333 | −0.001 [−0.087, 0.090] (5 of 12) | −0.001 [−0.087, 0.090] | 0.593 / 0.613 | +0.020 [−0.097, 0.132] (higher in 8) |

Public-library F1 moves the same way: W1 → W1 −0.029 [−0.118, 0.059]; W1 → W0 +0.058 [0.008, 0.111]; W0 → W1 +0.003 [−0.076, 0.086].

**Against the fixed readings: inconclusive (reading 4).**
- Reading 1 (order effect) needs W1 → W1's second run to be worse with an interval excluding 0. The point estimates lean that way (−0.041 F1, +0.089 loss), but both intervals include 0. Not met.
- Reading 2 (membership effect) needs W1 better in both mixed orders. W1 is worse in W1 → W0 (−0.070) and level in W0 → W1 (−0.001). Not met, and the data point against it.
- Reading 3 (causal passes involved) needs no difference in W1 → W1 and W1 → W0. W1 → W1 shows none, but W1 → W0 shows a nominal difference **in the opposite direction to Stage 2** (W0 better). Not met as written.

**Descriptive reading (plain terms; tuning data).**
1. **Without the causal passes between the two trainings, the Stage 2 pattern does not appear.**
   - In Stage 2, W1 beat W0 by 0.057 [0.031, 0.086] (60 targets), and W0 ended at a higher loss in both arms.
   - Here no order shows W1 ahead. Across the three orders, "second minus first" has no consistent sign: F1 −0.041, +0.070 and −0.001; loss +0.089, −0.095 and +0.020.
   - The one nominal result (W1 → W0, W0 better) is one of six comparisons at 95% and runs against Stage 2's direction, so it is probably chance.
2. **The target record does not make W1 train better.** If it did, W1 would lead in both mixed orders; it leads in neither.
3. **Training-to-training noise is large.** The first-trained W1 is the same setup (same target, seed and world) in W1 → W1 and W1 → W0, yet its 12-target mean RAG F1 is 0.363 in one and 0.332 in the other. One trained world is a noisy draw, about ±0.1 F1 per target, which is why 12 targets cannot resolve a 0.06 gap.
4. **What remains.** Stage 2's world difference is most plausibly tied to the work a paired job does between the two trainings: the causal passes and the W1 measurements (Reference scoring, retention, natural questions with the NLI model). Chance is not ruled out, but it is unlikely: Stage 2's interval excludes 0 with 60 targets, and both arms' losses agree. **This is not shown**: the diagnostic left out exactly those steps, so it can only say they are the remaining suspects.

**Next step (the researcher's decision).** A direct test would run real Stage 2 paired jobs (RG, ε = ∞, frozen template direction with release noise, every measurement) on the same 12 V targets and seeds. Their W1 − W0 can then be compared with the no-causal W1 → W0 above. This uses only the existing paired job kind, so it is a tool change only, with no core change; about 12 × 0.45 h ≈ 5.5 GPU-h. A separate question is how much a world difference moves the Reference AUC. That needs a control record that is in neither world, scored in both. The Stage 2 weights are retired, so it would be a new measurement (core change) on new runs.

**What it means for Stage 2.** Nothing is re-run or re-analysed. If reading 1 holds, the paper must state that H1's and H6's Reference AUCs include a world-order effect of unknown sign for the difference, and future paired jobs should train each world in its own process.

**Server steps** (the researcher runs these in tmux, in `/home/calc08/projects/LLMPrivacy/fl_grounded_stage0`, env `LLMPrivacy`, with `G` as above), after this commit is pushed:
1. `git fetch origin` and `git checkout --detach origin/feat/grounded-fl-stage0`; confirm the core fingerprint prints `0832ce3dbe5e`.
2. `prepare diagnostic "$G/diagnostic-1" --study "$G/build/study.json" --gate "$G/gate.json"` (add `--targets 20` for the larger option), then `resolve`.
3. `run --gpu 0 --max-jobs 36` (or 60), repeated until complete; `check` at any time (timings only).
4. `analyze "$G/diagnostic-1" "$G/diagnostic-1-analysis.json"`, and paste the output back.

## Descriptive secondaries: the `describe` tool (2026-10-12; tool only, core unchanged at `8b031bbfdcc1`)

**Why.** The confirmation `analyze` computed only H1–H7, the cross-channel rule, H2's no-noise secondary and H5's ε = 16 secondary. The other descriptive secondaries of proposal §9.2 were never computed (see "Deviations and gaps"). They need no GPU: every result JSON already keeps both Reference forms, both NQ scorers, refusal flags, gate records, F_P and no-context utility, answer NLL, the never-document scores and the per-pass causal AUCs.

**What `describe` does.** `python -m master_script.tools.grounded_fl_study describe ROOT OUTPUT [--final-analysis FILE]`.
- **Read-only.** It reads the finished final launch through `frozen_results` and never calls `completed()`, `check` or `analyze`. It checks each result against the launch's own completion record, its declared job file and the frozen cohort, and checks the launch's study and protocol copies against `launch.json`. It never checks today's tool hash, so it still reads the launch after the tool changed. Nothing is re-run.
- **Never decides anything.** It classifies no hypothesis and writes no outcome, reading or "claimed" field. Every section carries the label "descriptive secondary; does not change any primary outcome". Intervals are 95% only.
- **Writes a new file.** It refuses to overwrite and refuses to write inside the launch directory.
- **The same seeded draws as the confirmation analysis:** `default_rng(20260930)`, target draws at ε = ∞ and then at ε = 16, F article clusters, RG-public seeds; then F_P article clusters. Subsets (refusal-excluded, decided verbatim answers) draw their own from the same seed.
- **A built-in check.** With `--final-analysis`, it records that its recomputed primary values (H1 D and utility, H2 D and the no-noise D, H3 D, H4 R_RG and D_H4, H6 utility and Reference AUC, H7 D) equal the confirmation analysis's estimates and 95% intervals (`consistency_with_final_analysis`). It only reproduces them and never re-classifies them. A `false` would mean a reading bug, not a new result.

**What it reports, per budget (ε = ∞ with 60 targets, ε = 16 with 10).**

| Quantity | Construction |
|---|---|
| H1 sensitivity | Reference AUC of each arm in both forms (record, template), and CB-AO − RG for all four pairings of forms. The frozen pairing (CB-AO template, RG record) is marked; RG's record form was a pilot tie (0.611 vs 0.611) broken to the standard form |
| Causal AUCs | Mean per-target AUC per arm and pass (no noise, release noise) in the frozen (template) direction, and CB-AO − RG, at both budgets; plus the release-noise pass's own per-release privacy record |
| NQ cells, both scorers | Mean document score per cell, and pooled AUCs (with RG − CB-AO) for library-only vs neither (H3's contrast), training-only vs neither (the §6.4 secondary), both vs neither, both vs library-only, and both vs training-only |
| Entailment scorer | The same NQ table with the entailment scorer (F1 is the frozen scorer) |
| Refusal-excluded H3 | Library-only vs neither with refused probes dropped from each document score; targets kept only when both cells keep a scored probe (count reported) |
| Standard AUC against N | Each target's inserted passage (W0, library on) against the 16 never-inserted N documents scored by the same target's W0 model; pooled over targets, and a drawn target brings its positive and its 16 negatives |
| Verbatim baseline | Anderson-style yes/no query: "yes" rate per cell, and the library-only vs neither AUC over targets whose two answers were both decided (count reported) |
| Refusal rates | Per arm and cell: NQ probes in each cell, verbatim queries, never documents, retention probes (W1, W0), utility on F and F_P (W1, W0) |
| Gate | Share of queries whose top document the τ 0.45 gate would withhold: NQ probes per cell, verbatim queries per cell, never documents, utility F (W1, W0), P0, RG-public. It depends only on the query and the library, never on the model |
| Secondary utility | Per arm and world: RAG F1 and EM on F, no-context F1 and EM, answer NLL, public-library F1 and EM on F_P (crossed models × article clusters), with RG − CB-AO; the same for P0 and RG-public |
| Retention | R per arm (R_CB-AO is the §6.4 secondary) and RG − CB-AO |
| P0 against every arm | RAG F1, P0 minus each arm and budget and minus RG-public (oriented as H7) |

**Not covered:** the ε = 64 frontier (not run, decision B2), CB-LM (not run in Stage 2) and per-client results (the result files keep none).

**Tests (3 new; 812 passed, 2 skipped).** On a synthetic final launch run through `prepare`, `resolve`, `run` and `analyze`, with known refusals, gate decisions and an undecided verbatim answer:
- after a tool change, `check` and `analyze` refuse the launch and `describe` still reads it, leaving every launch file byte-identical;
- every recomputed primary value equals the confirmation analysis exactly, and no classification, reading or "claimed" field appears;
- the injected refusal rates, gate rates and decided-answer counts come back as built;
- it refuses an existing output, an output inside the launch, a non-final launch, another launch's analysis and an incomplete launch.

**Tool hash changed.** `check` and `analyze` will now refuse `$G/final` once the server pulls this commit. This is expected and harmless: the launch's files are untouched, and `final-analysis.json` stays the one confirmation analysis.

**Server steps** (the researcher runs these in tmux, in `/home/calc08/projects/LLMPrivacy/fl_grounded_stage0`, env `LLMPrivacy`, with `G` as above). CPU only, a few minutes.
1. `git pull --ff-only` on `feat/grounded-fl-stage0`; confirm the core fingerprint still prints `8b031bbfdcc1` and `final-analysis.json` still has sha256 `39bbd780…`.
2. `describe "$G/final" "$G/final-descriptives.json" --final-analysis "$G/final-analysis.json" | tee "$G/final-descriptives.txt"`.
3. Paste the printed lines back; they are recorded here as descriptive results.

**Results (server, 2026-10-16).** Written to `$G/final-descriptives.json` (printed copy `$G/final-descriptives.txt`) from launch sha256 `c2a1b6f1…`, `final-analysis.json` sha256 `39bbd780…`, launch tool `3921ed6a…`, describe tool `9f93e903…`, core `8b031bbfdcc1`, study `d7eac161…`. **All ten `consistency_with_final_analysis` checks are `True`:** `describe` reproduced every primary estimate and 95% interval exactly. Everything below is a descriptive secondary with 95% target-bootstrap intervals (utility: crossed with article clusters). None of it changes a primary outcome. ε = ∞: 60 targets; ε = 16: 10 targets.

*H1 sensitivity: Reference AUC in both attacker forms (ε = ∞).*

| | Record form | Template form |
|---|---|---|
| RG | 0.659 [0.603, 0.722] (frozen) | **0.917 [0.873, 0.959]** |
| CB-AO | 0.878 [0.818, 0.932] | 0.939 [0.894, 0.977] (frozen) |

| CB-AO − RG, by pairing of forms | ε = ∞ | ε = 16 |
|---|---|---|
| CB-AO template − RG record (**frozen; H1 / H6**) | 0.280 [0.205, 0.346] | −0.100 [−0.430, 0.220] |
| CB-AO record − RG record | 0.219 [0.143, 0.289] | 0.160 [−0.270, 0.590] |
| CB-AO template − RG template | 0.022 [−0.041, 0.081] | −0.190 [−0.400, −0.030] |
| CB-AO record − RG template | −0.039 [−0.117, 0.034] | 0.070 [−0.300, 0.420] |

*Datastore (W0, library on vs off; ε = ∞).*

| | RG | CB-AO | RG − CB-AO |
|---|---|---|---|
| NQ AUC, F1 scorer (H3's contrast) | 0.847 [0.782, 0.908] | 0.497 [0.443, 0.554] | 0.350 [0.263, 0.432] |
| NQ AUC, entailment scorer | 0.829 [0.767, 0.889] | 0.606 [0.547, 0.670] | 0.222 [0.146, 0.302] |
| Refusal-excluded (F1; all 60 targets kept) | 0.847 | 0.497 | 0.350 (identical: no refusals) |
| Inserted passage vs the 16 N documents (F1) | 0.839 [0.771, 0.902] | 0.434 [0.385, 0.488] | 0.405 [0.319, 0.488] |
| Inserted passage vs the 16 N documents (entailment) | 0.837 [0.780, 0.889] | 0.609 [0.539, 0.680] | 0.228 [0.169, 0.289] |
| Mean F1 document score, library on / off | 0.369 / 0.022 | 0.023 / 0.019 | |
| Verbatim yes/no baseline AUC (decided answers only) | 0.500 [0.467, 0.533] (45 targets) | 0.528 [0.458, 0.597] (36 targets) | |
| Verbatim "yes" rate, library on / off | 0.91 / 0.92 | 0.76 / 0.71 | |

*The four cells (natural-question AUCs, ε = ∞; F1 scorer, entailment in brackets).*

| Contrast | RG | CB-AO |
|---|---|---|
| training-only vs neither (§6.4 secondary) | 0.516 (0.513) | 0.493 (0.562) |
| both vs neither | 0.920 (0.828) | 0.584 (0.664) |
| both vs library-only (does training add to the library?) | 0.524 (0.507) | 0.581 (0.551) |
| both vs training-only (does the library add to training?) | 0.923 (0.835) | 0.588 (0.606) |

*Client updates (causal attack, frozen template direction, mean per-target AUC).*

| | ε = ∞ RG / CB-AO | CB-AO − RG | ε = 16 RG / CB-AO | CB-AO − RG |
|---|---|---|---|---|
| No noise | 0.993 / 0.995 | 0.002 [−0.006, 0.012] | 0.965 / 0.957 | −0.008 [−0.049, 0.027] |
| Release noise (σ_obs 1.0) | 0.703 / 0.651 | −0.052 [−0.094, −0.010] | 0.584 / 0.657 | **+0.073 [0.016, 0.133]** |

The release-noise pass accounts its own ε = 140.7 (δ 10⁻⁵, zCDP, 40 releases, per private batch), never merged with the training ε.

*Utility per arm (ε = ∞, member world W1; W0 in brackets).*

| | RG | CB-AO | RG-public (5 seeds) | P0 |
|---|---|---|---|---|
| RAG F1 on F | 0.378 (0.320) | 0.085 (0.082) | 0.281 | 0.167 |
| RAG EM on F | 0.253 (0.212) | 0.029 (0.028) | 0.175 | 0.013 |
| Public-library F1 on F_P | 0.404 (0.348) | 0.082 (0.080) | 0.315 | 0.192 |
| No-context F1 | 0.051 (0.046) | 0.076 (0.076) | 0.041 | 0.037 |
| Answer NLL (lower is better) | 3.77 (3.99) | 5.17 (5.28) | 4.29 | 5.64 |
| P0 − arm, RAG F1 | −0.211 [−0.258, −0.162] | **+0.081 [0.053, 0.113]** | −0.115 [−0.212, −0.018] | |

At ε = 16 every arm sits at P0's level: RAG F1 0.171 (RG) and 0.170 (CB-AO), F_P 0.197 / 0.196, P0 − arm −0.004 and −0.003.

*Other secondaries.*
- **R_CB-AO** (closed-book retention): 0.005 [−0.018, 0.030] at ε = ∞; 0.002 [−0.002, 0.008] at ε = 16.
- **Refusals:** none in any natural-question, verbatim, retention or ε = ∞ utility query. At ε = 16, 0.3–0.4% of F answers are refusals; P0 0.3%.
- **The gate** (recorded only; it withheld nothing in the jobs). It would withhold the top document for every verbatim query whose passage is in the library, for 11.1% of natural-question probes in the library-on cells (3.3% for the 10 ε = 16 targets), for none of the library-off probes, and for 10.0% of the F utility questions (every model; the gate depends only on the query and library). This is the share of F questions whose top document would be withheld, not the gold-document loss calibrated on V (4.4%), so the two numbers measure different things.
- **ε = 16 datastore and verbatim:** NQ AUC (F1) RG 0.860, CB-AO 0.880, difference −0.020 [−0.060, 0.000]; verbatim AUC 0.70 / 0.80 (10 targets).

**Readings (plain terms; descriptive only, they do not change any primary outcome).**
1. **H1 depends on the attacker's form for RG.** Under the frozen forms, RG leaks much less (0.28). But RG's frozen record form was chosen by a pilot tie-break (0.611 vs 0.611), and on the final data the template form is much stronger against RG (0.917 vs 0.659). If the attacker uses the template form for both arms, the difference is 0.022 [−0.041, 0.081], which is not a reduction. With the record form for both, RG still leaks less (0.219 [0.143, 0.289]). So grounded training lowers weight leakage for an attacker who scores the plain question–answer record, but not for one who scores the answer given the passage and question. H1 stays claimed as pre-registered, but any write-up must report this sensitivity next to it, and it weakens the "leakage leaves the weights" half of the cross-channel reading.
2. **H3 is mostly CB-AO ignoring the library.** CB-AO's natural-question AUC is at chance (0.497), its document scores are about 0.02 whether the passage is in the library or not, and its RAG F1 (0.085) is **below the untrained P0** (0.167; P0 − CB-AO +0.081 [0.053, 0.113]). Answer-only closed-book training taught the model to ignore retrieved text. RG's datastore AUC (0.847) is close to what P0 showed on V (0.90, a different cohort, so this is a rough comparison only). So the H3 difference reads better as "CB-AO hides the datastore by not using it" than as "RG adds datastore exposure beyond a normal RAG model". The entailment scorer and the N-document comparison agree on the direction.
3. **Training adds no natural-question signal.** The training-only cell is at chance for both arms and both scorers, and "both" is no more detectable than "library-only". This fits H4's equivalence: we see no passage-fact membership in the weights through these probes.
4. **The verbatim baseline fails on the fine-tuned models.** At ε = ∞ they answer "yes" to the yes/no question 70–90% of the time whether or not the passage is in the library (AUC about 0.5). On P0 (V) it reached 0.75.
5. **Release noise at ε = 16 reverses the ε = ∞ direction** (RG now leaks less: +0.073 [0.016, 0.133], 10 targets). The no-noise attack stays near the ceiling at ε = 16 (0.96), as expected: DP-SGD's training ε does not cover crafted-request releases (§7).
6. **Unexplained W1–W0 gap for RG (needs a check).** RG's member worlds answer better than its non-member worlds (RAG F1 0.378 vs 0.320; F_P 0.404 vs 0.348; NLL 3.77 vs 3.99). CB-AO shows no gap (0.085 vs 0.082), and neither arm does at ε = 16. The two worlds differ only in one of 128 training records, so a systematic utility gap is not expected.
   - The code gives no cause: client training is seeded per client and round, nothing changes a global torch setting, and answers are greedy.
   - The gap may be noise from the accepted training non-determinism (RG training diverges strongly between runs, as `timing-3`/`timing-4` showed), but the paired interval is not yet computed.
   - **Why it matters:** the Reference and NQ endpoints compare W1 with W0 per target. A world-level difference beyond the target record would add to every s₁ − s₀ for RG. It would push RG's Reference AUCs up, so it would work against H1, not for it.
   - A read-only check is below.

**Read-only check of the W1–W0 gap (server; it reads result files only and prints paired utility differences and training losses).**

```bash
python - <<'PY'
import json, os, glob
import numpy as np
G = os.environ['G']; rows = {}
for f in sorted(glob.glob(G + '/final/results/*/result.json')):
    r = json.load(open(f))
    if r['kind'] != 'paired':
        continue
    w1, w0 = (np.mean([q['f1'] for q in r['worlds'][w]['utility']['F']]) for w in ('W1', 'W0'))
    loss = [[h.get('mean_client_loss') for h in r[f'federated_history_{w}']] for w in ('W1', 'W0')]
    rows.setdefault((r['arm'], str(r['epsilon_budget'])), []).append((w1 - w0, loss))
rng = np.random.default_rng(20260930)
for key in sorted(rows):
    d = np.array([x[0] for x in rows[key]])
    boot = d[rng.integers(0, len(d), (10000, len(d)))].mean(axis=1)
    print(key, 'targets', len(d), 'mean W1-W0 F1', round(d.mean(), 3),
          '95% CI', [round(float(q), 3) for q in np.quantile(boot, [0.025, 0.975])], 'W1 better in', int((d > 0).sum()))
    losses = [x[1] for x in rows[key] if None not in x[1][0] + x[1][1]]
    if losses:
        l1, l0 = (np.array([l[w] for l in losses], float) for w in (0, 1))
        print('   mean client loss per round  W1', l1.mean(axis=0).round(3).tolist(), ' W0', l0.mean(axis=0).round(3).tolist())
        print('   corr(F1 gap, last-round loss gap W1-W0)',
              round(float(np.corrcoef(d[:len(l1)], l1[:, -1] - l0[:, -1])[0, 1]), 3))
PY
```

(The DP worlds record no per-round loss, so only the ε = ∞ rows print losses.)

**Result of the W1–W0 check (server, 2026-10-16): the gap is real and systematic.**

| Arm, budget | Targets | Mean W1 − W0 RAG F1 (95% CI) | W1 better | Mean client loss per round, W1 / W0 |
|---|---:|---|---:|---|
| RG, ε = ∞ | 60 | **0.057 [0.031, 0.086]** | 43 | 1.519 / 1.516 → 0.929 / 0.951 → **0.509 / 0.580** |
| CB-AO, ε = ∞ | 60 | 0.003 [−0.008, 0.015] | 28 | 3.830 / 3.817 → 2.700 / 2.720 → **1.813 / 1.862** |
| RG, ε = 16 | 10 | 0.001 [−0.000, 0.003] | 8 | not recorded (DP) |
| CB-AO, ε = 16 | 10 | 0.001 [−0.003, 0.005] | 5 | not recorded (DP) |

Correlation of the per-target F1 gap with the round-3 loss gap: −0.54 (RG), −0.31 (CB-AO).

- **What it shows.** In both arms at ε = ∞, the two worlds start level (round 1) and the non-member world W0 then trains to a higher loss (round 3: +0.071 for RG, +0.049 for CB-AO). For RG this carries into answer quality. Worlds that ended at a lower loss answer better.
- **Why it is unexpected.** The worlds share all 128 base records and the per-client, per-round seeds; they differ only in the target record versus its hold record in client 0. One record is 1/132 of the round's training data, so it cannot shift the mean client loss by 0.07 directly, and the accepted GPU non-determinism has no direction.
- **What does not explain it (code read, 2026-10-16):**
  - training seeds (identical per client and round);
  - model loading (fresh float32 load per client, no cache);
  - the training device and the Ray backend (fixed by config);
  - environment variables (only `gpu.py` sets `CUDA_VISIBLE_DEVICES`, before any training);
  - data shuffling (seeded);
  - answer generation (greedy).
  The order of work in a paired job is the leading candidate, but no mechanism is identified: W1 trains first in a fresh process, and W0 trains after the causal passes have run in that process. The ε = 16 worlds train through `private_train` and show no utility gap.
- **What it can and cannot affect (Stage 2):**
  - **Unaffected:** H3 and the datastore secondaries (both cells are W0), H2 and the causal secondaries (W1 only), and every utility endpoint (H1's utility, H5, H7 use W1).
  - **H4:** retention is F1(W1) − F1(W0) closed-book. A world advantage would push R upward, and R is about 0.006, so H4's "equivalent" reading stands.
  - **H1 (and H6) can be affected.** The Reference score compares the target record under W1 and W0. If W1 is generally better trained, both arms' Reference AUCs are inflated by an unknown amount. Because both arms show the loss gap, the effect on H1's difference is unknown in sign and size. This is a validity question for H1, recorded here; the primary outcome is not changed.
- **Diagnosis (2026-10-18):** the world-order diagnostic (12 V targets) is inconclusive by its fixed readings. Without the causal passes between the trainings, no order shows W1 ahead, and the target record does not make W1 train better. The work a paired job does between the worlds (causal passes, W1 measurements) remains the suspect; it is not shown.

## Decisions A and B; Stage 2 protocol frozen (2026-10-06)

**Fifth timing launch `timing-5` (core `8b031bbfdcc1`, both speed-ups; all checks verified).**

| | ε = ∞ | ε = 16 (accounted 16.0096 per world) |
|---|---:|---:|
| FL training, per world | 300 s / 299 s | 320 s / 314 s |
| Measurements, per world (W1 / W0) | 273 s / 428 s | 976 s / 1034 s |
| Causal: 2 directions × (plain + release noise), 160 trials | 818 s (was 3074 s) | 831 s (was 3037 s) |
| **Total** | **0.59 h** (was 1.18 h) | **0.97 h** (was 1.60 h) |

- **Per trial:** about 5.0 s of wall time (2.0–2.3 s of victim compute) in every pass, with or without release noise. `timing-4` took 12 s (plain) and 26 s (noise). The remaining ~3 s is the CPU scoring (`alignment_terms`) and moving the gradient off the GPU.
- **Every result records** `observation: {in_process: true, noise_on_device: true}`.
- **W0's ε = ∞ measurements took 428 s** (304 s in `timing-4`). This is probably run-to-run variation in answer length (training non-determinism, accepted 2026-10-04); it is not explained further.

**Re-costed Stage 2** (memo §4.1): about 0.48 h per ε = ∞ job and 0.85 h per ε = 16 job with one causal direction; ε = 64 assumed equal to ε = 16; P0 and RG-public about 1.6 h.

**Decisions (the researcher, 2026-10-06).**
- **A = 60 targets at ε = ∞** (option A3).
- **B2:** H6 at ε = 16 with 10 targets (the first 10 of the 60); the ε = 64 jobs are dropped. ε = 64 is a secondary of revision 4, so no primary definition changes.
- Budget about 76 GPU-h, inside the approved 40–80 (about 4 h of headroom; reruns may exceed it).

**`stage2_protocol.json` is frozen** with these decisions: 146 jobs (60 × {RG, CB-AO} at ε = ∞, 10 × {RG, CB-AO} at ε = 16, 5 RG-public seeds 9095–9099, 1 P0), all on fresh final targets (seeds 9000 + target index, so 9000–9059). The study has 124 eligible final targets.

**Server steps for Stage 2** (the researcher runs these in tmux, in `/home/calc08/projects/LLMPrivacy/fl_grounded_stage0`, env `LLMPrivacy`, with `G` as below), after reviewing `stage2_protocol.json`:
1. `git pull --ff-only` on `feat/grounded-fl-stage0`; confirm the core fingerprint is still `8b031bbfdcc1`.
2. `prepare final "$G/final" --study "$G/build/study.json" --gate "$G/gate.json"`, then `resolve`.
3. `run --gpu 0 --max-jobs N` repeatedly until all 146 jobs are complete; `check` at any time (integrity and timings only).
4. `analyze "$G/final" "$G/final-analysis.json"` once, after the last job.
 The frozen guard files and `causal_attack_validation.py` are byte-identical, and a test pins each.

## Observation speed-ups (2026-10-06; core fingerprint `4a001c5eb678` → `8b031bbfdcc1`)

**Why.** `timing-4` measured about 12 s per plain causal trial (2.5 s of victim compute) and about 26 s per release-noise trial (17 s of victim compute). About 9 s of every trial was Flower/Ray moving the 2 GB request and gradient; drawing the release noise as float64 NumPy on the CPU cost about 15 s. Both measures were agreed with the researcher on 2026-10-05. Both are opt-in; the guard studies' default paths are unchanged.

1. **Release noise on the GPU** (`defenses.protect_observation_on_device`, used through `causal_probe.protected_gradients(noise_on_device=True)`).
   - The same whole-vector clipping in float64 arithmetic, and Gaussian noise with the same std (σ_obs × C = 1.0 × 1.0), drawn by torch on the gradients' device from a freshly seeded generator (`secrets.randbits(63)`, as `private_train` does). Each tensor is released as float32 as soon as it is protected.
   - `protect_observation` itself is unchanged.
   - Tests: the clipped release equals `protect_observation`'s bit for bit (clip active, clip inactive, no defense); the noise has mean 0 and the configured std (within 1% over 400,000 draws); two calls draw different noise.
2. **In-process observation trials** (`amia.observe_in_process`, through `run_attack_trials(in_process=True)`).
   - Each trial computes the victim's released gradient in the job's own process, with the same `trial_member`, `sample_attack_batch` seed (`config.seed + 1009 + trial_id // 2`), `seed_training(config.seed + trial_id // 2)`, request parameters and `protected_gradients`, then scores it with `alignment_terms` / `score_from_terms`. There is no Flower/Ray round. The trial records and checkpoints have the same fields.
   - The request is loaded into the victim once per pass. Gradients come from `torch.autograd.grad` in eval mode and leave the model unchanged (the same argument as `cached_victim_model`, tested earlier). The attacker's model stays on the CPU; the victim model is released after the pass.
   - It refuses the probe-head attack, guards and calibrated thresholds, so only the grounded study can use it.
   - Test: on a tiny GPT-2, 6 counterbalanced trials, the in-process path (with and without GPU noise) releases bit-identical gradient arrays and gives identical trial records (`trial_id`, `truth_member`, `score`, `batch_pair_seed`, `alignment_terms`, …) to the existing `VictimClient` path driven through Flower's serialization (the simulation is replaced by a direct driver, without Ray), with no defense and with clipping.
3. **The grounded job opts in to both** (`grounded_job.OBSERVATION`), and each causal direction records `"observation": {"in_process": true, "noise_on_device": true}` in `result.json`.

**Not done.** "Two clients training at once" was okayed earlier but advised against (two full-model clients of about 9 GB each will not fit or speed up on the 20 GB vGPU slice). **Decision (2026-10-06, the researcher): dropped.** Batched answer generation was rejected (it changes the answers).

**Next.** A fifth timing launch (`timing-5`) on `8b031bbfdcc1` measures the saving before Stage 2 is costed. The scoring step (`alignment_terms`, NumPy over about 0.6 billion values) still runs on the CPU, unchanged, so it is part of what `timing-5` measures.

**Server steps for `timing-5`** (the researcher runs these in tmux, in `/home/calc08/projects/LLMPrivacy/fl_grounded_stage0`, env `LLMPrivacy`, with `G=/home/calc08/projects/LLMPrivacy/fl_llm_m_attack_experiments/outputs/grounded-fl-20260930`):
1. `git pull --ff-only` on `feat/grounded-fl-stage0`, then confirm the core fingerprint prints `8b031bbfdcc1`.
2. `prepare timing "$G/timing-5" --study "$G/build/study.json" --gate "$G/gate.json"`, then `resolve`, `run --gpu 0 --max-jobs 2` and `check`.
3. The per-trial split (read-only; prints timings only, never an AUC or score):

```bash
python - <<'PY'
import json, os, glob, statistics as st
G = os.environ['G']
for f in sorted(glob.glob(G + '/timing-5/results/*/result.json')):
    r = json.load(open(f))
    print(r['arm'], r['epsilon_budget'], r['implementation_fingerprint'],
          {k: round(v) for k, v in r['timings'].items()})
    for d, p in r.get('causal', {}).items():
        for name in ('plain', 'release_noise'):
            if name in p:
                t = p[name]['trials']
                print(' ', d, name, 'wall s/trial', round(p[name]['seconds'] / len(t), 1),
                      'victim s/trial', round(st.mean(x['response_seconds'] for x in t), 1),
                      'observation', p.get('observation'))
PY
```

Do not pull later commits until `timing-5` has finished and `check` has run: a launch refuses to run or be checked once the study tool or core code changes.

## Stage 2 tooling (2026-10-06; tool only, core unchanged at `8b031bbfdcc1`)

Built while `timing-5` runs, so that the protocol can be frozen as soon as decisions A and B are made. **Nothing in Stage 2 is prepared or run.**

- **`stage2_protocol.json`** (a draft at this commit; frozen later on 2026-10-06, above). It records everything decided so far:
  - the final cohort, seed band 9000–9099 and fresh final targets only (the first n in eligible order at ε = ∞; each DP arm takes the first n_ε of those; seed 9000 + target index); RG-public on seeds 9095–9099 and P0, both scored on the final F;
  - the frozen attacker choices (RG record/template/F1, CB-AO template/template/F1; CB-LM is not run);
  - C2 with its pre-registration notes (below), D1, H4 primary (seven primaries, 99.3%), margins 0.05, max_length 512, the gate (τ 0.45, 3-grams), accepted non-determinism, pooled choices, two clients dropped, batched generation rejected.
  - **Open in the file:** `targets.inf` (decision A), `targets.dp` (decision B) and the budget. Its status is `draft`, and `prepare final` refuses it until the status starts with `frozen`.
- **`prepare final`** reads the frozen protocol and checks every rule it may only tighten: at least 20 targets at ε = ∞; ε = 16 always present (H6) with at least 10 targets; only ε 64 and 16; DP targets a subset of the ε = ∞ targets; the final cohort and its seed band; 5 RG-public seeds above every target seed; the release-noise pass present when H2's primary uses it; H4 primary; margins and the Bonferroni level unchanged; the protocol names the committed Stage 0/1 protocol. Each paired job measures everything pre-registered, with the causal attack in the arm's frozen direction only (plain and release noise). The launch keeps copies of both protocols, and `resolve`, `run`, `check` and `analyze` refuse if either committed protocol changes.
- **Confirmation `analyze`** (launch kind `final`) runs the tested `hypotheses()` once, with the frozen choices, H2 on the release-noise pass (`h2_pass="release_noise"`; the no-noise D, interval and reading are reported as `secondary_no_noise` from the same target draws), utility clusters from the final F articles, and P0 and RG-public from the final launch. It refuses if the paired jobs, targets, seeds or utility questions differ from the protocol and the launch's declaration. `hypotheses()` keeps its default (`plain`), so the pilot analysis is unchanged (a test shows every other hypothesis is identical).
- **`check` never shows endpoint values.** It now builds every field from an allowlist: timings only from `*_seconds` numeric entries, sequence lengths only `max_tokens`/`mean_tokens`, per-pass causal seconds and row counts. A test smuggles AUC-, F1- and score-like values into a result's timings and sequence lengths and shows none reach the report.
- **Tool hash changed.** `timing-5` must finish and be checked before the server pulls this commit.

## Stage 1 pilot (V, seeds 8000–8099; tuning data, never evidence)

**Run.** 33 jobs: 4 targets × {RG, CB-AO} × ε {∞, 64, 16}, 4 CB-LM, and 5 RG-public seeds. P0 utility comes from the datastore launch. Accounted ε per world: ∞, 63.84 and 16.01, matched across arms. H4 is primary (from the 6-epoch control).

**Attacker choices (rule: per arm, pooling all budgets; the highest pooled tuning AUC wins, ties go to the standard form or F1).**

| Arm | Reference form (tuning AUC record / template) | Causal direction (record / template) | NQ scorer (F1 / entailment) |
|---|---|---|---|
| RG | **record** (0.611 / 0.611, a tie, so the standard form) | **template** (0.493 / 0.978) | **F1** (0.899 / 0.688) |
| CB-AO | **template** (0.458 / 0.597) | **template** (0.682 / 0.989) | **F1** (0.819 / 0.701) |
| CB-LM | **record** (1.000 / 0.563) | **record** (0.981; record only) | **F1** (0.844 / 0.813) |

**Projected decision half-widths for option (b)** (99.3% intervals; 20 targets at ε = ∞ and 10 for H6; normal approximation from 4 pilot targets, very rough).

| Endpoint | Pilot half-width | Projected | Margin |
|---|---:|---:|---:|
| H1 Reference AUC difference | 0.50 | 0.24 | 0.05 AUC |
| H1 utility (RG − CB-AO) | 0.098 | 0.043 | 0.05 F1 |
| H2 causal AUC difference | 0.00 | 0.00 (see below) | 0.05 AUC |
| H3 NQ AUC difference | 0.25 | 0.15 | 0.05 AUC |
| H4 R_RG | 0.10 | 0.052 | 0.05 F1 |
| H4 D_H4 | 0.17 | 0.087 | 0.05 F1 |
| H5 F1(RG-public) − F1(RG) | 0.116 | 0.054 | 0.05 F1 |
| H6 utility at ε = 16 | 0.007 | 0.004 | 0.05 F1 |
| H6 Reference AUC at ε = 16 | 0.25 | 0.17 | 0.05 AUC |
| H7 F1(P0) − F1(RG) | 0.074 | 0.033 | 0.05 F1 |

**Readings (for the Stage 2 decision; descriptive only).**
- **The AUC endpoints are far too imprecise at option (b).** Projected half-widths are 0.15–0.24 against a 0.05 margin, so H1, H3 and H6's leakage condition would almost surely be inconclusive.
  - Scaling the rough projections, a ±0.05 decision interval would need about 450 targets for H1, about 170 for H3 and about 110 for H6's AUC.
- **H2 is at the ceiling.** The zero spread means the template-direction causal AUC is identical in both arms on every pilot target. The other causal values (pooled about 0.98–0.99, saturated) suggest it is at or near 1. The projection cannot estimate precision there, and H2 would read "no meaningful change" or be degenerate.
- **The Reference attack is weak under answer-only training.** Tuning AUCs are about 0.46–0.61 for RG and CB-AO, against 1.0 for the full-LM CB-LM bridge.
- **The utility endpoints are close to the margin** (H1 utility, H5, H7: 0.03–0.054) and optimistic, because cluster and seed variance does not shrink with targets.
  - H6's utility spread (0.004) is suspiciously small; check the descriptive differences.
- **The causal attack stays effective at every budget.** DP-SGD's ε does not cover the crafted-request releases (§7), and the pooled tuning AUCs stay at about 0.98.

**Descriptive pilot differences** (estimate [95% CI]; 4 targets per arm and budget; tuning data, never evidence; §8: RG-vs-P0 changes nothing).

| Quantity | Estimate |
|---|---|
| H1 Reference AUC, CB-AO − RG (ε = ∞) | 0.19 [0.00, 0.75] |
| H1 utility, RG − CB-AO (ε = ∞) | **+0.27 F1 [0.20, 0.34]** |
| H2 causal AUC, CB-AO − RG | 0.00 [0.00, 0.00] (both at the ceiling) |
| H3 NQ AUC, RG − CB-AO | 0.19 [0.00, 0.47] |
| H4 R_RG / D_H4 | 0.05 [0.00, 0.15] / 0.08 [0.00, 0.25] |
| H5 RG-public − RG (ε = ∞) | −0.02 [−0.11, 0.07] |
| H5 secondary: RG-public − RG at ε = 64 / 16 | +0.19 [0.10, 0.28] / +0.20 [0.11, 0.29] |
| H6 utility, RG − CB-AO (ε = 16) | 0.001 [−0.003, 0.007] |
| H6 Reference AUC, CB-AO − RG (ε = 16) | −0.06 [−0.25, 0.19] |
| H7 P0 − RG (ε = ∞) | −0.23 [−0.28, −0.18] |

**Readings (design input only).**
- **Grounded training's utility gain at ε = ∞ is large and precise.** It is about +0.27 F1 over CB-AO and about +0.23 over P0.
- **RG-public is close to RG at ε = ∞.** The gain may come from learning the grounded answering format rather than private facts.
- **At the conservative accountant's ε = 64 and 16, the DP-trained models land at about P0's utility.** RG-public − RG(ε) ≈ +0.19–0.20, while RG-public ≈ RG(∞) ≈ P0 + 0.23. RG and CB-AO are then equal (H6 utility ≈ 0).
  - So DP-SGD at these σ (1.91, 5.45) erases the fine-tuning effect.
  - This also fits the slow DP-world measurements, which match P0's long generations.
- **Implication.** As designed, H6 and the DP frontier would compare two nearly pretrained models.

**Stage 2 design memo (2026-10-05, for the researcher's review).** See [`stage2_design_memo.md`](stage2_design_memo.md). It recommends S2:
- 40 targets at ε = ∞;
- H6 kept at ε = 16 with 10 targets;
- ε = 64 and the release-noise pass dropped (both secondary);
- H2 kept with a ceiling caveat.

S2 costs about 56 GPU-h with no primary definition changed. (Superseded: C2 on 2026-10-05; A = 60 and B2 on 2026-10-06, above.)

**Pilot causal AUCs with release noise (read 2026-10-05; tuning data).** Chosen template direction, ε = ∞:

| | No noise | Release noise |
|---|---|---|
| RG | 1.000 | 0.744 [0.68–0.78] |
| CB-AO | 1.000 | 0.621 [0.53–0.79] |

- **The noise takes the attack off the ceiling,** and suggests RG leaks more through updates.
- **Inverted record direction.** RG's record direction is inverted at ε = ∞ (no-noise AUC 0.31).
- **Memo §3 C** is updated with the full table, option C2 (✱, a selection-risk caution) and scenario S5 (30 targets with the noise pass, about 71 GPU-h).

**Decision C (2026-10-05, the researcher): C2.** H2's primary endpoint becomes the release-noise causal pass (σ_obs 1.0, the chosen template direction). This is a definition change made after seeing the pilot, so the Stage 2 pre-registration must:
- say openly that the choice was made from pilot data;
- keep the no-noise H2 as a reported secondary;
- state that H2 now measures leakage through noise-protected releases, not raw updates.

Stage 2 uses fresh final targets only. C2 implies D1 (the noise pass is kept). Decisions A and B followed on 2026-10-06 (above).

**The command used (read-only, for decision C in the memo).** This prints the pilot's mean causal AUC per arm, budget, direction and pass, without and with release noise:

```bash
python - <<'PY'
import json, os, glob, statistics as st
G = os.environ['G']; rows = {}
for f in glob.glob(G + '/pilot/results/*/result.json'):
    r = json.load(open(f))
    if r['kind'] != 'paired':
        continue
    for d, p in r['causal'].items():
        for name in ('plain', 'release_noise'):
            if name in p:
                rows.setdefault((r['arm'], str(r['epsilon_budget']), d, name), []).append(p[name]['auc'])
for k in sorted(rows):
    print(k, round(st.mean(rows[k]), 3), [round(x, 3) for x in rows[k]])
PY
```

## Stage 0 results so far (validation data on V; not evidence)

**Build (server, fingerprint `714ec988b38b`).** Study sha256 `d7eac161d178b6e855979fc879752a5d511516674e913788e590194d4dbecd92`.

- **Exclusions.** All 50 earlier target hashes matched SQuAD rows; the smoke target is the only allowed unmatched one. The cohorts are:

  | Cohort | Targets |
  |---|---|
  | Reserved finals | 4 |
  | Causal Stage A | 4 |
  | Causal Stage B | 10 |
  | Matched controls | 20 |
  | Guard v3/v4 development | 11 |

  52 questions were removed, because SQuAD repeats some identical question–answer pairs.
- **Skips.**
  - 2,972 passages fell outside 24–160 words.
  - 174 duplicate questions within a passage.
  - No article was skipped.
- **Eligible targets.**

  | | Targets | Limit | Fewer than 4 questions | Fewer than 3 qualifying probes | Eligible but no hold record |
  |---|---:|---|---:|---:|---:|
  | V | 186 | hold records | 63 | 55 | 311 |
  | final | 124 | hold records | 123 | 35 | 362 |

**Gate calibration (CPU, V).**
- The threshold is τ = 0.45 (3-gram overlap with the top document).
- Benign gold-document loss is 4.4% (13 of the 297 V utility questions whose gold document is retrieved), within the 5% limit.
- Verbatim membership probes are withheld 100% of the time.

**P0 datastore-sensitivity check (60 V targets, the same passage in vs out; launch sha256 `972362d8…`). It passed for both scorers, so H3 is evaluable and both scorers are eligible for §6.5.**

| Scorer | Pooled AUC | 95% CI (target bootstrap) | Passes (lower bound ≥ 0.70) | Secondary AUC vs 64 N documents |
|---|---:|---|---|---|
| F1 | 0.902 | [0.854, 0.947] | yes | 0.890 [0.826, 0.945] |
| Entailment | 0.887 | [0.837, 0.934] | yes | 0.907 [0.852, 0.953] |

- The verbatim yes/no baseline has AUC 0.75.
- No refusals occurred, so the refusal-excluded AUCs equal the primary ones.
- **Timing.** The P0 job took 31 min:
  - 14 min for the datastore pass;
  - 16.5 min for utility (300 F questions with and without context, plus 100 F_P).
- **Scope.** This validates the attack's sensitivity on the pretrained model only. It is not a privacy result.

**Timing launch (V timing target, seed 8081; two full paired RG jobs; all checks verified).**

| | ε = ∞ | ε = 16 (accounted 16.0096 per world, 51 steps) |
|---|---:|---:|
| FL training, per world | 299 s / 303 s | 319 s / 323 s |
| Measurements, per world (W1 / W0) | 271 s / 304 s | 1009 s / 1084 s |
| Causal: 2 directions × (plain + release noise), 160 trials | 5796 s (97 min) | 5802 s (97 min) |
| **Total** | **1.94 h** | **2.37 h** |

- **Sequence length.** The longest training record was 329 tokens (mean 199), so the 512 cap leaves room.
- **DP-SGD overhead.** About 7% on training time.
- **DP measurements are 3.7× slower.** The probable cause, not yet checked, is that the noisier model's answers run to the 64-token cap.
- **The causal attack dominates.** It takes about 36 s per trial: each trial reloads the model in a Ray client.
- **Budget projection.** At these rates the pilot as planned costs about 59 GPU-hours (budget 16–32). Stage 2 option (b) costs about 110–130 GPU-hours (budget 40–80), even with only the chosen direction. **Cost decision (2026-10-03).** The researcher chose one measure only: the victim client keeps its loaded model across observation trials (`amia.cached_victim_model`, opt-in through `run_attack_trials(reuse_victim_model=True)`). Each trial still overwrites every tensor with the request, and a test shows the gradients are bit-identical to a fresh load. Nothing else changes: the pilot keeps both directions and the release-noise pass, and no-context F1 stays. The guard studies' path is unchanged. Core fingerprint `714ec988b38b` → `4da0821d7111`. The 3- and 6-epoch control runs stay on `714ec988b38b`. A new timing launch must measure the saving before the pilot is costed.

**Second timing launch `timing-2` (2026-10-03).**
- **Ran on the old code.** `d171bed` had not been pushed, so the server worktree was still at `b7a185d`. This is inferred from the branch state; the launch's fingerprint has not been confirmed. **It does not test the fix.**
- **Repeatability check.** It does show that timings repeat. Totals: 6958 s vs 6984 s (ε = ∞) and 8577 s vs 8548 s (ε = 16). Causal: 5767 s and 5863 s.

**Third timing launch `timing-3` (fingerprint `4da0821d7111`, with the reuse fix).**
- **Saving.** The causal attack took 5567 s and 5558 s, only about 4% less. Model reloading was not the bottleneck.
- **Per-trial split (ε = ∞ job).**

  | Pass | Wall time per trial | Victim client compute |
  |---|---:|---:|
  | Plain | 27–28 s | 3 s |
  | Release noise | 41–43 s | 17–18 s |

- **Cause.** About 25 s per trial is simulation overhead. The likely cause, unconfirmed, is that the 2 GB request goes to all four clients and the three bystanders return it. Drawing release noise in float64 costs about 15 s.
- **Totals.** 1.87 h (ε = ∞) and 2.30 h (ε = 16).

**Second cost measure (2026-10-03, researcher's choice).** Observation rounds now simulate only the attacked client (`run_attack_trials(observe_target_only=True)`, opt-in; the guard studies' path is unchanged). It receives the same partition, seeds and request, so observations are unchanged; the three bystanders no longer receive and return the 2 GB request every trial. Core fingerprint `4da0821d7111` → `4a001c5eb678`. The pilot keeps both directions, the release-noise pass and no-context F1. A fourth timing launch measures the saving. The 6-epoch control rerun runs from a worktree pinned to `b7a185d` (fingerprint `714ec988b38b`), the same code as the 3-epoch run.

**Fourth timing launch `timing-4` (fingerprint `4a001c5eb678`, victim-only observation).**
- **Saving.** The causal attack took 3074 s and 3037 s, 45% less than `timing-3`.
- **Totals.** 1.18 h (ε = ∞) and 1.60 h (ε = 16). Training and measurement times are unchanged.
- **Re-costed pilot.** About 37 GPU-hours (budget 16–32):

  | Jobs | GPU-h |
  |---|---:|
  | 8 × ε = ∞ | 9.5 |
  | 8 × ε = 64 (unmeasured; assumed between the two measured budgets) | ≈ 11 |
  | 8 × ε = 16 | 12.8 |
  | 4 CB-LM, record direction only | ≈ 3 |
  | 5 RG-public | ≈ 1 |

- **Re-costed Stage 2 option (b).** About 75–80 GPU-hours (budget 40–80), with the chosen direction only:
  - 40 ε = ∞ jobs at about 0.76 h;
  - 40 DP jobs at up to about 1.18 h;
  - P0 and RG-public.
- **Remaining large cost.** The DP worlds' measurements take about 1000 s per world, against 275 s at ε = ∞.
- **Per-trial split (`timing-4`, ε = ∞).**

  | Pass | Wall time per trial | Victim compute |
  |---|---:|---:|
  | Plain | 12 s | 2.5 s |
  | Release noise | 26 s | 17 s |

- **Budget decision (2026-10-03).** The researcher accepted the pilot as designed, at about 37 GPU-hours.
- **Open: run-to-run agreement.** `timing-3` and `timing-4` used the same target and seed. The largest per-trial difference in the no-noise causal scores was 0.41 (record direction) and 0.92 (template direction), far more than GPU non-determinism should give. **The pilot waits until this is explained.**
- **Explained (2026-10-04).** The two runs trained different W1 models. Round losses agree in round 1 (1.42352 vs 1.42352), drift by round 2 (0.7948 vs 0.7950) and differ in round 3 (0.422 vs 0.326). The attack amplifies the difference: the template request ended at loss 17.4 vs 7.8. The victim saw identical labels and batch seeds, and the FL training code is identical in both fingerprints, so the cause is GPU non-determinism in training, not the observation changes.
- **Decision.** The researcher accepted run-to-run non-determinism. Each world is one random draw, as the paired design assumes; the noise affects W1 and W0 alike and is carried by the target bootstrap. Reruns are not bit-reproducible.

**Positive control, 6-epoch rerun (launch sha256 `aa367f4d…`, fingerprint `714ec988b38b`, worktree at `b7a185d`).**
- **Learning check: passes at the minimum.** 10 of 12 targets learned their passage.

  | | Passage-NLL ratio | Continuation match |
  |---|---|---|
  | 10 learned | 0.000–0.106 | all 1.0 |
  | 2 not learned | 0.138 and 0.031 (both pass) | 0.375 and 0.188 (both fail) |

- **Retention sensitivity (decisive): passes.** Mean r = 0.142 F1, 95% target-bootstrap CI [0.066, 0.225], lower bound > 0.05. **H4 stays primary: seven primaries, 99.3%.**
- **Secondary membership AUC (training-only cell, learned targets).** F1 0.80 [0.65, 0.95]; entailment 0.95 [0.87, 1.00]. Both are sensitive.
- **Scope.** This shows the probes detect deliberately memorized passage facts. It is not a finding about grounded training.

**Attacker-choice rule (decided 2026-10-04).** Each arm's choice pools its pilot jobs over all three budgets.

**Pilot analysis completed before the pilot is prepared (tool only; core fingerprint unchanged at `4a001c5eb678`).** `analyze` on the pilot launch now:
- reads P0's utility read-only from the datastore launch (`--p0-launch`);
- computes the attacker choices, the descriptive pilot differences for H1–H7 and projected decision half-widths for option (b): 20 targets at ε = ∞ and 10 for H6.

**Positive control, 3 epochs (12 V targets; launch sha256 `9d203eb0…`).**
- **Learning check.** 4 of 12 targets learned their passage; at least 10 are needed.
- **Rule.** By the pre-registered rule, the control is rerun once at 6 epochs, on the same 12 targets, lr 1e-4 and code fingerprint `714ec988b38b`.
- **Not computed.** Retention and AUC sensitivity are not computed until the rerun decides.

## What was built

| Stage 0 item (proposal §8, §12) | Where |
|---|---|
| Article-disjoint split builder, eligibility rules, skip counts, old-target exclusion, audit | `master_script/core/grounded.py` (`build_study`, `audit_study`); tool `build` |
| RG / CB-AO / RG-public encoders (the `rag._prepare_prompt` chat template, answer-only labels) and CB-LM (today's closed-book text) | `grounded.encode_record`, `grounded.training_record` |
| Answer-only loss in the AdamW client path and in `private_train` | `scoring.EncodedExample`, `amia.make_loader`, `defenses.private_train` |
| The same mask in the causal attack (victim gradients and the optional template direction) | `causal_probe._loss_inputs`, `optimize_request` |
| Supplied worlds for FL training; attack batches with a declared target | `amia.federated_fine_tune(world=…)`, `amia.run_attack_trials(target=…)` |
| New RAG study data from L, N, P and V, with F (300 questions, article ids kept), F_P and the probe sets | `study.json` written by `build` |
| Reference (record and template forms), causal (both directions, 40 trials, release-noise pass), the 3-query natural-question pass (F1 and pinned NLI), retention, utility | `master_script/core/grounded_job.py` |
| Per-run ε record; refusal to compare runs whose ε differs | `grounded.epsilon_record`, `same_epsilon`; tool `require_matched_epsilon` |
| P0 datastore-sensitivity check; positive control with learning check and the one 6-epoch rerun; retention sensitivity (decisive); AUC ≥ 0.60 (secondary) | tool `prepare datastore|control`, `analyze` |
| Verbatim-overlap library gate and its CPU calibration on V | `grounded.verbatim_overlap`, `gated_ranking`, `calibrate_gate`; tool `calibrate-gate` |
| Study tool: `build`, `calibrate-gate`, `prepare`, `resolve`, `run`, `job`, `check`, `analyze` | `master_script/tools/grounded_fl_study.py` |
| §9 constructions: paired target bootstrap, crossed article-cluster utility bootstrap, first-match six outcomes, H1–H7, H4 needing both endpoints, H7 non-inferiority, the cross-channel rule | tool `hypotheses` and helpers; tests in `tests/test_grounded_fl_study.py` |

The executor keeps the collector's guarantees without changing `collect_guard_traces.py`:
- frozen launches, with the core fingerprint, tool hash, protocol and study data all checked;
- CPU resolution of every target, world and token check before any GPU work;
- one job per process, one job per invocation by default;
- weights retired after verification.

`check` shows only integrity, accounted ε, timings and row counts, never a score, AUC, F1, refusal rate or learning value.

## Decisions made with the researcher (2026-09-30)

1. **Answer tokens.** The answer followed by the end-of-turn token (`<|im_end|>`, the tokenizer's EOS). The template must end the assistant turn with EOS, or encoding refuses.
2. **Library slot.** L = 255 fixed passages plus 1 slot. The slot holds a never-queried filler in L− and the target passage in L+, so the size is always 256. F comes only from the fixed passages.
3. **NLI.** `cross-encoder/nli-deberta-v3-base` at revision `6c749ce3425cd33b46d187e45b92bbf96ee12ec7` (label 1 is entailment; the label is read from the config). The premise is the passage. The hypothesis is `Question: <probe>\nAnswer: <response>`.
4. **Positive control.** The base arm is RG at ε = ∞. After FL, W1 gets a central AdamW fine-tune on the plain passage text: full LM loss, 1 step per epoch, lr 1e-4, 3 epochs, or 6 in the one rerun. W0 is not extra-tuned.
5. **Passage length.** 24–160 words in every group. Every T, T_hold and U record must fit `max_length` tokens untruncated for RG, CB-AO and CB-LM, or `build` refuses. `max_length` is **512** (see deviations). Inference uses top-4 retrieval, `max_context_tokens` 768 and `max_new_tokens` 64, as in the earlier configs.
6. **F and libraries.** L takes 4 passages from each of 64 articles (64 clusters). F is exactly 300 questions in seeded order, at most 2 per fixed L passage. P takes 8 passages from each of 32 articles. V has the same library layout.
7. **Base records.** For each target seed, 128 T records (one question per passage) are sampled from articles other than the target's. W1 and W0 share them exactly.
8. **Exclusion.** Question-level with a strict audit. Every manifest hash must match a SQuAD train row under the old closed-book formatting, or `build` refuses. The only allowed exception is the synthetic smoke target.

## Implementation choices (not fixed by the proposal; change any before Stage 0 runs)

- **Old record identity.** `_clean_record(_format_squad(row))` is recomputed at every `_char_budget` that `max_length` 32–512 could produce. `_clean_record` collapses whitespace, so today's records are single-line (`Question: … Answer: …`). CB-LM trains on exactly that string.
- **Sibling rules.** SQuAD v1.1 answer normalization, then containment of normalized token sequences, plus character-span overlap.
- **Seeded orders.** Study seed 20260930, with string-seeded `random.Random` for articles, groups, questions per passage and worlds. Target *k* is paired with T_hold record *k*.
- **Group sizes.**
  - Record groups take whole articles until a passage minimum is met:
    - T: 600 passages;
    - T_hold: 150 (V) or 100 (final);
    - U: 160;
    - C: 100, V only.
  - Library-like groups take a fixed number of articles:
    - L: 64 × 4;
    - N: 16 × 4;
    - P: 32 × 8.
  - A synthetic SQuAD-scale run of the builder used 270 of 442 articles.
  - The builder refuses fewer than 90 V or 40 final eligible targets.
  - **V is not smaller than final** (the proposal calls V "a smaller copy"). The library layout was kept equal so the datastore check transfers.
- **The C slice.** V's attacker-calibration slice (C) is built but not yet used by any Stage 0/1 measurement.
- **V target slices.** These are fixed from the study alone, in eligible order:
  - datastore: the first 60;
  - control: the next 12 whose passage has ≥ 64 tokens (skips counted);
  - pilot: 4;
  - CB-LM bridge: 4;
  - timing: 1.

  The seed is 8000 + target index; RG-public uses seeds 8095–8099.
- **Scorers.** The F1 scorer and utility F1 both use `rag.answer_utility` token F1, the existing utility metric, not SQuAD's normalized F1. The secondary answer NLL is the unchanged `rag.answer_nll`.
- **Verbatim baseline.** One Anderson yes/no query per document per condition, outside the 3-query budget.
- **Standard-attack AUC against N.** All 64 N documents in the P0 check; the first 16 in paired jobs (W0, library off).
- **Utility library.**
  - RG, CB-AO and CB-LM: both worlds use L+ (primary: W1).
  - P0 and RG-public: they have no target, so they use L with the filler.
- **ε per world.** 51 accounted steps, recorded for W1 and W0 and required to match. The two worlds are not composed: the pair is a counterfactual device, not two deployments.
- **DP-SGD clip norm.** 1.0, as in the earlier DP configs.
- **Gate.**
  - Rule: the share of the query's word 3-grams that occur verbatim in the top-ranked document; withhold that document at τ or above.
  - τ grid: 0.05–1.00. The smallest τ with ≤ 5% benign gold-document loss wins.
  - Benign set: V's F questions.
  - In jobs, the gate is recorded at retrieval level only (whether it would withhold, and the overlap); it triggers no extra generation.
- **Pilot attacker choices.** Made per arm, pooling that arm's pilot jobs over the three budgets. **Needs your OK before the Stage 1 analysis:** the alternative is ε = ∞ jobs only.
- **Learning-check continuation.** A positional token match over 32 greedy tokens (with `min_new_tokens` 32), given the first 32 passage tokens. This follows Biderman et al.'s 32/32 memorization score.

## Deviations and gaps

- **Sequence length 512, not 384 (2026-09-30).** The first server `build` refused because an RG training record (SQuAD question `57063fcb52bb8914006899b6`) needs 405 tokens. The researcher chose to raise `max_length` to 512 rather than skip long passages or tighten the word cap. The word range and the no-truncation rule are unchanged. The protocol's timing item "sequence length 384" is now measured at 512.
- **Related-work search: done 2026-10-12, after the confirmation** ([`related_work.md`](related_work.md)). Proposal §8 lists it as Stage 0 item 1; it was done last, so it informed none of the design or analysis choices. It is a documented structured search (queries, sources, criteria, counts), not a PRISMA review. 66 papers are included, each checked against its primary record (62 fully, 4 with the venue only from the arXiv comment or a search listing).
  - **Verdict:** no prior work found duplicates the three-channel design (grounded vs closed-book training, paired in/out worlds crossed with in/out of the library, weights + client updates + datastore, matched accounted DP). Every component has prior art, so there is still no "first" claim.
  - **Closest:** Zeng et al. 2024 and Huang et al. 2023 (training-data vs datastore leakage in one system, no FL or training-objective contrast), EPD (Fu et al. 2025), FedRepRAG (Peng et al. 2026) and Flemings et al. 2025.
  - **Limits:** DBLP and OpenReview forum pages blocked automated access. A manual check of both is advised before submission.
- **Pilot projections.** Wired into the pilot `analyze` (2026-10-04). They use a normal approximation, which is optimistic for the utility terms.
- **Stage 2.** `prepare final` and the confirmation `analyze` exist and are tested on synthetic data (2026-10-06). `stage2_protocol.json` is frozen (2026-10-06).
- **ε = 64 dropped from Stage 2 (decision B2, 2026-10-06).** The pre-registered frontier included ε = 64 as a secondary; Stage 2 runs only ε = ∞ and ε = 16. H5's secondary is reported against ε = 16 only.
- **60 targets, not 20, at ε = ∞ (decision A, 2026-10-06).** More targets than option (b) planned; this only narrows the intervals.
- **H2's primary changed after the pilot (C2, 2026-10-05).** The release-noise causal pass is H2's primary; the no-noise pass is a reported secondary. This is a post-pilot definition change and a new pre-registration, stated openly in `stage2_protocol.json`; it is not a tightening.
- **Stage 2 is analysed (2026-10-10).** The confirmation `analyze` ran once; its result is final. Nothing may be re-run or re-analysed to change an outcome.
- **Confirmation analysis scope.** The confirmation `analyze` reports H1–H7, the cross-channel rule, H2's no-noise secondary and H5's DP secondaries. The other descriptive secondaries were not in it. **2026-10-12:** they are computed by the read-only `describe` subcommand from the finished launch (run 2026-10-16), after the confirmation analysis, so they are post hoc descriptive reports and change no primary outcome. The ε = 64 frontier (not run) and per-client results (not recorded) stay unavailable.
- **Descriptive constructions chosen in `describe` (2026-10-12).** The proposal names these secondaries but not their exact construction, so these are implementation choices, not pre-registered definitions:
  - the "both" cell is reported through three contrasts (both vs neither, vs library-only, vs training-only);
  - the standard AUC against N pools each target's 16 N documents as negatives, scored by that target's W0 model;
  - the refusal-excluded H3 contrast and the verbatim baseline keep only targets whose two document scores or answers exist, with their own draws from the same seed;
  - the gate rate is the share of queries whose top-ranked document the gate would withhold.
- **Systematic W1–W0 world difference at ε = ∞ (2026-10-16, open; diagnosed 2026-10-16 to 2026-10-22, cause not shown).** The replicate (60 V targets) reproduces the gap in paired jobs (+0.037 [0.001, 0.074]) but not in no-causal pairs (+0.007), and the paired comparison is inconclusive (0.031 [−0.025, 0.086]). A read-only check of whether the gap moves the Reference scores is proposed in "Replicate". The read-only check confirms it: RG's member worlds answer 0.057 [0.031, 0.086] F1 better than its non-member worlds, and in both arms W0 ends training at a higher loss (round 3: +0.071 RG, +0.049 CB-AO) although the worlds differ in one record. No code cause was found; the order of work in a paired job is the leading candidate. It can inflate both arms' Reference AUCs, so H1's (and H6's) difference carries an unquantified world effect. H2, H3 and the utility endpoints are unaffected, and H4's equivalence stands. The paired design assumes the worlds differ only by the target record; this finding questions that assumption for the weights-channel endpoints. The world-order diagnostic (2026-10-18, 12 V targets, no causal passes between the trainings) is inconclusive by its fixed readings. It shows no membership effect and no consistent order effect, so the between-world work of a paired job remains the suspect; this is not shown. A direct test with real paired jobs on the same V targets is the researcher's decision.
- **H1 sensitivity is post hoc.** RG's template-form and CB-AO's record-form Reference AUCs are reported as a labelled sensitivity. H1's decision stays on the frozen forms.
- **DP jobs keep the causal passes.** Each DP paired job also runs the frozen-direction causal attack with release noise (secondary, the frontier), as the memo's cost estimates assumed.
- **Timing.** Timing is a separate launch: one full paired RG job at ε = ∞ and one at ε = 16, on the V timing target.
- **Observation speed-ups (2026-10-06).** Neither changes what is released or how it is scored, so no measurement definition changes. Two implementation differences are recorded:
  - GPU release noise uses torch's generator instead of NumPy's. The noise has the same distribution (Gaussian, std σ_obs × C, freshly seeded per release), but the two paths can never produce the same draw, so release-noise trials from `8b031bbfdcc1` and from earlier fingerprints are not draw-for-draw comparable (they never were: both draw from fresh seeds).
  - In-process trials skip the Flower/Ray transport. The simulated server still sees only the released arrays; the threat model is unchanged.

## Server steps (Stage 0; the researcher runs these)

Variables: `E=/home/calc08/projects/LLMPrivacy/fl_llm_m_attack_experiments/outputs`, `G=$E/grounded-fl-20260930`. Work in a separate worktree, `../fl_grounded_stage0`, in tmux.

The order:
1. `build`, with every exclusion source: the pilot's `cohort.json`, `splits.complete.json` and `launch.json`, the causal `stage-a-2` and `stage-b` splits, the matched-controls splits and the committed v3 copy. Paste back the printed report.
2. `calibrate-gate`.
3. `prepare datastore`, `resolve`, `run` and `analyze`.
4. `prepare timing`, `resolve`, `run --max-jobs 2` and `analyze`.
5. `prepare control`, `resolve`, `run --max-jobs 12` and `analyze --datastore-analysis`. If the analysis says `rerun_at_6_epochs`, prepare the one rerun with `--previous-analysis`.

Never edit a launch; on any refusal, prepare a new directory and record it here.
