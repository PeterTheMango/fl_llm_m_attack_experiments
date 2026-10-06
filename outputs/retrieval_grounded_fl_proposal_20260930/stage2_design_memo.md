# Stage 2 design memo: what the pilot tells us, and the choices before confirmation

*2026-10-05. For the researcher's review. Nothing in Stage 2 is prepared or run.*

Everything here comes from the Stage 1 pilot on V: 4 targets per arm and budget, plus the timing runs. The pilot is **tuning data and never evidence**. It is used only to size and shape Stage 2, as §8 allows. The protocol lets definitions, margins and rules be **tightened, never loosened**. Any change below that drops or redefines something needs your decision, is recorded in `status.md`, and in places needs a new pre-registration (marked ✱).

## 1. What the pilot shows

| Finding | Pilot numbers (estimate, 95% CI) | What it means for Stage 2 |
|---|---|---|
| Grounded training helps answers a lot at ε = ∞ | RG − CB-AO: **+0.27 F1** [0.20, 0.34]; P0 − RG: −0.23 [−0.28, −0.18] | H1's utility condition, H5 and H7 will be clearly measured |
| Public-only grounded training nearly matches private RG | RG-public − RG: −0.02 [−0.11, 0.07] | H5 is likely informative; the gain may come from learning the answering format |
| **DP at our budgets erases fine-tuning** | RG-public − RG(ε=64) +0.19; − RG(ε=16) +0.20; RG − CB-AO at ε = 16: 0.001 | DP-trained models sit near P0, so H6 and the DP frontier compare two almost pretrained models |
| The causal attack is at its ceiling | AUC ≈ 1 in RG and CB-AO; D = 0.00 [0.00, 0.00] | H2 cannot separate the arms; see §3 |
| The Reference attack is weak under answer-only loss | Tuning AUC 0.46–0.61 (RG, CB-AO) vs 1.0 for CB-LM | H1's AUC difference is noisy |
| The datastore attack works | NQ tuning AUC 0.82–0.90 (F1 scorer); H3 D = 0.19 [0.00, 0.47] | H3 is measurable |
| Retention is small | R_RG 0.05 [0.00, 0.15]; D_H4 0.08 [0.00, 0.25] | H4 needs enough targets to see a small effect |

**Why DP erases learning (plain terms).** Each client has 33 records and trains the whole 0.5-billion-parameter model with batches of 2. DP-SGD adds noise to every one of those parameters at every step.
- At σ = 1.91 or 5.45, the noise is thousands of times larger than the clipped gradient, so the model barely moves.
- A gentler accountant would not rescue this. The participation-based one (decision 5, rejected) would give σ ≈ 1.3 for ε = 16, and the noise would still dominate at batch size 2.
- Making DP-trained models learn would take a different training setup, for example LoRA, larger batches or more data per client. That is a different study ✱.

## 2. How precise Stage 2 would be

Projected 99.3% half-widths come from the pilot's spread, scaled as 1/√(targets). They rest on 4 pilot targets, so treat them as **rough, possibly off by 2×**.

There are two different questions, and they need very different precision:
- **Showing a difference** (outcomes 1–2) needs only the interval's lower end above 0. If the true effect is about 0.2, a half-width just under 0.2 is enough.
- **Showing no meaningful difference** ("equivalent") needs the whole interval inside ±0.05, which needs far more targets.

| Endpoint | Pilot estimate | Half-width at 20 targets | at 40 | at 60 | Targets to detect the pilot-sized effect* |
|---|---:|---:|---:|---:|---:|
| H1 Reference AUC difference | 0.19 | 0.24 | 0.17 | 0.14 | about 33 |
| H3 NQ AUC difference | 0.19 | 0.15 | 0.10 | 0.08 | about 12 (20 already enough) |
| H4 R_RG | 0.05 | 0.052 | 0.037 | 0.030 | about 22 |
| H4 D_H4 | 0.08 | 0.087 | 0.062 | 0.050 | about 22 |
| H1 utility, H5, H7 | 0.27 / −0.02 / −0.23 | 0.03–0.05 | smaller | smaller | already enough |
| H6 Reference AUC at ε = 16 (10 targets) | −0.06 | 0.17 | — | — | — (near-pretrained models) |

\*If the true effect equals the pilot estimate, which is itself very uncertain. Equivalence would need roughly 170 (H3) to 450 (H1) targets, which is out of reach.

## 3. The decisions

### A. How many targets at ε = ∞ (H1–H5, H7)

- **A1. 20 targets, option (b) as planned.** It detects large effects only. H1 is likely inconclusive, and H4 is borderline.
- **A2. 40 targets (recommended).** It roughly covers the pilot-sized effects for H1, H3 and H4, and costs about twice A1 at ε = ∞.
- **A3. 60 targets.** More margin for H1 and H4. Our 124 final targets allow it.

### B. The DP arms (H6 and the frontier)

- **B1. Keep as pre-registered:** 10 targets at ε = 64 and 16. This reports H6 honestly; the likely outcome is "no utility gain, because DP erases fine-tuning".
- **B2. Keep H6 at ε = 16 only, and drop the ε = 64 jobs (recommended).** ε = 64 is secondary (§9.2), and the pilot shows it behaves like ε = 16. H6 stays primary and unchanged.
- **B3. Downgrade H6 to secondary or drop it ✱.** This leaves six primaries at 99.2% and saves the DP cost, but it contradicts revision 4's "H6 is always kept".
- **B4. Redesign DP so models can learn ✱.** LoRA, larger batches or more data. This is a new study with its own pilot.

### C. H2 at the causal-attack ceiling

With the pre-registered definition, H2 would come out "no meaningful change". The reason is that both arms sit at AUC 1, not that their update leakage is equal.

**Pilot causal AUCs** (chosen template direction; mean over 4 targets, with each target's value in brackets):

| | RG, no noise | RG, release noise | CB-AO, no noise | CB-AO, release noise |
|---|---|---|---|---|
| ε = ∞ | 1.000 | **0.744** [0.77, 0.76, 0.78, 0.68] | 1.000 | **0.621** [0.79, 0.53, 0.60, 0.56] |
| ε = 64 | 0.935 | 0.633 | 0.999 | 0.698 |
| ε = 16 | 1.000 | 0.580 | 0.967 | 0.691 |

**What the table shows.**
- **Release noise takes the attack off the ceiling.** At ε = ∞ it suggests RG leaks *more* through updates than CB-AO: D = CB-AO − RG ≈ −0.12.
- **Projected precision for that D, from 4 targets and unpaired (target order not confirmed).** The 99.3% half-width would be about 0.075 at 20 targets, 0.061 at 30 and 0.053 at 40. At the pilot's size this would read "RG increases", and it would feed the cross-channel claim.
- **Side observation.** The record direction is *inverted* for RG at ε = ∞ (no-noise AUC 0.31); the chosen template direction is unaffected.

**Options.**
- **C1. Keep H2 as defined and report it with an explicit ceiling caveat.** No redesign. The release-noise result stays secondary and descriptive.
- **C2. Make the release-noise pass H2's primary endpoint ✱.** The no-noise pass becomes secondary. This is a definition change and needs a new pre-registration. **Caution:** choosing this endpoint *after* seeing that it shows a pilot difference is a selection risk. If you choose C2:
  - the pre-registration should state that the choice was made from the pilot;
  - it should keep the no-noise H2 as a reported secondary;
  - Stage 2 must use fresh final targets only, which it does by design.

  C2 requires keeping the release-noise pass (D1).

### D. The release-noise pass (secondary)

- **D1. Keep it in Stage 2.** It is 2/3 of each job's attack time (about 26 s per trial vs 12 s).
- **D2. Drop it in Stage 2 (recommended unless you choose C2).** It is secondary, and it is measured descriptively in the pilot already. This saves about 0.3 h per job.

## 4. Cost (GPU-hours, measured per-job times)

Per paired job with the chosen attack direction: ε = ∞ 0.76 h (0.46 h without the noise pass); ε = 16 1.18 h (0.88 h without it); ε = 64 about 1.1 h (not timed). P0 and RG-public add about 1.6 h.

| Scenario | Targets at ε = ∞ | DP arms | Noise pass | GPU-h |
|---|---:|---|---|---:|
| S0: option (b) as planned (A1, B1, D1) | 20 | ε = 64 and 16, 10 each | yes | ≈ 78 |
| S1: A2 + B2 + D1 | 40 | ε = 16, 10 targets | yes | ≈ 86 |
| **S2: A2 + B2 + D2 (recommended)** | 40 | ε = 16, 10 targets | no | **≈ 56** |
| S3: A3 + B2 + D2 | 60 | ε = 16, 10 targets | no | ≈ 74 |
| S5: 30 targets + B2 + C2 (keeps the noise pass) | 30 | ε = 16, 10 targets | yes | ≈ 71 |

The approved Stage 2 budget was 40–80 (option (b), provisional). S2, S3 and S5 fit; S1 does not.

**Data for decision C:** read on 2026-10-05; see §3 C.

### 4.1 Re-costed with `timing-5` (2026-10-06, core `8b031bbfdcc1`)

`timing-5` measured the two observation speed-ups. Every causal trial now takes about 5 s of wall time (about 2.1 s of victim compute), for the plain pass and the release-noise pass alike; `timing-4` took 12 s and 26 s.

| | ε = ∞ | ε = 16 |
|---|---:|---:|
| Measured job, both directions (`timing-5`) | 2128 s (0.59 h) | 3486 s (0.97 h) |
| Causal attack, both directions with both passes | 818 s | 831 s |
| **Per Stage 2 job, the chosen direction only** | **≈ 0.48 h** | **≈ 0.85 h** |

- ε = 64 is not timed; it is assumed to cost as much as ε = 16. P0 and RG-public stay at about 1.6 h.
- W0's measurements took 428 s at ε = ∞ (304 s in `timing-4`), so the ε = ∞ figure is on the cautious side; answer lengths vary with training non-determinism.

| Targets at ε = ∞ | B1: ε = 64 + 16 (10 each) | B2: ε = 16 only (10) |
|---:|---:|---:|
| 20 | ≈ 55 GPU-h | ≈ 38 GPU-h |
| 30 | ≈ 64 | ≈ 47 |
| 40 | ≈ 74 | ≈ 57 |
| 60 | ≈ 93 | **≈ 76** |

All scenarios keep the release-noise pass (C2 needs it). H2's release-noise D would have a 99.3% half-width of about 0.043 at 60 targets.

**Decisions (the researcher, 2026-10-06): A = 60 targets at ε = ∞ (A3) and B2 (ε = 16 only, 10 targets).** About 76 GPU-h, inside the approved 40–80 with about 4 h of headroom. They are frozen in `stage2_protocol.json`.

## 5. My recommendation (2026-10-05, before C and the re-cost)

It depends on decision C.

- **If you keep H2 as pre-registered (C1): S2.**
  - 40 targets at ε = ∞;
  - H6 at ε = 16 with 10 targets;
  - ε = 64 and the noise pass dropped;
  - about 56 GPU-h;
  - no new pre-registration.
- **If you want an informative H2 (C2): S5.**
  - 30 targets at ε = ∞, keeping the noise pass;
  - H6 at ε = 16 with 10 targets;
  - about 71 GPU-h;
  - a new pre-registration first.

  Its H1 and H4 precision is somewhat lower than S2's, because it has 30 targets rather than 40.

**I lean towards C1/S2.** It changes nothing after seeing the pilot. The release-noise H2 result can be reported as a secondary, descriptive finding from Stage 2 anyway, if you keep the noise pass. That costs about +30 GPU-h, which puts 40 targets over budget (S1); at 30 targets it fits. C2 is defensible, but it is a post-pilot endpoint switch toward the result the pilot showed.

## 6. What happens after you decide

1. I write the Stage 2 `protocol.json`: final cohort, seeds 9000–9099, the attacker choices frozen from the pilot, target counts, and your choices above. Definitions are tightened only.
2. I add `prepare final` and the confirmation `analyze` to the tool. `hypotheses()` is already tested, and the decision interval stays at 99.3% with seven primaries.
3. Then the usual sequence runs: you review, then prepare, resolve and run, one job per process.
4. The structured related-work search (Stage 0 item 1) is still open; it can be done in parallel and needs no GPU.
