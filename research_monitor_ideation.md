# Research Monitoring Dashboard — Functional Ideation

**Status:** Ideation only. This document defines *what the platform should do*. It does not implement the dashboard, and it does not define the centralized experimentation framework — that is built separately by the user.

**Date:** 2026-06-24

---

## 1. Context and what is being monitored

The dashboard observes a set of experiments that currently live as **separate Jupyter notebooks in the `adaptations` folder**. Each notebook corresponds to **one attack being experimented with individually** — specifically, membership-inference attacks adapted to federated LLM fine-tuning (the reference pattern is `AMIA_adaptation.ipynb`).

From the existing notebook conventions (`adaptations/AGENTS.md`), the relevant facts the dashboard can rely on are:

- Each run is identified by a **stable config hash** (`run_id`) computed from the complete config.
- Experiments are often **factor sweeps**: one base config expanded over a grid (`model_id`, `num_clients`, `federated_rounds`, `local_epochs`, `client_lr`, `epsilon`, `seed`, etc.), where **each expanded config is a separate experiment/run**.
- Durable results are persisted to **Firebase Firestore** with a known document schema: `run_id`, `status`, `updated_at_unix`, `config`, `methodology`, `federated_history`, `metrics` (`tpr`, `tnr`, `adv`, `num_trials`), `attack_trials[]`, and `artifacts`.
- The metric of record is **`Adv = 0.5·TPR + 0.5·TNR`** (Nguyen et al. 2023, Eq. 3), with secondary diagnostics optional.
- Firestore is written **after measurement**; a cache check happens **before** compute, so a complete document means "already done, skip recompute."

The dashboard treats **the notebook/attack as the unit of experiment**, and **the expanded config (`run_id`) as the unit of run**.

### 1.1 Data sources (the core architectural decision)

The dashboard's results detail only matters **once an experiment is complete**, so the platform does not build a granular live-progress layer (per-FL-round / per-trial heartbeats). Instead it reads from two sources, one authoritative and one optional:

| Source | Provides | Nature |
|---|---|---|
| **Firebase Firestore (real-time listeners)** | The authoritative results record: which runs are complete/failed, their metrics, configs, history, and trials — pushed to the UI in real time via Firestore's real-time listeners as documents land. | Authoritative, persistent. Written once per run, after measurement. |
| **Optional run-state report from the central script** | The set of run_ids **currently being run**, and (if available) the **planned sweep manifest** (the full expanded config list). This is what lets the monitor show "which experiments are running now" and a sweep denominator. | Optional, coarse, transient. Consumed only if the central framework publishes it. |

**Why Firestore can't do it alone.** Firestore is written only after a run finishes measurement, so a mid-flight run is simply *absent*. Firestore can therefore show completed and failed states but, on its own, cannot say which experiment is running right now or how large a sweep is planned to be.

**The optional report fills exactly that gap.** The central experimentation script (the single framework the user builds separately) can tell the monitor (a) which run_ids are currently executing and (b) the planned sweep grid — **if and only if it chooses to publish that state; otherwise the monitor relies on Firestore alone** and simply shows completed/failed results without a "currently running" set or sweep denominators. The recommended channel is for the script to publish this coarse state **into Firestore itself** (e.g. a small `monitor_state` document and/or per-run `status: "running"` markers), so the dashboard keeps a **single data source** and reuses the same real-time listeners — no separate local channel, no per-round emission hooks. This transient state is not a results record; per §1.2 it is never used to resume or reconstruct an experiment.

> **Dependency on the separately-built framework (not designed here):** the only thing the framework optionally provides is this coarse run-state/manifest publication. This document specifies *what the dashboard consumes*; the publication hook itself is part of the experimentation framework the user will build, and the dashboard degrades gracefully to Firestore-only if it is absent.

### 1.2 Freshness, restart, and concurrency

- **Freshness.** The results views are driven by **Firestore real-time listeners**, so completed/failed runs and metric updates appear without manual polling. The optional "currently running" set refreshes from whatever cadence the central script publishes its run-state at.
- **Restart.** The dashboard holds **no durable state of its own.** On restart it rebuilds entirely by re-reading Firestore (and re-attaching listeners). Nothing in-progress is persisted by the dashboard.
- **In-progress runs are never resumed.** To preserve academic integrity, an experiment interrupted by a restart (of the dashboard, the VM, or the run itself) is **re-run from the start** by the framework rather than resumed from a partial state. The dashboard reflects this: an interrupted run simply has no complete Firestore document until it is re-run cleanly.
- **Concurrency.** The VM has **2 GPUs, so at most two experiments run at a time.** The live view is designed around this small, bounded "currently running" set (at most two), not an arbitrarily large pool.

---

## 2. Live monitoring view

**Purpose:** at a glance, show *which experiments are currently being run and where each one stands.*

### 2.1 Top-level "now" panel

A panel showing the **currently-running experiments** — at most two, given the 2-GPU limit — sourced from the central script's optional run-state report. For each running experiment, surface:

- **Which attack** — the source notebook/adaptation name (e.g. `AMIA_adaptation`), so a viewer immediately knows *which attack* is being experimented with.
- **Which run within that attack** — the `run_id` (config hash) plus the human-meaningful config that distinguishes it within a sweep (e.g. `model_id=distilgpt2, federated_rounds=4, epsilon=8, seed=11`).
- **Run timing** — start time and elapsed wall-clock.
- **(If the script reports it) coarse stage** — e.g. fine-tuning vs. attack vs. measurement. This is optional enrichment; the platform does not require granular intra-run progress, since detailed results are only consumed once the run completes and lands in Firestore.

If the central script does not publish run-state, this panel indicates that the currently-running set is unavailable and the dashboard operates from completed/failed Firestore results alone.

### 2.2 Sweep / queue awareness

Because experiments are commonly **grid sweeps**, the live view groups runs **by parent sweep / by attack** and shows the sweep's overall shape. The counts come from two places: **complete and failed** counts are read directly from Firestore, and the **total / pending** counts require the **planned sweep manifest from the central script's optional report**. When the manifest is available this reads as "this sweep is 12/48 done, 1 running, 2 failed, 33 pending"; when it is not, the dashboard shows only what Firestore knows — completed and failed counts — without a denominator or pending set.

### 2.3 Recently finished

A short rolling feed of runs that **just landed in Firestore as complete or failed** — surfaced in real time by the Firestore listeners — with the headline metric (`Adv`) for completed runs and the failure indication for failed ones. This bridges the live view and the results view.

### 2.4 What "currently being run" explicitly means here

A run is shown as *currently running* when the central script's run-state report lists its `run_id` as executing and no complete Firestore document exists for it yet. The "currently running" set is therefore **only as good as the optional report**: with the report, the live view is exact (and bounded to two by the GPU limit); without it, the dashboard cannot identify in-progress runs from Firestore alone and says so, rather than guessing.

### 2.5 Resource & log visibility (nice-to-have)

Not required for the core monitoring view, but valuable given that runs are LLM fine-tuning jobs on a 2-GPU VM: a panel showing **per-GPU utilization / memory** (two GPUs, so naturally one indicator each, mappable to the at-most-two running experiments) and **live stdout/stderr tails** for the running experiments. For these jobs, GPU/memory state and live logs are often the clearest early signal that a run is healthy versus silently out-of-memory or hung — complementing the coarse run-state report. Marked as a future enhancement, outside the core functional scope.

---

## 3. Results view

**Purpose:** inspect the results of each experiment, **filter between individual experiments**, and **view all experiments together as a whole.** This view is **Firestore-driven** (the durable, authoritative results layer).

### 3.1 All-experiments overview (the "whole" view)

A single sortable, filterable table/grid where **each row is a run** (`run_id`) across all attacks, reading directly from the Firestore documents. Columns surface:

- Attack / source notebook, `run_id`, `status`, `updated_at`.
- Key config dimensions (`model_id`, `num_clients`, `federated_rounds`, `local_epochs`, `client_lr`, `ldp_mechanism`, `epsilon`, `seed`, …).
- Primary metrics: **`Adv`, `TPR`, `TNR`**, `num_trials`, plus any secondary diagnostics present.

This grid is the "all experiments together as a whole" requirement. On top of it:

- **Sort / rank** by any metric (e.g. highest `Adv`).
- **Aggregate summaries** — e.g. mean/median/spread of `Adv` per attack, or per config factor — so the whole corpus can be read at once rather than row by row.
- **Comparison plots across runs** — e.g. `Adv` vs. `federated_rounds`, `Adv` vs. `epsilon`, or `Adv` vs. `num_clients`, with series grouped by attack or by another factor. This is the natural way to read a sweep's effect.

### 3.2 Filtering between experiments

Filtering operates on two natural axes that the data already supports:

1. **By experiment/attack** — narrow to one notebook/adaptation (e.g. only `AMIA_adaptation` runs).
2. **By config factor** — filter on any sweep variable (e.g. `epsilon ≤ 8`, `model_id = distilgpt2`, `seed = 11`), and by `status` (complete / failed) and time window.

Filters are composable, so a viewer can isolate exactly one experiment or any cross-section, and the aggregate summaries/plots in 3.1 recompute over the active filter — making "filter between individual experiments vs. view all together" a single continuous control rather than two separate screens.

### 3.3 Per-experiment detail page

Selecting a `run_id` opens a detail page rendering that Firestore document in full:

- **Header** — attack name, `run_id`, `status`, `updated_at`, and a complete config dump.
- **Methodology block** — the document's `methodology` (`paper_attack`, `llm_adaptation`, `metric_definition`), so the page is self-documenting about *what attack this is and how it was adapted to FL*.
- **Headline metrics** — `Adv`, `TPR`, `TNR`, `num_trials`, prominently.
- **Federated training history** — `federated_history` rendered as a per-round curve (loss/utility over rounds), giving the FL context behind the attack result.
- **Attack-trial detail** — the `attack_trials[]` records (`trial_id`, `truth_member`, `score`, `pred_member`) rendered as an auditable table and as a **score distribution split by true membership** and/or **ROC-style view**, since TPR/TNR derive directly from these trials. This makes each result verifiable rather than just a headline number.
- **Artifacts** — the `artifacts` paths / `gs://` URIs, shown as references (note that artifacts are often cleaned up post-persist, so these may be pointers to removed local paths or to Cloud Storage).

### 3.4 Privacy-direction reading (within the experiment's environment)

Because several sweep factors are privacy-relevant (`ldp_mechanism`, `epsilon`) and the headline metric `Adv` is an *attack-advantage* measure, the results view should make the **privacy direction legible**: lower attack `Adv` within a given environment ⇒ privacy *improved*; higher `Adv` ⇒ privacy *declined*. Concretely, when an experiment or filtered set varies a privacy knob (e.g. `epsilon`, or `ldp_mechanism ∈ {none, BitRand, OME}`) against an otherwise-matched baseline, the dashboard should present `Adv` as a function of that knob so the viewer can read whether privacy improved or declined and by how much. This stays a *presentation* concern — the dashboard reports the direction implied by the runs; it does not compute new privacy claims of its own.

---

## 4. Remote-access mechanism (external viewing via tunnel)

**Purpose:** make the dashboard viewable **from outside the VM.** The VM **has outbound internet access**, but **users cannot connect to it directly** (e.g. via SSH or to its dashboard port) unless they are on the internal network / intranet VPN (GlobalProtect by Palo Alto). The problem is therefore *inbound reachability for external viewers*, not connectivity in general.

### 4.1 The constraint and why a tunnel is the answer

The dashboard binds to a local port on the VM, and there is **no inbound public route** to that port — an external viewer who is not on the VPN cannot reach it. The tunnel works precisely because the constraint is one-directional: **the VM can initiate outbound connections even though nothing can reach it inbound.** A tunnel agent running *on the VM* dials *out* to a tunneling provider (**Cloudflare** or **ngrok**), and the provider issues a **public URL** that proxies traffic back down that already-open outbound connection to the local dashboard port. No inbound firewall change and no VPN membership are required for viewers, because all connection establishment originates from inside the VM.

> **Prerequisite (must hold for this to work):** the VM's outbound access must permit reaching the chosen provider's edge endpoints (the `cloudflared`/ngrok control and data endpoints). Since the VM already has outbound internet, this is expected to hold; if outbound egress is filtered, those endpoints must be allowlisted, otherwise the tunnel cannot establish.

### 4.2 Configuration: API key + code

The mechanism is configured with **an API key and a code/token**, matching both providers' model:

- **ngrok** — an **authtoken** (the API key/credential) authenticates the agent to the account; a tunnel is created against a configured port, optionally with a **reserved domain / tunnel code** so the URL is stable across restarts.
- **Cloudflare** — an **API token/key** plus a **named-tunnel credential/connector token** ("the code") authenticates `cloudflared` and binds the tunnel to a hostname; the agent runs as an outbound connector and Cloudflare surfaces the public hostname.

In both cases the dashboard takes the provider's API key and the tunnel code, **creates the tunnel against the local dashboard port, and surfaces the resulting external link** for viewing the research monitor from outside the VM. The two providers should be interchangeable behind one small "tunnel" configuration surface (provider, API key, code/token, target port), so the user can pick whichever their environment allows.

### 4.3 Functional behavior the dashboard exposes around the tunnel

- **Start/stop control** — bring the tunnel up or down on demand; the dashboard should not require a tunnel to function locally (it works for on-VPN/local viewers regardless).
- **Link surfacing** — once the tunnel is established, display (and make copyable) the **external URL**, plus which provider and which target port it maps to.
- **Tunnel status / health** — show whether the tunnel is connected, the public URL's reachability, and the last connection time, so a stale or dropped tunnel is visible rather than silently broken.
- **Lifecycle clarity** — indicate when a URL is ephemeral (regenerated each session) vs. stable (reserved domain / named tunnel via the configured code), so external viewers aren't handed a dead link.

### 4.4 Access-exposure note

This mechanism deliberately makes an otherwise VPN-only dashboard reachable on the public internet, so the surfaced link is effectively the access boundary. The dashboard should treat tunnel exposure as an explicit, user-initiated action (start/stop control above) and surface clearly when external access is *currently live* — keeping the operator aware that the monitor is reachable beyond the intranet while the tunnel is up.

---

## 5. Scope boundaries

In scope (this ideation): the **functional definition** of the live monitoring view, the results/filtering view, and the external tunneling access mechanism — sufficient as the functional foundation for later implementation.

Explicitly **out of scope** and left to the user:

- The **single centralized Python experimentation framework** and any consolidation of the per-notebook attacks into it.
- The **dashboard implementation** itself (no code here).
- The framework's **optional run-state / sweep-manifest publication** (noted in §1.1 as a dependency the dashboard consumes when present, and degrades gracefully without; not designed here).

The dashboard is designed to **degrade gracefully**: full functionality (currently-running set, sweep denominators) when the central script publishes run-state, and a still-useful Firestore-only mode (complete/failed results, filtering, aggregation) when it does not. The resource/log panel (§2.5) is flagged as a nice-to-have future enhancement. No other features, integrations, or assumptions are introduced beyond the monitoring view, the results/filtering view, the Cloudflare/ngrok tunnel, the Firestore + optional run-state data sources, and the stated VM/VPN deployment environment.
