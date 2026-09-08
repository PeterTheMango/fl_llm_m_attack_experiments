---
type: "query"
date: "2026-08-23T23:59:41.078103+00:00"
question: "What command runs one baseline attack so the error can be copied?"
contributor: "graphify"
source_nodes: ["perform_experiments.py", "baseline_pythia410m.yaml"]
---

# Q: What command runs one baseline attack so the error can be copied?

## Answer

From the repository root, run python -m master_script.perform_experiments --config master_script/configs/baseline_pythia410m.yaml --attack zlib --max-parallel 1 --no-charts --log-level DEBUG. This isolates execution to one run at a time while preserving the baseline configuration and emitting a session log.

## Source Nodes

- perform_experiments.py
- baseline_pythia410m.yaml