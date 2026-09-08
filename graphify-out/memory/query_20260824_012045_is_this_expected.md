---
type: "query"
date: "2026-08-24T01:20:45.670737+00:00"
question: "Is this expected?"
contributor: "graphify"
source_nodes: ["Flower Simulation and Ray Stack", "run_hf_federated_finetune()", ".fit()", "build_client_partitions()"]
---

# Q: Is this expected?

## Answer

Expanded from original query via graph vocab: [flower, flwr, ray, simulation, client, clients, model, weights, aggregate, round, metrics, gpu, federation, fit, batch, local, server, resources]. The run is healthy: eight repeated weight loads are consistent with eight selected Flower clients, and aggregate_fit reports eight results with zero failures. Ray metrics-exporter connection messages only mean observability metrics are unavailable; they do not indicate an experiment failure. The federated execution path is run_hf_federated_finetune with client fit operations and client partitions.

## Source Nodes

- Flower Simulation and Ray Stack
- run_hf_federated_finetune()
- .fit()
- build_client_partitions()