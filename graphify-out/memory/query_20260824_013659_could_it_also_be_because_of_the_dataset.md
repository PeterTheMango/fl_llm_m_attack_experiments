---
type: "query"
date: "2026-08-24T01:36:59.818176+00:00"
question: "could it also be because of the dataset?"
contributor: "graphify"
source_nodes: ["build_client_partitions()", "run_hf_federated_finetune()", "base_metrics()"]
---

# Q: could it also be because of the dataset?

## Answer

Expanded from original query via graph vocab: [dataset, data, partition, partitions, client, clients, target, record, membership, sampling, rounds, trials, build, positive, negative, worlds]. Yes. The HF baseline uses a very small synthetic corpus in federation.py: four hand-written client partitions and generated two-record partitions. Client zero receives either the target record or a semantically related held-out clinic record. With 16 clients and eight sampled per round, the target client has only a 50 percent chance of participating in one round; if it is absent, the member record cannot affect training. Even when selected, its update is averaged with seven others. With only 16 balanced attack trials, the metric is also noisy. Diagnose by comparing ROC AUC and raw score distributions, checking whether client zero appears in federated_history, and seeing whether results improve from one to three to six rounds.

## Source Nodes

- build_client_partitions()
- run_hf_federated_finetune()
- base_metrics()