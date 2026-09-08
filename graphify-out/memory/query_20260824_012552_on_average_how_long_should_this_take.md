---
type: "query"
date: "2026-08-24T01:25:52.235683+00:00"
question: "on average how long should this take?"
contributor: "graphify"
source_nodes: ["run_attack_trials()", "run_hf_federated_finetune()", "run_sweep()", "Matched Positive and Negative Membership Worlds"]
---

# Q: on average how long should this take?

## Answer

Expanded from original query via graph vocab: [attack, trials, rounds, seed, sweep, runner, federation, fit, client, model, positive, negative]. For the full zlib-only pythia-410m grid, estimate 75 to 120 minutes on the observed A100 MIG, with 2 to 3 hours as a conservative upper bound. The configuration creates 9 experiment cells from 3 round settings and 3 seeds. Each cell runs 16 trials, and each trial performs real HF federated fine-tuning. Total work is 3 times 16 times the sum of 1, 3, and 6 rounds times 8 clients, or 3840 client fits across 480 federated rounds and 144 Flower simulations. The lighter proposed grid should take roughly 10 to 25 minutes.

## Source Nodes

- run_attack_trials()
- run_hf_federated_finetune()
- run_sweep()
- Matched Positive and Negative Membership Worlds