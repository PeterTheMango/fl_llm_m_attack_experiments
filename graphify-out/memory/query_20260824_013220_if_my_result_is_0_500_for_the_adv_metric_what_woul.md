---
type: "query"
date: "2026-08-24T01:32:20.078584+00:00"
question: "if my result is 0.500 for the ADV metric what would that mean?"
contributor: "graphify"
source_nodes: ["base_metrics()", "summarize()"]
---

# Q: if my result is 0.500 for the ADV metric what would that mean?

## Answer

Expanded from original query via graph vocab: [adv, advantage, membership, member, metric, summarize, accuracy, threshold, tpr, fpr, positive, negative]. In this project Adv is balanced accuracy, defined as 0.5 times TPR plus 0.5 times TNR. Adv 0.500 is the random-guessing baseline at the configured threshold and corresponds to conventional membership advantage TPR minus FPR equal to zero. It does not prove absence of leakage because a miscalibrated threshold can produce 0.500 even when raw scores separate members and nonmembers; inspect TPR, TNR, and threshold-independent ROC AUC. With one balanced 16-trial run, 0.500 usually means 8 correct classifications out of 16, though it can arise from asymmetric behavior such as TPR 1 and TNR 0.

## Source Nodes

- base_metrics()
- summarize()