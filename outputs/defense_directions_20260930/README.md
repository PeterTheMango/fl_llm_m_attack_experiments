# Defense directions for privacy-preserving RAG + FL LLMs (2026-09-30)

Goal, per the researcher: create a new defense, or improve an existing one, and test it in a setting where an FL-trained LLM answers with RAG. This note proposes directions for the researcher to choose among. Nothing here is implemented or run.

## What the evidence so far says

| Privacy channel (unit) | Attack in the harness | Status |
|---|---|---|
| Client gradient release (batch membership) | Causal gradient alignment | **Effective:** AUC 0.992 [0.981, 0.999] ([results](../causal_attack_validation_20260929/results.md)) |
| Trained weights (training-record membership) | Reference, paired worlds | 100% balanced accuracy in pilots. DP-SGD σ=2 brings it to chance; DP-FedAvg destroyed RAG F1 (0) |
| Retrieval datastore (document membership) | Yes/no RAG MIA | Weak attack (AUC ≈ 0.58–0.60). Mirabel removes the gold document for 81% of benign questions ([calibration](../mirabel_calibration_20260929/README.md)) |

What has been ruled out:

- **The request detector** does not separate attack from benign ([matched controls](../matched_reference_controls_20260930/results.md)).
- **The refusal instruction** was dropped: no refusals at smoke.
- **Mirabel** is a retrieval-removal control, not a defense.

**A structural finding.** FL clients train on closed-book records, `"Question: …\nAnswer: …"` (`datasets._format_squad` drops the SQuAD passage). RAG is attached only at inference, so FL and RAG are never connected during training. This has three consequences:

- **Clients memorize private facts.** The FL objective is to memorize private question-answer facts in the weights, which is the worst case for training-record privacy.
- **The objective cannot deliver utility.** A 0.5B model cannot learn extractive SQuAD answers closed-book. That explains the no-context F1 of 0.01–0.06, even for the pretrained model.
- **Utility lives in retrieval.** RAG F1 is 0.2–0.5, so the weights do not need to hold the facts.

## Directions

### A. Retrieval-grounded federated fine-tuning (recommended)

**Idea.** Move private knowledge out of the weights and into the retriever.

- Each client keeps its passages in its own datastore.
- Clients train on (retrieved passage + question → answer), with the loss on the **answer tokens only**. The passage is conditioning input and never an LM target.
- The model learns the *skill* of answering from context. The facts stay in an access-controlled store, where a retrieval-side protection (C) applies.

**Why it fits the theme.** It is a defense that exists only because RAG and FL are combined. The FL training objective is designed around the retriever.

**Hypotheses**, to pre-register:

- **H1.** At equal or better RAG F1, retrieval-grounded training lowers training-record membership (Reference) compared with the current closed-book training.
- **H2.** At a matched attack AUC, retrieval-grounded training plus DP-SGD needs less noise and keeps more RAG F1 than closed-book training plus DP-SGD. That would be a better privacy–utility frontier.
- **H3 (open).** The effect on the causal gradient attack. The attacker knows the candidate record and the public training objective, so an **adaptive** attacker must score with the same answer-only-given-context objective. Retrieval grounding may not reduce client-release leakage, and that is a valid outcome.

**Design sketch.**

- Four arms on the same fresh targets: closed-book, grounded, closed-book + DP-SGD, grounded + DP-SGD.
- Attacks:
  - Reference (training records);
  - causal gradient attack at the validated setting (cosine, 12 epochs), plus the adaptive-objective version;
  - the RAG datastore MIA.
- Utility: public and private RAG F1/EM (primary), plus answer NLL.
- Cohorts: new seed band. Stage A, Stage B, matched-controls and reserved-final targets are all excluded.

**Cost.** Code:

- a client formatter that keeps the passage;
- an answer-only loss mask;
- a per-client datastore;
- attack definitions updated to the new record form.

GPU: a pilot of about 4 arms × 5 targets at roughly 15–25 min each, so about 5–8 GPU-hours. A confirmation run would be sized from the pilot.

**Novelty (quick scan, not a systematic review).**

- Zeng et al. (ACL Findings 2024) observed that retrieval reduces an LLM's output of memorized training data at inference.
- EPD (arXiv 2512.03100) defends both fine-tuning and RAG against MIAs, but not in FL and not against active servers.
- FISGuard (arXiv 2608.27836) fixes a public representation subspace for FL LoRA, but only against an honest-but-curious server.

Nothing found uses the FL objective to shift private knowledge into a protected retriever and evaluates it against active FL attacks and RAG MIAs together. A proper related-work search is needed before any claim that it is "first".

### B. Calibrated retrieval gating (improve Mirabel)

Mirabel (arXiv 2505.22061, "detect-and-hide") fails here because a good factual question has a standout gold document, just like a membership probe does.

- **Improvement:** gate on what separates probes from questions. Membership probes quote long verbatim passage spans, whereas natural questions do not. Use n-gram overlap with the top document, with a threshold calibrated on benign questions, before withholding or paraphrasing.
- **Cost:** CPU-only calibration, like the Mirabel study, at hours of CPU and no GPU.
- **Limit:** paraphrasing attacks exist (for example "Riddle Me This", arXiv 2502.00306), so the gain is against verbatim attackers only. This is best as the retrieval-side half of A rather than as a stand-alone contribution.

### C. Three-channel privacy allocation (evaluation contribution)

- **Idea:** one system with a separate accountant per privacy unit:
  - DP-SGD on weights;
  - DP on retrieval, for example DP-RAG voting ([Grislain 2024](https://arxiv.org/abs/2412.19291)) or DP synthetic datastores ([DP-SynRAG](https://arxiv.org/abs/2510.06719));
  - observation noise on releases.
- **Measurement:** the privacy–utility frontier through RAG.
- **Novelty:** mostly in the joint FL + RAG evaluation, not in a new mechanism. It pairs naturally with A as the evaluation frame.

### Not recommended

- **Another request-classifier round.** The only exploratory signal (`mean_update_concentration`) is probably a batch-size artifact.
- **Request-direction projection** of released gradients. The attack's signal is the target's own gradient, which projecting out one request direction would not remove. A per-example clip plus noise is the known remedy, which is standard DP.

## Suggested path

1. Pre-register **A**, with **B** as its retrieval-side companion and **C**'s frontier as the reporting frame.
2. Replace the closed-book no-context endpoint with the utility protocol v4 already approved (parked item 1). Grounded training makes RAG F1 the natural primary utility.
3. Keep the validated attacks (causal cosine at 12 epochs, Reference, RAG MIA) as the fixed evaluation battery. Report every channel separately and never average them.

## Sources (quick scan)

- Zeng et al., "The Good and The Bad: Exploring Privacy Issues in RAG", ACL Findings 2024: https://aclanthology.org/2024.findings-acl.267.pdf
- Ensemble Privacy Defense (EPD): https://arxiv.org/pdf/2512.03100
- FISGuard: https://arxiv.org/pdf/2608.27836
- Mirabel ("Is This Query Too Close to Home?"): https://arxiv.org/pdf/2505.22061
- Riddle Me This (stealthy RAG MIA): https://arxiv.org/abs/2502.00306
- RAG with Differential Privacy (Grislain): https://arxiv.org/pdf/2412.19291
- DP-SynRAG: https://arxiv.org/pdf/2510.06719
- Private-RAG: https://arxiv.org/pdf/2511.07637
- SoK: Privacy Risks and Mitigations in RAG: https://arxiv.org/html/2601.03979v1
- Active MIA under LDP in FL (the AMI attack family): https://arxiv.org/abs/2302.12685
- MIA and defenses in FL, survey: https://arxiv.org/abs/2412.06157
