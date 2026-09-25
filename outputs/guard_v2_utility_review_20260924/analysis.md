# Pretrained utility control — 24 September 2026

The original pretrained checkpoint already performs poorly on this no-context question set. Fine-tuning slightly lowers decoded token F1 while improving the reported teacher-forced answer NLL. The earlier baseline/rules comparison produced identical answers, so this evidence does not attribute the poor answers to the guard. Neither checkpoint meets the original 0.10 no-context F1 screening floor.

| Metric | Pretrained | Fine-tuned |
|---|---:|---:|
| Token F1 | 0.025714 (2.57%) | 0.017835 (1.78%) |
| Exact match | 0 / 20 | 0 / 20 |
| Mean answer NLL, lower is better under this scoring convention | 9.020842 | 6.146161 |

The F1 change is **−0.788 percentage points**, or 69.4% of the pretrained score. This retention ratio compares model weights before and after training; it is **not** the defense-versus-matched-baseline utility gate. In the separate matched guard experiment, F1 retention was 100%, with both arms below the absolute floor. Four questions increase in token F1, six decrease and ten remain unchanged. Token overlap can reward incidental words without a correct answer. These counts should not be relabeled as numbers of semantically improved answers.

Reported per-question NLL improves on 19 of 20 questions. This measures probability assigned to the reference answer under teacher forcing, including the existing answer-token formatting convention. It does not establish that the model generates the reference answer. Here zero exact matches and the raw generations make that distinction concrete. Neither more training epochs nor a claim of broad catastrophic forgetting is justified by this small, mixed diagnostic.

The model-weight comparison held the questions, answer references, tokenizer, deterministic generation, 64-token limit and original empty-context prompt fixed. That prompt still asks the model to use context. Several questions depend on an absent passage; some answers contain relevant material but are penalized for extra words. This comparison therefore does not isolate missing context, general model knowledge and answer-format effects. It establishes low starting performance under the current protocol and a small absolute F1 decrease after training on these reused questions.

Verification:

- The source result SHA256 and study SHA256 match the received paired run and local pinned study.
- All 20 question IDs, question strings and answer references match the study.
- Exact match and token F1 were recomputed from the supplied generated answers. Their means reproduce the summary.
- Every fine-tuned answer exactly reproduces its earlier unguarded baseline answer; that earlier baseline also exactly matched the rules-only answers.
- The control-script hash equals the local script with one additional trailing newline. There is no unexplained substantive script mismatch.
- NLL means were recomputed from the supplied per-question values; forward passes were not rerun locally. The remote diagnostic reports approximately 48 seconds.

The original study gates remain unchanged and unmet. The latest matched pair still has 70.8% median round overhead, accepts all four tested architecture-preserving attack observations and has no learned detector. The utility control neither establishes privacy nor resolves those blockers. The measured RAG utilities remain separate: public/private token F1 is approximately 20.94%/40.67%, with identical baseline and rules-only answer scores.

This closes the immediate question of whether the low no-context starting score was created by the guard: no guard-specific difference was observed. Another repetition of the same smoke target is not needed for that question. The next engineering priorities are the remaining full-model hashing/reference-reading cost and storage-bounded trace collection across independent targets for the learned detector. Before a new effectiveness study, explicitly specify which task each utility gate measures. A passage-grounded QA protocol can be added as a separately declared endpoint; it must not retroactively turn the failed no-context screen into a pass or choose scoring rules from these final-looking results. Preserve the current failed gates and label this target/question set diagnostic.

No source code, model weights or historical results were changed during this review. No new GPU experiment was launched. The received control and a compact verified summary are saved alongside this report.
