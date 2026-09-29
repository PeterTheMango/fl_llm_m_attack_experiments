# Context-exposure audit repair (`context_exposure_v2`) — local verification

CPU only. No model weights, inference, GPU jobs, private experiments, reserved-final access, remote changes or pushes. Historical artifacts are unchanged.

| Item | Value |
|---|---|
| Base commit | `02a568f949e9adb3dfe8d5738eb9c6df165a5684` (repair uncommitted) |
| Core fingerprint | `489525a1da38` → **`4b82f0dd4eb4`** (changed: `core/rag.py`) |
| `master_script/core/rag.py` SHA-256 | before `11e70366f387b1987362d6513a86437ee9ea146cb5f3a838fb2cedc98807f27f`; after `72c96057b88eb58bea08734cd9891cb8a58950ac5f1e8efae963afec85f7835f` |
| Exported tokenizer | `tokenizer.json` SHA-256 `3fd169731d2cbde95e10bf356d66d5997fd885dd8dbb6fb4684da3f23b2585d8` (Qwen2 byte-level BPE, NFC normalizer) |
| Pinned study | `master_script/configs/research_data/squad_rag_study.json`, SHA-256 `6fbf28e7…2a03` (byte-identical to the export's pinned copy) |
| Libraries | `tokenizers` 0.23.2 (reviewer's version) and 0.22.2 with `transformers` 5.14.1 |

Full hashes, per-control rows and the matrix are in the two `verification_*.json` files.

## Defect

The historical audit encoded the candidate alone. It searched for those IDs in a separate encoding of `"\n\n".join(contexts)`. Byte-level BPE merges a passage's final punctuation with the following separator: standalone `.` becomes `.ĊĊ`. As a result, a fully present passage was reported invisible.

## Repair

- **Shared preparation.** `_prepare_prompt` is the single, torch-free source of prefix, suffix, joined context text, context IDs and the retained count. `_prompt_tokens` (generation and answer NLL) and the audit both call it.
- **Offset alignment.** The audit re-encodes the same text with offsets and refuses to classify if those IDs differ from the generation IDs. It finds every exact code-point occurrence of the candidate in the joined context only.
- **Retained boundary.** A code point counts as retained only if no dropped token overlaps it. This covers characters split across byte tokens and trimmed-offset gaps.
- **Classification.** Each record is classified `complete` / `partial` / `absent` (with its cause) / `unavailable` (with its reason). Any complete occurrence wins.
- **Normalization.** It is handled explicitly: an edge-crossing rewrite, or equivalent text with no exact occurrence, yields `unavailable`, never `absent`.

Semantics and private metadata are documented in [`master_script/docs/amia_reference_rag.md`](../../master_script/docs/amia_reference_rag.md#context-exposure-audit-context_exposure_v2).

The result `context_audit` now contains `schema`, `status`, `exposure`, cause or reason, the three historical count fields (unchanged meaning), and occurrence counts. The historical `candidate_tokens_visible` field is **not** emitted, so it cannot be silently reinterpreted. Spans, hashes and tokenizer identity go only to the mode-0600 `private-audit/rag-answers.jsonl`. They are not written to result JSON or detector inputs.

## Boundary controls (200; candidate + `\n\n` + different private document)

| | Before (exact token span) | After (v2) |
|---|---:|---:|
| Complete exposure recognized | 1/200 | **200/200** |
| False invisibility | 199/200 | 0/200 |

The "before" column reproduces the retained review's `boundary_diagnostic.json` row for row, under both library versions.

Negative and truncation controls, built from the same 200 candidates, are reported separately:

| Control | Result |
|---|---|
| Candidate only in the question; context = other document | 200/200 `absent: not_in_context` |
| Candidate after another document, truncated at its first code point | 200/200 `absent: removed_by_truncation` |
| Truncated halfway through the candidate | 200/200 `partial` |

Old `true` always coincided with new `complete`. These are constructed controls; they do not verify every historical prompt.

## Generation token IDs unchanged

The historical `rag.py` was loaded directly from `git show 02a568f`. Its `_prompt_tokens` was compared with the repaired version on the exported tokenizer over **7,200 prompts**:

- 200 candidates, with candidate position rotated
- plain and chat format
- instruction defense off and on
- `max_context_tokens` 768 / 64 / 16
- three query types: membership with contexts, empty contexts, and utility with `answer_tokens=12`

Result: **0 token-ID mismatches.** SHA-256 over all repaired prompts is `ce3298da…23b4`.

The historical `context_tokens_before`, `context_tokens_used` and `context_truncated` fields agree in all 2,400 comparisons. Unit tests also compare a verbatim copy of the historical function over a 64-case matrix, including capacity-bound truncation and overflow errors.

## Tests

`test_results.txt` holds the actual output:

- New file, full dependencies: 84 passed.
- `tokenizers` 0.23.2 without torch: 83 passed, 1 skipped (torch tensor check).
- numpy/pytest only: 81 passed, 3 skipped (torch; exported tokenizer).
- Full suite: **666 passed**.

The deterministic tests need no model weights, GPU or tokenizer library. The new tests cover:

- punctuation+newline merge;
- start, middle and end positions;
- leading whitespace and separators;
- full, partial and complete-removal truncation, including a capacity-bound budget;
- multiple and overlapping occurrences;
- candidate present only in the question;
- absent candidates and empty context;
- code-point offsets, including a byte-split emoji at the cut;
- NFC normalization edge cases;
- six unavailable-information paths;
- generation/audit metadata consistency;
- unchanged generation IDs;
- keeping private data out of public output.

Python 3.10 was not installed locally. Syntax was checked with `ast.parse(..., feature_version=(3, 10))` and the tests ran on 3.11 and 3.13. The tests then passed on the server's Python 3.10.12 (below).

```bash
/opt/anaconda3/envs/peter_experiments_fl/bin/python -m pytest tests/test_rag_context_exposure.py -q
HF_HUB_OFFLINE=1 /opt/anaconda3/envs/peter_experiments_fl/bin/python outputs/context_exposure_repair_20260927/verify.py /tmp/verify-0.22.2.json
PYTHONPATH=/private/tmp/guard-review-tokenizers /opt/anaconda3/bin/python outputs/context_exposure_repair_20260927/verify.py /tmp/verify-0.23.2.json
```

On another machine, set `RAG_EXPOSURE_TOKENIZER=/path/to/federated_model/tokenizer.json`. Optionally set `RAG_EXPOSURE_RESULTS=/dir` with `*/*/artifacts/*/result.json` for the historical tabulation, which is skipped when none are found. The output records the host and its `tokenizers`/`transformers` versions.

## Remote server run (LCALC08, 2026-09-28)

The branch was checked out into a separate git worktree (`../fl_exposure_check`), leaving the experiment checkout untouched, and run CPU-only (`CUDA_VISIBLE_DEVICES=""`, `HF_HUB_OFFLINE=1`).

- **Environment:** Python 3.10.12, `tokenizers` 0.22.2, `transformers` 5.14.1, torch 2.13.0+cu126.
- **Tests:** the new tests and the full suite all passed, as reported by the user; exact counts were not transcribed.
- **Tokenizer:** the server's original pilot `tokenizer.json` has SHA-256 `3fd16973…85d8`, byte-identical to the export.
- **Boundary controls and generation:** identical to the local runs. 199/200 false invisibility before, 200/200 complete after, all negative controls correct; 0 mismatches over 7,200 prompts, with the same all-prompt hash `ce3298da…23b4`.
- **Historical tabulation:** run against the original server results (`outputs/guard-v4-pilot/results`), not the exported copies. Same totals: 2,376 / 24 / 2,400 / 12 / 12.

These values are transcribed from pasted stdout into `remote_verification_LCALC08.json`; the full JSON with per-control rows stays on the server. The library versions are those installed now. The versions used by the original pilot runs remain unrecorded, although the pilot's model export was written by `transformers` 5.14.1, which is consistent.

## Smoke run on real outputs (LCALC08, 2026-09-28)

This was the first real run on the repaired code, using `master_script/configs/smoke/rag_exposure_smoke.yaml` with all four RAG defenses and the overlap cells. `check_smoke.py` reported **All checks passed**. The tokenizer backend hash `e360e6f6…` equals the exported tokenizer's hash.

| Conditions (both corpora) | Members retrieved | Member exposure | Nonmember exposure |
|---|---:|---|---|
| ordinary, instruction | 100/100 | 100 `complete` | 100 `absent` |
| mirabel, mirabel_instruction | 0/100 | 100 `absent` | 100 `absent` |

- **No unavailable results.** No record in any condition came back `unavailable`.
- **Truncation.** It hit 3 public and 0 private ordinary queries without removing any member passage.
- **Overlap cells.** The target is `complete` only when it is in the datastore without Mirabel, and `absent` in every other cell.
- **Mirabel removed a top document from 181/210 (public) and 172/210 (private) queries.** That is 81 and 72 beyond the 100 member queries, so its utility cost needs checking.
- **One checkpoint.** These are mechanics results from one checkpoint. The `adv=1.000` comes from 2 trials and is not interpreted.

Utility and membership metrics from the same run (read-only print from `result.json`):

| Condition | Public F1 / EM | Public AUC | Private F1 / EM | Private AUC |
|---|---|---:|---|---:|
| ordinary | 0.316 / 0.20 | 0.595 | 0.475 / 0.30 | 0.575 |
| instruction | 0.154 / 0.10 | 0.675 | 0.480 / 0.30 | 0.695 |
| mirabel | 0.040 / 0.00 | 0.51 | 0.185 / 0.10 | 0.555 |
| mirabel + instruction | 0.014 / 0.00 | 0.49 | 0.200 / 0.10 | 0.54 |
| no retrieval | F1 0.064, EM 0.00 | | | |

What these single-checkpoint numbers suggest:

- **Mirabel** brings AUC near chance mostly by removing retrieval. Public F1 falls below the no-retrieval baseline, and answer NLL approaches the no-retrieval value.
- **The instruction defense** produced no refusals and a higher AUC than ordinary retrieval.
- **The attack itself is weak.** With every member passage completely in context, ordinary-retrieval AUC is only 0.58–0.60.

Limits: utility rests on 10 questions per corpus, AUC standard errors are roughly 0.04, and there is one seed and one round. These are leads for development, not defense evaluations.

Details are in `smoke_run_LCALC08.json`, transcribed from the pasted checker output and metrics print.

## Research impact

**Unaffected, and still valid.** The following do not depend on the visibility flag:

- Membership-answer parsing and the conditional AUC ranges: public 0.6078–0.6750, private 0.6400–0.6950.
- Utility arithmetic, ledger/accounting checks, and all detector results.
- `retrieved` and the three token-count fields.
- The 24 historical `candidate_tokens_visible=true` records remain valid evidence of presence: a matched span decodes to the candidate.

**Derived, not rewritten.** Historical flags stay as recorded. Under the assumptions listed in `historical_flags` (exact-document `retrieved`, correct `context_truncated`, lossless byte-level tokenizer) and preconditions checked on the pinned study (no separator in any candidate, NFC-stable text, no candidate is a substring of another document), the recorded fields logically imply:

- 2,376 member records flagged `false` were **complete** exposures: all 1,200 private-ordinary and 1,176 public-ordinary records, retrieved and untruncated.
- 2,400 nonmember records were **absent**.

The earlier puzzle — "retrieved, untruncated, yet invisible" — is therefore explained by the audit defect, not by missing context.

**Still unverifiable:**

- 12 public members that were retrieved but truncated; their position in the context is unknown.
- 12 overlap nonmember cells; the target text is not in the pinned study.
- Exact historical prompt token sequences.
- The remote `tokenizers` version.

Any claim that relied on a `false` flag as evidence of absence is withdrawn.

**Unresolved by this repair:**

- 8.33% legitimate rejection (the gate is ≤1%).
- The failed no-context utility floor.
- The absence of an enforced, matched privacy comparison.

A correct exposure audit shows what the model received. It does not show privacy protection or reduced inference.

## Limitations

- Controls are constructed from pinned text, not the unsaved historical contexts.
- The Qwen chat template was not exported. A ChatML stand-in exercises the chat path; the context is tokenized separately from the template in both versions.
- Exposure refers to the exact candidate string. Paraphrase and partial textual overlap are not measured.
- The fingerprint changed, so historical experiment scopes must not be resumed under this code. Experiment keys embed the fingerprint and will not match.
