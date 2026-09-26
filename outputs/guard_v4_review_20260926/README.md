# Guard v4 delivery — September 26, 2026

**The old detector failed on architecture-preserving requests. The new features are a diagnostic candidate, not a demonstrated privacy defense.**

- Full diagnosis, definitions, limitations and exact remote commands: `master_script/docs/client_guard_v4_handoff.md`.
- Final evidence audit: `reproduced_evidence/review.json`. Reproduces 0/96 legitimate rejections, 8/8 probe detection, 0/8 causal detection, AUC .6667; verifies the historical Git fingerprint. Only one of 24 raw source results was supplied locally.
- Final verification and source/artifact hashes: `verification.json`.
- Current CPU/memory results: `benchmark_optimized.json` (256 MiB, serial full guard .819 s v1 / .857 s v2; ~2.36 MB extraction scratch).
- Maintained suite: `tests_final_verified.txt` — 554 passed, 11 skipped. Root-wide collection fails in archived vision tests without PyTorch.
- Locally validated configuration preview: `pilot-preview-final/` — nine development jobs and four final reservations; no actual target resolution or GPU execution here.
- Transfer patch: `implementation.patch` — verified to apply to the clean baseline and reproduce all 15 changed/new source, documentation and test files. Historical outputs are excluded.

New core fingerprint: `489525a1da38`. Features: `parameter_structure_v2`, explicitly opted in. V1 remains the default and its frozen detector reproduces the old decisions under current code.

The next action is remote CPU verification and cohort resolution. Future GPU execution is one job at a time with 20 GiB output and 4 GiB scratch headroom checks; keep the reserved final targets out of all development. All original utility, rejection and FL-overhead gates remain in force.

`evidence/`, `benchmark.json`, `pilot-preview/` and earlier test logs retain intermediate review work. The final files above supersede them. No historical detector was retuned, no GPU job was launched, and no historical artifact was deleted. Utility preservation, privacy effectiveness, generalization, population false rejection and matched FL overhead remain unestablished.
