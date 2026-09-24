# Next step after the CPU benchmark

Median complete guard check: 13.5066 → 11.4514 seconds (15.2% less time).
Median feature extraction: 5.5488 → 3.5494 seconds (36.0% less time).
All four checks accepted the initial snapshot; all reported feature values agree exactly, and the feature source SHA256 matches the local patch. There are only two warm-cache checks per implementation, with zero parameter delta. This does not establish real round overhead or effectiveness against nonzero malicious requests.

Hashing still costs about 4.03 seconds and reference loading 2.86 seconds per check. At four sequential clients, the current 11.45 seconds per request would add roughly 46 seconds to a round if those CPU timings transfer. The earlier baseline rounds took about 55–75 seconds, so the 10% overhead target remains unlikely without more optimization. That is a projection, not a measured post-change FL result.

Copy `guard_v2_paired_check.py` into the remote repository root and run:

```bash
python guard_v2_paired_check.py --gpu 0 --run
```

This explicitly launches two fresh matched AMIA runs (architecture-preserving probe, baseline and rules-only; one diagnostic target, four clients, three rounds, four attack observations). Estimated total: 25–35 minutes based on the earlier 626s and 842s runs, subject to hardware/load. The purpose is to measure round overhead after the code change, check legitimate training and capture no-context answers. It is not an independent efficacy study or the larger collection sweep.

The script validates the queue, creates a unique output directory, runs with Firestore/charts disabled, analyzes both results and packages JSON/config plus only the no-context answer audit. It does not package model checkpoints or change the old results. Return the printed `paired-check-review.tar.gz`.

Locally the script passed configuration validation and CLI dry run with exactly two planned runs. No GPU job was launched here. To prepare and inspect without launching remotely, omit `--run`.
