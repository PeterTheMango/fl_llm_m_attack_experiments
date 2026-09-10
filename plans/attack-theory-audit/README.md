# Attack theory audit

The audit identifies 12 attack families and 35 ranked findings: 17 theoretical misalignments, 13 design defects, and 5 simplification opportunities. Two findings are critical: AMIA does not observe the required client loss update, and the runner bypasses SPV-MIA self-prompt reference training.

- [Full audit](AUDIT.md): paper baselines, implementation evidence, severity and proposed corrections.
- [WBC resolution](wbc-resolution.md): explicit appendix/author-config schedule, separate equation-schedule sensitivity run, and uniform per-size aggregation correction.
- [Remote verification plan](remote-verification.md): discriminating checks to run on the research server after approval.
- [Initial inventory](initial-inventory.md): configuration fields and historical WBC blocker report; superseded by the full audit.

The audit was approved for implementation. See the [implementation record](../../master_script/docs/theory_corrections.md) for the current disposition of every finding and server verification commands. The linked audit is a pre-change snapshot. No training, inference, tests or remote jobs were run during the correction pass.
