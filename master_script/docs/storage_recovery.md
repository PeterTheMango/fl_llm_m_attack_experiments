# Recovering experiments after disk exhaustion

`OSError: [Errno 28] No space left on device` during NumPy serialization means the artifact filesystem cannot accept the write. A missing Ray `gcs_server.err` is a secondary startup symptom; disk exhaustion is a plausible common cause, but the missing file alone does not prove it.

Firestore records results and status; the experiment still needs local space for model weights, pinned guard references, Ray files and its release ledger.

## Recover on the server after syncing this patch

1. Confirm the experiment has stopped. Check `df -h` and `df -i` for both the output directory and `/tmp`. Check any user quota if the filesystem has capacity but writes still fail.
2. Archive old queues to another filesystem, or preview removal of model files from an old finalized queue:

   ```sh
   python -m master_script.tools.cleanup_queue_models master_script/outputs/queues/20260917-174553-6a47bb77b00d
   ```

   Inspect the printed list. If these model weights and pinned references are no longer needed, run the same command with `--apply` to permanently delete them. Export/archive them first if future model-based analysis or exact replay needs them. The tool keeps results, logs, configurations, observations, trial checkpoints and release ledgers, and writes a cleanup record. Failed runs are excluded. A running or stale manifest is refused; do not change its status until you have verified its process has stopped. Do not clean a queue while any process is using it.
3. Leave enough room for the next run's model artifacts, plus Ray's temporary files. The new 1 GiB headroom check is a minimum guardrail, not an estimate of the full experiment's storage needs. A larger disk is preferable for a sustained study. Existing `--queue-output /path/on/larger-disk/queues` moves queue outputs. To move Ray temporary files as well, create a writable directory on that disk and set `RAY_TMPDIR=/path/on/larger-disk/ray` in the experiment's environment before launch.
4. Retry the failed conditions first. Keep Firestore enabled as usual. Source changes alter implementation fingerprints and therefore run IDs, so replaying the entire pilot may rerun previously completed conditions. Preserve old results as provenance; do not assume cache reuse across this patch.

Never delete or reset a release ledger to bypass an error. A previously truncated pinned reference must be investigated or restored from its verified original; it is not silently overwritten. Start a distinct experiment with a new identity if recovery of the original state is impossible.

## Implemented safeguards

- Check available artifact space before runs and estimated model/reference writes, including 1 GiB headroom; check Ray's temporary filesystem before startup.
- Publish guard snapshots atomically only after the full file is written and synced. Validate existing pins instead of replacing them.
- Stop a sweep on storage exhaustion rather than generating more failures. Mark the queue `storage_exhausted` and unstarted entries `not_run`.
- Reserve 1 MiB during a queue for emergency manifest finalization and write manifests atomically. This is best effort under concurrent disk use or filesystem failure.
- Preserve the original failure if secondary local or Firestore failure reporting fails.

No automatic deletion, change to privacy accounting, or change to the scientific conditions is performed. Storage checks cannot guarantee future capacity, and an interrupted model save may still leave partial model weights in a failed run's artifacts.
