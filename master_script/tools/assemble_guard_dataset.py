"""Assemble real request traces using a versioned, explicit target split manifest.

Manifest: {schema: guard_splits_v2, sources: [{path, sha256}],
           groups: {target_sha256: train|validation|test|attacker_validation|final}}.
The same target across seeds/clients/worlds always retains one role. Missing
identities, source hash changes, and mixed code revisions fail closed.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from master_script.core.guard_features import feature_vector, FEATURE_SCHEMA, feature_names


def assemble(manifest, base, schema=FEATURE_SCHEMA):
    feature_names(schema)
    if manifest.get("schema") != "guard_splits_v2" or set(manifest) != {"schema", "sources", "groups", "held_out_variants"}:
        raise ValueError("Invalid versioned split manifest")
    roles = {"train", "validation", "test", "attacker_validation", "final"}
    if not manifest["groups"] or any(r not in roles for r in manifest["groups"].values()):
        raise ValueError("Every target needs a declared data role")
    if not isinstance(manifest["held_out_variants"], list) or not manifest["held_out_variants"]:
        raise ValueError("Declare at least one held-out malicious variant")
    rows, provenance, fingerprints = [], [], set()
    excluded = {"attacker_validation": 0, "final": 0, "no_features": 0, "public_calibration": 0, "held_out_variant": 0}
    for source in manifest["sources"]:
        path = (Path(base) / source["path"]).resolve()
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != source["sha256"]:
            raise ValueError("Source checksum mismatch")
        result = json.loads(raw)
        fingerprints.add(result["implementation_fingerprint"])
        targets = {}
        for trial in result["attack_trials"]:
            seed = trial.get("target_seed", trial.get("seed"))
            target = trial.get("target_sha256", trial.get("candidate_sha256"))
            if seed is None or not target:
                raise ValueError("Trace source needs explicit target and seed provenance")
            if seed in targets and targets[seed] != target:
                raise ValueError("Ambiguous target/seed mapping")
            targets[seed] = target
        for index, event in enumerate(result.get("guard_events", [])):
            kind, seed, *_ = event["scope"].split(":")
            if kind == "public_calibration":
                excluded["public_calibration"] += 1; continue
            if kind not in ("training", "observation") or int(seed) not in targets:
                raise ValueError("Cannot label an unknown request origin")
            target = targets[int(seed)]
            if target not in manifest["groups"]:
                raise ValueError("Unassigned target in split manifest")
            split = manifest["groups"][target]
            if split in excluded:
                excluded[split] += 1; continue
            if event.get("features") is None:
                excluded["no_features"] += 1; continue
            if event.get("feature_schema", FEATURE_SCHEMA) != schema:
                raise ValueError("Mixed or unexpected feature schemas")
            feature_vector(event["features"], schema)
            variant = result.get("config", {}).get("attack_variant", "probe_head") if kind == "observation" else "legitimate_training"
            if variant in manifest["held_out_variants"] and split != "test":
                excluded["held_out_variant"] += 1; continue
            rows.append({"group": target, "split": split, "malicious": kind == "observation",
                         "features": event["features"], "variant": variant})
            provenance.append({"source_sha256": source["sha256"], "event_index": index, "target_sha256": target,
                               "seed": int(seed), "client_id": event["client_id"], "round": event["round_id"],
                               "model_revision": result["config"].get("model_revision"),
                               "dataset_revision": result["config"].get("dataset_revision")})
    if len(fingerprints) != 1:
        raise ValueError("Detector trace dataset must use one implementation revision")
    return rows, {"schema": "guard_dataset_v2", "manifest_sha256": sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                  "implementation_fingerprint": next(iter(fingerprints)), "provenance": provenance, "excluded": excluded,
                  "feature_schema": schema,
                  "feature_scope": "public_parameters_only; identities_labels_and_paths_are_provenance_not_features"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-schema", default=FEATURE_SCHEMA); p.add_argument("manifest"); p.add_argument("output")
    args = p.parse_args(); path = Path(args.manifest)
    rows, metadata = assemble(json.loads(path.read_text()), path.parent, args.feature_schema)
    metadata["dataset_sha256"] = sha256((json.dumps(rows, indent=2, allow_nan=False)+"\n").encode()).hexdigest()
    for filename, data in ((args.output, rows), (args.output + ".provenance.json", metadata)):
        with Path(filename).open("x") as stream:
            json.dump(data, stream, indent=2, allow_nan=False); stream.write("\n")


if __name__ == "__main__":
    main()
