"""Export public feature traces with independently supplied group/label manifests.

Usage: python -m master_script.tools.build_guard_dataset EVENTS.json LABELS.json OUTPUT.json
Labels is a list of {event_index, group, split, malicious, variant}. It belongs to
research tooling, never the client detector. Pair/world grouping must be assigned
before fitting, with the same group across related runs and attack variants.
"""
import argparse
import json
from pathlib import Path
from master_script.core.guard_features import feature_vector


def build_dataset(events, labels):
    rows, seen = [], set()
    for label in labels:
        if set(label) != {"event_index", "group", "split", "malicious", "variant"}:
            raise ValueError("Invalid label manifest")
        index = label["event_index"]
        if type(index) is not int or not 0 <= index < len(events) or index in seen:
            raise ValueError("Invalid or duplicate event index")
        seen.add(index)
        features = events[index]["features"]
        feature_vector(features)  # Structural refusals have no behavioral features.
        rows.append({k: v for k, v in label.items() if k != "event_index"} | {"features": features})
    if not rows:
        raise ValueError("No usable public request traces")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events"); parser.add_argument("labels"); parser.add_argument("output")
    args = parser.parse_args()
    events = json.loads(Path(args.events).read_text())
    if isinstance(events, dict):
        events = events["guard_events"]
    rows = build_dataset(events, json.loads(Path(args.labels).read_text()))
    Path(args.output).write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
