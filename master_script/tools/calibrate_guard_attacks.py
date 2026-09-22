"""Freeze independent defended-validation score direction/threshold; apply to final data.

Supports fresh-batch AMIA, Reference training membership, datastore membership,
and separate decision/timing transcript attacks. Does not fit using final labels.
"""
import argparse
import json
from pathlib import Path
from master_script.core.guard_statistics import (fit_attack_calibration, apply_attack_calibration,
                                                privacy_report, calibration_binding, fit_transcript_calibration, apply_transcript_calibration)


def extract(result, rag_condition=None):
    if rag_condition:
        rows = [r for e in result.get("pipeline_evaluations", [])
                for r in e.get("rag_conditions", {}).get(rag_condition, {}).get("membership_trials", [])]
        if any("candidate_sha256" not in r for r in rows):
            raise ValueError("Historical RAG rows lack document identities; cannot calibrate them as independent data")
        unit = "retrieval_document"
    else:
        rows = result["attack_trials"]
        unit = "private_client_batch" if result["attack_name"] == "amia" else "training_record"
    return rows, calibration_binding(result, unit, rag_condition)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["fit", "evaluate"])
    p.add_argument("results", nargs="+"); p.add_argument("--output", required=True)
    p.add_argument("--calibration"); p.add_argument("--rag-condition")
    p.add_argument("--condition"); p.add_argument("--attack", choices=["amia","reference"])
    p.add_argument("--variant", choices=["probe_head","causal_gradient_alignment"])
    p.add_argument("--channel", choices=["score", "decision", "response_seconds", "transcript"], default="score")
    p.add_argument("--group-manifest", help="JSON mapping document/target SHA256 to attacker_validation or final; required to split repeated RAG queries")
    p.add_argument("--query-budget", type=int, default=1); p.add_argument("--fpr", type=float, default=.05)
    args = p.parse_args()
    if args.query_budget <= 0:
        p.error("query budget must be positive")
    rows, bindings = [], []
    for path in args.results:
        result = json.loads(Path(path).read_text())
        if args.condition and result.get("pipeline", {}).get("condition") != args.condition:
            continue
        if args.attack and result.get("attack_name") != args.attack:
            continue
        if args.variant and result.get("config", {}).get("attack_variant", "probe_head") != args.variant:
            continue
        if result.get("status") != "complete":
            raise ValueError("Cannot fit/evaluate operationally incomplete results")
        part, binding = extract(result, args.rag_condition)
        rows.extend(part); bindings.append(binding)
    if not bindings:
        raise ValueError("No matching results")
    if any(b != bindings[0] for b in bindings):
        raise ValueError("Do not combine changed implementations, defenses or configurations")
    if args.group_manifest:
        from master_script.core.guard_statistics import group_id
        roles = json.loads(Path(args.group_manifest).read_text())
        role = "attacker_validation" if args.mode == "fit" else "final"
        if any(group_id(r) not in roles for r in rows):
            raise ValueError("Group manifest must assign every input target/document")
        rows = [r for r in rows if roles[group_id(r)] == role]
    if args.mode == "fit":
        if args.channel == "transcript":
            output = fit_transcript_calibration(rows, target_fpr=args.fpr, query_budget=args.query_budget, binding=bindings[0])
        else:
            output = fit_attack_calibration(rows, target_fpr=args.fpr, channel=args.channel,
                                            query_budget=args.query_budget, binding=bindings[0])
    else:
        if not args.calibration:
            p.error("evaluate requires --calibration")
        frozen = json.loads(Path(args.calibration).read_text())
        apply = apply_transcript_calibration if frozen.get("channel") == "transcript" else apply_attack_calibration
        worlds = apply(rows, frozen, binding=bindings[0])
        output = {"calibration": frozen, "worlds": worlds, "metrics": privacy_report(worlds)}
    with Path(args.output).open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False); stream.write("\n")


if __name__ == "__main__":
    main()
