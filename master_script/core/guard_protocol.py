"""Explicit separate utility gates; never turn a missing/failed endpoint into a pass."""
import math


def validate_protocol(protocol):
    required = {"schema", "purpose", "endpoints", "minimum_f1", "minimum_relative_f1",
                "maximum_em_loss", "maximum_round_overhead", "maximum_benign_rejection", "interpretation"}
    if set(protocol) != required or protocol["schema"] != "guard_utility_protocol_v3":
        raise ValueError("Invalid utility protocol schema")
    if protocol["endpoints"] != ["no_retrieval", "public_ordinary", "private_ordinary"]:
        raise ValueError("Preserve all three separate required endpoints")
    for key in required - {"schema", "purpose", "endpoints", "interpretation"}:
        if type(protocol[key]) not in (float, int) or not math.isfinite(protocol[key]) or not 0 <= protocol[key] <= 1:
            raise ValueError(f"Invalid protocol threshold: {key}")
    return protocol


def utility_report(results, protocol):
    from master_script.tools.analyze_guard_study import pairing_key
    validate_protocol(protocol)
    baselines = {}
    for r in results:
        if r.get("pipeline", {}).get("condition") == "baseline" and r.get("status") == "complete":
            key = pairing_key(r)
            if key in baselines:
                raise ValueError("Ambiguous baseline")
            baselines[key] = r
    reports = []
    for r in results:
        if r.get("pipeline", {}).get("condition") == "baseline":
            continue
        base = baselines.get(pairing_key(r))
        report = {"run_id": r["run_id"], "baseline_run_id": base["run_id"] if base else None,
                  "endpoints": {}, "all_measured_utility_gates_pass": False}
        b_evals = {e["trial_id"]: e for e in (base or {}).get("pipeline_evaluations", [])}
        r_evals = {e["trial_id"]: e for e in r.get("pipeline_evaluations", [])}
        if len(b_evals) != len((base or {}).get("pipeline_evaluations", [])) or len(r_evals) != len(r.get("pipeline_evaluations", [])):
            raise ValueError("Duplicate evaluation identities")
        for endpoint in protocol["endpoints"]:
            pairs = []
            ids = set(b_evals) | set(r_evals)
            for trial in sorted(ids):
                def utility(e):
                    if endpoint == "no_retrieval":
                        return e.get("no_retrieval_utility", {})
                    return e.get("rag_conditions", {}).get(endpoint, {}).get("utility", {})
                bu, ru = utility(b_evals.get(trial, {})), utility(r_evals.get(trial, {}))
                valid = all(type(u.get(k)) in (float, int) and math.isfinite(u[k]) and 0 <= u[k] <= 1
                            for u in (bu, ru) for k in ("token_f1", "exact_match"))
                if not valid:
                    pairs.append({"trial_id": trial, "status": "unmeasured"})
                    continue
                floor = min(bu["token_f1"], ru["token_f1"]) >= protocol["minimum_f1"]
                relative = ru["token_f1"] >= protocol["minimum_relative_f1"] * bu["token_f1"]
                em = ru["exact_match"] >= bu["exact_match"] - protocol["maximum_em_loss"]
                pairs.append({"trial_id": trial, "baseline": bu, "defended": ru,
                              "absolute_floor_pass": floor, "relative_f1_pass": relative, "em_pass": em,
                              "status": "pass" if floor and relative and em else "fail"})
            operational = base is not None and r.get("status") == "complete"
            status = ("unmeasured" if not operational or not pairs or any(p["status"] == "unmeasured" for p in pairs)
                      else "fail" if any(p["status"] == "fail" for p in pairs) else "pass")
            report["endpoints"][endpoint] = {"status": status, "paired_evaluations": pairs}
        report["all_measured_utility_gates_pass"] = all(e["status"] == "pass" for e in report["endpoints"].values())
        reports.append(report)
    return {"protocol": protocol, "runs": reports,
            "interpretation": "Endpoint engineering checks only; no privacy score averaging or population claim. A protocol supplied after collection is a labeled reanalysis, not a predeclared study."}
