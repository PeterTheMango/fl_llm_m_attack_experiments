"""Check a context_exposure_v2 smoke run's result JSON and private audit.

Usage: python check_smoke.py RUN_ROOT [SUMMARY.json]

RUN_ROOT is searched for artifacts/*/result.json with a private-audit/rag-answers.jsonl
beside it. Prints per-condition exposure counts and consistency checks. Reads only;
the optional summary holds counts and hashes, never answers, contexts or token IDs.
"""
from collections import Counter
from pathlib import Path
import json
import sys

SCHEMA = "context_exposure_v2"
CLASSES = ("complete", "partial", "absent", "unavailable")


def main(root, output=None):
    found = [p for p in sorted(Path(root).rglob("result.json")) if (p.parent / "private-audit/rag-answers.jsonl").is_file()]
    if not found:
        sys.exit(f"No result.json with private-audit/rag-answers.jsonl under {root}")
    report, failures = [], []

    def check(name, ok):
        if not ok:
            failures.append(name)
        return ok

    for path in found:
        result = json.loads(path.read_text())
        rows = [json.loads(line) for line in (path.parent / "private-audit/rag-answers.jsonl").read_text().splitlines()]
        audited = [r for r in rows if r["kind"] in ("membership", "overlap")]
        private = {(r.get("corpus"), r.get("defense"), r.get("candidate_sha256")): r["context_exposure"]
                   for r in audited if r["kind"] == "membership"}
        check("audit rows carry v2 detail", audited and all(r.get("context_exposure", {}).get("schema") == SCHEMA for r in audited))
        check("membership rows linked by candidate_index and query_sha256",
              all("candidate_index" in r and "query_sha256" in r for r in audited if r["kind"] == "membership"))
        tokenizers = {json.dumps(r["context_exposure"].get("tokenizer"), sort_keys=True) for r in audited}
        entry = {"result": str(path), "audit_rows": len(rows), "tokenizer_identities": len(tokenizers),
                 "tokenizer": json.loads(sorted(tokenizers)[0]) if tokenizers else None, "conditions": {}}
        check("one tokenizer identity per run", len(tokenizers) == 1)
        for evaluation in result.get("pipeline_evaluations", []):
            for name, condition in evaluation.get("rag_conditions", {}).items():
                corpus, defense = name.split("_", 1)
                trials = condition["membership_trials"]
                members = [t for t in trials if t["truth_member"]]
                exposure = lambda group: dict(Counter(t["context_audit"].get("exposure") for t in group))
                check(f"{name}: v2 summaries only", all(t["context_audit"].get("schema") == SCHEMA and
                                                        "candidate_tokens_visible" not in t["context_audit"] for t in trials))
                check(f"{name}: diagnostics counts match trials",
                      condition["diagnostics"]["member_context_exposure"] == {k: exposure(members).get(k, 0) for k in CLASSES})
                check(f"{name}: public summary equals private summary", all(
                    private.get((corpus, defense, t["candidate_sha256"]), {}).get("summary") == t["context_audit"] for t in trials))
                check(f"{name}: retrieved and untruncated members are complete", all(
                    t["context_audit"]["exposure"] == "complete" for t in members
                    if t["retrieved"] and not t["context_audit"]["context_truncated"]))
                check(f"{name}: unretrieved candidates are never complete",
                      not any(t["context_audit"]["exposure"] == "complete" for t in trials if not t["retrieved"]))
                entry["conditions"].setdefault(name, []).append({
                    "hidden_document_queries": condition["hidden_document_queries"],
                    "members": len(members), "member_retrieved": sum(t["retrieved"] for t in members),
                    "member_exposure": exposure(members),
                    "nonmember_exposure": exposure([t for t in trials if not t["truth_member"]]),
                    "unavailable_reasons": dict(Counter(t["context_audit"].get("unavailable_reason") for t in trials
                                                        if t["context_audit"]["exposure"] == "unavailable")),
                    "truncated": sum(t["context_audit"]["context_truncated"] for t in trials)})
            for cell in evaluation.get("membership_overlap_cells", []):
                entry.setdefault("overlap_cells", []).append({k: cell[k] for k in ("defense", "training_member", "datastore_member", "retrieved")}
                                                             | {"exposure": cell["context_audit"].get("exposure")})
        report.append(entry)
    for entry in report:
        print(entry["result"])
        print("  tokenizer:", {k: entry["tokenizer"][k] for k in ("backend_sha256", "tokenizers_version", "transformers_version")})
        for name, blocks in entry["conditions"].items():
            for b in blocks:
                print(f"  {name:28} members {b['member_retrieved']}/{b['members']} retrieved, hidden {b['hidden_document_queries']:3}, "
                      f"member {b['member_exposure']}, nonmember {b['nonmember_exposure']}, truncated {b['truncated']}"
                      + (f", unavailable {b['unavailable_reasons']}" if b["unavailable_reasons"] else ""))
        for cell in entry.get("overlap_cells", []):
            print("  overlap", cell)
    print("FAILED: " + "; ".join(failures) if failures else "All checks passed.")
    if output:
        Path(output).write_text(json.dumps({"runs": report, "failures": failures}, indent=1) + "\n")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main(*sys.argv[1:3])
