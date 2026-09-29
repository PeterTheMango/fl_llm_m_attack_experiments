"""Mirabel significance calibration on benign SQuAD dev questions (see protocol.json).

Usage: HF_HUB_OFFLINE=1 python calibrate.py OUTPUT.json

CPU only: frozen MiniLM embeddings and the maintained rag.mirabel/rag.retrieve.
No LLM, generation, GPU or network. Writes one JSON with curves and checks.
"""
from hashlib import sha256
from pathlib import Path
import json
import math
import platform
import random
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "master_script/configs/research_data"
STUDY = DATA / "squad_rag_study.json"
SOURCE = DATA / "squad-dev-95aa6a52d5d6a735.json"
SOURCE_SHA256 = "95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEED, TOP_K, MAX_GOLD_LOSS = 20260929, 4, 0.05
GRID = [10 ** (-k / 4) for k in range(1, 49)]
SMOKE_HIDDEN = {"public": 181, "private": 172}


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def wilson(successes, n, z=1.959963984540054):
    if n == 0:
        return {"k": 0, "n": 0, "rate": None, "ci95": [None, None]}
    p = successes / n
    centre, half = (p + z * z / (2 * n)), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    scale = 1 + z * z / n
    return {"k": successes, "n": n, "rate": p, "ci95": [(centre - half) / scale, (centre + half) / scale]}


def log10_alpha_star(scores):
    """log10 of the significance above which rag.mirabel detects these scores."""
    import numpy as np
    index = int(np.argmax(scores))
    remaining = np.delete(scores, index)
    mean, std = float(remaining.mean()), float(remaining.std())
    scale = math.sqrt(2 * math.log(len(scores)))
    # detected iff max > mean + std*scale + c*std/scale, c = -log(-log1p(-alpha))
    c_star = (float(scores[index]) - mean - std * scale) * scale / std
    if c_star > 700:  # alpha_star = 1 - exp(-exp(-c)) ~ exp(-c) below float range
        return -c_star / math.log(10), index
    alpha = -math.expm1(-math.exp(-c_star)) if c_star > -700 else 1.0
    return (math.log10(alpha) if alpha > 0 else -c_star / math.log(10)), index


def auc(detectable, benign):
    """P(an attack query is more detectable, i.e. has smaller alpha_star, than a benign one)."""
    wins = 0.0
    ordered = sorted(benign)
    import bisect
    for value in detectable:
        lo, hi = bisect.bisect_left(ordered, value), bisect.bisect_right(ordered, value)
        wins += (len(ordered) - hi) + 0.5 * (hi - lo)
    return wins / (len(detectable) * len(ordered))


def main(output):
    import numpy as np
    from master_script.core import rag
    protocol = json.loads((HERE / "protocol.json").read_text())
    if digest(SOURCE) != SOURCE_SHA256:
        raise SystemExit("SQuAD dev source hash mismatch")
    study = json.loads(STUDY.read_text())
    source = json.loads(SOURCE.read_text())
    paragraphs = {}
    for article in source["data"]:
        for paragraph in article["paragraphs"]:
            paragraphs.setdefault(paragraph["context"].strip(), []).extend(paragraph["qas"])
    held_out = {q["id"].split("_", 1)[1] for q in study["utility_queries"]}

    corpora, benign = {}, []
    for name in ("public", "private"):
        documents = study[f"{name}_documents"]
        order = sorted(d["id"] for d in documents)
        random.Random(SEED).shuffle(order)
        calibration = set(order[:len(order) // 2])
        corpora[name] = documents
        for index, d in enumerate(documents):
            qas = paragraphs[d["text"]]  # KeyError here means the study and source disagree
            for q in qas:
                if q.get("answers") and q["answers"][0]["text"] in d["text"] and q["id"] not in held_out:
                    benign.append({"corpus": name, "question_id": q["id"], "question": q["question"],
                                   "gold": index, "split": "calibration" if d["id"] in calibration else "validation"})

    attacks = []
    for name, documents in corpora.items():
        members = {d["id"] for d in documents}
        index_of = {d["id"]: i for i, d in enumerate(documents)}
        for c in study["membership_candidates"]:
            member = c["id"] in members
            for form, text in (("yes_no", f'Does this: "{c["text"]}" appear in the context? Answer with Yes or No.'),
                               ("continuation", rag.continuation_question(c["text"])[0])):
                attacks.append({"corpus": name, "candidate_id": c["id"], "member": member, "form": form,
                                "candidate_index": index_of.get(c["id"]), "query": text})
    utility = [{"corpus": q["corpus"], "query": q["question"]} for q in study["utility_queries"]]

    vectors = {name: rag.embed([d["text"] for d in docs], EMBEDDING_MODEL) for name, docs in corpora.items()}
    queries = [r["question"] for r in benign] + [r["query"] for r in attacks] + [r["query"] for r in utility]
    encoded = rag.embed(queries, EMBEDDING_MODEL)
    for row, vector in zip(benign + attacks + utility, encoded):
        row["vector"] = vector

    mismatches = 0
    for row in benign + attacks:
        scores = vectors[row["corpus"]] @ row["vector"]
        row["log10_alpha_star"], row["top1"] = log10_alpha_star(scores)
        row["top4"] = np.argsort(-scores, kind="stable")[:TOP_K].tolist()
        for alpha in GRID[::6] + [0.05]:
            closed = alpha > 10 ** row["log10_alpha_star"]
            direct = rag.mirabel(scores, alpha)["detected"]
            mismatches += closed != direct and abs(math.log10(alpha) - row["log10_alpha_star"]) > 1e-9

    def detected(row, alpha):
        return math.log10(alpha) > row["log10_alpha_star"]

    reproduction = {}
    for name, docs in corpora.items():
        rows = [r for r in attacks if r["corpus"] == name and r["form"] == "yes_no"] + \
               [u for u in utility if u["corpus"] == name]
        hidden = sum(rag.retrieve(docs, vectors[name], r["vector"], TOP_K, 0.05, True)[1] for r in rows)
        removed = sum(detected(r, 0.05) and r["top1"] == r["candidate_index"]
                      for r in attacks if r["corpus"] == name and r["form"] == "yes_no" and r["member"])
        reproduction[name] = {"queries": len(rows), "hidden": hidden, "expected_hidden": SMOKE_HIDDEN[name],
                              "member_passages_removed": removed, "matches_smoke_run": hidden == SMOKE_HIDDEN[name]}

    def benign_rates(rows, alpha):
        return {"gold_loss": wilson(sum(detected(r, alpha) and r["top1"] == r["gold"] for r in rows), len(rows)),
                "detection": wilson(sum(detected(r, alpha) for r in rows), len(rows))}

    def attack_rates(rows, alpha):
        members = [r for r in rows if r["member"]]
        return {"member_passage_removed": wilson(sum(detected(r, alpha) and r["top1"] == r["candidate_index"] for r in members), len(members)),
                "nonmember_detection": wilson(sum(detected(r, alpha) for r in rows if not r["member"]), len(rows) - len(members))}

    split = {s: [r for r in benign if r["split"] == s] for s in ("calibration", "validation")}
    forms = {f: [r for r in attacks if r["form"] == f] for f in ("yes_no", "continuation")}
    curves = [{"alpha": a, "calibration": benign_rates(split["calibration"], a), "validation": benign_rates(split["validation"], a),
               **{f: attack_rates(rows, a) for f, rows in forms.items()}} for a in GRID]
    eligible = [c["alpha"] for c in curves if c["calibration"]["gold_loss"]["rate"] <= MAX_GOLD_LOSS]
    selected = max(eligible) if eligible else None

    report = None
    if selected is not None:
        v = benign_rates(split["validation"], selected)
        a = {f: attack_rates(rows, selected) for f, rows in forms.items()}
        per_corpus = {name: {"validation": benign_rates([r for r in split["validation"] if r["corpus"] == name], selected),
                             **{f: attack_rates([r for r in rows if r["corpus"] == name], selected) for f, rows in forms.items()}}
                      for name in corpora}
        viable = (v["gold_loss"]["ci95"][1] <= 0.10 and a["yes_no"]["member_passage_removed"]["rate"] >= 0.90)
        report = {"alpha_selected": selected, "validation": v, **a, "per_corpus": per_corpus,
                  "viable_by_protocol": viable}

    at_default = {"alpha": 0.05, "calibration": benign_rates(split["calibration"], 0.05),
                  "validation": benign_rates(split["validation"], 0.05),
                  **{f: attack_rates(rows, 0.05) for f, rows in forms.items()}}
    benign_log = [r["log10_alpha_star"] for r in split["validation"]]
    members = {f: [r["log10_alpha_star"] for r in rows if r["member"]] for f, rows in forms.items()}
    import torch
    import transformers
    snapshots = sorted(p.name for p in (Path.home() / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots").glob("*"))
    result = {
        "protocol_sha256": digest(HERE / "protocol.json"), "protocol": protocol,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "sources": {str(p.relative_to(ROOT)): digest(p) for p in (STUDY, SOURCE, HERE / "calibrate.py", ROOT / "master_script/core/rag.py")},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
                        "embedding_model": EMBEDDING_MODEL, "embedding_snapshots": snapshots, "device": "cpu"},
        "counts": {"benign": {s: len(rows) for s, rows in split.items()},
                   "benign_by_corpus": {n: sum(r["corpus"] == n for r in benign) for n in corpora},
                   "held_out_utility_questions": len(held_out), "attack_queries_per_form": len(forms["yes_no"])},
        "closed_form_vs_rag_mirabel_mismatches": mismatches,
        "reproduction_alpha_0_05": reproduction,
        "benign_gold_in_top4_ordinary": wilson(sum(r["gold"] in r["top4"] for r in benign), len(benign)),
        "at_alpha_0_05": at_default,
        "selection": {"rule": protocol["selection_rule"], "alpha_selected": selected}, "report": report,
        "detection_auc_vs_validation_benign": {f: auc(v, benign_log) for f, v in members.items()},
        "log10_alpha_star_quantiles": {
            name: {q: float(np.quantile(values, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
            for name, values in (("benign_validation", benign_log), ("member_yes_no", members["yes_no"]),
                                 ("member_continuation", members["continuation"]))},
        "curves": curves,
        "per_query": {"benign": [{k: r[k] for k in ("corpus", "question_id", "split", "gold", "top1", "log10_alpha_star")} for r in benign],
                      "attack": [{k: r[k] for k in ("corpus", "candidate_id", "form", "member", "candidate_index", "top1", "log10_alpha_star")} for r in attacks]},
    }
    Path(output).write_text(json.dumps(result, indent=1) + "\n")
    summary = {k: result[k] for k in ("counts", "closed_form_vs_rag_mirabel_mismatches", "reproduction_alpha_0_05",
                                        "benign_gold_in_top4_ordinary", "selection", "detection_auc_vs_validation_benign",
                                        "log10_alpha_star_quantiles")}
    summary["at_alpha_0_05"] = {k: ({m: x["rate"] for m, x in v.items()} if isinstance(v, dict) else v) for k, v in at_default.items()}
    summary["report"] = report
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
