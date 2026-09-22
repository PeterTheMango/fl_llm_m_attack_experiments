"""CPU mechanics only: actual linear SGD federation through the production guard.

Not an LLM experiment, a Flower run, or privacy-effectiveness evidence. Exercises
heterogeneous data, later-round changes, large updates, durable abort/retry,
versioned feature traces and grouped detector fitting without downloading data.
"""
import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
from master_script.core.guard_runtime import (parse_guard, prepare_guard, read_guard_events,
                                            reject_training_round, record_training_round, PolicyAbortedRound)
from master_script.core.guard_features import parameter_features
from master_script.core.guard_detector import fit_detector


def run(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    policy = output / "policy.json"
    policy.write_text(json.dumps({"schema": "client_guard_v1", "observation_architecture": "causal_lm"}))
    settings = replace(parse_guard({"mode": "rules", "policy_file": str(policy.resolve()),
                                   "release_budget": 4, "diagnostic": True}, "<config>"),
                       runtime_directory=str(output / "client-owned"))
    initial = np.ones(4, dtype=np.float64) * .1
    runtime = prepare_guard(settings, "training:toy", [initial], rounds=4)
    rng = np.random.default_rng(117)
    clients = [(rng.normal(loc=.2*client, scale=.5+client*.2, size=(12+client*3, 4))) for client in range(4)]
    truth = np.array([.2, -.3, .4, .1])
    weights, history = initial.copy(), []
    for round_id in range(1, 5):
        tick = time.perf_counter(); updates, responses = [], []
        for client, x in enumerate(clients):
            validated = runtime.authorize([weights], client, round_id)
            if validated is None:
                raise RuntimeError("Unexpected legitimate request rejection")
            local = validated[0]
            y = x @ truth
            # Heterogeneous local step counts, including a legitimate large update.
            for _ in range(1+client*2):
                local -= .08 * x.T @ (x @ local-y)/len(x)
            updates.append(local)
            responses.append((None, SimpleNamespace(num_examples=len(x), metrics={
                "partition_id": client, "train_loss": float(np.mean((x @ local-y)**2))})))
        reject_training_round(responses, [])
        weights = np.average(updates, axis=0, weights=[len(x) for x in clients])
        loss = float(np.mean([(x @ weights-x @ truth) @ (x @ weights-x @ truth)/len(x) for x in clients]))
        history.append({"round": round_id, "loss": loss})
        record_training_round(runtime, round_id, responses, status="completed", seconds=time.perf_counter()-tick)
    # A new worker reconstructs the same scope; duplicate and budget cannot reset.
    restarted = prepare_guard(settings, "training:toy", [initial], rounds=5)
    assert restarted.authorize([weights], 0, 4) is None
    assert restarted.authorize([weights], 0, 5) is None
    rejected = [(None, SimpleNamespace(num_examples=0, metrics={"partition_id":0, "guard_decision":"rejected"}))]
    try:
        reject_training_round(rejected, [])
    except PolicyAbortedRound:
        record_training_round(restarted, 5, rejected, status="policy_aborted", seconds=0.)
    else:
        raise AssertionError("Partial round accepted")
    # Reproducible real parameter transformations, explicitly toy provenance.
    rows = []
    for split, first in (("train", 10), ("validation", 30), ("test", 50)):
        for seed in range(first, first+8):
            rng = np.random.default_rng(seed)
            prior = rng.normal(size=16)
            benign = prior + rng.normal(scale=.015, size=16)
            malicious = prior.copy()
            if split == "test":
                malicious[0] += 10  # held-out coordinate variant
            else:
                malicious += rng.normal(scale=3, size=16)
            for label, values in ((False, benign), (True, malicious)):
                rows.append({"group": sha256(f"toy-target-{seed}".encode()).hexdigest(), "split": split,
                             "malicious": label, "variant": ("coordinate" if split == "test" else "dense") if label else "benign",
                             "features": parameter_features([values], [prior])})
    detector = fit_detector(rows)
    report = {"schema": "guard_local_smoke_v2", "scope": "CPU linear-SGD mechanics; not LLM evidence",
              "completed_rounds": 4, "aborted_rounds": 1, "recovery": "stop_and_preserve_ledger",
              "history": history, "convergence_improved": history[-1]["loss"] < history[0]["loss"],
              "guard_events": read_guard_events(output / "client-owned"),
              "llm_or_rag_effectiveness": "not_evaluated", "torch_flower_cuda": "not_required_for_this_mechanics_check"}
    for name, data in (("report.json", report), ("toy-dataset.json", rows), ("toy-detector.json", detector)):
        (output / name).write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("output")
    args = p.parse_args(); report = run(args.output)
    print(f"CPU mechanics: {report['completed_rounds']} completed rounds, {report['aborted_rounds']} deliberate abort; no LLM claim")


if __name__ == "__main__":
    main()
