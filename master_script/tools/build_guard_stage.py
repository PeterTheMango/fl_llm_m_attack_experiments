"""Resolve a bounded study stage without launching training or overwriting files."""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import yaml
from master_script.core.guard_detector import load_detector
from master_script.core.yaml_config import load_config_doc


ROOT = Path(__file__).resolve().parents[1] / "configs"


def build_stage(plan, *, detector=None, adaptive=False):
    if plan.get("schema") != "guard_stage_v2":
        raise ValueError("Unsupported stage definition")
    if plan["requires_detector"] and detector is None:
        raise ValueError("This stage requires a fitted real-trace detector; no placeholder artifact is allowed")
    doc = yaml.safe_load((ROOT / "client_guard_privacy_pilot.yaml").read_text())
    doc["defaults"].update(seed=plan["seed"], federated_rounds=plan["rounds"],
        num_clients=plan["clients"], clients_per_round=plan["clients"],
        calibration_nonmember_count=plan["calibration_nonmembers"],
        attack_trials=2*plan["targets"]*plan["queries_per_world"])
    rag = doc["pipeline"]["rag"]
    rag.update(study_file=str(ROOT / rag["study_file"]), evaluation_trials=2*plan["targets"],
               defenses=plan["retrieval_defenses"], membership_attack=plan["membership_attack"])
    pin = {}
    if detector:
        detector = Path(detector).resolve(); digest = sha256(detector.read_bytes()).hexdigest()
        artifact = load_detector(detector, digest)
        pin = {"detector_file": str(detector), "detector_sha256": digest, "feature_schema": artifact["schema"]}
    modes = {"rules_only": "rules", "classifier_only": "classifier", "rules_classifier": "rules_classifier",
             "rules_classifier_noise": "rules_classifier", "shadow": "shadow"}
    def pipeline(condition):
        value = {"condition": condition, "defense": {"mechanism": "dp_sgd" if condition == "training_dp" else "none"},
                 "rag": deepcopy(rag)}
        if condition in modes:
            mode = modes[condition]
            guard = {"mode": mode, "policy_file": str(ROOT / "guard/approved_training_policy.json"),
                     "release_budget": max(plan["rounds"], plan["queries_per_world"]), "diagnostic": True}
            if mode in ("classifier", "rules_classifier") or (mode == "shadow" and pin):
                guard.update(pin)
            value["client_guard"] = guard
        return value
    arms = []
    for variant in plan["variants"]:
        for condition in plan["conditions"]:
            protected = condition in ("noise_only", "rules_classifier_noise")
            arm = {"base": {"attack_variant": variant, "attack_targets": plan["targets"],
                            "probe_epochs": plan["probe_epochs"], "counterbalance_trials": True, "observation_defense": "gaussian" if protected else "none",
                            "observation_noise_multiplier": 1.0}, "pipeline": pipeline(condition)}
            # Same condition names across variants; analysis matches attack_variant.
            arms.append(arm)
    if adaptive:
        if not pin:
            raise ValueError("Adaptive stage needs a pinned detector")
        arm = deepcopy(next(a for a in arms if a["base"]["attack_variant"] == "causal_gradient_alignment"
                            and a["pipeline"]["condition"] == "rules_classifier_noise"))
        arm["base"]["adaptive_public_steps"] = 8
        arm["pipeline"]["condition"] = "adaptive_rules_classifier_noise"
        arms.append(arm)
    doc["attacks"]["amia"]["variants"] = arms
    doc["attacks"]["amia"]["sweep"] = {}
    doc["attacks"]["reference"]["sweep"] = {}
    doc["attacks"]["reference"]["base"]["attack_trials"] = 2*plan["targets"]
    doc["attacks"]["reference"]["variants"] = [{"pipeline": pipeline(c)} for c in plan["reference_conditions"]]
    return doc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["smoke", "collection", "validation", "confirmation"])
    p.add_argument("output", help="new output directory")
    p.add_argument("--detector"); p.add_argument("--adaptive", action="store_true")
    p.add_argument("--gpu", required=True, help="physical GPU index recorded in launch manifest")
    p.add_argument("--plan", help="Optional frozen custom stage definition")
    args = p.parse_args()
    if not args.gpu.isdigit():
        p.error("Select a numeric CUDA GPU; CPU fallback is not supported for these stages")
    path = Path(args.plan) if args.plan else ROOT / f"guard/stages/{args.stage}_v2.json"
    plan = json.loads(path.read_text())
    doc = build_stage(plan, detector=args.detector, adaptive=args.adaptive)
    output = Path(args.output).resolve()
    pairs = load_config_doc(doc, source=str(output / "experiments.yaml"))
    manifest = {"schema": "guard_launch_v2", "stage": plan, "stage_sha256": sha256(path.read_bytes()).hexdigest(),
                "gpu": args.gpu, "attacks": ["amia", "reference"], "expected_runs": len(pairs),
                "checkpoint_reuse": "disabled; each target/world is freshly trained, never shared as independent evidence",
                "output_root": str(output / "results"),
                "runs": [{"attack": spec.name, "condition": spec.pipeline.condition,
                          "variant": getattr(cfg, "attack_variant", None), "seed": cfg.seed,
                          "rounds": cfg.federated_rounds, "trials": cfg.attack_trials,
                          "targets": getattr(cfg, "attack_targets", cfg.attack_trials//2)} for cfg, spec in pairs],
                "runtime_estimate": {"source": "20260921 pilot", "historical_amia_seconds_per_target": 330,
                                     "historical_reference_seconds_per_target_pair": 734/2,
                                     "historical_round_seconds": [75, 81],
                                     "status": "not transferable to new causal probe; measure smoke stage first"}}
    output.mkdir(parents=True, exist_ok=False)
    (output / "experiments.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    (output / "launch.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Prepared {len(pairs)} runs. No training started. Review {output / 'launch.json'}")
    print(f"EXPERIMENT_GPU={args.gpu} python -m master_script.perform_experiments --queue '{output / 'experiments.yaml'}' --attack amia --attack reference --queue-output '{output / 'results'}' --no-firestore --no-charts")


if __name__ == "__main__":
    main()
