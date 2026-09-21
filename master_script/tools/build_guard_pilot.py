"""Create the six-arm AMIA comparison once an independently fitted detector exists.

All arms use matched training settings. Noiseless guard arms are explicit
ablation diagnostics. Output also contains matched Reference conditions.
"""
import argparse
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import yaml
from master_script.core.guard_detector import load_detector
from master_script.core.yaml_config import load_config_doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detector"); parser.add_argument("output")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "configs"
    detector = Path(args.detector).resolve()
    digest = sha256(detector.read_bytes()).hexdigest()
    load_detector(detector, digest)
    doc = yaml.safe_load((root / "client_guard_privacy_pilot.yaml").read_text())
    rag = doc["pipeline"]["rag"]
    rag["study_file"] = str((root / rag["study_file"]).resolve())
    guard = {"mode": "rules_classifier", "policy_file": str(root / "guard/diagnostic_probe_policy.json"),
             "release_budget": 10, "diagnostic": True,
             "detector_file": str(detector), "detector_sha256": digest}
    variants = []
    for name, mode, noise in (("baseline", None, False), ("rules_only", "rules", False),
                              ("classifier_only", "classifier", False), ("noise_only", None, True),
                              ("rules_classifier", "rules_classifier", False),
                              ("rules_classifier_noise", "rules_classifier", True)):
        pipeline = {"condition": name, "defense": {"mechanism": "none"}, "rag": rag}
        if mode:
            g = {**guard, "mode": mode}
            if mode == "rules":
                g.pop("detector_file"); g.pop("detector_sha256")
            pipeline["client_guard"] = g
        variants.append({"base": {"observation_defense": "gaussian" if noise else "none",
                                   "observation_noise_multiplier": 1.0}, "pipeline": pipeline})
    doc["attacks"]["amia"]["variants"] = variants
    # Reference does not consume AMIA observation parameters.
    doc["attacks"]["reference"]["variants"] = [{"pipeline": deepcopy(v["pipeline"])} for v in (variants[0], variants[4])]
    output = Path(args.output).resolve()
    pairs = load_config_doc(doc, source=str(output))
    output.write_text(yaml.safe_dump(doc, sort_keys=False))
    print(f"Validated {len(pairs)} runs; detector SHA256 {digest}")


if __name__ == "__main__":
    main()
