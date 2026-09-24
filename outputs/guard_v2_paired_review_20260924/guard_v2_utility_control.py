"""Compare cached pretrained and saved fine-tuned weights on the same utility task.

Run from the repository root. Inference only: no training, retrieval, downloads,
checkpoint writes, or changes to the original prompt, scoring or utility gate.
The small output includes raw answers and is created with mode 0600.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
from uuid import uuid4


def prepare(run_dir, root):
    run_dir, root = Path(run_dir).resolve(), Path(root).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    candidates = []
    for entry in manifest["entries"]:
        name = entry["result_file"]
        if Path(name).name != name:
            raise ValueError("Invalid result filename")
        path = run_dir / name
        result = json.loads(path.read_text())
        if result["run_id"] != entry["run_id"]:
            raise ValueError("Manifest and result identities differ")
        if (entry["status"] == result["status"] == "complete"
                and result.get("pipeline", {}).get("condition") == "baseline"
                and result.get("attack_name") == "amia"):
            candidates.append((path, result))
    if len(candidates) != 1:
        raise ValueError("Expected exactly one completed AMIA baseline")
    path, result = candidates[0]
    if result["config"].get("attack_targets") != 1:
        raise ValueError("This diagnostic expects one saved baseline model")
    rag = result["pipeline"]["rag"]
    study_path = Path(rag["study_file"])
    if not study_path.is_file():
        study_path = root / "master_script/configs/research_data/squad_rag_study.json"
    raw = study_path.read_bytes()
    if sha256(raw).hexdigest() != rag["study_sha256"]:
        raise ValueError("Utility study differs from the completed run")
    from master_script.core.config import implementation_fingerprint
    if implementation_fingerprint() != result["implementation_fingerprint"]:
        raise ValueError("Core code differs from the paired run; reconcile it before re-scoring")
    return {"result": result, "result_path": path, "study": json.loads(raw),
            "settings": SimpleNamespace(**rag),
            "checkpoint": run_dir / "artifacts" / path.stem / "federated_model",
            "result_sha256": sha256(path.read_bytes()).hexdigest()}


def aggregate(rows):
    return {k: statistics.mean(r[k] for r in rows) for k in ("exact_match", "token_f1", "answer_nll")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.gpu.isdigit():
        parser.error("Use the allocated numeric physical GPU index")
    root = Path.cwd()
    if not (root / "master_script/core/rag.py").is_file():
        parser.error("Run from the repository root")
    sys.path.insert(0, str(root))
    plan = prepare(args.run_dir, root)
    result, checkpoint = plan["result"], plan["checkpoint"]
    questions = plan["study"]["utility_queries"]
    print(f"{len(questions)} questions x 2 models; identical original no-context prompt and scoring.", flush=True)
    print(f"Fine-tuned checkpoint: {checkpoint} (exists: {checkpoint.is_dir()})", flush=True)
    if args.dry_run:
        print("Configuration/provenance verified. No model loading or GPU execution.")
        return
    if not checkpoint.is_dir():
        parser.error("The saved baseline checkpoint is required; run on the original host before deleting it")
    # Pin before importing torch; forbid network downloads to avoid filling disk.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from master_script.core.rag import generate_answer, answer_nll, answer_utility
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; this tool does not silently fall back to CPU")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    start = time.perf_counter()
    evaluated = {}
    for name, identifier, revision in (
            ("pretrained", result["config"]["model_id"], result["config"]["model_revision"]),
            ("finetuned", str(checkpoint), None)):
        print(f"Evaluating {name} using cached/local weights...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(identifier, revision=revision,
                    torch_dtype=torch.float32, local_files_only=True).to("cuda:0").eval()
        model.config.pad_token_id = tokenizer.pad_token_id
        bundle = {"model": model, "tokenizer": tokenizer, "device": "cuda:0"}
        rows = []
        for q in questions:
            answer = generate_answer(bundle, q["question"], [], plan["settings"])
            rows.append({"id": q["id"], "question": q["question"], "expected": q["answer"],
                         "answer": answer, **answer_utility(answer, q["answer"]),
                         "answer_nll": answer_nll(bundle, q["question"], q["answer"], [], plan["settings"])})
        evaluated[name] = {"summary": aggregate(rows), "rows": rows}
        del bundle, model
        torch.cuda.empty_cache()
    recorded = result["pipeline_evaluations"][0]["no_retrieval_utility"]
    fine = evaluated["finetuned"]["summary"]
    reproduced = all(abs(fine[k] - recorded[k]) < 1e-10 for k in ("exact_match", "token_f1"))
    report = {"schema": "guard_utility_control_v1", "run_id": result["run_id"],
              "source_result_sha256": plan["result_sha256"], "source_implementation": result["implementation_fingerprint"],
              "study_sha256": result["pipeline"]["rag"]["study_sha256"],
              "control_script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "model_id": result["config"]["model_id"], "model_revision": result["config"]["model_revision"],
              "tokenizer_source": str(checkpoint), "physical_gpu": args.gpu,
              "generation": {"prompt": "original no-context RAG prompt", "prompt_format": plan["settings"].prompt_format,
                             "max_new_tokens": plan["settings"].max_new_tokens, "do_sample": False},
              "recorded_finetuned": recorded, "finetuned_em_f1_reproduced": reproduced,
              "finetuned_minus_pretrained": {k: fine[k] - evaluated["pretrained"]["summary"][k] for k in fine},
              "evaluated": evaluated, "seconds": time.perf_counter() - start,
              "limitations": ["Same small diagnostic question set; not a new independent utility benchmark",
                              "Identical saved tokenizer isolates checkpoint weights; no prompt tuning",
                              "Comparison does not isolate missing context from model knowledge",
                              "If saved-model EM/F1 fails reproduction, investigate before interpreting differences"],
              "audit_outputs_private": True}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = args.run_dir / f"utility-control-{stamp}-{uuid4().hex[:8]}.json"
    with os.fdopen(os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v["summary"] for k, v in evaluated.items()}, indent=2))
    print(f"Saved-model EM/F1 reproduced: {reproduced}")
    print(f"Attach this file: {output}")


if __name__ == "__main__":
    main()
