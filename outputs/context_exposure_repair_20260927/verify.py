"""Local CPU-only verification of the context_exposure_v2 repair.

Usage: python verify.py OUTPUT.json

Optional environment: RAG_EXPOSURE_TOKENIZER=/path/to/tokenizer.json (default: the
guard-v4 export) and RAG_EXPOSURE_RESULTS=/dir containing */*/artifacts/*/result.json
(default: the export; the historical tabulation is skipped when none are found).

Needs the exported guard-v4 tokenizer and the tracked pinned study. With
torch+transformers it also compares historical (commit 02a568f) and repaired
generation token IDs; with only ``tokenizers`` it runs the boundary controls.
No model weights, inference, GPU or network access are used.
"""
from hashlib import sha256
from pathlib import Path
from types import ModuleType, SimpleNamespace
import json
import os
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HISTORICAL_COMMIT = "02a568f949e9adb3dfe8d5738eb9c6df165a5684"
HISTORICAL_FINGERPRINT = "489525a1da38"
STUDY = ROOT / "master_script/configs/research_data/squad_rag_study.json"
EXPORT = Path(os.environ.get("RAG_EXPOSURE_RESULTS",
                             ROOT / "outputs/guard_v4_retained_review_20260927/verified/inputs/results"))
TOKENIZER = Path(os.environ["RAG_EXPOSURE_TOKENIZER"]) if os.environ.get("RAG_EXPOSURE_TOKENIZER") else sorted(
    (ROOT / "outputs/guard_v4_retained_review_20260927/verified/inputs/results").glob(
        "job-000/*/artifacts/*/federated_model/tokenizer.json"))[0]
# The export has no chat template. This ChatML stand-in only shapes the lead/tail
# text, which is tokenized separately from the context in both implementations.
STANDIN_CHAT_TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
                         "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def digest(data):
    return sha256(data).hexdigest()


def shown(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def load_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError:
        import tokenizers

        class Wrapped:  # Same calls the audit makes on a fast HF tokenizer.
            name_or_path = str(TOKENIZER)
            backend_tokenizer = tokenizers.Tokenizer.from_file(str(TOKENIZER))

            def encode(self, text, add_special_tokens=False):
                return self.backend_tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

            def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
                e = self.backend_tokenizer.encode(text, add_special_tokens=add_special_tokens)
                return {"input_ids": e.ids, "offset_mapping": e.offsets}

        return Wrapped(), "tokenizers.Tokenizer (wrapped)"
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER.parent))
    tokenizer.chat_template = STANDIN_CHAT_TEMPLATE
    return tokenizer, "transformers.AutoTokenizer"


def historical_rag():
    source = git("show", f"{HISTORICAL_COMMIT}:master_script/core/rag.py")
    module = ModuleType("master_script.core.rag_02a568f")
    module.__package__ = "master_script.core"
    exec(compile(source, "rag.py@02a568f", "exec"), module.__dict__)
    return module, digest(source.encode())


def main(output):
    from master_script.core import rag
    from master_script.core.config import implementation_fingerprint
    tokenizer, loader = load_tokenizer()
    config = json.loads((TOKENIZER.parent / "config.json").read_text())
    bundle = {"tokenizer": tokenizer, "device": "cpu",
              "model": SimpleNamespace(config=SimpleNamespace(max_position_embeddings=config["max_position_embeddings"]))}
    study = json.loads(STUDY.read_text())
    private = [d["text"] for d in study["private_documents"]]

    def settings(**kw):
        return SimpleNamespace(**{"max_context_tokens": 768, "max_new_tokens": 64, "prompt_format": "plain",
                                  "instruction_defense": False, **kw})

    def question(text):
        return f'Does this: "{text}" appear in the context? Answer with Yes or No.'

    def limit_at(contexts, char_index):
        e = tokenizer(rag.CONTEXT_SEPARATOR.join(contexts), add_special_tokens=False, return_offsets_mapping=True)
        return sum(s < char_index for s, _ in e["offset_mapping"])

    def legacy_visible(contexts, text, retained):
        visible = tokenizer.encode(rag.CONTEXT_SEPARATOR.join(contexts), add_special_tokens=False)[:retained]
        target = tokenizer.encode(text, add_special_tokens=False)
        return any(visible[i:i + len(target)] == target for i in range(len(visible) - len(target) + 1))

    # 1. The 200 boundary controls of the retained review, plus negative controls.
    controls = []
    for c in study["membership_candidates"]:
        text = c["text"]
        control = next(d for d in private if d != text)
        q = question(text)
        positive, detail = rag.context_exposure_record(bundle, q, [text, control], settings(), text)
        start = len(control) + len(rag.CONTEXT_SEPARATOR)
        row = {"candidate_sha256": digest(text.encode()),
               "old_exact_span_visible": legacy_visible([text, control], text, positive["context_tokens_used"]),
               "new_exposure": positive["exposure"], "context_truncated": positive["context_truncated"],
               "legacy_field_in_private_detail": detail["legacy_exact_token_span_visible"],
               "negative_absent_question_only": rag.context_exposure(bundle, q, [control], settings(), text),
               "negative_removed_by_truncation": rag.context_exposure(
                   bundle, q, [control, text], settings(max_context_tokens=limit_at([control, text], start)), text),
               "partial_truncation": rag.context_exposure(
                   bundle, q, [control, text], settings(max_context_tokens=limit_at([control, text], start + len(text) // 2)), text)}
        controls.append(row)

    def count(key, field="exposure"):
        values = {}
        for r in controls:
            v = r[key] if isinstance(r[key], str) else r[key][field]
            values[v] = values.get(v, 0) + 1
        return values

    boundary = {
        "controls": len(controls),
        "construction": "candidate + '\\n\\n' + first different private document (as in the retained review)",
        "before_old_exact_span_visible": sum(r["old_exact_span_visible"] for r in controls),
        "before_false_invisible": sum(not r["old_exact_span_visible"] for r in controls),
        "after_complete_exposure": sum(r["new_exposure"] == "complete" for r in controls),
        "after_positive_classes": count("new_exposure"),
        "old_true_implies_new_complete": all(r["new_exposure"] == "complete" for r in controls if r["old_exact_span_visible"]),
        "any_positive_truncated": any(r["context_truncated"] for r in controls),
        "negative_question_only_classes": count("negative_absent_question_only"),
        "negative_question_only_causes": count("negative_absent_question_only", "absence_cause"),
        "negative_truncated_away_classes": count("negative_removed_by_truncation"),
        "negative_truncated_away_causes": count("negative_removed_by_truncation", "absence_cause"),
        "partial_truncation_classes": count("partial_truncation"),
    }

    # 2. Generation token IDs and retained-count fields: historical vs repaired.
    generation = {"status": "skipped", "reason": "torch or transformers unavailable"}
    try:
        import torch
        import transformers  # noqa: F401
    except ImportError:
        old = None
    else:
        old, old_sha = historical_rag()
    if old is not None:
        compared = mismatched = field_mismatch = 0
        old_true_not_complete = 0
        hashes = sha256()
        for index, c in enumerate(study["membership_candidates"]):
            others = [private[(index * 7 + k) % len(private)] for k in range(1, 5) if private[(index * 7 + k) % len(private)] != c["text"]][:3]
            contexts = others[:index % 4] + [c["text"]] + others[index % 4:]
            for fmt in ("plain", "chat"):
                for defended in (False, True):
                    for budget in (768, 64, 16):
                        s = settings(prompt_format=fmt, instruction_defense=defended, max_context_tokens=budget)
                        for q, ctx, answer_tokens in ((question(c["text"]), contexts, 0), (question(c["text"]), [], 0),
                                                      (study["utility_queries"][index % 20]["question"], contexts, 12)):
                            before = old._prompt_tokens(bundle, q, ctx, s, answer_tokens)
                            after = rag._prompt_tokens(bundle, q, ctx, s, answer_tokens)
                            compared += 1
                            mismatched += not torch.equal(before, after)
                            hashes.update(json.dumps(after[0].tolist()).encode())
                        a = old.context_exposure(bundle, question(c["text"]), contexts, s, c["text"])
                        b = rag.context_exposure(bundle, question(c["text"]), contexts, s, c["text"])
                        field_mismatch += any(a[k] != b[k] for k in ("context_tokens_before", "context_tokens_used", "context_truncated"))
                        old_true_not_complete += a["candidate_tokens_visible"] and b["exposure"] != "complete"
        generation = {"status": "compared", "historical_rag_sha256": old_sha, "prompts_compared": compared,
                      "token_id_mismatches": mismatched, "sha256_of_all_repaired_prompt_ids": hashes.hexdigest(),
                      "retained_count_field_mismatches": field_mismatch,
                      "old_visible_true_but_new_not_complete": old_true_not_complete,
                      "matrix": "200 candidates x {plain, chat(stand-in template)} x {instruction off, on} x "
                                "max_context_tokens {768, 64, 16} x {membership query with contexts, empty contexts, "
                                "utility query with answer_tokens=12}; candidate position rotates 0..3"}

    # 3. Read-only tabulation of historical flags. Derivation is logical, from recorded
    # fields only; historical files are not modified and flags are not rewritten.
    import unicodedata
    texts = {digest(t.encode()): t for t in [c["text"] for c in study["membership_candidates"]] +
             [d["text"] for k in ("public_documents", "private_documents") for d in study[k]]}
    corpora = {k: [d["text"] for d in study[f"{k}_documents"]] for k in ("public", "private")}
    preconditions = {
        "no_candidate_contains_separator": all(rag.CONTEXT_SEPARATOR not in t for t in texts.values()),
        "all_text_nfc_stable": all(unicodedata.normalize("NFC", t) == t for t in texts.values()),
        "no_text_starts_with_combining_mark": all(not unicodedata.combining(t[0]) for t in texts.values()),
        "no_candidate_substring_of_other_document": all(
            not any(c["text"] in d and c["text"] != d for d in docs)
            for c in study["membership_candidates"] for docs in corpora.values()),
    }
    table = {}

    def tabulate(scope, row, text):
        audit = row.get("context_audit", {})
        old_flag = audit.get("candidate_tokens_visible")
        if audit.get("status") != "checked":
            derived = "undetermined"
        elif old_flag:  # A matched token span decodes to the candidate: presence is shown.
            derived = "complete (old true)"
        elif text is None:
            derived = "undetermined (text not in pinned study)"
        elif row["retrieved"] and not audit["context_truncated"]:
            derived = "complete (retrieved, untruncated)"
        elif not row["retrieved"] and all(preconditions.values()):
            derived = "absent: not_in_context (not retrieved)"
        else:
            derived = "undetermined (retrieved, truncated)"
        key = (scope, bool(row["retrieved"]), audit.get("context_truncated"), old_flag, derived)
        table[key] = table.get(key, 0) + 1

    results = sorted(EXPORT.glob("*/*/artifacts/*/result.json"))
    for path in results:
        for evaluation in json.loads(path.read_text()).get("pipeline_evaluations", []):
            for name, condition in evaluation.get("rag_conditions", {}).items():
                for t in condition["membership_trials"]:
                    tabulate(f"{name}/{'member' if t['truth_member'] else 'nonmember'}", t, texts.get(t["candidate_sha256"]))
            for cell in evaluation.get("membership_overlap_cells", []):
                text = texts.get(cell["target_sha256"])
                tabulate(f"overlap/{cell['defense']}/datastore_{'member' if cell['datastore_member'] else 'nonmember'}",
                         cell, text if text is not None and not any(text in d and text != d for d in corpora["private"]) else None)
    historical = {
        "status": "tabulated" if results else "skipped: no result.json under RAG_EXPOSURE_RESULTS",
        "source": shown(EXPORT) + " (read-only)",
        "result_files": len(results), "records": sum(table.values()), "derivation_preconditions": preconditions,
        "assumptions": [
            "retrieved == candidate is an exact element of the contexts list (rag.py@02a568f).",
            "context_truncated was computed from the same encoding as the retained prompt, so false means every context token was retained.",
            "The remote tokenizer is the exported byte-level BPE (lossless; offsets cover every character) with an NFC normalizer.",
            "Contexts contain corpus documents only; a candidate cannot span the separator (it contains none).",
        ],
        "rows": [dict(zip(("scope", "retrieved", "context_truncated", "old_candidate_tokens_visible", "derived_v2_class"), k),
                      records=v) for k, v in sorted(table.items(), key=lambda kv: str(kv[0]))],
        "derived_totals": {d: sum(v for k, v in table.items() if k[4] == d) for d in sorted({k[4] for k in table})},
    }

    tokenizers = sys.modules.get("tokenizers") or __import__("tokenizers")
    transformers = sys.modules.get("transformers")
    report = {
        "purpose": "CPU-only verification of context_exposure_v2; no weights, inference, GPU or network.",
        "git_head": git("rev-parse", "HEAD").strip(),
        "working_tree_changes": git("status", "--porcelain", "--", "master_script", "tests").splitlines(),
        "sources": {p: digest((ROOT / p).read_bytes()) for p in (
            "master_script/core/rag.py", "tests/test_rag_context_exposure.py",
            "outputs/context_exposure_repair_20260927/verify.py")},
        "historical_commit": HISTORICAL_COMMIT, "historical_core_fingerprint": HISTORICAL_FINGERPRINT,
        "core_fingerprint": implementation_fingerprint(),
        "study": {"path": str(STUDY.relative_to(ROOT)), "sha256": digest(STUDY.read_bytes())},
        "tokenizer": {"path": shown(TOKENIZER), "file_sha256": digest(TOKENIZER.read_bytes()),
                      "loader": loader, "tokenizers_version": tokenizers.__version__,
                      "transformers_version": getattr(transformers, "__version__", None),
                      "chat_template": "stand-in ChatML (not exported)" if transformers else "not used"},
        "python": platform.python_version(), "host": platform.node(),
        "boundary_controls": boundary, "generation_token_ids": generation, "historical_flags": historical,
        "limitations": [
            "Controls are constructed from pinned study text; they are not the historical membership contexts, "
            "whose rankings and final prompt tokens were never saved.",
            "The original remote tokenizers version is unrecorded; results are reported per local version.",
            "The Qwen chat template was not exported; a ChatML stand-in exercises the chat code path only.",
            "Python 3.10 was not available locally; compatibility was checked statically and on 3.11/3.13.",
            "Audit correctness is not evidence of privacy protection.",
        ],
        "per_control": [{k: v for k, v in r.items() if not isinstance(v, dict)} for r in controls],
    }
    Path(output).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("core_fingerprint", "tokenizer", "boundary_controls", "generation_token_ids")}, indent=1))
    print(json.dumps({k: historical[k] for k in ("status", "records", "derivation_preconditions", "derived_totals")}, indent=1))
    for row in historical["rows"]:
        print(row)


if __name__ == "__main__":
    main(sys.argv[1])
