"""Inference RAG, Mirabel detection, and separate datastore-MIA/utility audits.

Mirabel follows Choi et al. (2025), Algorithm 1. The black-box datastore attack
uses Anderson et al. (2024)'s membership question; it never receives retrieval
IDs or private similarities. Audit diagnostics are not a DP-protected release.
"""
from collections import Counter
from functools import lru_cache
import json
import math
import re
import time


def validate_study(study):
    required = {"public_documents", "private_documents", "membership_candidates", "utility_queries"}
    if not isinstance(study, dict) or set(study) != required:
        raise ValueError(f"RAG study must contain exactly: {', '.join(sorted(required))}")
    for section in required:
        rows = study[section]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{section} must be a nonempty list")
        seen = set()
        keys = {"id", "question", "answer"} if section == "utility_queries" else {"id", "text"}
        for row in rows:
            if not isinstance(row, dict) or set(row) != keys:
                raise ValueError(f"{section}: each row must contain {sorted(keys)}")
            if any(not isinstance(row[k], str) or not row[k].strip() for k in keys):
                raise ValueError(f"{section}: fields must be nonempty strings")
            if row["id"] in seen:
                raise ValueError(f"{section}: duplicate id")
            seen.add(row["id"])
    for section in ("public_documents", "private_documents"):
        documents = study[section]
        if len(documents) < 3:
            raise ValueError(f"{section} requires at least three documents for Mirabel")
        texts = [d["text"] for d in documents]
        if len(set(texts)) != len(texts):
            raise ValueError(f"{section}: duplicate document text")
        lookup = {d["id"]: d["text"] for d in documents}
        labels = []
        for candidate in study["membership_candidates"]:
            present = candidate["id"] in lookup
            if present and lookup[candidate["id"]] != candidate["text"]:
                raise ValueError("Candidate ID and corpus text disagree")
            if not present and candidate["text"] in texts:
                raise ValueError("Candidate occurs under another corpus ID")
            labels.append(present)
        if not any(labels) or all(labels):
            raise ValueError(f"{section}: candidates must include both members and nonmembers")


def mirabel(scores, significance=0.05):
    """Full-corpus cosine statistics excluding one maximum, then Gumbel cutoff."""
    import numpy as np

    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) < 3 or not np.isfinite(scores).all():
        raise ValueError("Mirabel needs at least three finite similarity scores")
    if not 0 < significance < 1:
        raise ValueError("Mirabel significance must be between zero and one")
    index = int(np.argmax(scores))
    remaining = np.delete(scores, index)
    mean, std = float(remaining.mean()), float(remaining.std())
    scale = math.sqrt(2 * math.log(len(scores)))
    critical = -math.log(-math.log1p(-significance))
    threshold = mean + std * scale + critical * std / scale
    return {"detected": bool(scores[index] > threshold), "index": index, "threshold": threshold}


@lru_cache(maxsize=2)
def _embedding_bundle(model_id):
    from transformers import AutoModel, AutoTokenizer
    return AutoModel.from_pretrained(model_id).to("cpu").eval(), AutoTokenizer.from_pretrained(model_id)


def embed(texts, model_id):
    """Attention-mask mean pooling and L2 normalization for the frozen encoder."""
    import numpy as np
    import torch

    model, tokenizer = _embedding_bundle(model_id)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), 32):
            tokens = tokenizer(texts[start:start + 32], padding=True, truncation=True,
                               max_length=256, return_tensors="pt")
            hidden = model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            chunks.append(torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy())
    return np.concatenate(chunks)


def retrieve(documents, embeddings, query_embedding, top_k, significance, defended):
    import numpy as np

    scores = embeddings @ query_embedding
    detection = mirabel(scores, significance)
    ranking = np.argsort(-scores, kind="stable").tolist()
    if defended and detection["detected"]:
        ranking.remove(detection["index"])
    selected = ranking[:top_k]
    return [documents[i]["text"] for i in selected], bool(defended and detection["detected"])


def _prompt_tokens(bundle, question, contexts, settings, answer_tokens=0):
    import torch

    tokenizer, model = bundle["tokenizer"], bundle["model"]
    capacity = getattr(model.config, "max_position_embeddings", getattr(model.config, "n_positions", 1024))
    capacity = int(capacity or 1024)
    lead = "Use the context to answer the question.\nQuestion: " + question + "\nContext:\n"
    tail = "\nAnswer:"
    if getattr(settings, "prompt_format", "plain") == "chat":
        # Keep the same context budget, inside the model's native user turn.
        marker = "__RETRIEVED_CONTEXT_SLOT_9217__"
        if marker in question:
            raise ValueError("Question contains the reserved context marker")
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": lead + marker + tail}],
            tokenize=False, add_generation_prompt=True,
        )
        lead, tail = formatted.split(marker)
    prefix = tokenizer.encode(lead, add_special_tokens=False)
    suffix = tokenizer.encode(tail, add_special_tokens=False)
    reserve = max(settings.max_new_tokens, answer_tokens)
    available = capacity - len(prefix) - len(suffix) - reserve
    if available < 0:
        raise ValueError("RAG question exceeds model context; use shorter study candidates or a larger model")
    context = tokenizer.encode("\n\n".join(contexts), add_special_tokens=False)
    context = context[:min(available, settings.max_context_tokens)]
    return torch.tensor([prefix + context + suffix], dtype=torch.long, device=bundle["device"])


def generate_answer(bundle, question, contexts, settings):
    import torch

    tokens = _prompt_tokens(bundle, question, contexts, settings)
    bundle["model"].eval()
    with torch.no_grad():
        generated = bundle["model"].generate(
            input_ids=tokens, attention_mask=torch.ones_like(tokens), do_sample=False,
            max_new_tokens=settings.max_new_tokens,
            pad_token_id=bundle["tokenizer"].pad_token_id,
            eos_token_id=bundle["tokenizer"].eos_token_id,
        )
    return bundle["tokenizer"].decode(generated[0, tokens.shape[1]:], skip_special_tokens=True)


def answer_nll(bundle, question, answer, contexts, settings):
    """Score only answer tokens, excluding the question and retrieved documents."""
    import torch

    ids = bundle["tokenizer"].encode(" " + answer, add_special_tokens=False)
    prefix = _prompt_tokens(bundle, question, contexts, settings, answer_tokens=len(ids))
    suffix = torch.tensor([ids], dtype=torch.long, device=bundle["device"])
    tokens = torch.cat([prefix, suffix], dim=1)
    labels = tokens.clone()
    labels[:, :prefix.shape[1]] = -100
    with torch.no_grad():
        return float(bundle["model"](input_ids=tokens, attention_mask=torch.ones_like(tokens), labels=labels).loss)


def _words(text):
    return re.findall(r"\w+", text.casefold())


def answer_utility(answer, expected):
    actual, target = _words(answer), _words(expected)
    overlap = sum((Counter(actual) & Counter(target)).values())
    return {"exact_match": float(actual == target),
            "token_f1": 2 * overlap / (len(actual) + len(target)) if actual or target else 1.0}


def evaluate_pipeline(bundle, config, pipeline, trial_id=None):
    evaluation = {"training_privacy": bundle.get("privacy", {}),
                  "audit_outputs_private": True}
    if pipeline.rag is None:
        return evaluation
    from .metrics import base_metrics, roc_auc

    settings = pipeline.rag
    if settings.evaluation_trials is not None and trial_id is not None and trial_id >= settings.evaluation_trials:
        evaluation["rag_evaluation_skipped"] = True
        return evaluation
    study = json.loads(settings.study_json)
    training_records = set(bundle.get("training_records", []))
    # Utility is held out from FL training; retrieved evidence may of course
    # contain its answer. Exact overlap is checked, not semantic disjointness.
    for q in study["utility_queries"]:
        if q["question"] in training_records or f"Question: {q['question']}\nAnswer: {q['answer']}" in training_records:
            raise ValueError("RAG utility query overlaps a training record")
    queries = [f'Does this: "{c["text"]}" appear in the context? Answer with Yes or No.'
               for c in study["membership_candidates"]]
    queries += [q["question"] for q in study["utility_queries"]]
    query_vectors = embed(queries, settings.embedding_model)
    conditions = {}
    for corpus_name in ("public", "private"):
        documents = study[f"{corpus_name}_documents"]
        vectors = embed([d["text"] for d in documents], settings.embedding_model)
        ids = {d["id"] for d in documents}
        for defended in (False, True):
            start = time.perf_counter()
            membership, utility = [], []
            hidden = 0
            for i, query in enumerate(queries):
                contexts, did_hide = retrieve(documents, vectors, query_vectors[i], settings.top_k,
                                               settings.significance, defended)
                hidden += int(did_hide)
                generated = generate_answer(bundle, query, contexts, settings)
                if i < len(study["membership_candidates"]):
                    candidate = study["membership_candidates"][i]
                    # Published black-box convention: no yes/no answer -> nonmember.
                    response = re.search(r"\b(yes|no)\b", generated, flags=re.I)
                    prediction = bool(response and response.group(1).lower() == "yes")
                    membership.append({"candidate_index": i, "truth_member": candidate["id"] in ids,
                                       "training_member": candidate["text"] in training_records,
                                       "score": float(prediction), "pred_member": prediction,
                                       "answer_recognized": response is not None})
                else:
                    q = study["utility_queries"][i - len(study["membership_candidates"])]
                    utility.append({"query_index": i - len(study["membership_candidates"]),
                                    **answer_utility(generated, q["answer"]),
                                    "answer_nll": answer_nll(bundle, query, q["answer"], contexts, settings)})
            metrics = base_metrics(membership)
            metrics["roc_auc"] = roc_auc([r["truth_member"] for r in membership], [r["score"] for r in membership])
            conditions[f"{corpus_name}_{'mirabel' if defended else 'ordinary'}"] = {
                "membership_target": "retrieval_document", "attack": "anderson2024_black_box",
                "metrics": metrics, "membership_trials": membership,
                "utility": {key: sum(r[key] for r in utility) / len(utility)
                            for key in ("exact_match", "token_f1", "answer_nll")},
                "utility_trials": utility, "hidden_document_queries": hidden,
                "seconds": time.perf_counter() - start,
            }
    # A no-context condition distinguishes retrieval gains from the trained LM.
    baseline = []
    for q in study["utility_queries"]:
        response = generate_answer(bundle, q["question"], [], settings)
        baseline.append({**answer_utility(response, q["answer"]),
                         "answer_nll": answer_nll(bundle, q["question"], q["answer"], [], settings)})
    evaluation.update(rag_conditions=conditions,
                      no_retrieval_utility={key: sum(r[key] for r in baseline) / len(baseline)
                                            for key in ("exact_match", "token_f1", "answer_nll")})
    return evaluation
