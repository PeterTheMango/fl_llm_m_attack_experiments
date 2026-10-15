"""One generalization-study job on the GPU: paired worlds, a LiRA shadow world, the LoRA positive control or P0.

Run by `python -m master_script.tools.grounded_generalization_study job`, one job per
process, after the tool's CPU resolution. It reuses the parent study's measurement
functions unchanged (grounded_job.natural_questions, retention, learning_measures,
Session) and adds, opt-in:

- training by cell: the parent's Flower path with full fp32 fine-tuning (A-full), or
  LoRA adapters on a frozen bf16 base with in-process FedAvg (core/lora.py);
- the lean measurement set (decision 10): utility on W1 only, without the no-context
  and public-library F1;
- weight scores with their raw NLLs (the Reference ratio is computed exactly as the
  parent's score_candidate_hf and template form), for the target and for population
  records (U), so LiRA and RMIA can be computed after the fact;
- shadow worlds for LiRA (proposal §6.2).

Under LoRA the reference model is the same base with its adapters switched off.
Raw responses go only to the mode-0600 private audit file. Nothing here makes a
privacy claim.
"""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import time

from . import generalization, grounded, lora
from .config import METHOD_VERSION, implementation_fingerprint, resolve_run_config, validate_attack_config
from .grounded_job import Session, amia_config, learning_measures, natural_questions, retention
from .queue import write_json

JOB_SCHEMA, RESULT_SCHEMA = "generalization_job_v1", "generalization_result_v1"
KINDS = ("paired", "shadow", "control", "p0")
MEASUREMENTS = ("reference", "retention", "utility", "nq", "n_documents", "population")
FORMS = ("record", "template")


def validate_job(job):
    """Structural checks shared by prepare, resolve and execution."""
    if job.get("schema") != JOB_SCHEMA or job.get("kind") not in KINDS:
        raise ValueError("Not a generalization job")
    kind, arm = job["kind"], job["arm"]
    allowed = {"paired": ("RG", "CB-AO"), "shadow": ("RG", "CB-AO"), "control": ("RG",), "p0": ("P0",)}
    if arm not in allowed[kind]:
        raise ValueError(f"{kind} jobs use {allowed[kind]}, not {arm!r}")
    spec = generalization.cell(job["cell"])
    training = job["training"]
    if training["method"] != spec["training"] or job["clients"] != spec["clients"]:
        raise ValueError("The job's training method or client scheme differs from its cell")
    profile = spec["profile"]
    config = job["config"]
    if (config["model_id"], config["model_revision"]) != (profile["model_id"], profile["revision"]):
        raise ValueError("The job's model differs from its cell's pinned model")
    if training["method"] == "lora":
        lora_spec = {k: training.get(k) for k in ("rank", "alpha", "dropout", "dtype")}
        if lora_spec != {k: generalization.LORA[k] for k in ("rank", "alpha", "dropout", "dtype")}:
            raise ValueError("LoRA settings differ from the pre-registered recipe")
        if training.get("lr") not in (generalization.LORA["lr"], generalization.LORA["escalation_lr"]):
            raise ValueError("LoRA lr is 2e-4, or 5e-4 after a failed learning check")
    if job["defense"] != {"mechanism": "none"}:
        raise ValueError("The generalization study trains without DP (decision 12)")
    if job["cohort"] not in grounded.COHORTS:
        raise ValueError("Unknown cohort")
    if set(job["measurements"]) - set(MEASUREMENTS):
        raise ValueError("Unknown measurement")
    if kind in ("paired", "control") and not isinstance(job.get("target_index"), int):
        raise ValueError("Paired and control jobs name one target")
    if kind == "shadow":
        shadow = job.get("shadow") or {}
        if len(shadow.get("targets", [])) != len(shadow.get("in_row", [])) or not shadow.get("targets"):
            raise ValueError("A shadow job names its targets and its design row")
    if kind == "control":
        control = job.get("control") or {}
        if training["method"] != "lora" or control.get("steps") not in (6, 12) or control.get("lr") != 1e-3:
            raise ValueError("The LoRA positive control trains 6 steps (or the one 12-step rerun) at lr 1e-3")
    if kind == "p0" and not job.get("datastore_targets") and not job["measurements"].get("utility"):
        raise ValueError("A P0 job measures the datastore, utility or both")
    grounded.RagSettings(**job["rag"])
    amia_config(config)
    return True


# ------------------------------------------------------------------ helpers

def _now():
    return time.perf_counter()


def _sha(text):
    return sha256(text.encode()).hexdigest()


def _release():
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _bundle(model, tokenizer):
    return {"model": model.eval(), "tokenizer": tokenizer, "device": next(model.parameters()).device}


def tokenizer_for(config, profile):
    from transformers import AutoConfig, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return generalization.pinned(tokenizer, profile), AutoConfig.from_pretrained(config.model_id, revision=config.model_revision)


def record_nll(bundle, text):
    """Mean full-LM NLL of a record text, exactly as score_candidate_hf's mean_nll."""
    import torch
    encoded = bundle["tokenizer"](text, return_tensors="pt", truncation=True, max_length=10 ** 6)
    encoded = {k: v.to(bundle["device"]) for k, v in encoded.items()}
    with torch.no_grad():
        return float(bundle["model"](**encoded, labels=encoded["input_ids"]).loss.detach().cpu())


def template_nll(bundle, example):
    """Mean answer-only NLL of an encoded example (the parent's template form)."""
    import torch
    from .scoring import encoded_batch
    batch = encoded_batch([example], bundle["tokenizer"].pad_token_id, bundle["device"])
    with torch.no_grad():
        return float(bundle["model"](**batch).loss.detach().cpu())


def weight_scores(bundle, reference, items, max_length):
    """For each (closed_text, template_example): both forms' model and reference NLLs, token counts and
    the Carlini ratio (larger is more member-like). reference(fn) evaluates fn on the reference model."""
    from .attacks.reference import calibrated_reference_score
    from .scoring import effective_record
    tokenizer = bundle["tokenizer"]
    texts = [effective_record(tokenizer, closed, max_length) for closed, _ in items]
    model = [(record_nll(bundle, t), template_nll(bundle, e)) for t, (_, e) in zip(texts, items)]
    ref = reference(lambda b: [(record_nll(b, t), template_nll(b, e)) for t, (_, e) in zip(texts, items)])
    out = []
    for text, (_, example), (m_rec, m_tpl), (r_rec, r_tpl) in zip(texts, items, model, ref):
        out.append({"record": {"ratio": calibrated_reference_score(m_rec, r_rec), "model_nll": m_rec, "reference_nll": r_rec,
                               "tokens": len(tokenizer(text)["input_ids"]) - 1},
                    "template": {"ratio": calibrated_reference_score(m_tpl, r_tpl), "model_nll": m_tpl, "reference_nll": r_tpl,
                                 "tokens": sum(label != -100 for label in example.labels[1:])}})
    return out


def lean_utility(bundle, session, library, world):
    """RAG F1 on F with EM, refusal, gold retrieval, gate and answer NLL; no no-context or public-library F1."""
    from .grounded_job import _generate, _retrieve
    from .rag import answer_nll, answer_utility, is_refusal
    rows = []
    for q in session.cohort["F"]:
        _, contexts, gate = _retrieve(session, library, q["question"])
        response = _generate(bundle, session, q["question"], contexts)
        session.audit({"kind": "utility", "world": world, "query_id": q["id"], "answer": response})
        scored = answer_utility(response, q["answer"])
        rows.append({"query_id": q["id"], "article": q["article"], "f1": scored["token_f1"], "em": scored["exact_match"],
                     "refused": is_refusal(response), "gold_retrieved": session.passage(q["passage"]) in contexts,
                     "gate": gate, "answer_nll": answer_nll(bundle, q["question"], q["answer"], contexts, session.settings)})
    return {"F": rows}


# ---------------------------------------------------------------- training

class Trainer:
    """Trains one world for the job's cell and yields the model with its reference."""

    def __init__(self, job, config, tokenizer, device):
        self.job, self.config, self.tokenizer, self.device = job, config, tokenizer, device
        self.method = job["training"]["method"]
        self.profile = generalization.cell(job["cell"])["profile"]

    def train(self, world, directory, timings, name):
        started = _now()
        if self.method == "full":
            from .attacks import amia
            from .pipeline import Defense, Pipeline
            model, _, _, history, _ = amia.federated_fine_tune(self.config, directory, Pipeline(defense=Defense(mechanism="none")),
                                                               world=world)
            provenance = getattr(model, "_training_provenance", {})
        else:
            model = self.base()
            lora.attach(model, self.job["training"]["rank"], self.job["training"]["alpha"], self.job["training"]["dropout"],
                        seed=self.config.seed)
            history, provenance = lora.federated_fine_tune(self.config, model, self.tokenizer, world, self.job["training"]["lr"])
            lora.save_adapter(model, Path(directory) / "adapter.npz")
        timings[f"{name}_training_seconds"] = _now() - started
        return model, history, provenance

    def base(self):
        model = lora.load_base(self.config.model_id, self.config.model_revision, self.job["training"]["dtype"], self.device)
        model.config.pad_token_id = self.tokenizer.pad_token_id
        return model.eval()

    def reference(self, model):
        """A callable that evaluates fn(bundle) on the pretrained reference, then frees what it loaded."""
        if self.method == "lora":
            def run(fn):
                with lora.adapters_disabled(model):
                    return fn(_bundle(model, self.tokenizer))
            return run

        def run_full(fn):
            from .federation import load_reference_bundle
            reference = load_reference_bundle(self.config)
            try:
                return fn({**reference, "tokenizer": self.tokenizer})
            finally:
                del reference
                _release()
        return run_full


# --------------------------------------------------------------------- run

def run_job(job, study, output_dir):
    """Execute one validated job; writes the result twice (for checkpoint retirement)."""
    import torch
    from .attacks.amia import client_device
    validate_job(job)
    if grounded.study_sha256(study) != job["study_sha256"]:
        raise ValueError("Study data differ from the job's declaration")
    started = _now()
    config = resolve_run_config(amia_config(job["config"]))
    validate_attack_config(config)
    if config.num_clients != study["sizes"]["clients"] or config.target_client_id != 0:
        raise ValueError("The study's worlds place the target in client 0 of its declared clients")
    output = Path(output_dir).resolve()
    artifact = output / "artifacts" / "result"
    artifact.mkdir(parents=True, exist_ok=False)
    device = torch.device(client_device(config))
    session = Session(job, study, artifact / "private-audit", "cuda" if torch.cuda.is_available() else "cpu")
    profile = generalization.cell(job["cell"])["profile"]
    tokenizer, model_config = tokenizer_for(config, profile)
    timings = {"setup_seconds": _now() - started}
    runner = {"paired": _paired, "shadow": _shadow, "control": _control, "p0": _p0}[job["kind"]]
    body = runner(job, study, config, session, artifact, timings, tokenizer, model_config, device)
    digest = _sha(json.dumps(job, sort_keys=True))
    result = {"schema": RESULT_SCHEMA, "status": "complete", "run_id": f"generalization_{digest[:24]}",
              "job_sha256": digest, "kind": job["kind"], "cell": job["cell"], "arm": job["arm"], "cohort": job["cohort"],
              "seed": config.seed, "target_index": job.get("target_index"), "method_version": METHOD_VERSION,
              "implementation_fingerprint": implementation_fingerprint(), "config": asdict(config),
              "training": job["training"], "clients": job["clients"], "rag": job["rag"], "nli": job["nli"],
              "gate": job.get("gate"), "measurements": job["measurements"], **body,
              "timings": {**timings, "total_seconds": _now() - started}, "audit_outputs_private": True,
              "scope": "Generalization-study measurement; attack effectiveness only, no defense or privacy claim"}
    write_json(artifact / "result.json", result)
    write_json(output / "result.json", result)
    return result


def _encode(tokenizer, model_config, session, settings, max_length, arm):
    return lambda r: grounded.training_record(tokenizer, model_config, r, session.passage(r["passage"]), arm, settings, max_length)


def _population(job, session, tokenizer, model_config, config):
    """The first n U records (public, never trained in any world), in both attacker forms."""
    n = job["measurements"].get("population", 0)
    records = session.cohort["records"]["U"][:n]
    return [(grounded.closed_book_record(r["question"], r["answer"]),
             grounded.template_example(tokenizer, model_config, r, session.passage(r["passage"]), job["arm"], session.settings,
                                       config.max_length)) for r in records], [r["id"] for r in records]


def _target_items(target, session, tokenizer, model_config, config, arm):
    closed = grounded.closed_book_record(target["training"]["question"], target["training"]["answer"])
    template = grounded.template_example(tokenizer, model_config, target["training"], session.passage(target["passage"]),
                                         arm, session.settings, config.max_length)
    return closed, template


def _paired(job, study, config, session, artifact, timings, tokenizer, model_config, device):
    arm, cohort, index = job["arm"], job["cohort"], job["target_index"]
    m = job["measurements"]
    target = session.cohort["targets"][index]
    item = _target_items(target, session, tokenizer, model_config, config, arm)
    population, population_ids = _population(job, session, tokenizer, model_config, config)
    encode = _encode(tokenizer, model_config, session, session.settings, config.max_length, arm)
    trainer = Trainer(job, config, tokenizer, device)
    body = {"target_sha256": target["target_sha256"], "worlds": {}, "privacy": {}, "population_ids": population_ids}
    document = session.passage(target["passage"])
    for name, member in (("W1", True), ("W0", False)):
        records = generalization.world_records(study, cohort, index, config.seed, member, job["clients"])
        grounded.check_world(study, cohort, records, index, member)
        if name == "W1":
            body["client_skew"] = generalization.skew_summary(study, cohort, records)
        world = {"partitions": [[encode(r) for r in part] for part in records], "target": encode(target["training"]),
                 "held_out": encode(target["hold"]), "member": member}
        if name == "W1":
            sizes = [len(x.input_ids) for part in world["partitions"] for x in part]
            body["sequence_lengths"] = {"max_tokens": max(sizes), "mean_tokens": sum(sizes) / len(sizes)}
        model, history, provenance = trainer.train(world, artifact / name.lower(), timings, name)
        body["privacy"][name] = {"epsilon": "inf", "mechanism": "none"}
        body[f"federated_history_{name}"], body[f"training_provenance_{name}"] = history, provenance
        bundle = _bundle(model, tokenizer)
        started, out = _now(), {}
        if m.get("reference"):
            scores = weight_scores(bundle, trainer.reference(model), [item] + population, config.max_length)
            out["reference"], out["population"] = scores[0], scores[1:]
        if m.get("retention"):
            out["retention"] = retention(bundle, session, target["probes"], name)
        if m.get("nq"):
            cells = {"both": True, "training_only": False} if member else {"library_only": True, "neither": False}
            out["natural_questions"] = {cell: natural_questions(bundle, session, document, target["probes"],
                                                                session.library(document if on else None), cell, name)
                                        for cell, on in cells.items()}
            if not member and m.get("n_documents"):
                off = session.library()
                out["never_documents"] = [natural_questions(bundle, session, session.passage(d["passage"]), d["probes"], off,
                                                            "never_library", name)
                                          for d in session.cohort["N"][:m["n_documents"]]]
        if member and m.get("utility"):
            out["utility"] = lean_utility(bundle, session, session.library(document), name)
        timings[f"{name}_measurement_seconds"] = _now() - started
        body["worlds"][name] = out
        del model, bundle
        _release()
    return body


def _shadow(job, study, config, session, artifact, timings, tokenizer, model_config, device):
    shadow, cohort = job["shadow"], job["cohort"]
    records = generalization.shadow_records(study, cohort, shadow["targets"], shadow["in_row"], config.seed, job["clients"])
    generalization.check_shadow_world(study, cohort, records, shadow["targets"], shadow["in_row"])
    encode = _encode(tokenizer, model_config, session, session.settings, config.max_length, job["arm"])
    world = {"partitions": [[encode(r) for r in part] for part in records], "target": None, "held_out": None, "member": None}
    trainer = Trainer(job, config, tokenizer, device)
    model, history, provenance = trainer.train(world, artifact / "shadow", timings, "shadow")
    targets = [session.cohort["targets"][i] for i in shadow["targets"]]
    items = [_target_items(t, session, tokenizer, model_config, config, job["arm"]) for t in targets]
    population, population_ids = _population(job, session, tokenizer, model_config, config)
    started = _now()
    scores = weight_scores(_bundle(model, tokenizer), trainer.reference(model), items + population, config.max_length)
    timings["shadow_scoring_seconds"] = _now() - started
    del model
    _release()
    return {"shadow": dict(shadow), "federated_history_shadow": history, "training_provenance_shadow": provenance,
            "client_skew": generalization.skew_summary(study, cohort, records), "privacy": {"shadow": {"epsilon": "inf", "mechanism": "none"}},
            "targets": [{"target_index": i, "target_sha256": t["target_sha256"], "in": bool(inside), "weights": s}
                        for i, t, inside, s in zip(shadow["targets"], targets, shadow["in_row"], scores)],
            "population": scores[len(items):], "population_ids": population_ids}


def _control(job, study, config, session, artifact, timings, tokenizer, model_config, device):
    cohort, index = job["cohort"], job["target_index"]
    target = session.cohort["targets"][index]
    passage = session.passage(target["passage"])
    encode = _encode(tokenizer, model_config, session, session.settings, config.max_length, "RG")
    trainer = Trainer(job, config, tokenizer, device)
    body = {"target_sha256": target["target_sha256"], "worlds": {}, "privacy": {}, "control": dict(job["control"])}
    for name, member in (("W1", True), ("W0", False)):
        records = generalization.world_records(study, cohort, index, config.seed, member, job["clients"])
        grounded.check_world(study, cohort, records, index, member)
        world = {"partitions": [[encode(r) for r in part] for part in records], "target": encode(target["training"]),
                 "held_out": encode(target["hold"]), "member": member}
        model, history, _ = trainer.train(world, artifact / name.lower(), timings, name)
        body["privacy"][name] = {"epsilon": "inf", "mechanism": "none"}
        body[f"federated_history_{name}"] = history
        if member:
            started = _now()
            body["control"]["memorization_loss"] = lora.memorize(model, tokenizer, passage, job["control"]["steps"],
                                                                 job["control"]["lr"], config.seed + 7777)
            timings["memorization_seconds"] = _now() - started
        bundle = _bundle(model, tokenizer)
        started = _now()
        body["worlds"][name] = {
            "learning": learning_measures(bundle, passage),
            "retention": retention(bundle, session, target["probes"], name),
            "natural_questions": {"training_only" if member else "neither":
                                  natural_questions(bundle, session, passage, target["probes"], session.library(),
                                                    "library_off", name)}}
        timings[f"{name}_measurement_seconds"] = _now() - started
        del model, bundle
        _release()
    return body


def _p0(job, study, config, session, artifact, timings, tokenizer, model_config, device):
    """The cell's pretrained model (in its training dtype): the datastore check and lean utility.
    The result layout matches the parent's P0 job, so the parent's analyze_datastore applies."""
    if job["training"]["method"] == "full":
        from .federation import load_reference_bundle
        model = load_reference_bundle(config)["model"]
    else:
        model = Trainer(job, config, tokenizer, device).base()
    bundle = _bundle(model, tokenizer)
    body = {"privacy": {"P0": {"epsilon": "inf", "mechanism": "none", "note": "no fine-tuning"}}, "worlds": {"P0": {}}}
    started = _now()
    rows = []
    for index in job.get("datastore_targets", []):
        target = session.cohort["targets"][index]
        document = session.passage(target["passage"])
        rows.append({"target_index": index, "target_sha256": target["target_sha256"],
                     "cells": {"library_on": natural_questions(bundle, session, document, target["probes"],
                                                               session.library(document), "library_on", "P0"),
                               "library_off": natural_questions(bundle, session, document, target["probes"],
                                                                session.library(), "library_off", "P0")}})
    if rows:
        body["worlds"]["P0"]["datastore"] = rows
        off = session.library()
        body["worlds"]["P0"]["never_documents"] = [
            natural_questions(bundle, session, session.passage(d["passage"]), d["probes"], off, "never_library", "P0")
            for d in session.cohort["N"][:job["measurements"].get("n_documents", len(session.cohort["N"]))]]
    timings["datastore_seconds"] = _now() - started
    if job["measurements"].get("utility"):
        started = _now()
        body["worlds"]["P0"]["utility"] = lean_utility(bundle, session, session.library(), "P0")
        timings["utility_seconds"] = _now() - started
    del bundle, model
    _release()
    return body
