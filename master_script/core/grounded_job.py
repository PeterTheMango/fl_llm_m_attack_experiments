"""One grounded-study job on the GPU: paired worlds, the positive control, RG-public or P0.

Run by `python -m master_script.tools.grounded_fl_study job`, one job per process,
after the tool's CPU resolution. Measurements follow proposal §6 exactly:

- Reference: Carlini log-perplexity ratio against the pretrained model, in the
  record form (today's closed-book text) and the template form (answer-only
  NLL given the arm's own prompt).
- Causal: the validated request (cosine, probe_epochs 12, probe_lr 0.005) in W1,
  40 counterbalanced trials per attacker direction, plus a release-noise pass.
- Natural questions: each document's 3 probes through retrieval (top-k over the
  private library with the slot holding the target or the filler), scored by F1
  and a pinned NLI model from the same responses; refusals are flagged.
- Retention: the same probes with an empty context slot.
- Utility: RAG F1 on F, public-library F1 on F_P, no-context F1 and answer NLL.

Raw responses are written only to a mode-0600 private audit file; results carry
scores, flags and hashes. Nothing here makes a privacy claim.
"""
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import time

from . import grounded
from .attacks import amia
from .config import METHOD_VERSION, implementation_fingerprint, resolve_run_config, validate_attack_config
from .pipeline import Defense, Pipeline
from .queue import write_json

JOB_SCHEMA, RESULT_SCHEMA = "grounded_job_v1", "grounded_result_v1"
KINDS = ("paired", "control", "public", "p0")
DIRECTIONS = ("record", "template")
CAUSAL = {"attack_variant": "causal_gradient_alignment", "causal_score": "cosine", "probe_epochs": 12,
          "probe_lr": 0.005, "attack_trials": 40, "counterbalance_trials": True, "attack_targets": 1}
RELEASE_NOISE = {"observation_defense": "gaussian", "observation_noise_multiplier": 1.0, "observation_clip_norm": 1.0}
MEASUREMENTS = ("reference", "retention", "utility", "nq", "causal_directions", "release_noise", "n_documents")
CONTINUATION = 32


def amia_config(values):
    """The job's AmiaConfig. Only the two model-selection keys AmiaConfig lacks are
    accepted beyond its fields, and they must name the real pretrained base model."""
    from dataclasses import fields
    names = {f.name for f in fields(amia.AmiaConfig)}
    extra = set(values) - names
    if extra - {"use_hf_models", "reference_model_id"}:
        raise ValueError(f"Unknown config fields: {sorted(extra)}")
    if values.get("use_hf_models", True) is not True or values.get("reference_model_id", values["model_id"]) != values["model_id"]:
        raise ValueError("Grounded jobs use real models, with the pretrained base model as the reference")
    return amia.AmiaConfig(**{k: v for k, v in values.items() if k in names})


def validate_job(job):
    """Structural checks shared by prepare, resolve and execution."""
    if job.get("schema") != JOB_SCHEMA or job.get("kind") not in KINDS:
        raise ValueError("Not a grounded job")
    kind, arm, defense = job["kind"], job["arm"], job["defense"]
    allowed = {"paired": ("RG", "CB-AO", "CB-LM"), "control": ("RG",), "public": ("RG-public",), "p0": ("P0",)}
    if arm not in allowed[kind]:
        raise ValueError(f"{kind} jobs train {allowed[kind]}, not {arm!r}")
    if defense.get("mechanism") not in ("none", "dp_sgd") or (kind != "paired" and defense["mechanism"] != "none"):
        raise ValueError("Only paired jobs may use DP-SGD")
    if job["cohort"] not in grounded.COHORTS:
        raise ValueError("Unknown cohort")
    measurements = job["measurements"]
    if set(measurements) - set(MEASUREMENTS):
        raise ValueError("Unknown measurement")
    directions = measurements.get("causal_directions", [])
    if any(d not in DIRECTIONS for d in directions) or len(set(directions)) != len(directions):
        raise ValueError("causal_directions must be distinct record/template")
    if directions and kind != "paired":
        raise ValueError("The causal attack runs in paired member worlds only")
    if measurements.get("release_noise") and not directions:
        raise ValueError("Release noise is a second pass of the causal attack")
    config = job["config"]
    if directions and any(config.get(k) != v for k, v in CAUSAL.items()):
        raise ValueError("The causal attack must use the validated configuration set explicitly: "
                         "cosine score, probe_epochs 12, probe_lr 0.005, 40 counterbalanced trials")
    if kind in ("paired", "control") and not isinstance(job.get("target_index"), int):
        raise ValueError("Paired and control jobs name one target")
    if kind == "p0" and not job.get("datastore_targets") and not measurements.get("utility"):
        raise ValueError("A P0 job measures the datastore, utility or both")
    if kind == "control":
        control = job.get("control", {})
        if control.get("epochs") not in (3, 6) or control.get("lr") != 1e-4:
            raise ValueError("The positive control fine-tunes 3 epochs (or the one 6-epoch rerun) at lr 1e-4")
    grounded.RagSettings(**job["rag"])
    amia_config(config)
    return True


# ------------------------------------------------------------------ helpers

def _now():
    return time.perf_counter()


def _sha(text):
    return sha256(text.encode()).hexdigest()


class Embedder:
    """The pinned retrieval encoder on the CPU, with rag.embed's pooling and normalization."""

    def __init__(self, settings):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model, revision=settings.embedding_revision)
        self.model = AutoModel.from_pretrained(settings.embedding_model, revision=settings.embedding_revision).to("cpu").eval()

    def __call__(self, texts):
        import numpy as np
        import torch
        chunks = []
        with torch.no_grad():
            for start in range(0, len(texts), 32):
                tokens = self.tokenizer(texts[start:start + 32], padding=True, truncation=True,
                                        max_length=256, return_tensors="pt")
                hidden = self.model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                chunks.append(torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy())
        return np.concatenate(chunks)


class Scorers:
    """The pinned embedding and NLI models, loaded once per job."""

    def __init__(self, settings, nli, device):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.settings, self.device = settings, device
        self.embed = Embedder(settings)
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli["model"], revision=nli["revision"])
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli["model"], revision=nli["revision"]).to(device).eval()
        labels = {int(k): v.lower() for k, v in self.nli_model.config.id2label.items()}
        entail = [k for k, v in labels.items() if v == "entailment"]
        if len(entail) != 1:
            raise ValueError("The pinned NLI model has no single entailment label")
        self.entailment_index = entail[0]

    def entail(self, premises, hypotheses):
        import torch
        with torch.no_grad():
            tokens = self.nli_tokenizer(list(premises), list(hypotheses), truncation="only_first", max_length=512,
                                        padding=True, return_tensors="pt").to(self.device)
            probs = torch.softmax(self.nli_model(**tokens).logits.float(), dim=-1)
        return [float(p) for p in probs[:, self.entailment_index].cpu()]


class Library:
    """A fixed document list with one slot: filler (library-off) or the target passage (library-on)."""

    def __init__(self, texts, vectors):
        self.texts, self.vectors = list(texts), vectors

    def with_slot(self, text, vector):
        import numpy as np
        return Library(self.texts + [text], np.vstack([self.vectors, vector[None, :]]))

    def ranking(self, query_vector):
        import numpy as np
        return np.argsort(-(self.vectors @ query_vector), kind="stable").tolist()


class Session:
    """Shared state of one job: settings, scorers, libraries and the private audit file."""

    def __init__(self, job, study, audit_directory, device):
        self.job, self.study = job, study
        self.cohort = study["cohorts"][job["cohort"]]
        self.settings = grounded.RagSettings(**job["rag"])
        self.audit_settings = type("Audit", (), {"runtime_audit_directory": str(audit_directory)})()
        self.gate = job.get("gate")
        self.scorers = Scorers(self.settings, job["nli"], device)
        text = lambda p: self.cohort["passages"][p]["text"]
        fixed = [text(p) for p in self.cohort["library"]["fixed"]]
        self.private = Library(fixed, self.scorers.embed(fixed))
        self.filler_text = text(self.cohort["library"]["filler"])
        self.filler_vector = self.scorers.embed([self.filler_text])[0]
        public = [text(p) for p in self.cohort["public_library"]]
        self.public = Library(public, self.scorers.embed(public))
        self._queries = {}

    def passage(self, pid):
        return self.cohort["passages"][pid]["text"]

    def library(self, slot_text=None):
        if slot_text is None:
            return self.private.with_slot(self.filler_text, self.filler_vector)
        return self.private.with_slot(slot_text, self.scorers.embed([slot_text])[0])

    def query_vector(self, query):
        if query not in self._queries:
            self._queries[query] = self.scorers.embed([query])[0]
        return self._queries[query]

    def audit(self, row):
        from .rag import audit_answer
        audit_answer(self.audit_settings, row)


def _retrieve(session, library, query):
    ranking = library.ranking(session.query_vector(query))
    gate = None
    if session.gate is not None:
        _, withheld, overlap = grounded.gated_ranking(ranking, library.texts, query, session.gate["threshold"],
                                                      session.gate["ngram"])
        gate = {"withheld_top": withheld, "overlap": overlap}
    return ranking, [library.texts[i] for i in ranking[:session.settings.top_k]], gate


def _generate(bundle, session, question, contexts):
    from .rag import generate_answer
    return generate_answer(bundle, question, contexts, session.settings)


def natural_questions(bundle, session, document, probes, library, cell, world):
    """3 probe queries for one document through retrieval, both scorers, plus one verbatim query."""
    from .rag import is_refusal, parse_membership_answer
    rows, responses = [], []
    for probe in probes:
        ranking, contexts, gate = _retrieve(session, library, probe["question"])
        response = _generate(bundle, session, probe["question"], contexts)
        responses.append(response)
        rank = next((position for position, i in enumerate(ranking) if library.texts[i] == document), None)
        session.audit({"kind": "natural_question", "world": world, "cell": cell, "probe_id": probe["id"],
                       "document_sha256": _sha(document), "answer": response})
        rows.append({"probe_id": probe["id"], "document_rank": rank, "retrieved": document in contexts,
                     "gate": gate, "response_sha256": _sha(response), "_refused": is_refusal(response)})
    entailment = session.scorers.entail([document] * len(probes),
                                        [grounded.nli_hypothesis(p["question"], r) for p, r in zip(probes, responses)])
    scored = []
    for row, probe, response, entail in zip(rows, probes, responses, entailment):
        refused = row.pop("_refused")
        scored.append({**row, **grounded.probe_scores(response, probe["answer"], entail, refused)})
    query = f'Does this: "{document}" appear in the context? Answer with Yes or No.'
    ranking, contexts, gate = _retrieve(session, library, query)
    answer = _generate(bundle, session, query, contexts)
    parsed = parse_membership_answer(answer)
    session.audit({"kind": "verbatim_probe", "world": world, "cell": cell, "document_sha256": _sha(document),
                   "answer": answer})
    verbatim = {"score": None if parsed["pred_member"] is None else float(parsed["pred_member"]),
                "refused": parsed["refused"], "retrieved": document in contexts, "gate": gate}
    return {"document_sha256": _sha(document), "probes": scored, "verbatim": verbatim}


def retention(bundle, session, probes, world):
    """Closed-book probes: the same template with an empty context slot; refusals score 0."""
    from .rag import is_refusal
    rows = []
    for probe in probes:
        response = _generate(bundle, session, probe["question"], [])
        session.audit({"kind": "retention", "world": world, "probe_id": probe["id"], "answer": response})
        refused = is_refusal(response)
        rows.append({"probe_id": probe["id"], "refused": refused, "f1": 0.0 if refused else grounded.f1(response, probe["answer"]),
                     "raw_f1": grounded.f1(response, probe["answer"])})
    return {"probes": rows, "f1": sum(r["f1"] for r in rows) / len(rows)}


def utility(bundle, session, library, world):
    """Per-question RAG F1 on F (with EM, refusal, NLL, gate), F_P on the public library, no-context F1."""
    from .rag import answer_nll, answer_utility, is_refusal
    rows = []
    for q in session.cohort["F"]:
        _, contexts, gate = _retrieve(session, library, q["question"])
        response = _generate(bundle, session, q["question"], contexts)
        closed = _generate(bundle, session, q["question"], [])
        session.audit({"kind": "utility", "world": world, "query_id": q["id"], "answer": response, "no_context_answer": closed})
        scored, bare = answer_utility(response, q["answer"]), answer_utility(closed, q["answer"])
        rows.append({"query_id": q["id"], "article": q["article"], "f1": scored["token_f1"], "em": scored["exact_match"],
                     "refused": is_refusal(response), "gold_retrieved": session.passage(q["passage"]) in contexts,
                     "gate": gate, "no_context_f1": bare["token_f1"], "no_context_em": bare["exact_match"],
                     "answer_nll": answer_nll(bundle, q["question"], q["answer"], contexts, session.settings)})
    public = []
    for q in session.cohort["F_P"]:
        _, contexts, _ = _retrieve(session, session.public, q["question"])
        response = _generate(bundle, session, q["question"], contexts)
        session.audit({"kind": "public_utility", "world": world, "query_id": q["id"], "answer": response})
        scored = answer_utility(response, q["answer"])
        public.append({"query_id": q["id"], "f1": scored["token_f1"], "em": scored["exact_match"],
                       "refused": is_refusal(response)})
    return {"F": rows, "F_P": public}


def _nll(bundle, example):
    import torch
    from .scoring import encoded_batch
    batch = encoded_batch([example], bundle["tokenizer"].pad_token_id, bundle["device"])
    with torch.no_grad():
        return float(bundle["model"](**batch).loss)


def reference_scores(bundle, reference, closed_text, template, max_length):
    """Both attacker forms of the Carlini ratio; larger is more member-like."""
    from .attacks.reference import calibrated_reference_score, score_candidate_hf
    return {"record": score_candidate_hf(bundle, reference, closed_text, max_length=max_length),
            "template": calibrated_reference_score(_nll(bundle, template), _nll(reference, template))}


def causal_passes(config, model_path, tokenizer, clients, member_record, candidates, release_noise, directory):
    """Per direction: one optimized request, a 40-trial pass and optionally a release-noise pass."""
    import torch
    from .attacks.causal_probe import optimize_request, raw_gradients
    from .defenses import privacy_bound
    from .metrics import roc_auc
    from .model_io import load_causal_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    for direction, candidate in candidates.items():
        started = _now()
        probe = load_causal_model(model_path).to(device)
        history = optimize_request(probe, tokenizer, candidate, config)
        probe._public_direction = raw_gradients(probe, tokenizer, [candidate], config)
        probe.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        passes = {"plain": config}
        if release_noise:
            passes["release_noise"] = replace(config, **RELEASE_NOISE)
        out[direction] = {"request_loss": history, "request_seconds": _now() - started}
        for name, pass_config in passes.items():
            started = _now()
            trials = amia.run_attack_trials(model_path, probe, clients, pass_config,
                                            checkpoint_dir=Path(directory) / f"{direction}-{name}",
                                            checkpoint_metadata={"direction": direction, "pass": name},
                                            target=member_record, reuse_victim_model=True,
                                            observe_target_only=True)
            clean = [{k: t[k] for k in ("trial_id", "truth_member", "score", "batch_pair_seed", "alignment_terms",
                                        "response_seconds") if k in t} for t in trials]
            out[direction][name] = {"trials": clean, "seconds": _now() - started,
                                    "auc": roc_auc([t["truth_member"] for t in trials], [t["score"] for t in trials])}
            if name == "release_noise":
                bound = privacy_bound(len(trials), pass_config.observation_noise_multiplier, pass_config.observation_delta)
                out[direction][name]["release_privacy"] = {
                    **bound, "privacy_unit": "private_batch", "scope": "these 40 observation releases only; never merged with the training epsilon"}
        del probe
    return out


def memorize(model, tokenizer, passage, epochs, lr, seed):
    """Positive control: central full-LM AdamW on the plain passage text, one step per epoch."""
    import torch
    from .federation import seed_training
    seed_training(seed)
    device = next(model.parameters()).device
    ids = torch.tensor([tokenizer.encode(passage, add_special_tokens=False)], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses = []
    for _ in range(epochs):
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
    model.eval()
    return losses


def learning_measures(bundle, passage):
    """Passage-token NLL and greedy 32-token continuation match given the first 32 tokens."""
    import torch
    ids = bundle["tokenizer"].encode(passage, add_special_tokens=False)
    if len(ids) < 2 * CONTINUATION:
        raise ValueError("Control passages need at least 64 tokens")
    tensor = torch.tensor([ids], device=bundle["device"])
    with torch.no_grad():
        nll = float(bundle["model"](input_ids=tensor, labels=tensor).loss)
        prompt = tensor[:, :CONTINUATION]
        generated = bundle["model"].generate(input_ids=prompt, attention_mask=torch.ones_like(prompt), do_sample=False,
                                             max_new_tokens=CONTINUATION, min_new_tokens=CONTINUATION,
                                             pad_token_id=bundle["tokenizer"].pad_token_id)
    produced = generated[0, CONTINUATION:2 * CONTINUATION].tolist()
    expected = ids[CONTINUATION:2 * CONTINUATION]
    return {"passage_nll": nll, "passage_tokens": len(ids),
            "continuation_match": sum(a == b for a, b in zip(produced, expected)) / CONTINUATION}


# ------------------------------------------------------------------ worlds

def _encoder(config, arm, settings):
    from transformers import AutoConfig, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.model_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_config = AutoConfig.from_pretrained(config.model_id, revision=config.model_revision)
    return tokenizer, model_config


def encode_world(study, cohort, partitions, arm, tokenizer, model_config, settings, max_length):
    text = lambda r: study["cohorts"][cohort]["passages"][r["passage"]]["text"]
    return [[grounded.training_record(tokenizer, model_config, r, text(r), arm, settings, max_length) for r in part]
            for part in partitions]


def _lengths(partitions):
    sizes = [len(x.input_ids) if hasattr(x, "input_ids") else None for part in partitions for x in part]
    sizes = [s for s in sizes if s is not None]
    return {"max_tokens": max(sizes), "mean_tokens": sum(sizes) / len(sizes)} if sizes else {}


def _bundle(model, tokenizer):
    return {"model": model.eval(), "tokenizer": tokenizer, "device": next(model.parameters()).device}


def _release():
    """Callers delete their own model references first; this reclaims the memory."""
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------- run

def run_job(job, study, output_dir):
    """Execute one validated job; returns and writes the result (twice, for checkpoint retirement)."""
    import torch
    validate_job(job)
    if grounded.study_sha256(study) != job["study_sha256"]:
        raise ValueError("Study data differ from the job's declaration")
    started = _now()
    config = amia_config(job["config"])
    validate_attack_config(config)
    config = resolve_run_config(config)
    if config.dataset_revision != study["source"]["revision"]:
        raise ValueError("Pinned dataset revision differs from the study source")
    if config.num_clients != study["sizes"]["clients"] or config.target_client_id != 0:
        raise ValueError("The study's worlds place the target in client 0 of its declared clients")
    output = Path(output_dir).resolve()
    artifact = output / "artifacts" / "result"
    artifact.mkdir(parents=True, exist_ok=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    session = Session(job, study, artifact / "private-audit", device)
    timings = {"setup_seconds": _now() - started}
    kind = job["kind"]
    runner = {"paired": _paired, "control": _control, "public": _public, "p0": _p0}[kind]
    body = runner(job, study, config, session, artifact, timings)
    job_digest = _sha(json.dumps(job, sort_keys=True))
    result = {"schema": RESULT_SCHEMA, "status": "complete", "run_id": f"grounded_{job_digest[:24]}",
              "job_sha256": job_digest, "kind": kind, "arm": job["arm"], "cohort": job["cohort"],
              "epsilon_budget": job.get("epsilon"), "seed": config.seed, "target_index": job.get("target_index"),
              "method_version": METHOD_VERSION, "implementation_fingerprint": implementation_fingerprint(),
              "config": asdict(config), "defense": job["defense"], "rag": job["rag"], "nli": job["nli"],
              "gate": job.get("gate"), "measurements": job["measurements"], **body,
              "timings": {**timings, "total_seconds": _now() - started},
              "audit_outputs_private": True,
              "scope": "Grounded-FL study measurement; attack effectiveness only, no defense or privacy claim"}
    write_json(artifact / "result.json", result)
    write_json(output / "result.json", result)
    return result


def _pipeline(job):
    d = job["defense"]
    return Pipeline(defense=Defense(**{k: d[k] for k in ("mechanism", "clip_norm", "noise_multiplier", "delta") if k in d}))


def _world(job, study, config, session, arm, member, tokenizer, model_config):
    cohort, index = job["cohort"], job["target_index"]
    records = grounded.world_records(study, cohort, index, config.seed, member)
    grounded.check_world(study, cohort, records, index, member)
    target = study["cohorts"][cohort]["targets"][index]
    encode = lambda r: grounded.training_record(tokenizer, model_config, r, session.passage(r["passage"]), arm,
                                                session.settings, config.max_length)
    partitions = encode_world(study, cohort, records, arm, tokenizer, model_config, session.settings, config.max_length)
    return {"partitions": partitions, "target": encode(target["training"]), "held_out": encode(target["hold"]),
            "member": member}


def _train(job, config, world, directory, timings, name):
    started = _now()
    model, tokenizer, clients, history, path = amia.federated_fine_tune(config, directory, _pipeline(job), world=world)
    timings[f"{name}_training_seconds"] = _now() - started
    privacy = grounded.epsilon_record(getattr(model, "_training_privacy", {}), job["defense"]["mechanism"])
    return model, tokenizer, clients, history, path, privacy


def _world_measures(job, session, bundle, reference, target, closed, template, slot_cells, world, timings, n_documents=0):
    m, out = job["measurements"], {}
    started = _now()
    if m.get("reference"):
        out["reference"] = reference_scores(bundle, reference, closed, template, session.job["config"]["max_length"])
    if m.get("retention"):
        out["retention"] = retention(bundle, session, target["probes"], world)
    if m.get("nq"):
        document = session.passage(target["passage"])
        out["natural_questions"] = {
            cell: natural_questions(bundle, session, document, target["probes"],
                                    session.library(document if on else None), cell, world)
            for cell, on in slot_cells.items()}
        if n_documents:
            off = session.library()
            out["never_documents"] = [natural_questions(bundle, session, session.passage(d["passage"]), d["probes"],
                                                        off, "never_library", world)
                                      for d in session.cohort["N"][:n_documents]]
    if m.get("utility"):
        out["utility"] = utility(bundle, session, session.library(session.passage(target["passage"])), world)
    timings[f"{world}_measurement_seconds"] = _now() - started
    return out


def _paired(job, study, config, session, artifact, timings):
    from .federation import load_reference_bundle
    arm = job["arm"]
    tokenizer, model_config = _encoder(config, arm, session.settings)
    target = session.cohort["targets"][job["target_index"]]
    closed = grounded.closed_book_record(target["training"]["question"], target["training"]["answer"])
    template = grounded.template_example(tokenizer, model_config, target["training"], session.passage(target["passage"]),
                                         arm, session.settings, config.max_length)
    worlds = {name: _world(job, study, config, session, arm, member, tokenizer, model_config)
              for name, member in (("W1", True), ("W0", False))}
    body = {"target_sha256": target["target_sha256"], "worlds": {}, "privacy": {},
            "sequence_lengths": _lengths(worlds["W1"]["partitions"])}
    m = job["measurements"]
    model, tok, clients, history, path, privacy = _train(job, config, worlds["W1"], artifact / "w1", timings, "W1")
    # The pretrained reference joins only after training, keeping the GPU free for the clients.
    reference = load_reference_bundle(config)
    body["privacy"]["W1"], body["federated_history_W1"] = privacy, history
    body["training_provenance_W1"] = getattr(model, "_training_provenance", {})
    body["worlds"]["W1"] = _world_measures(job, session, _bundle(model, tok), reference, target, closed, template,
                                           {"both": True, "training_only": False}, "W1", timings)
    del model
    reference["model"].to("cpu")  # parked while the causal passes and W0 use the GPU
    _release()
    if m.get("causal_directions"):
        candidates = {"record": closed, "template": template}
        started = _now()
        body["causal"] = causal_passes(config, path, tok, clients, worlds["W1"]["target"],
                                       {d: candidates[d] for d in m["causal_directions"]}, m.get("release_noise"),
                                       artifact / "observations")
        timings["causal_seconds"] = _now() - started
    model, tok, _, history, _, privacy = _train(job, config, worlds["W0"], artifact / "w0", timings, "W0")
    reference["model"].to(reference["device"])
    body["privacy"]["W0"], body["federated_history_W0"] = privacy, history
    body["training_provenance_W0"] = getattr(model, "_training_provenance", {})
    body["worlds"]["W0"] = _world_measures(job, session, _bundle(model, tok), reference, target, closed, template,
                                           {"library_only": True, "neither": False}, "W0", timings,
                                           n_documents=m.get("n_documents", 0))
    if not grounded.same_epsilon(body["privacy"]["W1"], body["privacy"]["W0"]):
        raise RuntimeError("The paired worlds were accounted at different epsilon")
    del model, reference
    _release()
    return body


def _control(job, study, config, session, artifact, timings):
    tokenizer, model_config = _encoder(config, "RG", session.settings)
    target = session.cohort["targets"][job["target_index"]]
    passage = session.passage(target["passage"])
    worlds = {name: _world(job, study, config, session, "RG", member, tokenizer, model_config)
              for name, member in (("W1", True), ("W0", False))}
    body = {"target_sha256": target["target_sha256"], "worlds": {}, "privacy": {}, "control": dict(job["control"])}
    for name in ("W1", "W0"):
        model, tok, _, history, _, privacy = _train(job, config, worlds[name], artifact / name.lower(), timings, name)
        body["privacy"][name] = privacy
        if name == "W1":
            started = _now()
            body["control"]["memorization_loss"] = memorize(model, tok, passage, job["control"]["epochs"],
                                                            job["control"]["lr"], config.seed + 7777)
            timings["memorization_seconds"] = _now() - started
        bundle = _bundle(model, tok)
        started = _now()
        body["worlds"][name] = {
            "learning": learning_measures(bundle, passage),
            "retention": retention(bundle, session, target["probes"], name),
            "natural_questions": {"training_only" if name == "W1" else "neither":
                                  natural_questions(bundle, session, passage, target["probes"], session.library(),
                                                    "library_off", name)}}
        timings[f"{name}_measurement_seconds"] = _now() - started
        del model, bundle
        _release()
    return body


def _public(job, study, config, session, artifact, timings):
    tokenizer, model_config = _encoder(config, "RG-public", session.settings)
    records = grounded.public_records(study, job["cohort"], config.seed)
    grounded.check_world(study, job["cohort"], records)
    partitions = encode_world(study, job["cohort"], records, "RG-public", tokenizer, model_config, session.settings,
                              config.max_length)
    world = {"partitions": partitions, "target": None, "held_out": None, "member": None}
    model, tok, _, history, _, privacy = _train(job, config, world, artifact / "public", timings, "public")
    started = _now()
    body = {"privacy": {"public": privacy}, "sequence_lengths": _lengths(partitions),
            "worlds": {"public": {"utility": utility(_bundle(model, tok), session, session.library(), "public")}}}
    timings["public_measurement_seconds"] = _now() - started
    del model
    _release()
    return body


def _p0(job, study, config, session, artifact, timings):
    import torch
    from .federation import load_reference_bundle
    pretrained = load_reference_bundle(config)
    bundle = {"model": pretrained["model"], "tokenizer": pretrained["tokenizer"], "device": pretrained["device"]}
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
        body["worlds"]["P0"]["utility"] = utility(bundle, session, session.library(), "P0")
        timings["utility_seconds"] = _now() - started
    del bundle, pretrained
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return body
