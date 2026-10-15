"""Generalization study (outputs/grounded_generalization_proposal_20261012): Stage 0 code.

Model profiles and the pinned template, the non-IID worlds, shadow worlds, the synthetic
records, LoRA adapters, in-process FedAvg (bit-identical to the Flower path on full
parameters), weight scores with raw NLLs, LiRA/RMIA statistics, and the study tool's
prepare / resolve / check / analyze rules. CPU only; no Hub access.
"""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from master_script.core import generalization, grounded
from tests.grounded_fixtures import SMALL_LAYOUT, SMALL_SIZES, small_study
from tests.test_grounded_data import MODEL_CONFIG, PASSAGE, RECORD, SETTINGS, ChatTokenizer, examples


class WordTokenizer:
    """Word-level tokenizer with a ChatML-shaped template (ids stay below 50), offsets included."""
    eos_token, eos_token_id, pad_token, pad_token_id, name_or_path = "<|im_end|>", 1, "<pad>", 0, "fake-words"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if "date_string" in kwargs:
            text = f"Today Date: {kwargs['date_string']}\n" + text
        return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def _spans(self, text):
        spans, start = [], None
        for i, ch in enumerate(text + " "):
            if ch.isspace():
                if start is not None:
                    spans.append((start, i))
                    start = None
            elif start is None:
                start = i
        return spans

    def encode(self, text, add_special_tokens=False):
        return [sum(map(ord, text[a:b])) % 45 + 3 for a, b in self._spans(text)]

    def __call__(self, text, truncation=False, max_length=None, return_offsets_mapping=False, return_tensors=None, **kwargs):
        spans = self._spans(text)
        if truncation and max_length:
            spans = spans[:max_length]
        out = {"input_ids": [sum(map(ord, text[a:b])) % 45 + 3 for a, b in spans]}
        if return_offsets_mapping:
            out["offset_mapping"] = spans
        if return_tensors == "pt":
            import torch
            out = {"input_ids": torch.tensor([out["input_ids"]]), "attention_mask": torch.ones(1, len(spans), dtype=torch.long)}
        return out


@pytest.fixture
def tiny():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.GPT2Config(vocab_size=50, n_positions=512, n_embd=16, n_layer=1, n_head=2)
    return transformers.GPT2LMHeadModel(config).eval()


# ------------------------------------------------------------ profiles

def test_every_model_is_pinned_and_every_cell_changes_one_thing():
    for profile in generalization.MODELS.values():
        assert len(profile["revision"]) == 40 and int(profile["revision"], 16) >= 0
    anchor = generalization.CELLS["A-LoRA"]
    for name, spec in generalization.CELLS.items():
        changed = [k for k in ("model", "training", "data", "clients") if spec[k] != anchor[k]]
        assert len(changed) == (0 if name == "A-LoRA" else 1), name
    assert generalization.CELLS["A-full"]["training"] == "full"
    assert {spec["family"] for spec in generalization.CELLS.values()} == {"RQ0", "RQ1", "RQ2", "RQ3", "RQ4"}
    with pytest.raises(ValueError, match="Unknown cell"):
        generalization.cell("Gemma-2B")  # tier 2, not run


def test_the_frozen_protocol_matches_the_code():
    from master_script.tools import grounded_generalization_study as tool
    frozen = tool.protocol()
    assert frozen["status"].startswith("frozen 2026-10-14")
    assert frozen["decisions"]["decided"].startswith("2026-10-14")
    assert frozen["data"]["squad_study_sha256"] == tool.SQUAD_STUDY_SHA256


def test_the_pinned_template_fixes_llamas_date_and_checks_eos():
    profile = generalization.MODELS["llama3.2-1b"]
    tokenizer = generalization.pinned(WordTokenizer(), profile)
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": "q"}], tokenize=False, add_generation_prompt=True)
    assert rendered.startswith("Today Date: 26 Jul 2024")
    assert generalization.template_check(tokenizer)["end_of_turn_id"] == 1
    assert generalization.pinned(WordTokenizer(), generalization.MODELS["qwen2.5-0.5b"]).__class__ is WordTokenizer
    tokenizer.pad_token = "<pad2>"  # attribute writes reach the wrapped tokenizer
    assert tokenizer._tokenizer.pad_token == "<pad2>"

    class Clock(WordTokenizer):
        calls = 0

        def apply_chat_template(self, messages, **kwargs):
            Clock.calls += 1
            return super().apply_chat_template(messages, **{**kwargs, "date_string": str(Clock.calls)})
    with pytest.raises(ValueError, match="renders differently"):
        generalization.template_check(Clock())


def test_pinned_records_encode_identically_to_the_prompt_the_model_answers():
    from master_script.core import rag
    tokenizer = generalization.pinned(WordTokenizer(), {"template_kwargs": {"date_string": "26 Jul 2024"}})
    first = grounded.encode_record(tokenizer, MODEL_CONFIG, RECORD, PASSAGE, "RG", SETTINGS)
    assert first == grounded.encode_record(tokenizer, MODEL_CONFIG, RECORD, PASSAGE, "RG", SETTINGS)
    bundle = {"tokenizer": tokenizer, "model": SimpleNamespace(config=MODEL_CONFIG)}
    prompt = rag._prompt_ids(rag._prepare_prompt(bundle, RECORD["question"], [PASSAGE], SETTINGS))
    assert list(first.input_ids[:len(prompt)]) == prompt and tokenizer.encode("Today") == [prompt[0]]


# ------------------------------------------------------------ non-IID

def test_dirichlet_worlds_share_base_records_and_skew_clients():
    study = small_study()
    sizes = study["sizes"]
    w1 = generalization.world_records(study, "final", 0, 10060, True, "dirichlet")
    w0 = generalization.world_records(study, "final", 0, 10060, False, "dirichlet")
    target = study["cohorts"]["final"]["targets"][0]
    assert [len(p) for p in w1] == [sizes["records_per_client"] + 1] + [sizes["records_per_client"]] * (sizes["clients"] - 1)
    assert w1[1:] == w0[1:] and w1[0][:-1] == w0[0][:-1]
    assert w1[0][-1] == target["training"] and w0[0][-1] == target["hold"]
    grounded.check_world(study, "final", w1, 0, True)
    grounded.check_world(study, "final", w0, 0, False)
    assert generalization.world_records(study, "final", 0, 10060, True, "dirichlet") == w1
    assert generalization.world_records(study, "final", 0, 10060, True, "iid") == grounded.world_records(study, "final", 0, 10060, True)
    with pytest.raises(ValueError, match="iid or dirichlet"):
        generalization.world_records(study, "final", 0, 1, True, "topic")


def test_dirichlet_alpha_point_one_concentrates_each_client_on_few_articles():
    pool = [{"id": f"r{a}-{i}", "article": f"a{a}"} for a in range(16) for i in range(40)]
    article = lambda r: r["article"]
    skewed, iid = [], []
    study = {"cohorts": {"x": {"passages": {r["id"]: {"article": r["article"]} for r in pool}}}}
    as_records = lambda parts: [[{"passage": r["id"]} for r in p] for p in parts]
    for seed in range(20):
        parts, meta = generalization.dirichlet_partitions(pool, article, 4, 32, f"k{seed}", alpha=0.1)
        assert all(len(p) == 32 for p in parts) and len({r["id"] for p in parts for r in p}) == 128
        skewed += generalization.skew_summary(study, "x", as_records(parts))["effective_articles_per_client"]
        mixed, _ = generalization.dirichlet_partitions(pool, article, 4, 32, f"k{seed}", alpha=1000.0)
        iid += generalization.skew_summary(study, "x", as_records(mixed))["effective_articles_per_client"]
    # Measured: about 3.3 effective articles per client at alpha 0.1 against about 12 near IID (of 16).
    assert np.mean(skewed) < 4.5 and np.mean(iid) > 10


def test_skew_summary_reports_structure_only():
    study = small_study()
    summary = generalization.skew_summary(study, "final", generalization.world_records(study, "final", 0, 3, True, "dirichlet"))
    assert set(summary) == {"effective_articles_per_client", "mean_pairwise_js_divergence_nats"}
    assert all(x >= 1 for x in summary["effective_articles_per_client"])


# ------------------------------------------------------------ shadows

@pytest.mark.parametrize("n", [1, 6, 7, 30])
def test_shadow_design_is_balanced(n):
    rows = generalization.shadow_design(n, 16, "key")
    assert len(rows) == 16
    assert all(sum(r[i] for r in rows) == 8 for i in range(n))
    assert all(abs(sum(r) - n / 2) <= 0.5 for r in rows)
    assert rows == generalization.shadow_design(n, 16, "key") != generalization.shadow_design(n, 16, "other") or n == 1
    with pytest.raises(ValueError, match="even"):
        generalization.shadow_design(n, 5, "key")


@pytest.mark.parametrize("scheme", ["iid", "dirichlet"])
def test_shadow_worlds_hold_exactly_their_in_targets(scheme):
    study = small_study()
    c = study["cohorts"]["final"]
    targets = [t["index"] for t in c["targets"]][:4]
    row = [True, False, True, False]
    parts = generalization.shadow_records(study, "final", targets, row, 10150, scheme)
    assert [len(p) for p in parts] == [study["sizes"]["records_per_client"]] * study["sizes"]["clients"]
    generalization.check_shadow_world(study, "final", parts, targets, row)
    ids = {r["id"] for p in parts for r in p}
    for index, inside in zip(targets, row):
        assert (c["targets"][index]["training"]["id"] in ids) is inside
    blocked = {c["targets"][i]["passage"] for i, inside in zip(targets, row) if not inside}
    assert not blocked & {r["passage"] for p in parts for r in p}
    assert parts == generalization.shadow_records(study, "final", targets, row, 10150, scheme)
    with pytest.raises(ValueError, match="design row"):
        generalization.check_shadow_world(study, "final", parts, targets, [False, False, True, False])


# ------------------------------------------------------------ synthetic records

def test_synthetic_rows_are_deterministic_with_exact_spans_and_distinct_answers():
    rows = generalization.synthetic_rows(organizations=6, people=4, seed=7)
    assert rows == generalization.synthetic_rows(organizations=6, people=4, seed=7)
    assert rows != generalization.synthetic_rows(organizations=6, people=4, seed=8)
    by_context = {}
    for row in rows:
        start, answer = row["answers"]["answer_start"][0], row["answers"]["text"][0]
        assert row["context"][start:start + len(answer)] == answer
        by_context.setdefault(row["context"], []).append(row)
    assert len(by_context) == 24 and {len(v) for v in by_context.values()} == {len(generalization._ATTRIBUTES)}
    for context, group in by_context.items():
        assert 24 <= len(context.split()) <= 160
        answers = [grounded._tokens(r["answers"]["text"][0]) for r in group]
        assert all(a and not grounded._contains(a, b) for a in answers for b in answers if a is not b)
    assert len({r["title"] for r in rows}) == 6


def test_the_full_synthetic_build_meets_the_parent_layout():
    study = grounded.build_study(generalization.synthetic_rows(), set(), source=dict(generalization.SYNTH_SOURCE))
    assert study["audit"]["V"]["targets"] >= 90 and study["audit"]["final"]["targets"] >= 40
    assert not study["skips"]["rows"].get("passage_outside_word_range")
    assert all(len(t["probes"]) == 3 for t in study["cohorts"]["final"]["targets"])


# ------------------------------------------------------------ LoRA

def test_lora_starts_as_the_base_trains_only_adapters_and_switches_off(tiny):
    import torch
    from master_script.core import lora
    base = deepcopy(tiny)
    ids = torch.tensor([[3, 4, 5, 6, 7]])
    names = lora.attach(tiny, rank=4, alpha=8, seed=1)
    assert names == lora.target_module_names(base) and all("lm_head" not in n for n in names)
    assert torch.equal(tiny(input_ids=ids).logits, base(input_ids=ids).logits)
    assert all(not p.requires_grad for n, p in tiny.named_parameters() if "lora_" not in n)
    with torch.no_grad():
        for _, p in lora.adapter_parameters(tiny):
            p.add_(0.1)
    assert not torch.allclose(tiny(input_ids=ids).logits, base(input_ids=ids).logits)
    with lora.adapters_disabled(tiny):
        assert torch.equal(tiny(input_ids=ids).logits, base(input_ids=ids).logits)
    arrays = lora.adapter_arrays(tiny)
    other = deepcopy(base)
    lora.attach(other, rank=4, alpha=8, seed=2)
    lora.set_adapter_arrays(other, arrays)
    assert torch.equal(other(input_ids=ids).logits, tiny(input_ids=ids).logits)
    with pytest.raises(ValueError, match="already"):
        lora.attach(tiny, rank=4, alpha=8)
    with pytest.raises(ValueError, match="shape"):
        lora.set_adapter_arrays(other, [a[:1] for a in arrays])


def test_lora_attach_seed_fixes_the_init_without_touching_the_training_stream(tiny):
    import torch
    from master_script.core import lora
    a, b = deepcopy(tiny), deepcopy(tiny)
    torch.manual_seed(5)
    lora.attach(a, rank=2, alpha=4, seed=9)
    after_a = torch.rand(1)
    torch.manual_seed(5)
    lora.attach(b, rank=2, alpha=4, seed=9)
    after_b = torch.rand(1)
    assert all(np.array_equal(x, y) for x, y in zip(lora.adapter_arrays(a), lora.adapter_arrays(b)))
    assert torch.equal(after_a, after_b)


def _flower_training(monkeypatch):
    """Drive amia.federated_fine_tune's real clients and strategy with Flower's serialization, without Ray."""
    import flwr.client
    import flwr.server
    from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
    from master_script.core import runtime_memory
    monkeypatch.setattr(flwr.server, "ServerApp", lambda server_fn: SimpleNamespace(server_fn=server_fn))
    monkeypatch.setattr(flwr.client, "ClientApp", lambda client_fn: SimpleNamespace(client_fn=client_fn))

    def simulate(server_app, client_app, num_supernodes, backend_config):
        components = server_app.server_fn(None)
        strategy, rounds = components.strategy, components.config.num_rounds
        current = strategy.initial_parameters
        for round_id in range(1, rounds + 1):
            fit_config = strategy.on_fit_config_fn(round_id)
            results = []
            for node in range(num_supernodes):
                client = client_app.client_fn(SimpleNamespace(node_config={"partition-id": node})).numpy_client
                arrays, n, metrics = client.fit(parameters_to_ndarrays(current), fit_config)
                results.append((None, FitRes(status=Status(code=Code.OK, message=""), parameters=ndarrays_to_parameters(arrays),
                                             num_examples=n, metrics=metrics)))
            current, _ = strategy.aggregate_fit(round_id, results, [])
    monkeypatch.setattr(runtime_memory, "run_simulation", simulate)


class SavingTokenizer(ChatTokenizer):
    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


def test_in_process_fedavg_is_bit_identical_to_the_flower_path_on_full_parameters(tiny, tmp_path, monkeypatch):
    from master_script.core import lora
    from master_script.core.attacks import amia
    monkeypatch.setattr(amia, "build_model_and_tokenizer", lambda config: (deepcopy(tiny), SavingTokenizer()))
    _flower_training(monkeypatch)
    a, b, c = examples()
    extra = grounded.encode_record(ChatTokenizer(), MODEL_CONFIG, dict(RECORD, id="r9", answer="Seine"), PASSAGE, "CB-AO", SETTINGS)
    partitions = [[a, b, extra], [c]]
    config = amia.AmiaConfig(max_length=384, num_clients=2, clients_per_round=2, federated_rounds=3, local_batch_size=2,
                             client_lr=1e-2, seed=8123)
    flower_model, *_ = amia.federated_fine_tune(config, tmp_path / "flower", None,
                                                world={"partitions": partitions, "target": None, "held_out": None, "member": None})
    client_cls = amia._ami_flower_client_cls()

    def update(arrays, partition_id, round_id):
        return client_cls(partition_id, partitions[partition_id], config).fit(arrays, {"server_round": round_id})
    arrays, history = lora.fedavg_rounds(config, amia.get_parameters(tiny), update)
    assert len(history) == 3 and [h["selected_clients"] for h in history] == [[0, 1]] * 3
    expected = amia.get_parameters(flower_model)
    assert len(arrays) == len(expected) and all(np.array_equal(x, y) for x, y in zip(arrays, expected))
    assert not all(np.array_equal(x, y) for x, y in zip(arrays, amia.get_parameters(tiny)))


def test_lora_federated_training_moves_only_adapters_and_repeats_exactly(tiny):
    from master_script.core import lora
    from master_script.core.attacks import amia
    a, b, c = examples()
    world = {"partitions": [[a, b], [c]], "target": a, "held_out": None, "member": True}
    config = amia.AmiaConfig(max_length=384, num_clients=2, clients_per_round=2, federated_rounds=2, local_batch_size=2, seed=3)
    runs = []
    for _ in range(2):
        model = deepcopy(tiny)
        frozen = {k: v.clone() for k, v in model.state_dict().items()}
        lora.attach(model, rank=4, alpha=8, seed=config.seed)
        start = lora.adapter_arrays(model)
        history, provenance = lora.federated_fine_tune(config, model, ChatTokenizer(), world, lr=1e-2)
        runs.append(lora.adapter_arrays(model))
        assert [h["round"] for h in history] == [1, 2] and provenance
        assert not all(np.array_equal(x, y) for x, y in zip(start, runs[-1]))
        state = model.state_dict()
        assert all(np.array_equal(state[k.replace("c_attn.", "c_attn.base.").replace("c_proj.", "c_proj.base.")
                                       .replace("c_fc.", "c_fc.base.")].numpy(), v.numpy())
                   for k, v in frozen.items() if k in state or k.replace("c_attn.", "c_attn.base.") in state)
        assert all(not p.requires_grad for p in model.parameters())
    assert all(np.array_equal(x, y) for x, y in zip(*runs))
    with pytest.raises(ValueError, match="membership"):
        lora.federated_fine_tune(config, deepcopy(model), ChatTokenizer(), {**world, "member": False}, lr=1e-2)


def test_lora_memorization_trains_the_adapters_only(tiny):
    import torch
    from master_script.core import lora
    tokenizer = ChatTokenizer()
    lora.attach(tiny, rank=4, alpha=8, seed=1)
    base = {n: p.clone() for n, p in tiny.named_parameters() if "lora_" not in n}
    losses = lora.memorize(tiny, tokenizer, "the tower was built in paris for the fair", steps=6, lr=1e-2, seed=4)
    assert len(losses) == 6 and losses[-1] < losses[0]
    assert all(torch.equal(p, base[n]) for n, p in tiny.named_parameters() if "lora_" not in n)


# ------------------------------------------------------------ weight scores

def test_weight_scores_reproduce_the_parent_ratios_and_record_raw_nlls(tiny):
    from master_script.core import generalization_job, lora
    from master_script.core.attacks.reference import score_candidate_hf
    from master_script.core.grounded_job import reference_scores
    tokenizer = WordTokenizer()
    reference = deepcopy(tiny)
    lora.attach(tiny, rank=4, alpha=8, seed=1)
    import torch
    with torch.no_grad():
        for _, p in lora.adapter_parameters(tiny):
            p.add_(0.05)
    bundle = {"model": tiny, "tokenizer": tokenizer, "device": "cpu"}
    ref_bundle = {"model": reference, "tokenizer": tokenizer, "device": "cpu"}
    closed = "Question: who built the tower? Answer: Gustave Eiffel"
    template = grounded.encode_record(tokenizer, MODEL_CONFIG, RECORD, PASSAGE, "RG", SETTINGS)

    def adapters_off(fn):
        with lora.adapters_disabled(tiny):
            return fn(bundle)
    scores = generalization_job.weight_scores(bundle, adapters_off, [(closed, template)], 384)[0]
    parent = reference_scores(bundle, ref_bundle, closed, template, 384)
    assert scores["record"]["ratio"] == pytest.approx(score_candidate_hf(bundle, ref_bundle, closed, max_length=384), rel=1e-6)
    assert scores["record"]["ratio"] == pytest.approx(parent["record"], rel=1e-6)
    assert scores["template"]["ratio"] == pytest.approx(parent["template"], rel=1e-6)
    assert scores["record"]["tokens"] == len(closed.split()) - 1
    assert scores["template"]["tokens"] == sum(label != -100 for label in template.labels)
    assert scores["template"]["ratio"] == pytest.approx(-scores["template"]["model_nll"] / scores["template"]["reference_nll"])


# ------------------------------------------------------------ statistics

def test_lira_separates_in_from_out_and_needs_both():
    from master_script.tools import grounded_generalization_study as tool
    from master_script.tools.grounded_fl_study import auc
    rng = np.random.default_rng(1)
    n, shadows = 30, 16
    difficulty = rng.normal(0, 3, n)  # per-target offsets the pretrained-reference ratio cannot remove
    mask = np.array(generalization.shadow_design(n, shadows, "x"))
    shadow_phi = difficulty + 1.0 * mask + rng.normal(0, 0.3, (shadows, n))
    fit = tool.lira_fit(shadow_phi, mask)
    member = difficulty + 1.0 + rng.normal(0, 0.3, n)
    nonmember = difficulty + rng.normal(0, 0.3, n)
    assert auc(tool.lira_scores(member, fit), tool.lira_scores(nonmember, fit)) > 0.9
    assert auc(member, nonmember) < 0.8  # without calibration, target difficulty swamps the signal
    assert auc(tool.lira_offline_scores(member, fit), tool.lira_offline_scores(nonmember, fit)) > 0.9
    with pytest.raises(ValueError, match="IN and one OUT"):
        tool.lira_fit(shadow_phi[:1], mask[:1])


def test_rmia_and_phi():
    from master_script.tools import grounded_generalization_study as tool
    scores = tool.rmia_scores([0.0, 2.0], [0.0, 0.0], [0.5, 1.0, 1.5], [0.0, 0.0, 0.0])
    assert scores.tolist() == [0.0, 1.0]
    row = {"template": {"model_nll": 0.5, "tokens": 4}}
    assert tool.phi(row, "template") == -2.0


# ------------------------------------------------------------ tool

def word_fits():
    tokenizer = WordTokenizer()

    def fits(record, arm):
        grounded.training_record(tokenizer, MODEL_CONFIG, record, record["passage_text"], arm, SETTINGS)
        return True
    return tokenizer, fits


@pytest.fixture(scope="module")
def synth_study(tmp_path_factory):
    from master_script.tools import grounded_generalization_study as tool
    tokenizer, _ = word_fits()
    out = tmp_path_factory.mktemp("synth") / "build"
    report = tool.build_synth(out, tokenizer=tokenizer, model_config=MODEL_CONFIG)
    assert report["study_sha256"] == grounded.study_sha256(json.loads((out / "study.json").read_bytes()))
    return out / "study.json"


def test_v_slices_come_after_every_parent_slice(synth_study):
    from master_script.tools import grounded_fl_study as parent
    from master_script.tools import grounded_generalization_study as tool
    study = json.loads(synth_study.read_bytes())
    slices = tool.v_allocation(study)
    used = parent.allocation(study)
    assert slices["datastore"] == used["datastore"]
    earlier = used["control"] + used["pilot"] + used["bridge"] + used["timing"] + used["datastore"]
    fresh = slices["control"] + slices["pilot"] + slices["smoke"]
    assert min(fresh) > max(earlier) and len(set(fresh)) == len(fresh) == 17
    assert [tool.v_seed(slices, i) for i in fresh] == list(range(8100, 8117))


def test_prepare_refuses_steps_a_cell_does_not_need_and_foreign_studies(synth_study, tmp_path):
    from master_script.tools import grounded_generalization_study as tool
    with pytest.raises(ValueError, match="not a Stage 0 step"):
        tool.prepare("control", "Non-IID", tmp_path / "a", synth_study)
    with pytest.raises(ValueError, match="not a Stage 0 step"):
        tool.prepare("datastore", "A-full", tmp_path / "b", synth_study)
    with pytest.raises(ValueError, match="reuse the parent build"):
        tool.prepare("smoke", "A-LoRA", tmp_path / "c", synth_study)
    study = small_study()
    path = tmp_path / "small.json"
    path.write_text(json.dumps(study))
    with pytest.raises(ValueError, match="synthetic build"):
        tool.prepare("smoke", "Synth", tmp_path / "d", path)


def test_prepare_and_resolve_a_synth_smoke_launch(synth_study, tmp_path):
    from master_script.tools import grounded_generalization_study as tool
    launch = tool.prepare("smoke", "Synth", tmp_path / "smoke", synth_study)
    kinds = [(e["kind"], e["arm"]) for e in launch["jobs"]]
    assert kinds == [("paired", "RG"), ("paired", "CB-AO"), ("shadow", "RG"), ("shadow", "CB-AO")]
    jobs = [json.loads((tmp_path / "smoke" / e["file"]).read_bytes()) for e in launch["jobs"]]
    assert {j["training"]["lr"] for j in jobs} == {2e-4} and {j["training"]["method"] for j in jobs} == {"lora"}
    assert jobs[0]["measurements"] == tool.LEAN and jobs[0]["config"]["seed"] == jobs[1]["config"]["seed"] == 8116
    assert jobs[2]["shadow"] == {"number": 0, "targets": [jobs[0]["target_index"]], "in_row": [True]}
    assert launch["tool_sha256"] == tool.tool_digests() and launch["protocol_status"].startswith("frozen")
    state = tool.execute(tmp_path / "smoke" / "launch.json", resolve_only=True, tokenizer=WordTokenizer(),
                         model_config=MODEL_CONFIG)
    cohort = json.loads((tmp_path / "smoke" / "cohort.json").read_bytes())
    assert len(state["targets"]) == 4 and cohort["template"]["end_of_turn_id"] == 1
    # A changed job file is refused.
    path = tmp_path / "smoke" / launch["jobs"][0]["file"]
    path.write_text(path.read_text().replace('"RG"', '"CB-AO"', 1))
    with pytest.raises(ValueError, match="changed after declaration"):
        tool._load_launch(tmp_path / "smoke")


def test_control_launch_and_its_one_rerun(synth_study, tmp_path):
    from master_script.tools import grounded_generalization_study as tool
    launch = tool.prepare("control", "Synth", tmp_path / "control", synth_study)
    jobs = [json.loads((tmp_path / "control" / e["file"]).read_bytes()) for e in launch["jobs"]]
    assert len(jobs) == 12 and {j["control"]["steps"] for j in jobs} == {6} and {j["arm"] for j in jobs} == {"RG"}
    rerun = tmp_path / "control-analysis.json"
    rerun.write_text(json.dumps({"schema": "generalization_control_analysis_v1", "status": "rerun_at_12_steps", "cell": "Synth"}))
    again = tool.prepare("control", "Synth", tmp_path / "control-12", synth_study, previous=rerun)
    assert {json.loads((tmp_path / "control-12" / e["file"]).read_bytes())["control"]["steps"] for e in again["jobs"]} == {12}
    rerun.write_text(json.dumps({"schema": "generalization_control_analysis_v1", "status": "rerun_at_12_steps", "cell": "Q-3B"}))
    with pytest.raises(ValueError, match="12-step rerun"):
        tool.prepare("control", "Synth", tmp_path / "control-x", synth_study, previous=rerun)


def test_job_validation_refuses_departures_from_the_cell():
    from master_script.core.generalization_job import validate_job
    from master_script.tools import grounded_generalization_study as tool
    study = small_study()
    job = tool.job_document(study, "Q-1.5B", "paired", "RG", 8100, target_index=0, measurements=tool.LEAN)
    assert job["config"]["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct" and job["training"]["rank"] == 16
    for change in ({"training": {**job["training"], "rank": 8}}, {"training": {**job["training"], "lr": 1e-3}},
                   {"defense": {"mechanism": "dp_sgd"}}, {"clients": "dirichlet"},
                   {"config": {**job["config"], "model_revision": "main"}}, {"measurements": {"causal_directions": ["record"]}}):
        with pytest.raises(ValueError):
            validate_job({**job, **change})
    full = tool.job_document(study, "A-full", "paired", "CB-AO", 8100, target_index=0, measurements=tool.LEAN)
    assert full["training"] == {"method": "full", "lr": 2e-5, "dtype": "float32"}


def _rows(kind, arm, result):
    return {"result": {"kind": kind, "arm": arm, "cell": "Synth", "training": {"method": "lora", "lr": 2e-4},
                       "timings": {"total_seconds": 1.0}, **result}}


def test_smoke_learning_check_escalates_once_then_drops():
    from master_script.tools import grounded_generalization_study as tool
    utility = {"F": [{"f1": 0.6}, {"f1": 0.4}]}
    learning = {"federated_history_W1": [{"mean_client_loss": 2.0}, {"mean_client_loss": 1.5}, {"mean_client_loss": 1.0}],
                "worlds": {"W1": {"utility": utility}}}
    rows = [_rows("paired", "RG", learning), _rows("paired", "CB-AO", learning), _rows("shadow", "RG", {})]
    assert tool.analyze_smoke(rows, {"cell": "Synth", "p0_v_utility_mean_f1": 0.4})["learning_check"]["status"] == "passes"
    assert tool.analyze_smoke(rows, {"cell": "Synth", "p0_v_utility_mean_f1": 0.47})["learning_check"]["status"] == "escalate"
    rows[0]["result"]["training"] = {"method": "lora", "lr": 5e-4}
    assert tool.analyze_smoke(rows, {"cell": "Synth", "p0_v_utility_mean_f1": 0.47})["learning_check"]["status"] == "drop_cell"
    with pytest.raises(ValueError, match="belongs to 'A-LoRA'"):
        tool.analyze_smoke(rows, {"cell": "A-LoRA", "p0_v_utility_mean_f1": 0.4})
    for row in rows:
        row["result"]["cell"] = "Non-IID"
    assert tool.analyze_smoke(rows, {"cell": "A-LoRA", "p0_v_utility_mean_f1": 0.4})["learning_check"]["status"] == "passes"
    with pytest.raises(ValueError, match="datastore analysis"):
        tool.analyze_smoke(rows, None)


def _control_rows(learned, steps):
    rows = []
    for i in range(12):
        ok = i < learned
        rows.append({"result": {"cell": "Synth", "target_sha256": f"t{i}", "control": {"steps": steps}, "worlds": {
            "W1": {"learning": {"passage_nll": 0.1 if ok else 0.9, "continuation_match": 1.0 if ok else 0.0},
                   "retention": {"f1": 0.4 + 0.01 * i}, "natural_questions": {"training_only": {"probes": [
                       {"f1": 0.9, "entailment": 0.9, "refused": False}] * 3}}},
            "W0": {"learning": {"passage_nll": 1.0, "continuation_match": 0.0}, "retention": {"f1": 0.1},
                   "natural_questions": {"neither": {"probes": [{"f1": 0.1, "entailment": 0.1, "refused": False}] * 3}}}}}})
    return rows


def test_control_analysis_mirrors_the_parent_with_the_lora_escalation():
    from master_script.tools import grounded_generalization_study as tool
    assert tool.analyze_control(_control_rows(9, 6))["status"] == "rerun_at_12_steps"
    failed = tool.analyze_control(_control_rows(9, 12))
    assert failed["status"] == "failed_to_learn" and failed["h4_primary"] is False
    passed = tool.analyze_control(_control_rows(11, 6))
    assert passed["status"] == "sensitive" and passed["h4_primary"] is True and passed["learned"] == 11


def test_check_never_shows_endpoint_values(synth_study, tmp_path):
    from master_script.core.config import implementation_fingerprint
    from master_script.core.queue import write_json
    from master_script.tools import grounded_generalization_study as tool
    root = tmp_path / "smoke"
    launch = tool.prepare("smoke", "Synth", root, synth_study)
    tool.execute(root / "launch.json", resolve_only=True, tokenizer=WordTokenizer(), model_config=MODEL_CONFIG)
    state = json.loads((root / "progress.json").read_bytes())
    job = json.loads((root / launch["jobs"][0]["file"]).read_bytes())
    result = {"schema": "generalization_result_v1", "status": "complete", "kind": "paired", "cell": "Synth", "arm": "RG",
              "seed": 8116, "target_sha256": state["targets"]["0"], "implementation_fingerprint": implementation_fingerprint(),
              "job_sha256": sha256(json.dumps(job, sort_keys=True).encode()).hexdigest(),
              "training": {"method": "lora", "lr": 2e-4, "dtype": "bfloat16", "secret_auc": 0.987},
              "timings": {"W1_training_seconds": 5.0, "auc_seconds_leak": "0.91", "f1": 0.777},
              "sequence_lengths": {"max_tokens": 200, "mean_tokens": 150.0, "auc": 0.876},
              "client_skew": {"effective_articles_per_client": [1.0], "rg_f1": 0.555},
              "worlds": {"W1": {"reference": {"record": {"ratio": 0.4242}}, "population": [{}, {}]}}}
    path = root / "results" / "job-000" / "result.json"
    path.parent.mkdir(parents=True)
    write_json(path, result)
    from master_script.core.checkpoint_retention import file_digest
    state["completed"].append({"job": 0, "result": "results/job-000/result.json", "sha256": file_digest(path), "reclaimed_bytes": 0})
    write_json(root / "progress.json", state)
    report = json.dumps(tool.check(root))
    for leaked in ("0.987", "0.91", "0.777", "0.876", "0.555", "0.4242", "secret_auc"):
        assert leaked not in report
    assert '"W1_population_rows": 2' in report


# ------------------------------------------------------------ job orchestration

def test_the_paired_job_measures_lean_and_trains_both_worlds_by_scheme(monkeypatch, tmp_path):
    from master_script.core import generalization_job as gj
    study = small_study()
    cohort = study["cohorts"]["final"]
    calls = []

    class FakeTrainer:
        def __init__(self, job, config, tokenizer, device):
            self.job = job

        def train(self, world, directory, timings, name):
            calls.append(("train", name, world["member"], [len(p) for p in world["partitions"]]))
            return SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device="cpu")]), eval=lambda: None), [{"round": 1}], {}

        def reference(self, model):
            return lambda fn: None
    session = SimpleNamespace(cohort=cohort, settings=SETTINGS, passage=lambda p: cohort["passages"][p]["text"],
                              library=lambda text=None: ("library", text))
    monkeypatch.setattr(gj, "Trainer", FakeTrainer)
    monkeypatch.setattr(gj, "_bundle", lambda model, tokenizer: {"model": model})
    monkeypatch.setattr(gj, "weight_scores", lambda bundle, ref, items, max_length: [{"n": i} for i in range(len(items))])
    monkeypatch.setattr(gj, "retention", lambda *a: calls.append(("retention", a[-1])) or {"f1": 0.0})
    monkeypatch.setattr(gj, "natural_questions", lambda *a: calls.append(("nq", a[-2], a[-1])) or {})
    monkeypatch.setattr(gj, "lean_utility", lambda *a: calls.append(("utility", a[-1])) or {"F": []})
    monkeypatch.setattr(gj, "_release", lambda: None)
    monkeypatch.setattr(grounded, "template_example", lambda *a: "template")
    monkeypatch.setattr(grounded, "training_record",
                        lambda tok, cfg, r, passage, arm, settings, max_length=512: SimpleNamespace(input_ids=(1, 2, 3), id=r["id"]))
    from master_script.tools import grounded_generalization_study as tool
    job = {**tool.job_document(study, "Non-IID", "paired", "RG", 10060, target_index=0, measurements=tool.LEAN), "cohort": "final"}
    config = SimpleNamespace(seed=10060, max_length=512)
    monkeypatch.setattr(gj, "validate_partition_tokens", lambda *a, **k: None, raising=False)
    body = gj._paired(job, study, config, session, tmp_path, {}, None, MODEL_CONFIG, "cpu")
    trains = [c for c in calls if c[0] == "train"]
    assert [(c[1], c[2]) for c in trains] == [("W1", True), ("W0", False)]
    assert [c for c in calls if c[0] == "utility"] == [("utility", "W1")]
    assert ("nq", "never_library", "W0") in calls and not any(c[0] == "nq" and c[1] == "never_library" and c[2] == "W1" for c in calls)
    assert {c[1] for c in calls if c[0] == "nq" and c[2] == "W1"} == {"both", "training_only"}
    assert len(body["worlds"]["W1"]["population"]) == min(tool.POPULATION, len(cohort["records"]["U"]))
    assert body["client_skew"]["mean_pairwise_js_divergence_nats"] >= 0
    expected = generalization.world_records(study, "final", 0, 10060, True, "dirichlet")
    assert trains[0][3] == [len(p) for p in expected]
