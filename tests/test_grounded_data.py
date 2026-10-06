"""Grounded study data: eligibility, splits, exclusions, worlds, prompt parity and the answer-only loss."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from master_script.core import datasets as data_sources
from master_script.core import grounded, rag
from master_script.core.scoring import EncodedExample, encoded_batch, validate_partition_tokens
from tests.grounded_fixtures import SMALL_LAYOUT, SMALL_SIZES, passage_rows, small_study, squad_rows

TOOLS = Path(grounded.__file__).resolve().parents[1] / "tools"


def test_the_validated_causal_tool_is_byte_identical():
    digest = sha256((TOOLS / "causal_attack_validation.py").read_bytes()).hexdigest()
    assert digest == "9acd6cd1e5a119f526d1921f83d3307cbce9b3fdfe2978d76f49bf225f206c9c"


# ------------------------------------------------------------ eligibility

def q(answer, start, question="What is it?", id="x"):
    return {"id": id, "question": question, "answer": answer, "answer_start": start}


def test_squad_normalization():
    assert grounded.normalize_answer("The  Eiffel-Tower!") == "eiffeltower"
    assert grounded.normalize_answer("an Apple, a pear") == "apple pear"


@pytest.mark.parametrize("sibling, ok", [
    (q("Paris", 50), True),
    (q("Louvre", 50), False),            # occurs in the training question
    (q("1887", 50), False),              # occurs in the training answer
    (q("in 1887 and 1889", 50), False),  # the training answer occurs in it
    (q("Seine", 12), False),             # its span overlaps the training span
    (q("The", 50), False),               # empty after normalization
    (q("paris.", 50), True),
])
def test_absent_fact_rules(sibling, ok):
    training = q("1887 and 1889", 10, question="When was the Louvre pyramid built?")
    assert grounded.qualifies(sibling, training) is ok


def test_target_probes_take_the_next_three_qualifying_siblings():
    training = q("red", 0, "Which colour?")
    siblings = [q("blue", 10, id="a"), q("red car", 20, id="b"), q("green", 30, id="c"), q("amber", 40, id="d"),
                q("violet", 50, id="e")]
    chosen, reason = grounded.target_probes([training, *siblings])
    assert reason is None and [p["id"] for p in chosen[1]] == ["a", "c", "d"]
    assert grounded.target_probes([training, *siblings[:2]]) == (None, "fewer_than_4_questions")
    assert grounded.target_probes([training, siblings[1], siblings[1], siblings[0]])[1] == "fewer_than_3_qualifying_siblings"


# ---------------------------------------------------------------- splits

def test_split_is_article_disjoint_with_one_question_per_passage():
    study = small_study()
    assert grounded.audit_study(study)
    articles = {}
    for name, cohort in study["cohorts"].items():
        for pid, p in cohort["passages"].items():
            articles.setdefault(p["article"], set()).add((name, p["group"]))
        for group, rows in cohort["records"].items():
            assert len({r["passage"] for r in rows}) == len(rows)
    assert all(len(owners) == 1 for owners in articles.values())
    for cohort in study["cohorts"].values():
        library = cohort["library"]
        assert library["filler"] not in library["fixed"]
        assert len(library["fixed"]) + 1 == SMALL_LAYOUT["V"]["L"]["articles"] * SMALL_LAYOUT["V"]["L"]["per_article"]
        assert len(cohort["F"]) == SMALL_SIZES["F"]
        assert all(q["passage"] in library["fixed"] for q in cohort["F"])
        assert max(sum(q["passage"] == p for q in cohort["F"]) for p in library["fixed"]) <= 2
        assert all(len(t["probes"]) == 3 and all(grounded.qualifies(p, t["training"]) for p in t["probes"])
                   for t in cohort["targets"])
        assert len({t["hold"]["id"] for t in cohort["targets"]}) == len(cohort["targets"])
    assert "C" in study["cohorts"]["V"]["records"] and "C" not in study["cohorts"]["final"]["records"]


def test_split_is_deterministic_and_seeded():
    assert grounded.study_sha256(small_study()) == grounded.study_sha256(small_study())
    assert grounded.study_sha256(small_study(seed=1)) != grounded.study_sha256(small_study())


def old_pool_hash(row, max_length=128):
    """How the earlier studies named a target: the cleaned closed-book pool string."""
    text = data_sources._clean_record(data_sources._format_squad(row), max_chars=data_sources._char_budget(max_length))
    return sha256(text.encode()).hexdigest()


def test_earlier_target_questions_are_excluded_by_recomputed_hashes():
    rows = squad_rows()
    victims = [rows[0], rows[173]]
    study = small_study(rows, excluded={old_pool_hash(victims[0]), old_pool_hash(victims[1], 64)})
    ids = {r["id"] for c in study["cohorts"].values() for rows_ in c["records"].values() for r in rows_}
    ids |= {q["id"] for c in study["cohorts"].values() for t in c["targets"] for q in [t["training"], *t["probes"]]}
    ids |= {q["id"] for c in study["cohorts"].values() for q in c["F"] + c["F_P"]}
    assert not ids & {v["id"] for v in victims}
    assert study["exclusions"]["matched"] == 2 and study["exclusions"]["excluded_questions"] == 2
    with pytest.raises(ValueError, match="match no SQuAD train row"):
        small_study(rows, excluded={"0" * 64})
    assert small_study(rows, excluded={"0" * 64}, allowed_unmatched=["0" * 64])["exclusions"]["allowed_unmatched"] == ["0" * 64]


def test_audit_refuses_an_excluded_question_in_use():
    study = small_study()
    target = study["cohorts"]["V"]["targets"][0]["training"]
    study["exclusions"]["hashes"] = [old_pool_hash({"question": target["question"], "answers": {"text": [target["answer"]]}})]
    with pytest.raises(ValueError, match="excluded"):
        grounded.audit_study(study)


def test_audit_refuses_shared_articles_and_duplicate_passages():
    study = small_study()
    tampered = deepcopy(study)
    v = tampered["cohorts"]["V"]
    pid = v["library"]["fixed"][0]
    v["passages"][pid]["article"] = next(p["article"] for p in v["passages"].values() if p["group"] == "T")
    with pytest.raises(ValueError, match="two groups"):
        grounded.audit_study(tampered)
    tampered = deepcopy(study)
    rows = tampered["cohorts"]["V"]["records"]["T"]
    rows.append(dict(rows[0], id="other"))
    with pytest.raises(ValueError, match="one question per passage"):
        grounded.audit_study(tampered)


def test_structural_skips_are_counted():
    rows = squad_rows()
    long = passage_rows(99, 0, words=200)
    broken = passage_rows(98, 0)
    broken[0]["answers"]["answer_start"] = [3]
    study = small_study(rows + long + broken)
    assert study["skips"]["rows"]["answer_span_mismatch"] == 1
    assert study["skips"]["rows"]["passage_outside_word_range"] >= 1
    assert all(p["article"] != "Article 99" for c in study["cohorts"].values() for p in c["passages"].values())


def test_builder_refuses_records_that_do_not_fit():
    with pytest.raises(ValueError, match="do not fit"):
        small_study(fits=lambda record, arm: arm != "RG")


def test_eligibility_skips_are_recorded():
    rows = [r for r in squad_rows(questions=5) if not (r["title"] in ("Article 3", "Article 7") and r["id"].endswith("-4"))]
    study = small_study(rows)
    total = sum(sum(a["eligibility_skips"].values()) for a in study["audit"].values())
    assert total > 0


# ---------------------------------------------------------------- worlds

def test_paired_worlds_differ_only_in_the_target_record():
    study = small_study()
    for index in range(3):
        w1 = grounded.world_records(study, "V", index, 8000 + index, True)
        w0 = grounded.world_records(study, "V", index, 8000 + index, False)
        target = study["cohorts"]["V"]["targets"][index]
        assert [len(p) for p in w1] == [SMALL_SIZES["records_per_client"] + 1] + [SMALL_SIZES["records_per_client"]] * (SMALL_SIZES["clients"] - 1)
        assert w1[1:] == w0[1:] and w1[0][:-1] == w0[0][:-1]
        assert w1[0][-1] == target["training"] and w0[0][-1] == target["hold"]
        assert grounded.check_world(study, "V", w1, index, True) and grounded.check_world(study, "V", w0, index, False)
        passages = study["cohorts"]["V"]["passages"]
        assert all(passages[r["passage"]]["article"] != target["article"] for r in w1[0][:-1] + [x for p in w1[1:] for x in p])
        with pytest.raises(ValueError, match="membership"):
            grounded.check_world(study, "V", w1, index, False)
    assert grounded.world_records(study, "V", 0, 8000, True) != grounded.world_records(study, "V", 0, 8001, True)


def test_public_worlds_train_on_u_only():
    study = small_study()
    records = grounded.public_records(study, "V", 8095)
    assert grounded.check_world(study, "V", records)
    assert all(study["cohorts"]["V"]["passages"][r["passage"]]["group"] == "U" for p in records for r in p)


# ------------------------------------------------------------- encoding

class ChatTokenizer:
    """Character tokenizer with a Qwen-shaped chat template (ids stay below 50)."""
    eos_token, eos_token_id, pad_token, pad_token_id, name_or_path = "<|im_end|>", 1, "<pad>", 0, "fake"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 45 + 3 for c in text]

    def __call__(self, text, truncation=False, max_length=None, **kwargs):
        ids = self.encode(text)
        return {"input_ids": ids[:max_length] if truncation and max_length else ids}


MODEL_CONFIG = SimpleNamespace(max_position_embeddings=4096)
SETTINGS = grounded.RagSettings()
RECORD = {"id": "r1", "question": "Who built it?", "answer": "Gustave", "answer_start": 0}
PASSAGE = "Gustave built the tower in Paris for the fair."


def inference_prompt(question, contexts, settings=SETTINGS):
    bundle = {"tokenizer": ChatTokenizer(), "model": SimpleNamespace(config=MODEL_CONFIG)}
    return rag._prompt_ids(rag._prepare_prompt(bundle, question, contexts, settings))


def test_rg_and_cb_ao_training_prompts_match_the_rag_prompt_exactly():
    tokenizer = ChatTokenizer()
    rg = grounded.encode_record(tokenizer, MODEL_CONFIG, RECORD, PASSAGE, "RG", SETTINGS)
    cb = grounded.encode_record(tokenizer, MODEL_CONFIG, RECORD, PASSAGE, "CB-AO", SETTINGS)
    answer = tokenizer.encode("Gustave") + [tokenizer.eos_token_id]
    for example, contexts in ((rg, [PASSAGE]), (cb, [])):
        prompt = inference_prompt(RECORD["question"], contexts)
        assert list(example.input_ids) == prompt + answer
        assert list(example.labels) == [-100] * len(prompt) + answer
    # Same template: the arms differ only by the passage tokens in the context slot.
    bundle = {"tokenizer": tokenizer, "model": SimpleNamespace(config=MODEL_CONFIG)}
    full = rag._prepare_prompt(bundle, RECORD["question"], [PASSAGE], SETTINGS)
    empty = rag._prepare_prompt(bundle, RECORD["question"], [], SETTINGS)
    assert (full["prefix"], full["suffix"]) == (empty["prefix"], empty["suffix"]) and empty["context_ids"] == []
    assert list(rg.input_ids) == full["prefix"] + full["context_ids"] + full["suffix"] + answer
    public = grounded.encode_record(tokenizer, MODEL_CONFIG, RECORD, PASSAGE, "RG-public", SETTINGS)
    assert public.input_ids == rg.input_ids


def test_grounded_records_are_never_truncated():
    tokenizer = ChatTokenizer()
    with pytest.raises(ValueError, match="truncate the passage"):
        grounded.encode_record(tokenizer, MODEL_CONFIG, RECORD, PASSAGE * 20, "RG", grounded.RagSettings(max_context_tokens=64))
    with pytest.raises(ValueError, match="max_length"):
        grounded.encode_record(tokenizer, MODEL_CONFIG, RECORD, PASSAGE * 5, "RG", SETTINGS, max_length=128)


def test_cb_lm_is_todays_closed_book_record():
    tokenizer = ChatTokenizer()
    text = grounded.training_record(tokenizer, MODEL_CONFIG, RECORD, PASSAGE, "CB-LM", SETTINGS)
    # _clean_record collapses whitespace, so today's pool records are single-line.
    assert text == "Question: Who built it? Answer: Gustave" == grounded.closed_book_record("Who built it?", "Gustave")
    with pytest.raises(ValueError):
        grounded.encode_record(tokenizer, MODEL_CONFIG, RECORD, PASSAGE, "CB-LM", SETTINGS)
    rg = grounded.template_example(tokenizer, MODEL_CONFIG, RECORD, PASSAGE, "RG-public", SETTINGS)
    cb = grounded.template_example(tokenizer, MODEL_CONFIG, RECORD, PASSAGE, "CB-LM", SETTINGS)
    assert rg == grounded.encode_record(tokenizer, MODEL_CONFIG, RECORD, PASSAGE, "RG", SETTINGS)
    assert cb == grounded.encode_record(tokenizer, MODEL_CONFIG, RECORD, PASSAGE, "CB-AO", SETTINGS)


def test_the_assistant_turn_must_end_with_eos():
    class Plain(ChatTokenizer):
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            return "".join(m["content"] + "\n" for m in messages)
    with pytest.raises(ValueError, match="EOS"):
        grounded.end_of_turn_id(Plain())


def test_encoded_identity_and_membership_checks():
    tokenizer = ChatTokenizer()
    a = grounded.encode_record(tokenizer, MODEL_CONFIG, RECORD, PASSAGE, "RG", SETTINGS)
    b = grounded.encode_record(tokenizer, MODEL_CONFIG, dict(RECORD, id="r2", answer="Eiffel"), PASSAGE, "RG", SETTINGS)
    c = grounded.encode_record(tokenizer, MODEL_CONFIG, dict(RECORD, id="r3", answer="Paris"), PASSAGE, "RG", SETTINGS)
    validate_partition_tokens([[a, b]], tokenizer, 384, target=a, held_out=c, expected_membership=True)
    validate_partition_tokens([[c, b]], tokenizer, 384, target=a, held_out=c, expected_membership=False)
    with pytest.raises(ValueError, match="membership"):
        validate_partition_tokens([[c, b]], tokenizer, 384, target=a, held_out=c, expected_membership=True)
    with pytest.raises(ValueError, match="never truncated"):
        validate_partition_tokens([[a]], tokenizer, 16)
    with pytest.raises(ValueError):
        EncodedExample((1, 2), (-100, -100), "none")


# ------------------------------------------------------- training paths

@pytest.fixture
def tiny():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.GPT2Config(vocab_size=50, n_positions=512, n_embd=16, n_layer=1, n_head=2)
    return transformers.GPT2LMHeadModel(config).eval()


def examples():
    tokenizer = ChatTokenizer()
    return [grounded.encode_record(tokenizer, MODEL_CONFIG, dict(RECORD, id=f"r{i}", answer=a), PASSAGE[:20 + i], arm, SETTINGS)
            for i, (a, arm) in enumerate((("Gustave", "RG"), ("Eiffel", "CB-AO"), ("Paris", "RG")))]


def manual_answer_loss(model, batch):
    import torch
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1]
    labels = batch["labels"][:, 1:]
    keep = labels != -100
    return torch.nn.functional.cross_entropy(logits[keep], labels[keep])


def test_the_adamw_client_loader_takes_loss_on_answer_tokens_only(tiny):
    from master_script.core.attacks import amia
    config = SimpleNamespace(max_length=384, local_batch_size=3)
    batch = next(iter(amia.make_loader(examples(), ChatTokenizer(), config, shuffle=False)))
    lengths = [len(e.input_ids) for e in examples()]
    assert batch["input_ids"].shape == (3, max(lengths))
    for row, e in enumerate(examples()):
        assert batch["labels"][row, :len(e.labels)].tolist() == list(e.labels)
        assert (batch["labels"][row, len(e.labels):] == -100).all() and (batch["attention_mask"][row, len(e.labels):] == 0).all()
    assert float(tiny(**batch).loss.detach()) == pytest.approx(float(manual_answer_loss(tiny, batch).detach()), rel=1e-5)
    with pytest.raises(ValueError, match="mix"):
        amia.make_loader(examples() + ["plain text"], ChatTokenizer(), config, shuffle=False)
    with pytest.raises(ValueError, match="never truncated"):
        amia.make_loader(examples(), ChatTokenizer(), SimpleNamespace(max_length=40, local_batch_size=3), shuffle=False)


class Recorder:
    """Wraps a model and keeps every labels tensor it is trained with."""
    def __init__(self, model):
        self.model, self.labels = model, []

    def parameters(self):
        return self.model.parameters()

    def train(self):
        self.model.train()

    def __call__(self, **kwargs):
        self.labels.append(kwargs["labels"].clone())
        return self.model(**kwargs)


def test_private_train_uses_the_answer_only_mask(tiny):
    from master_script.core.defenses import private_train
    config = SimpleNamespace(client_lr=1e-3, local_epochs=1, local_batch_size=2, max_length=384)
    defense = SimpleNamespace(clip_norm=1.0, noise_multiplier=0.5)
    recorder = Recorder(tiny)
    steps = private_train(recorder, ChatTokenizer(), examples(), config, defense)
    assert steps == 2 and len(recorder.labels) == 3
    expected = sorted(tuple(e.labels) for e in examples())
    assert sorted(tuple(l[0].tolist()) for l in recorder.labels) == expected
    with pytest.raises(ValueError, match="never truncated"):
        private_train(Recorder(tiny), ChatTokenizer(), examples(), SimpleNamespace(**{**vars(config), "max_length": 30}), defense)


def test_causal_gradients_and_request_use_the_answer_only_loss(tiny):
    import torch
    from master_script.core.attacks.causal_probe import optimize_request, raw_gradients
    config = SimpleNamespace(max_length=384, probe_epochs=3, probe_lr=0.01)
    batch = examples()
    released = raw_gradients(tiny, ChatTokenizer(), batch, config)
    encoded = encoded_batch(batch, 0)
    tiny.zero_grad()
    manual = torch.autograd.grad(tiny(**encoded).loss, tuple(tiny.parameters()))
    named = dict(tiny.named_parameters())
    first = next(iter(tiny.state_dict()))
    assert np.allclose(released[0], manual[list(named).index(first)].numpy(), atol=1e-6)
    request = deepcopy(tiny)
    history = optimize_request(request, ChatTokenizer(), batch[0], config)
    one = encoded_batch([batch[0]], 0)
    assert float(request(**one).loss.detach()) > history[0]
    # The plain-text path is unchanged: full LM loss with padding masked.
    plain = raw_gradients(tiny, TextTokenizer(), ["abc", "abcdef"], config)
    enc = TextTokenizer()(["abc", "abcdef"], padding=True, return_tensors="pt")
    labels = enc["input_ids"].clone(); labels[enc["attention_mask"] == 0] = -100
    ref = torch.autograd.grad(tiny(**enc, labels=labels).loss, tuple(tiny.parameters()))
    assert np.allclose(plain[0], ref[list(named).index(first)].numpy(), atol=1e-6)


class TextTokenizer:
    pad_token_id = 0

    def __call__(self, texts, padding=False, truncation=True, max_length=32, return_tensors="pt"):
        import torch
        rows = [texts] if isinstance(texts, str) else texts
        ids = [[ord(c) % 45 + 3 for c in t][:max_length] for t in rows]
        width = max(map(len, ids))
        return {"input_ids": torch.tensor([x + [0] * (width - len(x)) for x in ids]),
                "attention_mask": torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in ids])}


def test_attack_batches_use_the_declared_encoded_target():
    import random
    from master_script.core.attacks import amia
    batch = examples()
    config = SimpleNamespace(attack_batch_size=2)
    member = amia.sample_attack_batch(batch, config, True, random.Random(1), target=batch[0])
    other = amia.sample_attack_batch(batch, config, False, random.Random(1), target=batch[0])
    assert batch[0] in member and batch[0] not in other and len(member) == len(other) == 2


def test_supplied_worlds_are_validated_before_any_simulation(monkeypatch, tiny):
    from master_script.core.attacks import amia
    from master_script.core import guard_runtime

    class Stop(Exception):
        pass

    def stop(*args, **kwargs):
        raise Stop
    monkeypatch.setattr(amia, "build_model_and_tokenizer", lambda config: (tiny, ChatTokenizer()))
    monkeypatch.setattr(guard_runtime, "prepare_guard", stop)
    config = amia.AmiaConfig(max_length=384)
    a, b, c = examples()
    with pytest.raises(Stop):
        amia.federated_fine_tune(config, "unused", None, world={"partitions": [[a, b], [c]], "target": a,
                                                                 "held_out": grounded.encode_record(ChatTokenizer(), MODEL_CONFIG, dict(RECORD, id="h", answer="Hold"), PASSAGE, "RG", SETTINGS),
                                                                 "member": True})
    with pytest.raises(ValueError, match="membership"):
        amia.federated_fine_tune(config, "unused", None, world={"partitions": [[b], [c]], "target": a, "held_out": None,
                                                                 "member": True})


# ------------------------------------------------------------- gate/scorers

def test_verbatim_gate_withholds_only_a_quoted_top_document():
    docs = ["the tower was built in paris for the fair", "a river flows through the city", "bread is baked daily"]
    probe = 'Does this: "the tower was built in paris for the fair" appear in the context?'
    assert grounded.verbatim_overlap(probe, docs[0]) > 0.5 and grounded.verbatim_overlap("Who built it?", docs[0]) == 0
    assert grounded.gated_ranking([0, 1, 2], docs, probe, 0.5)[:2] == ([1, 2], True)
    assert grounded.gated_ranking([0, 1, 2], docs, "Who built the tower?", 0.5)[:2] == ([0, 1, 2], False)
    assert grounded.gated_ranking([0, 1, 2], docs, probe, None)[:2] == ([0, 1, 2], False)


def test_gate_calibration_picks_the_most_protective_threshold_within_five_percent():
    docs = [f"alpha beta gamma delta epsilon zeta {i}" for i in range(4)]
    benign = [{"query": "which alpha beta gamma delta", "gold": 0, "ranking": [0, 1, 2, 3], "documents": docs}]
    benign += [{"query": f"unrelated question {i}", "gold": i % 4, "ranking": [i % 4, 0, 1, 2], "documents": docs}
               for i in range(1, 40)]
    gate = grounded.calibrate_gate(benign, top_k=1)
    assert gate["threshold"] == 0.05 and gate["gold_loss"] == pytest.approx(1 / 40)
    strict = grounded.calibrate_gate(benign, top_k=1, max_loss=0.0)
    assert strict["threshold"] > 0.5 and strict["gold_loss"] == 0
    with pytest.raises(ValueError):
        grounded.calibrate_gate([{"query": "q", "gold": 9, "ranking": [0], "documents": docs}], top_k=1)


def test_refusals_score_zero_and_documents_average_their_three_probes():
    rows = [grounded.probe_scores("Paris", "Paris", 0.9, False), grounded.probe_scores("I cannot answer", "Paris", 0.8, True),
            grounded.probe_scores("Lyon", "Paris", 0.1, False)]
    assert [r["f1"] for r in rows] == [1.0, 0.0, 0.0] and rows[1]["raw_entailment"] == 0.8 and rows[1]["entailment"] == 0
    assert grounded.document_score(rows, "entailment") == pytest.approx(1.0 / 3)
    assert grounded.document_score(rows, "entailment", exclude_refusals=True) == pytest.approx(0.5)
    refused = [dict(r, refused=True) for r in rows]
    assert grounded.document_score(refused, "f1", exclude_refusals=True) is None
    with pytest.raises(ValueError):
        grounded.document_score(rows[:2], "f1")
    assert grounded.nli_hypothesis("Who?", "Me") == "Question: Who?\nAnswer: Me"


def test_epsilon_records():
    from master_script.core.defenses import privacy_bound
    bound = privacy_bound(51, 5.45, 1e-5)
    record = grounded.epsilon_record(bound, "dp_sgd")
    assert record["epsilon"] == pytest.approx(16.0, abs=0.05) and record["steps"] == 51
    assert grounded.epsilon_record({}, "none") == {"epsilon": "inf", "mechanism": "none"}
    with pytest.raises(RuntimeError):
        grounded.epsilon_record({}, "dp_sgd")
    other = grounded.epsilon_record(privacy_bound(51, 1.91, 1e-5), "dp_sgd")
    assert grounded.same_epsilon(record, record) and not grounded.same_epsilon(record, other)
    assert not grounded.same_epsilon(record, {"epsilon": "inf"})


# ------------------------------------------------------------ job passes

class FakeScorers:
    def embed(self, texts):
        vectors = np.array([[1.0 if "tower" in t else 0.0, 1.0] for t in texts])
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def entail(self, premises, hypotheses):
        return [0.9 if "Paris" in h else 0.2 for h in hypotheses]


def fake_session(audit, gate=None):
    from master_script.core.grounded_job import Library
    scorers = FakeScorers()
    return SimpleNamespace(settings=grounded.RagSettings(top_k=1), gate=gate, scorers=scorers, audit=audit.append,
                           query_vector=lambda q: scorers.embed([q])[0]), Library


def test_natural_question_pass_scores_the_same_three_responses(monkeypatch):
    from master_script.core import grounded_job
    audit = []
    session, Library = fake_session(audit, gate={"threshold": 0.5, "ngram": 3})
    long = "the tower stands in paris near the river and was finished for the fair of that year by a large team of workers"
    docs = ["a river crosses the city", "bread is baked daily", long]
    library = Library(docs, session.scorers.embed(docs))
    answers = iter(["Paris", "I cannot answer that", "Lyon", "Yes"])
    monkeypatch.setattr(rag, "generate_answer", lambda bundle, question, contexts, settings: next(answers))
    probes = [{"id": f"p{i}", "question": f"Where is the tower {i}?", "answer": "Paris"} for i in range(3)]
    out = grounded_job.natural_questions({}, session, docs[2], probes, library, "library_on", "W0")
    assert [p["f1"] for p in out["probes"]] == [1.0, 0.0, 0.0]
    assert [p["refused"] for p in out["probes"]] == [False, True, False]
    assert [p["entailment"] for p in out["probes"]] == [0.9, 0.0, 0.2] and out["probes"][1]["raw_entailment"] == 0.2
    assert all(p["retrieved"] and p["document_rank"] == 0 for p in out["probes"])
    assert out["verbatim"]["score"] == 1.0 and out["verbatim"]["gate"]["withheld_top"]
    assert not any("answer" in k for p in out["probes"] for k in p) and len(audit) == 4
    assert {row["kind"] for row in audit} == {"natural_question", "verbatim_probe"}


def test_retention_pass_uses_an_empty_context(monkeypatch):
    from master_script.core import grounded_job
    seen = []
    monkeypatch.setattr(rag, "generate_answer", lambda b, q, contexts, s: seen.append(contexts) or "Paris")
    session, _ = fake_session([])
    out = grounded_job.retention({}, session, [{"id": "a", "question": "Q", "answer": "Paris"}] * 3, "W1")
    assert seen == [[], [], []] and out["f1"] == 1.0


def test_learning_check_and_memorization(tiny):
    from master_script.core import grounded_job
    tokenizer = ChatTokenizer()
    passage = "The tower was completed for the world fair and stood as the tallest structure for decades."
    bundle = {"model": tiny, "tokenizer": tokenizer, "device": "cpu"}
    before = grounded_job.learning_measures(bundle, passage)
    losses = grounded_job.memorize(tiny, tokenizer, passage, epochs=3, lr=1e-2, seed=1)
    after = grounded_job.learning_measures(bundle, passage)
    assert len(losses) == 3 and after["passage_nll"] < before["passage_nll"]
    assert 0 <= after["continuation_match"] <= 1 and after["passage_tokens"] == len(passage)
    with pytest.raises(ValueError, match="64 tokens"):
        grounded_job.learning_measures(bundle, "short")


def test_reference_template_form_is_the_answer_only_ratio(tiny, monkeypatch):
    from master_script.core import grounded_job
    from master_script.core.attacks import reference as reference_attack
    monkeypatch.setattr(reference_attack, "score_candidate_hf", lambda *a, **k: 0.25)
    example = examples()[0]
    other = deepcopy(tiny)
    with __import__("torch").no_grad():
        next(other.parameters()).add_(0.1)
    bundle = {"model": tiny, "tokenizer": ChatTokenizer(), "device": "cpu"}
    reference = {"model": other, "tokenizer": ChatTokenizer(), "device": "cpu"}
    scores = grounded_job.reference_scores(bundle, reference, "closed", example, 384)
    expected = -grounded_job._nll(bundle, example) / grounded_job._nll(reference, example)
    assert scores == {"record": 0.25, "template": pytest.approx(expected)}


def test_jobs_refuse_other_study_data(tmp_path):
    from master_script.core import grounded_job
    from master_script.tools import grounded_fl_study as gs
    study = small_study()
    job = gs.job_document(study, "control", "RG", "V", target_index=0, control={"epochs": 3, "lr": 1e-4})
    with pytest.raises(ValueError, match="Study data"):
        grounded_job.run_job(job, small_study(seed=5), tmp_path / "out")


def test_a_reused_victim_model_observes_exactly_what_a_fresh_load_does(tiny, tmp_path, monkeypatch):
    import torch
    from transformers import AutoTokenizer
    from master_script.core.attacks import amia
    from master_script.core.attacks.causal_probe import raw_gradients
    from master_script.core.model_io import load_causal_model
    tiny.save_pretrained(tmp_path / "model")
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", staticmethod(lambda path: ChatTokenizer()))
    base = amia.get_parameters(tiny)
    request = [p + 0.01 for p in base]
    other = [p - 0.03 for p in base]
    config = SimpleNamespace(max_length=384)
    batch = examples()[:2]

    fresh = load_causal_model(tmp_path / "model").eval()
    amia.set_parameters(fresh, request)
    expected = raw_gradients(fresh, ChatTokenizer(), batch, config)

    amia._VICTIM_MODEL.clear()
    cached, _ = amia.cached_victim_model(tmp_path / "model", "cpu")
    amia.set_parameters(cached, other)  # an earlier, different trial
    raw_gradients(cached, ChatTokenizer(), batch, config)
    again, _ = amia.cached_victim_model(tmp_path / "model", "cpu")
    assert again is cached
    amia.set_parameters(again, request)
    observed = raw_gradients(again, ChatTokenizer(), batch, config)
    assert all(np.array_equal(a, b) for a, b in zip(expected, observed))
    assert all(p.grad is None for p in again.parameters())
    amia._VICTIM_MODEL.clear()


@pytest.mark.parametrize("target_only, nodes", [(True, 1), (False, 4)])
def test_observation_rounds_can_simulate_the_victim_alone(monkeypatch, target_only, nodes):
    import flwr.client
    import flwr.server
    from master_script.core import runtime_memory
    from master_script.core.attacks import amia
    seen = {}
    monkeypatch.setattr(flwr.server, "ServerApp", lambda server_fn: SimpleNamespace(server_fn=server_fn))
    monkeypatch.setattr(flwr.client, "ClientApp", lambda client_fn: SimpleNamespace(client_fn=client_fn))

    def simulate(server_app, client_app, num_supernodes, backend_config):
        seen["nodes"] = num_supernodes
        seen["strategy"] = server_app.server_fn(None).strategy
        seen["partitions"] = [client_app.client_fn(SimpleNamespace(node_config={"partition-id": i})).numpy_client.partition_id
                              for i in range(num_supernodes)]
    monkeypatch.setattr(runtime_memory, "run_simulation", simulate)
    probe = SimpleNamespace(_public_direction=[np.zeros(1)])
    monkeypatch.setattr(amia, "get_parameters", lambda model: [np.zeros(1, "float32")])
    config = SimpleNamespace(attack_variant="causal_gradient_alignment", num_clients=4, target_client_id=2,
                             attack_trials=2, sim_num_gpus=0.0, sim_max_concurrent_clients=1)
    with pytest.raises(RuntimeError, match="Incomplete"):
        amia.run_attack_trials("unused", probe, [["a"], ["b"], ["c"], ["d"]], config, observe_target_only=target_only)
    assert seen["nodes"] == nodes
    assert (seen["strategy"].min_fit_clients, seen["strategy"].min_available_clients) == (nodes, nodes)
    assert seen["partitions"] == ([2] if target_only else [0, 1, 2, 3])


# ------------------------------------------------ observation speed-ups (opt-in)

def _arrays(seed=3):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, 0.7, (300, 40)).astype("float32"), rng.normal(0, 0.2, (5000,)).astype("float32"),
            np.zeros((7,), "float32")]


@pytest.mark.parametrize("mechanism, clip", [("none", 1.0), ("clip", 1.0), ("clip", 1e6)])
def test_device_release_noise_clips_exactly_as_the_numpy_path(mechanism, clip):
    import torch
    from master_script.core.defenses import protect_observation, protect_observation_on_device
    config = SimpleNamespace(observation_defense=mechanism, observation_clip_norm=clip, observation_noise_multiplier=1.0)
    arrays = _arrays()
    expected = protect_observation(arrays, config)
    released = protect_observation_on_device([torch.from_numpy(a.copy()) for a in arrays], config)
    assert [r.dtype for r in released] == [np.dtype("float32")] * 3
    assert all(np.array_equal(a, b) for a, b in zip(expected, released))
    if mechanism == "clip" and clip == 1.0:
        assert np.sqrt(sum(float(np.sum(r.astype(float) ** 2)) for r in released)) == pytest.approx(1.0, rel=1e-6)


def test_device_release_noise_has_the_configured_std_and_fresh_seeds():
    import torch
    from master_script.core.defenses import protect_observation_on_device
    arrays = [np.full((400_000,), 3.0, "float32"), np.zeros((1000,), "float32")]
    tensors = [torch.from_numpy(a.copy()) for a in arrays]
    clip = SimpleNamespace(observation_defense="clip", observation_clip_norm=2.0, observation_noise_multiplier=0.5)
    gaussian = SimpleNamespace(observation_defense="gaussian", observation_clip_norm=2.0, observation_noise_multiplier=0.5)
    clipped = protect_observation_on_device(tensors, clip)
    first = protect_observation_on_device(tensors, gaussian)
    second = protect_observation_on_device(tensors, gaussian)
    noise = np.concatenate([(f.astype(float) - c) for f, c in zip(first, clipped)])
    assert noise.mean() == pytest.approx(0.0, abs=0.01)
    assert noise.std() == pytest.approx(0.5 * 2.0, rel=0.01)  # std = noise multiplier x clip norm
    assert not np.array_equal(first[0], second[0])  # each call draws from a freshly seeded generator
    seeded = lambda: torch.Generator().manual_seed(5)
    again = [protect_observation_on_device(tensors, gaussian, generator=seeded()) for _ in range(2)]
    assert all(np.array_equal(a, b) for a, b in zip(*again))


def _victim_records():
    tokenizer = ChatTokenizer()
    answers = ("Gustave", "Eiffel", "Paris", "fair", "tower", "iron", "1889")
    return [grounded.encode_record(tokenizer, MODEL_CONFIG, dict(RECORD, id=f"v{i}", answer=a), PASSAGE[:18 + 3 * i],
                                   "RG", SETTINGS) for i, a in enumerate(answers)]


def _flower_simulation(monkeypatch, config):
    """Drive the real VictimClient and ObserveGradient, with Flower's serialization, without Ray."""
    import flwr.client
    import flwr.server
    from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
    from master_script.core import runtime_memory
    monkeypatch.setattr(flwr.server, "ServerApp", lambda server_fn: SimpleNamespace(server_fn=server_fn))
    monkeypatch.setattr(flwr.client, "ClientApp", lambda client_fn: SimpleNamespace(client_fn=client_fn))

    def simulate(server_app, client_app, num_supernodes, backend_config):
        strategy = server_app.server_fn(None).strategy
        current = strategy.initial_parameters
        for round_id in range(1, config.attack_trials + 1):
            fit_config = strategy.on_fit_config_fn(round_id)
            results = []
            for node in range(num_supernodes):
                client = client_app.client_fn(SimpleNamespace(node_config={"partition-id": node})).numpy_client
                arrays, examples_, metrics = client.fit(parameters_to_ndarrays(current), fit_config)
                results.append((None, FitRes(status=Status(code=Code.OK, message=""), parameters=ndarrays_to_parameters(arrays),
                                             num_examples=examples_, metrics=metrics)))
            current, _ = strategy.aggregate_fit(round_id, results, [])
    monkeypatch.setattr(runtime_memory, "run_simulation", simulate)


@pytest.mark.parametrize("mechanism", ["none", "clip"])
def test_in_process_observation_releases_and_scores_exactly_what_the_victim_client_does(tiny, tmp_path, monkeypatch, mechanism):
    import torch
    from transformers import AutoTokenizer
    from master_script.core.attacks import amia, causal_probe
    tiny.save_pretrained(tmp_path / "model")
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", staticmethod(lambda path: ChatTokenizer()))
    records = _victim_records()
    target = records[0]
    clients = [records, list(reversed(records[1:]))]
    config = amia.AmiaConfig(attack_variant="causal_gradient_alignment", causal_score="cosine", attack_trials=6,
                             counterbalance_trials=True, attack_batch_size=3, max_length=384, num_clients=2,
                             target_client_id=0, seed=8123, observation_defense=mechanism, observation_clip_norm=0.05)
    probe = deepcopy(tiny)
    with torch.no_grad():
        for parameter in probe.parameters():
            parameter.add_(0.02 * torch.randn_like(parameter))
    probe._public_direction = causal_probe.raw_gradients(probe, ChatTokenizer(), [target], config)
    released = []
    original = causal_probe.alignment_terms
    monkeypatch.setattr(causal_probe, "alignment_terms",
                        lambda arrays, direction: released.append([np.array(a) for a in arrays]) or original(arrays, direction))
    _flower_simulation(monkeypatch, config)
    paths = {}
    for name, options in (("victim_client", {"reuse_victim_model": True, "observe_target_only": True}),
                          ("in_process", {"in_process": True}),
                          ("in_process_device_noise", {"in_process": True, "noise_on_device": True})):
        released.clear()
        amia._VICTIM_MODEL.clear()
        trials = amia.run_attack_trials(str(tmp_path / "model"), probe, clients, config, target=target,
                                        checkpoint_dir=tmp_path / name, **options)
        paths[name] = ([{k: v for k, v in t.items() if k != "response_seconds"} for t in trials], list(released))
        assert all(t["response_seconds"] >= 0 for t in trials)
    amia._VICTIM_MODEL.clear()
    expected_trials, expected_released = paths["victim_client"]
    assert len(expected_trials) == 6 and {t["truth_member"] for t in expected_trials} == {True, False}
    assert len({t["score"] for t in expected_trials}) > 1
    for name in ("in_process", "in_process_device_noise"):
        trials, arrays = paths[name]
        assert trials == expected_trials  # trial_id, truth_member, score, batch_pair_seed, alignment_terms, ...
        assert len(arrays) == len(expected_released) == 6
        assert all(np.array_equal(a, b) for x, y in zip(arrays, expected_released) for a, b in zip(x, y))
    assert all(p.grad is None for p in probe.parameters())


def test_in_process_observation_refuses_guards_calibration_and_the_probe_head():
    from master_script.core.attacks import amia
    causal = SimpleNamespace(attack_variant="causal_gradient_alignment")
    with pytest.raises(ValueError, match="causal request only"):
        amia.run_attack_trials("unused", None, [[]], SimpleNamespace(attack_variant="probe_head"), in_process=True)
    with pytest.raises(ValueError, match="no guard"):
        amia.run_attack_trials("unused", None, [[]], causal, guard_runtime=object(), in_process=True)
    with pytest.raises(ValueError, match="no guard"):
        amia.run_attack_trials("unused", None, [[]], causal, calibration={"threshold": 0.0}, in_process=True)


def test_grounded_causal_passes_opt_in_to_both_observation_speed_ups(monkeypatch, tmp_path):
    from master_script.core import grounded_job, model_io
    from master_script.core.attacks import amia, causal_probe
    calls = []
    model = SimpleNamespace(to=lambda device: model)
    monkeypatch.setattr(model_io, "load_causal_model", lambda path: model)
    monkeypatch.setattr(causal_probe, "optimize_request", lambda *a: [1.0])
    monkeypatch.setattr(causal_probe, "raw_gradients", lambda *a: [np.zeros(1)])

    def trials(model_path, probe, clients, config, **options):
        calls.append((config.observation_defense, options))
        return [{"trial_id": i, "truth_member": i % 2 == 0, "score": float(i % 2 == 0), "batch_pair_seed": 1,
                 "alignment_terms": {}, "response_seconds": 0.1} for i in range(4)]
    monkeypatch.setattr(amia, "run_attack_trials", trials)
    config = amia.AmiaConfig(attack_variant="causal_gradient_alignment")
    out = grounded_job.causal_passes(config, "path", None, [[]], "target", {"template": "candidate"}, True, tmp_path)
    assert [c[0] for c in calls] == ["none", "gaussian"]
    assert all(c[1]["in_process"] is True and c[1]["noise_on_device"] is True for c in calls)
    assert out["template"]["observation"] == {"in_process": True, "noise_on_device": True}
    assert set(out["template"]) >= {"plain", "release_noise"}
