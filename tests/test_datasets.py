from types import SimpleNamespace

import pytest

from master_script.core import datasets as data_sources


def _config(**overrides):
    values = {
        "dataset_name": "squad",
        "seed": 7,
        "max_length": 64,
        "num_clients": 3,
        "target_client_id": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _records(n=128):
    return tuple(f"Question: real question {i}? Answer: real answer number {i}." for i in range(n))


def test_catalog_includes_requested_general_and_domain_profiles():
    assert {
        "nq_open",
        "trivia_qa",
        "squad",
        "hotpot_qa",
        "medical_medquad",
        "financial_phrasebank",
        "corporate_enron",
    } <= set(data_sources.available_dataset_names())


def test_real_worlds_differ_only_in_target_payload(monkeypatch):
    monkeypatch.setattr(data_sources, "_load_dataset_pool", lambda *a, **k: _records())
    cfg = _config()
    positive = data_sources.build_real_membership_world(cfg, truth_member=True)
    negative = data_sources.build_real_membership_world(cfg, truth_member=False)

    assert positive.target_record == negative.target_record
    assert positive.held_out_record == negative.held_out_record
    assert [part[:-1] if i == cfg.target_client_id else part for i, part in enumerate(positive.partitions)] == [
        part[:-1] if i == cfg.target_client_id else part for i, part in enumerate(negative.partitions)
    ]
    assert positive.partitions[cfg.target_client_id][-1] == positive.target_record
    assert negative.partitions[cfg.target_client_id][-1] == negative.held_out_record


def test_real_world_is_deterministic_and_seeded(monkeypatch):
    monkeypatch.setattr(data_sources, "_load_dataset_pool", lambda *a, **k: _records())
    first = data_sources.build_real_membership_world(_config(seed=7), truth_member=True)
    again = data_sources.build_real_membership_world(_config(seed=7), truth_member=True)
    moved = data_sources.build_real_membership_world(_config(seed=11), truth_member=True)
    assert first == again
    assert first.target_record != moved.target_record


def test_calibration_records_are_disjoint_from_world(monkeypatch):
    monkeypatch.setattr(data_sources, "_load_dataset_pool", lambda *a, **k: _records())
    cfg = _config()
    world = data_sources.build_real_membership_world(cfg, truth_member=True)
    calibration = data_sources.calibration_records(cfg, 8)
    used = {world.target_record, world.held_out_record}
    used.update(record for partition in world.partitions for record in partition)
    assert set(calibration).isdisjoint(used)


def test_unknown_dataset_fails_before_compute():
    with pytest.raises(ValueError, match="unknown dataset_name"):
        data_sources.validate_dataset_name("not-a-real-profile")


def test_qa_formatters_use_actual_row_fields():
    assert data_sources._format_squad({
        "question": "Who?",
        "answers": {"text": ["Ada"], "answer_start": [0]},
    }) == "Question: Who?\nAnswer: Ada"
    assert data_sources._format_trivia_qa({
        "question": "Where?",
        "answer": {"value": "Toronto", "aliases": ["Toronto, Ontario"]},
    }) == "Question: Where?\nAnswer: Toronto"
