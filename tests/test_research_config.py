"""Small regressions for the larger research profile; no downloaded models/data."""
from collections import Counter
from dataclasses import replace

import pytest
import yaml

from master_script.core import datasets
from master_script.core.config import experiment_key
from master_script.core.registry import ATTACKS
from master_script.core.yaml_config import ConfigError, load_config_doc
from master_script.paths import CONFIGS_DIR


def test_research_file_expands_matched_conditions_with_unique_keys():
    doc = yaml.safe_load((CONFIGS_DIR / "pipeline_research_master.yaml").read_text())
    for variant in doc["pipeline"]:
        variant["rag"]["study_file"] = str(CONFIGS_DIR / "pipeline_demo_study.json")
    pairs = load_config_doc(doc)
    assert len(pairs) == len({experiment_key(c, s) for c, s in pairs}) == 99
    counts = Counter(s.pipeline.defense.mechanism for c, s in pairs)
    assert counts == {"none": 33, "dp_sgd": 33, "dp_fedavg": 33}
    for name in ATTACKS:
        grid = [(c, s) for c, s in pairs if s.name == name]
        assert len(grid) == 9
        assert {c.seed for c, s in grid} == {7, 107, 207}
        assert all(c.sim_num_gpus == 1 and c.attack_trials == 200 for c, s in grid)


@pytest.mark.parametrize("variants", [[], [None], [[{}]], ["dp_sgd"]])
def test_invalid_pipeline_lists_are_rejected(variants):
    with pytest.raises(ConfigError):
        load_config_doc({"pipeline": variants, "attacks": {"zlib": {}}})


def test_research_world_and_loss_calibration_use_same_partition_boundaries(monkeypatch):
    monkeypatch.setattr(datasets, "_load_dataset_pool",
                        lambda dataset_name, pool_size, max_chars: tuple(f"record {i}" for i in range(pool_size)))
    cfg = replace(ATTACKS["loss"].config_cls(), dataset_name="squad_research", num_clients=4)
    member = datasets.build_real_membership_world(cfg, True)
    nonmember = datasets.build_real_membership_world(cfg, False)
    assert [len(p) for p in member.partitions] == [33, 32, 32, 32]
    assert member.target_record == datasets.target_record_for(cfg, "unused")
    assert member.partitions[1:] == nonmember.partitions[1:]
    assert member.partitions[0][:-1] == nonmember.partitions[0][:-1]
    calibration = set(datasets.calibration_records(cfg, 64))
    training = {t for parts in (member.partitions, nonmember.partitions) for p in parts for t in p}
    assert len(calibration) == 64 and not calibration & training
    legacy = datasets.build_real_membership_world(replace(cfg, dataset_name="squad"), True)
    assert [len(p) for p in legacy.partitions] == [5, 4, 4, 4]


def test_preparer_is_deterministic_and_balanced():
    from master_script.prepare_research_study import build_study
    paragraphs = [{"context": "word " * 25 + f"answer{i}",
                   "qas": [{"id": str(i), "question": f"Question {i}?",
                            "answers": [{"text": f"answer{i}"}]}]} for i in range(520)]
    source = {"data": [{"title": "fixture", "paragraphs": paragraphs}]}
    study, provenance = build_study(source)
    assert (study, provenance) == build_study(source)
    for corpus in ("public_documents", "private_documents"):
        assert len(study[corpus]) == 256
        ids = {d["id"] for d in study[corpus]}
        assert sum(c["id"] in ids for c in study["membership_candidates"]) == 100
    assert len(study["utility_queries"]) == 20


def test_skipped_rag_evaluation_keeps_training_accounting():
    from master_script.core.rag import evaluate_pipeline
    from types import SimpleNamespace
    privacy = {"steps": 51}
    pipeline = SimpleNamespace(rag=SimpleNamespace(evaluation_trials=2))
    result = evaluate_pipeline({"privacy": privacy}, None, pipeline, trial_id=2)
    assert result["training_privacy"] == privacy
    assert result["rag_evaluation_skipped"]
    assert "rag_conditions" not in result
