"""Grounded FL study tool: exclusions, frozen launches, execution, check, validation rules and §9 statistics."""
from hashlib import sha256
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pytest

from master_script.core import grounded
from master_script.core.grounded_job import CAUSAL, validate_job
from master_script.tools import grounded_fl_study as gs
from master_script.tools.collect_guard_traces import SMOKE_TARGET
from tests.grounded_fixtures import SMALL_LAYOUT, SMALL_SIZES, squad_rows
from tests.test_grounded_data import ChatTokenizer, old_pool_hash


class WordTokenizer(ChatTokenizer):
    """Word-level ids, so synthetic passages fit the 384-token budget."""
    def encode(self, text, add_special_tokens=False):
        return [3 + sum(map(ord, w)) % 45 for w in re.findall(r"\S+|\n", text)]


MODEL_CONFIG = SimpleNamespace(max_position_embeddings=4096)


# ------------------------------------------------------------ exclusions

def manifests(tmp_path, rows, split=True):
    """guard_splits_v2 cohorts holding hashes of real (synthetic) rows, as the earlier studies wrote them."""
    hashes = iter(old_pool_hash(r) for r in rows[::7])
    roles = {"final": 4, "attacker_tuning": 4, "attacker_evaluation": 10, "matched_control_evaluation": 20, "train": 12}
    paths = []
    for role, count in roles.items():
        path = tmp_path / f"{role}.json"
        path.write_text(json.dumps({"schema": "guard_splits_v2", "groups": {next(hashes): role for _ in range(count)}}))
        paths.append(path)
    return paths


def test_exclusions_need_every_earlier_cohort(tmp_path):
    rows = squad_rows()
    paths = manifests(tmp_path, rows)
    excluded, sources, roles = gs.exclusions(paths)
    assert SMOKE_TARGET in excluded and len(excluded) == 51 and roles["matched_control_evaluation"] == 20
    with pytest.raises(ValueError, match="matched_control_evaluation"):
        gs.exclusions([p for p in paths if "matched" not in p.name])
    launch = tmp_path / "launch.json"
    launch.write_text(json.dumps({"schema": "guard_collection_v4",
                                  "excluded_targets": list(json.loads(paths[0].read_text())["groups"])}))
    # A launch's generic exclusion never hides a cohort's own role, in either order.
    assert gs.exclusions([launch, *paths])[2] == gs.exclusions([*paths, launch])[2]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "other", "groups": {"a": "b"}}))
    with pytest.raises(ValueError, match="guard_splits_v2"):
        gs.exclusions([bad, *paths])


@pytest.fixture
def small(monkeypatch):
    for name, value in (("DATASTORE_TARGETS", 2), ("CONTROL_TARGETS", 2), ("PILOT_TARGETS", 1), ("BRIDGE_TARGETS", 1),
                        ("CONTROL_MIN_TOKENS", 10)):
        monkeypatch.setattr(gs, name, value)


@pytest.fixture
def built(tmp_path, small):
    rows = squad_rows()
    report = gs.build(tmp_path / "build", manifests(tmp_path, rows), rows=rows, tokenizer=WordTokenizer(),
                      model_config=MODEL_CONFIG, layout=SMALL_LAYOUT, sizes=SMALL_SIZES)
    return tmp_path / "build" / "study.json", report


def test_build_writes_the_study_and_its_audit(built):
    path, report = built
    study = json.loads(path.read_bytes())
    assert report["study_sha256"] == grounded.study_sha256(study)
    assert report["exclusions"]["matched"] == 50 and report["exclusions"]["excluded_questions"] == 50
    assert report["exclusions"]["allowed_unmatched"] == [SMOKE_TARGET] and "hashes" not in report["exclusions"]
    assert all("tokens" in p for p in study["cohorts"]["V"]["passages"].values())


# ---------------------------------------------------------------- prepare

def test_v_slices_are_disjoint_and_control_passages_are_long_enough(built):
    study = json.loads(built[0].read_bytes())
    slices = gs.allocation(study)
    used = slices["datastore"] + slices["control"] + slices["pilot"] + slices["bridge"] + slices["timing"]
    assert len(used) == len(set(used)) == 7
    tokens = lambda i: study["cohorts"]["V"]["passages"][study["cohorts"]["V"]["targets"][i]["passage"]]["tokens"]
    assert all(tokens(i) >= gs.CONTROL_MIN_TOKENS for i in slices["control"])


def test_jobs_set_the_validated_causal_configuration_explicitly(built):
    study = json.loads(built[0].read_bytes())
    jobs, _ = gs.plan("pilot", study)
    paired = [j for j in jobs if j["kind"] == "paired"]
    assert all({k: j["config"][k] for k in CAUSAL} == CAUSAL for j in paired)
    assert all((j["config"]["causal_score"], j["config"]["probe_epochs"], j["config"]["probe_lr"]) == ("cosine", 12, 0.005)
               for j in paired)
    assert {(j["arm"], j["epsilon"]) for j in paired} == {(a, e) for a in ("RG", "CB-AO") for e in ("inf", 64, 16)} | {("CB-LM", "inf")}
    sigma = {str(j["epsilon"]): j["defense"].get("noise_multiplier") for j in paired}
    assert sigma == {"inf": None, "64": 1.91, "16": 5.45}
    assert all(j["config"]["max_length"] == 384 and j["config"]["threshold_mode"] == "fixed" for j in jobs)
    assert sorted(j["config"]["seed"] for j in jobs if j["kind"] == "public") == list(gs.PUBLIC_SEEDS)
    assert all(gs.V_SEED_BAND[0] <= j["config"]["seed"] <= gs.V_SEED_BAND[1] for j in jobs)
    bad = dict(paired[0], config={**paired[0]["config"], "causal_score": "projection"})
    with pytest.raises(ValueError, match="validated configuration"):
        validate_job(bad)


def test_job_rules():
    study = {"cohorts": {}}  # job_document only hashes the study
    job = gs.job_document(study, "control", "RG", "V", target_index=3, control={"epochs": 3, "lr": 1e-4})
    assert job["config"]["seed"] == 8003
    with pytest.raises(ValueError, match="1e-4"):
        gs.job_document(study, "control", "RG", "V", target_index=3, control={"epochs": 3, "lr": 2e-5})
    with pytest.raises(ValueError, match="band"):
        gs.job_document(study, "paired", "RG", "V", target_index=200)
    with pytest.raises(ValueError, match="DP-SGD"):
        validate_job(dict(job, defense={"mechanism": "dp_sgd"}))
    with pytest.raises(ValueError, match="train"):
        validate_job(dict(job, arm="CB-AO"))
    with pytest.raises(ValueError, match="Unknown config"):
        validate_job(dict(job, config={**job["config"], "typo_field": 1}))
    assert gs.expected_epsilon(64)[0] == 51 and abs(gs.expected_epsilon(16)[1] - 16) < 0.05


def test_prepare_freezes_the_launch_and_the_rerun_needs_its_rule(built, tmp_path):
    study_path, _ = built
    launch = gs.prepare("control", tmp_path / "control", study_path)
    assert launch["stage"] == 0 and len(launch["jobs"]) == 2
    assert all(json.loads((tmp_path / "control" / j["file"]).read_bytes())["control"] == {"epochs": 3, "lr": 1e-4}
               for j in launch["jobs"])
    assert (tmp_path / "control" / "protocol.json").read_bytes() == gs.PROTOCOL.read_bytes()
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps({"schema": "grounded_control_analysis_v1", "status": "sensitive"}))
    with pytest.raises(ValueError, match="6-epoch"):
        gs.prepare("control", tmp_path / "rerun-refused", study_path, previous=previous)
    previous.write_text(json.dumps({"schema": "grounded_control_analysis_v1", "status": "rerun_at_6_epochs"}))
    rerun = gs.prepare("control", tmp_path / "rerun", study_path, previous=previous)
    assert all(json.loads((tmp_path / "rerun" / j["file"]).read_bytes())["control"]["epochs"] == 6 for j in rerun["jobs"])
    assert [j["target_index"] for j in rerun["jobs"]] == [j["target_index"] for j in launch["jobs"]]
    with pytest.raises(FileExistsError):
        gs.prepare("control", tmp_path / "control", study_path)


# ---------------------------------------------------------- resolve and run

def fake_result(job, target, kind_body):
    return {"schema": "grounded_result_v1", "status": "complete", "run_id": "grounded_x",
            "job_sha256": sha256(json.dumps(job, sort_keys=True).encode()).hexdigest(), "kind": job["kind"],
            "arm": job["arm"], "cohort": job["cohort"], "epsilon_budget": job["epsilon"], "seed": job["config"]["seed"],
            "target_index": job["target_index"], "implementation_fingerprint": gs.implementation_fingerprint(),
            "timings": {"total_seconds": 1.0}, **kind_body, **({"target_sha256": target} if isinstance(target, str) else {})}


def probes(f1, entail=None, refused=False):
    entail = f1 if entail is None else entail
    return {"probes": [{"f1": f1, "entailment": entail, "refused": refused} for _ in range(3)],
            "verbatim": {"score": 1.0 if f1 > 0.5 else 0.0}}


def learning(nll, match):
    return {"passage_nll": nll, "continuation_match": match, "passage_tokens": 80}


def control_body(learned=True, r=0.3, auc_gap=0.5):
    return {"control": {"epochs": 3, "lr": 1e-4}, "privacy": {"W1": {"epsilon": "inf"}, "W0": {"epsilon": "inf"}},
            "worlds": {"W1": {"learning": learning(1.0 if learned else 3.0, 0.8), "retention": {"f1": r, "probes": []},
                              "natural_questions": {"training_only": probes(0.5 + auc_gap)}},
                       "W0": {"learning": learning(3.0, 0.1), "retention": {"f1": 0.0, "probes": []},
                              "natural_questions": {"neither": probes(0.5)}}}}


def install_runner(monkeypatch, launch_dir, bodies):
    """Stand-in for the job subprocess: write a result and a weight file for retirement."""
    def run(command, env, check):
        job = json.loads(Path(command[-2]).read_bytes())
        out = Path(command[-1])
        artifact = out / "artifacts" / "result"
        artifact.mkdir(parents=True)
        (artifact / "model.safetensors").write_bytes(b"weights")
        cohort = json.loads((launch_dir / "cohort.json").read_bytes())
        index = int(Path(command[-1]).name.split("-")[1])
        result = fake_result(job, cohort["targets"][str(index)], bodies(job, cohort["targets"][str(index)]))
        for path in (artifact / "result.json", out / "result.json"):
            path.write_text(json.dumps(result))
    monkeypatch.setattr(gs.subprocess, "run", run)


def test_resolve_run_check_and_analyze_the_control(built, tmp_path, monkeypatch):
    study_path, _ = built
    root = tmp_path / "control"
    gs.prepare("control", root, study_path)
    with pytest.raises(ValueError, match="numeric GPU"):
        gs.execute(root / "launch.json", gpu="x")
    state = gs.execute(root / "launch.json", resolve_only=True, tokenizer=WordTokenizer(), model_config=MODEL_CONFIG)
    study = json.loads(study_path.read_bytes())
    launch = json.loads((root / "launch.json").read_bytes())
    assert state["targets"] == {str(i): study["cohorts"]["V"]["targets"][j["target_index"]]["target_sha256"]
                                for i, j in enumerate(launch["jobs"])}
    install_runner(monkeypatch, root, lambda job, target: control_body())
    with pytest.raises(ValueError, match="not complete"):
        gs.analyze(root, tmp_path / "early.json")
    gs.execute(root / "launch.json", gpu="0", max_jobs=1, scratch_root=tmp_path)
    assert not list((root / "results" / "job-000").rglob("*.safetensors"))
    assert (root / "results" / "job-000" / "artifacts" / "result" / "checkpoint-retirement.json").exists()
    report = gs.check(root)
    assert report["completed"] == 1 and report["planned"] == 2
    text = json.dumps(report)
    for forbidden in ("auc", "f1", "passage_nll", "continuation", "entailment", "score", "retention\"", "learned"):
        assert forbidden not in text, forbidden
    gs.execute(root / "launch.json", gpu="0", max_jobs=5, scratch_root=tmp_path)
    monkeypatch.setattr(gs, "CONTROL_TARGETS", 2)
    monkeypatch.setattr(gs, "CONTROL_LEARNED", 2)
    analysis = gs.analyze(root, tmp_path / "control-analysis.json")
    assert analysis["learned"] == 2 and analysis["status"] in ("sensitive", "not_evaluable")
    with pytest.raises(ValueError, match="already exists"):
        gs.analyze(root, tmp_path / "control-analysis.json")


def test_changed_files_are_refused(built, tmp_path):
    study_path, _ = built
    root = tmp_path / "timing"
    launch = gs.prepare("timing", root, study_path)
    job = root / launch["jobs"][0]["file"]
    job.write_text(job.read_text().replace('"seed": 80', '"seed": 81'))
    with pytest.raises(ValueError, match="job file changed"):
        gs.execute(root / "launch.json", resolve_only=True, tokenizer=WordTokenizer(), model_config=MODEL_CONFIG)


def test_results_must_match_their_declared_job(built, tmp_path):
    study_path, _ = built
    root = tmp_path / "datastore"
    launch = gs.prepare("datastore", root, study_path)
    job = json.loads((root / launch["jobs"][0]["file"]).read_bytes())
    targets = gs.resolve_targets(json.loads(study_path.read_bytes()), launch, [job])["0"]
    body = {"worlds": {"P0": {"datastore": [{"target_sha256": t} for t in targets]}}}
    result = fake_result(job, targets, body)
    assert gs.verify_result(result, launch, job, targets)
    with pytest.raises(ValueError, match="datastore targets"):
        gs.verify_result(fake_result(job, targets, {"worlds": {"P0": {"datastore": []}}}), launch, job, targets)
    with pytest.raises(ValueError, match="Executed job"):
        gs.verify_result(dict(result, job_sha256="0"), launch, job, targets)
    with pytest.raises(ValueError, match="Executed code"):
        gs.verify_result(dict(result, implementation_fingerprint="old"), launch, job, targets)


# ------------------------------------------------------------ statistics

def test_mann_whitney_auc_counts_ties_as_half():
    assert gs.auc([1, 2], [0, 0]) == 1.0 and gs.auc([1], [1]) == 0.5 and gs.auc([0, 2], [1, 1]) == 0.5
    rng = np.random.default_rng(0)
    pos, neg = rng.normal(1, 1, 6), rng.normal(0, 1, 6)
    idx = gs.target_indices(6, draws=50)
    brute = [gs.auc(pos[row], neg[row]) for row in idx]
    assert np.allclose(gs.auc_resampled(pos, neg, idx), brute)


def test_paired_target_bootstrap_is_seeded_and_shared():
    a, b = gs.target_indices(20), gs.target_indices(20)
    assert (a == b).all() and a.shape == (10_000, 20)
    assert (gs.target_indices(20) == np.random.default_rng(20260930).integers(0, 20, size=(10_000, 20))).all()


def test_crossed_cluster_bootstrap_keeps_siblings_together():
    rng = np.random.default_rng(3)
    clusters = ["a", "a", "b", "c", "c", "c"]
    weights = gs.cluster_weights(clusters, rng, draws=200)
    assert (weights[:, 0] == weights[:, 1]).all() and (weights[:, 3] == weights[:, 5]).all()
    assert (weights[:, [0, 2, 3]].sum(axis=1) == 3).all()  # three clusters drawn per resample
    scores = rng.random((4, 6))
    means = gs.model_means(scores, weights)
    brute = np.array([[np.average(scores[m], weights=w) for m in range(4)] for w in weights])
    assert np.allclose(means, brute)
    idx = rng.integers(0, 4, size=(200, 4))
    assert np.allclose(gs.arm_means(means, idx), [means[b, idx[b]].mean() for b in range(200)])


@pytest.mark.parametrize("lo, hi, number", [
    (0.05, 0.2, 1), (0.051, 0.1, 1), (0.01, 0.2, 2), (0.0, 0.2, 6), (-0.3, -0.05, 3), (-0.3, -0.01, 4),
    (-0.04, 0.04, 5), (-0.05, 0.04, 6), (-0.01, 0.06, 6), (-0.2, 0.2, 6),
])
def test_first_match_six_outcome_classification(lo, hi, number):
    assert gs.classify(lo, hi)["number"] == number


def test_readings_and_decision_levels():
    assert gs.h2_reading(gs.classify(0.1, 0.2)) == "RG reduces" and gs.h2_reading(gs.classify(-0.2, -0.01)) == "RG increases"
    assert gs.h2_reading(gs.classify(-0.01, 0.01)) == "no meaningful change"
    assert gs.h7_reading(-0.04, 0.1).startswith("fine-tuning not needed")
    assert gs.h7_reading(-0.3, -0.06) == "RG is meaningfully better" and gs.h7_reading(-0.2, 0.1) == "inconclusive"
    assert gs.h7_reading(-0.05, 0.1) == "inconclusive"
    assert gs.decision_level() == pytest.approx(1 - 0.05 / 7) and gs.decision_level(False) == pytest.approx(1 - 0.05 / 6)


# ------------------------------------------------------------ hypotheses

QUESTIONS = 12
CLUSTERS = [f"article-{i // 3}" for i in range(QUESTIONS)]


def record(arm, eps, i, ref, causal, nq, retention, utility, epsilon=None):
    epsilon = {"epsilon": "inf"} if eps == "inf" else {"epsilon": epsilon or gs.expected_epsilon(eps)[1]}
    rng = np.random.default_rng(i)
    return {"target": f"t{i}", "arm": arm, "epsilon_budget": eps, "privacy": {"W1": epsilon, "W0": epsilon},
            "reference": {f: (ref + rng.normal(0, .1), rng.normal(0, .1)) for f in gs.FORMS},
            "causal": {"record": {"plain": causal + rng.normal(0, .01)}, "template": {"plain": causal}},
            "nq": {s: {"d_plus": nq + rng.normal(0, .1), "d_minus": rng.normal(0, .1)} for s in gs.NQ_SCORERS},
            "retention": retention + rng.normal(0, .01),
            "utility": list(np.clip(utility + rng.normal(0, .02, QUESTIONS), 0, 1)),
            "utility_ids": [f"q{k}" for k in range(QUESTIONS)]}


CHOICES = {arm: {"reference_form": "record", "causal_direction": "record", "nq_scorer": "f1"} for arm in ("RG", "CB-AO")}


def cohort(rg=None, cb=None, n=12):
    rg = {"ref": 0.0, "causal": 0.9, "nq": 1.0, "retention": 0.2, "utility": 0.7, **(rg or {})}
    cb = {"ref": 1.0, "causal": 0.9, "nq": 0.0, "retention": 0.0, "utility": 0.5, **(cb or {})}
    rows = []
    for eps in ("inf", 16):
        rows += [record("RG", eps, i, **rg) for i in range(n)] + [record("CB-AO", eps, i, **cb) for i in range(n)]
    return rows


def run_hypotheses(rows, public_level=0.7, p0_level=0.7, h4_primary=True):
    public = [list(np.full(QUESTIONS, public_level) + 0.01 * s) for s in range(5)]
    return gs.hypotheses(rows, public, list(np.full(QUESTIONS, p0_level)), CHOICES, h4_primary, CLUSTERS)


def test_hypotheses_follow_the_pre_registered_rules():
    out = run_hypotheses(cohort())
    assert out["decision_level"] == pytest.approx(1 - 0.05 / 7) and out["primaries"] == 7
    assert out["H1"]["claimed"] and out["H1"]["D"]["classification"]["number"] == 1
    assert out["H2"]["reading"] == "no meaningful change"
    assert out["H3"]["claimed"] and out["H4"]["claimed"]
    assert out["H5"]["claimed"] and out["H6"]["claimed"]
    assert out["H7"]["reading"].startswith("fine-tuning not needed")
    assert out["cross_channel"]["claimed"]  # H1 and H3


def test_identical_arms_give_exactly_zero_paired_differences():
    same = {"ref": 0.5, "causal": 0.7, "nq": 0.5, "retention": 0.1, "utility": 0.6}
    out = run_hypotheses(cohort(rg=same, cb=same))
    for key in ("H1", "H2", "H3"):
        assert out[key]["D"]["ci95"] == [0.0, 0.0]
    assert out["H4"]["D_H4"]["ci95"] == [0.0, 0.0] and out["H1"]["utility_RG_minus_CBAO"]["ci95"] == [0.0, 0.0]


def test_h4_needs_both_endpoints():
    worse = run_hypotheses(cohort(rg={"retention": 0.0}, cb={"retention": -0.3}))
    assert worse["H4"]["D_H4"]["classification"]["supported"] and not worse["H4"]["R_RG"]["classification"]["supported"]
    assert not worse["H4"]["claimed"]
    matched = run_hypotheses(cohort(rg={"retention": 0.2}, cb={"retention": 0.2}))
    assert matched["H4"]["R_RG"]["classification"]["supported"] and not matched["H4"]["claimed"]
    assert run_hypotheses(cohort(), h4_primary=False)["decision_level"] == pytest.approx(1 - 0.05 / 6)


def test_h1_needs_acceptable_utility_and_h6_needs_no_worse_leakage():
    harmful = run_hypotheses(cohort(rg={"utility": 0.3}))
    assert harmful["H1"]["D"]["classification"]["supported"] and not harmful["H1"]["claimed"]
    assert not harmful["H6"]["claimed"]
    leaky = run_hypotheses(cohort(rg={"ref": 1.0}, cb={"ref": 0.0}))
    assert not leaky["H6"]["reference_auc_CBAO_minus_RG"]["no_worse"] and not leaky["H6"]["claimed"]


def test_h2_is_two_sided_and_feeds_the_cross_channel_rule():
    out = run_hypotheses(cohort(rg={"causal": 1.0, "nq": 0.0}, cb={"causal": 0.6, "nq": 0.0}))
    assert out["H2"]["reading"] == "RG increases" and not out["H3"]["claimed"]
    assert out["cross_channel"]["claimed"]
    none = run_hypotheses(cohort(rg={"nq": 0.0}, cb={"nq": 0.0}))
    assert not none["cross_channel"]["claimed"]


def test_h5_and_h7_readings():
    out = run_hypotheses(cohort(), public_level=0.4, p0_level=0.4)
    assert not out["H5"]["claimed"] and out["H7"]["reading"] == "RG is meaningfully better"
    assert "secondary_vs_eps16" in out["H5"]


def test_runs_with_different_epsilon_are_never_compared():
    rows = cohort()
    rows[-1] = record("CB-AO", 16, 11, 1.0, 0.9, 0.0, 0.0, 0.5, epsilon=17.0)
    with pytest.raises(ValueError, match="refusing the comparison"):
        run_hypotheses(rows)


# ------------------------------------------------------------ validation

def p0_row(gap, n=60, refuse_all=False):
    rng = np.random.default_rng(1)
    targets = [{"target_sha256": f"t{i}", "cells": {"library_on": probes(float(np.clip(0.5 + gap + rng.normal(0, .2), 0, 1))),
                                                    "library_off": probes(float(np.clip(0.5 + rng.normal(0, .2), 0, 1)),
                                                                          refused=refuse_all)}} for i in range(n)]
    never = [probes(float(np.clip(0.5 + rng.normal(0, .2), 0, 1))) for _ in range(8)]
    return [{"result": {"worlds": {"P0": {"datastore": targets, "never_documents": never}}}}]


def test_datastore_check_passes_only_with_a_lower_bound_of_070():
    strong = gs.analyze_datastore(p0_row(0.5))
    assert strong["eligible_scorers"] == ["f1", "entailment"] and strong["H3"] == "evaluable"
    assert strong["scorers"]["f1"]["ci95"][0] >= 0.70 and "secondary_vs_never_documents" in strong["scorers"]["f1"]
    weak = gs.analyze_datastore(p0_row(0.0))
    assert weak["eligible_scorers"] == [] and weak["H3"].startswith("not evaluable")
    assert gs.analyze_datastore(p0_row(0.5, refuse_all=True))["scorers"]["f1"]["refusal_excluded"]["targets"] == 0
    with pytest.raises(ValueError, match="at least 60"):
        gs.analyze_datastore(p0_row(0.5, n=59))


def test_learning_check_thresholds():
    ok = {"learning": learning(1.5, 0.5)}
    base = {"learning": learning(3.0, 0.0)}
    assert gs.learned(ok, base)
    assert not gs.learned({"learning": learning(1.51, 0.9)}, base)
    assert not gs.learned({"learning": learning(1.0, 0.49)}, base)


def control_rows(learned=12, r=0.3, auc_gap=0.5, epochs=3):
    rows = []
    for i in range(12):
        body = control_body(learned=i < learned, r=r, auc_gap=auc_gap)
        body["control"]["epochs"] = epochs
        body["target_sha256"] = f"t{i}"
        rows.append({"result": body})
    return rows


def test_positive_control_rules():
    rerun = gs.analyze_control(control_rows(learned=9))
    assert rerun["status"] == "rerun_at_6_epochs" and rerun["h4_primary"] is None and "retention_sensitivity" not in rerun
    failed = gs.analyze_control(control_rows(learned=9, epochs=6))
    assert failed["status"] == "failed_to_learn" and failed["h4_primary"] is False and "99.2%" in failed["H4"]
    good = gs.analyze_control(control_rows(learned=10))
    assert good["status"] == "sensitive" and good["h4_primary"] is True
    assert good["retention_sensitivity"]["learned_targets"] == 10
    flat = gs.analyze_control(control_rows(learned=12, r=0.04))
    assert flat["status"] == "not_evaluable" and flat["h4_primary"] is False
    # The membership AUC is secondary: insensitive, yet H4 keeps its primary status.
    insensitive = gs.analyze_control(control_rows(learned=12, auc_gap=0.0))
    assert insensitive["h4_primary"] is True and not insensitive["secondary_membership_auc"]["f1"]["sensitive"]
    with pytest.raises(ValueError, match="12 targets"):
        gs.analyze_control(control_rows()[:11])


def test_attacker_choices_prefer_the_standard_form_on_ties():
    assert gs.choose({"record": 0.7, "template": 0.7}, "record") == "record"
    assert gs.choose({"record": 0.6, "template": 0.7}, "record") == "template"
    rows = [r for r in cohort() if r["epsilon_budget"] == "inf"]
    choices = gs.attacker_choices(rows, eligible=["f1"])
    assert set(choices) == {"RG", "CB-AO"} and choices["RG"]["nq_scorer"] == "f1"
    assert "entailment" not in choices["RG"]["nq_tuning_auc"]
    assert gs.attacker_choices(rows, eligible=[])["RG"]["nq_scorer"] is None
    assert gs.projection(np.random.default_rng(0).normal(0, 0.1, 10_000), 4, 20, 0.95) == pytest.approx(0.0877, abs=0.005)


def test_paired_records_are_built_from_job_results():
    cell = lambda v: probes(v)
    result = {"target_sha256": "t", "arm": "RG", "epsilon_budget": "inf", "privacy": {"W1": {"epsilon": "inf"}, "W0": {"epsilon": "inf"}},
              "worlds": {"W1": {"reference": {"record": 1.0, "template": 2.0}, "retention": {"f1": 0.6},
                                "natural_questions": {"both": cell(0.9), "training_only": cell(0.4)},
                                "utility": {"F": [{"query_id": "q", "f1": 0.5}], "F_P": []}},
                         "W0": {"reference": {"record": 0.5, "template": 1.0}, "retention": {"f1": 0.2},
                                "natural_questions": {"library_only": cell(0.8), "neither": cell(0.1)},
                                "utility": {"F": [{"query_id": "q", "f1": 0.3}], "F_P": []}}},
              "causal": {"record": {"plain": {"auc": 0.9}, "release_noise": {"auc": 0.6}}}}
    r = gs.paired_record(result)
    assert r["reference"]["template"] == (2.0, 1.0) and r["retention"] == pytest.approx(0.4)
    assert r["nq"]["f1"]["d_plus"] == pytest.approx(0.8) and r["nq"]["f1"]["d_minus"] == pytest.approx(0.1)
    assert r["causal"]["record"] == {"plain": 0.9, "release_noise": 0.6} and r["utility"] == [0.5]
