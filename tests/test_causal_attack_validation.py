"""Causal-attack validation: score telemetry, frozen launches, selection and analysis."""
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import json
import sys
from types import ModuleType

import numpy as np
import pytest

from master_script.core.attacks import amia, causal_probe
from master_script.core.calibration import nonmember_threshold
from master_script.tools import causal_attack_validation as cav


def legacy_alignment_score(released, public_direction):
    """alignment_score before telemetry (commit 75612fe)."""
    dot = norm = 0.
    for value, direction in zip(released, public_direction):
        a, b = value.ravel(), direction.ravel()
        for start in range(0, a.size, 65536):
            x, y = a[start:start+65536].astype(float), b[start:start+65536].astype(float)
            dot += float(np.sum(x * y)); norm += float(np.sum(y * y))
    return dot / max(np.sqrt(norm), 1e-12)


def test_alignment_terms_keep_the_score_bit_identical_and_add_norms():
    rng = np.random.default_rng(0)
    released = [rng.normal(size=(300, 400)).astype(np.float32), rng.normal(size=7).astype(np.float32)]
    direction = [rng.normal(size=(300, 400)).astype(np.float32), rng.normal(size=7).astype(np.float32)]
    terms = causal_probe.alignment_terms(released, direction)
    assert causal_probe.alignment_score(released, direction) == legacy_alignment_score(released, direction)
    assert causal_probe.score_from_terms(terms) == legacy_alignment_score(released, direction)
    flat = lambda arrays: np.concatenate([a.ravel().astype(float) for a in arrays])
    assert terms["released_norm"] == pytest.approx(np.linalg.norm(flat(released)), rel=1e-12)
    assert terms["direction_norm"] == pytest.approx(np.linalg.norm(flat(direction)), rel=1e-12)
    assert terms["dot"] == pytest.approx(flat(released) @ flat(direction), rel=1e-9)
    with pytest.raises(ValueError):
        causal_probe.alignment_terms([np.ones(3)], [np.ones(2)])


def stub_flower(monkeypatch):
    class Client:
        def to_client(self): return self

    class App:
        def __init__(self, client_fn=None, server_fn=None): self.client_fn, self.server_fn = client_fn, server_fn

    class Strategy:
        def __init__(self, **kwargs): pass

    exports = {"flwr": {}, "flwr.client": {"NumPyClient": Client, "ClientApp": App},
               "flwr.common": {"ndarrays_to_parameters": lambda a: a, "parameters_to_ndarrays": lambda a: a},
               "flwr.server": {"ServerApp": App, "ServerAppComponents": SimpleNamespace, "ServerConfig": SimpleNamespace},
               "flwr.server.strategy": {"FedAvg": Strategy},
               "transformers": {"AutoModelForCausalLM": object, "AutoTokenizer": object}}
    for name, values in exports.items():
        module = ModuleType(name); module.__dict__.update(values); monkeypatch.setitem(sys.modules, name, module)


def test_observed_causal_trials_record_server_visible_terms(monkeypatch):
    from master_script.core import runtime_memory
    stub_flower(monkeypatch)
    monkeypatch.setattr(amia, "get_parameters", lambda p: [np.ones(2)])
    released = [[np.array([3., 4.])], [np.array([0., 2.])]]

    def simulate(server_app, client_app, **kwargs):
        strategy = server_app.server_fn(None).strategy
        for i, arrays in enumerate(released):
            strategy.aggregate_fit(i + 1, [(None, SimpleNamespace(parameters=arrays, num_examples=4,
                                                                  metrics={"partition_id": 0}))], [])
    monkeypatch.setattr(runtime_memory, "run_simulation", simulate)
    monkeypatch.setattr(runtime_memory, "simulation_backend", lambda c: {})
    config = SimpleNamespace(num_clients=1, target_client_id=0, attack_trials=2, gradient_threshold=0., seed=7,
                             attack_variant="causal_gradient_alignment")
    probe = SimpleNamespace(_public_direction=[np.array([1., 0.])])
    rows = amia.run_attack_trials("unused", probe, [["SECRET-PARTITION-TEXT"]], config)
    assert [r["score"] for r in rows] == [3., 0.]
    assert rows[0]["alignment_terms"] == {"dot": 3., "direction_norm": 1., "released_norm": 5.}
    assert rows[1]["alignment_terms"]["released_norm"] == 2.
    assert all("SECRET-PARTITION-TEXT" not in json.dumps(r) for r in rows)


def test_calibration_records_terms_for_every_public_batch(monkeypatch):
    records = [f"public record {i}" for i in range(40)]
    monkeypatch.setattr(amia.dataset_sources, "uses_real_dataset", lambda c: True)
    monkeypatch.setattr(amia.dataset_sources, "calibration_records", lambda c, n: records[:n])
    monkeypatch.setattr(amia.dataset_sources, "target_record_for", lambda c, d: "the target")
    monkeypatch.setattr(causal_probe, "protected_gradients",
                        lambda probe, tok, batch, cfg: [np.array([float(len("".join(batch))), 1.])])
    tokenizer = lambda text, truncation, max_length: {"input_ids": [ord(c) for c in text] + [0]}
    config = SimpleNamespace(calibration_nonmember_count=20, adversary_negative_count=4, attack_batch_size=4, seed=7,
                             attack_variant="causal_gradient_alignment", calibration_fpr=0.05, max_length=64)
    probe = SimpleNamespace(_public_direction=[np.array([1., 1.])])
    result = amia.calibrate_probe(None, tokenizer, probe, config)
    assert len(result["alignment_terms"]) == 20
    scores = [causal_probe.score_from_terms(t) for t in result["alignment_terms"]]
    assert result["threshold"] == nonmember_threshold(scores, 0.05)["threshold"]


def splits_manifest(path, groups):
    path.write_text(json.dumps({"schema": "guard_splits_v2", "sources": [], "groups": groups, "held_out_variants": []}))
    return path


@pytest.fixture
def pilot_cohort(tmp_path):
    groups = {f"{i:064x}": role for i, role in enumerate(["train", "validation", "test", "final", "final", "final", "final"])}
    return splits_manifest(tmp_path / "cohort.json", groups)


def test_prepare_stage_a_writes_frozen_pipeline_free_causal_jobs(tmp_path, pilot_cohort):
    from master_script.core.queue import load_batch
    launch = cav.prepare(tmp_path / "a", "a", [pilot_cohort])
    assert launch["schema"] == "guard_collection_v4" and len(launch["jobs"]) == 8
    assert [j["seed"] for j in launch["jobs"]] == [6000, 6000, 6001, 6001, 6002, 6002, 6003, 6003]
    assert {j["probe_epochs"] for j in launch["jobs"]} == {12, 80}
    assert set(json.loads(pilot_cohort.read_text())["groups"]) | {cav.SMOKE_TARGET} == set(launch["excluded_targets"])
    for job in launch["jobs"]:
        config, spec = load_batch([tmp_path / "a" / job["config"]]).pairs[0]
        assert spec.name == "amia" and spec.pipeline is None
        assert (config.attack_variant, config.attack_trials, config.attack_targets) == ("causal_gradient_alignment", 40, 1)
        assert (config.calibration_nonmember_count, config.observation_defense, config.counterbalance_trials) == (100, "none", True)
        assert config.probe_epochs == job["probe_epochs"] and config.seed == job["seed"]
    assert launch["protocol_sha256"] == cav.digest(cav.PROTOCOL)


def test_prior_collection_launch_contributes_its_exclusions(tmp_path, pilot_cohort):
    v3_target = "cd" * 32
    prior = tmp_path / "v4-launch.json"
    prior.write_text(json.dumps({"schema": "guard_collection_v4", "excluded_targets": [v3_target, cav.SMOKE_TARGET]}))
    launch = cav.prepare(tmp_path / "a", "a", [pilot_cohort, prior])
    assert v3_target in launch["excluded_targets"]
    assert [s["schema"] for s in launch["exclusion_sources"]] == ["guard_splits_v2", "guard_collection_v4"]
    unknown = tmp_path / "other.json"
    unknown.write_text(json.dumps({"schema": "something_else", "groups": {v3_target: "x"}}))
    with pytest.raises(ValueError, match="exclusions need"):
        cav.prepare(tmp_path / "b", "a", [pilot_cohort, unknown])


def test_prepare_refuses_unsafe_cohorts(tmp_path, pilot_cohort):
    no_final = splits_manifest(tmp_path / "dev.json", {"ab" * 32: "train"})
    with pytest.raises(ValueError, match="reserved-final"):
        cav.prepare(tmp_path / "x", "a", [no_final])
    with pytest.raises(ValueError, match="band"):
        cav.prepare(tmp_path / "y", "a", [pilot_cohort], first_seed=6097)
    with pytest.raises(ValueError, match="selection"):
        cav.prepare(tmp_path / "z", "b", [pilot_cohort])
    cav.prepare(tmp_path / "a", "a", [pilot_cohort])
    with pytest.raises(FileExistsError):
        cav.prepare(tmp_path / "a", "a", [pilot_cohort])


def fake_target(config, default=""):
    return f"private target for seed {config.seed}"


def test_unchanged_collector_resolves_and_guards_the_launch(tmp_path, pilot_cohort, monkeypatch):
    from master_script.core import datasets
    from master_script.tools import collect_guard_traces
    monkeypatch.setattr(datasets, "target_record_for", fake_target)
    launch = cav.prepare(tmp_path / "a", "a", [pilot_cohort])
    collect_guard_traces.execute(tmp_path / "a" / "launch.json", resolve_only=True)
    cohort = json.loads((tmp_path / "a" / "cohort.json").read_text())
    assert sorted(cohort["groups"].values()) == ["attacker_tuning"] * 4
    # A historical target selected again is refused before any job.
    consumed = sha256(fake_target(SimpleNamespace(seed=6002)).encode()).hexdigest()
    reused = splits_manifest(tmp_path / "old.json", {**json.loads(pilot_cohort.read_text())["groups"], consumed: "test"})
    cav.prepare(tmp_path / "b", "a", [reused])
    with pytest.raises(ValueError, match="excluded from an earlier cohort") as refused:
        collect_guard_traces.execute(tmp_path / "b" / "launch.json", resolve_only=True)
    message = str(refused.value)
    assert f"seed 6002 selected target {consumed[:12]}," in message
    assert consumed not in message and "smoke" not in message and "private target" not in message
    assert not (tmp_path / "b" / "cohort.json").exists() and not (tmp_path / "b" / "progress.json").exists()


def test_collector_names_the_smoke_target_only_when_it_is_selected(tmp_path, pilot_cohort, monkeypatch):
    from master_script.core import datasets
    from master_script.tools import collect_guard_traces
    monkeypatch.setattr(datasets, "target_record_for", fake_target)
    monkeypatch.setattr(collect_guard_traces.subprocess, "run", lambda *a, **kw: pytest.fail("GPU job launched"))
    smoke = sha256(fake_target(SimpleNamespace(seed=6001)).encode()).hexdigest()
    monkeypatch.setattr(collect_guard_traces, "SMOKE_TARGET", smoke)
    monkeypatch.setattr(cav, "SMOKE_TARGET", smoke)
    cav.prepare(tmp_path / "a", "a", [pilot_cohort])
    for resolve_only in (True, False):
        with pytest.raises(ValueError, match="^Job seed 6001 selected the diagnostic smoke target$"):
            collect_guard_traces.execute(tmp_path / "a" / "launch.json", gpu="0", resolve_only=resolve_only)
    assert not (tmp_path / "a" / "cohort.json").exists() and not (tmp_path / "a" / "results").exists()


def fabricate(root, strength, *, private_shift=0.0, seed=0):
    """Complete a prepared launch with synthetic results in the collector's layout."""
    rng = np.random.default_rng(seed)
    launch = json.loads((root / "launch.json").read_text())
    targets = {str(j["seed"]): sha256(fake_target(SimpleNamespace(seed=j["seed"])).encode()).hexdigest() for j in launch["jobs"]}

    def terms(member, effect):
        released = float(rng.lognormal(0, 0.6))  # batch size dominates the projection
        cosine = 0.10 + rng.normal(0, 0.01) + (effect if member else 0.0)
        return {"dot": cosine * released * 2.0, "direction_norm": 2.0, "released_norm": released}

    completed = []
    for index, job in enumerate(launch["jobs"]):
        effect = strength(job)
        trials = []
        for pair in range(20):
            for member in (True, False):
                t = terms(member, effect)
                t["dot"] += private_shift * 2.0 * t["released_norm"]
                trials.append({"trial_id": len(trials), "truth_member": member, "batch_pair_seed": 1000 + pair,
                               "target_sha256": targets[str(job["seed"])], "alignment_terms": t,
                               "score": cav.score(t, "projection")})
        calibration = [terms(False, 0.0) for _ in range(100)]
        result = {"status": "complete", "implementation_fingerprint": launch["implementation_fingerprint"],
                  "config": {"probe_epochs": job["probe_epochs"]}, "attack_trials": trials,
                  "calibration": {"alignment_terms": calibration, "threshold": nonmember_threshold(
                      [cav.score(t, "projection") for t in calibration], 0.05)["threshold"]}}
        path = root / "results" / f"job-{index:03d}" / "run" / "0000-x.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(result))
        completed.append({"job": index, "result": str(path.relative_to(root)), "sha256": cav.digest(path)})
    (root / "progress.json").write_text(json.dumps({"completed": completed, "active_job": None, "targets": targets}))
    groups = {t: launch["jobs"][0]["role"] for t in targets.values()}
    splits_manifest(root / "splits.complete.json", groups)


def test_selection_and_analysis_follow_the_fixed_rules(tmp_path, pilot_cohort, monkeypatch):
    monkeypatch.setattr(cav, "BOOTSTRAP", 2000)
    monkeypatch.setattr(cav, "PERMUTATIONS", 300)
    cav.prepare(tmp_path / "a", "a", [pilot_cohort])
    fabricate(tmp_path / "a", lambda job: 0.03 if job["probe_epochs"] == 80 else 0.0)
    selection = cav.select(tmp_path / "a", tmp_path / "selection.json")
    assert selection["selected"] == {"probe_epochs": 80, "score": "cosine"}
    with pytest.raises(ValueError, match="overwrite"):
        cav.select(tmp_path / "a", tmp_path / "selection.json")
    with pytest.raises(ValueError, match="Stage A targets"):
        cav.prepare(tmp_path / "bad", "b", [pilot_cohort], tmp_path / "selection.json")
    stage_a = tmp_path / "a" / "splits.complete.json"
    launch = cav.prepare(tmp_path / "b", "b", [pilot_cohort, stage_a], tmp_path / "selection.json")
    assert [j["seed"] for j in launch["jobs"]] == list(range(6004, 6014))
    assert {j["probe_epochs"] for j in launch["jobs"]} == {80}
    fabricate(tmp_path / "b", lambda job: 0.03, seed=1)
    report = cav.analyze(tmp_path / "b", tmp_path / "selection.json", tmp_path / "analysis.json")
    assert report["decision"] == "effective" and report["primary"]["ci95"][0] >= 0.60
    assert report["paired"]["sign_test_p_one_sided"] < 1e-6
    assert not report["calibration_failure"] and report["operating_point"]["fpr"] <= 0.15


def test_analysis_reports_null_attacks_and_calibration_failure(tmp_path, pilot_cohort, monkeypatch):
    monkeypatch.setattr(cav, "BOOTSTRAP", 2000)
    monkeypatch.setattr(cav, "PERMUTATIONS", 300)
    cav.prepare(tmp_path / "a", "a", [pilot_cohort])
    fabricate(tmp_path / "a", lambda job: 0.0)
    cav.select(tmp_path / "a", tmp_path / "selection.json")
    for name, shift in (("null", 0.0), ("shifted", 0.05)):
        cav.prepare(tmp_path / name, "b", [pilot_cohort, tmp_path / "a" / "splits.complete.json"], tmp_path / "selection.json")
        fabricate(tmp_path / name, lambda job: 0.0, private_shift=shift, seed=2)
    null = cav.analyze(tmp_path / "null", tmp_path / "selection.json", tmp_path / "null.json")
    assert null["decision"] == "not_effective" and null["primary"]["ci95"][1] < 0.60
    shifted = cav.analyze(tmp_path / "shifted", tmp_path / "selection.json", tmp_path / "shifted.json")
    assert shifted["calibration_failure"] and shifted["operating_point"]["fpr_ci95"][0] > 0.10
    with pytest.raises(ValueError, match="selection"):
        other = tmp_path / "other-selection.json"
        other.write_text((tmp_path / "selection.json").read_text() + " ")
        cav.analyze(tmp_path / "null", other, tmp_path / "x.json")


def test_analysis_refuses_changed_or_incomplete_results(tmp_path, pilot_cohort):
    cav.prepare(tmp_path / "a", "a", [pilot_cohort])
    fabricate(tmp_path / "a", lambda job: 0.0)
    state = json.loads((tmp_path / "a" / "progress.json").read_text())
    path = tmp_path / "a" / state["completed"][0]["result"]
    result = json.loads(path.read_text()); result["attack_trials"][0].pop("alignment_terms")
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="changed"):
        cav.select(tmp_path / "a", tmp_path / "s.json")
    state["completed"][0]["sha256"] = cav.digest(path)
    (tmp_path / "a" / "progress.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="telemetry"):
        cav.select(tmp_path / "a", tmp_path / "s.json")
    state["completed"].pop()
    (tmp_path / "a" / "progress.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="not complete"):
        cav.select(tmp_path / "a", tmp_path / "s.json")
