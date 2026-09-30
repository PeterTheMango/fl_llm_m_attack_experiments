"""Matched-reference benign controls: request construction, frozen launches and analysis."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from master_script.core.attacks import amia, matched_controls as mc
from master_script.core.guard_features import STRUCTURE_NAMES, STRUCTURE_SCHEMA
from master_script.tools import matched_reference_controls as mrc

CORE = Path(mrc.__file__).resolve().parents[1] / "core"


def test_the_frozen_detector_code_is_byte_identical_to_the_pilot():
    # Hashes of these files at 25cca4d, the guard-v4 pilot commit (fingerprint 489525a1da38).
    pinned = {"guard_features.py": "7ff56e508cedb9f9d5d7b484eedf729aee369659e89816be7f2d8a1ee30bbcae",
              "guard_detector.py": "391a3c5ec8bb22d91e781666eb3d784bb282df7d6f05e51f41b4391ff3a10503",
              "guard_runtime.py": "13c4cc8616470ddf363896f6ff2e4b6382169d5d99aa6a39eaadc1bccc644c32",
              "guard_replay.py": "3cf43687e952f9d8454fa1f5f7e72ebc2929d910d19cba8ac053a0aa6692eaaf"}
    assert {name: sha256((CORE / name).read_bytes()).hexdigest() for name in pinned} == pinned


# ---------------------------------------------------------------- requests

class Tokenizer:
    """Character tokenizer with the call signature the causal code uses."""
    def __call__(self, texts, padding=False, truncation=True, max_length=32, return_tensors="pt"):
        import torch
        rows = [texts] if isinstance(texts, str) else texts
        ids = [[ord(c) % 49 + 1 for c in text][:max_length] for text in rows]
        width = max(map(len, ids))
        return {"input_ids": torch.tensor([x + [0] * (width - len(x)) for x in ids]),
                "attention_mask": torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in ids])}


@pytest.fixture
def tiny_model():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.GPT2Config(vocab_size=50, n_positions=32, n_embd=16, n_layer=1, n_head=2)
    return transformers.GPT2LMHeadModel(config).eval()


def test_step_matched_control_uses_the_causal_request_optimizer(tiny_model):
    from master_script.core.attacks.causal_probe import optimize_request
    config = SimpleNamespace(probe_epochs=4, probe_lr=0.005, max_length=32)
    tokenizer, candidate = Tokenizer(), "the private candidate passage"
    ascent, twin = deepcopy(tiny_model), deepcopy(tiny_model)
    history = optimize_request(ascent, tokenizer, candidate, config)
    twin_history = mc.normalized_steps(twin, mc.candidate_loss(tokenizer, candidate, config), 4, 0.005, +1)
    assert history == twin_history
    assert all(np.array_equal(a, b) for a, b in zip(amia.get_parameters(ascent), amia.get_parameters(twin)))
    # Descent on a public batch: same step count and size, so the same summed step length.
    descent = deepcopy(tiny_model)
    loss = mc.batch_loss(tokenizer, ["public one", "another public record"], config)
    steps = mc.normalized_steps(descent, loss, 4, 0.005, -1)
    assert len(steps) == 4 and float(loss(descent).detach()) < steps[0]
    # Each step has length probe_lr over the unique parameters (tied weights once).
    unique = lambda m: [p.detach().numpy().copy() for p in m.parameters()]
    assert 0 < mc.delta_norm(unique(tiny_model), unique(descent)) <= 4 * 0.005 * (1 + 1e-5)


def test_norm_matched_scaling_and_fedavg():
    rng = np.random.default_rng(1)
    approved = [rng.normal(0, .05, size=(300, 200)).astype("float32"), rng.normal(size=7).astype("float32")]
    updated = [a + rng.normal(0, 1e-3, size=a.shape).astype("float32") for a in approved]
    target_norm = 2.3e-3
    scale = target_norm / mc.delta_norm(approved, updated)
    request = mc.scaled_request(approved, updated, scale)
    assert [r.dtype for r in request] == [a.dtype for a in approved]
    assert mc.delta_norm(approved, request) == pytest.approx(target_norm, rel=1e-4)
    with pytest.raises(ValueError):
        mc.scaled_request(approved, updated, 0.)
    a, b = [np.ones(3, "float32")], [np.full(3, 4., "float32")]
    assert np.array_equal(mc.fedavg([(a, 1), (b, 3)])[0], (a[0] * 1 + b[0] * 3) / 4)


def test_honest_round_trains_every_scheduled_client_from_the_approved_model(monkeypatch):
    calls = []

    class Client:
        def __init__(self, partition, texts, config, **kwargs):
            assert not kwargs  # no guard and no training defense
            self.partition, self.texts = partition, texts

        def fit(self, parameters, fit_config):
            calls.append((self.partition, fit_config["server_round"]))
            parameters[0] += 1.  # a client must receive its own copy
            return [parameters[0] * (self.partition + 1)], len(self.texts), {}

    monkeypatch.setattr(amia, "_ami_flower_client_cls", lambda: Client)
    config = SimpleNamespace(seed=7000, num_clients=4, clients_per_round=4, federated_rounds=3)
    approved = [np.zeros(2, "float32")]
    clients = [["a"], ["b", "c"], ["d"], ["e", "f", "g"]]
    updated, round_id = mc.honest_round(config, clients, approved)
    assert round_id == 4 and sorted(calls) == [(p, 4) for p in range(4)]
    assert np.array_equal(approved[0], np.zeros(2))
    assert np.allclose(updated[0], (1 * 1 + 2 * 2 + 3 * 1 + 4 * 3) / 7)


def test_public_batches_follow_every_other_public_slice(monkeypatch):
    records = [f"public {i}" for i in range(200)]
    monkeypatch.setattr(amia.dataset_sources, "calibration_records", lambda c, n: records[:n])
    config = SimpleNamespace(matched_descent_requests=3, attack_batch_size=4, adversary_negative_count=16,
                             threshold_mode="fixed", calibration_nonmember_count=100)
    assert mc.public_records(config) == records[16:28]
    assert mc.public_batches(config) == [records[16:20], records[20:24], records[24:28]]
    assert mc.public_records(replace_ns(config, threshold_mode="calibrated")) == records[116:128]
    assert amia.matched_public_records(replace_ns(config, matched_descent_requests=0)) == []


def replace_ns(namespace, **changes):
    return SimpleNamespace(**{**vars(namespace), **changes})


def fake_detector(path, threshold=0.5):
    path.write_text(json.dumps({"schema": STRUCTURE_SCHEMA, "features": list(STRUCTURE_NAMES),
                                "mean": [0.] * len(STRUCTURE_NAMES), "scale": [1.] * len(STRUCTURE_NAMES),
                                "weights": [1.] + [0.] * (len(STRUCTURE_NAMES) - 1), "intercept": 0.,
                                "threshold": threshold,
                                "split_groups": {"train": ["a"], "validation": ["b"], "test": ["c"]}}))
    return path, sha256(path.read_bytes()).hexdigest()


def guard_settings(tmp_path):
    from master_script.core.guard_runtime import parse_guard
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"schema": "client_guard_v1", "observation_architecture": "causal_lm"}))
    detector, digest = fake_detector(tmp_path / "detector.json")
    value = {"mode": "shadow", "policy_file": str(policy), "release_budget": 3, "diagnostic": True,
             "feature_schema": STRUCTURE_SCHEMA, "detector_file": str(detector), "detector_sha256": digest}
    return replace(parse_guard(value, "<config>"), runtime_directory=str(tmp_path / "local"))


def test_controls_share_the_causal_reference_and_take_no_ledger_debit(tmp_path, tiny_model, monkeypatch):
    from master_script.core import model_io
    from master_script.core.attacks.causal_probe import optimize_request
    from master_script.core.guard_runtime import prepare_guard, read_guard_events
    records = [f"public record number {i}" for i in range(40)]
    monkeypatch.setattr(amia.dataset_sources, "calibration_records", lambda c, n: records[:n])
    monkeypatch.setattr(model_io, "load_causal_model", lambda path: deepcopy(tiny_model))
    monkeypatch.setattr(amia, "client_device", lambda config: "cpu")
    config = SimpleNamespace(matched_descent_requests=3, matched_honest_round=True, attack_batch_size=2,
                             adversary_negative_count=4, threshold_mode="fixed", calibration_nonmember_count=0,
                             probe_epochs=3, probe_lr=0.005, max_length=32, seed=7000, federated_rounds=3)
    tokenizer, target = Tokenizer(), "the private candidate passage"
    approved = amia.get_parameters(tiny_model)
    causal_model = deepcopy(tiny_model)
    optimize_request(causal_model, tokenizer, target, config)
    causal = amia.get_parameters(causal_model)
    honest = [a + np.float32(1e-3) for a in approved]
    monkeypatch.setattr(mc, "honest_round", lambda cfg, clients, w: (honest, 4))
    guard = prepare_guard(guard_settings(tmp_path), "observation:7000", approved, rounds=2)
    out = mc.run_matched_controls(config, "unused", tokenizer, [["SECRET-CLIENT-TEXT"]], approved, causal, target, guard)
    arms = [c["arm"] for c in out["controls"]]
    assert arms == ["M1", "M1", "M1", "M2", "M2raw"]
    events = {e["scope"]: e for e in read_guard_events(tmp_path / "local")}
    assert set(events) == {c["scope"] for c in out["controls"]}
    assert {e["reference_sha256"] for e in events.values()} == {sha256("".join(guard.reference_digests).encode()).hexdigest()}
    assert all(e["reservation_count"] == 0 and e["accounting_scope"] == "observation" for e in events.values())
    assert all(e["feature_schema"] == STRUCTURE_SCHEMA and e["detector_score"] is not None for e in events.values())
    m2 = next(c for c in out["controls"] if c["arm"] == "M2")
    assert m2["delta_norm_ratio_to_causal"] == pytest.approx(1, abs=1e-4)
    assert all(c["loss_after"] < c["loss_before"] and c["steps"] == 3 for c in out["controls"] if c["arm"] == "M1")
    assert out["causal_candidate_loss_after"] > mc.candidate_loss(tokenizer, target, config)(tiny_model).item()
    text = json.dumps(out)
    assert "SECRET-CLIENT-TEXT" not in text and "public record" not in text and target not in text


def test_matched_controls_are_explicit_causal_only_fields():
    from master_script.core.config import validate_attack_config
    base = replace(amia.AmiaConfig(), attack_variant="causal_gradient_alignment")
    assert (amia.AmiaConfig().matched_descent_requests, amia.AmiaConfig().matched_honest_round) == (0, False)
    validate_attack_config(replace(base, matched_honest_round=True))
    for bad, message in ((dict(matched_descent_requests=-1), "matched_descent_requests"),
                         (dict(matched_honest_round="yes"), "matched_descent_requests"),
                         (dict(attack_variant="probe_head", matched_honest_round=True), "causal variant"),
                         (dict(adaptive_public_steps=2, matched_honest_round=True), "unadapted"),
                         (dict(matched_descent_requests=2), "real models")):
        with pytest.raises(ValueError, match=message):
            validate_attack_config(replace(base, **bad))


def test_controls_refuse_before_training_without_a_pinned_detector(monkeypatch):
    monkeypatch.setattr(amia, "federated_fine_tune", lambda *a, **k: pytest.fail("training started"))
    config = replace(amia.AmiaConfig(), attack_variant="causal_gradient_alignment", matched_honest_round=True)
    no_detector = SimpleNamespace(client_guard=SimpleNamespace(detector_file=None))
    for pipeline in (None, SimpleNamespace(client_guard=None), no_detector):
        with pytest.raises(ValueError, match="pinned detector"):
            amia._run_target(config, "unused", pipeline)


# ---------------------------------------------------------------- launches

def splits(path, groups):
    path.write_text(json.dumps({"schema": "guard_splits_v2", "sources": [], "groups": groups, "held_out_variants": []}))
    return path


@pytest.fixture
def study(tmp_path, monkeypatch):
    """A final protocol pinning a fake detector, and every earlier cohort."""
    detector, digest = fake_detector(tmp_path / "pilot-detector.json")
    protocol = json.loads(mrc.PROTOCOL.read_text())
    protocol["detector"]["sha256"] = digest
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    monkeypatch.setattr(mrc, "PROTOCOL", path)
    ids = iter(f"{i:064x}" for i in range(1, 1000))
    cohort = splits(tmp_path / "cohort.json", {next(ids): r for r in ["train", "validation", "test"] + ["final"] * 4})
    stage_a = splits(tmp_path / "stage-a-2.json", {next(ids): "attacker_tuning" for _ in range(4)})
    stage_b = splits(tmp_path / "stage-b.json", {next(ids): "attacker_evaluation" for _ in range(10)})
    launch = tmp_path / "pilot-launch.json"
    launch.write_text(json.dumps({"schema": "guard_collection_v4", "excluded_targets": [next(ids) for _ in range(8)]}))
    return SimpleNamespace(detector=detector, digest=digest, manifests=[cohort, launch, stage_a, stage_b])


def test_the_committed_protocol_is_final_and_pins_the_pilot_detector():
    protocol = mrc._protocol()
    assert protocol["detector"]["sha256"] == "2588f09e544574e81685b00edd6bf6b9381fef930594f8f42884c66de8af1611"
    assert protocol["detector"]["baseline_D0_threshold"] == 0.0023743033104398573
    assert (protocol["cohort"]["targets"], protocol["cohort"]["seed_band"]) == (mrc.TARGETS, list(mrc.SEED_BAND))


def test_prepare_writes_frozen_shadow_jobs_with_the_pinned_detector(tmp_path, study):
    from master_script.core.queue import load_batch
    launch = mrc.prepare(tmp_path / "run", study.manifests, study.detector)
    assert launch["schema"] == "guard_collection_v4" and launch["study"] == mrc.STUDY
    assert [j["seed"] for j in launch["jobs"]] == list(range(7000, 7020))
    assert {j["role"] for j in launch["jobs"]} == {"matched_control_evaluation"}
    assert launch["detector"]["sha256"] == study.digest and len(launch["excluded_targets"]) == 30
    for job in launch["jobs"]:
        config, spec = load_batch([tmp_path / "run" / job["config"]]).pairs[0]
        guard = spec.pipeline.client_guard
        assert (guard.mode, guard.detector_sha256, guard.feature_schema) == ("shadow", study.digest, STRUCTURE_SCHEMA)
        assert spec.pipeline.rag is None and spec.pipeline.defense.mechanism == "none"
        assert (config.attack_variant, config.probe_epochs, config.causal_score, config.probe_lr) == (
            "causal_gradient_alignment", 12, "cosine", 0.005)
        assert (config.attack_trials, config.threshold_mode, config.request_interpolation) == (4, "fixed", 1.0)
        assert (config.matched_descent_requests, config.matched_honest_round) == (10, True)
    assert launch["protocol_sha256"] == mrc.digest(mrc.PROTOCOL)


def test_prepare_refuses_unsafe_cohorts_and_other_detectors(tmp_path, study):
    cohort, launch, stage_a, stage_b = study.manifests
    for manifests, message in (([cohort, launch, stage_a], "attacker_evaluation"),
                               ([launch, stage_a, stage_b], "final"),
                               ([cohort, stage_a, stage_b], "fewer targets")):
        with pytest.raises(ValueError, match=message):
            mrc.prepare(tmp_path / "x", manifests, study.detector)
    with pytest.raises(ValueError, match="band"):
        mrc.prepare(tmp_path / "y", study.manifests, study.detector, first_seed=7090)
    other, _ = fake_detector(tmp_path / "other.json", threshold=0.4)
    with pytest.raises(ValueError, match="pinned"):
        mrc.prepare(tmp_path / "z", study.manifests, other)
    mrc.prepare(tmp_path / "a", study.manifests, study.detector)
    with pytest.raises(FileExistsError):
        mrc.prepare(tmp_path / "a", study.manifests, study.detector)


def fake_target(config, default=""):
    return f"private target for seed {config.seed}"


def test_collector_resolves_the_launch_and_detector_fitting_fails_closed(tmp_path, study, monkeypatch):
    from master_script.core import datasets
    from master_script.tools import collect_guard_traces
    from master_script.tools.assemble_guard_dataset import assemble
    monkeypatch.setattr(datasets, "target_record_for", fake_target)
    mrc.prepare(tmp_path / "run", study.manifests, study.detector)
    collect_guard_traces.execute(tmp_path / "run" / "launch.json", resolve_only=True)
    cohort = json.loads((tmp_path / "run" / "cohort.json").read_text())
    assert sorted(set(cohort["groups"].values())) == ["matched_control_evaluation"]
    with pytest.raises(ValueError, match="declared data role"):
        assemble(cohort, tmp_path, STRUCTURE_SCHEMA)
    # A benign-control event is never labelled as legitimate or malicious training data.
    target = next(iter(cohort["groups"]))
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"implementation_fingerprint": "x", "config": {},
                                  "attack_trials": [{"target_seed": 7000, "target_sha256": target}],
                                  "guard_events": [{"scope": "benign_control:7000:M1:0", "features": {}}]}))
    manifest = {"schema": "guard_splits_v2", "sources": [{"path": "result.json", "sha256": mrc.digest(result)}],
                "groups": {target: "test"}, "held_out_variants": ["causal_gradient_alignment"]}
    with pytest.raises(ValueError, match="unknown request origin"):
        assemble(manifest, tmp_path, STRUCTURE_SCHEMA)


# ---------------------------------------------------------------- analysis

def event(scope, accept, request, *, accounting="observation", round_id=1, reference="ref", delta=1e-4):
    features = {name: 0. for name in STRUCTURE_NAMES}
    features["relative_delta"] = delta
    return {"scope": scope, "accounting_scope": accounting, "would_accept": accept, "decision": "accepted",
            "reason": "accepted" if accept else "detector_threshold", "request_sha256": request,
            "reference_sha256": reference, "round_id": round_id, "features": features,
            "detector_score": 0.1 if accept else 0.9}


def fabricate(root, *, causal=lambda i: True, m1=lambda i: 0, m2=lambda i: False, bad_reference=None):
    """Complete a prepared launch with synthetic results in the collector's layout."""
    launch = json.loads((root / "launch.json").read_text())
    targets = {str(j["seed"]): sha256(fake_target(SimpleNamespace(seed=j["seed"])).encode()).hexdigest()
               for j in launch["jobs"]}
    completed = []
    for index, job in enumerate(launch["jobs"]):
        seed = job["seed"]
        events = [event(f"training:{seed}:True", True, f"round-{r}", accounting="training", round_id=r)
                  for r in (1, 2, 3) for _ in range(4)]
        events += [event(f"observation:{seed}:world-{w}", not causal(index), "causal") for w in (0, 1, 0, 1)]
        controls = []
        for arm, count in (("M1", 10), ("M2", 1), ("M2raw", 1)):
            for k in range(count):
                scope = f"benign_control:{seed}:{arm}:{k}"
                flagged = k < m1(index) if arm == "M1" else m2(index) if arm == "M2" else False
                reference = "other" if bad_reference == index and arm == "M2" else "ref"
                events.append(event(scope, not flagged, f"{arm}-{k}", reference=reference))
                controls.append({"arm": arm, "index": k, "scope": scope, "delta_norm": 1.,
                                 "delta_norm_ratio_to_causal": 1. if arm != "M2raw" else 30.,
                                 "loss_before": 3., "loss_after": 2.9})
        result = {"status": "complete", "implementation_fingerprint": launch["implementation_fingerprint"],
                  "config": {"probe_epochs": 12, "causal_score": "cosine", "matched_descent_requests": 10,
                             "matched_honest_round": True},
                  "pipeline": {"client_guard": {"mode": "shadow", "detector_sha256": launch["detector"]["sha256"]}},
                  "attack_trials": [{"trial_id": i, "target_sha256": targets[str(seed)]} for i in range(4)],
                  "probe_training_loss": [2.0, 2.1], "guard_events": events,
                  "matched_controls": {"schema": "matched_controls_v1", "causal_delta_norm": 1.,
                                       "causal_candidate_loss_after": 2.5, "controls": controls}}
        path = root / "results" / f"job-{index:03d}" / "run" / "0000-x.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(result))
        completed.append({"job": index, "result": str(path.relative_to(root)), "sha256": mrc.digest(path)})
    (root / "progress.json").write_text(json.dumps({"completed": completed, "active_job": None, "targets": targets}))


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(mrc, "BOOTSTRAP", 2000)


def test_analysis_separates_only_without_any_matched_rejection(tmp_path, study, fast):
    mrc.prepare(tmp_path / "run", study.manifests, study.detector)
    fabricate(tmp_path / "run")
    report = mrc.analyze(tmp_path / "run", tmp_path / "analysis.json")
    assert report["decision"] == ["separates_at_this_resolution"]
    e = report["endpoints"]
    assert e["E1"]["count"] == 20 and e["E1"]["decision_interval"][0] == pytest.approx(0.832, abs=1e-3)
    # All-zero rejections: the bootstrap interval is a point, so the Clopper-Pearson bound applies.
    assert e["E2"]["bootstrap_ci95"] == [0., 0.] and e["E2"]["decision_interval"][1] == pytest.approx(0.168, abs=1e-3)
    s = report["secondary"]
    assert s["S1_training_rejection_per_event"]["events"] == 240
    assert s["S4_exploratory_within_target_auc_c_vs_m1"]["detector_score"]["pooled"] == 1.
    with pytest.raises(ValueError, match="exists"):
        mrc.analyze(tmp_path / "run", tmp_path / "analysis.json")


@pytest.mark.parametrize("scenario,expected", [
    (dict(m2=lambda i: i < 2), ["fails_benign_flagged"]),
    (dict(m1=lambda i: 3), ["fails_benign_flagged"]),
    (dict(causal=lambda i: i < 5), ["fails_causal_missed"]),
    (dict(causal=lambda i: i < 5, m1=lambda i: 3), ["fails_benign_flagged", "fails_causal_missed"]),
    (dict(m1=lambda i: int(i == 0)), ["inconclusive"]),
    (dict(causal=lambda i: i < 18), ["inconclusive"]),
])
def test_analysis_applies_the_preregistered_rule(tmp_path, study, fast, scenario, expected):
    mrc.prepare(tmp_path / "run", study.manifests, study.detector)
    fabricate(tmp_path / "run", **scenario)
    assert mrc.analyze(tmp_path / "run", tmp_path / "analysis.json")["decision"] == expected


def test_analysis_withholds_a_decision_when_matching_fails(tmp_path, study, fast):
    mrc.prepare(tmp_path / "run", study.manifests, study.detector)
    fabricate(tmp_path / "run", bad_reference=3)
    report = mrc.analyze(tmp_path / "run", tmp_path / "analysis.json")
    assert report["decision"] is None and len(report["invalid_targets"]) == 1
    assert "endpoints" not in report
    assert report["per_target"][3]["problems"] == ["reference differs across arms"]


def test_check_reports_integrity_without_any_detector_decision(tmp_path, study):
    mrc.prepare(tmp_path / "run", study.manifests, study.detector)
    fabricate(tmp_path / "run", causal=lambda i: i % 2 == 0, m1=lambda i: 2, bad_reference=1)
    state = json.loads((tmp_path / "run" / "progress.json").read_text())
    state["completed"] = state["completed"][:3]
    (tmp_path / "run" / "progress.json").write_text(json.dumps(state))
    report = mrc.check(tmp_path / "run")
    assert (report["completed"], report["planned"]) == (3, 20)
    assert [j["valid"] for j in report["jobs"]] == [True, False, True]
    text = json.dumps(report)
    for hidden in ("would_accept", "detector_score", "\"c\"", "\"m1\"", "reasons", "within_target_auc"):
        assert hidden not in text
    with pytest.raises(ValueError, match="not complete"):
        mrc.analyze(tmp_path / "run", tmp_path / "a.json")


def test_analysis_refuses_changed_or_incomplete_results(tmp_path, study):
    mrc.prepare(tmp_path / "run", study.manifests, study.detector)
    fabricate(tmp_path / "run")
    state = json.loads((tmp_path / "run" / "progress.json").read_text())
    path = tmp_path / "run" / state["completed"][0]["result"]
    result = json.loads(path.read_text()); result["config"]["probe_epochs"] = 80
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="changed"):
        mrc.analyze(tmp_path / "run", tmp_path / "a.json")
    state["completed"][0]["sha256"] = mrc.digest(path)
    (tmp_path / "run" / "progress.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="declared job"):
        mrc.analyze(tmp_path / "run", tmp_path / "a.json")
    state["completed"].pop()
    (tmp_path / "run" / "progress.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="not complete"):
        mrc.analyze(tmp_path / "run", tmp_path / "a.json")


def test_clopper_pearson_matches_known_bounds():
    for (k, n), expected in {(0, 20): (0., .1684), (2, 20): (.0123, .3170), (19, 20): (.7513, .9987),
                             (5, 20): (.0866, .4910)}.items():
        assert mrc.clopper_pearson(k, n) == pytest.approx(list(expected), abs=1e-4)
