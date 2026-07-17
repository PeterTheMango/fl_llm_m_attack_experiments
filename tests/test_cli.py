import pytest

from master_script.perform_experiments import build_parser, main


def test_list_attacks_prints_all_eleven(capsys):
    assert main(["--list-attacks"]) == 0
    out = capsys.readouterr().out
    for name in ("zlib", "min_k", "min_k_plus_plus", "neighborhood", "recall",
                 "reference", "samia", "spv_mia", "wbc", "amia", "loss"):
        assert name in out


def test_dry_run_prints_run_ids_and_runs_nothing(tmp_path, capsys, monkeypatch):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("attacks:\n  zlib:\n    sweep:\n      seed: [7, 11]\n")

    import master_script.perform_experiments as cli

    monkeypatch.setattr(
        cli, "_run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    assert main(["--config", str(cfg), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "2 run(s)" in out


def test_attack_flag_filters_subset(tmp_path, capsys):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("attacks:\n  zlib: {}\n  min_k: {}\n")
    main(["--config", str(cfg), "--dry-run", "--attack", "zlib"])
    out = capsys.readouterr().out
    assert "zlib" in out and "min_k" not in out


def test_unknown_config_key_exits_nonzero_before_compute(tmp_path, capsys):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("attacks:\n  zlib:\n    base: {epsilon: 8}\n")
    assert main(["--config", str(cfg), "--dry-run"]) == 2
    assert "epsilon" in capsys.readouterr().err


def test_max_parallel_rejects_more_than_two():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--max-parallel", "3"])


def test_defaults_to_smoke_config():
    args = build_parser().parse_args([])
    assert args.config.name == "smoke.yaml"
    assert args.max_parallel == 1


def test_parallel_yaml_roundtrip_preserves_run_id(tmp_path):
    """--max-parallel 2 ships each config to a subprocess as YAML.

    wbc.window_sizes is a tuple that returns from YAML as a list. json.dumps
    renders both as [2,3], so the hash survives -- but this is load-bearing and
    silent, so pin it. If this breaks, parallel runs recompute everything.
    """
    from dataclasses import asdict

    import yaml

    from master_script.core.config import experiment_key
    from master_script.core.registry import ATTACKS
    from master_script.core.yaml_config import load_config_file

    for name in ("wbc", "zlib"):
        spec = ATTACKS[name]
        cfg = spec.config_cls()
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump({"attacks": {name: {"base": asdict(cfg)}}}))
        (reloaded, reloaded_spec), = load_config_file(path)
        assert experiment_key(reloaded, reloaded_spec) == experiment_key(cfg, spec), name
