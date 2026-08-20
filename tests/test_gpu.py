from master_script.core import gpu


def test_env_var_forces_selection(monkeypatch):
    monkeypatch.setenv("EXPERIMENT_GPU", "1")
    assert gpu.select_gpu() == "1"


def test_cpu_forced_sets_empty_visible_devices(monkeypatch):
    monkeypatch.setenv("EXPERIMENT_GPU", "cpu")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert gpu.apply_gpu_selection() == "cpu"
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_no_nvidia_smi_returns_none(monkeypatch):
    monkeypatch.delenv("EXPERIMENT_GPU", raising=False)
    monkeypatch.setattr(gpu.shutil, "which", lambda _: None)
    monkeypatch.setattr(gpu, "_read_env_file_var", lambda _: None)
    assert gpu.select_gpu() is None


def test_auto_selects_gpu_with_most_free_memory(monkeypatch):
    monkeypatch.delenv("EXPERIMENT_GPU", raising=False)
    monkeypatch.setattr(gpu, "_read_env_file_var", lambda _: None)
    monkeypatch.setattr(gpu, "_gpu_free_memory", lambda: [("0", 1000), ("1", 8000)])
    assert gpu.select_gpu() == "1"


# ---------- webui/gpustats.py: nvidia-smi field availability ----------

def _fake_smi(monkeypatch, output: str):
    from master_script.webui import gpustats

    monkeypatch.setattr(gpustats.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(gpustats.subprocess, "check_output", lambda *a, **k: output)
    return gpustats


def test_gpustats_keeps_mig_gpu_whose_util_and_temp_are_na(monkeypatch):
    """The bug: '[N/A]' util/temp dropped the row, so a real GPU read as absent."""
    gpustats = _fake_smi(monkeypatch, "0, GRID A100D-1-20C, [N/A], 1, 20480, [N/A]\n")
    gpus = gpustats.poll()
    assert len(gpus) == 1
    gpu = gpus[0]
    assert gpu["id"] == 0 and gpu["name"] == "GRID A100D-1-20C"
    assert gpu["util"] is None and gpu["temp_c"] is None
    assert gpu["mem_total_gib"] == 20480 / 1024.0


def test_gpustats_unknown_util_is_none_not_zero(monkeypatch):
    """0% and 'not reported' are different claims; the UI must be able to tell."""
    gpustats = _fake_smi(monkeypatch, "0, GRID A100D-1-20C, [N/A], 1, 20480, [N/A]\n")
    util = gpustats.poll()[0]["util"]
    assert util is None
    assert util != 0


def test_gpustats_still_parses_a_fully_reporting_gpu(monkeypatch):
    gpustats = _fake_smi(monkeypatch, "0, NVIDIA A100-SXM4-40GB, 73, 8192, 40960, 61\n")
    gpu = gpustats.poll()[0]
    assert gpu["util"] == 73.0 and gpu["temp_c"] == 61.0
    assert gpu["mem_used_gib"] == 8.0


def test_gpustats_drops_only_rows_with_no_index(monkeypatch):
    gpustats = _fake_smi(
        monkeypatch,
        "[N/A], broken, [N/A], [N/A], [N/A], [N/A]\n0, GRID A100D-1-20C, [N/A], 1, 20480, [N/A]\n",
    )
    assert [g["id"] for g in gpustats.poll()] == [0]


def test_gpustats_empty_without_nvidia_smi(monkeypatch):
    from master_script.webui import gpustats

    monkeypatch.setattr(gpustats.shutil, "which", lambda _: None)
    assert gpustats.poll() == []
