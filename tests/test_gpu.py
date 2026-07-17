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
