import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
from master_script.core import runner, gpu
from master_script.core.registry import ATTACKS


def test_completed_trials_survive_later_interrupt_and_attempts_are_isolated(monkeypatch, tmp_path):
    spec = ATTACKS['reference']
    cfg = spec.config_cls(attack_trials=4)
    def run(config, spec, trial_id, truth_member):
        if trial_id == 1:
            raise KeyboardInterrupt()
        return {'trial_id': trial_id, 'score': -.7, 'truth_member': truth_member}
    monkeypatch.setattr(runner, 'run_attack_trial', run)
    for _ in range(2):
        with pytest.raises(KeyboardInterrupt):
            runner.run_attack_trials(cfg, spec, trial_directory=tmp_path)
    attempts = list(tmp_path.iterdir())
    assert len(attempts) == 2
    for attempt in attempts:
        progress = json.loads((attempt / 'progress.json').read_text())
        assert progress['status'] == 'partial'
        assert progress['completed_trials'] == 1
        saved = json.loads((attempt / 'trial-000000.json').read_text())
        assert saved['trial']['score'] == -.7
        assert saved['status'] == 'partial'
        assert not (attempt / 'trial-000001.json').exists()


def test_successful_run_saves_trials_before_final_result(tmp_path):
    spec = ATTACKS['reference']
    cfg = spec.config_cls(attack_trials=2)
    result = runner.run_single_experiment(cfg, spec, use_firestore=False, artifact_directory=tmp_path)
    assert result['status'] == 'complete'
    progress = json.loads(next(tmp_path.rglob('progress.json')).read_text())
    assert progress['completed_trials'] == 2
    assert len(list(tmp_path.rglob('trial-*.json'))) == 2
    assert (tmp_path / 'result.json').exists()


@pytest.mark.parametrize('available,requested,expected', [(False, 0, 'cpu'), (True, 1, 'cuda')])
def test_explicit_device_selection(monkeypatch, available, requested, expected):
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    assert gpu.training_device(SimpleNamespace(sim_num_gpus=requested)) == expected


def test_gpu_request_never_silently_falls_back_to_cpu(monkeypatch):
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                            version=SimpleNamespace(cuda=None), __version__='cpu-only')
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setenv('EXPERIMENT_GPU', 'cpu')
    with pytest.raises(RuntimeError, match='Refusing to silently fall back') as error:
        gpu.training_device(SimpleNamespace(sim_num_gpus=1))
    assert 'cpu-only' in str(error.value)
    assert "EXPERIMENT_GPU='cpu'" in str(error.value)


def test_dotenv_can_explicitly_force_cpu(monkeypatch):
    monkeypatch.delenv('EXPERIMENT_GPU', raising=False)
    monkeypatch.setattr(gpu, '_read_env_file_var', lambda name: 'cpu')
    monkeypatch.setattr(gpu, '_gpu_free_memory', lambda: [('0', 20000)])
    assert gpu.apply_gpu_selection() == 'cpu'
