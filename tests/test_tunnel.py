import pytest

from master_script.webui import envfile, tunnel
from master_script.webui.tunnel import TunnelConfig, TunnelManager


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    monkeypatch.setattr(envfile, "env_path", lambda: path)
    return path


def test_status_disconnected_before_start():
    assert TunnelManager().status["connected"] is False


def test_ngrok_url_without_reserved_domain_is_ephemeral():
    cfg = TunnelConfig(provider="ngrok", api_key="tok", code="", port=8080)
    assert cfg.is_ephemeral is True


def test_ngrok_url_with_reserved_domain_is_stable():
    cfg = TunnelConfig(provider="ngrok", api_key="tok", code="my.domain", port=8080)
    assert cfg.is_ephemeral is False


def test_cloudflare_named_tunnel_is_stable():
    cfg = TunnelConfig(provider="cloudflare", api_key="k", code="tunnel-token", port=8080)
    assert cfg.is_ephemeral is False


def test_unknown_provider_rejected():
    with pytest.raises(ValueError, match="provider"):
        TunnelConfig(provider="dropbox", api_key="k", code="", port=8080)


def test_missing_api_key_rejected():
    with pytest.raises(ValueError, match="api_key"):
        TunnelConfig(provider="ngrok", api_key="", code="", port=8080)


def test_start_builds_expected_ngrok_command(monkeypatch):
    m = TunnelManager()
    captured = {}
    monkeypatch.setattr(m, "_spawn", lambda cmd, env: captured.setdefault("cmd", cmd))
    m.start(TunnelConfig(provider="ngrok", api_key="tok", code="", port=8080))
    assert "ngrok" in captured["cmd"][0]
    assert "8080" in " ".join(captured["cmd"])


def test_start_builds_expected_cloudflared_command(monkeypatch):
    m = TunnelManager()
    captured = {}
    monkeypatch.setattr(m, "_spawn", lambda cmd, env: captured.setdefault("cmd", cmd))
    m.start(TunnelConfig(provider="cloudflare", api_key="k", code="tok", port=8080))
    assert "cloudflared" in captured["cmd"][0]


def test_stop_marks_disconnected(monkeypatch):
    m = TunnelManager()
    monkeypatch.setattr(m, "_spawn", lambda cmd, env: None)
    m.start(TunnelConfig(provider="ngrok", api_key="tok", code="", port=8080))
    m.stop()
    assert m.status["connected"] is False


# ---------- persistence ----------

def test_saved_config_round_trips_through_the_env_file(env_file):
    tunnel.save_config("ngrok", "tok-123", "my-lab-monitor", 9100)
    assert tunnel.load_saved() == {
        "provider": "ngrok", "api_key": "tok-123",
        "code": "my-lab-monitor", "port": 9100,
    }


def test_load_saved_falls_back_when_nothing_is_saved(env_file):
    assert tunnel.load_saved() == {
        "provider": tunnel.DEFAULT_PROVIDER, "api_key": "",
        "code": "", "port": tunnel.DEFAULT_PORT,
    }


def test_load_saved_ignores_a_hand_edited_nonsense_provider_or_port(env_file):
    env_file.write_text("TUNNEL_PROVIDER=dropbox\nTUNNEL_PORT=not-a-port\n")
    saved = tunnel.load_saved()
    assert saved["provider"] == tunnel.DEFAULT_PROVIDER
    assert saved["port"] == tunnel.DEFAULT_PORT


def test_saved_payload_masks_the_credentials_it_reports(env_file):
    tunnel.save_config("cloudflare", "cf-token-abcdefgh", "connector-token-xyz", 8080)
    payload = tunnel.saved_payload()
    assert "cf-token-abcdefgh" not in str(payload)
    assert "connector-token-xyz" not in str(payload)
    assert payload["api_key_set"] is True and payload["code_set"] is True
    assert payload["api_key_mask"].endswith("efgh")


def test_saving_keeps_unrelated_keys_in_the_env_file(env_file):
    env_file.write_text("FIREBASE_PROJECT_ID=canary\n")
    tunnel.save_config("cloudflare", "k", "", 8080)
    assert "FIREBASE_PROJECT_ID=canary" in env_file.read_text()
