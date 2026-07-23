import pytest

from master_script.webui.tunnel import TunnelConfig, TunnelManager


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
