# tests/test_webui_pages.py
"""Wiring tests for the CANARY Monitor API. Data logic is in test_webui_state.py."""
import os
import time

from fastapi.testclient import TestClient

from master_script.webui import api, app as app_module, catalog, monitor, results
from master_script.webui.state import DashboardState

DOCS = [
    {"run_id": "a", "status": "complete", "updated_at_unix": 100,
     "config": {"attack_name": "zlib", "federated_rounds": 1, "seed": 7,
                "model_id": "distilbert/distilgpt2", "ldp_mechanism": "none", "epsilon": None},
     "metrics": {"adv": 0.75, "tpr": 0.8, "tnr": 0.7, "num_trials": 2},
     "attack_trials": [
         {"trial_id": 0, "truth_member": True, "score": 0.9, "pred_member": True},
         {"trial_id": 1, "truth_member": False, "score": 0.1, "pred_member": False},
     ],
     "federated_history": [{"trial_id": 0, "rounds": [{"round": 1, "mean_client_loss": 4.2}]}],
     "artifacts": {"artifact_dir": "artifacts/zlib/a", "federated_model_path": None}},
    {"run_id": "c", "status": "failed", "updated_at_unix": 300,
     "config": {"attack_name": "min_k", "federated_rounds": 1, "seed": 7},
     "error": "RuntimeError: CUDA out of memory"},
]


def _client(docs=DOCS, host="127.0.0.1"):
    """TestClient over the real app with the shared state pre-seeded.

    ensure_fresh() would otherwise reach for Firestore; stamping last_sync_unix
    keeps these wiring tests offline. The default client address is loopback,
    which is what the settings routes require -- pass a routable host to stand
    in for a viewer arriving over the tunnel.
    """
    api.STATE.runs.clear()
    api.STATE.running = []
    api.STATE.manifest = None
    api.STATE.ingest(docs)
    api.STATE.last_sync_unix = time.time()
    return TestClient(app_module.app, client=(host, 45678))


def test_spa_routes_all_serve_the_same_shell():
    client = _client()
    for path in ["/", "/results", "/results/a", "/access", "/launch", "/settings"]:
        response = client.get(path)
        assert response.status_code == 200
        assert "CANARY Monitor" in response.text


def test_settings_page_and_api_are_refused_to_a_non_loopback_caller():
    """The one surface that reads credentials must not follow the tunnel out."""
    remote = _client(host="203.0.113.7")
    assert remote.get("/settings").status_code == 403
    assert remote.get("/api/settings").status_code == 403
    assert remote.get("/api/settings/reveal/FIREBASE_PROJECT_ID").status_code == 403
    assert remote.post("/api/settings", json={"updates": {"X": "1"}}).status_code == 403
    # Everything else stays reachable: only settings are local-only.
    assert remote.get("/").status_code == 200
    assert remote.get("/access").status_code == 200
    assert remote.get("/api/live").status_code == 200


def test_a_proxied_request_is_not_local_even_though_its_socket_is():
    """The tunnel agent dials the dashboard over loopback; the viewer is remote."""
    client = _client()
    assert client.get("/settings", headers={"X-Forwarded-For": "203.0.113.7"}).status_code == 403
    assert client.get("/api/settings", headers={"X-Forwarded-For": "203.0.113.7"}).status_code == 403


def test_loopback_recognises_every_form_of_the_local_address():
    from master_script.webui import localguard

    for host in ["127.0.0.1", "127.0.0.53", "::1", "::ffff:127.0.0.1"]:
        assert localguard.is_loopback(host) is True, host
    for host in ["", "192.168.1.4", "10.0.0.2", "203.0.113.7", "::ffff:10.0.0.2", "localhost"]:
        assert localguard.is_loopback(host) is False, host


def test_session_endpoint_tells_the_client_whether_settings_are_available():
    assert _client().get("/api/session").json()["local"] is True
    assert _client(host="203.0.113.7").get("/api/session").json()["local"] is False


def test_static_assets_are_served_locally():
    """Chart.js is vendored: the dashboard must render without a CDN."""
    client = _client()
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/chart.umd.min.js").status_code == 200


def test_live_log_tail_ships_follow_and_manual_scroll_policies():
    """A rerender follows the tail only while the viewer is already at bottom."""
    javascript = _client().get("/static/app.js").text
    assert "LOG_BOTTOM_EPSILON_PX = 2" in javascript
    assert "const logScrollTop = logPane ? logPane.scrollTop : null" in javascript
    assert "pane.scrollTop = pane.scrollHeight" in javascript
    assert "pane.scrollTop = logScrollTop" in javascript


def test_live_endpoint_reports_running_set_unavailable_without_a_manifest():
    """§2.4: say so rather than guessing."""
    payload = _client().get("/api/live").json()
    assert payload["run_state_available"] is False
    assert payload["running"] == []


def test_live_endpoint_groups_sweep_progress_by_attack():
    sweeps = {s["attack"]: s for s in _client().get("/api/live").json()["sweeps"]}
    assert sweeps["zlib"]["complete"] == 1
    assert sweeps["min_k"]["failed"] == 1
    assert sweeps["zlib"]["total"] is None  # no manifest -> no denominator


def test_results_endpoint_returns_every_run_and_the_attack_catalog():
    payload = _client().get("/api/results").json()
    assert {r["run_id"] for r in payload["runs"]} == {"a", "c"}
    assert "zlib" in payload["attacks"]


def test_detail_endpoint_returns_trials_and_normalized_history():
    payload = _client().get("/api/runs/a").json()
    assert payload["attack"] == "zlib"
    assert len(payload["trials"]) == 2
    assert payload["federated_history"] == [{"round": 1, "mean_loss": 4.2}]


def test_detail_endpoint_404s_for_an_unknown_run():
    assert _client().get("/api/runs/nope").status_code == 404


def test_detail_lists_the_firestore_doc_as_authoritative():
    artifacts = _client().get("/api/runs/a").json()["artifacts"]
    assert any(a["k"] == "firestore_doc" and a.get("authoritative") for a in artifacts)


def test_tunnel_endpoint_reports_disconnected_before_any_start():
    assert _client().get("/api/tunnel").json()["connected"] is False


def _env(tmp_path, monkeypatch):
    from master_script.webui import envfile

    path = tmp_path / ".env"
    monkeypatch.setattr(envfile, "env_path", lambda: path)
    return path


def test_settings_endpoint_lists_the_complete_catalog_without_an_env_file(tmp_path, monkeypatch):
    from master_script.webui import envfile

    _env(tmp_path, monkeypatch)
    entries = _client().get("/api/settings").json()["entries"]

    assert [entry["key"] for entry in entries] == list(envfile.KNOWN_KEYS)
    assert all(entry["known"] for entry in entries)
    assert all(entry["present"] is False for entry in entries)


def test_tunnel_credentials_persist_to_the_env_file(tmp_path, monkeypatch):
    path = _env(tmp_path, monkeypatch)
    body = _client().post("/api/tunnel/save", json={
        "provider": "ngrok", "api_key": "tok-123", "code": "my-lab", "port": 9100,
    }).json()
    assert body["ok"] is True
    text = path.read_text()
    assert "TUNNEL_API_KEY=tok-123" in text and "TUNNEL_PORT=9100" in text
    # A later read prefills the form from disk without echoing the token back.
    saved = _client().get("/api/tunnel").json()["saved"]
    assert saved["provider"] == "ngrok" and saved["port"] == 9100
    assert saved["api_key_set"] is True
    assert "tok-123" not in str(saved)


def test_an_untouched_credential_field_keeps_the_saved_value(tmp_path, monkeypatch):
    """null means "leave it alone" -- the browser never had the token to resend."""
    path = _env(tmp_path, monkeypatch)
    client = _client()
    client.post("/api/tunnel/save", json={"provider": "ngrok", "api_key": "tok-123",
                                          "code": "my-lab", "port": 9100})
    client.post("/api/tunnel/save", json={"provider": "ngrok", "port": 9200})
    assert "TUNNEL_API_KEY=tok-123" in path.read_text()
    assert "TUNNEL_PORT=9200" in path.read_text()


def test_an_emptied_credential_field_clears_the_saved_value(tmp_path, monkeypatch):
    path = _env(tmp_path, monkeypatch)
    client = _client()
    client.post("/api/tunnel/save", json={"provider": "ngrok", "api_key": "tok-123",
                                          "code": "my-lab", "port": 9100})
    client.post("/api/tunnel/save", json={"provider": "ngrok", "api_key": ""})
    assert "tok-123" not in path.read_text()
    assert _client().get("/api/tunnel").json()["saved"]["api_key_set"] is False


def test_starting_a_tunnel_saves_the_config_even_if_the_agent_is_missing(tmp_path, monkeypatch):
    path = _env(tmp_path, monkeypatch)
    monkeypatch.setattr(api.tunnel.MANAGER, "_spawn",
                        lambda cmd, env: (_ for _ in ()).throw(FileNotFoundError("ngrok")))
    body = _client().post("/api/tunnel/start", json={
        "provider": "ngrok", "api_key": "tok-123", "code": "", "port": 8080,
    }).json()
    assert body["ok"] is False
    assert "TUNNEL_API_KEY=tok-123" in path.read_text()


def test_starting_a_tunnel_reuses_the_saved_credentials(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    captured = {}
    monkeypatch.setattr(api.tunnel.MANAGER, "_spawn",
                        lambda cmd, env: captured.update(cmd=cmd, env=env))
    client = _client()
    client.post("/api/tunnel/save", json={"provider": "ngrok", "api_key": "tok-123",
                                          "code": "my-lab", "port": 9100})
    body = client.post("/api/tunnel/start", json={}).json()
    api.tunnel.MANAGER.stop()
    assert body["ok"] is True
    assert captured["env"]["NGROK_AUTHTOKEN"] == "tok-123"
    assert "my-lab" in " ".join(captured["cmd"])


def test_every_registered_attack_has_a_label_and_colour():
    from master_script.core.registry import ATTACKS

    entries = catalog.catalog()
    assert set(entries) == set(ATTACKS)
    assert all(e["label"] and e["color"].startswith("#") for e in entries.values())


def test_privacy_direction_label_reads_lower_adv_as_improved():
    """§3.4: lower Adv within an environment => privacy improved."""
    assert "improved" in results.privacy_direction(baseline_adv=0.9, current_adv=0.6).lower()
    assert "declined" in results.privacy_direction(baseline_adv=0.6, current_adv=0.9).lower()


def test_privacy_direction_reports_unchanged_within_tolerance():
    assert "unchanged" in results.privacy_direction(0.7, 0.7).lower()


def test_monitor_reports_unavailable_running_set_without_manifest():
    """§2.4: say so rather than guessing."""
    s = DashboardState()
    s.ingest([{"run_id": "a", "status": "complete", "config": {}, "metrics": {"adv": 1.0}}])
    assert monitor.running_set_available(s) is False


def test_running_row_marks_an_unreported_stage_rather_than_guessing():
    s = DashboardState()
    s.ingest([{"run_id": "monitor_state", "manifest": [],
               "running": [{"run_id": "z", "attack": "zlib", "started_unix": 10}]}])
    row = monitor.live_payload(s)["running"][0]
    assert row["stage_index"] == -1
    assert row["stage_label"] == "stage unreported"


def test_settings_endpoint_never_returns_a_secret_in_the_clear(tmp_path, monkeypatch):
    from master_script.webui import envfile

    path = tmp_path / ".env"
    path.write_text("FIREBASE_SERVICE_ACCOUNT_JSON=super-secret-value-1234\n")
    monkeypatch.setattr(envfile, "env_path", lambda: path)

    body = _client().get("/api/settings").json()
    assert "super-secret-value-1234" not in str(body)
    entry = next(e for e in body["entries"] if e["key"] == "FIREBASE_SERVICE_ACCOUNT_JSON")
    assert entry["masked"] is True


def test_reveal_is_a_separate_explicit_request(tmp_path, monkeypatch):
    from master_script.webui import envfile

    path = tmp_path / ".env"
    path.write_text("FIREBASE_SERVICE_ACCOUNT_JSON=super-secret-value-1234\n")
    monkeypatch.setattr(envfile, "env_path", lambda: path)

    client = _client()
    assert client.get("/api/settings/reveal/FIREBASE_SERVICE_ACCOUNT_JSON").json()["value"] \
        == "super-secret-value-1234"
    assert client.get("/api/settings/reveal/NOT_SET").status_code == 404


def test_saving_settings_reloads_them_into_the_running_process(tmp_path, monkeypatch):
    from master_script.webui import envfile

    path = tmp_path / ".env"
    path.write_text("FIREBASE_PROJECT_ID=old\n")
    monkeypatch.setattr(envfile, "env_path", lambda: path)
    monkeypatch.delenv("FIREBASE_PROJECT_ID", raising=False)

    body = _client().post("/api/settings", json={"updates": {"FIREBASE_PROJECT_ID": "new"}}).json()
    assert body["ok"] is True
    assert "FIREBASE_PROJECT_ID" in body["changed"]
    assert os.environ["FIREBASE_PROJECT_ID"] == "new"


def test_settings_endpoint_rejects_a_malformed_key(tmp_path, monkeypatch):
    from master_script.webui import envfile

    monkeypatch.setattr(envfile, "env_path", lambda: tmp_path / ".env")
    body = _client().post("/api/settings", json={"updates": {"BAD=KEY": "x"}}).json()
    assert body["ok"] is False
    assert not (tmp_path / ".env").exists()


def test_config_validate_endpoint_writes_nothing_and_counts_runs(tmp_path, monkeypatch):
    from master_script.webui import configs

    monkeypatch.setattr(configs, "CONFIGS_DIR", tmp_path)
    body = _client().post(
        "/api/configs/validate",
        json={"text": "attacks:\n  zlib:\n    sweep:\n      seed: [7, 11]\n"},
    ).json()
    assert body["ok"] is True and body["runs"] == 2
    assert list(tmp_path.iterdir()) == []


def test_config_save_endpoint_refuses_a_name_outside_the_configs_dir(tmp_path, monkeypatch):
    from master_script.webui import configs

    monkeypatch.setattr(configs, "CONFIGS_DIR", tmp_path)
    body = _client().post(
        "/api/configs/save",
        json={"name": "../escaped", "text": "attacks:\n  zlib: {}\n"},
    ).json()
    assert body["ok"] is False
    assert not (tmp_path.parent / "escaped.yaml").exists()


# ---------- run-state staleness (the hard-kill fix) ----------

def _state_with_running(**overrides):
    entry = {"run_id": "z", "attack": "zlib", "stage": "attack",
             "started_unix": time.time(), "heartbeat_unix": time.time()}
    entry.update(overrides)
    s = DashboardState()
    s.ingest([
        {"run_id": "monitor_state", "manifest": [{"run_id": "z", "attack": "zlib"}],
         "running": [entry]},
        {"run_id": "done", "status": "complete", "config": {"attack_name": "zlib"},
         "metrics": {"adv": 0.6}},
    ])
    return s


def test_a_beating_run_is_reported_live():
    payload = monitor.live_payload(_state_with_running())
    assert payload["running"][0]["stale"] is False
    assert payload["running_live"] == 1 and payload["running_stale"] == 0


def test_a_run_whose_heartbeat_stopped_is_reported_stale():
    """A hard kill never runs on_run_end, so the reader has to age the entry out."""
    dead = time.time() - (monitor.STALE_AFTER_SECONDS + 60)
    payload = monitor.live_payload(_state_with_running(heartbeat_unix=dead))
    row = payload["running"][0]
    assert row["stale"] is True
    assert row["heartbeat_age_seconds"] > monitor.STALE_AFTER_SECONDS
    assert payload["running_live"] == 0 and payload["running_stale"] == 1


def test_a_long_run_that_keeps_beating_is_not_called_stale():
    """Elapsed time must not be mistaken for staleness -- runs take hours."""
    payload = monitor.live_payload(
        _state_with_running(started_unix=time.time() - 6 * 3600, heartbeat_unix=time.time() - 5)
    )
    assert payload["running"][0]["stale"] is False


def test_a_stale_entry_does_not_occupy_a_slot_in_the_sweep_bars():
    dead = time.time() - (monitor.STALE_AFTER_SECONDS + 60)
    sweeps = {s["attack"]: s for s in monitor.live_payload(_state_with_running(heartbeat_unix=dead))["sweeps"]}
    assert sweeps["zlib"]["running"] == 0
    # ...and the ghost must not be double-counted against the manifest total
    assert sweeps["zlib"]["pending"] == 0


def test_an_entry_with_no_timestamp_at_all_is_stale():
    """Nothing vouches for it, so it must not read as a live run."""
    s = DashboardState()
    s.ingest([{"run_id": "monitor_state", "manifest": [],
               "running": [{"run_id": "z", "attack": "zlib"}]}])
    assert monitor.live_payload(s)["running"][0]["stale"] is True


def test_live_endpoint_exposes_the_staleness_threshold():
    payload = _client().get("/api/live").json()
    assert payload["stale_after_seconds"] == monitor.STALE_AFTER_SECONDS


# ---------- stale error on a recovered run ----------

_RECOVERED = [{
    "run_id": "r", "status": "complete", "updated_at_unix": 100,
    "config": {"attack_name": "zlib"}, "metrics": {"adv": 0.6, "tpr": 0.6, "tnr": 0.6},
    # left behind by an earlier failed attempt, before save_result cleared it
    "error": "RuntimeError: CUDA out of memory",
}]


def test_a_completed_run_is_not_reported_as_errored():
    """Legacy documents still carry the old error; it must not read as failure."""
    payload = _client(_RECOVERED).get("/api/runs/r").json()
    assert payload["status"] == "complete"
    assert payload["error"] is None


def test_the_earlier_failure_is_still_visible_as_history():
    payload = _client(_RECOVERED).get("/api/runs/r").json()
    assert "CUDA out of memory" in payload["prior_error"]


def test_a_genuinely_failed_run_still_reports_its_error():
    payload = _client().get("/api/runs/c").json()
    assert payload["status"] == "failed"
    assert "CUDA out of memory" in payload["error"]
    assert payload["prior_error"] is None


def test_the_results_grid_does_not_mark_a_recovered_run_as_errored():
    row = next(r for r in _client(_RECOVERED).get("/api/results").json()["runs"] if r["run_id"] == "r")
    assert row["error"] is None
