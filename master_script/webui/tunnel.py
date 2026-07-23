"""Outbound tunnel for external viewing (spec §4).

The VM has outbound internet but no inbound route; a tunnel agent dials out and
the provider proxies a public URL back down that connection. Exposure is always
explicit and user-initiated (§4.4) -- never auto-start.
"""
from dataclasses import dataclass
from typing import List, Optional
import re
import subprocess
import threading
import time

from nicegui import ui

PROVIDERS = ("ngrok", "cloudflare")
_URL_RE = re.compile(r"https://[^\s\"']+")


@dataclass
class TunnelConfig:
    provider: str
    api_key: str
    code: str
    port: int

    def __post_init__(self):
        if self.provider not in PROVIDERS:
            raise ValueError(f"unknown provider {self.provider!r}; expected one of {PROVIDERS}")
        if not self.api_key:
            raise ValueError("api_key is required")

    @property
    def is_ephemeral(self) -> bool:
        """ngrok without a reserved domain regenerates its URL each session."""
        return self.provider == "ngrok" and not self.code


class TunnelManager:
    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._cfg: Optional[TunnelConfig] = None
        self.url: str = ""
        self.last_connected_unix: Optional[int] = None

    def _build_command(self, cfg: TunnelConfig) -> List[str]:
        if cfg.provider == "ngrok":
            cmd = ["ngrok", "http", str(cfg.port), "--log", "stdout"]
            if cfg.code:
                cmd += ["--domain", cfg.code]
            return cmd
        cmd = ["cloudflared", "tunnel", "--no-autoupdate"]
        if cfg.code:
            cmd += ["run", "--token", cfg.code]
        else:
            cmd += ["--url", f"http://localhost:{cfg.port}"]
        return cmd

    def _spawn(self, cmd: List[str], env: dict):
        return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)

    def start(self, cfg: TunnelConfig) -> None:
        import os

        self.stop()
        self._cfg = cfg
        env = {**os.environ}
        if cfg.provider == "ngrok":
            env["NGROK_AUTHTOKEN"] = cfg.api_key
        else:
            env["CLOUDFLARE_API_TOKEN"] = cfg.api_key
        self._proc = self._spawn(self._build_command(cfg), env)
        if self._proc is not None:
            threading.Thread(target=self._watch, daemon=True).start()
        self.last_connected_unix = int(time.time())

    def _watch(self) -> None:
        """Scrape the agent's stdout for the public URL."""
        for line in self._proc.stdout:
            match = _URL_RE.search(line)
            if match and not self.url:
                self.url = match.group(0)

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            self._proc = None
        self.url = ""

    @property
    def status(self) -> dict:
        connected = self._proc is not None and self._proc.poll() is None
        return {
            "connected": connected,
            "url": self.url,
            "provider": self._cfg.provider if self._cfg else None,
            "port": self._cfg.port if self._cfg else None,
            "ephemeral": self._cfg.is_ephemeral if self._cfg else None,
            "last_connected_unix": self.last_connected_unix,
        }


MANAGER = TunnelManager()


def render() -> None:
    with ui.card().classes("w-full"):
        ui.label("External access tunnel").classes("text-lg font-bold")
        provider = ui.select(list(PROVIDERS), value="ngrok", label="Provider")
        api_key = ui.input("API key / authtoken", password=True)
        code = ui.input("Tunnel code / reserved domain (optional)")
        port = ui.number("Target port", value=8080, format="%d")
        banner = ui.label("").classes("text-sm")

        @ui.refreshable
        def status_panel():
            s = MANAGER.status
            if not s["connected"]:
                ui.label("Tunnel down — dashboard reachable on the VM/VPN only.")
                return
            ui.label("EXTERNAL ACCESS IS LIVE").classes("text-red-600 font-bold")
            ui.label(f"{s['provider']} -> localhost:{s['port']}")
            if s["url"]:
                ui.link(s["url"], s["url"], new_tab=True)
                ui.button("Copy", on_click=lambda: ui.clipboard.write(s["url"]))
            ui.label("Ephemeral URL — regenerated each session." if s["ephemeral"]
                     else "Stable URL — reserved domain / named tunnel.")

        def _start():
            try:
                MANAGER.start(TunnelConfig(provider.value, api_key.value, code.value, int(port.value)))
                banner.set_text("")
            except (ValueError, FileNotFoundError) as exc:
                banner.set_text(f"Could not start tunnel: {exc}")
            status_panel.refresh()

        def _stop():
            MANAGER.stop()
            status_panel.refresh()

        with ui.row():
            ui.button("Start tunnel", on_click=_start)
            ui.button("Stop tunnel", on_click=_stop)
        status_panel()
