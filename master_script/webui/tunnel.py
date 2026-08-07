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

PROVIDERS = ("ngrok", "cloudflare")
DEFAULT_PROVIDER = "cloudflare"
DEFAULT_PORT = 8080
_URL_RE = re.compile(r"https://[^\s\"']+")

# Where a saved config lives in the .env. Keeping it there rather than in a
# private file means the settings page can see and revoke these credentials
# alongside every other one the program holds.
ENV_KEYS = {
    "provider": "TUNNEL_PROVIDER",
    "api_key": "TUNNEL_API_KEY",
    "code": "TUNNEL_CODE",
    "port": "TUNNEL_PORT",
}


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


def load_saved() -> dict:
    """The last config saved from the Access page, in the clear.

    Server-side only -- see saved_payload() for what the browser is allowed to
    see. Anything unreadable falls back to the default rather than raising: a
    hand-edited .env should not be able to break the page that edits it.
    """
    from . import envfile

    pairs = envfile.read_pairs()
    provider = (pairs.get(ENV_KEYS["provider"]) or "").strip().lower()
    port = (pairs.get(ENV_KEYS["port"]) or "").strip()
    return {
        "provider": provider if provider in PROVIDERS else DEFAULT_PROVIDER,
        "api_key": pairs.get(ENV_KEYS["api_key"], ""),
        "code": pairs.get(ENV_KEYS["code"], ""),
        "port": int(port) if port.isdigit() else DEFAULT_PORT,
    }


def save_config(provider: str, api_key: str, code: str, port: int) -> None:
    """Persist a config so the next process start finds the form filled in.

    Takes loose values rather than a TunnelConfig: saving is also how a token
    gets cleared, and an empty api_key is not a startable config.
    """
    from . import envfile

    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; expected one of {PROVIDERS}")
    envfile.write({
        ENV_KEYS["provider"]: provider,
        ENV_KEYS["api_key"]: api_key,
        ENV_KEYS["code"]: code,
        ENV_KEYS["port"]: str(int(port)),
    })


def saved_payload() -> dict:
    """What the Access page prefills with. Credentials go out masked only.

    The page is reachable through the tunnel itself, so a saved token is never
    sent back in the clear -- the client shows the mask and posts null for any
    field it did not retype, meaning "keep what is on disk".
    """
    from . import envfile

    saved = load_saved()
    return {
        "provider": saved["provider"],
        "port": saved["port"],
        "api_key_set": bool(saved["api_key"]),
        "api_key_mask": envfile.mask(saved["api_key"]),
        "code_set": bool(saved["code"]),
        "code_mask": envfile.mask(saved["code"]),
    }


class TunnelManager:
    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._cfg: Optional[TunnelConfig] = None
        self.url: str = ""
        self.last_connected_unix: Optional[int] = None
        self.started_unix: Optional[int] = None
        self.error: str = ""

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
        self.error = ""
        env = {**os.environ}
        if cfg.provider == "ngrok":
            env["NGROK_AUTHTOKEN"] = cfg.api_key
        else:
            env["CLOUDFLARE_API_TOKEN"] = cfg.api_key
        self._proc = self._spawn(self._build_command(cfg), env)
        if getattr(self._proc, "stdout", None) is not None:
            threading.Thread(target=self._watch, daemon=True).start()
        self.last_connected_unix = int(time.time())
        self.started_unix = self.last_connected_unix

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
        self.started_unix = None

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
            "started_unix": self.started_unix if connected else None,
            "code_set": bool(self._cfg.code) if self._cfg else False,
            "error": self.error,
        }


MANAGER = TunnelManager()
