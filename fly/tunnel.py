"""Public URL for the visualiser from outside the network, no account needed.

Providers:
  ngrok      ngrok http 127.0.0.1:<port>  -> https://<id>.ngrok-free.app   (default when installed;
             install: winget install Ngrok.Ngrok, then once: ngrok config add-authtoken <your token>)
  lhr        localhost.run over plain SSH (ships with Windows):  ssh -R 80:127.0.0.1:<port> nokey@localhost.run
             -> https://<id>.lhr.life   (no install, no account, but slow)
  cloudflare Cloudflare quick tunnel:  cloudflared tunnel --url http://127.0.0.1:<port>
             -> https://<words>.trycloudflare.com   (api.trycloudflare.com is blocked on some networks)
  auto       ngrok if installed, otherwise lhr

Either way anyone with the link can watch (the page has no controls), the link changes on every
start, and the session is restarted automatically if the provider drops it.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time

CLOUDFLARED = [
    r"C:\Program Files (x86)\cloudflared\cloudflared.exe",
    r"C:\Program Files\cloudflared\cloudflared.exe",
    "/usr/local/bin/cloudflared", "/usr/bin/cloudflared", "/opt/homebrew/bin/cloudflared",
]
NGROK = [
    os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Links", "ngrok.exe"),
    r"C:\Program Files\ngrok\ngrok.exe", "/usr/local/bin/ngrok", "/opt/homebrew/bin/ngrok",
]
URL_RE = {
    "lhr": re.compile(r"https://[a-z0-9-]+\.lhr\.life"),
    "cloudflare": re.compile(r"https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com"),
    "ngrok": re.compile(r"https://[a-z0-9.-]+\.ngrok(?:-free)?\.(?:app|io|dev)"),
}


def find_cloudflared() -> str | None:
    return shutil.which("cloudflared") or next((p for p in CLOUDFLARED if os.path.exists(p)), None)


def find_ngrok() -> str | None:
    return shutil.which("ngrok") or next((p for p in NGROK if os.path.exists(p)), None)


def resolve(provider: str) -> str:
    if provider == "auto":
        return "ngrok" if find_ngrok() else "lhr"
    return provider


class Tunnel:
    def __init__(self, port: int, provider: str = "auto", on_url=None):
        provider = resolve(provider)
        if provider not in URL_RE:
            raise ValueError(f"unknown tunnel provider {provider!r} (ngrok | lhr | cloudflare | auto)")
        self.port = port
        self.provider = provider
        self.on_url = on_url                      # callback(url) when a (new) public URL is known
        self.proc: subprocess.Popen | None = None
        self.url: str | None = None
        self.log: list[str] = []
        self._stop = False

    def _command(self) -> list[str]:
        if self.provider == "ngrok":
            exe = find_ngrok()
            if not exe:
                raise RuntimeError("ngrok not found - install with: winget install Ngrok.Ngrok")
            return [exe, "http", f"127.0.0.1:{self.port}", "--log=stdout", "--log-format=json"]
        if self.provider == "cloudflare":
            exe = find_cloudflared()
            if not exe:
                raise RuntimeError("cloudflared not found - install with: winget install Cloudflare.cloudflared")
            return [exe, "tunnel", "--url", f"http://127.0.0.1:{self.port}", "--no-autoupdate"]
        ssh = shutil.which("ssh")
        if not ssh:
            raise RuntimeError("ssh not found (Windows: Settings > Apps > Optional features > OpenSSH Client)")
        return [ssh, "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30",
                "-o", "ServerAliveCountMax=3", "-o", "ExitOnForwardFailure=yes", "-o", "ConnectTimeout=20",
                "-T", "-R", f"80:127.0.0.1:{self.port}", "nokey@localhost.run"]

    def start(self, timeout: float = 40.0) -> str:
        self._stop = False
        self._spawn()
        t0 = time.time()
        while self.url is None and time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                tail = " ".join(self.log[-3:])[-400:]
                if self.provider == "ngrok" and "authtoken" in tail.lower():
                    raise RuntimeError("ngrok needs your authtoken once: ngrok config add-authtoken <token> "
                                       "(dashboard.ngrok.com -> Your Authtoken)")
                raise RuntimeError(f"{self.provider} tunnel exited: {tail}")
            time.sleep(0.2)
        if self.url is None:
            self.stop()
            raise RuntimeError(f"{self.provider} tunnel gave no URL in {timeout:.0f}s: "
                               + " ".join(self.log[-3:])[-400:])
        threading.Thread(target=self._watchdog, daemon=True).start()
        return self.url

    def _spawn(self):
        self.url = None
        self.proc = subprocess.Popen(
            self._command(), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()

    def _pump(self, proc):
        for line in proc.stdout:
            line = line.rstrip()
            self.log.append(line)
            if len(self.log) > 200:
                del self.log[:100]
            m = URL_RE[self.provider].search(line)
            if m and self.url is None and proc is self.proc:
                self.url = m.group(0)
                if self.on_url:
                    self.on_url(self.url)

    def _watchdog(self):
        """Free tunnels get dropped now and then: reconnect and publish the new link."""
        while not self._stop:
            time.sleep(2)
            if self._stop or self.proc is None:
                return
            if self.proc.poll() is not None:
                print(f"viz: {self.provider} tunnel dropped, reconnecting ...")
                try:
                    self._spawn()
                except Exception as e:
                    print(f"viz: tunnel restart failed: {e}")
                    time.sleep(15)

    def stop(self):
        self._stop = True
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc, self.url = None, None

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None and self.url is not None


def open_tunnel(port: int, provider: str = "auto", on_url=None) -> Tunnel:
    """Start the wanted provider; if it cannot (ngrok without an authtoken, blocked cloudflare), fall
    back to localhost.run so the fly still gets a public link. Raises only if everything failed."""
    first = resolve(provider)
    order = [first] + (["lhr"] if first != "lhr" else [])
    errors = []
    for prov in order:
        t = Tunnel(port, prov, on_url=on_url)
        try:
            t.start()
            if prov != first:
                print(f"viz: {first} failed ({errors[-1]}); using {prov} instead")
            return t
        except Exception as e:
            errors.append(str(e))
            t.stop()
    raise RuntimeError("; ".join(errors))
