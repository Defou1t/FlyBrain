"""Public URL for the visualiser from outside the network, no account needed.

Providers:
  lhr        localhost.run over plain SSH (ships with Windows):  ssh -R 80:127.0.0.1:<port> nokey@localhost.run
             -> https://<id>.lhr.life   (default: works where Cloudflare's quick-tunnel API is blocked)
  cloudflare Cloudflare quick tunnel:  cloudflared tunnel --url http://127.0.0.1:<port>
             -> https://<words>.trycloudflare.com   (install: winget install Cloudflare.cloudflared)

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
URL_RE = {
    "lhr": re.compile(r"https://[a-z0-9-]+\.lhr\.life"),
    "cloudflare": re.compile(r"https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com"),
}


def find_cloudflared() -> str | None:
    return shutil.which("cloudflared") or next((p for p in CLOUDFLARED if os.path.exists(p)), None)


class Tunnel:
    def __init__(self, port: int, provider: str = "lhr", on_url=None):
        if provider not in URL_RE:
            raise ValueError(f"unknown tunnel provider {provider!r} (lhr | cloudflare)")
        self.port = port
        self.provider = provider
        self.on_url = on_url                      # callback(url) when a (new) public URL is known
        self.proc: subprocess.Popen | None = None
        self.url: str | None = None
        self.log: list[str] = []
        self._stop = False

    def _command(self) -> list[str]:
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
                raise RuntimeError(f"{self.provider} tunnel exited: " + " ".join(self.log[-3:])[-400:])
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
