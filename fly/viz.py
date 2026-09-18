"""Live visualiser: tiny HTTP + Server-Sent-Events server feeding viz/index.html (Three.js).

Streams: neuron activity on the 3-D soma cloud every few ticks, the page screenshot with link/button
boxes, the readout's softmax, the chosen click, the reward and (casino mode) balance / bet / win.

Remote viewing: the server listens on all interfaces; remote clients are only served while
`public` is on (CLI --public, or toggled at runtime from a localhost browser / GET /admin/public?on=1).
"""
from __future__ import annotations

import base64
import json
import os
import queue
import socket
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from .connectome import Brain

STATIC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "viz")

GROUPS = {   # superclass prefix -> colour group id used by the page
    "ol_": 1, "visual": 1, "cb_": 2, "vnc_": 3, "ascending": 3, "descending": 4, "vnc_motor": 5, "cb_motor": 5,
}
LOCAL = ("127.0.0.1", "::1", "::ffff:127.0.0.1")


def lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return socket.gethostbyname(socket.gethostname())


def all_ips() -> list[str]:
    """Every non-loopback IPv4 of this host (LAN, VPN/Tailscale, virtual switches) - colleagues may need any."""
    ips = [lan_ip()]
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith(("127.", "169.254.")) and ip not in ips:
                ips.append(ip)
    except OSError:
        pass
    if os.name == "nt":   # VPN (RAS) adapters are missing from getaddrinfo - read them off ipconfig
        import re
        import subprocess
        try:
            out = subprocess.run(["ipconfig"], capture_output=True, text=True, timeout=5, errors="replace").stdout
            for ip in re.findall(r"IPv4[^:]*:\s*(\d+\.\d+\.\d+\.\d+)", out):
                if not ip.startswith(("127.", "169.254.")) and ip not in ips:
                    ips.append(ip)
        except Exception:
            pass
    return ips


class VizServer:
    def __init__(self, brain: Brain, port: int = 8765, max_points: int = 120_000, every: int = 4,
                 tick_delay: float = 0.03, open_browser: bool = True, public: bool = False):
        self.brain = brain
        self.every = every
        self.tick_delay = tick_delay
        self.public = public
        self.clients: list[queue.Queue] = []
        self.lock = threading.Lock()

        has = ~np.isnan(brain.soma[:, 0])
        idx = np.flatnonzero(has)
        rng = np.random.default_rng(0)
        if idx.size > max_points:
            idx = rng.choice(idx, max_points, replace=False)
        pam = brain.reward_dopamine if brain.reward_dopamine is not None else np.zeros(0, np.int64)
        must = np.concatenate([brain.readout, brain.dopamine, pam, brain.retina_L[brain.retina_L >= 0],
                               brain.retina_R[brain.retina_R >= 0]])
        self.idx = np.unique(np.concatenate([idx, must[has[must]]]))
        self.act = np.zeros(self.idx.size, np.float32)

        pos = brain.soma[self.idx]
        pos = (pos - np.nanmean(pos, axis=0)) / (np.nanmax(np.abs(pos - np.nanmean(pos, axis=0))) + 1e-6)
        self.soma_bin = pos.astype(np.float32).tobytes()
        group = np.zeros(self.idx.size, np.uint8)
        sc = brain.superclass[self.idx].astype(str)
        for prefix, g in GROUPS.items():
            group[np.char.startswith(sc, prefix)] = g
        group[np.isin(self.idx, brain.descending)] = 4
        group[np.isin(self.idx, brain.motor)] = 5
        retina = np.concatenate([brain.retina_L[brain.retina_L >= 0], brain.retina_R[brain.retina_R >= 0]])
        group[np.isin(self.idx, retina)] = 6
        group[np.isin(self.idx, brain.dopamine)] = 7
        group[np.isin(self.idx, pam)] = 8
        # where the readout neurons (DN first, then motor) sit inside the streamed activity vector
        pos_in_idx = {int(v): i for i, v in enumerate(self.idx)}
        readout_pos = [pos_in_idx.get(int(v), -1) for v in brain.readout]
        self.meta = {"n": int(self.idx.size), "total": int(brain.n), "synapses": int(brain.W.nnz),
                     "n_descending": int(brain.descending.size), "readout": readout_pos,
                     "group": base64.b64encode(group.tobytes()).decode()}

        self.port = port
        self.ips = all_ips()
        self.lan = self.ips[0]
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def _send(self, body: bytes, ctype: str, code: int = 200):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                local = self.client_address[0] in LOCAL
                if not local and not server.public:
                    return self._send("доступ выключен / remote access is off".encode(), "text/plain; charset=utf-8", 403)
                u = urlparse(self.path)
                if u.path == "/":
                    with open(os.path.join(STATIC, "index.html"), "rb") as f:   # re-read: edit & refresh
                        self._send(f.read(), "text/html; charset=utf-8")
                elif u.path.startswith("/assets/") and ".." not in u.path:
                    fp = os.path.join(STATIC, u.path.lstrip("/").replace("/", os.sep))
                    if not os.path.isfile(fp):
                        return self.send_error(404)
                    ctype = {"js": "application/javascript", "glb": "model/gltf-binary", "png": "image/png",
                             "css": "text/css"}.get(fp.rsplit(".", 1)[-1], "application/octet-stream")
                    with open(fp, "rb") as f:
                        self._send(f.read(), ctype)
                elif u.path == "/soma.bin":
                    self._send(server.soma_bin, "application/octet-stream")
                elif u.path == "/meta.json":
                    meta = dict(server.meta, local=local, public=server.public,
                                lan_url=f"http://{server.lan}:{server.port}/",
                                lan_urls=[f"http://{ip}:{server.port}/" for ip in server.ips])
                    self._send(json.dumps(meta).encode(), "application/json")
                elif u.path == "/admin/public":
                    if not local:
                        return self._send(b"forbidden", "text/plain", 403)
                    q = parse_qs(u.query)
                    if "on" in q:
                        server.set_public(q["on"][0] in ("1", "true", "on"))
                    self._send(json.dumps({"public": server.public, "lan_url": f"http://{server.lan}:{server.port}/"})
                               .encode(), "application/json")
                elif u.path == "/events":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    qq: queue.Queue = queue.Queue(maxsize=64)
                    with server.lock:
                        server.clients.append(qq)
                    try:
                        while True:
                            msg = qq.get()
                            if msg is None:            # remote access switched off -> drop remote viewers
                                if not local:
                                    break
                                continue
                            self.wfile.write(b"data: " + msg + b"\n\n")
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        pass
                    finally:
                        with server.lock:
                            server.clients.remove(qq)
                else:
                    self.send_error(404)

        self.httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        remote = ", ".join(f"http://{ip}:{port}/" for ip in self.ips)
        print(f"viz: http://127.0.0.1:{port}/   remote: {remote} "
              f"({'ON' if public else 'off - toggle in the page or GET /admin/public?on=1'})")
        if open_browser:
            webbrowser.open(f"http://127.0.0.1:{port}/")

    def set_public(self, on: bool):
        self.public = bool(on)
        print(f"viz: remote access {'ON  -> http://%s:%d/' % (self.lan, self.port) if on else 'off'}")
        self.send({"t": "public", "on": self.public, "lan_url": f"http://{self.lan}:{self.port}/"})
        if not on:
            with self.lock:
                for q in self.clients:
                    try:
                        q.put_nowait(None)
                    except queue.Full:
                        pass

    def wait_for_client(self, timeout: float = 30.0):
        t0 = time.time()
        while not self.clients and time.time() - t0 < timeout:
            time.sleep(0.1)

    def send(self, event: dict, drop_if_busy: bool = False):
        msg = json.dumps(event).encode()
        with self.lock:
            for q in self.clients:
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    if not drop_if_busy:
                        q.get_nowait()
                        q.put_nowait(msg)

    # ---- hooks called from the agent loop -------------------------------------------------
    def tick(self, spikes: np.ndarray, tick: int):
        self.act *= 0.75
        self.act[spikes[self.idx]] = 1.0
        if tick % self.every == 0:
            a = np.clip(self.act * 255, 0, 255).astype(np.uint8)
            self.send({"t": "act", "tick": tick, "a": base64.b64encode(a.tobytes()).decode()}, drop_if_busy=True)
        if self.tick_delay:
            time.sleep(self.tick_delay)

    def page(self, obs: dict, step: int, episode: int):
        self.send({"t": "page", "step": step, "episode": episode, "url": obs["url"], "title": obs.get("title", ""),
                   "session": bool(obs.get("session")), "casino": obs.get("casino"),
                   "png": base64.b64encode(obs["png"]).decode(),
                   "links": [{"url": u, "text": t, "box": b} for (u, t), b in zip(obs["links"], obs["boxes"])]})

    def decision(self, probs: np.ndarray, action: int, rate: float):
        self.send({"t": "decision", "p": [round(float(x), 4) for x in probs], "a": int(action), "rate": rate})

    def reward(self, r: float, total: float, done: bool, casino: dict | None = None):
        self.send({"t": "reward", "r": r, "total": total, "done": done, "casino": casino})
