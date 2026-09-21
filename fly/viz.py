"""Live visualiser: tiny HTTP + Server-Sent-Events server feeding viz/index.html (Three.js).

Streams: neuron activity on the 3-D soma cloud every few ticks, the page screenshot with link/button
boxes, the readout's softmax, the chosen click, the reward and (casino mode) balance / bet / win.

Remote viewing: the server listens on all interfaces; remote clients are only served while
`public` is on (CLI --public, or toggled at runtime from a localhost browser / GET /admin/public?on=1).
"""
from __future__ import annotations

import base64
import gzip
import io
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

from .browser import SESSION_FILE
from .connectome import Brain
LOGCSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dopamine.csv")

STATIC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "viz")

GROUPS = {   # superclass prefix -> colour group id used by the page
    "ol_": 1, "visual": 1, "cb_": 2, "vnc_": 3, "ascending": 3, "descending": 4, "vnc_motor": 5, "cb_motor": 5,
}
LOCAL = ("127.0.0.1", "::1", "::ffff:127.0.0.1")


class Client:
    """One SSE viewer. Reliable events queue up; the live streams (brain activity, video frames) keep
    only the latest message per kind, so a slow link (tunnel) always gets fresh data, never a backlog.
    Remote viewers also get a lower rate: they are on a tunnel with a few hundred KB/s at best."""

    def __init__(self, local: bool):
        self.local = local
        self.q: queue.Queue = queue.Queue(maxsize=256)
        self.latest: dict[str, bytes] = {}
        self.sent_at: dict[str, float] = {}
        self.min_gap = {"act": 0.045, "frame": 0.04} if local else {"act": 0.25, "frame": 0.15}
        self.wake = threading.Event()

    def put(self, kind: str, msg: bytes, droppable: bool):
        if droppable:
            self.latest[kind] = msg
        else:
            try:
                self.q.put_nowait(msg)
            except queue.Full:
                pass
        self.wake.set()

    def next(self) -> bytes | None:
        """Blocks until something is due; None = still nothing after a short wait."""
        try:
            return self.q.get_nowait()
        except queue.Empty:
            pass
        now = time.time()
        for kind in ("frame", "act"):
            msg = self.latest.get(kind)
            if msg is not None and now - self.sent_at.get(kind, 0.0) >= self.min_gap[kind]:
                del self.latest[kind]
                self.sent_at[kind] = now
                return msg
        self.wake.clear()
        self.wake.wait(0.05)
        return None


def to_jpeg(png: bytes, quality: int = 72) -> bytes:
    """The brain sees the PNG; viewers get a JPEG a quarter of the size."""
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(png)).convert("RGB")
        out = io.BytesIO()
        im.save(out, "JPEG", quality=quality, optimize=True)
        return out.getvalue()
    except Exception:
        return png


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
                 tick_delay: float = 0.012, open_browser: bool = True, public: bool = False,
                 tunnel_provider: str = "auto", tunnel_enabled: bool = False):
        self.brain = brain
        self.every = every
        self.tick_delay = tick_delay
        self.public = public
        self.tunnel = None
        self.tunnel_provider = tunnel_provider
        self.tunnel_enabled = tunnel_enabled or bool(os.environ.get("FLY_TUNNEL_URL"))   # off unless --tunnel was given
        self.commands: queue.Queue = queue.Queue()          # page -> agent loop
        self.state = {"paused": False, "mode": "browse", "broke": False}    # what the fly is doing now
        self.clients: list[Client] = []
        self.lock = threading.Lock()
        self.gz_cache: dict[str, bytes] = {}
        self.raw_cache: dict[str, bytes] = {}
        self.latest_jpg: bytes | None = None                 # newest screencast frame for /stream.mjpg
        self.frame_seq = 0
        self.frame_cv = threading.Condition()
        self.history: list[dict] = []                        # spins of this session (page reload keeps the charts)
        self._load_history()

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
        self.spiked = np.zeros(self.idx.size, bool)

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

            def _send_static(self, body: bytes, ctype: str, key: str):
                """Big static blobs: gzip once (cached), let the browser keep them for a day - the page
                reloads after every fly restart and must not re-download 8 MB through the tunnel."""
                gz = "gzip" in (self.headers.get("Accept-Encoding") or "")
                if gz:
                    body = server.gz_cache.get(key) or server.gz_cache.setdefault(key, gzip.compress(body, 6))
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "public, max-age=86400")
                if gz:
                    self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_cached(self, fp: str, ctype: str):
                key = fp + str(os.stat(fp).st_mtime)
                body = server.raw_cache.get(key)
                if body is None:
                    with open(fp, "rb") as f:
                        body = f.read()
                    server.raw_cache[key] = body
                self._send_static(body, ctype, key)

            def do_GET(self):
                # viewers coming through a tunnel connect from 127.0.0.1: they are told apart by proxy headers
                # and by the Host they asked for (the tunnel's hostname, not 127.0.0.1/localhost)
                proxied = bool(self.headers.get("Cf-Connecting-Ip") or self.headers.get("X-Forwarded-For")
                               or self.headers.get("X-Forwarded-Host"))
                host = (self.headers.get("Host") or "").split(":")[0].lower()
                local = (self.client_address[0] in LOCAL and not proxied
                         and host in ("127.0.0.1", "localhost", "::1", "[::1]"))
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
                    self._send_cached(fp, ctype)
                elif u.path == "/soma.bin":
                    self._send_static(server.soma_bin, "application/octet-stream", "soma.bin")
                elif u.path == "/history.json":
                    with server.lock:
                        body = json.dumps({"spins": server.history[-2000:]}).encode()
                    self._send(body, "application/json")
                elif u.path == "/stream.mjpg":
                    # live video of the fly's browser as motion-JPEG: the <img> decodes natively, no JSON,
                    # no base64; local viewers get every frame (~60 fps), tunnel viewers ~8 fps
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    gap = 1 / 60 if local else 1 / 8
                    seen, last = -1, 0.0
                    try:
                        while True:
                            with server.frame_cv:
                                server.frame_cv.wait_for(lambda: server.frame_seq != seen, timeout=1.0)
                                jpg, seen = server.latest_jpg, server.frame_seq
                            if not jpg:
                                continue
                            wait = gap - (time.time() - last)
                            if wait > 0:
                                time.sleep(wait)
                            last = time.time()
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                        pass
                elif u.path == "/meta.json":
                    meta = dict(server.meta, local=local, public=server.public,
                                lan_url=f"http://{server.lan}:{server.port}/",
                                lan_urls=[f"http://{ip}:{server.port}/" for ip in server.ips],
                                tunnel_url=server.tunnel_url, tunnel_managed=server.tunnel_managed, tunnel_enabled=server.tunnel_enabled,
                                state=server.state, session=os.path.exists(SESSION_FILE) or bool(os.environ.get("FLY_PHPSESSID")))
                    self._send(json.dumps(meta).encode(), "application/json")
                elif u.path == "/admin/cmd":          # page -> fly: pause / resume / casino / browse
                    if not local:
                        return self._send(b"forbidden", "text/plain", 403)
                    do = parse_qs(u.query).get("do", [""])[0]
                    if do not in ("pause", "resume", "casino", "browse", "refill", "reset"):
                        return self._send(b"unknown command", "text/plain", 400)
                    server.commands.put(do)
                    self._send(json.dumps({"queued": do, "state": server.state}).encode(), "application/json")
                elif u.path == "/admin/session":      # first run: the viewer pastes their own PHPSESSID (local only)
                    if not local:
                        return self._send(b"forbidden", "text/plain", 403)
                    q = parse_qs(u.query)
                    val = (q.get("value") or [""])[0].strip()
                    if not re.fullmatch(r"[A-Za-z0-9,_-]{16,128}", val):
                        return self._send(json.dumps({"ok": False, "error": "that does not look like a PHPSESSID"}).encode(),
                                          "application/json", 400)
                    os.makedirs(os.path.dirname(SESSION_FILE), exist_ok=True)
                    with open(SESSION_FILE, "w", encoding="utf-8") as f:
                        f.write(val + "\n")
                    server.commands.put("restart")        # the worker exits cleanly; fly.serve starts it again
                    self._send(json.dumps({"ok": True}).encode(), "application/json")
                elif u.path in ("/admin/public", "/admin/tunnel"):
                    if not local:
                        return self._send(b"forbidden", "text/plain", 403)
                    q = parse_qs(u.query)
                    err = None
                    if "on" in q:
                        on = q["on"][0] in ("1", "true", "on")
                        if u.path == "/admin/public":
                            server.set_public(on)
                        else:
                            try:
                                server.set_tunnel(on)
                            except Exception as e:
                                err = str(e)
                    self._send(json.dumps({"public": server.public, "lan_url": f"http://{server.lan}:{server.port}/",
                                           "tunnel_url": server.tunnel_url, "error": err}).encode(),
                               "application/json")
                elif u.path == "/events":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    cl = Client(local)
                    with server.lock:
                        server.clients.append(cl)
                    try:
                        while True:
                            msg = cl.next()
                            if msg is None:
                                continue
                            if msg == b"":             # remote access switched off -> drop remote viewers
                                if not local:
                                    break
                                continue
                            self.wfile.write(b"data: " + msg + b"\n\n")
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        pass
                    finally:
                        with server.lock:
                            server.clients.remove(cl)
                else:
                    self.send_error(404)

        class Server(ThreadingHTTPServer):
            allow_reuse_address = False      # Windows would happily let two flies share the port otherwise

        try:
            self.httpd = Server(("0.0.0.0", port), Handler)
        except OSError as e:
            raise SystemExit(f"viz: port {port} is already in use (another fly running?) - pick --port: {e}")
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        threading.Thread(target=self._watch_page, daemon=True).start()
        remote = ", ".join(f"http://{ip}:{port}/" for ip in self.ips)
        print(f"viz: http://127.0.0.1:{port}/   remote: {remote} "
              f"({'ON' if public else 'off - toggle in the page or GET /admin/public?on=1'})")
        if open_browser:
            webbrowser.open(f"http://127.0.0.1:{port}/")

    @property
    def tunnel_url(self) -> str | None:
        if self.tunnel and self.tunnel.alive:
            return self.tunnel.url
        return os.environ.get("FLY_TUNNEL_URL") or None      # tunnel owned by fly.serve (survives restarts)

    @property
    def tunnel_managed(self) -> bool:
        return bool(os.environ.get("FLY_TUNNEL_URL"))

    def set_tunnel(self, on: bool, provider: str | None = None):
        """Start/stop a public tunnel (localhost.run over ssh, or cloudflared); starting also switches
        remote access on. Viewers through the tunnel count as remote (proxy headers), never as admins."""
        from .tunnel import open_tunnel
        if not self.tunnel_enabled:
            raise RuntimeError("the tunnel is disabled: start with --tunnel (or --tunnel manual) to allow it")
        if self.tunnel_managed:
            print("viz: the tunnel is owned by fly.serve - stop/start it there")
            return
        if on:
            if not self.tunnel_url:
                if self.tunnel:
                    self.tunnel.stop()
                self.tunnel = open_tunnel(self.port, provider or self.tunnel_provider, on_url=self._tunnel_changed)
                print(f"viz: tunnel ON ({self.tunnel.provider}) -> {self.tunnel.url}   "
                      "(anyone with the link can watch; new link on every start)")
            if not self.public:
                self.set_public(True)
        elif self.tunnel:
            self.tunnel.stop()
            print("viz: tunnel off")
        self.send({"t": "public", "on": self.public, "lan_url": f"http://{self.lan}:{self.port}/",
                   "tunnel_url": self.tunnel_url})

    def _tunnel_changed(self, url: str):
        print(f"viz: public link -> {url}")
        self.send({"t": "public", "on": self.public, "lan_url": f"http://{self.lan}:{self.port}/", "tunnel_url": url})

    def set_public(self, on: bool):
        self.public = bool(on)
        print(f"viz: remote access {'ON  -> http://%s:%d/' % (self.lan, self.port) if on else 'off'}")
        self.send({"t": "public", "on": self.public, "lan_url": f"http://{self.lan}:{self.port}/",
                   "tunnel_url": self.tunnel_url})
        if not on:
            with self.lock:
                for cl in self.clients:
                    cl.put("kick", b"", droppable=False)

    def _watch_page(self):
        """viz/index.html edited -> tell open pages to reload (no restart needed for the page itself)."""
        path = os.path.join(STATIC, "index.html")
        last = os.stat(path).st_mtime
        while True:
            time.sleep(2)
            try:
                m = os.stat(path).st_mtime
            except OSError:
                continue
            if m != last:
                last = m
                time.sleep(1)                       # let the editor finish writing
                print("viz: index.html changed -> reloading open pages")
                self.send({"t": "reload"})

    def wait_for_client(self, timeout: float = 30.0):
        t0 = time.time()
        while not self.clients and time.time() - t0 < timeout:
            time.sleep(0.1)

    def send(self, event: dict, drop_if_busy: bool = False):
        msg = json.dumps(event).encode()
        with self.lock:
            for cl in self.clients:
                cl.put(event["t"], msg, droppable=drop_if_busy)

    # ---- hooks called from the agent loop -------------------------------------------------
    def tick(self, spikes: np.ndarray, tick: int):
        # a spike bitmask (1 bit per neuron, ~15 KB) instead of 8-bit activity (120 KB); the page decays it
        self.spiked |= spikes[self.idx]
        if tick % self.every == 0:
            bits = np.packbits(self.spiked)
            self.spiked[:] = False
            self.send({"t": "act", "tick": tick, "n": int(self.idx.size),
                       "s": base64.b64encode(bits.tobytes()).decode()}, drop_if_busy=True)
        if self.tick_delay:
            time.sleep(self.tick_delay)

    def page(self, obs: dict, step: int, episode: int):
        self.send({"t": "page", "step": step, "episode": episode, "url": obs["url"], "title": obs.get("title", ""),
                   "session": bool(obs.get("session")), "casino": obs.get("casino"),
                   "jpg": base64.b64encode(to_jpeg(obs["png"])).decode(),
                   "links": [{"url": u, "text": t, "box": b} for (u, t), b in zip(obs["links"], obs["boxes"])]})

    def frame(self, jpg: bytes):
        """Live view of the browser (Chromium screencast). Served as MJPEG at /stream.mjpg; every viewer
        takes the newest frame at its own pace (~60 fps local, ~8 fps through a tunnel)."""
        with self.frame_cv:
            self.latest_jpg = jpg
            self.frame_seq += 1
            self.frame_cv.notify_all()

    def decision(self, probs: np.ndarray, action: int, rate: float):
        self.send({"t": "decision", "p": [round(float(x), 4) for x in probs], "a": int(action), "rate": rate})

    def reward(self, r: float, total: float, done: bool, casino: dict | None = None):
        if casino and casino.get("bet") is not None:
            d = casino.get("drive") or {}
            row = {"ts": round(time.time(), 1), "r": round(float(r), 3), "bet": casino.get("bet"), "win": casino.get("win"),
                   "bank": casino.get("balance"), "desire": d.get("desire"), "slot": d.get("name")}
            with self.lock:
                self.history.append(row)
                self.history = self.history[-5000:]
            casino = dict(casino, row=row)
        self.send({"t": "reward", "r": r, "total": total, "done": done, "casino": casino})

    def _load_history(self):
        """Spins of the current bank session from data/dopamine.csv (rows after the last refill/reset)."""
        try:
            import csv
            from .drive import STATE
            since = 0.0
            try:
                with open(STATE, encoding="utf-8") as f:
                    since = float(json.load(f).get("since_ts", 0.0))
            except (OSError, ValueError):
                pass
            with open(LOGCSV, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            out = []
            for x in rows:
                try:
                    ts = float(x["ts"])
                    if ts < since:
                        continue
                    extra = x.get(None) or []          # rows written after the bank/desire columns were added to an old file
                    bank = x.get("bank") or (extra[0] if len(extra) > 0 else None)
                    desire = x.get("desire") or (extra[1] if len(extra) > 1 else None)
                    out.append({"ts": ts, "r": float(x["reward"]), "bet": float(x["bet"]), "win": float(x["win"]),
                                "bank": float(bank) if bank else None, "desire": float(desire) if desire else None, "slot": None})
                except (KeyError, ValueError):
                    continue
            self.history = out[-5000:]
        except OSError:
            self.history = []

    def clear_history(self):
        with self.lock:
            self.history = []
        self.send({"t": "reset"})

    def set_state(self, **kw):
        self.state.update(kw)
        self.send({"t": "state", **self.state})
