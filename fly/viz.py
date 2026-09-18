"""Live visualiser: tiny HTTP + Server-Sent-Events server feeding viz/index.html (Three.js).

Streams: neuron activity on the 3-D soma cloud every few ticks, the page screenshot with link boxes,
the readout's softmax over links, the chosen click and the reward.
"""
from __future__ import annotations

import base64
import json
import os
import queue
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .connectome import Brain

STATIC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "viz")

GROUPS = {   # superclass prefix -> colour group id used by the page
    "ol_": 1, "visual": 1, "cb_": 2, "vnc_": 3, "ascending": 3, "descending": 4, "vnc_motor": 5, "cb_motor": 5,
}


class VizServer:
    def __init__(self, brain: Brain, port: int = 8765, max_points: int = 120_000, every: int = 4,
                 tick_delay: float = 0.03, open_browser: bool = True):
        self.brain = brain
        self.every = every
        self.tick_delay = tick_delay
        self.clients: list[queue.Queue] = []
        self.lock = threading.Lock()

        has = ~np.isnan(brain.soma[:, 0])
        idx = np.flatnonzero(has)
        rng = np.random.default_rng(0)
        if idx.size > max_points:
            idx = rng.choice(idx, max_points, replace=False)
        must = np.concatenate([brain.readout, brain.dopamine, brain.retina_L[brain.retina_L >= 0],
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
        group[np.isin(self.idx, brain.readout)] = np.where(np.isin(self.idx[np.isin(self.idx, brain.readout)],
                                                                    brain.descending), 4, 5)
        retina = np.concatenate([brain.retina_L[brain.retina_L >= 0], brain.retina_R[brain.retina_R >= 0]])
        group[np.isin(self.idx, retina)] = 6
        group[np.isin(self.idx, brain.dopamine)] = 7
        # where the readout neurons (DN first, then motor) sit inside the streamed activity vector
        pos_in_idx = {int(v): i for i, v in enumerate(self.idx)}
        readout_pos = [pos_in_idx.get(int(v), -1) for v in brain.readout]
        self.meta = json.dumps({"n": int(self.idx.size), "total": int(brain.n), "synapses": int(brain.W.nnz),
                                "n_descending": int(brain.descending.size), "readout": readout_pos,
                                "group": base64.b64encode(group.tobytes()).decode()})

        self.port = port
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def _send(self, body: bytes, ctype: str):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/":
                    with open(os.path.join(STATIC, "index.html"), "rb") as f:   # re-read: edit & refresh
                        self._send(f.read(), "text/html; charset=utf-8")
                elif self.path.startswith("/assets/") and ".." not in self.path:
                    fp = os.path.join(STATIC, self.path.lstrip("/").replace("/", os.sep))
                    if not os.path.isfile(fp):
                        return self.send_error(404)
                    ctype = {"js": "application/javascript", "glb": "model/gltf-binary", "png": "image/png",
                             "css": "text/css"}.get(fp.rsplit(".", 1)[-1], "application/octet-stream")
                    with open(fp, "rb") as f:
                        self._send(f.read(), ctype)
                elif self.path == "/soma.bin":
                    self._send(server.soma_bin, "application/octet-stream")
                elif self.path == "/meta.json":
                    self._send(server.meta.encode(), "application/json")
                elif self.path == "/events":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    q: queue.Queue = queue.Queue(maxsize=64)
                    with server.lock:
                        server.clients.append(q)
                    try:
                        while True:
                            msg = q.get()
                            self.wfile.write(b"data: " + msg + b"\n\n")
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        pass
                    finally:
                        with server.lock:
                            server.clients.remove(q)
                else:
                    self.send_error(404)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        print(f"viz: http://127.0.0.1:{port}/")
        if open_browser:
            webbrowser.open(f"http://127.0.0.1:{port}/")

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
                   "session": bool(obs.get("session")), "png": base64.b64encode(obs["png"]).decode(),
                   "links": [{"url": u, "text": t, "box": b} for (u, t), b in zip(obs["links"], obs["boxes"])]})

    def decision(self, probs: np.ndarray, action: int, rate: float):
        self.send({"t": "decision", "p": [round(float(x), 4) for x in probs], "a": int(action), "rate": rate})

    def reward(self, r: float, total: float, done: bool):
        self.send({"t": "reward", "r": r, "total": total, "done": done})
