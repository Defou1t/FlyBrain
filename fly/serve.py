"""Keep the fly running: a supervisor that runs `fly.run` forever and hot-restarts it when the code
changes, so improving the project never stops the fly for more than a restart.

    python -m fly.serve --casino --viz --tunnel [any other fly.run flags]

* the worker (`python -m fly.run ... --episodes 0`) runs endlessly; the readout checkpoint survives
  restarts, so learning continues;
* every file under fly/ is watched: a change restarts the worker - but only after the changed files
  compile, so a half-typed edit is ignored and the running fly keeps going; viz/index.html needs no
  restart (the page reloads itself);
* a crashed worker is restarted after a short pause;
* the public tunnel (--tunnel) is owned by the supervisor, so the link stays the same across restarts.
Ctrl+C stops everything.
"""
from __future__ import annotations

import os
import py_compile
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WATCH_DIR = os.path.join(ROOT, "fly")
SUPERVISOR_ONLY = {"--tunnel", "--public", "--port", "--episodes"}


def snapshot() -> dict[str, float]:
    out = {}
    for name in os.listdir(WATCH_DIR):
        if name.endswith(".py") and name != "serve.py":
            p = os.path.join(WATCH_DIR, name)
            try:
                out[p] = os.stat(p).st_mtime
            except OSError:
                pass
    return out


def compiles(paths) -> bool:
    ok = True
    for p in paths:
        try:
            py_compile.compile(p, doraise=True)
        except py_compile.PyCompileError as e:
            print(f"serve: {os.path.basename(p)} does not compile yet, keeping the old fly running:\n  "
                  + str(e).strip().splitlines()[-1])
            ok = False
    return ok


def split_args(argv: list[str]):
    """Flags the supervisor handles itself vs. flags passed through to fly.run."""
    mine, rest, i = {}, [], 0
    while i < len(argv):
        a = argv[i]
        if a == "--public":
            mine["public"] = True
        elif a == "--tunnel":
            if i + 1 < len(argv) and argv[i + 1] in ("lhr", "cloudflare"):
                mine["tunnel"] = argv[i + 1]
                i += 1
            else:
                mine["tunnel"] = "lhr"
        elif a in ("--port", "--episodes"):
            mine[a.lstrip("-")] = argv[i + 1]
            i += 1
        else:
            rest.append(a)
        i += 1
    return mine, rest


class Worker:
    def __init__(self, args: list[str], env: dict):
        self.args = args
        self.env = env
        self.proc: subprocess.Popen | None = None
        self.started = 0.0

    def start(self):
        cmd = [sys.executable, "-u", "-m", "fly.run", *self.args]
        print("serve: starting  " + " ".join(cmd[2:]))
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.proc = subprocess.Popen(cmd, cwd=ROOT, env=self.env, creationflags=flags)
        self.started = time.time()

    def stop(self, grace: float = 20.0):
        if not self.proc or self.proc.poll() is not None:
            return
        print("serve: stopping the fly (closing browser, saving readout) ...")
        try:
            if os.name == "nt":
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)     # -> KeyboardInterrupt in fly.run
            else:
                self.proc.send_signal(signal.SIGINT)
            self.proc.wait(grace)
        except (subprocess.TimeoutExpired, OSError):
            self.proc.kill()
            self.proc.wait(5)

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


def main():
    mine, rest = split_args(sys.argv[1:])
    port = int(mine.get("port", 8765))
    if "--viz" not in rest:
        rest.append("--viz")
    worker_args = [*rest, "--episodes", str(mine.get("episodes", 0)), "--port", str(port), "--no-open"]
    if mine.get("public") or mine.get("tunnel"):
        worker_args.append("--public")
    env = dict(os.environ, PYTHONUNBUFFERED="1")

    tunnel = None
    if mine.get("tunnel"):
        from .tunnel import Tunnel
        tunnel = Tunnel(port, mine["tunnel"], on_url=lambda u: print(f"serve: public link -> {u}"))
        try:
            env["FLY_TUNNEL_URL"] = tunnel.start()
            print(f"serve: tunnel ON -> {env['FLY_TUNNEL_URL']}  (stays the same across restarts)")
        except Exception as e:
            print(f"serve: tunnel failed: {e}")
            tunnel = None

    worker = Worker(worker_args, env)
    worker.start()
    seen = snapshot()
    pending: dict[str, float] = {}
    crashes = 0
    try:
        while True:
            time.sleep(1.0)
            if tunnel and tunnel.url and env.get("FLY_TUNNEL_URL") != tunnel.url:   # reconnected with a new link
                env["FLY_TUNNEL_URL"] = tunnel.url
            now = snapshot()
            changed = [p for p, m in now.items() if seen.get(p) != m]
            for p in changed:
                pending[p] = time.time()
            seen = now
            if pending and time.time() - max(pending.values()) > 1.5:      # debounce: wait for the save burst
                files = list(pending)
                pending.clear()
                if compiles(files):
                    print("serve: code changed (" + ", ".join(os.path.basename(p) for p in files) + ") -> restart")
                    worker.stop()
                    worker.start()
                    crashes = 0
            if not worker.alive:
                code = worker.proc.returncode
                if code == 0:
                    print("serve: the fly finished its episodes, starting again")
                    worker.start()
                else:
                    crashes += 1
                    wait = min(60, 5 * crashes)
                    print(f"serve: the fly crashed (exit {code}), restart in {wait}s "
                          f"(fix the code - a good save restarts it immediately)")
                    t0 = time.time()
                    while time.time() - t0 < wait:
                        time.sleep(1)
                        now = snapshot()
                        if any(seen.get(p) != m for p, m in now.items()):
                            break
                    seen = snapshot()
                    worker.start()
    except KeyboardInterrupt:
        print("serve: shutting down")
    finally:
        worker.stop()
        if tunnel:
            tunnel.stop()


if __name__ == "__main__":
    main()
