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
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WATCH_DIR = os.path.join(ROOT, "fly")
LOG = os.path.join(ROOT, "data", "fly.log")      # everything the supervisor and the fly print, with times
LOG_MAX = 8 * 1024 * 1024


def say(line: str, echo: bool = True):
    """Print and append to data/fly.log, so 'did the fly break?' can be answered after the terminal is gone."""
    line = line.rstrip("\n")
    if echo:
        print(line, flush=True)
    try:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        if os.path.exists(LOG) and os.path.getsize(LOG) > LOG_MAX:      # keep the tail, drop the rest
            with open(LOG, "rb") as f:
                f.seek(-LOG_MAX // 4, 2)
                tail = f.read()
            with open(LOG, "wb") as f:
                f.write(tail)
        with open(LOG, "a", encoding="utf-8", errors="replace") as f:
            f.write(time.strftime("%d.%m %H:%M:%S ") + line + "\n")
    except OSError:
        pass
SUPERVISOR_ONLY = {"--tunnel", "--public", "--port", "--episodes"}


def bind_children_to_me():
    """Windows: put this process into a Job Object with KILL_ON_JOB_CLOSE. Everything it spawns (the
    fly, its Chromium, ngrok/ssh) inherits the job and is killed the moment the supervisor dies for any
    reason - Ctrl+C, closed terminal, crash - so no orphan can keep the port."""
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.windll.kernel32
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.GetCurrentProcess.restype = wintypes.HANDLE          # pseudo-handle -1 must stay 64-bit
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    job = k32.CreateJobObjectW(None, None)
    if not job:
        return

    class LIMIT(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class IO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                                    "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class EXT(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", LIMIT), ("IoInfo", IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    info = EXT()
    info.BasicLimitInformation.LimitFlags = 0x2000          # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))   # ExtendedLimitInformation
    if ok and k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
        globals()["_JOB"] = job                                # keep the handle alive for the whole run
    else:
        print("serve: could not create the job object; children may outlive the supervisor")


def listener_pid(port: int) -> int | None:
    """PID listening on the port (Windows netstat / Linux ss)."""
    try:
        if os.name == "nt":
            out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, timeout=10).stdout
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 5 and parts[1].endswith(f":{port}") and parts[3] == "LISTENING":
                    return int(parts[4])
        else:
            out = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True, timeout=10).stdout
            for line in out.splitlines():
                if f":{port} " in line and "pid=" in line:
                    return int(line.split("pid=")[1].split(",")[0])
    except Exception:
        pass
    return None


def free_port(port: int):
    """A stale fly (a worker whose supervisor died earlier) still holding the port is ours to kill;
    anything else on the port is not - say so and stop."""
    pid = listener_pid(port)
    if not pid or pid == os.getpid():
        return
    cmd = ""
    try:
        if os.name == "nt":   # (wmic is gone on recent Windows 11 builds)
            cmd = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"],
                                 capture_output=True, text=True, timeout=15).stdout.strip()
        else:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\\0", b" ").decode(errors="replace")
    except Exception:
        pass
    if "fly.run" in cmd or "fly.serve" in cmd:
        print(f"serve: port {port} is held by a stale fly (pid {pid}) - stopping it")
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"] if os.name == "nt" else ["kill", "-9", str(pid)],
                       capture_output=True)
        time.sleep(2)
    else:
        raise SystemExit(f"serve: port {port} is in use by another program (pid {pid}: {cmd[:80] or '?'}); "
                         f"pick another --port")


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
            if i + 1 < len(argv) and argv[i + 1] in ("auto", "ngrok", "lhr", "cloudflare"):
                mine["tunnel"] = argv[i + 1]
                i += 1
            else:
                mine["tunnel"] = "auto"
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
        say("serve: starting  " + " ".join(cmd[2:]))
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.proc = subprocess.Popen(cmd, cwd=ROOT, env=self.env, creationflags=flags,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.started = time.time()
        threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()

    @staticmethod
    def _pump(proc):
        with proc.stdout:
            for raw in iter(proc.stdout.readline, b""):
                say(raw.decode("utf-8", "replace"))

    def stop(self, grace: float = 20.0):
        if not self.proc or self.proc.poll() is not None:
            return
        say("serve: stopping the fly (closing browser, saving readout) ...")
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
    say(f"serve: supervisor started (pid {os.getpid()}) " + " ".join(sys.argv[1:]))
    bind_children_to_me()
    free_port(port)
    if "--viz" not in rest:
        rest.append("--viz")
    worker_args = [*rest, "--episodes", str(mine.get("episodes", 0)), "--port", str(port), "--no-open"]
    if mine.get("public") or mine.get("tunnel"):
        worker_args.append("--public")
    env = dict(os.environ, PYTHONUNBUFFERED="1")

    tunnel = None
    if mine.get("tunnel"):
        from .tunnel import open_tunnel
        try:
            tunnel = open_tunnel(port, mine["tunnel"], on_url=lambda u: say(f"serve: public link -> {u}"))
            env["FLY_TUNNEL_URL"] = tunnel.url
            say(f"serve: tunnel ON ({tunnel.provider}) -> {tunnel.url}  (stays the same across restarts)")
        except Exception as e:
            say(f"serve: tunnel failed: {e}")
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
                    say("serve: code changed (" + ", ".join(os.path.basename(p) for p in files) + ") -> restart")
                    worker.stop()
                    worker.start()
                    crashes = 0
            if not worker.alive:
                code = worker.proc.returncode
                if code == 0:
                    say("serve: the fly finished its episodes, starting again")
                    worker.start()
                else:
                    crashes += 1
                    wait = min(60, 5 * crashes)
                    say(f"serve: the fly crashed (exit {code}), restart in {wait}s "
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
        say("serve: shutting down")
    finally:
        worker.stop()
        if tunnel:
            tunnel.stop()
        say("serve: supervisor exited")


if __name__ == "__main__":
    main()
