"""Build a shareable zip of FlyBrain: code, page, vendored assets, docs, start.bat - and nothing that
belongs to this machine (downloaded data, the brain, the session cookie, logs, checkpoints, venv).

    python tools/pack.py            -> dist/FlyBrain-<date>.zip
"""
from __future__ import annotations

import os
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCLUDE = ["fly", "viz", "docs", "tools", "README.md", "requirements.txt", "start.bat", "start.sh", ".gitignore", ".gitattributes"]
SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", "node_modules", "dist", "data", "runtime"}
SKIP_EXT = {".pyc", ".log"}


def files():
    for item in INCLUDE:
        p = os.path.join(ROOT, item)
        if os.path.isfile(p):
            yield p
            continue
        for d, dirs, names in os.walk(p):
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
            for n in names:
                if os.path.splitext(n)[1] in SKIP_EXT:
                    continue
                yield os.path.join(d, n)


def main():
    os.makedirs(os.path.join(ROOT, "dist"), exist_ok=True)
    name = sys.argv[1] if len(sys.argv) > 1 else f"FlyBrain-{time.strftime('%Y-%m-%d')}.zip"
    out = os.path.join(ROOT, "dist", name)
    total = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files():
            rel = os.path.relpath(p, ROOT).replace(os.sep, "/")
            if rel == "data/session.txt":          # never
                continue
            z.write(p, "FlyBrain/" + rel)
            total += os.path.getsize(p)
        z.writestr("FlyBrain/data/README.txt",
                   "Runtime data lives here: the MaleCNS download and data/brain.npz (python -m fly.build), "
                   "readout checkpoints, dopamine.csv, drive.json, events.jsonl, fly.log, and your own "
                   "session.txt (PHPSESSID) if you choose to give the fly your account. Nothing here is shared.\n")
    print(f"{out}  ({os.path.getsize(out) / 1e6:.1f} MB zipped, {total / 1e6:.1f} MB of files)")


if __name__ == "__main__":
    main()
