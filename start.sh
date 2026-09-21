#!/usr/bin/env bash
# FlyBrain on Linux / macOS: a private virtualenv in runtime/venv (needs python3 >= 3.11 on the
# system - `sudo apt install python3 python3-venv` / `brew install python`), dependencies, Chromium,
# the brain (first run, ~560 MB), then the fly.   ./start.sh [--tunnel] [--synthetic] [--setup]
set -e
cd "$(dirname "$0")"
export PYTHONNOUSERSITE=1
PY=runtime/venv/bin/python
if [ ! -x "$PY" ]; then
  command -v python3 >/dev/null || { echo "FlyBrain: python3 (3.11+) is required"; exit 1; }
  echo "FlyBrain: creating a private virtualenv in runtime/venv ..."
  mkdir -p runtime && python3 -m venv runtime/venv
fi
"$PY" -m pip install -q --disable-pip-version-check -r requirements.txt
"$PY" -m playwright install chromium
if [ ! -f data/brain.npz ] && [ "$1" != "--synthetic" ]; then
  echo "FlyBrain: building the brain from the MaleCNS connectome - one time, ~560 MB download ..."
  "$PY" -m fly.build
fi
[ "$1" = "--setup" ] && { echo "FlyBrain: setup complete."; exit 0; }
echo "FlyBrain: starting - open http://127.0.0.1:8765/   (Ctrl+C stops it)"
exec "$PY" -m fly.serve --casino "$@"
