@echo off
rem FlyBrain: one-click start on Windows. Creates a venv, installs dependencies and Chromium,
rem builds the brain (first run only, ~560 MB download) and starts the fly with the live page.
setlocal
cd /d "%~dp0"
where python >nul 2>nul || (echo Python 3.11+ is required: https://www.python.org/downloads/ & pause & exit /b 1)
if not exist .venv (
  echo creating a virtual environment ...
  python -m venv .venv || (pause & exit /b 1)
)
call .venv\Scripts\activate.bat
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt || (pause & exit /b 1)
python -m playwright install chromium || (pause & exit /b 1)
if not exist data\brain.npz (
  echo building the brain from the MaleCNS connectome (one time, ~560 MB download) ...
  python -m fly.build || (pause & exit /b 1)
)
echo.
echo starting the fly: http://127.0.0.1:8765/   (Ctrl+C stops it)
echo add --tunnel for a public link, --synthetic for a quick test without the real brain
echo.
python -m fly.serve --casino %*
pause
