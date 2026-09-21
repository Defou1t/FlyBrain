@echo off
rem FlyBrain: update to the latest version from GitHub. The running fly picks the new code up by
rem itself (fly.serve hot-restarts it, the page reloads); dependencies are re-checked by start.bat.
setlocal
cd /d "%~dp0"
where git >nul 2>nul || (echo git is required for updates - or download the ZIP from https://github.com/Defou1t/FlyBrain again and replace the files. & pause & exit /b 1)
if not exist .git (echo This folder is not a git checkout - download the ZIP from https://github.com/Defou1t/FlyBrain and replace the files. & pause & exit /b 1)
git pull --ff-only || (pause & exit /b 1)
if exist runtime\python\python.exe (
  set "PYTHONNOUSERSITE=1"
  runtime\python\python.exe -m pip install -q --disable-pip-version-check --no-warn-script-location -r requirements.txt
  runtime\python\python.exe -m playwright install chromium
)
echo FlyBrain: up to date. A running fly restarts itself with the new code; otherwise run start.bat.
pause
