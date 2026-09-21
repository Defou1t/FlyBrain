@echo off
rem FlyBrain: stop the running fly (its supervisor, browser and tunnel). The page's gear menu has the
rem same "Stop the fly" item; Ctrl+C or closing the start.bat window works too.
setlocal
cd /d "%~dp0"
set "PYTHONNOUSERSITE=1"
set "PY=%~dp0runtime\python\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" -m fly.serve --stop %*
pause
