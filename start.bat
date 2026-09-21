@echo off
rem FlyBrain: one-click start on Windows. Needs no Python on the machine: a private Python 3.12 is
rem downloaded into runtime\python (the official "embeddable" build, ~11 MB) and everything is
rem installed there - nothing else on the computer is touched, no version conflicts with any other
rem Python. Then Chromium is installed, the brain is built (first run only, ~560 MB download) and
rem the fly starts with the live page.
rem
rem   start.bat                 run the fly (casino, live page on http://127.0.0.1:8765/)
rem   start.bat --tunnel        ... plus a public link (opt-in)
rem   start.bat --synthetic     quick test on a random brain, nothing to download
rem   start.bat --setup         only install, do not start
setlocal
cd /d "%~dp0"
set "PYTHONNOUSERSITE=1"
rem ^ never pick up packages of another Python installed for this user (no version conflicts)
set "PY=%~dp0runtime\python\python.exe"
set "PYVER=3.12.10"

if exist "%PY%" goto :have_python
echo FlyBrain: installing a private Python %PYVER% into runtime\python ...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol='Tls12';" ^
  "New-Item -ItemType Directory -Force runtime | Out-Null;" ^
  "$z='runtime\python.zip'; Invoke-WebRequest ('https://www.python.org/ftp/python/%PYVER%/python-%PYVER%-embed-amd64.zip') -OutFile $z;" ^
  "Expand-Archive $z -DestinationPath runtime\python -Force; Remove-Item $z;" ^
  "$pth = Get-ChildItem runtime\python\python3*._pth | Select-Object -First 1;" ^
  "(Get-Content $pth) -replace '^#import site','import site' | Set-Content $pth;" ^
  "Add-Content $pth 'Lib\site-packages'; Add-Content $pth '..\..';" ^
  "Invoke-WebRequest 'https://bootstrap.pypa.io/get-pip.py' -OutFile runtime\get-pip.py"
if errorlevel 1 goto :fail
"%PY%" runtime\get-pip.py --no-warn-script-location --disable-pip-version-check
if errorlevel 1 goto :fail
del runtime\get-pip.py
echo FlyBrain: Python ready.

:have_python
echo FlyBrain: installing dependencies ...
"%PY%" -m pip install -q --disable-pip-version-check --no-warn-script-location -r requirements.txt
if errorlevel 1 goto :fail
"%PY%" -m playwright install chromium
if errorlevel 1 goto :fail
if exist data\brain.npz goto :built
if "%~1"=="--synthetic" goto :run
echo FlyBrain: building the brain from the MaleCNS connectome - one time, ~560 MB download ...
"%PY%" -m fly.build
if errorlevel 1 goto :fail
:built
if not "%~1"=="--setup" goto :run
echo FlyBrain: setup complete.
goto :eof

:run
echo.
echo FlyBrain: starting - open http://127.0.0.1:8765/   (Ctrl+C stops it)
echo.
"%PY%" -m fly.serve --casino %*
goto :eof

:fail
echo.
echo FlyBrain: setup failed (see the message above). Check the internet connection and run start.bat again.
pause
exit /b 1
