REM Robot gia lap: khong can Dobot, in lenh va cho 2 giay moi bo.
@echo off
setlocal
cd /d "%~dp0"
set "PY=python"
if exist "venv_python.txt" set /p PY=<venv_python.txt
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
set "PYTHONPATH=%CD%"
"%PY%" scripts\robot_worker.py --sim
