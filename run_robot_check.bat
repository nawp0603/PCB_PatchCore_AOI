@echo off
REM Kiem tra JSON cau hinh Dobot. Lenh nay khong ket noi va khong di chuyen robot.
setlocal
cd /d "%~dp0"
set "PY=python"
if exist "venv_python.txt" set /p PY=<venv_python.txt
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
set "PYTHONPATH=%CD%"
"%PY%" scripts\robot_worker.py --check-config
