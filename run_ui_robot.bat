REM UI + gui lenh phan loai cho robot (chay kem run_robot_sim.bat hoac run_robot.bat).
@echo off
setlocal
cd /d "%~dp0"
set "PY=python"
if exist "venv_python.txt" set /p PY=<venv_python.txt
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
set "PYTHONPATH=%CD%"
"%PY%" src\ui\tkinter_app.py --weights real_3class.pt --robot %*
