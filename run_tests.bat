@echo off
REM Chay test cua ban UI.
setlocal
cd /d "%~dp0"
set "PY=python"
if exist "venv_python.txt" set /p PY=<venv_python.txt
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
set "PYTHONPATH=%CD%"
"%PY%" -m pytest tests -q %*
