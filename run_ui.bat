@echo off
REM Mo UI. Khong truyen tham so -> --weights real_3class.pt (model 3 lop, 23/11/2026)
setlocal
cd /d "%~dp0"
set "PY=python"
if exist "venv_python.txt" set /p PY=<venv_python.txt
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
set "PYTHONPATH=%CD%"
if "%~1"=="" (
  "%PY%" src\ui\tkinter_app.py --weights real_3class.pt
) else (
  "%PY%" src\ui\tkinter_app.py %*
)
