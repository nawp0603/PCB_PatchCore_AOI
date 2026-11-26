@echo off
setlocal
cd /d "%~dp0"

set "PY=python"
where py >nul 2>nul && set "PY=py"

if not exist ".venv\Scripts\python.exe" (
  echo [1/4] Tao moi truong Python...
  %PY% -m venv .venv
  if errorlevel 1 goto :error
)

set "VPY=.venv\Scripts\python.exe"
echo [2/4] Cap nhat pip...
"%VPY%" -m pip install --upgrade pip
if errorlevel 1 goto :error

echo [3/4] Cai PyTorch...
"%VPY%" -m pip install torch torchvision
if errorlevel 1 goto :error

echo [4/4] Cai thu vien AOI va Dobot...
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo Mo giao dien AOI...
set "PYTHONPATH=%CD%"
if /I "%~1"=="--robot-sim" (
  call "%~dp0run_ui_robot_all.bat"
) else if /I "%~1"=="--robot-real" (
  call "%~dp0run_ui_robot_all.bat" --real
) else if "%~1"=="" (
  "%VPY%" src\ui\tkinter_app.py --weights real_3class.pt
) else (
  "%VPY%" src\ui\tkinter_app.py %*
)
exit /b %errorlevel%

:error
echo Cai dat that bai. Kiem tra Python va ket noi Internet.
exit /b 1
