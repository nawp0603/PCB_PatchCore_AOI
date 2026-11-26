@echo off
REM Launcher duy nhat cho tram AOI: mac dinh robot gia lap, --real dung Dobot that.
setlocal
cd /d "%~dp0"

if /I "%~1"=="--real" (
  call "%~dp0run_robot_check.bat"
  if errorlevel 1 goto :error
  call "%~dp0run_ui_robot_all.bat" --real
) else (
  call "%~dp0run_ui_robot_all.bat"
)
exit /b %errorlevel%

:error
echo Kiem tra cau hinh robot that bai. UI khong duoc khoi dong.
exit /b 1
