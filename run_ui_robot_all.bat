@echo off
REM UI tu khoi dong worker robot. Mac dinh dung robot gia lap.
setlocal
cd /d "%~dp0"

if /I "%~1"=="--real" (
  call "%~dp0run_ui_robot.bat" --robot-autostart real
) else (
  call "%~dp0run_ui_robot.bat" --robot-autostart sim
)
