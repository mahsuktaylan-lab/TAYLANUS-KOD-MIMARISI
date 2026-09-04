@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_windows.ps1" %*
if errorlevel 1 (
  echo.
  echo TAY installation failed.
  exit /b 1
)
endlocal
