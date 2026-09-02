@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0my_run_stapler_qgf_normal.ps1" %*
exit /b %ERRORLEVEL%
