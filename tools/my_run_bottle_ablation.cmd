@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0my_run_bottle_ablation.ps1" %*
exit /b %ERRORLEVEL%
