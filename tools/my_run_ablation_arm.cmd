@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0my_run_ablation_arm.ps1" %*
exit /b %ERRORLEVEL%
