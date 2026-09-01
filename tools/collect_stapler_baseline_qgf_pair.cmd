@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0collect_stapler_baseline_qgf_pair.ps1" %*
exit /b %ERRORLEVEL%
