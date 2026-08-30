@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0my_run_pair_compare.ps1" %*
exit /b %ERRORLEVEL%
