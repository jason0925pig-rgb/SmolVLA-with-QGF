@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0my_run_task_rollout.ps1" %*
exit /b %ERRORLEVEL%
