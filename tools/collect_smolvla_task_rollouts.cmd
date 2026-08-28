@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0collect_smolvla_task_rollouts.ps1" %*
exit /b %ERRORLEVEL%
