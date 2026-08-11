param(
    [string]$Task = "",
    [string]$SshTarget = "armstrong-orin"
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($Task)) {
    $Task = [System.Text.Encoding]::UTF8.GetString(
        [Convert]::FromBase64String("5oqK55+/5rOJ5rC05pS+6L+b57q4566x6YeM44CC")
    )
}
$project = "/home/nvidia/work/telop/SmolVLA-with-QGF"
$taskBytes = [System.Text.Encoding]::UTF8.GetBytes($Task)
$taskBase64 = [Convert]::ToBase64String($taskBytes)
$remoteCommand = "cd '$project' && export SMOLVLA_TASK_B64='$taskBase64' && ./tools/run_smolvla_orin.sh"

Write-Host "Starting the staged SmolVLA launcher on $SshTarget."
Write-Host "The robot does not move until the remote prompt explicitly says so."
Write-Host "Task: $Task"
Write-Host ""

& ssh -tt $SshTarget $remoteCommand
$sshCode = $LASTEXITCODE
if ($sshCode -eq 0 -or $sshCode -eq 130) {
    Write-Host "SmolVLA session stopped. The remote cleanup sequence was requested."
    exit 0
}
throw "SmolVLA remote launcher failed with SSH exit code $sshCode."
