param(
    [int]$EpisodeCount = 5,
    [string]$SshTarget = "armstrong-orin",
    # Keep parcel rollouts out of the historical water/QGF dataset root.
    [string]$DatasetRoot = "/home/nvidia/work/telop/parcel_real_rollouts",
    [string]$Notes = "parcel_task; new_ckpt=015000",
    [ValidateSet("baseline", "qgf")]
    [string]$Mode = "baseline",
    [double]$Beta = 0.0,
    # Selected checkpoint 015000 is the default; -UseAltCheckpoint switches to
    # the 020000 alternative stored inside the same bundle.
    [switch]$UseAltCheckpoint
)

# Parcel-sorting task entry point (new 50-demo checkpoint, 2026-08-26).
#
# Thin wrapper over collect_qgf_rollouts.ps1: it only fixes the task text,
# the notes tag and the model bundle for the parcel task, then delegates
# everything (session lifecycle, ARM/MOVE prompts, S/F/D labelling, safety
# cleanup) to the verified collector. The old water-bottle invocations are
# untouched: run collect_qgf_rollouts.ps1 directly for that task.

$ErrorActionPreference = "Stop"

$collector = Join-Path $PSScriptRoot "collect_qgf_rollouts.ps1"
if (-not (Test-Path -LiteralPath $collector -PathType Leaf)) {
    throw "Base collector is missing: $collector"
}

$bundle = "/home/nvidia/work/telop/models/smolvla_20260825_parcel_50"
# Frozen task text (stored as base64 so no console codepage can corrupt it):
# 把箱子里面的红色包裹放到箱子外面左侧，绿色包裹放到箱子外面右侧
$taskB64 = "5oqK566x5a2Q6YeM6Z2i55qE57qi6Imy5YyF6KO55pS+5Yiw566x5a2Q5aSW6Z2i5bem5L6n77yM57u/6Imy5YyF6KO55pS+5Yiw566x5a2Q5aSW6Z2i5Y+z5L6n"
$task = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($taskB64))

$forward = @{
    Task = $task
    EpisodeCount = $EpisodeCount
    SshTarget = $SshTarget
    DatasetRoot = $DatasetRoot
    Notes = $Notes
    Mode = $Mode
    ModelBundle = $bundle
}
if ($Beta -gt 0.0) { $forward.Beta = $Beta }
if ($UseAltCheckpoint) {
    $forward.ModelCheckpointPath = "$bundle/checkpoint_alt_020000"
    $forward.Notes = ($Notes -replace "new_ckpt=015000", "new_ckpt=020000_alt")
}

Write-Host "Parcel-sorting rollout: checkpoint $(if ($UseAltCheckpoint) { '020000 (alt)' } else { '015000 (selected)' }); episodes=$EpisodeCount; mode=$Mode"
& $collector @forward
exit $LASTEXITCODE
