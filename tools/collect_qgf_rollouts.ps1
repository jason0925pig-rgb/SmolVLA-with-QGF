param(
    [string]$Task = "",
    [int]$EpisodeCount = 50,
    [string]$SshTarget = "armstrong-orin",
    [string]$DatasetRoot = "/home/nvidia/work/telop/qgf_real_rollouts",
    [string]$Notes = ""
)

$ErrorActionPreference = "Stop"
if ($EpisodeCount -le 0) { throw "EpisodeCount must be positive." }
if ([string]::IsNullOrWhiteSpace($Task)) {
    $Task = [System.Text.Encoding]::UTF8.GetString(
        [Convert]::FromBase64String("5oqK55+/5rOJ5rC05pS+6L+b57q4566x6YeM44CC")
    )
}

$remoteProject = "/home/nvidia/work/telop/SmolVLA-with-QGF"
$taskBase64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($Task))
$notesBase64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($Notes))
$batchId = Get-Date -Format "yyyyMMdd_HHmmss"
$saved = 0
$attempt = 0

Write-Host "============================================================"
Write-Host "QGF real-robot rollout collector"
Write-Host "Task          : $Task"
Write-Host "Target kept   : $EpisodeCount episode(s)"
Write-Host "SSH target    : $SshTarget"
Write-Host "Dataset root  : $DatasetRoot"
Write-Host ""
Write-Host "Each round still requires ARM and MOVE confirmation."
Write-Host "After the robot stops: S saves success, F saves failure, D deletes the round."
Write-Host "Discarded and failed-to-start rounds do not consume an episode number."
Write-Host "============================================================"

while ($saved -lt $EpisodeCount) {
    $attempt++
    $token = "${batchId}_attempt_$($attempt.ToString('D4'))"
    $staging = "$DatasetRoot/.staging/$token"
    $remoteRun = "cd '$remoteProject' && export QGF_EPISODE_DIR='$staging' SMOLVLA_TASK_B64='$taskBase64' && ./tools/run_qgf_episode.sh"
    Write-Host ""
    Write-Host "--- Attempt $attempt / kept $saved of $EpisodeCount ---"
    Write-Host "The robot remains gated until you type ARM and then MOVE."
    & ssh -tt $SshTarget $remoteRun
    $runCode = $LASTEXITCODE

    if ($runCode -ne 0 -and $runCode -ne 130) {
        Write-Warning "Rollout launcher ended with code $runCode; this attempt will be deleted."
        $discard = "cd '$remoteProject' && source tools/smolvla_orin_env.sh && `"`${SMOLVLA_ORIN_VENV}/bin/python`" tools/finalize_qgf_episode.py --staging '$staging' --dataset-root '$DatasetRoot' --outcome discard --task x"
        & ssh $SshTarget $discard
        $again = (Read-Host "Press Enter to retry, or type Q to stop").Trim().ToUpperInvariant()
        if ($again -eq "Q") { break }
        continue
    }

    do {
        $choice = (Read-Host "Episode outcome: S=save success, F=save failure, D=discard/delete, Q=discard and stop").Trim().ToUpperInvariant()
    } while ($choice -notin @("S", "F", "D", "Q"))
    $outcome = switch ($choice) {
        "S" { "success" }
        "F" { "failure" }
        default { "discard" }
    }
    $remoteFinalize = @"
cd '$remoteProject' && source tools/smolvla_orin_env.sh && TASK=`$(printf '%s' '$taskBase64' | base64 --decode) && NOTES=`$(printf '%s' '$notesBase64' | base64 --decode) && "`${SMOLVLA_ORIN_VENV}/bin/python" tools/finalize_qgf_episode.py --staging '$staging' --dataset-root '$DatasetRoot' --outcome '$outcome' --task "`$TASK" --notes "`$NOTES"
"@.Trim()
    & ssh $SshTarget $remoteFinalize
    if ($LASTEXITCODE -ne 0) {
        throw "QGF episode finalization failed; staging data remains at $staging"
    }
    if ($outcome -ne "discard") {
        $saved++
        Write-Host "Kept episodes: $saved / $EpisodeCount"
    }
    if ($choice -eq "Q") { break }
}

Write-Host "QGF_COLLECTION_FINISHED kept=$saved attempts=$attempt dataset=$DatasetRoot"
