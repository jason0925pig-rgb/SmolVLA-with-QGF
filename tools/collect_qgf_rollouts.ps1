param(
    [string]$Task = "",
    [int]$EpisodeCount = 50,
    [string]$SshTarget = "armstrong-orin",
    [string]$DatasetRoot = "/home/nvidia/work/telop/qgf_real_rollouts",
    [string]$Notes = "",
    [string]$ComparisonTag = "",
    [ValidateSet("baseline", "qgf")]
    [string]$Mode = "baseline",
    [double]$Beta = 0.0
)

$ErrorActionPreference = "Stop"
if ($EpisodeCount -le 0) { throw "EpisodeCount must be positive." }
if ($Mode -eq "qgf" -and $Beta -le 0.0) {
    throw "For -Mode qgf, -Beta must be positive. The actual Q guidance coefficient is 1/Beta."
}
if ([string]::IsNullOrWhiteSpace($Task)) {
    $Task = [System.Text.Encoding]::UTF8.GetString(
        [Convert]::FromBase64String("5oqK55+/5rOJ5rC05pS+6L+b57q4566x6YeM44CC")
    )
}

$remoteProject = "/home/nvidia/work/telop/SmolVLA-with-QGF"
$taskBase64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($Task))
$betaText = $Beta.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$runNotes = if ($Mode -eq "qgf") {
    "$Notes; policy_mode=qgf; beta=$betaText; q_coefficient=$((1.0 / $Beta).ToString([System.Globalization.CultureInfo]::InvariantCulture))"
} else {
    "$Notes; policy_mode=baseline"
}
$notesBase64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($runNotes))
$comparisonTagBase64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($ComparisonTag))
$remoteCommand = @"
cd '$remoteProject' && export SMOLVLA_TASK_B64='$taskBase64' QGF_NOTES_B64='$notesBase64' QGF_COMPARISON_TAG_B64='$comparisonTagBase64' QGF_EPISODE_COUNT='$EpisodeCount' QGF_DATASET_ROOT='$DatasetRoot' QGF_RUN_MODE='$Mode' QGF_BETA='$betaText' && ./tools/run_qgf_collection_session.sh
"@.Trim()

Write-Host "Starting one persistent QGF collection session on $SshTarget."
Write-Host "ARM and MOVE are entered once; model, cameras, power and enable remain active between rounds."
Write-Host "Ctrl+C or a robot safety stop deletes the complete current episode before shutdown."
Write-Host "Task: $Task"
Write-Host "Target kept episodes: $EpisodeCount"
if (-not [string]::IsNullOrWhiteSpace($ComparisonTag)) {
    Write-Host "Comparison cohort tag: $ComparisonTag"
}
if ($Mode -eq "qgf") {
    Write-Host "Policy mode: QGF; beta=$betaText; actual Q coefficient=1/beta=$((1.0 / $Beta).ToString([System.Globalization.CultureInfo]::InvariantCulture))"
} else {
    Write-Host "Policy mode: baseline (no Q guidance)."
}
Write-Host ""

& ssh -tt $SshTarget $remoteCommand
$sshCode = $LASTEXITCODE
if ($sshCode -eq 0 -or $sshCode -eq 130) {
    Write-Host "QGF collection session ended; ordered robot shutdown was requested."
    exit 0
}
throw "QGF collection session failed with SSH exit code $sshCode. The current episode should have been deleted by remote cleanup."
