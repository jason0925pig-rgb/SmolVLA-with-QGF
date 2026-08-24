param(
    [string]$Task = "",
    [string]$SshTarget = "armstrong-orin",
    [string]$DatasetRoot = "/home/nvidia/work/telop/qgf_real_rollouts",
    # Keep this aligned with the currently active real-robot perturbation
    # cohort.  Existing episodes 221+ use this plain note tag (without a
    # separate paired_cohort field), so the default live statistics include
    # both those earlier saved rounds and the new paired rounds.
    [string]$Notes = "distractor_perturbation",
    [ValidateRange(1, 100000)]
    [int]$BaselineEpisodeCount = 1,
    [ValidateRange(1, 100000)]
    [int]$QgfEpisodeCount = 1,
    [ValidateSet("ask", "baseline", "qgf")]
    [string]$InitialMode = "ask",
    # Leave empty to include every episode tagged with $Notes, including the
    # earlier rounds tagged with the same perturbation. Set this only when a
    # later experiment needs an isolated subset.
    [string]$PairCohort = "",
    [double]$Beta = 2.0
)

$ErrorActionPreference = "Stop"
if ($Beta -le 0.0) {
    throw "Beta must be positive. The actual Q guidance coefficient is 1/Beta."
}
function Resolve-InitialMode {
    if ($InitialMode -ne "ask") {
        return $InitialMode
    }
    while ($true) {
        $answer = (Read-Host "First round: Baseline or QGF? Enter B or Q").Trim().ToUpperInvariant()
        switch ($answer) {
            "B" { return "baseline" }
            "BASELINE" { return "baseline" }
            "Q" { return "qgf" }
            "QGF" { return "qgf" }
            default { Write-Host "Please enter B (baseline first) or Q (QGF first)." -ForegroundColor Yellow }
        }
    }
}

if ([string]::IsNullOrWhiteSpace($Task)) {
    $Task = [System.Text.Encoding]::UTF8.GetString(
        [Convert]::FromBase64String("5oqK55+/5rOJ5rC05pS+6L+b57q4566x6YeM44CC")
    )
}

$resolvedInitialMode = Resolve-InitialMode
$remoteProject = "/home/nvidia/work/telop/SmolVLA-with-QGF"
if ([string]::IsNullOrWhiteSpace($PairCohort)) {
    # Historical runs predate paired_cohort.  Match on the shared experiment
    # note so they are included in the live success-rate table.
    $notesForRun = $Notes
    $comparisonTag = $Notes
    $cohortDescription = "all episodes tagged '$Notes' (including earlier saved rounds)"
} else {
    $notesForRun = "$Notes; paired_cohort=$PairCohort"
    $comparisonTag = "paired_cohort=$PairCohort"
    $cohortDescription = "isolated cohort '$PairCohort'"
}
$taskBase64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($Task))
$notesBase64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($notesForRun))
$comparisonTagBase64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($comparisonTag))
$betaText = $Beta.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$coefficient = (1.0 / $Beta).ToString([System.Globalization.CultureInfo]::InvariantCulture)
$remoteCommand = @"
cd '$remoteProject' && export SMOLVLA_TASK_B64='$taskBase64' QGF_NOTES_B64='$notesBase64' QGF_COMPARISON_TAG_B64='$comparisonTagBase64' QGF_RUN_MODE='paired' QGF_INITIAL_MODE='$resolvedInitialMode' QGF_BASELINE_EPISODE_COUNT='$BaselineEpisodeCount' QGF_QGF_EPISODE_COUNT='$QgfEpisodeCount' QGF_DATASET_ROOT='$DatasetRoot' QGF_BETA='$betaText' && ./tools/run_qgf_collection_session.sh
"@.Trim()

Write-Host "============================================================"
Write-Host "Interactive paired Baseline/QGF real-robot collection"
Write-Host "Baseline target: $BaselineEpisodeCount; QGF target: $QgfEpisodeCount"
Write-Host "First policy: $resolvedInitialMode"
Write-Host "QGF beta=$Beta; actual Q coefficient=1/beta=$coefficient"
Write-Host "Notes: $Notes"
Write-Host "Statistics scope: $cohortDescription"
Write-Host "After each F/S/D label, the remote terminal prints both success rates."
Write-Host "Then choose B=baseline, Q/G=QGF, or X=finish; after B/Q/G press Enter to start."
Write-Host "ARM and MOVE are entered once. Power/enable stay on between rounds."
Write-Host "============================================================"
Write-Host ""

& ssh -tt $SshTarget $remoteCommand
$sshCode = $LASTEXITCODE
if ($sshCode -eq 0 -or $sshCode -eq 130) {
    Write-Host "Paired collection session ended; ordered robot shutdown was requested."
    exit 0
}
throw "Paired collection session failed with SSH exit code $sshCode. The current episode should have been deleted by remote cleanup."
