param(
    [ValidateSet("normal", "light", "distractor")]
    [string]$Condition = "normal",
    [ValidateRange(1, 100000)]
    [int]$BaselineEpisodeCount = 1,
    [ValidateRange(1, 100000)]
    [int]$QgfEpisodeCount = 1,
    [ValidateSet("ask", "baseline", "qgf")]
    [string]$InitialMode = "ask",
    [ValidateRange(0.000001, 1000000.0)]
    [double]$Beta = 0.5,
    [string]$Notes = "",
    [string]$SshTarget = "armstrong-orin"
)

$ErrorActionPreference = "Stop"

# Keep source ASCII-only so Windows PowerShell 5.1 cannot misdecode the task
# string when the launcher is checked out without a UTF-8 BOM.
$task = [System.Text.Encoding]::UTF8.GetString(
    [Convert]::FromBase64String("5oqK5rC05p2v5pS+5Yiw57Sr6Imy55qE566x5a2Q5LiK")
)
$datasetRoot = "/home/nvidia/work/telop/mug_purple_box_real_rollouts"
$bundle = "/home/nvidia/work/telop/models/smolvla_20260827_mug_purple_box"
$critic = "/home/nvidia/work/telop/models/qgf/mug_purple_box_single_q_45_5_20260829/critic_member_00.pt"
$comparisonTag = "mug_$Condition"
$conditionNote = @{
    normal = "normal"
    light = "lighting_perturbation"
    distractor = "distractor_perturbation"
}[$Condition]
if ([string]::IsNullOrWhiteSpace($Notes)) {
    $Notes = "mug purple box; $conditionNote"
}
$notesForRun = "$Notes; condition=$Condition; comparison_cohort=$comparisonTag"

function Resolve-InitialMode {
    if ($InitialMode -ne "ask") { return $InitialMode }
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

$resolvedInitialMode = Resolve-InitialMode
$culture = [System.Globalization.CultureInfo]::InvariantCulture
$taskBase64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($task))
$notesBase64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($notesForRun))
$comparisonTagBase64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes("comparison_cohort=$comparisonTag"))
$betaText = $Beta.ToString($culture)
$coefficient = (1.0 / $Beta).ToString($culture)
$remoteProject = "/home/nvidia/work/telop/SmolVLA-with-QGF"
$remoteCommand = @"
cd '$remoteProject' && export SMOLVLA_TASK_B64='$taskBase64' QGF_NOTES_B64='$notesBase64' QGF_COMPARISON_TAG_B64='$comparisonTagBase64' QGF_RUN_MODE='paired' QGF_INITIAL_MODE='$resolvedInitialMode' QGF_BASELINE_EPISODE_COUNT='$BaselineEpisodeCount' QGF_QGF_EPISODE_COUNT='$QgfEpisodeCount' QGF_DATASET_ROOT='$datasetRoot' QGF_BETA='$betaText' SMOLVLA_QGF_CRITIC_PATH='$critic' SMOLVLA_ORIN_BUNDLE='$bundle' SMOLVLA_SERVER_MODEL_PATH='$bundle/checkpoint' SMOLVLA_EXPECTED_CHECKPOINT='$bundle/checkpoint' SMOLVLA_GRIPPER_OPEN_THRESHOLD='0.15' SMOLVLA_GRIPPER_CLOSE_THRESHOLD='0.85' SMOLVLA_GRIPPER_CONFIRMATION_FRAMES='5' SMOLVLA_CANONICALIZE_POLICY_OBSERVATION='false' SMOLVLA_INITIAL_POSE_TOLERANCE_RAD='0.3490658503988659' && ./tools/run_qgf_collection_session.sh
"@
$remoteCommand = $remoteCommand.Trim()

Write-Host "============================================================"
Write-Host "Interactive paired Mug-to-Purple-Box Baseline/QGF collection"
Write-Host "Condition: $Condition (isolated comparison cohort: $comparisonTag)"
Write-Host "Baseline target: $BaselineEpisodeCount; QGF target: $QgfEpisodeCount"
Write-Host "First policy: $resolvedInitialMode"
Write-Host "QGF beta=$Beta; actual Q coefficient=1/beta=$coefficient"
Write-Host "Notes: $notesForRun"
Write-Host "After each S/F/D label, only this condition's baseline/QGF success rates are printed."
Write-Host "Then choose B=baseline, Q/G=QGF, or X=finish; after B/Q/G press Enter to start."
Write-Host "ARM and MOVE are entered once. Power/enable stay on between rounds."
Write-Host "============================================================"

& ssh -tt $SshTarget $remoteCommand
$sshCode = $LASTEXITCODE
if ($sshCode -eq 0 -or $sshCode -eq 130) { exit 0 }
throw "Mug paired collection failed with SSH exit code $sshCode."
