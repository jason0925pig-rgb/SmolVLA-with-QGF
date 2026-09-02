param(
    [ValidateSet("normal", "light", "distractor")]
    [string]$Condition = "normal",
    [ValidateRange(1, 100000)]
    [int]$BaselineEpisodeCount = 20,
    [ValidateRange(1, 100000)]
    [int]$QgfEpisodeCount = 20,
    [ValidateRange(0, 100000)]
    [int]$ExistingBaselineCount = 0,
    [ValidateRange(0, 100000)]
    [int]$ExistingQgfCount = 0,
    [ValidateSet("ask", "baseline", "qgf")]
    [string]$InitialMode = "ask",
    [ValidateRange(0.000001, 1000000.0)]
    [double]$Beta = 0.35,
    [string]$Notes = "",
    [string]$SshTarget = "armstrong-orin"
)

$ErrorActionPreference = "Stop"
$task = [System.Text.Encoding]::UTF8.GetString(
    [Convert]::FromBase64String("5oqK6K6i5Lmm5py65pS+6L+b5b+r6YCS57q455uS")
)
$datasetRoot = "/home/nvidia/work/telop/stapler_real_rollouts"
$bundle = "/home/nvidia/work/telop/models/smolvla_20260827_stapler_into_box"
$critic = "/home/nvidia/work/telop/models/qgf/stapler_into_box_single_q_45_5_20260830/critic_member_00.pt"
$comparisonTag = if ($Condition -eq "normal") { "lighting=normal" } else { "comparison_cohort=stapler_$Condition" }
$conditionNote = @{
    normal = "lighting=normal"
    light = "lighting_perturbation"
    distractor = "distractor_perturbation"
}[$Condition]
if ([string]::IsNullOrWhiteSpace($Notes)) { $Notes = "stapler into box; $conditionNote" }
$notesForRun = "$Notes; condition=$Condition; comparison_cohort=stapler_$Condition"

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
$comparisonTagBase64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($comparisonTag))
$betaText = $Beta.ToString($culture)
$coefficient = (1.0 / $Beta).ToString($culture)
$remoteProject = "/home/nvidia/work/telop/SmolVLA-with-QGF"
$remoteCommand = @"
cd '$remoteProject' && export SMOLVLA_TASK_B64='$taskBase64' QGF_NOTES_B64='$notesBase64' QGF_COMPARISON_TAG_B64='$comparisonTagBase64' QGF_RUN_MODE='paired' QGF_INITIAL_MODE='$resolvedInitialMode' QGF_BASELINE_EPISODE_COUNT='$BaselineEpisodeCount' QGF_QGF_EPISODE_COUNT='$QgfEpisodeCount' QGF_INITIAL_SAVED_BASELINE='$ExistingBaselineCount' QGF_INITIAL_SAVED_QGF='$ExistingQgfCount' QGF_DATASET_ROOT='$datasetRoot' QGF_BETA='$betaText' SMOLVLA_QGF_CRITIC_PATH='$critic' SMOLVLA_ORIN_BUNDLE='$bundle' SMOLVLA_SERVER_MODEL_PATH='$bundle/checkpoint' SMOLVLA_EXPECTED_CHECKPOINT='$bundle/checkpoint' SMOLVLA_GRIPPER_OPEN_THRESHOLD='0.15' SMOLVLA_GRIPPER_CLOSE_THRESHOLD='0.85' SMOLVLA_GRIPPER_CONFIRMATION_FRAMES='5' SMOLVLA_CANONICALIZE_POLICY_OBSERVATION='false' SMOLVLA_INITIAL_POSE_TOLERANCE_RAD='0.3490658503988659' && ./tools/run_qgf_collection_session.sh
"@
$remoteCommand = $remoteCommand.Trim()

Write-Host "============================================================"
Write-Host "Interactive paired Stapler-into-Box Baseline/QGF collection"
Write-Host "Condition: $Condition (comparison filter: $comparisonTag)"
Write-Host "Baseline target: $BaselineEpisodeCount; QGF target: $QgfEpisodeCount"
Write-Host "Previously saved in this cohort: baseline=$ExistingBaselineCount; QGF=$ExistingQgfCount"
Write-Host "First policy: $resolvedInitialMode"
Write-Host "QGF beta=$Beta; actual Q coefficient=1/beta=$coefficient"
Write-Host "After each S/F/D label, only this condition's baseline/QGF success rates are printed."
Write-Host "Then choose B=baseline, Q/G=QGF, or X=finish; after B/Q/G press Enter to start."
Write-Host "============================================================"

& ssh -tt $SshTarget $remoteCommand
$sshCode = $LASTEXITCODE
if ($sshCode -eq 0 -or $sshCode -eq 130) { exit 0 }
throw "Stapler paired collection failed with SSH exit code $sshCode."
