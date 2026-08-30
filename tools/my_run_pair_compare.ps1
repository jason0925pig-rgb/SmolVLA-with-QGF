<#
Baseline-vs-QGF collection for ONE task under ONE named condition.

Why this exists instead of tools/collect_baseline_qgf_pair.ps1:
that script was written for the old water-bottle experiment.  It hardcodes its
task text and dataset root and exports NEITHER SMOLVLA_ORIN_BUNDLE NOR
SMOLVLA_QGF_CRITIC_PATH, so both fall back to tools/smolvla_orin_env.sh, whose
defaults are the old water-bottle checkpoint and the old water-bottle critic
(real_17_116_single_qcritic).  Running it unchanged for the mug would have
compared the wrong policy against the wrong critic on the wrong task and
written into the wrong directory.

policy_mode, and why it differs by mode -- verified against
run_qgf_collection_session.sh:

    finalize_current() { local notes="${NOTES}"
      if (( PAIRED_MODE )); then notes="${NOTES}; policy_mode=${CURRENT_MODE}"; fi

  * paired  : the remote APPENDS policy_mode, so this script must NOT write it,
              or the summariser's regex sees two and silently takes the first.
  * qgf     : the remote appends NOTHING, so this script MUST write it, or the
    baseline   summariser skips every episode and reports 0/0.

The summariser matches --tag as a PLAIN SUBSTRING of the notes.
#>
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("mug", "stapler", "red_parcel")]
    [string]$TaskProfile,

    [Parameter(Mandatory = $true)]
    [ValidateSet("normal", "medium", "dark", "distractor", "background")]
    [string]$Condition,

    # paired = alternate B/Q in one session (the clean design).
    # qgf / baseline = collect one arm only, e.g. when the other arm already
    # exists from an earlier session.  Cross-session arms are NOT paired and
    # are confounded with anything that drifted between sessions.
    [ValidateSet("paired", "qgf", "baseline")]
    [string]$Mode = "paired",

    [ValidateRange(1, 100000)]
    [int]$BaselineCount = 20,
    [ValidateRange(1, 100000)]
    [int]$QgfCount = 20,
    # single-arm modes only
    [ValidateRange(1, 100000)]
    [int]$EpisodeCount = 20,

    [double]$Beta = 2.0,
    [int]$TimeoutSeconds = 0,

    [ValidateSet("ask", "baseline", "qgf")]
    [string]$InitialMode = "ask",

    [ValidateSet("", "normal", "medium", "dark")]
    [string]$Lighting = "",

    # Override the derived dataset root / cohort tag.  Needed when a single-arm
    # run must land beside episodes that already exist elsewhere, so the
    # summariser can see both arms in one root.
    [string]$DatasetRoot = "",
    [string]$CohortTag = "",

    [double]$GripperOpenThreshold = 0.15,
    [double]$GripperCloseThreshold = 0.85,
    [int]$GripperConfirmationFrames = 5,

    [string]$SshTarget = "armstrong-orin"
)

$ErrorActionPreference = "Stop"
if ($Beta -le 0.0) { throw "Beta must be positive; the Q guidance coefficient is 1/Beta." }

# The mug 50 baseline rollouts were collected at 180 s, so comparisons use
# 180 s to stay comparable with them.
$timeoutDefaults = @{ mug = 180; stapler = 180; red_parcel = 180 }

# distractor/background are run under normal lighting so the scene is the only
# manipulated variable.
$lightingForCondition = @{
    normal = "normal"; medium = "medium"; dark = "dark"
    distractor = "normal"; background = "normal"
}

$profiles = @{
    mug = @{
        Task       = "把水杯放到紫色的箱子上"
        Bundle     = "/home/nvidia/work/telop/models/smolvla_20260827_mug_purple_box"
        CriticPath = "/home/nvidia/work/telop/models/qgf/mug_purple_box_single_q_45_5_20260829/critic_member_00.pt"
        RootPrefix = "mug"
        CanonicalizePolicyObservation = "false"
        InitialPoseToleranceRad       = "0.3490658503988659"
        Joint2MaxTargetErrorRad       = "0.50"
    }
    stapler = @{
        Task       = "把订书机放进快递纸盒"
        Bundle     = "/home/nvidia/work/telop/models/smolvla_20260827_stapler_into_box"
        CriticPath = "/home/nvidia/work/telop/models/qgf/stapler_into_box_single_q_45_5_20260830/critic_member_00.pt"
        RootPrefix = "stapler"
        CanonicalizePolicyObservation = "false"
        InitialPoseToleranceRad       = "0.3490658503988659"
        Joint2MaxTargetErrorRad       = "0.50"
    }
    red_parcel = @{
        Task       = "把箱子里的红色包裹拿出来放到桌子上。"
        Bundle     = "/home/nvidia/work/telop/models/smolvla_20260828_red_parcel_clean"
        CriticPath = ""   # no red parcel Q critic trained yet
        RootPrefix = "red_parcel"
        CanonicalizePolicyObservation = "false"
        InitialPoseToleranceRad       = "0.3490658503988659"
        Joint2MaxTargetErrorRad       = "0.75"
    }
}

$profile = $profiles[$TaskProfile]
if ($Mode -ne "baseline" -and [string]::IsNullOrWhiteSpace($profile.CriticPath)) {
    throw "No Q critic is configured for task profile '$TaskProfile'. Train and deploy one first."
}
if ($TimeoutSeconds -le 0) { $TimeoutSeconds = $timeoutDefaults[$TaskProfile] }
if ([string]::IsNullOrWhiteSpace($Lighting)) { $Lighting = $lightingForCondition[$Condition] }

$culture   = [System.Globalization.CultureInfo]::InvariantCulture
$openText  = $GripperOpenThreshold.ToString($culture)
$closeText = $GripperCloseThreshold.ToString($culture)
$betaText  = $Beta.ToString($culture)
$coefText  = (1.0 / $Beta).ToString($culture)

if ([string]::IsNullOrWhiteSpace($DatasetRoot)) {
    $DatasetRoot = "/home/nvidia/work/telop/$($profile.RootPrefix)_compare_$Condition"
}
if ([string]::IsNullOrWhiteSpace($CohortTag)) {
    # "cmp=" keeps the tag from colliding with fields such as task_profile=mug.
    $CohortTag = "cmp=$($profile.RootPrefix)_$Condition"
}

$criticName = if ([string]::IsNullOrWhiteSpace($profile.CriticPath)) { "none" }
              else { Split-Path -Leaf (Split-Path -Parent $profile.CriticPath) }
$bundleName = Split-Path -Leaf $profile.Bundle

# When the tag was overridden to reuse an existing field (e.g. "lighting=medium"
# so the summariser can see baselines collected earlier), do not repeat it.
$noteParts = @()
if ($CohortTag -notin @("lighting=$Lighting", "task_profile=$TaskProfile", "condition=$Condition")) {
    $noteParts += $CohortTag
}
$noteParts += @(
    "task_profile=$TaskProfile"
    "condition=$Condition"
    "lighting=$Lighting"
    "episode_timeout_s=$TimeoutSeconds"
    "qgf_beta=$betaText"
    "critic=$criticName"
    "bundle=$bundleName"
    "gripper_open_threshold=$openText"
    "gripper_close_threshold=$closeText"
    "gripper_confirmation_frames=$GripperConfirmationFrames"
    "initial_pose_tolerance_rad=$($profile.InitialPoseToleranceRad)"
    "joint2_max_target_error_rad=$($profile.Joint2MaxTargetErrorRad)"
)
# See the header: only the single-arm modes need policy_mode written here.
if ($Mode -ne "paired") { $noteParts += "policy_mode=$Mode" }
$runNotes = $noteParts -join "; "

function Resolve-InitialMode {
    if ($InitialMode -ne "ask") { return $InitialMode }
    while ($true) {
        $answer = (Read-Host "First round: Baseline or QGF? Enter B or Q").Trim().ToUpperInvariant()
        switch ($answer) {
            "B" { return "baseline" } "BASELINE" { return "baseline" }
            "Q" { return "qgf" }      "QGF"      { return "qgf" }
            default { Write-Host "Please enter B or Q." -ForegroundColor Yellow }
        }
    }
}

if ($Mode -eq "paired") {
    $resolvedInitialMode = Resolve-InitialMode
    $modeEnv = "QGF_RUN_MODE='paired' QGF_INITIAL_MODE='$resolvedInitialMode' " +
               "QGF_BASELINE_EPISODE_COUNT='$BaselineCount' QGF_QGF_EPISODE_COUNT='$QgfCount'"
    $targetText = "baseline=$BaselineCount  qgf=$QgfCount   first=$resolvedInitialMode"
} else {
    $modeEnv = "QGF_RUN_MODE='$Mode' QGF_EPISODE_COUNT='$EpisodeCount'"
    $targetText = "$Mode only, $EpisodeCount episodes (single arm - NOT paired)"
}

$taskBase64   = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($profile.Task))
$notesBase64  = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($runNotes))
$cohortBase64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($CohortTag))
$remoteProject = "/home/nvidia/work/telop/SmolVLA-with-QGF"

$remoteCommand = @"
cd '$remoteProject' && export SMOLVLA_TASK_B64='$taskBase64' QGF_NOTES_B64='$notesBase64' QGF_COMPARISON_TAG_B64='$cohortBase64' $modeEnv QGF_DATASET_ROOT='$DatasetRoot' QGF_BETA='$betaText' QGF_ROLLOUT_TIMEOUT_SECONDS='$TimeoutSeconds' SMOLVLA_ORIN_BUNDLE='$($profile.Bundle)' SMOLVLA_SERVER_MODEL_PATH='$($profile.Bundle)/checkpoint' SMOLVLA_EXPECTED_CHECKPOINT='$($profile.Bundle)/checkpoint' SMOLVLA_QGF_CRITIC_PATH='$($profile.CriticPath)' SMOLVLA_QGF_GRAD_CLIP_NORM='1.0' SMOLVLA_GRIPPER_OPEN_THRESHOLD='$openText' SMOLVLA_GRIPPER_CLOSE_THRESHOLD='$closeText' SMOLVLA_GRIPPER_CONFIRMATION_FRAMES='$GripperConfirmationFrames' SMOLVLA_CANONICALIZE_POLICY_OBSERVATION='$($profile.CanonicalizePolicyObservation)' SMOLVLA_INITIAL_POSE_TOLERANCE_RAD='$($profile.InitialPoseToleranceRad)' SMOLVLA_JOINT2_MAX_TARGET_ERROR_RAD='$($profile.Joint2MaxTargetErrorRad)' && ./tools/run_qgf_collection_session.sh
"@.Trim()

Write-Host "============================================================"
Write-Host "$TaskProfile / condition=$Condition / mode=$Mode"
Write-Host "============================================================"
Write-Host "Task            : $($profile.Task)"
Write-Host "Bundle          : $($profile.Bundle)"
Write-Host "Q critic        : $(if ($profile.CriticPath) { $profile.CriticPath } else { '<none>' })"
Write-Host "Dataset root    : $DatasetRoot"
Write-Host "Cohort tag      : $CohortTag"
Write-Host "Lighting        : $Lighting"
Write-Host "Targets         : $targetText"
Write-Host "QGF beta        : $betaText   (guidance coefficient = 1/beta = $coefText)"
Write-Host "Episode limit   : $TimeoutSeconds s"
Write-Host "Gripper filter  : open<=$openText, close>=$closeText, confirmation=$GripperConfirmationFrames frames"
Write-Host "Observation     : $(if ($profile.CanonicalizePolicyObservation -eq 'true') { 'canonical' } else { 'raw' })"
Write-Host "Start tolerance : $($profile.InitialPoseToleranceRad) rad"
Write-Host "J2 guard        : $($profile.Joint2MaxTargetErrorRad) rad"
if ($Mode -ne "paired") {
    Write-Host ""
    Write-Host "NOTE: single-arm run. The other arm must already exist in this same" -ForegroundColor Yellow
    Write-Host "      dataset root and match the cohort tag, or the printed table is" -ForegroundColor Yellow
    Write-Host "      one-sided. Cross-session arms are not paired." -ForegroundColor Yellow
}
Write-Host ""
Write-Host "After each S/F/D label the remote prints both success rates."
Write-Host "ARM and MOVE are entered once; power stays on between rounds."
Write-Host "============================================================"
Write-Host ""

& ssh -tt $SshTarget $remoteCommand
$sshCode = $LASTEXITCODE
if ($sshCode -eq 0 -or $sshCode -eq 130) {
    Write-Host "Session ended; ordered robot shutdown was requested."
    exit 0
}
throw "Session failed with SSH exit code $sshCode."
