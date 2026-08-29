param(
    [ValidateSet("red_parcel", "stapler", "mug")]
    [string]$TaskProfile,
    [int]$EpisodeCount = 1,
    [string]$Notes = "",
    [string]$SshTarget = "armstrong-orin",
    [double]$GripperOpenThreshold = 0.15,
    [double]$GripperCloseThreshold = 0.85,
    [int]$GripperConfirmationFrames = 5
)

$ErrorActionPreference = "Stop"
if ($EpisodeCount -le 0) { throw "EpisodeCount must be positive." }
if ($GripperOpenThreshold -lt 0.0 -or $GripperCloseThreshold -gt 1.0 -or $GripperOpenThreshold -ge $GripperCloseThreshold) {
    throw "Require 0 <= GripperOpenThreshold < GripperCloseThreshold <= 1."
}
if ($GripperConfirmationFrames -lt 1) { throw "GripperConfirmationFrames must be at least 1." }

$profiles = @{
    red_parcel = @{
        Task = "把箱子里的红色包裹拿出来放到桌子上。"
        Bundle = "/home/nvidia/work/telop/models/smolvla_20260828_red_parcel_clean"
        DatasetRoot = "/home/nvidia/work/telop/red_parcel_real_rollouts"
        # The clean checkpoint was trained on the controller's raw negative-J5
        # representation. Safety checks still canonicalize wraparound joints
        # internally; only the observation sent to the policy stays raw.
        CanonicalizePolicyObservation = "false"
        # Permit small manual scene-reset drift (5 degrees) without accepting
        # the much wider 20-degree envelope used by the other task profiles.
        InitialPoseToleranceRad = "0.0872664625997165"
        Joint2MaxTargetErrorRad = "0.75"
    }
    stapler = @{
        Task = "把订书机放进快递纸盒"
        Bundle = "/home/nvidia/work/telop/models/smolvla_20260827_stapler_into_box"
        DatasetRoot = "/home/nvidia/work/telop/stapler_real_rollouts"
        CanonicalizePolicyObservation = "false"
        InitialPoseToleranceRad = "0.3490658503988659"
    }
    mug = @{
        Task = "把水杯放到紫色的箱子上"
        Bundle = "/home/nvidia/work/telop/models/smolvla_20260827_mug_purple_box"
        DatasetRoot = "/home/nvidia/work/telop/mug_purple_box_real_rollouts"
        CanonicalizePolicyObservation = "false"
        InitialPoseToleranceRad = "0.3490658503988659"
    }
}

$profile = $profiles[$TaskProfile]
$culture = [System.Globalization.CultureInfo]::InvariantCulture
$openText = $GripperOpenThreshold.ToString($culture)
$closeText = $GripperCloseThreshold.ToString($culture)
$taskBase64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($profile.Task))
$runNotes = "$Notes; task_profile=$TaskProfile; policy_mode=baseline; gripper_open_threshold=$openText; gripper_close_threshold=$closeText; gripper_confirmation_frames=$GripperConfirmationFrames"
$notesBase64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($runNotes))
$comparisonTagBase64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($TaskProfile))
$canonicalizePolicyObservation = if ($profile.ContainsKey("CanonicalizePolicyObservation")) {
    $profile.CanonicalizePolicyObservation
} else {
    "true"
}
$initialPoseToleranceRad = if ($profile.ContainsKey("InitialPoseToleranceRad")) {
    $profile.InitialPoseToleranceRad
} else {
    "0.0"
}
$joint2MaxTargetErrorRad = if ($profile.ContainsKey("Joint2MaxTargetErrorRad")) {
    $profile.Joint2MaxTargetErrorRad
} else {
    "0.50"
}
$remoteProject = "/home/nvidia/work/telop/SmolVLA-with-QGF"
$remoteCommand = @"
cd '$remoteProject' && export SMOLVLA_TASK_B64='$taskBase64' QGF_NOTES_B64='$notesBase64' QGF_COMPARISON_TAG_B64='$comparisonTagBase64' QGF_EPISODE_COUNT='$EpisodeCount' QGF_DATASET_ROOT='$($profile.DatasetRoot)' QGF_RUN_MODE='baseline' QGF_BETA='0' SMOLVLA_ORIN_BUNDLE='$($profile.Bundle)' SMOLVLA_SERVER_MODEL_PATH='$($profile.Bundle)/checkpoint' SMOLVLA_EXPECTED_CHECKPOINT='$($profile.Bundle)/checkpoint' SMOLVLA_GRIPPER_OPEN_THRESHOLD='$openText' SMOLVLA_GRIPPER_CLOSE_THRESHOLD='$closeText' SMOLVLA_GRIPPER_CONFIRMATION_FRAMES='$GripperConfirmationFrames' SMOLVLA_CANONICALIZE_POLICY_OBSERVATION='$canonicalizePolicyObservation' SMOLVLA_INITIAL_POSE_TOLERANCE_RAD='$initialPoseToleranceRad' SMOLVLA_JOINT2_MAX_TARGET_ERROR_RAD='$joint2MaxTargetErrorRad' && ./tools/run_qgf_collection_session.sh
"@.Trim()

Write-Host "SmolVLA task rollout: $TaskProfile"
Write-Host "Task: $($profile.Task)"
Write-Host "Checkpoint bundle: $($profile.Bundle)"
Write-Host "Dataset root: $($profile.DatasetRoot)"
Write-Host "Gripper filter: open<=$openText, close>=$closeText, confirmation=$GripperConfirmationFrames frames"
Write-Host "Policy joint observation coordinates: $(if ($canonicalizePolicyObservation -eq 'true') { 'canonical' } else { 'raw' })"
Write-Host "Initial pose tolerance: $initialPoseToleranceRad rad (20 degrees for mug/stapler)"
Write-Host "Joint 2 target-error threshold: $joint2MaxTargetErrorRad rad"
Write-Host "ARM and MOVE will be requested by the remote attended rollout."

& ssh -tt $SshTarget $remoteCommand
$sshCode = $LASTEXITCODE
if ($sshCode -eq 0 -or $sshCode -eq 130) { exit 0 }
throw "Task rollout failed with SSH exit code $sshCode."
