# Water-bottle-into-carton: ablation launcher (own file, touches no shared profile table).
#
# WHY A SEPARATE FILE
#   tools/collect_smolvla_task_rollouts.ps1 has no bottle profile (ValidateSet is
#   red_parcel|stapler|mug) and is held by another session with uncommitted edits.
#   The 08-22 legacy launcher collect_baseline_qgf_pair.ps1 hard-codes the old
#   dataset root, records no condition tag, and passes no tolerance/JOINT2 exports.
#   This file mirrors the AUDITED export line of collect_red_parcel_baseline_qgf_pair.ps1
#   (commit f1bc8c5) and makes every setting explicit.
#
# WHAT THE BOTTLE HAS ON DISK (2026-09-03)
#   bundle : /home/nvidia/work/telop/models/smolvla_onearm_20k_20260805   (the env default)
#   critic : /home/nvidia/work/telop/models/qgf/real_17_116_single_qcritic/critic_member_00.pt
#            trained 2026-08-17 on bottle rollouts ep17-116 (47S/53F). THOSE ROLLOUTS WERE
#            DELETED on 2026-08-28; the weights and the held-out feature cache (ep158-197,
#            Orin tmp/fig4/features) survive. The August 85%-vs-70% numbers are NOT reusable
#            as controls (sequential days, beta=2 by oral confirmation only, code state
#            unrecoverable), so every bottle arm here is collected fresh.
#
# DATASET ROOT
#   A NEW root, bottle_real_rollouts. The old qgf_real_rollouts is the emptied August tree
#   and must not receive new episodes (its episodes.jsonl still indexes ep0-241).
#
# ARMS (guidance_mode is read by the policy server once the qgf.py switch lands):
#   baseline            plain SmolVLA
#   qgf                 critic gradient guidance, coefficient 1/beta
#   qgf_random          matched-norm random direction (same |grad|, clip, 1/beta; random direction)
#   qgf_zero            B0-LM: critic forward AND backward run, applied guidance zeroed at qgf.py:138
#
# BETA
#   Mandatory. No bottle run at any beta other than 2 exists, and 2 rests on oral
#   confirmation. Do not default it. The bottle critic's gradient scale is being
#   measured offline (bottle_grad_scale.py); set -Beta from that, on the same
#   target-ratio basis as the other tasks.

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("baseline", "qgf", "qgf_random", "qgf_zero")]
    [string]$Arm,

    # Mandatory on purpose (see BETA above). Ignored for baseline.
    [double]$Beta = [double]::NaN,

    [ValidateRange(1, 1000)]
    [int]$EpisodeCount = 20,

    # Random-direction seed for qgf_random; recorded in notes.
    [int]$RandomSeed = 20260903,

    # Bottle start-pose overshoot tolerance. 0.0 = the August setting (the envelope in
    # configuration_armstrong_ros2.py IS the bottle's own, widened +15 deg on 08-21).
    # Raise only if code 23 (arm not at start pose) keeps tripping on reset.
    [double]$InitialPoseToleranceRad = 0.0,

    # Bottle keeps the config default. red_parcel needed 0.75; the bottle never did.
    [double]$Joint2MaxTargetErrorRad = 0.50,

    [string]$Notes = "",
    [string]$SshTarget = "armstrong-orin",
    [string]$RemoteProject = "/home/nvidia/work/telop/SmolVLA-with-QGF"
)

$ErrorActionPreference = "Stop"

# ---- task prompt: base64 + sha256 guard (the red-parcel launcher caught a one-codepoint
# ---- corruption in a handoff note this way; do the same here).
$taskBase64 = "5oqK55+/5rOJ5rC05pS+6L+b57q4566x6YeM44CC"   # 把矿泉水放进纸箱里。
$taskBytes = [Convert]::FromBase64String($taskBase64)
$task = [System.Text.Encoding]::UTF8.GetString($taskBytes)
$sha = [System.BitConverter]::ToString([System.Security.Cryptography.SHA256]::Create().ComputeHash($taskBytes)).Replace("-", "").ToLower()
if ($task.Length -ne 10 -or $sha -ne "faf63ad00eb909c4b935f75ffa0302b5a8dab225c2475a0150e75e32bfb5cd07") {
    throw "bottle prompt failed its guard: length=$($task.Length) (expect 10) sha256=$sha (expect faf63ad00eb909c4b935f75ffa0302b5a8dab225c2475a0150e75e32bfb5cd07). Do not run with a corrupted prompt."
}

$bundle   = "/home/nvidia/work/telop/models/smolvla_onearm_20k_20260805"
$critic   = "/home/nvidia/work/telop/models/qgf/real_17_116_single_qcritic/critic_member_00.pt"
$datasetRoot = "/home/nvidia/work/telop/bottle_real_rollouts"

# ---- refuse qgf_random / qgf_zero unless the DEPLOYED policy server actually reads the
# ---- guidance_mode env. Otherwise the server silently runs plain QGF and the arm becomes
# ---- a mislabelled second QGF cohort (the failure mode the ablation exists to avoid).
if ($Arm -in @("qgf_random", "qgf_zero")) {
    $probe = ssh $SshTarget "grep -q SMOLVLA_QGF_GUIDANCE_MODE '$RemoteProject/lerobot_robot_armstrong_ros2/src/lerobot_robot_armstrong_ros2/policy_server_qgf.py' && grep -q guidance_mode '$RemoteProject/qgf/src/guided_action_flow/guidance/qgf.py' && echo OK || echo MISSING"
    if ($probe -notmatch "OK") {
        throw "arm '$Arm' needs the guidance_mode switch deployed on the Orin (policy_server_qgf.py + qgf.py). It is not there yet; refusing to launch."
    }
}

$isGuided = $Arm -ne "baseline"
if ($isGuided -and ([double]::IsNaN($Beta) -or $Beta -le 0.0 -or [double]::IsInfinity($Beta))) {
    throw "-Beta is mandatory and must be a positive number for arm '$Arm'. There is no defensible default for the bottle."
}
$betaText = if ($isGuided) { $Beta.ToString([Globalization.CultureInfo]::InvariantCulture) } else { "0" }
$guidanceMode = switch ($Arm) { "qgf" { "critic" } "qgf_random" { "random_matched_norm" } "qgf_zero" { "zero" } default { "" } }
$runMode = if ($isGuided) { "qgf" } else { "baseline" }

$cohort = "bottle_ablation_normal"
$comparisonTag = "comparison_cohort=$cohort"
if ([string]::IsNullOrWhiteSpace($Notes)) { $Notes = "water bottle into carton; ablation" }
$notesForRun = "$Notes; condition=normal; $comparisonTag; task_profile=bottle; arm=$Arm; policy_mode=$runMode; guidance_mode=$guidanceMode; qgf_beta=$betaText; qgf_random_seed=$RandomSeed; qgf_critic=$critic; policy_bundle=$bundle"

$enc = [System.Text.Encoding]::UTF8
$notesBase64 = [Convert]::ToBase64String($enc.GetBytes($notesForRun))
$comparisonTagBase64 = [Convert]::ToBase64String($enc.GetBytes($comparisonTag))
$tolText = $InitialPoseToleranceRad.ToString([Globalization.CultureInfo]::InvariantCulture)
$j2Text  = $Joint2MaxTargetErrorRad.ToString([Globalization.CultureInfo]::InvariantCulture)

Write-Host "============================================================"
Write-Host "Water bottle into carton  -  arm: $Arm"
Write-Host "============================================================"
Write-Host "  task            : $task"
Write-Host "  bundle          : $bundle"
Write-Host "  critic          : $critic"
Write-Host "  dataset root    : $datasetRoot   (NEW; not the emptied qgf_real_rollouts)"
Write-Host "  run mode        : $runMode    guidance_mode=$guidanceMode"
Write-Host "  beta            : $betaText" -ForegroundColor $(if ($isGuided) { "Yellow" } else { "Gray" })
Write-Host "  episodes        : $EpisodeCount"
Write-Host "  pose tolerance  : $tolText rad    joint2 max target error: $j2Text rad"
Write-Host "  canonicalize    : true (the bottle predates the flag; smolvla_orin_env.sh never sets it, so the config default True applies. mug, stapler and red_parcel all set false.)"
Write-Host "  notes -> metadata: $notesForRun"
Write-Host "============================================================"
Write-Host ""

# Mirrors f1bc8c5's audited export line. SMOLVLA_JOINT2_MAX_TARGET_ERROR_RAD and
# SMOLVLA_INITIAL_POSE_TOLERANCE_RAD are exported explicitly: a missing export
# silently falls back to a different value and has already killed a cohort once.
$remote = "cd '$RemoteProject' && export SMOLVLA_TASK_B64='$taskBase64' QGF_NOTES_B64='$notesBase64' QGF_COMPARISON_TAG_B64='$comparisonTagBase64' QGF_RUN_MODE='$runMode' QGF_EPISODE_COUNT='$EpisodeCount' QGF_DATASET_ROOT='$datasetRoot' QGF_BETA='$betaText' SMOLVLA_QGF_GUIDANCE_MODE='$guidanceMode' SMOLVLA_QGF_RANDOM_SEED='$RandomSeed' SMOLVLA_QGF_CRITIC_PATH='$critic' SMOLVLA_ORIN_BUNDLE='$bundle' SMOLVLA_SERVER_MODEL_PATH='$bundle/checkpoint' SMOLVLA_EXPECTED_CHECKPOINT='$bundle/checkpoint' SMOLVLA_GRIPPER_OPEN_THRESHOLD='0.15' SMOLVLA_GRIPPER_CLOSE_THRESHOLD='0.85' SMOLVLA_GRIPPER_CONFIRMATION_FRAMES='5' SMOLVLA_CANONICALIZE_POLICY_OBSERVATION='true' SMOLVLA_INITIAL_POSE_TOLERANCE_RAD='$tolText' SMOLVLA_JOINT2_MAX_TARGET_ERROR_RAD='$j2Text' && ./tools/run_qgf_collection_session.sh"

ssh -t $SshTarget $remote
exit $LASTEXITCODE
