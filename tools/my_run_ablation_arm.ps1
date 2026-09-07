# Ablation arm launcher for ONE task and ONE arm per session.
#
#   .\tools\my_run_ablation_arm.cmd -Task red_parcel -Arm qgf_random -Beta 0.5
#   .\tools\my_run_ablation_arm.cmd -Task red_parcel -Arm qgf_zero   -Beta 0.5
#   .\tools\my_run_ablation_arm.cmd -Task bottle     -Arm baseline
#
# One arm per session on purpose: tools/run_qgf_collection_session.sh only knows the
# modes baseline|qgf|paired and counts a third mode as QGF (:319/:328/:590/:599), and
# its baseline branch does not unset arm env vars (:361-364). Running each arm as its
# own single-mode session sidesteps both. Arms are told apart by the notes fields
# arm= / guidance_mode= / comparison_cohort= written into every episode's metadata.
#
# Per-task literals come from the audited sources:
#   red_parcel : collect_smolvla_task_rollouts.ps1 profile and the f1bc8c5 paired launcher
#                (canonicalize=false because the clean checkpoint was trained on the raw
#                negative-J5 representation; JOINT2 0.75; start tolerance 5 deg)
#   bottle     : smolvla_orin_env.sh defaults (bundle, critic) and configuration_armstrong_ros2.py
#                defaults (canonicalize=true, JOINT2 0.50, the bottle's own start envelope)
#
# BETA is mandatory for every guided arm. No default. The only defensible values are the
# ones decided per task in Ablation最终方案_20260903.md.

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("red_parcel", "bottle")]
    [string]$Task,

    [Parameter(Mandatory = $true)]
    [ValidateSet("baseline", "qgf", "qgf_random", "qgf_zero")]
    [string]$Arm,

    [double]$Beta = [double]::NaN,

    [ValidateRange(1, 1000)]
    [int]$EpisodeCount = 20,

    [int]$RandomSeed = 20260903,

    # Override the per-task defaults only when the guard keeps tripping on reset.
    [double]$InitialPoseToleranceRad = [double]::NaN,
    [double]$Joint2MaxTargetErrorRad = [double]::NaN,

    [string]$CohortSuffix = "ablation_normal",
    [string]$Notes = "",
    [string]$SshTarget = "armstrong-orin",
    [string]$RemoteProject = "/home/nvidia/work/telop/SmolVLA-with-QGF",
    # Verify every precondition and print the remote command, but start nothing.
    # Running this launcher without it IS starting a rollout on the real robot;
    # that has been mistaken for a test before.
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$profiles = @{
    red_parcel = @{
        TaskBase64   = "5oqK566x5a2Q6YeM55qE57qi6Imy5YyF6KO55ou/5Ye65p2l5pS+5Yiw5qGM5a2Q5LiK44CC"
        TaskSha256   = "8ccd69472e895f31fb374da5956d42cdfcb3368f9ffc5810bc66dda9d0e85b81"
        TaskLength   = 18
        Bundle       = "/home/nvidia/work/telop/models/smolvla_20260828_red_parcel_clean"
        Critic       = "/home/nvidia/work/telop/models/qgf/red_parcel_single_q_45_5_20260902/critic_member_00.pt"
        DatasetRoot  = "/home/nvidia/work/telop/red_parcel_real_rollouts"
        Canonicalize = "false"
        PoseTolRad   = 0.0872664625997165   # 5 deg
        Joint2Rad    = 0.75
        Label        = "red parcel out of box"
    }
    bottle = @{
        TaskBase64   = "5oqK55+/5rOJ5rC05pS+6L+b57q4566x6YeM44CC"
        TaskSha256   = "faf63ad00eb909c4b935f75ffa0302b5a8dab225c2475a0150e75e32bfb5cd07"
        TaskLength   = 10
        Bundle       = "/home/nvidia/work/telop/models/smolvla_onearm_20k_20260805"
        Critic       = "/home/nvidia/work/telop/models/qgf/real_17_116_single_qcritic/critic_member_00.pt"
        DatasetRoot  = "/home/nvidia/work/telop/bottle_real_rollouts"   # NEW root; the emptied qgf_real_rollouts stays untouched
        Canonicalize = "true"
        PoseTolRad   = 0.0
        Joint2Rad    = 0.50
        Label        = "water bottle into carton"
    }
}
$p = $profiles[$Task]

# ---- prompt guard: base64 -> bytes -> sha256 + codepoint count
$taskBytes = [Convert]::FromBase64String($p.TaskBase64)
$taskText = [System.Text.Encoding]::UTF8.GetString($taskBytes)
$sha = [System.BitConverter]::ToString([System.Security.Cryptography.SHA256]::Create().ComputeHash($taskBytes)).Replace("-", "").ToLower()
if ($taskText.Length -ne $p.TaskLength -or $sha -ne $p.TaskSha256) {
    throw "task prompt for '$Task' failed its guard: length=$($taskText.Length) (expect $($p.TaskLength)) sha256=$sha (expect $($p.TaskSha256))."
}

# ---- one pre-flight round trip: is the robot free, is the switch deployed,
# ---- and are the bundle and critic actually on the Orin?
#
# The busy gate must see BOTH kinds of session, because both hold the arm, the
# servo nodes and the cameras. The bracketed letters stop pgrep -f from matching
# the ssh command line that carries this very pattern, and GATE_OK proves the
# check ran: "no match" and "pgrep is not on the non-interactive PATH" are
# otherwise indistinguishable, and a gate that cannot be evaluated must refuse.
# The busy gate is a SEPARATE ssh call on purpose. `pgrep -f` matches the whole
# command line of every process, including the ssh that carries the check, so any
# sibling check mentioning a name in the pattern would make the gate fire on an
# idle robot. That happened: a grep against
# .../lerobot_robot_armstrong_ros2/policy_server_qgf.py matched
# policy_server_qg[f] and refused every run. Keeping the gate alone in its own
# command line makes that impossible rather than a property of the current list.
$busyPattern = "run_qgf_collection_sessio[n]|policy_server_qg[f]|policy_server_telemetr[y]|async_clien[t]|safe_one_arm_serv[o]|safe_gripper_controlle[r]|udp_leader_bridg[e]|ros2_episode_recorde[r]|dataset_cameras.launch.p[y]"
$busy = @(ssh $SshTarget "if command -v pgrep >/dev/null 2>&1; then pgrep -f '$busyPattern' >/dev/null 2>&1 && echo BUSY; echo GATE_OK; else echo GATE_FAILED; fi")
$busyCode = $LASTEXITCODE

$checks = @(
    "test -s '$($p.Bundle)/checkpoint/config.json' || echo MISSING:$($p.Bundle)/checkpoint/config.json"
)
if ($Arm -ne "baseline") {
    $checks += "test -s '$($p.Critic)' || echo MISSING:$($p.Critic)"
}
if ($Arm -in @("qgf_random", "qgf_zero")) {
    $checks += "grep -q SMOLVLA_QGF_GUIDANCE_MODE '$RemoteProject/lerobot_robot_armstrong_ros2/src/lerobot_robot_armstrong_ros2/policy_server_qgf.py' || echo NO_SWITCH_SERVER"
    $checks += "grep -q guidance_mode '$RemoteProject/qgf/src/guided_action_flow/guidance/qgf.py' || echo NO_SWITCH_QGF"
}
$preflight = @(ssh $SshTarget ($checks -join "; "))
$sshCode = $LASTEXITCODE

$problems = @()
if ($busyCode -ne 0) {
    $problems += "Could not reach $SshTarget for the busy check (ssh exit $busyCode)."
} elseif ($busy -notcontains "GATE_OK") {
    $problems += "The busy check did not run on $SshTarget (no GATE_OK). Refusing rather than assuming the robot is free."
} elseif ($busy -contains "BUSY") {
    $problems += "A collection or teleop session is already running on $SshTarget. Finish it, or clear stale nodes, before starting another."
}
if ($sshCode -ne 0) {
    $problems += "Could not reach $SshTarget to pre-flight the paths (ssh exit $sshCode)."
}
if ($preflight -contains "NO_SWITCH_SERVER" -or $preflight -contains "NO_SWITCH_QGF") {
    $problems += "arm '$Arm' needs the guidance_mode switch deployed on the Orin; it is not there. Without it the server would silently run plain QGF and this cohort would be a mislabelled second QGF arm. Deploy with tools/qgf_ablation/deploy_guidance_mode.py."
}
$missing = @($preflight | Where-Object { $_ -like "MISSING:*" })
if ($missing.Count -gt 0) {
    $problems += "Not deployed on the Orin yet:`n$($missing -join "`n")"
}

$isGuided = $Arm -ne "baseline"
if ($isGuided -and ([double]::IsNaN($Beta) -or $Beta -le 0.0 -or [double]::IsInfinity($Beta))) {
    throw "-Beta is mandatory and must be positive for arm '$Arm'."
}
$inv = [Globalization.CultureInfo]::InvariantCulture
$betaText = if ($isGuided) { $Beta.ToString($inv) } else { "0" }
$guidanceMode = switch ($Arm) { "qgf" { "critic" } "qgf_random" { "random_matched_norm" } "qgf_zero" { "zero" } default { "" } }
$runMode = if ($isGuided) { "qgf" } else { "baseline" }
$tol = if ([double]::IsNaN($InitialPoseToleranceRad)) { $p.PoseTolRad } else { $InitialPoseToleranceRad }
$j2  = if ([double]::IsNaN($Joint2MaxTargetErrorRad)) { $p.Joint2Rad } else { $Joint2MaxTargetErrorRad }
$tolText = ([double]$tol).ToString($inv)
$j2Text  = ([double]$j2).ToString($inv)

$cohort = "${Task}_${CohortSuffix}"
$comparisonTag = "comparison_cohort=$cohort"
if ([string]::IsNullOrWhiteSpace($Notes)) { $Notes = "$($p.Label); ablation" }
$notesForRun = "$Notes; condition=normal; $comparisonTag; task_profile=$Task; arm=$Arm; policy_mode=$runMode; guidance_mode=$guidanceMode; qgf_beta=$betaText; qgf_random_seed=$RandomSeed; qgf_critic=$($p.Critic); policy_bundle=$($p.Bundle)"

$enc = [System.Text.Encoding]::UTF8
$notesBase64 = [Convert]::ToBase64String($enc.GetBytes($notesForRun))
$comparisonTagBase64 = [Convert]::ToBase64String($enc.GetBytes($comparisonTag))

Write-Host "============================================================"
Write-Host "$($p.Label)  -  arm: $Arm"
Write-Host "============================================================"
Write-Host "  task            : $taskText"
Write-Host "  bundle          : $($p.Bundle)"
Write-Host "  critic          : $($p.Critic)"
Write-Host "  dataset root    : $($p.DatasetRoot)"
Write-Host "  cohort          : $cohort"
Write-Host "  run mode        : $runMode    guidance_mode=$guidanceMode"
Write-Host "  beta            : $betaText" -ForegroundColor $(if ($isGuided) { "Yellow" } else { "Gray" })
Write-Host "  episodes        : $EpisodeCount"
Write-Host "  canonicalize    : $($p.Canonicalize)    pose tol: $tolText rad    joint2 max err: $j2Text rad"
Write-Host "  notes -> metadata: $notesForRun"
Write-Host "============================================================"
Write-Host ""

# Mirrors the audited f1bc8c5 export line. Tolerance and JOINT2 are exported explicitly:
# a missing export silently falls back to another value and has killed a cohort before.
$remote = "cd '$RemoteProject' && export SMOLVLA_TASK_B64='$($p.TaskBase64)' QGF_NOTES_B64='$notesBase64' QGF_COMPARISON_TAG_B64='$comparisonTagBase64' QGF_RUN_MODE='$runMode' QGF_EPISODE_COUNT='$EpisodeCount' QGF_DATASET_ROOT='$($p.DatasetRoot)' QGF_BETA='$betaText' SMOLVLA_QGF_GUIDANCE_MODE='$guidanceMode' SMOLVLA_QGF_RANDOM_SEED='$RandomSeed' SMOLVLA_QGF_CRITIC_PATH='$($p.Critic)' SMOLVLA_ORIN_BUNDLE='$($p.Bundle)' SMOLVLA_SERVER_MODEL_PATH='$($p.Bundle)/checkpoint' SMOLVLA_EXPECTED_CHECKPOINT='$($p.Bundle)/checkpoint' SMOLVLA_GRIPPER_OPEN_THRESHOLD='0.15' SMOLVLA_GRIPPER_CLOSE_THRESHOLD='0.85' SMOLVLA_GRIPPER_CONFIRMATION_FRAMES='5' SMOLVLA_CANONICALIZE_POLICY_OBSERVATION='$($p.Canonicalize)' SMOLVLA_INITIAL_POSE_TOLERANCE_RAD='$tolText' SMOLVLA_JOINT2_MAX_TARGET_ERROR_RAD='$j2Text' && ./tools/run_qgf_collection_session.sh"

if ($DryRun) {
    Write-Host ""
    if ($problems.Count -gt 0) {
        Write-Host "-DryRun: these would BLOCK a real run:" -ForegroundColor Yellow
        foreach ($item in $problems) { Write-Host "  - $item" -ForegroundColor Yellow }
    } else {
        Write-Host "-DryRun: every precondition passed." -ForegroundColor Green
    }
    Write-Host ""
    Write-Host "Remote command that would run:" -ForegroundColor DarkGray
    Write-Host $remote -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "-DryRun: nothing started." -ForegroundColor Cyan
    if ($busy -contains "GATE_FAILED") {
        Write-Host "Raw busy check (it could not run):" -ForegroundColor DarkGray
        $busy | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
    }
    if ($problems.Count -gt 0) { exit 3 }
    exit 0
}

if ($problems.Count -gt 0) {
    throw ($problems -join "`n")
}

ssh -t $SshTarget $remote
exit $LASTEXITCODE
