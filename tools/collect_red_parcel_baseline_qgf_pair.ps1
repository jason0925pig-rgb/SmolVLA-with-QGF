# Paired 40-episode red-parcel Baseline/QGF collection launcher (20 + 20).
#
# Modelled on tools/collect_stapler_baseline_qgf_pair.ps1.  The task-specific
# settings below are carried over verbatim from the red_parcel profile in
# tools/collect_smolvla_task_rollouts.ps1 - they are policy and safety settings,
# not cosmetics, and changing any of them either breaks the policy or trips the
# safety guard:
#   CanonicalizePolicyObservation = false
#       The clean checkpoint was trained on the controller's raw negative-J5
#       representation.  Safety checks still canonicalize wraparound joints
#       internally; only the observation handed to the policy stays raw.
#   InitialPoseToleranceRad = 0.0872664625997165  (5 degrees)
#       Permits small manual scene-reset drift without accepting the 20-degree
#       envelope the mug and stapler profiles use.
#   Joint2MaxTargetErrorRad = 0.75
#       The ROS client default is 0.50, which is tighter than this task needs
#       and would abort episodes on the guard.
#
# WHY THE TWO ARMS ARE INTERLEAVED IN ONE SESSION (required, not a convenience)
#   The 50 red-parcel baseline rollouts already on the Orin are the exact
#   episodes the Q critic red_parcel_single_q_45_5_20260902 was TRAINED on, and
#   they were collected on 2026-09-02 - a different day, under mixed lighting
#   (30 normal / 10 medium_light / 10 dark-light, per
#   docs/qgf/RED_PARCEL_SINGLE_Q_45_5_RESULT_20260902.md).  Scoring fresh QGF
#   episodes against them would therefore be both
#     (a) circular   - the critic has already seen those states and outcomes, so
#                      the baseline arm would be drawn from the critic's own
#                      training set, and
#     (b) cross-day  - different scene reset, lighting, calibration, operator.
#   The only defensible comparison is a new cohort in which baseline and QGF
#   episodes alternate B, Q, B, Q, ... inside a single session, so both arms
#   share the day, the scene, the hardware state and the operator.  That is what
#   QGF_RUN_MODE=paired does, and it is why this launcher has no QGF-only path.
#
#   The new cohort is separated from the 2026-09-02 rollouts by its comparison
#   cohort tag, not by directory: every episode lands in the same DatasetRoot,
#   and tools/summarize_qgf_comparison.py selects a cohort by substring-matching
#   the tag against each episode's notes field.  Reusing the old
#   red_parcel_normal tag would silently fold the 50 critic-training episodes
#   into the on-screen baseline success rate - the exact contamination this
#   launcher exists to avoid - so the default tag here is a distinct
#   red_parcel_paired_<condition>.

param(
    # MANDATORY on purpose.  There is no defensible default: the only beta ever
    # recorded anywhere in this project is 0.5 (33 of the 60 mug QGF episodes;
    # the other 27 record no beta at all), the stapler launcher's 0.35 default
    # has no supporting data behind it, and the 40 stapler QGF episodes from
    # 2026-09-01 recorded no beta whatsoever and are consequently
    # unreproducible.  The operator must state the value out loud.
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateRange(0.000001, 1000000.0)]
    [double]$Beta,

    [ValidateSet("normal", "medium_light", "light", "distractor")]
    [string]$Condition = "normal",

    # 20 + 20 = the 40-episode paired cohort.  Both must be positive: the remote
    # paired branch rejects zero or non-integer targets.
    [ValidateRange(1, 100000)]
    [int]$BaselineEpisodeCount = 20,
    [ValidateRange(1, 100000)]
    [int]$QgfEpisodeCount = 20,

    # Episodes of THIS cohort (this tag) already finalized on disk.  Both are 0
    # for a fresh cohort; they exist so an interrupted session can be resumed
    # without restarting the count.  They do NOT refer to the 50 rollouts from
    # 2026-09-02, which carry different tags and are filtered out.
    [ValidateRange(0, 100000)]
    [int]$ExistingBaselineCount = 0,
    [ValidateRange(0, 100000)]
    [int]$ExistingQgfCount = 0,

    [ValidateSet("ask", "baseline", "qgf")]
    [string]$InitialMode = "ask",

    # Cohort tag body.  Override only to keep a second paired cohort collected
    # under the same condition separate from this one.
    [ValidatePattern('^[A-Za-z0-9_]*$')]
    [string]$CohortId = "",

    [string]$Notes = "",
    [string]$SshTarget = "armstrong-orin"
)

$ErrorActionPreference = "Stop"

# Keep this source file pure ASCII.  Windows PowerShell 5.1 reads a BOM-less
# UTF-8 script as ANSI, which would corrupt a literal Chinese task string.  The
# prompt is therefore carried as base64 and decoded at runtime.
$task = [System.Text.Encoding]::UTF8.GetString(
    [Convert]::FromBase64String("5oqK566x5a2Q6YeM55qE57qi6Imy5YyF6KO55ou/5Ye65p2l5pS+5Yiw5qGM5a2Q5LiK44CC")
)

# Guard the literal above.  The SmolVLA checkpoint was trained on this exact
# 18-character prompt and all 50 rollouts record it verbatim; a single wrong
# codepoint silently changes the task token sequence and the run stops being
# comparable.  This is not theoretical - the base64 supplied in the handoff
# notes for this launcher was off by one character (it decoded U+88FF where the
# trained prompt has U+88F9) and would otherwise have been used unnoticed.
$taskSha256 = [BitConverter]::ToString(
    [System.Security.Cryptography.SHA256]::Create().ComputeHash(
        [System.Text.Encoding]::UTF8.GetBytes($task))).Replace("-", "").ToLowerInvariant()
if ($task.Length -ne 18 -or $taskSha256 -ne "8ccd69472e895f31fb374da5956d42cdfcb3368f9ffc5810bc66dda9d0e85b81") {
    throw "Red-parcel task prompt failed its integrity check (length=$($task.Length), sha256=$taskSha256). The embedded base64 literal is corrupt; do not collect with it."
}

# Carried over from the red_parcel profile in collect_smolvla_task_rollouts.ps1.
$datasetRoot = "/home/nvidia/work/telop/red_parcel_real_rollouts"
$bundle = "/home/nvidia/work/telop/models/smolvla_20260828_red_parcel_clean"
$critic = "/home/nvidia/work/telop/models/qgf/red_parcel_single_q_45_5_20260902/critic_member_00.pt"
# The red_parcel profile in collect_smolvla_task_rollouts.ps1 deliberately widens
# the joint-2 target-error guard to 0.75 rad.  The defaults below it are 0.50
# (configuration_armstrong_ros2.py) and 0.25 (smolvla_guard.py), and a real
# session has already tripped this guard at 0.5830 rad, so omitting this export
# would abort episodes mid-cohort.  It is NOT optional for this task.
$joint2MaxTargetErrorRad = "0.75"
$canonicalizePolicyObservation = "false"
$initialPoseToleranceRad = "0.0872664625997165"
$joint2MaxTargetErrorRad = "0.75"

$conditionNote = @{
    normal = "lighting=normal"
    medium_light = "lighting=medium"
    light = "lighting_perturbation"
    distractor = "distractor_perturbation"
}[$Condition]
if ([string]::IsNullOrWhiteSpace($CohortId)) { $CohortId = "red_parcel_paired_$Condition" }
if ([string]::IsNullOrWhiteSpace($Notes)) { $Notes = "red parcel out of box; $conditionNote" }

# The paired branch of the remote tools/run_qgf_collection_session.sh enforces
#     QGF_BASELINE_EPISODE_COUNT and QGF_QGF_EPISODE_COUNT are positive integers
#     0 <= QGF_INITIAL_SAVED_BASELINE <= QGF_BASELINE_EPISODE_COUNT
#     0 <= QGF_INITIAL_SAVED_QGF      <= QGF_QGF_EPISODE_COUNT
# and exits 2 otherwise - after the ssh handshake but before ARM, so a violation
# costs a wasted trip to the robot.  The ValidateRange attributes above already
# cover positivity and the lower bounds; the upper bounds are checked here.  For
# a fresh cohort both existing counts are 0 and the constraint holds trivially,
# but it is checked rather than assumed, because -ExistingBaselineCount and
# -ExistingQgfCount exist precisely for resuming an interrupted cohort, and a
# resume is where the upper bound actually bites.
if ($ExistingBaselineCount -gt $BaselineEpisodeCount) {
    throw "ExistingBaselineCount ($ExistingBaselineCount) exceeds BaselineEpisodeCount ($BaselineEpisodeCount); the remote session requires QGF_INITIAL_SAVED_BASELINE <= QGF_BASELINE_EPISODE_COUNT. Raise -BaselineEpisodeCount to the full cohort target."
}
if ($ExistingQgfCount -gt $QgfEpisodeCount) {
    throw "ExistingQgfCount ($ExistingQgfCount) exceeds QgfEpisodeCount ($QgfEpisodeCount); the remote session requires QGF_INITIAL_SAVED_QGF <= QGF_QGF_EPISODE_COUNT. Raise -QgfEpisodeCount to the full cohort target."
}

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

# Always format numbers with the invariant culture: these strings go into the
# episode metadata and into the remote shell command, where a comma decimal
# separator from a localised Windows install would be silently wrong.
$culture = [System.Globalization.CultureInfo]::InvariantCulture
$betaText = $Beta.ToString($culture)
$coefficient = (1.0 / $Beta).ToString($culture)

# The upstream session script records neither beta nor the critic path anywhere,
# which is why the 40 stapler QGF episodes from 2026-09-01 are unreproducible.
# The -Notes string IS written into every episode's metadata, so both are baked
# in here.  Baseline rounds carry the same notes string, so qgf_beta and
# qgf_critic describe what the QGF arm of this cohort ran at, not what that one
# episode ran; the authoritative per-episode arm label is the
# policy_mode=baseline|qgf token the remote script appends.  comparison_cohort
# must stay in the string verbatim - it is the substring
# tools/summarize_qgf_comparison.py filters on to print the running success
# rates after each S/F/D label.
$comparisonTag = "comparison_cohort=$CohortId"
$notesForRun = "$Notes; condition=$Condition; $comparisonTag; task_profile=red_parcel; qgf_beta=$betaText; qgf_critic=$critic; policy_bundle=$bundle"

$taskBase64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($task))
$notesBase64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($notesForRun))
$comparisonTagBase64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($comparisonTag))
$remoteProject = "/home/nvidia/work/telop/SmolVLA-with-QGF"
$remoteCommand = @"
cd '$remoteProject' && export SMOLVLA_TASK_B64='$taskBase64' QGF_NOTES_B64='$notesBase64' QGF_COMPARISON_TAG_B64='$comparisonTagBase64' QGF_RUN_MODE='paired' QGF_INITIAL_MODE='$resolvedInitialMode' QGF_BASELINE_EPISODE_COUNT='$BaselineEpisodeCount' QGF_QGF_EPISODE_COUNT='$QgfEpisodeCount' QGF_INITIAL_SAVED_BASELINE='$ExistingBaselineCount' QGF_INITIAL_SAVED_QGF='$ExistingQgfCount' QGF_DATASET_ROOT='$datasetRoot' QGF_BETA='$betaText' SMOLVLA_QGF_CRITIC_PATH='$critic' SMOLVLA_ORIN_BUNDLE='$bundle' SMOLVLA_SERVER_MODEL_PATH='$bundle/checkpoint' SMOLVLA_EXPECTED_CHECKPOINT='$bundle/checkpoint' SMOLVLA_GRIPPER_OPEN_THRESHOLD='0.15' SMOLVLA_GRIPPER_CLOSE_THRESHOLD='0.85' SMOLVLA_GRIPPER_CONFIRMATION_FRAMES='5' SMOLVLA_CANONICALIZE_POLICY_OBSERVATION='$canonicalizePolicyObservation' SMOLVLA_INITIAL_POSE_TOLERANCE_RAD='$initialPoseToleranceRad' SMOLVLA_JOINT2_MAX_TARGET_ERROR_RAD='$joint2MaxTargetErrorRad' SMOLVLA_JOINT2_MAX_TARGET_ERROR_RAD='$joint2MaxTargetErrorRad' && ./tools/run_qgf_collection_session.sh
"@
$remoteCommand = $remoteCommand.Trim()

$totalEpisodes = $BaselineEpisodeCount + $QgfEpisodeCount

Write-Host "============================================================"
Write-Host "Interactive paired Red-Parcel Baseline/QGF collection"
Write-Host "============================================================"
Write-Host "  cohort               : $CohortId   (filter: $comparisonTag)"
Write-Host "  condition            : $Condition ($conditionNote)"
Write-Host "  episodes this cohort : $totalEpisodes = $BaselineEpisodeCount baseline + $QgfEpisodeCount QGF, interleaved"
Write-Host "  already saved        : baseline=$ExistingBaselineCount  qgf=$ExistingQgfCount"
Write-Host "  first policy         : $resolvedInitialMode"
Write-Host ""
Write-Host "  >>> QGF BETA = $betaText   (applied Q coefficient = 1/beta = $coefficient) <<<" -ForegroundColor Cyan
if ($betaText -ne "0.5") {
    Write-Host "  WARNING: beta is not 0.5, the only value ever recorded in this project." -ForegroundColor Yellow
    Write-Host "           Mixing values across tasks makes them non-comparable and reads" -ForegroundColor Yellow
    Write-Host "           as per-task tuning.  Confirm this is deliberate." -ForegroundColor Yellow
}
Write-Host ""
Write-Host "  bundle               : $bundle"
Write-Host "  critic               : $critic"
Write-Host "  dataset root         : $datasetRoot"
Write-Host "  policy observation   : raw joint coordinates (canonicalize=$canonicalizePolicyObservation)"
Write-Host "  initial pose tol.    : $initialPoseToleranceRad rad (5 degrees)"
Write-Host "  joint 2 target error : $joint2MaxTargetErrorRad rad"
Write-Host "  task prompt (base64) : $taskBase64"
Write-Host "  task prompt          : $task"
Write-Host "  notes into metadata  : $notesForRun"
Write-Host "============================================================"
Write-Host "Both arms run in this one session so they share the day, the scene and"
Write-Host "the operator.  The 50 rollouts from 2026-09-02 are the critic's own"
Write-Host "training data and are NOT part of this comparison."
Write-Host "After each S/F/D label, only this cohort's baseline/QGF success rates"
Write-Host "are printed.  Then choose B=baseline, Q/G=QGF, or X=finish; after"
Write-Host "B/Q/G press Enter to start."
Write-Host "============================================================"

& ssh -tt $SshTarget $remoteCommand
$sshCode = $LASTEXITCODE
if ($sshCode -eq 0 -or $sshCode -eq 130) { exit 0 }
throw "Red parcel paired collection failed with SSH exit code $sshCode."
