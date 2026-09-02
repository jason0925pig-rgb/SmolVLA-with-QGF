# Thin, audited wrapper around collect_stapler_baseline_qgf_pair.ps1 for the
# stapler / normal-lighting QGF arm.
#
# Why this wrapper exists:
#   The upstream launcher defaults to -Beta 0.35.  Every recorded mug QGF episode
#   ran at beta = 0.5 (33 episodes carry qgf_beta=0.5; the other 27 record no beta
#   field at all).  Running the stapler at 0.35 would be per-task beta tuning,
#   which the review explicitly warned against.  This wrapper pins beta = 0.5 so
#   the value cannot be forgotten on the command line.
#
#   It also pins the condition and the already-collected baseline count, so the
#   on-screen running success rates are computed against the right cohort.

param(
    # How many QGF episodes to collect this session.
    [ValidateRange(1, 1000)]
    [int]$QgfEpisodeCount = 20,

    # Baseline episodes to collect alongside.  0 = QGF only (compare against the
    # 50 baseline episodes already on disk).  Any value > 0 runs the upstream
    # paired B/Q flow, which is the statistically stronger option.
    [ValidateRange(0, 1000)]
    [int]$BaselineEpisodeCount = 0,

    # Baseline episodes already saved in this cohort, used only for the running
    # success-rate display.  50 = the stapler clean baseline collected 2026-08-30.
    [ValidateRange(0, 100000)]
    [int]$ExistingBaselineCount = 50,

    [ValidateRange(0, 100000)]
    [int]$ExistingQgfCount = 0,

    # Locked deliberately.  Do not change without also re-running every other
    # QGF comparison at the new value.
    [ValidateRange(0.000001, 1000000.0)]
    [double]$Beta = 0.5,

    [string]$SshTarget = "armstrong-orin"
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$upstream = Join-Path $here "collect_stapler_baseline_qgf_pair.ps1"
if (-not (Test-Path $upstream)) {
    throw "Upstream launcher not found: $upstream"
}

if ($Beta -ne 0.5) {
    Write-Host ""
    Write-Host "  WARNING: beta = $Beta, not the project-wide 0.5." -ForegroundColor Yellow
    Write-Host "  Every existing QGF comparison ran at 0.5.  Mixing values makes" -ForegroundColor Yellow
    Write-Host "  the tasks non-comparable and looks like per-task tuning." -ForegroundColor Yellow
    Write-Host ""
}

# For a QGF-only session the remote session script still needs a baseline
# target, and it enforces  QGF_INITIAL_SAVED_BASELINE <= QGF_BASELINE_EPISODE_COUNT.
# So set the target EQUAL to what is already on disk: the baseline arm then reads
# as already complete (50/50), the round loop keeps going until the QGF target is
# met, and choosing B by mistake is refused with "Baseline target is already
# complete."  Passing 1 here instead fails validation whenever ExistingBaselineCount > 1.
$qgfOnly = $BaselineEpisodeCount -le 0
$baselineArg = if ($qgfOnly) { [math]::Max(1, $ExistingBaselineCount) } else { $BaselineEpisodeCount }
$initialMode = if ($qgfOnly) { "qgf" } else { "ask" }

if ($qgfOnly -and $ExistingBaselineCount -lt 1) {
    Write-Host "  NOTE: no existing baseline recorded, so the baseline arm shows 0/1." -ForegroundColor Yellow
    Write-Host "        The session will not stop on its own - press X after the last QGF episode." -ForegroundColor Yellow
}

Write-Host "============================================================"
Write-Host "Stapler / normal lighting  -  QGF arm"
Write-Host "============================================================"
Write-Host "  condition            : normal"
Write-Host "  beta                 : $Beta   (Q coefficient = 1/beta = $([math]::Round(1.0/$Beta, 6)))"
Write-Host "  QGF episodes         : $QgfEpisodeCount"
if ($qgfOnly) {
    Write-Host "  baseline episodes    : 0  (QGF only)" -ForegroundColor Cyan
    Write-Host "                         baseline arm pinned complete at $([math]::Max(1,$ExistingBaselineCount))/$([math]::Max(1,$ExistingBaselineCount)); choose Q every round"
    Write-Host ""
    Write-Host "  NOTE: comparing against the 50 baseline episodes from 2026-08-30." -ForegroundColor Yellow
    Write-Host "        That is a cross-day, unpaired comparison.  Running paired" -ForegroundColor Yellow
    Write-Host "        baseline in the same session is stronger - see -BaselineEpisodeCount." -ForegroundColor Yellow
} else {
    Write-Host "  baseline episodes    : $BaselineEpisodeCount  (paired B/Q in this session)" -ForegroundColor Green
}
Write-Host "  already saved        : baseline=$ExistingBaselineCount  qgf=$ExistingQgfCount"
Write-Host "  critic               : stapler_into_box_single_q_45_5_20260830/critic_member_00.pt"
Write-Host "  bundle               : smolvla_20260827_stapler_into_box"
Write-Host "  notes                : beta recorded into every episode metadata"
Write-Host "============================================================"
Write-Host ""

# The upstream launcher records neither beta nor the critic path in the episode
# metadata.  All 40 stapler light/distractor QGF episodes from 2026-09-01 are
# therefore unreproducible - nothing on disk says what beta they ran at.  The
# -Notes string IS written into every episode's metadata, so bake beta into it.
# 'lighting=normal' must stay in the string: it is the comparison filter the
# session uses to compute the running success rate against the existing 50
# baseline episodes.
$notes = "stapler into box; lighting=normal; qgf_beta=$Beta"

& $upstream `
    -Condition normal `
    -Notes $notes `
    -BaselineEpisodeCount $baselineArg `
    -QgfEpisodeCount $QgfEpisodeCount `
    -ExistingBaselineCount $ExistingBaselineCount `
    -ExistingQgfCount $ExistingQgfCount `
    -InitialMode $initialMode `
    -Beta $Beta `
    -SshTarget $SshTarget

exit $LASTEXITCODE
