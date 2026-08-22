param(
    [string]$Task = "",
    [string]$SshTarget = "armstrong-orin",
    [string]$DatasetRoot = "/home/nvidia/work/telop/qgf_real_rollouts",
    [string]$Notes = "background_perturbation",
    [string]$ComparisonTag = "background_perturbation",
    [ValidateRange(1, 100000)]
    [int]$BaselineEpisodeCount = 1,
    [ValidateRange(1, 100000)]
    [int]$QgfEpisodeCount = 1,
    [ValidateSet("ask", "baseline_first", "qgf_first")]
    [string]$Order = "ask",
    [double]$Beta = 2.0
)

$ErrorActionPreference = "Stop"
if ($Beta -le 0.0) {
    throw "Beta must be positive. The actual Q guidance coefficient is 1/Beta."
}

$collector = Join-Path $PSScriptRoot "collect_qgf_rollouts.ps1"
if (-not (Test-Path -LiteralPath $collector -PathType Leaf)) {
    throw "Existing collector was not found: $collector"
}

function Invoke-OneRollout {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet("baseline", "qgf")]
        [string]$Mode,
        [Parameter(Mandatory = $true)]
        [int]$EpisodeCount
    )

    $arguments = @(
        "-NoLogo",
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $collector,
        "-EpisodeCount", $EpisodeCount.ToString(),
        "-SshTarget", $SshTarget,
        "-DatasetRoot", $DatasetRoot,
        "-Notes", $Notes,
        "-ComparisonTag", $ComparisonTag,
        "-Mode", $Mode
    )
    if (-not [string]::IsNullOrWhiteSpace($Task)) {
        $arguments += @("-Task", $Task)
    }
    if ($Mode -eq "qgf") {
        $arguments += @("-Beta", $Beta.ToString([System.Globalization.CultureInfo]::InvariantCulture))
    }

    & powershell.exe @arguments
    $collectorCode = $LASTEXITCODE
    if ($collectorCode -ne 0) {
        throw "$Mode collection failed with exit code $collectorCode. The next collection stage will not be started."
    }
}

function Resolve-Order {
    if ($Order -ne "ask") {
        return $Order
    }

    while ($true) {
        $answer = (Read-Host "Run baseline first or QGF first? Enter B or Q").Trim().ToUpperInvariant()
        switch ($answer) {
            "B" { return "baseline_first" }
            "BASELINE" { return "baseline_first" }
            "Q" { return "qgf_first" }
            "QGF" { return "qgf_first" }
            default { Write-Host "Please enter B (baseline first) or Q (QGF first)." -ForegroundColor Yellow }
        }
    }
}

$coefficient = (1.0 / $Beta).ToString([System.Globalization.CultureInfo]::InvariantCulture)
$resolvedOrder = Resolve-Order
Write-Host "============================================================"
Write-Host "Paired real-robot comparison"
Write-Host "Both rounds reuse the existing recorder, S/F/D outcome flow,"
Write-Host "automatic completion, timeout, safety cleanup and data format."
Write-Host "Baseline target: $BaselineEpisodeCount; QGF target: $QgfEpisodeCount"
Write-Host "Run order: $resolvedOrder"
Write-Host "QGF beta=$Beta; actual Q coefficient=1/beta=$coefficient"
Write-Host "Notes: $Notes"
Write-Host "Comparison cohort: $ComparisonTag"
Write-Host "============================================================"
Write-Host ""

if ($resolvedOrder -eq "baseline_first") {
    Write-Host "PAIR_STAGE_1_BASELINE"
    Invoke-OneRollout -Mode "baseline" -EpisodeCount $BaselineEpisodeCount
    Write-Host ""
    Write-Host "BASELINE_COLLECTION_FINISHED"
    Write-Host "PAIR_STAGE_2_QGF"
    Invoke-OneRollout -Mode "qgf" -EpisodeCount $QgfEpisodeCount
} else {
    Write-Host "PAIR_STAGE_1_QGF"
    Invoke-OneRollout -Mode "qgf" -EpisodeCount $QgfEpisodeCount
    Write-Host ""
    Write-Host "QGF_COLLECTION_FINISHED"
    Write-Host "PAIR_STAGE_2_BASELINE"
    Invoke-OneRollout -Mode "baseline" -EpisodeCount $BaselineEpisodeCount
}

Write-Host ""
Write-Host "PAIRED_BASELINE_QGF_COLLECTION_FINISHED"
