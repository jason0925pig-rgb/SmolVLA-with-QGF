param(
    [string]$Task = "",
    [string]$SshTarget = "armstrong-orin",
    [string]$DatasetRoot = "/home/nvidia/work/telop/qgf_real_rollouts",
    [string]$Notes = "official; lighting=medium; paired_trial",
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
        [string]$Mode
    )

    $arguments = @(
        "-NoLogo",
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $collector,
        "-EpisodeCount", "1",
        "-SshTarget", $SshTarget,
        "-DatasetRoot", $DatasetRoot,
        "-Notes", $Notes,
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
        throw "$Mode rollout failed with exit code $collectorCode. The QGF rollout will not be started after a baseline failure."
    }
}

$coefficient = (1.0 / $Beta).ToString([System.Globalization.CultureInfo]::InvariantCulture)
Write-Host "============================================================"
Write-Host "Paired real-robot comparison: one baseline, then one QGF"
Write-Host "Both rounds reuse the existing recorder, S/F/D outcome flow,"
Write-Host "automatic completion, timeout, safety cleanup and data format."
Write-Host "QGF beta=$Beta; actual Q coefficient=1/beta=$coefficient"
Write-Host "Notes: $Notes"
Write-Host "============================================================"
Write-Host ""

Write-Host "PAIR_STAGE_1_BASELINE"
Invoke-OneRollout -Mode "baseline"

Write-Host ""
Write-Host "BASELINE_ROUND_FINISHED"
Write-Host "PAIR_STAGE_2_QGF"
Invoke-OneRollout -Mode "qgf"

Write-Host ""
Write-Host "PAIRED_BASELINE_QGF_COLLECTION_FINISHED"

