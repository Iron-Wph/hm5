param(
    [string]$Config = "config.json",
    [switch]$Install,
    [switch]$SkipColmap,
    [switch]$SkipTrain,
    [switch]$SkipEval,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Invoke-Step {
    param(
        [string]$Title,
        [string[]]$ArgsList
    )
    Write-Host ""
    Write-Host "==== $Title ===="
    & python @ArgsList
}

if ($Install) {
    Write-Host "Installing Python helper packages from requirements.txt"
    & python -m pip install --upgrade pip
    & python -m pip install -r requirements.txt
    Write-Host ""
    Write-Host "Nerfstudio/CUDA packages are intentionally not installed here because they depend on your GPU/CUDA setup."
    Write-Host "Install them using the commands in 一键完成式实验执行说明.md before running training."
}

$videoPath = Join-Path $PSScriptRoot "data/raw/toy.mp4"
if (-not (Test-Path $videoPath)) {
    New-Item -ItemType Directory -Force -Path (Split-Path $videoPath) | Out-Null
    throw "Missing video file: $videoPath. Put toy.mp4 at this path, then run this script again."
}

Invoke-Step "1. Extract and score frames" @("scripts/00_extract_score_frames.py", "--config", $Config)
Invoke-Step "2. Filter frames" @("scripts/01_filter_frames.py", "--config", $Config)

if (-not $SkipColmap) {
    $args = @("scripts/02_process_colmap.py", "--config", $Config)
    if ($DryRun) { $args += "--dry-run" }
    Invoke-Step "3. COLMAP/Nerfstudio data processing" $args
} else {
    Write-Host "Skipping COLMAP/Nerfstudio data processing."
}

Invoke-Step "4. Pose-aware split" @("scripts/03_pose_stratified_split.py", "--config", $Config)

if (-not $SkipTrain) {
    $args = @("scripts/04_train_all.py", "--config", $Config)
    if ($DryRun) { $args += "--dry-run" }
    Invoke-Step "5. Train NeRF and 3DGS models" $args
} else {
    Write-Host "Skipping model training."
}

if (-not $SkipEval) {
    $args = @("scripts/05_eval_metrics.py", "--config", $Config)
    if ($DryRun) { $args += "--dry-run" }
    Invoke-Step "6. Evaluate trained models" $args
} else {
    Write-Host "Skipping evaluation."
}

Invoke-Step "7. Build figures" @("scripts/06_make_figures.py", "--config", $Config)
Invoke-Step "8. Build qualitative examples" @("scripts/08_make_qualitative_examples.py", "--config", $Config)
Invoke-Step "9. Write report outline" @("scripts/07_write_report_outline.py", "--config", $Config)

Write-Host ""
Write-Host "Pipeline finished. Check:"
Write-Host "  results/tables/"
Write-Host "  results/figures/"
Write-Host "  report/final_report_outline.md"
