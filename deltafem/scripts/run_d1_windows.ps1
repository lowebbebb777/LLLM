param(
    [string]$Model = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",
    [ValidateSet("last_token", "aligned_sequence")]
    [string]$View = "last_token",
    [int]$MaxLength = 128,
    [int]$GenerationSteps = 3
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$ProjectRoot = Join-Path $RepoRoot "deltafem"
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Missing .venv. Run deltafem\scripts\setup_windows.ps1 first."
}

$env:HF_HOME = Join-Path $RepoRoot ".cache\huggingface"
$env:PYTHONUTF8 = "1"
New-Item -ItemType Directory -Force $env:HF_HOME | Out-Null

Push-Location $ProjectRoot
try {
    & $Python scripts\run_d1.py `
        --mode hf `
        --model $Model `
        --device $Device `
        --view $View `
        --max-length $MaxLength `
        --generation-steps $GenerationSteps
}
finally {
    Pop-Location
}
