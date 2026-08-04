$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$ProjectRoot = Join-Path $RepoRoot "deltafem"
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Missing .venv. Run deltafem\scripts\setup_windows.ps1 first."
}

Push-Location $ProjectRoot
try {
    & $Python -m pytest -q
    & $Python scripts\run_d1.py --mode toy --device cpu --generation-steps 2 --max-length 64 --output results\d1_toy_smoke
}
finally {
    Pop-Location
}
