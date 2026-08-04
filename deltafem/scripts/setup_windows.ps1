param(
    [ValidateSet("cu128", "cu130", "cpu")]
    [string]$TorchBuild = "cu128",
    [string]$TorchVersion = "2.11.0",
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$ProjectRoot = Join-Path $RepoRoot "deltafem"
$VenvRoot = Join-Path $RepoRoot ".venv"

if ($Recreate -and (Test-Path $VenvRoot)) {
    Remove-Item -Recurse -Force $VenvRoot
}

if (-not (Test-Path $VenvRoot)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.11 -m venv $VenvRoot
        if ($LASTEXITCODE -ne 0) {
            & py -3 -m venv $VenvRoot
        }
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv $VenvRoot
    }
    else {
        throw "Python was not found. Install Python 3.11 or newer and retry."
    }
}

$Python = Join-Path $VenvRoot "Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Virtual environment creation failed: $Python not found"
}

& $Python -m pip install --upgrade pip setuptools wheel
$TorchIndex = "https://download.pytorch.org/whl/$TorchBuild"
& $Python -m pip install "torch==$TorchVersion" --index-url $TorchIndex
& $Python -m pip install -r (Join-Path $ProjectRoot "requirements-d1.txt")
& $Python -m pip install --editable $ProjectRoot --no-deps

$env:HF_HOME = Join-Path $RepoRoot ".cache\huggingface"
New-Item -ItemType Directory -Force $env:HF_HOME | Out-Null

& $Python -c "import torch; print('Python OK'); print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

Write-Host ""
Write-Host "DeltaFEM D1 environment is ready."
Write-Host "Venv: $VenvRoot"
Write-Host "Activate: $VenvRoot\Scripts\Activate.ps1"
Write-Host "Verify: powershell -ExecutionPolicy Bypass -File $ProjectRoot\scripts\verify_windows.ps1"
