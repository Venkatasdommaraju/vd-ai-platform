Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force
$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Definition
$SITE = Join-Path $ROOT "vsd-portfolio"

if (-not (Test-Path $SITE)) {
    Write-Host "[ERROR] vsd-portfolio folder not found at: $SITE" -ForegroundColor Red
    exit 1
}

Set-Location $SITE

if (-not (Test-Path (Join-Path $SITE "node_modules"))) {
    Write-Host "node_modules missing, running npm install..." -ForegroundColor Yellow
    npm install
    if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] npm install failed." -ForegroundColor Red; exit 1 }
}

Write-Host "Starting portfolio dev server at http://localhost:5173" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop."
Write-Host ""

npm run dev
