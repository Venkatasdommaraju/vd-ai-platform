param(
    [ValidateSet("iot", "portfolio", "langgraph", "")]
    [string]$Backend = ""
)

Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force
$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Definition

$BACKENDS = @{
    iot = @{
        Name   = "IOT-Anomaly-Detector"
        Dir    = Join-Path $ROOT "IOT-Anomaly-Detector"
        Port   = 8080
        Module = "service.main:app"
    }
    portfolio = @{
        Name   = "AI-Portfolio-Assistant"
        Dir    = Join-Path $ROOT "AI-Portfolio-Assistant"
        Port   = 8081
        Module = "app.main:app"
    }
    langgraph = @{
        Name   = "Multi-Agent-Langgraph"
        Dir    = Join-Path $ROOT "Multi-Agent-Langgraph"
        Port   = 8082
        Module = "app.main:app"
    }
}

if ($Backend -eq "") {
    Write-Host ""
    Write-Host "Which backend?" -ForegroundColor Cyan
    Write-Host "  [1]  IOT-Anomaly-Detector    -> http://localhost:8080"
    Write-Host "  [2]  AI-Portfolio-Assistant  -> http://localhost:8081"
    Write-Host "  [3]  Multi-Agent-Langgraph   -> http://localhost:8082"
    Write-Host ""
    $choice = Read-Host "Enter 1, 2 or 3"
    switch ($choice.Trim()) {
        "1" { $Backend = "iot" }
        "2" { $Backend = "portfolio" }
        "3" { $Backend = "langgraph" }
        default { Write-Host "Invalid choice." -ForegroundColor Red; exit 1 }
    }
}

$cfg  = $BACKENDS[$Backend]
$DIR  = $cfg.Dir
$PORT = $cfg.Port

if (-not (Test-Path $DIR)) {
    Write-Host "[ERROR] Folder not found: $DIR" -ForegroundColor Red
    exit 1
}

Set-Location $DIR

$venvPath = $null
foreach ($candidate in @(".venv", "venv")) {
    $p = Join-Path $DIR $candidate
    if (Test-Path (Join-Path $p "Scripts\python.exe")) {
        $venvPath = $p
        break
    }
}

if (-not $venvPath) {
    Write-Host "[ERROR] No virtual environment found in $DIR. Create one with 'python -m venv .venv' and install requirements." -ForegroundColor Red
    exit 1
}

$uvicorn = Join-Path $venvPath "Scripts\uvicorn.exe"
$python  = Join-Path $venvPath "Scripts\python.exe"

Write-Host ""
Write-Host "Starting $($cfg.Name) on port $PORT..." -ForegroundColor Green
Write-Host "URL: http://localhost:$PORT" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop."
Write-Host ""

if (Test-Path $uvicorn) {
    & $uvicorn $cfg.Module --reload --host 0.0.0.0 --port $PORT
} else {
    & $python -m uvicorn $cfg.Module --reload --host 0.0.0.0 --port $PORT
}
