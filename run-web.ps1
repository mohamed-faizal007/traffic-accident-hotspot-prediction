<#
.SYNOPSIS
  Starts the FastAPI backend and the React frontend together, for demo
  convenience. Ctrl+C stops both cleanly.

.DESCRIPTION
  Runs `uvicorn` (WITHOUT --reload, since this is a demo launcher, not an
  active-development workflow) and `npm run dev` as two child processes of
  this console, then waits. Pressing Ctrl+C unwinds the try/finally below
  and force-stops both processes, so nothing is left listening on
  port 8000 or 5173.

  Does not touch any frozen ML artifact, the ML pipeline, or the Streamlit
  fallback app (legacy/streamlit_dashboard) -- it only starts the two
  existing web/backend and web/frontend dev servers.

.EXAMPLE
  .\run-web.ps1
#>

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$venvPython = Join-Path $root "venv\Scripts\python.exe"
$backendDir = Join-Path $root "web\backend"
$frontendDir = Join-Path $root "web\frontend"

if (-not (Test-Path $venvPython)) {
    Write-Error "Python venv not found at $venvPython. Set it up first (see README.md 'Reproducibility')."
    exit 1
}
if (-not (Test-Path (Join-Path $frontendDir "node_modules"))) {
    Write-Error "web/frontend/node_modules not found. Run 'npm install' in web/frontend first (see web/README.md)."
    exit 1
}

function Test-PortOpen($port) {
    # Try both loopback addresses directly (no hostname/DNS resolution,
    # which was a source of flaky timing) -- Vite defaults to the IPv6
    # loopback ([::1]) in this setup, which an IPv4-only check would miss.
    foreach ($addr in @([System.Net.IPAddress]::Loopback, [System.Net.IPAddress]::IPv6Loopback)) {
        try {
            # Windows PowerShell 5.1 / .NET Framework: TcpClient()'s
            # parameterless constructor creates an IPv4-only socket, which
            # throws if BeginConnect is later given an IPv6 address -- must
            # construct with the matching AddressFamily explicitly.
            $client = New-Object System.Net.Sockets.TcpClient($addr.AddressFamily)
            $iar = $client.BeginConnect($addr, $port, $null, $null)
            $ok = $iar.AsyncWaitHandle.WaitOne(500)
            if ($ok -and $client.Connected) { $client.Close(); return $true }
            $client.Close()
        } catch { }
    }
    return $false
}

Write-Host "Starting backend  (FastAPI / uvicorn, no --reload)..." -ForegroundColor Cyan
$backend = Start-Process -FilePath $venvPython `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--port", "8000" `
    -WorkingDirectory $backendDir -NoNewWindow -PassThru

Write-Host "Starting frontend (Vite dev server)..." -ForegroundColor Cyan
$frontend = Start-Process -FilePath "npm.cmd" `
    -ArgumentList "run", "dev" `
    -WorkingDirectory $frontendDir -NoNewWindow -PassThru

function Stop-Both {
    Write-Host "`nStopping backend and frontend..." -ForegroundColor Yellow
    foreach ($p in @($backend, $frontend)) {
        if ($p -and -not $p.HasExited) {
            try { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } catch {}
        }
    }
    Write-Host "Stopped. Ports 8000 and 5173 should now be free." -ForegroundColor Yellow
}

try {
    $backendReady = $false
    $frontendReady = $false
    for ($i = 0; $i -lt 30; $i++) {
        if ($backend.HasExited -or $frontend.HasExited) { break }
        if (-not $backendReady)  { $backendReady  = Test-PortOpen 8000 }
        if (-not $frontendReady) { $frontendReady = Test-PortOpen 5173 }
        if ($backendReady -and $frontendReady) { break }
        Start-Sleep -Seconds 1
    }

    if ($backend.HasExited -or $frontend.HasExited) {
        Write-Host "One of the processes exited during startup -- check the output above." -ForegroundColor Red
    } else {
        Write-Host ""
        Write-Host "=================================================================" -ForegroundColor Green
        Write-Host "  Backend  API :  http://localhost:8000   (docs at /docs)" -ForegroundColor Green
        Write-Host "  Frontend App :  http://localhost:5173" -ForegroundColor Green
        Write-Host "=================================================================" -ForegroundColor Green
        Write-Host "Press Ctrl+C to stop both.`n"
    }

    while ($true) {
        if ($backend.HasExited -or $frontend.HasExited) {
            Write-Host "One of the processes exited unexpectedly; stopping the other." -ForegroundColor Red
            break
        }
        Start-Sleep -Seconds 1
    }
}
finally {
    Stop-Both
}
