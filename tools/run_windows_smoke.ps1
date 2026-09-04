param(
    [Parameter(Mandatory = $true)]
    [string]$BasePython,
    [Parameter(Mandatory = $true)]
    [string]$Wheel,
    [Parameter(Mandatory = $true)]
    [string]$Venv,
    [Parameter(Mandatory = $true)]
    [string]$Workspace,
    [Parameter(Mandatory = $true)]
    [string]$CfdExample
)

$ErrorActionPreference = "Stop"

if (Test-Path -LiteralPath $Venv) {
    throw "Smoke-test venv already exists: $Venv"
}

New-Item -ItemType Directory -Path $Workspace -Force | Out-Null
& $BasePython -m venv $Venv
if ($LASTEXITCODE -ne 0) { throw "Could not create smoke-test venv." }

$python = Join-Path $Venv "Scripts\python.exe"
$tay = Join-Path $Venv "Scripts\tay.exe"
& $python -m pip install $Wheel
if ($LASTEXITCODE -ne 0) { throw "Wheel installation failed." }
& $python -m pip install "scipy>=1.10" "numba>=0.58"
if ($LASTEXITCODE -ne 0) { throw "TAYLANUS extra installation failed." }

Push-Location $Workspace
try {
    & $tay --version
    if ($LASTEXITCODE -ne 0) { throw "tay --version failed." }

    & $tay doctor
    if ($LASTEXITCODE -ne 0) { throw "tay doctor failed." }

    & $python -c (
        "from pathlib import Path; import taylang, taylanus_core; " +
        "paths=[Path(taylang.__file__).resolve(), " +
        "Path(taylanus_core.__file__).resolve()]; " +
        "print('INSTALLED_PATHS=' + '|'.join(map(str, paths))); " +
        "assert all('site-packages' in str(p) for p in paths)"
    )
    if ($LASTEXITCODE -ne 0) { throw "Installed import-path check failed." }

    & $tay init demo
    if ($LASTEXITCODE -ne 0) { throw "tay init failed." }
    & $tay run demo\hello.tay --quiet
    if ($LASTEXITCODE -ne 0) { throw "Starter program failed." }
    & $tay notebook demo\explore.taynb
    if ($LASTEXITCODE -ne 0) { throw "Notebook smoke failed." }

    @("a=5", "a+10", ":quit") | & $tay repl
    if ($LASTEXITCODE -ne 0) { throw "Scripted REPL smoke failed." }

    & $tay run $CfdExample --quiet
    if ($LASTEXITCODE -ne 0) { throw "Installed TAYLANUS CFD example failed." }
} finally {
    Pop-Location
}

foreach ($artifact in @(
    (Join-Path $Workspace "demo\output\final.npy"),
    (Join-Path $Workspace "demo\output\slice.png"),
    (Join-Path $Workspace "demo\output\mass.png"),
    (Join-Path $Workspace "demo\explore.report.json")
)) {
    if (-not (Test-Path -LiteralPath $artifact)) {
        throw "Missing smoke artifact: $artifact"
    }
}

Write-Host "WINDOWS_WHEEL_SMOKE=PASS"
