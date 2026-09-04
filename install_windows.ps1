param(
    [string]$Python = "",
    [switch]$UpgradePip,
    [switch]$NoDependencies
)

$ErrorActionPreference = "Stop"

Write-Host "TAY Language + TAYLANUS v3 installer" -ForegroundColor Cyan

function Find-WorkingPython {
    param([string]$Requested)

    $candidates = [System.Collections.Generic.List[object]]::new()
    if ($Requested) {
        $candidates.Add([pscustomobject]@{ Command = $Requested; Prefix = @() })
    }

    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        $candidates.Add(
            [pscustomobject]@{ Command = $launcher.Source; Prefix = @("-3") }
        )
    }

    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        $candidates.Add(
            [pscustomobject]@{ Command = $pythonCommand.Source; Prefix = @() }
        )
    }

    foreach ($known in @(
        (Join-Path $env:USERPROFILE "anaconda3\python.exe"),
        (Join-Path $env:USERPROFILE "miniconda3\python.exe"),
        "C:\ProgramData\anaconda3\python.exe",
        "C:\ProgramData\miniconda3\python.exe"
    )) {
        if (Test-Path -LiteralPath $known) {
            $candidates.Add(
                [pscustomobject]@{ Command = $known; Prefix = @() }
            )
        }
    }

    foreach ($candidate in $candidates) {
        try {
            $prefix = @($candidate.Prefix)
            $probe = & $candidate.Command @prefix -c (
                "import sys; " +
                "assert sys.version_info >= (3, 10); " +
                "print(sys.executable)"
            ) 2>$null
            if ($LASTEXITCODE -eq 0 -and $probe) {
                return [pscustomobject]@{
                    Command = $candidate.Command
                    Prefix = $prefix
                    Executable = ($probe | Select-Object -Last 1)
                }
            }
        } catch {
            # A Windows Store alias can be discoverable but not executable.
        }
    }

    throw (
        "No working Python 3.10+ interpreter was found. " +
        "Pass one explicitly: .\install_windows.ps1 -Python C:\path\python.exe"
    )
}

$selected = Find-WorkingPython -Requested $Python
$pythonPrefix = @($selected.Prefix)
Write-Host "Using Python: $($selected.Executable)"

$venv = Join-Path $PSScriptRoot ".tay-venv"
if (-not (Test-Path -LiteralPath $venv)) {
    Write-Host "Creating virtual environment: $venv"
    & $selected.Command @pythonPrefix -m venv $venv
    if ($LASTEXITCODE -ne 0) {
        throw "Virtual environment creation failed."
    }
}

$venvPython = Join-Path $venv "Scripts\python.exe"
$venvTay = Join-Path $venv "Scripts\tay.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "The virtual environment is incomplete: $venvPython"
}

if ($UpgradePip) {
    Write-Host "Upgrading pip..."
    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw "pip upgrade failed."
    }
}

Write-Host "Installing TAY and the TAYLANUS CPU engine..."
$installArgs = @("-m", "pip", "install")
if ($NoDependencies) {
    $installArgs += "--no-deps"
}
$installArgs += "$PSScriptRoot[taylanus]"
& $venvPython @installArgs
if ($LASTEXITCODE -ne 0) {
    throw "TAY installation failed."
}

if (-not (Test-Path -LiteralPath $venvTay)) {
    throw "Installation completed without creating tay.exe."
}

Write-Host ""
Write-Host "Installation complete." -ForegroundColor Green
Write-Host "Version:"
& $venvTay --version
Write-Host "Doctor check:"
& $venvTay doctor

Write-Host ""
Write-Host "Run the CFD example with:"
Write-Host "  `"$venvTay`" run examples\taylanus_vortex.tay"
Write-Host ""
Write-Host "Optional PyTorch array backend:"
Write-Host "  `"$venvPython`" -m pip install torch"
Write-Host "TAYLANUS itself remains on its validated CPU NumPy/Numba/SciPy path."
