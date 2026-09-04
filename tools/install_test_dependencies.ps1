param(
    [Parameter(Mandatory = $true)]
    [string]$Python,
    [switch]$WithTorch
)

$ErrorActionPreference = "Stop"

& $Python -m pip install pytest matplotlib build wheel
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install the standard test/build dependencies."
}

if ($WithTorch) {
    & $Python -m pip install torch
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install the optional PyTorch test dependency."
    }
}
