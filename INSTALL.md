# TAY v0.8 + TAYLANUS v3 — Installation

## Requirements

- Python 3.10 or newer
- Windows, macOS or Linux
- TAY base runtime: NumPy, pandas and Matplotlib
- TAYLANUS CPU extra: SciPy and Numba
- Optional ordinary TAY Torch backend: PyTorch

TAYLANUS GPU execution is not validated in v3.

## Windows installer

From the repository or extracted package directory:

```powershell
.\install_windows.ps1
```

The installer:

1. probes candidate interpreters by actually executing them;
2. ignores an inaccessible Windows Store `python.exe` alias;
3. falls back to common Anaconda/Miniconda locations;
4. creates `.tay-venv`;
5. installs `.[taylanus]`;
6. runs `tay --version` and `tay doctor`.

An interpreter can be supplied explicitly:

```powershell
.\install_windows.ps1 -Python C:\Python313\python.exe
```

Optional switches:

- `-UpgradePip`: upgrade pip before installation;
- `-NoDependencies`: reinstall the package without resolving dependencies
  (only for an environment where they are already installed).

The batch wrapper forwards the same arguments:

```bat
install_windows.bat -Python C:\Python313\python.exe
```

## pip source installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\python -m pip install ".[taylanus]"
# macOS/Linux
.venv/bin/python -m pip install ".[taylanus]"
```

The extra installs the TAYLANUS CPU runtime. The `taylanus_core` module itself
is included in the same wheel as `taylang`; no `sys.path` modification is
used.

## Build and install the wheel

```powershell
.\.venv\Scripts\python.exe -m build --wheel --no-isolation
python -m venv clean-venv
.\clean-venv\Scripts\python.exe -m pip install .\dist\taylang-0.8.0.dev2-py3-none-any.whl
.\clean-venv\Scripts\python.exe -m pip install "scipy>=1.10" "numba>=0.58"
```

Verify that imports come from `site-packages`, not the source tree:

```powershell
.\clean-venv\Scripts\python.exe -c "import taylang,taylanus_core; print(taylang.__file__); print(taylanus_core.__file__)"
```

Both printed paths must contain `site-packages`.

## Smoke test

```powershell
.\.tay-venv\Scripts\tay.exe --version
.\.tay-venv\Scripts\tay.exe doctor
.\.tay-venv\Scripts\tay.exe init demo
.\.tay-venv\Scripts\tay.exe run demo\hello.tay --quiet
.\.tay-venv\Scripts\tay.exe notebook demo\explore.taynb
.\.tay-venv\Scripts\tay.exe repl
.\.tay-venv\Scripts\tay.exe run examples\taylanus_vortex.tay --quiet
```

`tay doctor` reports ordinary TAY backends and the TAYLANUS engine
independently.

## Optional Torch backend

```powershell
.\.tay-venv\Scripts\python.exe -m pip install torch
.\.tay-venv\Scripts\tay.exe doctor
```

This enables `BACKEND TORCH` for ordinary TAY array programs. It does not
turn TAYLANUS into a Torch or GPU solver. The CFD bridge rejects such a request
instead of silently running something different.

## Common Windows issue

If typing `python` opens the Store or reports an inaccessible
`WindowsApps\python.exe`, use the bundled installer. It validates candidates
and found `C:\Users\TAYLAN\anaconda3\python.exe` automatically on the
reference Windows machine.

## Status

`taylang 0.8.0.dev2` is a developer preview. `TAYLANUS
3.0.0-research` is a stable research core, not a production CFD package.
