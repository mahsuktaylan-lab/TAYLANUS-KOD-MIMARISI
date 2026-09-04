# (.tay) KOD MİMARİSİ

TAY Language v0.8 and TAYLANUS v3 now form one installable research workflow:

- **TAY Language** is the readable scientific-programming and orchestration
  layer.
- **TAYLANUS** is the adaptive incompressible-flow research solver.
- **TAYLANUS remains the solver source of truth.** The language adapter calls
  the existing `UnifiedTaylanusIR`, planner, time-step and projection APIs; it
  does not reimplement the CFD equations.

The integration release is `taylang 0.8.0.dev2`. TAYLANUS keeps its
`3.0.0-research` identity.

## Install

On Windows:

```powershell
.\install_windows.ps1
.\.tay-venv\Scripts\tay.exe doctor
```

On any supported Python 3.10+ environment:

```bash
python -m pip install ".[taylanus]"
tay doctor
```

See [INSTALL.md](INSTALL.md) for wheel installation, verification and Windows
troubleshooting.

## Run the real CFD example

```powershell
.\.tay-venv\Scripts\tay.exe run examples\taylanus_vortex.tay
```

The TAY source is intentionally small:

```tay
BACKEND NUMPY
ENGINE TAYLANUS

RESOLUTION 32
DT 0.005
TEND 0.20
MODE AUTO
VISCOSITY 0.0033333333333333335
REFERENCE "../references/localized_vortex_N32_t0p20.npy"
OUTPUT "../outputs/taylanus_vortex"

RUN TAYLANUS
```

`BACKEND` and `ENGINE` are separate:

- `BACKEND NUMPY|TORCH|GPU` selects the ordinary TAY array/device runtime.
- `ENGINE TAYLANUS` selects the CFD solver.
- TAYLANUS v3 is validated only on CPU NumPy/Numba/SciPy. A GPU or Torch
  TAYLANUS request fails closed; it never silently falls back to CPU.

## CFD outputs

The example writes real solver-derived data under
`outputs/taylanus_vortex/`:

- adaptive mesh-level slice and level distribution;
- velocity-magnitude and vorticity-magnitude slices;
- conservative finite-volume divergence slice;
- kinetic-energy history;
- numerical CSV and JSON diagnostics;
- final velocity, mesh level and divergence NumPy arrays.

No pressure plot is fabricated. The frozen core does not expose a pressure
field as a public result, so the report records
`pressure_field_available=false`.

## Architecture

```text
.tay source
   ├─ BACKEND -> NumPy / Torch / CUDA array runtime
   └─ ENGINE TAYLANUS
         -> validated CFD configuration
         -> topology + UnifiedCompileCache
         -> UnifiedTaylanusIR
         -> planner
              ├─ SUBFACE_SPARSE
              └─ MODAL_STREAM
         -> existing RK2 + pressure projection
         -> diagnostics + reproducible outputs
```

The CFD bridge is implemented in `taylang/engines/taylanus.py`; the unchanged
solver remains in `taylanus_core.py`.

## Validation

Current Windows verification:

- original TAY v0.8 baseline: **83 passed, 0 failed, 0 skipped**;
- combined TAY + bridge suite: **98 passed, 0 failed, 0 skipped**;
- TAYLANUS numerical regression: **4/4 cases passed**;
- isolated wheel, console command and `site-packages` import smoke: **PASS**;
- Windows installer with automatic interpreter fallback: **PASS**.

The NF=32, t=0.20 example reproduced:

- representation: `SUBFACE_SPARSE`;
- leaf count: 2,402;
- state DOFs: 29,364;
- relative reference L2: 0.05311733082469636;
- RMS conservative divergence: 1.8287457244460903e-17.

Exact commands and environment evidence are in
`reports/TAY_TAYLANUS_INTEGRATION_REPORT.md`.

## Scientific scope

This is a stable **research core** plus a developer-preview language. It does
not establish ANSYS equivalence, production CFD validation, general speed
superiority or validated GPU performance. The localized periodic-vortex case
is a short-time validation problem favorable to local adaptivity. See
`docs/KNOWN_LIMITATIONS.md` and `docs/TAYLANUS_ENGINE.md`.

## Development gates

```powershell
.\.venv\Scripts\python.exe -m pytest -q -o addopts= -ra
.\.venv\Scripts\python.exe tests\run_regression.py
.\.venv\Scripts\python.exe benchmarks\run_benchmarks.py
.\.venv\Scripts\python.exe -m build --wheel --no-isolation
```

The integration lives on `feature/tay-taylanus-integration`; `main` is not
merged automatically.
