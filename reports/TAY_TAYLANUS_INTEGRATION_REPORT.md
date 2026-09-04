# TAY v0.8 + TAYLANUS v3 Integration Report

Date: 2026-09-04
Project name: `(.tay) KOD MİMARİSİ`
Branch: `feature/tay-taylanus-integration`
Frozen `main` base: `d5a4a944435f9f27b95a85f7a99867b0fdff3e57`
Host: Microsoft Windows NT 10.0.19045.0, x64, 4 logical processors
Python: 3.14.6

## Outcome

TAY Language and TAYLANUS are now one installable workflow. A `.tay` program
can select `ENGINE TAYLANUS`, validate CFD parameters, invoke the existing
`UnifiedTaylanusIR` pipeline, run either validated representation, and write
structured numerical/graphical results.

The CFD equations, topology, operators, pressure projection and time steps
remain in the original `taylanus_core.py`. The bridge is an adapter, not a
second solver.

## Architecture decision

```text
BACKEND NUMPY|TORCH|GPU
    ordinary TAY array/device implementation

ENGINE TAYLANUS
    domain solver selection
        -> configuration validation
        -> UnifiedCompileCache
        -> UnifiedTaylanusIR
        -> AUTO/FAST/COMPACT representation selection
        -> existing RK2/projection step methods
        -> diagnostics and artifacts
```

This separation prevents a TAY numerical backend from being mistaken for a
TAYLANUS implementation. TAYLANUS v3 rejects Torch/GPU engine requests and
does not silently fall back.

## Packaging

- Distribution: `taylang`
- Integration version: `0.8.0.dev2`
- TAYLANUS identity: `3.0.0-research`
- Wheel:
  `dist/taylang-0.8.0.dev2-py3-none-any.whl`
- Wheel SHA-256:
  `68691207e38066282cbaa5a79dcb3c755fea83df965684674b1c2fd55ca6a677`
- Included installed modules:
  - `taylang`
  - `taylang.engines.taylanus`
  - `taylanus_core`
- Optional CPU engine dependencies:
  `taylanus = [scipy>=1.10, numba>=0.58]`

There is no `sys.path` workaround.

## Test environments

Development environment:

| Component | Version |
|---|---:|
| Python | 3.14.6 |
| NumPy | 2.5.2 |
| SciPy | 1.18.1 |
| Numba | 0.67.0 |
| pandas | 3.0.5 |
| Matplotlib | 3.11.1 |
| pytest | 9.1.1 |
| PyTorch CPU | 2.14.0+cpu |
| CUDA available | false |

The clean-wheel environment was newly created from
`C:\Users\TAYLAN\anaconda3\python.exe`. It installed the wheel and all
declared CPU dependencies without inheriting the source package.

## Exact verification commands and results

### Unmodified TAY v0.8 baseline

Working directory:
`C:\Users\TAYLAN\Documents\ChatGPT\ANACONDA çalışmaları\TAYLANG_v0_8_BASELINE`

```powershell
& "C:\Users\TAYLAN\Documents\ChatGPT\ANACONDA çalışmaları\TAYLANUS-KOD-MIMARISI\.venv\Scripts\python.exe" -m pytest -q -o addopts= -ra
```

Result: **83 passed, 0 failed, 0 skipped** in 10.93 s.

Source CLI baseline:

```powershell
& "<dev-python>" -m taylang --version
& "<dev-python>" -m taylang doctor
```

Result: `tay 0.8.0.dev1`; NumPy available, PyTorch/GPU initially absent in
that environment.

### Final combined suite

```powershell
.\.venv\Scripts\python.exe -m pytest -q -o addopts= -ra
```

Result: **98 passed, 0 failed, 0 skipped** in 38.82 s.

Breakdown:

- preserved TAY v0.1-v0.8 suite: 83;
- engine/bridge tests: 11;
- TAYLANUS release-gate tests: 4.

The bridge tests execute both real representations, the real AUTO planner,
interpreter and generated-Python paths; validate output files and finite data;
exercise invalid resolution/time-step/mode settings; and assert fail-closed
Torch/GPU behavior.

### TAYLANUS numerical regression

```powershell
.\.venv\Scripts\python.exe tests\run_regression.py
```

Result: **4/4 cases passed**.

| Resolution | Representation | Relative L2 | Energy error | State DOFs | RMS divergence | Result |
|---:|---|---:|---:|---:|---:|---|
| 32 | SUBFACE_SPARSE | 0.05311733 | 0.02119380 | 29,364 | 1.829e-17 | PASS |
| 32 | MODAL_STREAM | 0.06270016 | 0.02231101 | 11,329 | 1.607e-17 | PASS |
| 64 | SUBFACE_SPARSE | 0.04467618 | 0.00477869 | 125,520 | 3.073e-17 | PASS |
| 64 | MODAL_STREAM | 0.05899064 | 0.00648833 | 33,926 | 2.878e-17 | PASS |

### TAYLANUS release gates

```powershell
.\.venv\Scripts\python.exe -m pytest -q -o addopts= -ra tests\test_taylanus_release_gates.py
```

Result: **4 passed, 0 failed, 0 skipped** in 19.03 s.

Verified:

- frozen manifest flags;
- planner COMPACT/memory/runtime/tie-band policy;
- actual geometry and subface IR cache hits;
- generated GPU backend remains explicitly unvalidated.

### Benchmark execution

```powershell
.\.venv\Scripts\python.exe benchmarks\run_benchmarks.py
```

Result: exit code 0.

| Resolution | Representation | Steps | Runtime |
|---:|---|---:|---:|
| 32 | SUBFACE_SPARSE | 40 | 0.361274 s |
| 32 | MODAL_STREAM | 40 | 2.457234 s |
| 64 | SUBFACE_SPARSE | 80 | 3.041031 s |
| 64 | MODAL_STREAM | 80 | 8.822186 s |

These are host-specific execution observations, not general speed claims.

### Build

```powershell
.\.venv\Scripts\python.exe -m build --wheel --no-isolation
```

Result: **PASS**. The wheel contains `taylanus_core.py`, all `taylang`
modules, both engine modules, metadata and the `tay` console entry point.

### Clean installed-wheel smoke

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools\run_windows_smoke.ps1 -BasePython "C:\Users\TAYLAN\anaconda3\python.exe" -Wheel "<repo>\dist\taylang-0.8.0.dev2-py3-none-any.whl" -Venv "<repo>\.smoke-final-venv" -Workspace "<workspace>\TAY_WHEEL_SMOKE_FINAL" -CfdExample "<repo>\examples\taylanus_vortex.tay"
```

Result: **WINDOWS_WHEEL_SMOKE=PASS**.

Verified:

- `tay --version`;
- `tay doctor`;
- `tay init`;
- starter `tay run`;
- notebook;
- scripted REPL;
- installed `tay` running the real CFD example;
- starter NPY/PNG/report artifacts;
- `taylang.__file__` and `taylanus_core.__file__` both under the fresh
  venv's `Lib\site-packages`.

After the final compiler compatibility guard was added, the rebuilt wheel was
force-reinstalled into that isolated venv and checked again from the external
workspace. Both imports remained under `site-packages`,
`FINAL_WHEEL_SITE_PACKAGES_COMPILER_AGG=PASS`, the full CFD command returned
exit code 0, the installed starter plot ran headlessly, and `pip check`
reported no broken requirements.

### Windows installer

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File install_windows.ps1
```

Result: **PASS**.

The script rejected the inaccessible Windows Store Python alias, automatically
selected `C:\Users\TAYLAN\anaconda3\python.exe`, created `.tay-venv`,
installed `.[taylanus]`, produced `tay.exe`, and passed version/doctor.
An explicit `-Python` run also passed.

## Installed CFD demonstration

Command:

```powershell
.\.smoke-final-venv\Scripts\tay.exe run examples\taylanus_vortex.tay --quiet
```

Configuration:

- localized periodic incompressible vortex, Re=300 reference setup;
- equivalent resolution: 32;
- `DT=0.005`;
- `TEND=0.20`;
- 40 steps;
- `MODE=AUTO`;
- viscosity: 1/300.

Actual result from the installed wheel:

| Diagnostic | Value |
|---|---:|
| Status | PASS |
| Adaptive leaves | 2,402 |
| Selected representation | SUBFACE_SPARSE |
| State/subface DOFs | 29,364 / 29,364 |
| Geometry compile | 0.0814584000 s |
| Subface IR compile | 12.3814157000 s |
| Modal setup during planning | 1.6339472999 s |
| Topology + IR setup | 12.9217556000 s |
| Planner | 2.2166948000 s |
| 40-step measured runtime | 1.0219659001 s |
| Initial kinetic energy | 0.00028426737222169557 |
| Final kinetic energy | 0.000279597493998419 |
| RMS conservative divergence | 1.8287457244460903e-17 |
| Maximum conservative divergence | 1.946972788330584e-15 |
| Displayed-slice maximum divergence | 2.5870121187473043e-16 |
| Relative reference L2 | 0.05311733082469636 |

The AUTO planner measured both representations. It selected
`SUBFACE_SPARSE` because the modal setup cost outweighed its measured
per-step advantage for this 40-step cold run. `MODAL_STREAM` remains
available through `MODE COMPACT`.

## Output files

Directory: `outputs/taylanus_vortex/`

- `diagnostics.json`
- `kinetic_energy.csv`
- `kinetic_energy_history.png`
- `mesh_level_slice.png`
- `level_distribution.png`
- `velocity_magnitude_slice.png`
- `vorticity_magnitude_slice.png`
- `divergence_slice.png`
- `velocity_final.npy`
- `mesh_levels.npy`
- `conservative_divergence.npy`

All files are non-empty. The NPY arrays were loaded during tests and checked
for expected shape and finite values. The PNG files were opened for visual
inspection; titles, axes and colorbars are present.

Pressure is used internally by projection, but the frozen public core does not
expose a pressure field/history. Therefore no pressure plot was created:
`pressure_field_available=false`,
`pressure_plot_written=false`.

## Files added or changed

- Packaging/entry: `pyproject.toml`, `taylang/version.py`,
  `taylang/__init__.py`, `.gitignore`.
- Language/runtime: `taylang/core.py`, `taylang/compiler.py`,
  `taylang/cli.py`, `taylang/session.py`.
- New engine package: `taylang/engines/__init__.py`,
  `taylang/engines/taylanus.py`.
- Preserved TAY modules: `taylang/backends.py`,
  `taylang/backend_runtime.py`, `taylang/numpy_backend.py`,
  `taylang/table.py`, `taylang/__main__.py`.
- CLI compatibility scripts: `tayrun.py`, `tayrepl.py`, `taynb.py`.
- Examples: preserved TAY examples plus `examples/taylanus_vortex.tay`.
- Tests: eight preserved TAY test files,
  `tests/test_taylanus_integration.py`,
  `tests/test_taylanus_release_gates.py`.
- Windows: `install_windows.ps1`, `install_windows.bat`.
- Verification tools: `tools/install_test_dependencies.ps1`,
  `tools/run_windows_smoke.ps1`.
- Documentation: `README.md`, `INSTALL.md`, `QUICKSTART_TR.md`,
  `docs/TAY_LANGUAGE_V0_8_SPEC.md`, `docs/TAYLANUS_ENGINE.md`.
- Reports: `reports/integration_audit.md`, this report.
- Build/result artifacts: `dist/taylang-0.8.0.dev2-py3-none-any.whl` and
  `outputs/taylanus_vortex/*`.

`taylanus_core.py` itself was not modified.

## Problems found and repaired

1. Separate packages and no installed import path: fixed by packaging the
   existing core module with TAY and adding an optional CPU extra.
2. No engine abstraction: fixed with a registry and explicit
   `ENGINE TAYLANUS` semantics.
3. Interpreter/compiler mismatch risk: engine directives were implemented and
   tested in both paths.
4. Windows Store alias selected as Python: fixed by execution-based probing and
   common Conda fallbacks.
5. Installer forced an unnecessary pip upgrade: made opt-in.
6. Missing engine diagnostics/artifacts: added structured results and eleven
   reproducible files.
7. Ambiguous divergence visualization: the final plot uses conservative
   face-flux divergence mapped from adaptive leaves.
8. GPU ambiguity: engine status and execution are explicitly fail-closed.
9. Matplotlib could select a broken Windows Tk GUI during batch tests: both
   interpreter and generated-Python plot paths now force the headless Agg
   renderer.

Intermediate development checks exposed four Windows path-literal failures in
the new tests, one missing local wheel-build dependency, and the order-sensitive
Tk renderer failure. All were repaired; the final suite has zero failures and
zero skips.

## Fluid-workflow interpretation

TAY now acts as a compact, reviewable scientific experiment description.
TAYLANUS remains the numerical machinery. That means researchers can change
resolution, time step, viscosity, planning objective, reference field and
output location without editing the CFD implementation.

The most valuable next uses are controlled parameter sweeps and comparison
reports across FAST/COMPACT modes. Scientifically meaningful expansion should
add new validated flows and reference gates before adding more surface syntax:
Taylor–Green vortex, lid-driven cavity, channel flow, longer-time convergence,
additional boundary conditions, exposed pressure results, then accelerator
validation.

## Remaining limitations

- TAY is still pre-alpha/developer preview.
- TAYLANUS is a research core, not production CFD.
- Only equivalent resolutions 32 and 64 are accepted.
- The bridge limits `TEND` to the validated 0.20 horizon.
- The reference case is periodic and favorable to local adaptivity.
- Modal compression is real; general modal CPU speed superiority is not.
- Pressure output is unavailable.
- TAYLANUS GPU/Torch execution remains unvalidated and disabled.
- No ANSYS equivalence or general solver superiority is claimed.

## Merge readiness

All requested technical gates are green on the feature branch. The branch is
ready for human review and merge consideration. `main` has not been changed
or merged.
