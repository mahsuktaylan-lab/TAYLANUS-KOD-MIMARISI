# TAY v0.8 + TAYLANUS v3 Integration Audit

Date: 2026-09-04
Integration branch: `feature/tay-taylanus-integration`
Frozen TAYLANUS `main` starting commit: `d5a4a944435f9f27b95a85f7a99867b0fdff3e57`

This audit was written before any production-code integration change. The
branch starts from a clean `main` worktree. No merge into `main` is part of
this work.

## Source packages examined

### TAY Language v0.8 Developer Preview

- Source: `C:\Users\TAYLAN\Desktop\TAYLANG_v0_8_Developer_Preview`
- Package: `taylang`
- Declared version: `0.8.0.dev1`
- Build backend: setuptools / PEP 517
- Console entry point: `tay = taylang.cli:main`
- Declared Python: 3.10+
- Required dependencies: NumPy, pandas, Matplotlib
- Optional dependencies: PyTorch and pytest
- Existing tests: eight files, v0.1 through v0.8
- Existing distribution: `dist/taylang-0.8.0.dev1-py3-none-any.whl`

The interpreter is line-oriented. `_clean` strips comments and blank lines,
`_parse` builds a small tuple AST, and `TAYRuntime` executes the AST. The
Python transpiler independently emits Python from the same tuple AST. Runtime
backend selection is handled by `BACKEND NUMPY|TORCH|GPU`.

### TAYLANUS v3 stable research core

- Repository: `mahsuktaylan-lab/TAYLANUS-KOD-MIMARISI`
- Core module: `taylanus_core.py`
- Release identity: `3.0.0-research`
- CPU dependencies: NumPy, SciPy, Numba, pandas
- Optional/unvalidated GPU dependency: CuPy
- Validated equivalent resolutions: 32 and 64
- Reference problem: localized periodic incompressible vortex, Re=300
- Representations: `SUBFACE_SPARSE` and `MODAL_STREAM`

The public integration seam is the existing `UnifiedTaylanusIR` API:
topology is built with `build_initial_topology`, a shared
`UnifiedCompileCache` is supplied, `plan` selects a representation, initial
state is produced with `make_initial_subface`, and the real time loop calls
`step_subface` or `step_modal`. The existing
`cell_velocity_fast`, divergence helpers, and energy helper expose the
diagnostics needed by a language adapter without duplicating the solver.

## Baseline evidence already present

The checked-in TAY report records 83/83 tests passing under Python 3.13.5 with
PyTorch 2.10 CPU, an installed-wheel smoke test, and a fail-closed CUDA result.
Those records are historical evidence only; this branch will rerun the
available gates.

The current TAYLANUS checkout was independently rerun before this audit:

- NF=32, SUBFACE_SPARSE: L2 0.05311733, DOFs 29364, RMS divergence
  1.829e-17
- NF=32, MODAL_STREAM: L2 0.06270016, DOFs 11329, RMS divergence
  1.607e-17
- NF=64, SUBFACE_SPARSE: L2 0.04467618, DOFs 125520, RMS divergence
  3.073e-17
- NF=64, MODAL_STREAM: L2 0.05899064, DOFs 33926, RMS divergence
  2.878e-17

Result: `TAYLANUS v3.0 numerical regression: PASS`.

## Problems and risks found

1. TAY and TAYLANUS are separate source trees; the TAY wheel cannot import the
   solver from `site-packages`.
2. TAY has no engine abstraction. Reusing `BACKEND` for TAYLANUS would mix
   array/device choice with solver choice and would be semantically wrong.
3. TAY's parser, interpreter, CLI and transpiler know nothing about an
   `ENGINE` directive or engine configuration.
4. TAYLANUS is a single repository-root module and has no install metadata.
   It must be included as an installed module; a `sys.path` workaround is not
   acceptable.
5. TAYLANUS relies on module-level grid context variables. The adapter must set
   them deliberately before topology/IR construction and must serialize access
   or otherwise avoid cross-run context corruption.
6. TAYLANUS `plan` benchmarks both representations. This is accurate but can
   dominate short runs; explicit modes should not benchmark unnecessarily.
7. The existing Windows installer discovers `python` with
   `Get-Command`, but on this machine that resolves to an inaccessible
   Windows Store alias. A real interpreter probe is required.
8. The Windows installer always upgrades pip, making installation unnecessarily
   network-dependent.
9. This Windows host currently has a usable Python 3.14.6 environment with the
   TAYLANUS CPU stack, but pytest/Matplotlib/PyTorch were not initially present.
   The inaccessible Store alias is the default `python` command.
10. TAY v0.8 tests import the project root by editing `sys.path`; therefore a
    separate installed-wheel smoke is essential to prove packaging.
11. TAY's compiler duplicates execution semantics. Engine execution should be
    an explicit generated call or should fail clearly; silent fallback to the
    normal numerical runtime would be unsafe.
12. TAYLANUS pressure projection is internal. No pressure-history plot can be
    promised unless a real pressure field is exposed by the existing API.
13. GPU support is unvalidated in TAYLANUS v3. The bridge must reject GPU/Torch
    solver execution instead of falling back to CPU.

## Integration decision

The integration keeps the two axes separate:

```text
BACKEND NUMPY|TORCH|GPU     array/device execution for ordinary TAY programs
ENGINE TAYLANUS             selects the CFD solver engine
MODE AUTO|FAST|COMPACT|
     SUBFACE_SPARSE|
     MODAL_STREAM           selects/plans TAYLANUS representation
```

The combined distribution will:

- retain the real `taylanus_core.py` implementation as the single source of
  solver truth;
- package that module alongside `taylang`;
- add a thin adapter under `taylang.engines`;
- add a dedicated CFD configuration block to the TAY parser/runtime;
- expose engine status in `tay doctor`;
- generate diagnostics and plots only from actual solver state;
- keep CPU integration independent of optional Torch/CUDA support;
- preserve the existing TAY and TAYLANUS regression suites.

## Planned verification gates

1. Rerun the complete TAY suite and record pass/fail/skip counts.
2. Rerun TAYLANUS regression and benchmark/release gates.
3. Build a wheel and install it into a fresh virtual environment.
4. Verify `taylang` and `taylanus_core` resolve from `site-packages`.
5. Run version, doctor, init, example, notebook and scripted REPL smoke tests.
6. Run bridge unit, invalid-input, fail-closed backend, end-to-end CFD, artifact
   and finite/divergence tests.
7. Run the real NF=32 localized-vortex example to t=0.20 and retain numerical
   JSON/CSV/PNG artifacts.

The work is ready to proceed on the feature branch; it is not yet ready to
merge.
