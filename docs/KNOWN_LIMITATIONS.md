# Known Limitations — TAYLANUS v3.0

1. The primary validation problem is a short, localized, periodic incompressible vortex benchmark; it is intentionally favorable to local adaptivity.
2. Long-time testing at 32³-equivalent resolution showed material error accumulation: L2 rose to roughly 0.15 by physical time 1.0.
3. The modal branch compresses state but is less accurate than the sparse-subface branch at matched topology in current tests.
4. The modal branch still evaluates nonlinear momentum through the shared sparse subface operator; state compression and compute compression are not fully identical.
5. Dynamic topology transfer uses subface helper representation.
6. Pressure projection is compatible in tested prototypes, but this is not a production MAC-AMR implementation.
7. CPU tests use Python/Numba/SciPy research code. Results are benchmark-specific.
8. GPU/CuPy code is generated only and has not been executed or benchmarked.
9. No ANSYS equivalence, validation, or speed comparison is claimed.
10. No claim is made that TAYLANUS numerical methods such as finite differences, AMR, MUSCL/TVD, RK2, pressure projection, or modal bases are themselves novel.
