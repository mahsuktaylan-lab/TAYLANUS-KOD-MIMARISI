# TAYLANUS v3.0 — Stable Research Core

TAYLANUS is a research computing architecture built around the abstraction:

**field + direction + scale + neighbor relation → mathematical operator**

v3.0 consolidates the experimentally developed v2.x line into one reproducible CPU research core.

## Validated CPU representations

- `SUBFACE_SPARSE`: sparse active face-normal DOFs; current default for `FAST`/`AUTO`.
- `MODAL_STREAM`: local face modes (mean + tangential slopes); current `COMPACT` choice.

Both use the same octree topology, sparse direction+scale+neighbor operator substrate, and compatible pressure projection.

## v3.0 release gates

All four packaged numerical regression cases passed:

- 32³-equivalent `SUBFACE_SPARSE`
- 32³-equivalent `MODAL_STREAM`
- 64³-equivalent `SUBFACE_SPARSE`
- 64³-equivalent `MODAL_STREAM`

The release also passed planner-policy and topology-cache gates.

Run:

```bash
python tests/run_regression.py
```

## Current planner policy

- `FAST` → `SUBFACE_SPARSE`
- `AUTO` → `SUBFACE_SPARSE`
- `COMPACT` → `MODAL_STREAM`

The modal branch is not claimed to be faster on CPU; its demonstrated advantage is state compression. At 64³-equivalent resolution it used 33,926 modal state DOFs versus 125,520 sparse-subface state DOFs (~3.70× compression), with an accuracy cost.

## GPU status

`backends/taylanus_gpu_cupy_generated.py` is **generated only**. It has not been executed or validated on CUDA hardware in this release.

## Scientific scope

This is a research CFD prototype. It does **not** establish:

- equivalence to ANSYS or other production CFD solvers,
- general CFD speed superiority,
- production-grade AMR/MAC correctness,
- validated GPU performance.

See `docs/KNOWN_LIMITATIONS.md`.

## Release lineage

v3.0 stabilizes the work through v2.23 rather than adding a new numerical method.
