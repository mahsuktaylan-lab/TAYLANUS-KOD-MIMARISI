# v3.0 Architecture

```text
.tay / semantic intent
        ↓
Semantic IR
        ↓
Topology signature + shared cache
        ↓
Sparse direction + scale + neighbor operator IR
        ↓
Planner
   ┌───────────────┬──────────────┐
   │ SUBFACE_SPARSE│ MODAL_STREAM │
   │ FAST/AUTO     │ COMPACT      │
   └───────────────┴──────────────┘
        ↓
Compatible pressure projection
        ↓
CPU NumPy/Numba/SciPy runtime
        ↓
Generated GPU backend hook (not validated)
```

Core local primitive:

```text
ACTIVE FACE → direction → scale → neighbor(s) + weight(s) → local operator
```
