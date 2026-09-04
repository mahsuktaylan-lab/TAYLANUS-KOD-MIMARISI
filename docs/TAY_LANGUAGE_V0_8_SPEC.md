# TAY Language v0.8 — Backend Separation Specification

## Goal

v0.8 tests a central TAY architectural claim:

> TAY source semantics should be separable from the numerical execution backend.

The same TAY source can now be executed externally with:

- `NUMPY`
- `TORCH`
- `GPU` when CUDA is genuinely available

without rewriting the scientific source.

---

## 1. External backend selection

Runtime API:

```python
run_tay(source, backend="NUMPY")
run_tay(source, backend="TORCH")
run_tay(source, backend="GPU")
```

Transpiler:

```python
compile_to_python(source, backend="TORCH")
```

CLI:

```bash
python tayrun.py model.tay --backend NUMPY
python tayrun.py model.tay --backend TORCH
python tayrun.py model.tay --backend GPU
```

Notebook:

```bash
python taynb.py model.taynb --backend TORCH
```

REPL:

```bash
python tayrepl.py --backend TORCH
```

A source-level `BACKEND ...` declaration is still accepted and is authoritative
when present.

---

## 2. Implemented numerical backends

### NUMPY

Storage:

`numpy.ndarray`

Device:

CPU.

### TORCH

Storage:

`torch.Tensor`

Current reference device:

CPU.

Reference floating-field dtype:

`torch.float64`

This is a real PyTorch tensor backend, not a wrapper that converts every
operation back to NumPy.

Field constructors, relation operators, differential operators and reductions
execute through PyTorch tensor operations.

### GPU

The GPU path uses the same Torch backend with a CUDA device.

It opens only when:

```python
torch.cuda.is_available() == True
```

If CUDA is unavailable, `BACKEND GPU` fails closed.

---

## 3. Backend-dispatched semantics

v0.8 dispatches these numerical operations through the selected backend:

- `ZEROS`
- `ONES`
- `FULL`
- `LINSPACE`
- `RANGE`
- `ACC6`
- `ACC26`
- `AVG6`
- `AVG26`
- `DX`
- `DY`
- `DZ`
- `GRAD`
- `LAPLACE6`
- `LAPLACE`
- `SCALE`
- `SUM`
- `MEAN`
- `MIN`
- `MAX`
- `ABS`
- `SQRT`
- `NORM`
- `CLIP`

Normal TAY arithmetic over backend arrays uses the native array/tensor
operators of that backend.

`NEXT` / `COMMIT` therefore stages native backend values.

---

## 4. Boundaries

Both NumPy and Torch implement:

- `BOUNDARY ZERO`
- `BOUNDARY WRAP`
- `BOUNDARY EDGE`

The same finite-difference and neighborhood semantics are retained.

---

## 5. I/O and plotting

Backend numerical arrays are converted to host NumPy representation only when
needed for external I/O or plotting.

Examples:

```tay
SAVE A TO "A.npy"
PLOT SLICE A Z 10 TO "slice.png"
```

This conversion boundary is explicit in the backend runtime.

Trace history remains a host NumPy array in v0.8.

---

## 6. Tables

`TAYTable` remains a CPU/pandas-backed structured-data system.

Selecting `TORCH` or `GPU` for numerical arrays does not move tables onto a GPU.

The numerical backend and table backend are deliberately separate.

---

## 7. Backend selection timing

v0.8 rejects changing the numerical backend after numerical vector/field state
already exists.

Allowed:

```tay
BACKEND TORCH
FIELD A = ONES(64,64,64)
```

Rejected:

```tay
FIELD A = ONES(64,64,64)
BACKEND TORCH
```

Reason:

silent mixed NumPy/Torch state is more dangerous than a visible failure.

External backend selection avoids this problem entirely.

---

## 8. Externally supplied arrays

When a NumPy array is supplied through the runtime API or CLI while
`backend="TORCH"` is selected, v0.8 moves it onto the selected Torch device.

Boolean and integer dtypes are preserved where needed for masks/indexing.

Typed `FIELD` and `VECTOR` values are converted to the selected numerical
backend.

---

## 9. Interactive execution

`TAYSession`, REPL and notebook state preserve the selected backend.

Transactional rollback now clones native backend arrays/tensors rather than
forcing them through NumPy.

---

## 10. GPU validation gate

The package contains:

```bash
python gpu_validation_v0_8.py
```

Behavior:

- if CUDA is absent: `SKIP`, with no GPU claim;
- if CUDA is present:
  - run identical NumPy and GPU TAY workloads,
  - compare numerical outputs,
  - synchronize CUDA for timing,
  - report median end-to-end timings and speed ratio.

This validation script must be run on actual CUDA hardware before any TAY GPU
speed claim is accepted.

---

## 11. Current limitations

v0.8 does not claim that every historical scientific helper is fully
backend-native.

The core numerical array/PDE path is backend-dispatched.

Some higher-level v0.3 scientific helpers, such as legacy least-squares,
optimization or descriptive-statistics implementations, remain primarily
NumPy-oriented and should be revisited separately if they become priority
workloads.

No automatic kernel fusion or JIT compilation is implemented.

No GPU execution was available on the development host used for the v0.8
reference package.

---

## 12. Architectural result

The v0.8 design is:

```text
                  ┌── NumPy / CPU
TAY source ──────►│
                  ├── Torch / CPU
                  │
                  └── Torch / CUDA GPU  [when available]
```

The same TAY relation/state semantics remain above these numerical backends.
