"""Thin TAY Language adapter for the real TAYLANUS v3 research core."""

from __future__ import annotations

import csv
import importlib
import importlib.util
import json
import math
import os
import platform
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class EngineError(RuntimeError):
    """Raised when an engine request is unavailable, unsafe, or invalid."""


_VALID_RESOLUTIONS = {32: 0.005, 64: 0.0025}
_VALID_MODES = {
    "AUTO",
    "FAST",
    "COMPACT",
    "SUBFACE_SPARSE",
    "MODAL_STREAM",
}
_RUN_LOCK = threading.RLock()
_CACHE_BY_RESOLUTION: dict[int, Any] = {}


@dataclass(frozen=True)
class TaylanusConfig:
    resolution: int = 32
    dt: float = 0.005
    tend: float = 0.20
    mode: str = "AUTO"
    viscosity: float = 1.0 / 300.0
    output: str = "outputs/taylanus_vortex"
    reference: str | None = None
    slice_index: int | None = None
    planner_reps: int = 3

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "TaylanusConfig":
        raw = {str(k).lower(): v for k, v in dict(values or {}).items()}
        known = {
            "resolution",
            "dt",
            "tend",
            "mode",
            "viscosity",
            "output",
            "reference",
            "slice_index",
            "planner_reps",
        }
        unknown = sorted(set(raw) - known)
        if unknown:
            raise EngineError(
                "Unknown TAYLANUS configuration key(s): " + ", ".join(unknown)
            )

        try:
            resolution = int(raw.get("resolution", cls.resolution))
            dt = float(raw.get("dt", _VALID_RESOLUTIONS.get(resolution, cls.dt)))
            tend = float(raw.get("tend", cls.tend))
            viscosity = float(raw.get("viscosity", cls.viscosity))
            planner_reps = int(raw.get("planner_reps", cls.planner_reps))
            slice_raw = raw.get("slice_index")
            slice_index = None if slice_raw is None else int(slice_raw)
        except (TypeError, ValueError) as exc:
            raise EngineError(f"Invalid numeric TAYLANUS configuration: {exc}") from exc

        mode = str(raw.get("mode", cls.mode)).strip().upper()
        output = str(raw.get("output", cls.output))
        reference_raw = raw.get("reference")
        reference = None if reference_raw in (None, "", "NONE") else str(reference_raw)

        if resolution not in _VALID_RESOLUTIONS:
            raise EngineError(
                "RESOLUTION must be 32 or 64; these are the v3 validated "
                "equivalent resolutions."
            )
        if not math.isfinite(dt) or dt <= 0:
            raise EngineError("DT must be a finite positive number.")
        stable_dt = _VALID_RESOLUTIONS[resolution]
        if dt > stable_dt * (1.0 + 1e-12):
            raise EngineError(
                f"DT={dt:g} exceeds the validated limit {stable_dt:g} "
                f"for RESOLUTION {resolution}."
            )
        if not math.isfinite(tend) or tend <= 0:
            raise EngineError("TEND must be a finite positive number.")
        if tend > 0.20 * (1.0 + 1e-12):
            raise EngineError(
                "TEND exceeds the v3 short-time validation envelope (0.20)."
            )
        steps_float = tend / dt
        steps = int(round(steps_float))
        if steps < 1 or not math.isclose(
            steps_float, steps, rel_tol=1e-10, abs_tol=1e-10
        ):
            raise EngineError("TEND must be an integer multiple of DT.")
        if mode not in _VALID_MODES:
            raise EngineError(
                "MODE must be AUTO, FAST, COMPACT, SUBFACE_SPARSE, "
                "or MODAL_STREAM."
            )
        if not math.isfinite(viscosity) or viscosity <= 0:
            raise EngineError("VISCOSITY must be a finite positive number.")
        if not output.strip():
            raise EngineError("OUTPUT must name a non-empty directory.")
        if slice_index is not None and not 0 <= slice_index < resolution:
            raise EngineError(
                f"SLICE must be between 0 and {resolution - 1}."
            )
        if not 1 <= planner_reps <= 20:
            raise EngineError("PLANNER_REPS must be between 1 and 20.")

        return cls(
            resolution=resolution,
            dt=dt,
            tend=tend,
            mode=mode,
            viscosity=viscosity,
            output=output,
            reference=reference,
            slice_index=slice_index,
            planner_reps=planner_reps,
        )


def _load_core():
    try:
        return importlib.import_module("taylanus_core")
    except Exception as exc:
        raise EngineError(
            "TAYLANUS engine is installed but its CPU runtime could not be "
            "loaded. Install the combined package with the 'taylanus' extra "
            "(SciPy and Numba are required). "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc


def taylanus_status() -> dict:
    spec = importlib.util.find_spec("taylanus_core")
    status: dict[str, Any] = {
        "installed": spec is not None,
        "available": False,
        "version": "3.0.0-research",
        "device": "cpu",
        "supported_tay_backends": ["NUMPY"],
        "gpu_validated": False,
    }
    if spec is None:
        status["error"] = "taylanus_core module is not installed"
        return status
    try:
        core = _load_core()
        import numba
        import pandas
        import scipy

        status.update(
            {
                "available": True,
                "module": str(Path(core.__file__).resolve()),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "numba": numba.__version__,
                "pandas": pandas.__version__,
            }
        )
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
    return status


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _cell_derivatives(velocity: np.ndarray, spacing: float) -> np.ndarray:
    deriv = np.empty(velocity.shape + (3,), dtype=float)
    for component in range(3):
        for axis in range(3):
            deriv[..., component, axis] = (
                np.roll(velocity[..., component], -1, axis=axis)
                - np.roll(velocity[..., component], 1, axis=axis)
            ) / (2.0 * spacing)
    return deriv


def _field_products(velocity: np.ndarray, spacing: float):
    derivatives = _cell_derivatives(velocity, spacing)
    vorticity = np.stack(
        [
            derivatives[..., 2, 1] - derivatives[..., 1, 2],
            derivatives[..., 0, 2] - derivatives[..., 2, 0],
            derivatives[..., 1, 0] - derivatives[..., 0, 1],
        ],
        axis=-1,
    )
    speed = np.linalg.norm(velocity, axis=-1)
    vorticity_magnitude = np.linalg.norm(vorticity, axis=-1)
    return speed, vorticity_magnitude


def _conservative_divergence(face, geom) -> np.ndarray:
    """Map the solver's finite-volume leaf divergence to the finest grid."""
    divergence_numerator = np.zeros(len(geom.keys), dtype=float)
    for component, geometry in zip(face, geom.faces):
        flux = component * geom.face_area
        np.add.at(divergence_numerator, geometry["ia"], +flux)
        np.add.at(divergence_numerator, geometry["ib"], -flux)
    leaf_divergence = divergence_numerator / geom.volumes
    return leaf_divergence[geom.owner]


def _plot_slice(
    data: np.ndarray,
    index: int,
    path: Path,
    title: str,
    colorbar_label: str,
    cmap: str = "viridis",
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise EngineError(
            "Matplotlib is required to write TAYLANUS diagnostic plots."
        ) from exc

    domain = 2.0 * np.pi
    fig, axis = plt.subplots(figsize=(6.4, 5.2))
    image = axis.imshow(
        data[:, :, index].T,
        origin="lower",
        extent=(0.0, domain, 0.0, domain),
        aspect="equal",
        cmap=cmap,
    )
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label(colorbar_label)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_level_distribution(levels: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unique, counts = np.unique(levels, return_counts=True)
    fig, axis = plt.subplots(figsize=(6.4, 4.4))
    axis.bar(unique.astype(str), counts, color="#4472C4")
    axis.set_xlabel("Octree level")
    axis.set_ylabel("Leaf count")
    axis.set_title("TAYLANUS adaptive mesh level distribution")
    for x, count in zip(range(len(unique)), counts):
        axis.text(x, int(count), str(int(count)), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_energy(history: list[dict[str, float]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6.4, 4.4))
    axis.plot(
        [row["time"] for row in history],
        [row["kinetic_energy"] for row in history],
        marker="o",
        markersize=2.5,
        linewidth=1.4,
    )
    axis.set_xlabel("Physical time")
    axis.set_ylabel("Mean kinetic energy")
    axis.set_title("TAYLANUS kinetic energy history")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_outputs(
    *,
    core,
    config: TaylanusConfig,
    base_dir: Path,
    ir,
    face,
    velocity: np.ndarray,
    history: list[dict[str, float]],
    diagnostics: dict[str, Any],
) -> dict[str, str]:
    output_dir = _resolve_path(base_dir, config.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    mpl_config = Path(tempfile.gettempdir()) / "taylang-matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    index = config.slice_index
    if index is None:
        index = config.resolution // 2

    levels = ir.geom.levels[ir.geom.owner]
    speed, vorticity_magnitude = _field_products(velocity, ir.geom.hf)
    conservative_divergence = _conservative_divergence(face, ir.geom)

    paths = {
        "mesh_level_slice": output_dir / "mesh_level_slice.png",
        "level_distribution": output_dir / "level_distribution.png",
        "velocity_magnitude_slice": output_dir / "velocity_magnitude_slice.png",
        "vorticity_magnitude_slice": output_dir
        / "vorticity_magnitude_slice.png",
        "divergence_slice": output_dir / "divergence_slice.png",
        "kinetic_energy_history": output_dir / "kinetic_energy_history.png",
        "energy_csv": output_dir / "kinetic_energy.csv",
        "diagnostics_json": output_dir / "diagnostics.json",
        "velocity_npy": output_dir / "velocity_final.npy",
        "mesh_levels_npy": output_dir / "mesh_levels.npy",
        "divergence_npy": output_dir / "conservative_divergence.npy",
    }

    _plot_slice(
        levels,
        index,
        paths["mesh_level_slice"],
        f"Adaptive mesh level, z-index {index}",
        "Octree level",
        cmap="viridis",
    )
    _plot_level_distribution(ir.geom.levels, paths["level_distribution"])
    _plot_slice(
        speed,
        index,
        paths["velocity_magnitude_slice"],
        f"Velocity magnitude, z-index {index}",
        "|u|",
        cmap="magma",
    )
    _plot_slice(
        vorticity_magnitude,
        index,
        paths["vorticity_magnitude_slice"],
        f"Vorticity magnitude, z-index {index}",
        "|curl u|",
        cmap="inferno",
    )
    abs_limit = float(np.max(np.abs(conservative_divergence[:, :, index])))
    _plot_slice(
        conservative_divergence,
        index,
        paths["divergence_slice"],
        f"Conservative leaf divergence, z-index {index}",
        "finite-volume div(u)",
        cmap="coolwarm",
    )
    _plot_energy(history, paths["kinetic_energy_history"])

    with paths["energy_csv"].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "time", "kinetic_energy"])
        writer.writeheader()
        writer.writerows(history)

    np.save(paths["velocity_npy"], velocity)
    np.save(paths["mesh_levels_npy"], levels)
    np.save(paths["divergence_npy"], conservative_divergence)
    diagnostics["conservative_divergence_slice_max_abs"] = abs_limit
    diagnostics["outputs"] = {key: str(path) for key, path in paths.items()}
    paths["diagnostics_json"].write_text(
        json.dumps(_json_safe(diagnostics), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {key: str(path) for key, path in paths.items()}


class TaylanusEngine:
    name = "TAYLANUS"

    def run(
        self,
        values: Mapping[str, Any] | None,
        *,
        base_dir: str | Path = ".",
        backend: str = "NUMPY",
    ) -> dict[str, Any]:
        normalized_backend = str(backend).upper()
        if normalized_backend != "NUMPY":
            raise EngineError(
                "TAYLANUS v3 execution is validated only on its CPU "
                "NumPy/Numba/SciPy path. BACKEND "
                f"{normalized_backend} is not a TAYLANUS GPU/Torch engine and "
                "will not fall back silently."
            )

        config = TaylanusConfig.from_mapping(values)
        base_path = Path(base_dir).resolve()
        core = _load_core()
        nsteps = int(round(config.tend / config.dt))

        with _RUN_LOCK:
            previous_viscosity = core.nu
            try:
                core.nu = config.viscosity
                setup_start = time.perf_counter()
                core.CURRENT_NF, core.CURRENT_MAX_LEVEL, core.CURRENT_HF, (
                    core.CURRENT_FACE_AREA
                ) = core.make_context(config.resolution)
                keys = core.build_initial_topology(config.resolution)
                cache = _CACHE_BY_RESOLUTION.setdefault(
                    config.resolution, core.UnifiedCompileCache()
                )
                cache_before = {
                    "hits": dict(cache.hits),
                    "misses": dict(cache.misses),
                }
                ir = core.UnifiedTaylanusIR(config.resolution, keys, cache)
                topology_ir_setup_s = time.perf_counter() - setup_start

                planner_rows: list[dict[str, Any]] = []
                planner_metrics: dict[str, Any] = {}
                planner_start = time.perf_counter()
                if config.mode == "AUTO":
                    representation, table, metrics = ir.plan(
                        config.dt, nsteps, objective="AUTO", reps=config.planner_reps
                    )
                    planner_rows = table.to_dict(orient="records")
                    planner_metrics = {
                        key: asdict(value) for key, value in metrics.items()
                    }
                elif config.mode in ("FAST", "SUBFACE_SPARSE"):
                    representation = "SUBFACE_SPARSE"
                else:
                    representation = "MODAL_STREAM"
                planner_s = time.perf_counter() - planner_start

                face = ir.make_initial_subface(config.dt)
                if representation == "MODAL_STREAM":
                    state = ir.subface_to_modal(face, config.dt)
                    face = ir.modal_to_subface(state)
                else:
                    state = face

                history = [
                    {
                        "step": 0,
                        "time": 0.0,
                        "kinetic_energy": core.energy_face(
                            face, ir.geom, ir.sub_ir
                        ),
                    }
                ]
                runtime_start = time.perf_counter()
                for step in range(1, nsteps + 1):
                    if representation == "SUBFACE_SPARSE":
                        state = ir.step_subface(state, config.dt)
                        face = state
                    else:
                        state = ir.step_modal(state, config.dt)
                        face = ir.modal_to_subface(state)
                    history.append(
                        {
                            "step": step,
                            "time": step * config.dt,
                            "kinetic_energy": core.energy_face(
                                face, ir.geom, ir.sub_ir
                            ),
                        }
                    )
                runtime_s = time.perf_counter() - runtime_start

                velocity = core.cell_velocity_fast(face, ir.geom, ir.sub_ir)
                rms_divergence, max_divergence = core.subface_divergence(
                    face, ir.geom
                )
                subface_dofs = int(sum(len(component) for component in face))
                state_dofs = int(sum(len(component) for component in state))
                compression = (
                    float(subface_dofs / state_dofs) if state_dofs else None
                )

                reference_error = None
                reference_path = None
                if config.reference:
                    candidate = _resolve_path(base_path, config.reference)
                    if not candidate.is_file():
                        raise EngineError(
                            f"REFERENCE file does not exist: {candidate}"
                        )
                    reference = np.load(candidate, allow_pickle=False)
                    if reference.shape != velocity.shape:
                        raise EngineError(
                            "REFERENCE shape does not match reconstructed "
                            f"velocity: {reference.shape} != {velocity.shape}"
                        )
                    reference_error = float(
                        np.linalg.norm(velocity - reference)
                        / np.linalg.norm(reference)
                    )
                    reference_path = str(candidate)

                diagnostics: dict[str, Any] = {
                    "engine": self.name,
                    "engine_version": "3.0.0-research",
                    "status": "PASS",
                    "config": asdict(config),
                    "resolution": config.resolution,
                    "steps": nsteps,
                    "leaf_count": int(len(keys)),
                    "representation": representation,
                    "state_dofs": state_dofs,
                    "subface_dofs": subface_dofs,
                    "compression_vs_subface": compression,
                    "geometry_compile_s": float(ir.geometry_compile_s),
                    "subface_compile_s": float(ir.subface_compile_s),
                    "modal_setup_s": (
                        None
                        if ir.modal_setup_s is None
                        else float(ir.modal_setup_s)
                    ),
                    "topology_ir_setup_s": topology_ir_setup_s,
                    "planner_s": planner_s,
                    "runtime_s": runtime_s,
                    "kinetic_energy_initial": history[0]["kinetic_energy"],
                    "kinetic_energy_final": history[-1]["kinetic_energy"],
                    "rms_divergence": rms_divergence,
                    "max_divergence": max_divergence,
                    "reference_relative_l2": reference_error,
                    "reference_path": reference_path,
                    "planner": planner_rows,
                    "planner_metrics": planner_metrics,
                    "cache_before": cache_before,
                    "cache_after": {
                        "hits": dict(cache.hits),
                        "misses": dict(cache.misses),
                    },
                    "pressure_field_available": False,
                    "pressure_plot_written": False,
                    "backend": {
                        "tay_backend": normalized_backend,
                        "taylanus_runtime": "CPU NumPy/Numba/SciPy",
                        "gpu_validated": False,
                    },
                    "environment": {
                        "python": sys.version.split()[0],
                        "executable": sys.executable,
                        "platform": platform.platform(),
                        "numpy": np.__version__,
                    },
                }
                outputs = _write_outputs(
                    core=core,
                    config=config,
                    base_dir=base_path,
                    ir=ir,
                    face=face,
                    velocity=velocity,
                    history=history,
                    diagnostics=diagnostics,
                )
                diagnostics["outputs"] = outputs
                return _json_safe(diagnostics)
            finally:
                core.nu = previous_viscosity

    @staticmethod
    def environment_values(result: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "CFD_RESULT": dict(result),
            "CFD_STATUS": result["status"],
            "CFD_OUTPUT_DIR": str(Path(result["outputs"]["diagnostics_json"]).parent),
            "CFD_REPRESENTATION": result["representation"],
            "CFD_LEAF_COUNT": result["leaf_count"],
            "CFD_STATE_DOFS": result["state_dofs"],
            "CFD_RMS_DIVERGENCE": result["rms_divergence"],
            "CFD_MAX_DIVERGENCE": result["max_divergence"],
            "CFD_KINETIC_ENERGY": result["kinetic_energy_final"],
            "CFD_REFERENCE_L2": result["reference_relative_l2"],
        }


__all__ = [
    "EngineError",
    "TaylanusConfig",
    "TaylanusEngine",
    "taylanus_status",
]
