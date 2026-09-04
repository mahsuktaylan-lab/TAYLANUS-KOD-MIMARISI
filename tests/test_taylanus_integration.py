import json
import os
from pathlib import Path

import numpy as np
import pytest

from taylang import TAYError, compile_to_python, run_tay
from taylang.engines.taylanus import (
    EngineError,
    TaylanusConfig,
    TaylanusEngine,
    taylanus_status,
)


def quick_source(output, mode="FAST"):
    output = Path(output).as_posix()
    return f"""
BACKEND NUMPY
ENGINE TAYLANUS
RESOLUTION 32
DT 0.005
TEND 0.005
MODE {mode}
VISCOSITY 0.0033333333333333335
SLICE 16
PLANNER_REPS 1
OUTPUT "{output}"
REFERENCE NONE
RUN TAYLANUS
"""


def test_taylanus_config_validation():
    config = TaylanusConfig.from_mapping(
        {"resolution": 32, "dt": 0.005, "tend": 0.01, "mode": "compact"}
    )
    assert config.resolution == 32
    assert config.mode == "COMPACT"

    invalid = [
        ({"resolution": 16}, "32 or 64"),
        ({"resolution": 32, "dt": 0}, "positive"),
        ({"resolution": 32, "dt": 0.006}, "validated limit"),
        ({"resolution": 32, "dt": 0.004, "tend": 0.005}, "integer multiple"),
        ({"resolution": 32, "tend": 0.25}, "validation envelope"),
        ({"resolution": 32, "mode": "GPU"}, "MODE must be"),
    ]
    for values, message in invalid:
        with pytest.raises(EngineError, match=message):
            TaylanusConfig.from_mapping(values)


def test_engine_run_requires_selection():
    with pytest.raises(TAYError, match="requires ENGINE TAYLANUS"):
        run_tay("RUN TAYLANUS")


def test_unknown_engine_fails_closed():
    with pytest.raises(TAYError, match="Unknown TAY engine"):
        run_tay("ENGINE NOT_A_SOLVER")


def test_engine_words_remain_normal_variables_without_engine():
    source = "DT = 2\nMODE = 3\nOUTPUT = DT + MODE"
    interpreted = run_tay(source)
    generated = {}
    exec(compile_to_python(source), generated, generated)
    assert interpreted["OUTPUT"] == 5
    assert generated["OUTPUT"] == 5


@pytest.mark.parametrize("backend", ["TORCH", "GPU"])
def test_taylanus_rejects_unvalidated_backend(backend, tmp_path):
    with pytest.raises(EngineError, match="will not fall back silently"):
        TaylanusEngine().run(
            {
                "resolution": 32,
                "dt": 0.005,
                "tend": 0.005,
                "mode": "FAST",
                "output": str(tmp_path / "unused"),
            },
            backend=backend,
        )
    assert not (tmp_path / "unused").exists()


def test_engine_status_reports_real_cpu_stack():
    status = taylanus_status()
    assert status["installed"] is True
    assert status["available"] is True
    assert status["device"] == "cpu"
    assert status["gpu_validated"] is False
    assert status["supported_tay_backends"] == ["NUMPY"]
    assert Path(status["module"]).name == "taylanus_core.py"


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("FAST", "SUBFACE_SPARSE"),
        ("COMPACT", "MODAL_STREAM"),
    ],
)
def test_real_engine_modes_write_finite_outputs(tmp_path, mode, expected):
    output = tmp_path / mode.lower()
    env = run_tay(quick_source(output, mode=mode), base_dir=tmp_path)
    result = env["CFD_RESULT"]

    assert result["status"] == "PASS"
    assert result["representation"] == expected
    assert result["steps"] == 1
    assert result["leaf_count"] > 0
    assert result["state_dofs"] > 0
    assert np.isfinite(result["kinetic_energy_final"])
    assert np.isfinite(result["rms_divergence"])
    assert np.isfinite(result["max_divergence"])
    assert result["rms_divergence"] < 1e-14
    assert result["max_divergence"] < 1e-12
    assert result["pressure_field_available"] is False
    assert result["pressure_plot_written"] is False

    required = {
        "mesh_level_slice",
        "level_distribution",
        "velocity_magnitude_slice",
        "vorticity_magnitude_slice",
        "divergence_slice",
        "kinetic_energy_history",
        "energy_csv",
        "diagnostics_json",
        "velocity_npy",
        "mesh_levels_npy",
        "divergence_npy",
    }
    assert set(result["outputs"]) == required
    for path in result["outputs"].values():
        assert Path(path).is_file()
        assert Path(path).stat().st_size > 0

    persisted = json.loads(Path(result["outputs"]["diagnostics_json"]).read_text())
    assert persisted["representation"] == expected
    velocity = np.load(result["outputs"]["velocity_npy"], allow_pickle=False)
    divergence = np.load(result["outputs"]["divergence_npy"], allow_pickle=False)
    assert velocity.shape == (32, 32, 32, 3)
    assert divergence.shape == (32, 32, 32)
    assert np.isfinite(velocity).all()
    assert np.isfinite(divergence).all()


def test_auto_mode_uses_real_planner(tmp_path):
    env = run_tay(quick_source(tmp_path / "auto", mode="AUTO"), base_dir=tmp_path)
    result = env["CFD_RESULT"]
    assert result["representation"] in {"SUBFACE_SPARSE", "MODAL_STREAM"}
    assert len(result["planner"]) == 2
    assert set(result["planner_metrics"]) == {
        "SUBFACE_SPARSE",
        "MODAL_STREAM",
    }
    assert sum(result["cache_after"]["hits"].values()) >= sum(
        result["cache_before"]["hits"].values()
    )


def test_compiled_engine_program_executes(tmp_path):
    source = quick_source(tmp_path / "compiled", mode="FAST")
    generated = compile_to_python(source)
    namespace = {}
    previous = Path.cwd()
    try:
        os.chdir(tmp_path)
        exec(generated, namespace, namespace)
    finally:
        os.chdir(previous)
    assert namespace["CFD_STATUS"] == "PASS"
    assert namespace["CFD_REPRESENTATION"] == "SUBFACE_SPARSE"
    assert Path(namespace["CFD_OUTPUT_DIR"]).is_dir()
