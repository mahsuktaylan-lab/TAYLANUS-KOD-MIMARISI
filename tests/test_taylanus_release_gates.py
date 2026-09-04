import json
from pathlib import Path

import taylanus_core as core


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_release_manifest_gates_are_explicit():
    manifest = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text())
    assert manifest["version"] == "3.0.0-research"
    assert manifest["release_gates"]["numerical_regression"] is True
    assert manifest["release_gates"]["planner_policy"] is True
    assert manifest["release_gates"]["topology_cache"] is True
    assert manifest["release_gates"]["gpu_execution_validated"] is False


def test_planner_policy_preserves_compact_memory_and_tie_band():
    choose = core.select_representation
    assert choose("COMPACT", 1.0, 2.0, 1000, 100) == "MODAL_STREAM"
    assert (
        choose("AUTO", 1.0, 2.0, 1000, 100, memory_budget_bytes=500)
        == "MODAL_STREAM"
    )
    assert choose("AUTO", 1.0, 0.90, 1000, 100) == "MODAL_STREAM"
    assert choose("AUTO", 1.0, 0.96, 1000, 100) == "SUBFACE_SPARSE"
    assert choose("FAST", 1.0, 1.1, 1000, 100) == "SUBFACE_SPARSE"


def test_real_topology_compile_cache_hits():
    resolution = 32
    (
        core.CURRENT_NF,
        core.CURRENT_MAX_LEVEL,
        core.CURRENT_HF,
        core.CURRENT_FACE_AREA,
    ) = core.make_context(resolution)
    keys = core.build_initial_topology(resolution)
    cache = core.UnifiedCompileCache()

    first = core.UnifiedTaylanusIR(resolution, keys, cache)
    second = core.UnifiedTaylanusIR(resolution, keys, cache)

    assert first.sig == second.sig
    assert first.geom is second.geom
    assert first.sub_ir is second.sub_ir
    assert cache.misses["geometry"] == 1
    assert cache.misses["subface"] == 1
    assert cache.hits["geometry"] == 1
    assert cache.hits["subface"] == 1
    assert second.geometry_compile_s == 0.0
    assert second.subface_compile_s == 0.0


def test_generated_gpu_backend_remains_unvalidated():
    manifest = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text())
    generated = ROOT / "backends" / "taylanus_gpu_cupy_generated.py"
    assert generated.is_file()
    assert manifest["release_gates"]["gpu_execution_validated"] is False
