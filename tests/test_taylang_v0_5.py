import sys, os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from taylang import run_tay, compile_to_python, TAYError

def test_array_creation_primitives():
    out=run_tay("""
    A = ZEROS(2,3,4)
    B = ONES(2,3)
    C = FULL(7,2,2)
    X = LINSPACE(0,1,5)
    R = RANGE(1,5,1)
    """)
    assert out["A"].shape==(2,3,4)
    assert np.all(out["A"]==0)
    assert np.all(out["B"]==1)
    assert np.all(out["C"]==7)
    assert np.allclose(out["X"],np.linspace(0,1,5))
    assert np.allclose(out["R"],np.arange(1,5,1))

def test_region_assignment():
    src="""
    FIELD A = ZEROS(6,6,6)
    REGION A[2:4,1:5,3:6] = 9
    """
    out=run_tay(src)
    exp=np.zeros((6,6,6))
    exp[2:4,1:5,3:6]=9
    assert np.array_equal(out["A"],exp)

def test_region_assignment_transpiler():
    src="""
    FIELD A = ZEROS(5,5,5)
    REGION A[1:4,2:5,0:2] = 3
    """
    ref=run_tay(src)
    env={}
    exec(compile_to_python(src),env,env)
    assert np.array_equal(env["A"],ref["A"])

def test_trace():
    src="""
    x = 1
    REPEAT 4
        x = x * 2
        TRACE growth = x
    END
    """
    out=run_tay(src)
    assert np.array_equal(out["TRACE_growth"],np.array([2.,4.,8.,16.]))
    assert out["growth"]==16.0

def test_trace_transpiler():
    src="""
    x = 1
    REPEAT 3
        x = x + 1
        TRACE t = x
    END
    """
    ref=run_tay(src)
    env={}
    exec(compile_to_python(src),env,env)
    assert np.array_equal(env["TRACE_t"],ref["TRACE_t"])

def test_backend_numpy():
    out=run_tay("BACKEND NUMPY\nx=1")
    assert out["BACKEND"]=="NUMPY"

def test_unimplemented_backend_fails_closed():
    try:
        run_tay("BACKEND GPU")
        assert False
    except TAYError as e:
        msg=str(e).lower()
        assert ("not implemented" in msg) or ("cuda" in msg and ("false" in msg or "available" in msg))

def test_plot_slice_and_trace(tmp_path):
    src="""
    FIELD A = ONES(5,5,5)
    TRACE m = SUM(A)
    PLOT SLICE A Z 2 TO "slice.png"
    PLOT TRACE m TO "trace.png"
    """
    out=run_tay(src,base_dir=tmp_path)
    assert (tmp_path/"slice.png").exists()
    assert (tmp_path/"trace.png").exists()

def test_full_workflow_interpreter(tmp_path):
    src=(ROOT/"examples"/"full_scientific_workflow.tay").read_text(encoding="utf-8")
    # Reduce workload for test.
    src=src.replace("PARAM n=12, steps=25","PARAM n=8, steps=3")
    src=src.replace("REGION U[4:8,4:8,4:8] = 0.50","REGION U[2:6,2:6,2:6] = 0.50")
    src=src.replace("REGION V[4:8,4:8,4:8] = 0.25","REGION V[2:6,2:6,2:6] = 0.25")
    src=src.replace("PLOT SLICE V Z 6","PLOT SLICE V Z 4")
    out=run_tay(src,base_dir=tmp_path)
    assert out["U"].shape==(8,8,8)
    assert out["V"].shape==(8,8,8)
    assert len(out["TRACE_mass"])==3
    assert (tmp_path/"output"/"U_final.npy").exists()
    assert (tmp_path/"output"/"V_final.npy").exists()
    assert (tmp_path/"output"/"mass_trace.csv").exists()
    assert (tmp_path/"output"/"V_slice.png").exists()
    assert (tmp_path/"output"/"mass_trace.png").exists()


def test_param_group():
    out=run_tay("PARAM a=1, b=2.5, c=a+b")
    assert out["a"]==1
    assert out["b"]==2.5
    assert out["c"]==3.5

def test_param_group_transpiler():
    src="PARAM a=1, b=2, c=a+b"
    ref=run_tay(src)
    env={}
    exec(compile_to_python(src),env,env)
    assert env["a"]==ref["a"] and env["b"]==ref["b"] and env["c"]==ref["c"]
