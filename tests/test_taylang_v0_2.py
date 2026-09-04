import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from taylang import run_tay, compile_to_python, TAYError

def explicit_acc26(a):
    p=np.pad(a,1)
    nx,ny,nz=a.shape
    s=np.zeros_like(a,dtype=float)
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            for dz in (-1,0,1):
                if (dx,dy,dz)==(0,0,0):
                    continue
                s += p[1+dx:1+dx+nx,1+dy:1+dy+ny,1+dz:1+dz+nz]
    return s

def test_function():
    a=np.arange(27,dtype=float).reshape(3,3,3)
    src="""
    FUNC F(X, k)
        RETURN X + k
    END
    B = F(A, 2)
    """
    out=run_tay(src,{"A":a})
    assert np.allclose(out["B"],a+2)

def test_if_else():
    out=run_tay("""
    X = 5
    IF X > 3
        Y = 10
    ELSE
        Y = 20
    END
    """)
    assert out["Y"]==10

def test_typed_declarations():
    out=run_tay("""
    SCALAR a = 2
    VECTOR v = [1,2,3]
    """)
    assert out["a"]==2
    assert out["v"].shape==(3,)

def test_bad_type_fails():
    try:
        run_tay("VECTOR v = 2")
        assert False
    except TAYError:
        assert True

def test_one_line_solve_converges():
    a=np.ones((4,4,4),dtype=float)
    src="""
    SCALAR eps = 0.01
    SOLVE A = A * 0.5 UNTIL CHANGE < eps MAX 100
    """
    out=run_tay(src,{"A":a})
    assert out["SOLVED"] is True
    assert out["ITER"] < 100
    assert out["CHANGE"] < out["eps"]

def test_block_solve():
    out=run_tay("""
    x = 0
    SOLVE 20 UNTIL x >= 5
        x = x + 1
    END
    """)
    assert out["x"]==5
    assert out["SOLVED"] is True
    assert out["ITER"]==5

def test_save_load_npy(tmp_path):
    base=tmp_path
    a=np.arange(12).reshape(3,4)
    src='SAVE A TO "x.npy"\nLOAD B FROM "x.npy"'
    out=run_tay(src,{"A":a},base_dir=base)
    assert np.array_equal(out["B"],a)

def test_use_module():
    base=ROOT/"examples"
    a=np.ones((3,3,3),dtype=float)
    src='USE "math_module.tay"\nB = DIFFUSE(A, alpha)'
    out=run_tay(src,{"A":a,"alpha":0.1},base_dir=base)
    exp=a+0.1*(explicit_acc26(a)/26.0-a)
    assert np.allclose(out["B"],exp)

def test_transpiler_function_if_solve():
    a=np.ones((3,3,3),dtype=float)
    src="""
    FUNC STEP(X, k)
        RETURN X * k
    END
    IF alpha < 1
        SOLVE A = STEP(A, alpha) UNTIL CHANGE < eps MAX 50
    ELSE
        A = STEP(A, alpha)
    END
    """
    ref=run_tay(src,{"A":a.copy(),"alpha":0.5,"eps":0.01})
    env={"A":a.copy(),"alpha":0.5,"eps":0.01}
    exec(compile_to_python(src),env,env)
    assert np.allclose(env["A"],ref["A"])
    assert env["SOLVED"]==ref["SOLVED"]
    assert env["ITER"]==ref["ITER"]
