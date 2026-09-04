import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from taylang import run_tay

def explicit_acc26(a):
    p=np.pad(a,1)
    nx,ny,nz=a.shape
    s=np.zeros_like(a,dtype=float)
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            for dz in (-1,0,1):
                if (dx,dy,dz)==(0,0,0): continue
                s += p[1+dx:1+dx+nx,1+dy:1+dy+ny,1+dz:1+dz+nz]
    return s

def test_acc26():
    rng=np.random.default_rng(1)
    a=rng.normal(size=(6,6,6))
    got=run_tay("B = ACC26(A)",{"A":a})["B"]
    assert np.allclose(got,explicit_acc26(a))

def test_diffusion_repeat_next_commit():
    rng=np.random.default_rng(2)
    a=rng.normal(size=(5,5,5)); alpha=.1
    src="""
    REPEAT 3
        NEXT A = A + alpha*(AVG26(A)-A)
        COMMIT
    END
    """
    got=run_tay(src,{"A":a.copy(),"alpha":alpha})["A"]
    exp=a.copy()
    for _ in range(3):
        exp=exp+alpha*(explicit_acc26(exp)/26.0-exp)
    assert np.allclose(got,exp)

def test_next_is_synchronous():
    out=run_tay("NEXT A = B\nNEXT B = A\nCOMMIT",{"A":1.0,"B":2.0})
    assert out["A"]==2.0 and out["B"]==1.0

def test_scale_and_reduce():
    a=np.arange(64).reshape(4,4,4)
    out=run_tay("B = SCALE(A,2)\nS = SUM(A)",{"A":a})
    assert out["B"].shape==(2,2,2)
    assert out["S"]==a.sum()


def test_norm_and_clip():
    a=np.array([-3.0,4.0])
    out=run_tay("N = NORM(A)\nB = CLIP(A,-2,2)",{"A":a})
    assert abs(out["N"]-5.0)<1e-12
    assert np.allclose(out["B"],[-2,2])


def test_transpiler_matches_interpreter():
    from taylang import compile_to_python
    rng=np.random.default_rng(8)
    a=rng.normal(size=(4,4,4))
    src="""
    BOUNDARY ZERO
    REPEAT 4
        NEXT A = A + alpha*(AVG26(A)-A)
        COMMIT
    END
    S = SUM(A)
    """
    interpreted=run_tay(src,{"A":a.copy(),"alpha":.1})
    env={"A":a.copy(),"alpha":.1}
    exec(compile_to_python(src),env,env)
    assert np.allclose(env["A"],interpreted["A"])
    assert abs(env["S"]-interpreted["S"])<1e-12
