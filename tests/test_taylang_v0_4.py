import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from taylang import run_tay, compile_to_python

def ref_ops(a,hx,hy,hz):
    p=np.pad(a,1,mode="wrap")
    nx,ny,nz=a.shape
    dx=(p[2:2+nx,1:1+ny,1:1+nz]-p[0:nx,1:1+ny,1:1+nz])/(2*hx)
    dy=(p[1:1+nx,2:2+ny,1:1+nz]-p[1:1+nx,0:ny,1:1+nz])/(2*hy)
    dz=(p[1:1+nx,1:1+ny,2:2+nz]-p[1:1+nx,1:1+ny,0:nz])/(2*hz)
    c=p[1:1+nx,1:1+ny,1:1+nz]
    lap=(p[2:2+nx,1:1+ny,1:1+nz]-2*c+p[0:nx,1:1+ny,1:1+nz])/(hx*hx)
    lap+=(p[1:1+nx,2:2+ny,1:1+nz]-2*c+p[1:1+nx,0:ny,1:1+nz])/(hy*hy)
    lap+=(p[1:1+nx,1:1+ny,2:2+nz]-2*c+p[1:1+nx,1:1+ny,0:nz])/(hz*hz)
    return dx,dy,dz,lap

def test_spacing_aware_derivatives():
    rng=np.random.default_rng(44)
    a=rng.normal(size=(6,5,4))
    hx,hy,hz=.5,.75,1.25
    ref=ref_ops(a,hx,hy,hz)
    src="""
    BOUNDARY WRAP
    X = DX(A,hx)
    Y = DY(A,hy)
    Z = DZ(A,hz)
    L = LAPLACE(A,hx,hy,hz)
    """
    out=run_tay(src,{"A":a,"hx":hx,"hy":hy,"hz":hz})
    assert np.allclose(out["X"],ref[0])
    assert np.allclose(out["Y"],ref[1])
    assert np.allclose(out["Z"],ref[2])
    assert np.allclose(out["L"],ref[3])

def test_derivative_transpiler_matches():
    rng=np.random.default_rng(45)
    a=rng.normal(size=(5,5,5))
    src="""
    BOUNDARY WRAP
    X = DX(A,0.5)
    L = LAPLACE(A,0.5,0.75,1.0)
    """
    ref=run_tay(src,{"A":a.copy()})
    env={"A":a.copy()}
    exec(compile_to_python(src),env,env)
    assert np.allclose(env["X"],ref["X"])
    assert np.allclose(env["L"],ref["L"])

def test_two_field_synchronous_reaction():
    u=np.ones((3,3,3))
    v=np.full((3,3,3),2.0)
    src="""
    R = U * V
    NEXT U = U - 0.1 * R
    NEXT V = V + 0.1 * R
    COMMIT
    """
    out=run_tay(src,{"U":u.copy(),"V":v.copy()})
    assert np.allclose(out["U"],0.8)
    assert np.allclose(out["V"],2.2)
