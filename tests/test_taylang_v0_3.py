import sys, math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from taylang import run_tay, compile_to_python

def exec_compiled(src,env):
    e=dict(env)
    exec(compile_to_python(src),e,e)
    return e

def test_linear():
    A=np.array([[3.,1.],[1.,2.]])
    b=np.array([9.,8.])
    src="LINEAR x = A, b"
    ref=np.linalg.solve(A,b)
    out=run_tay(src,{"A":A,"b":b})
    comp=exec_compiled(src,{"A":A,"b":b})
    assert np.allclose(out["x"],ref)
    assert np.allclose(comp["x"],ref)

def test_fit():
    X=np.c_[np.ones(5),np.arange(5,dtype=float)]
    y=2+3*np.arange(5,dtype=float)
    src="FIT beta = X, y"
    ref=np.linalg.lstsq(X,y,rcond=None)[0]
    out=run_tay(src,{"X":X,"y":y})
    comp=exec_compiled(src,{"X":X,"y":y})
    assert np.allclose(out["beta"],ref)
    assert np.allclose(comp["beta"],ref)

def test_smooth():
    x=np.array([1.,2.,3.,4.,5.])
    src="SMOOTH y = x WINDOW 3"
    p=np.pad(x,(1,1),mode="edge")
    ref=np.convolve(p,np.ones(3)/3,mode="valid")
    out=run_tay(src,{"x":x})
    comp=exec_compiled(src,{"x":x})
    assert np.allclose(out["y"],ref)
    assert np.allclose(comp["y"],ref)

def test_integrate():
    x=np.linspace(0,1,101)
    y=x*x
    src="INTEGRATE I = y OVER x"
    ref=float(np.trapezoid(y,x) if hasattr(np,"trapezoid") else np.trapz(y,x))
    out=run_tay(src,{"x":x,"y":y})
    comp=exec_compiled(src,{"x":x,"y":y})
    assert abs(out["I"]-ref)<1e-15
    assert abs(comp["I"]-ref)<1e-15

def test_stats():
    x=np.array([1.,2.,3.,4.])
    src="STATS s = x"
    ref=np.array([4,2.5,np.std(x,ddof=1),1,4],dtype=float)
    out=run_tay(src,{"x":x})
    comp=exec_compiled(src,{"x":x})
    assert np.allclose(out["s"],ref)
    assert np.allclose(comp["s"],ref)

def test_ode_rk4_scalar():
    src="""
    FUNC F(y, t)
        RETURN -y
    END
    ODE y = F(y,t) FROM 0 TO 1 STEP 0.01 METHOD RK4
    """
    out=run_tay(src,{"y":1.0})
    comp=exec_compiled(src,{"y":1.0})
    ref=math.exp(-1)
    assert abs(out["y"]-ref)<1e-9
    assert abs(comp["y"]-ref)<1e-9
    assert out["STEPS"]==100
    assert comp["STEPS"]==100

def test_optimize_scalar():
    src="""
    FUNC LOSS(x)
        RETURN (x - 3) * (x - 3)
    END
    OPTIMIZE x = LOSS(x) LR 0.1 UNTIL CHANGE < 0.000001 MAX 500
    """
    out=run_tay(src,{"x":10.0})
    comp=exec_compiled(src,{"x":10.0})
    assert abs(out["x"]-3)<1e-5
    assert abs(comp["x"]-3)<1e-5
    assert out["SOLVED"] is True
    assert comp["SOLVED"] is True

def test_optimize_vector():
    src="""
    FUNC LOSS(x)
        RETURN SUM((x - target) * (x - target))
    END
    OPTIMIZE x = LOSS(x) LR 0.2 UNTIL CHANGE < 0.000001 MAX 500
    """
    env={"x":np.array([4.,-3.]),"target":np.array([1.,2.])}
    out=run_tay(src,env)
    assert np.linalg.norm(out["x"]-env["target"])<1e-5
