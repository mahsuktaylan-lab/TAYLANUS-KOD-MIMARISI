import sys, os
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from taylang import run_tay, compile_to_python, TAYSession, TAYError, backend_status

torch=pytest.importorskip("torch")

def run_compiled(src,cwd):
    env={}
    old=os.getcwd()
    try:
        os.chdir(cwd)
        exec(compile_to_python(src),env,env)
    finally:
        os.chdir(old)
    return env

def test_backend_status_reports_torch_and_gpu_truthfully():
    s=backend_status()
    assert s["NUMPY"]["available"] is True
    assert "TORCH" in s and "GPU" in s
    assert s["TORCH"]["available"] is True
    assert s["GPU"]["available"] == bool(torch.cuda.is_available())

def test_torch_backend_creates_real_tensors():
    out=run_tay("""
BACKEND TORCH
FIELD A = ONES(3,4,5)
VECTOR x = LINSPACE(0,1,5)
s = SUM(A)
""")
    assert torch.is_tensor(out["A"])
    assert torch.is_tensor(out["x"])
    assert out["A"].device.type=="cpu"
    assert out["A"].dtype==torch.float64
    assert out["BACKEND"]=="TORCH"
    assert out["DEVICE"]=="cpu"
    assert float(out["s"])==60.0

@pytest.mark.parametrize("boundary",["ZERO","WRAP","EDGE"])
def test_numpy_torch_field_primitive_parity(tmp_path,boundary):
    rng=np.random.default_rng(808)
    a=rng.normal(size=(6,5,4))
    np.save(tmp_path/"a.npy",a)
    body=f"""
BOUNDARY {boundary}
LOAD A FROM "a.npy"
B6 = ACC6(A)
B26 = ACC26(A)
M26 = AVG26(A)
X = DX(A,0.5)
Y = DY(A,0.75)
Z = DZ(A,1.25)
L = LAPLACE(A,0.5,0.75,1.25)
G = GRAD(A)
N = NORM(A)
S = SUM(A)
"""
    np_out=run_tay("BACKEND NUMPY\n"+body,base_dir=tmp_path)
    th_out=run_tay("BACKEND TORCH\n"+body,base_dir=tmp_path)
    for name in ["A","B6","B26","M26","X","Y","Z","L","G"]:
        got=th_out[name].detach().cpu().numpy()
        assert np.allclose(got,np.asarray(np_out[name]),rtol=0,atol=1e-12), name
    assert abs(float(th_out["N"])-float(np_out["N"]))<1e-12
    assert abs(float(th_out["S"])-float(np_out["S"]))<1e-12

def test_torch_scale_clip_sqrt_and_slicing():
    src="""
BACKEND TORCH
FIELD A = RANGE(0,64,1)
"""
    # RANGE is 1D, so construct a field in Python-generated TAY using FULL + assignments instead.
    out=run_tay("""
BACKEND TORCH
FIELD A = ONES(4,4,4)
A[1:3,1:3,1:3] = 9
B = SCALE(A,2)
C = CLIP(B,1,5)
D = SQRT(C)
slice = A[:,2,:]
""")
    assert torch.is_tensor(out["B"])
    assert tuple(out["B"].shape)==(2,2,2)
    assert torch.all(out["C"]<=5)
    assert torch.allclose(out["D"],torch.sqrt(out["C"]))
    assert tuple(out["slice"].shape)==(4,4)

def test_backend_switch_after_numeric_state_fails_closed():
    with pytest.raises(TAYError) as e:
        run_tay("""
FIELD A = ONES(3,3,3)
BACKEND TORCH
""")
    assert "before numerical" in str(e.value)

def test_torch_session_transaction_rolls_back():
    s=TAYSession()
    s.run_cell("""
BACKEND TORCH
FIELD A = ONES(3,3)
""")
    before=s.runtime.env["A"].clone()
    with pytest.raises(TAYError):
        s.run_cell("""
A[0,:] = 9
unknown_symbol
""")
    assert torch.equal(s.runtime.env["A"],before)
    assert s.runtime.backend=="TORCH"

def test_gpu_backend_fail_closed_or_real_cuda():
    if torch.cuda.is_available():
        out=run_tay("""
BACKEND GPU
FIELD A = ONES(2,2,2)
s = SUM(A)
""")
        assert out["A"].device.type=="cuda"
        assert float(out["s"])==8.0
    else:
        with pytest.raises(TAYError) as e:
            run_tay("BACKEND GPU")
        msg=str(e.value).lower()
        assert "cuda" in msg and ("false" in msg or "available" in msg)

def test_transpiler_torch_tensor_execution(tmp_path):
    src="""
BACKEND TORCH
BOUNDARY WRAP
FIELD A = ONES(5,5,5)
A[1:4,1:4,1:4] = 2
REPEAT 3
    NEXT A = A + 0.1 * LAPLACE(A,1,1,1)
    COMMIT
    TRACE mass = SUM(A)
END
SAVE A TO "a.npy"
"""
    ref=run_tay(src,base_dir=tmp_path)
    env=run_compiled(src,tmp_path)
    assert torch.is_tensor(env["A"])
    assert np.allclose(env["A"].detach().cpu().numpy(),ref["A"].detach().cpu().numpy(),rtol=0,atol=1e-12)
    assert np.allclose(env["TRACE_mass"],ref["TRACE_mass"],rtol=0,atol=1e-12)
    assert (tmp_path/"a.npy").exists()

def test_table_workflow_still_works_under_torch_selected(tmp_path):
    import pandas as pd
    pd.DataFrame({"age":[40,60,70],"sex":[0,1,1],"x":[1.0,np.nan,3.0]}).to_csv(tmp_path/"d.csv",index=False)
    out=run_tay("""
BACKEND TORCH
TABLE T = CSV("d.csv")
FILL T.x = MEDIAN
FILTER T WHERE age >= 50
GROUP G = T BY sex SUMMARIZE x:MEAN
""",base_dir=tmp_path)
    assert out["G"].shape==(1,2)
    assert out["BACKEND"]=="TORCH"


def test_external_numpy_env_is_coerced_to_torch():
    a=np.arange(27,dtype=float).reshape(3,3,3)
    out=run_tay("B = A * 2",{"A":a},backend="TORCH")
    assert torch.is_tensor(out["A"])
    assert torch.is_tensor(out["B"])
    assert np.array_equal(out["B"].detach().cpu().numpy(),a*2)

def test_same_source_external_backend_selection():
    src="""
BOUNDARY WRAP
FIELD A = ONES(5,5,5)
A[1:4,1:4,1:4] = 2
REPEAT 4
    NEXT A = A + 0.1 * LAPLACE(A,1,1,1)
    COMMIT
END
TOTAL = SUM(A)
"""
    np_out=run_tay(src,backend="NUMPY")
    th_out=run_tay(src,backend="TORCH")
    assert np.allclose(np_out["A"],th_out["A"].detach().cpu().numpy(),rtol=0,atol=1e-12)
    assert abs(float(np_out["TOTAL"])-float(th_out["TOTAL"]))<1e-12

def test_same_source_transpiler_external_backend_selection():
    src="""
BOUNDARY EDGE
FIELD A = ONES(4,4,4)
A[1:3,1:3,1:3] = 4
B = AVG26(A)
TOTAL = SUM(B)
"""
    env_np={}
    exec(compile_to_python(src,backend="NUMPY"),env_np,env_np)
    env_th={}
    exec(compile_to_python(src,backend="TORCH"),env_th,env_th)
    assert torch.is_tensor(env_th["A"])
    assert np.allclose(env_np["B"],env_th["B"].detach().cpu().numpy(),rtol=0,atol=1e-12)
    assert abs(float(env_np["TOTAL"])-float(env_th["TOTAL"]))<1e-12

def test_torch_session_summary_is_compact():
    s=TAYSession(backend="TORCH")
    s.run_cell("FIELD A = ONES(2,2,2)")
    info=s.vars()["A"]
    assert info["type"]=="torch.Tensor"
    assert info["device"]=="cpu"
    assert info["shape"]==[2,2,2]
