import sys, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from taylang import run_tay, compile_to_python, TAYSession, run_notebook, split_notebook

def exec_compiled(src,env=None):
    e=dict(env or {})
    exec(compile_to_python(src),e,e)
    return e

def test_slice_read_basic():
    a=np.arange(4*5*6).reshape(4,5,6)
    out=run_tay("B = A[:,1:4,2]",{"A":a})
    assert np.array_equal(out["B"],a[:,1:4,2])

def test_slice_read_step_negative_and_ellipsis():
    a=np.arange(5*6*7).reshape(5,6,7)
    src="""
    B = A[::2,1:5:2,-1]
    C = A[...,2]
    D = A[-1,:,:]
    """
    out=run_tay(src,{"A":a})
    assert np.array_equal(out["B"],a[::2,1:5:2,-1])
    assert np.array_equal(out["C"],a[...,2])
    assert np.array_equal(out["D"],a[-1,:,:])

def test_slice_write_basic():
    a=np.zeros((6,6,6))
    src="""
    A[1:5:2,2:6,3] = 7
    """
    out=run_tay(src,{"A":a.copy()})
    exp=a.copy()
    exp[1:5:2,2:6,3]=7
    assert np.array_equal(out["A"],exp)

def test_slice_write_transpiler_matches():
    a=np.arange(4*5*6,dtype=float).reshape(4,5,6)
    src="""
    B = A[:,1:4,2]
    A[1:3,:,::2] = -5
    C = A[...,1]
    """
    ref=run_tay(src,{"A":a.copy()})
    env=exec_compiled(src,{"A":a.copy()})
    assert np.array_equal(env["A"],ref["A"])
    assert np.array_equal(env["B"],ref["B"])
    assert np.array_equal(env["C"],ref["C"])

def test_integer_list_and_boolean_mask_indexing():
    a=np.arange(10)
    mask=np.array([True,False,True,False,True,False,True,False,True,False])
    out=run_tay("""
    B = A[[1,3,7]]
    C = A[mask]
    """,{"A":a,"mask":mask})
    assert np.array_equal(out["B"],a[[1,3,7]])
    assert np.array_equal(out["C"],a[mask])

def test_bare_expression_result():
    s=TAYSession()
    r=s.run_cell("a=5\na*3")
    assert r.result==15
    assert s.runtime.env["_"]==15

def test_bare_comparison_is_not_assignment():
    s=TAYSession()
    s.run_cell("a=5")
    r=s.run_cell("a == 5")
    assert r.result is True

def test_session_persists_values_and_functions():
    s=TAYSession()
    s.run_cell("a=4")
    s.run_cell("""
    FUNC SQUARE(x)
        RETURN x*x
    END
    """)
    r=s.run_cell("SQUARE(a)")
    assert r.result==16

def test_session_cell_result_does_not_leak():
    s=TAYSession()
    assert s.run_cell("2+3").result==5
    r=s.run_cell("a=7")
    assert r.result is None

def test_session_snapshot_is_copy():
    s=TAYSession()
    s.run_cell("A = ONES(3,3)")
    snap=s.snapshot()
    s.run_cell("A[0,:]=9")
    assert np.all(snap["A"][0,:]==1)
    assert np.all(s.runtime.env["A"][0,:]==9)

def test_session_reset():
    s=TAYSession()
    s.run_cell("a=3")
    s.reset()
    assert "a" not in s.runtime.env
    assert s.cell_count==0
    assert s.history==[]

def test_notebook_split():
    text="""%% setup
a=2
%% compute
b=a*4
b
"""
    cells=split_notebook(text)
    assert len(cells)==2
    assert cells[0]["title"]=="setup"
    assert cells[1]["title"]=="compute"

def test_notebook_persistent_execution(tmp_path):
    nb=tmp_path/"demo.taynb"
    nb.write_text("""%% setup
PARAM n=4
FIELD A = ZEROS(n,n,n)
A[1:3,1:3,1:3] = 2
%% inspect
mid = A[:,:,2]
SUM(mid)
%% function
FUNC SCALE2(x)
    RETURN x*2
END
%% use
B = SCALE2(A)
SUM(B)
""",encoding="utf-8")
    rp=tmp_path/"report.json"
    report=run_notebook(nb,rp)
    assert report["status"]=="PASS"
    assert len(report["cells"])==4
    assert report["cells"][1]["result"]==8.0
    assert report["cells"][3]["result"]==32.0
    assert rp.exists()

def test_compiled_bare_expression_sets_underscore():
    env=exec_compiled("a=6\na+4")
    assert env["_"]==10


def test_session_failed_cell_rolls_back_memory():
    from taylang import TAYError
    s=TAYSession()
    s.run_cell("A = ONES(3,3)")
    before=s.snapshot()["A"].copy()
    try:
        s.run_cell("""
        A[0,:] = 9
        definitely_unknown_name + 1
        """)
        assert False
    except TAYError:
        pass
    assert np.array_equal(s.runtime.env["A"],before)
    assert s.cell_count==1

def test_session_failed_cell_rolls_back_function_definition():
    from taylang import TAYError
    s=TAYSession()
    s.run_cell("""
    FUNC F(x)
        RETURN x+1
    END
    """)
    assert s.run_cell("F(3)").result==4
    try:
        s.run_cell("""
        FUNC F(x)
            RETURN x+100
        END
        missing_symbol
        """)
        assert False
    except TAYError:
        pass
    assert s.run_cell("F(3)").result==4
