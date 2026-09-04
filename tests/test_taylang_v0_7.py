import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from taylang import run_tay, compile_to_python, TAYTable, TAYSession, run_notebook

def make_df():
    return pd.DataFrame({
        "age":[34,67,58,72,49,61,55,44],
        "sex":[0,1,0,1,1,0,1,0],
        "ddimer":[240,820,np.nan,1100,430,760,510,np.nan],
        "outcome":[0,1,0,1,0,1,0,0],
    })

def test_table_type_and_csv(tmp_path):
    p=tmp_path/"d.csv"
    make_df().to_csv(p,index=False)
    out=run_tay('TABLE T = CSV("d.csv")',base_dir=tmp_path)
    assert isinstance(out["T"],TAYTable)
    assert out["T"].shape==(8,4)

def test_keep_drop_and_column(tmp_path):
    p=tmp_path/"d.csv"; make_df().to_csv(p,index=False)
    src='''
TABLE T = CSV("d.csv")
DROP T = outcome
KEEP T = age, ddimer
VECTOR a = COL(T,"age")
'''
    out=run_tay(src,base_dir=tmp_path)
    assert out["T"].columns==["age","ddimer"]
    assert np.array_equal(out["a"],make_df()["age"].to_numpy())

def test_fill_median_matches_pandas(tmp_path):
    p=tmp_path/"d.csv"; df=make_df(); df.to_csv(p,index=False)
    out=run_tay('TABLE T = CSV("d.csv")\nFILL T.ddimer = MEDIAN',base_dir=tmp_path)
    ref=df.copy()
    ref["ddimer"]=ref["ddimer"].fillna(ref["ddimer"].median())
    pd.testing.assert_frame_equal(out["T"].df,ref,check_dtype=False)

def test_filter_boolean_words_and_external_scalar(tmp_path):
    p=tmp_path/"d.csv"; df=make_df(); df.to_csv(p,index=False)
    src='''
TABLE T = CSV("d.csv")
threshold = 50
FILTER T WHERE age >= threshold AND (sex == 1 OR ddimer > 700)
'''
    out=run_tay(src,base_dir=tmp_path)
    ref=df[(df.age>=50)&((df.sex==1)|(df.ddimer>700))].reset_index(drop=True)
    pd.testing.assert_frame_equal(out["T"].df,ref,check_dtype=False)

def test_filter_missing_present(tmp_path):
    p=tmp_path/"d.csv"; df=make_df(); df.to_csv(p,index=False)
    out=run_tay('TABLE T = CSV("d.csv")\nFILTER T WHERE MISSING(ddimer)',base_dir=tmp_path)
    ref=df[df.ddimer.isna()].reset_index(drop=True)
    pd.testing.assert_frame_equal(out["T"].df,ref,check_dtype=False)

def test_sort_matches_pandas(tmp_path):
    p=tmp_path/"d.csv"; df=make_df(); df.to_csv(p,index=False)
    out=run_tay('TABLE T = CSV("d.csv")\nSORT T BY age DESC',base_dir=tmp_path)
    ref=df.sort_values("age",ascending=False,kind="mergesort").reset_index(drop=True)
    pd.testing.assert_frame_equal(out["T"].df,ref,check_dtype=False)

def test_group_summary_matches_pandas(tmp_path):
    p=tmp_path/"d.csv"; df=make_df(); df.to_csv(p,index=False)
    src='''
TABLE T = CSV("d.csv")
FILL T.ddimer = MEDIAN
GROUP G = T BY sex SUMMARIZE age:MEAN, ddimer:MEDIAN, outcome:MEAN
'''
    out=run_tay(src,base_dir=tmp_path)
    d=df.copy()
    d["ddimer"]=d["ddimer"].fillna(d["ddimer"].median())
    ref=d.groupby(["sex"],dropna=False,sort=True).agg(
        age_mean=("age","mean"),
        ddimer_median=("ddimer","median"),
        outcome_mean=("outcome","mean"),
    ).reset_index()
    pd.testing.assert_frame_equal(out["G"].df,ref,check_dtype=False)

def test_save_table_csv(tmp_path):
    p=tmp_path/"d.csv"; df=make_df(); df.to_csv(p,index=False)
    src='''
TABLE T = CSV("d.csv")
KEEP T = age, sex
SAVE T TO "out.csv"
'''
    run_tay(src,base_dir=tmp_path)
    got=pd.read_csv(tmp_path/"out.csv")
    pd.testing.assert_frame_equal(got,df[["age","sex"]],check_dtype=False)

def test_rows_ncols_missingcount(tmp_path):
    p=tmp_path/"d.csv"; df=make_df(); df.to_csv(p,index=False)
    out=run_tay('TABLE T = CSV("d.csv")\nr=ROWS(T)\nc=NCOLS(T)\nm=MISSINGCOUNT(T,"ddimer")',base_dir=tmp_path)
    assert out["r"]==8 and out["c"]==4 and out["m"]==2

def test_table_transpiler_full_chain(tmp_path,monkeypatch):
    p=tmp_path/"d.csv"; df=make_df(); df.to_csv(p,index=False)
    src='''
TABLE T = CSV("d.csv")
KEEP T = age, sex, ddimer, outcome
FILL T.ddimer = MEDIAN
FILTER T WHERE age >= 50 AND ddimer > 500
SORT T BY ddimer DESC
GROUP G = T BY sex SUMMARIZE age:MEAN, ddimer:MEDIAN, outcome:MEAN
SAVE G TO "g.csv"
'''
    ref=run_tay(src,base_dir=tmp_path)
    env={}
    monkeypatch.chdir(tmp_path)
    exec(compile_to_python(src),env,env)
    pd.testing.assert_frame_equal(env["G"].df,ref["G"].df,check_dtype=False)
    assert (tmp_path/"g.csv").exists()

def test_table_session_transaction_rolls_back(tmp_path):
    p=tmp_path/"d.csv"; make_df().to_csv(p,index=False)
    s=TAYSession(base_dir=tmp_path)
    s.run_cell('TABLE T = CSV("d.csv")')
    before=s.runtime.env["T"].df.copy()
    try:
        s.run_cell('''
FILTER T WHERE age > 50
definitely_missing_name
''')
        assert False
    except Exception:
        pass
    pd.testing.assert_frame_equal(s.runtime.env["T"].df,before,check_dtype=False)

def test_table_notebook(tmp_path):
    (tmp_path/"data").mkdir()
    make_df().to_csv(tmp_path/"data"/"study.csv",index=False)
    nb=tmp_path/"x.taynb"
    nb.write_text('''%% load
TABLE T = CSV("data/study.csv")
%% clean
FILL T.ddimer = MEDIAN
FILTER T WHERE age >= 50
ROWS(T)
%% group
GROUP G = T BY sex SUMMARIZE age:MEAN, outcome:MEAN
ROWS(G)
''',encoding="utf-8")
    report=run_notebook(nb)
    assert report["status"]=="PASS"
    assert report["cells"][1]["result"]==5
    assert report["cells"][2]["result"]==2


def test_fill_constant_mode_and_drop(tmp_path):
    df=pd.DataFrame({
        "x":[1.0,np.nan,3.0,np.nan],
        "cat":["a",None,"a","b"],
    })
    p=tmp_path/"d.csv"; df.to_csv(p,index=False)

    out=run_tay('TABLE T = CSV("d.csv")\nFILL T.x = 9',base_dir=tmp_path)
    assert out["T"].df["x"].tolist()==[1.0,9.0,3.0,9.0]

    out=run_tay('TABLE T = CSV("d.csv")\nFILL T.cat = MODE',base_dir=tmp_path)
    assert out["T"].df["cat"].tolist()==["a","a","a","b"]

    out=run_tay('TABLE T = CSV("d.csv")\nFILL T.x = DROP',base_dir=tmp_path)
    assert len(out["T"].df)==2
    assert out["T"].missing_count("x")==0

def test_filter_string_column(tmp_path):
    df=pd.DataFrame({"site":["A","B","A","C"],"value":[1,2,3,4]})
    p=tmp_path/"d.csv"; df.to_csv(p,index=False)
    out=run_tay('TABLE T = CSV("d.csv")\nFILTER T WHERE site == "A"',base_dir=tmp_path)
    ref=df[df.site=="A"].reset_index(drop=True)
    pd.testing.assert_frame_equal(out["T"].df,ref,check_dtype=False)

def test_table_session_summary(tmp_path):
    p=tmp_path/"d.csv"; make_df().to_csv(p,index=False)
    s=TAYSession(base_dir=tmp_path)
    result=s.run_cell('TABLE T = CSV("d.csv")')
    summary=s.vars()["T"]
    assert summary["type"]=="table"
    assert summary["rows"]==8
    assert summary["columns"]==["age","sex","ddimer","outcome"]
    assert summary["missing"]==2
