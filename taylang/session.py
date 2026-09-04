from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import copy
import json
import numpy as np

from .core import TAYRuntime, TAYError
from .table import TAYTable

def summarize_value(value):
    try:
        import torch
        if torch.is_tensor(value):
            a=value.detach().cpu().numpy()
            if value.ndim==0:
                return value.item()
            out={
                "type":"torch.Tensor",
                "device":str(value.device),
                "shape":list(value.shape),
                "dtype":str(value.dtype),
            }
            if a.size and np.issubdtype(a.dtype,np.number):
                out.update({
                    "min":float(np.min(a)),
                    "max":float(np.max(a)),
                    "mean":float(np.mean(a)),
                })
            if a.size<=16:
                out["value"]=a.tolist()
            return out
    except Exception:
        pass

    if isinstance(value,TAYTable):
        return {
            "type":"table",
            "rows":len(value),
            "columns":value.columns,
            "shape":list(value.shape),
            "missing":value.missing_count(),
        }
    if isinstance(value,np.ndarray):
        out={
            "type":"ndarray",
            "shape":list(value.shape),
            "dtype":str(value.dtype),
        }
        if value.size:
            if np.issubdtype(value.dtype,np.number):
                out.update({
                    "min":float(np.min(value)),
                    "max":float(np.max(value)),
                    "mean":float(np.mean(value)),
                })
            if value.size<=16:
                out["value"]=value.tolist()
        return out
    if isinstance(value,np.generic):
        return value.item()
    if isinstance(value,(int,float,bool,str)) or value is None:
        return value
    return {"type":type(value).__name__,"repr":repr(value)}

@dataclass
class CellResult:
    cell: int
    result: object
    new_names: list[str]
    visible_names: list[str]

    def as_dict(self):
        return {
            "cell":self.cell,
            "result":summarize_value(self.result),
            "new_names":self.new_names,
            "visible_names":self.visible_names,
        }

class TAYSession:
    """Persistent TAY execution session for REPL/notebook-style workflows."""

    def __init__(self,env=None,boundary="ZERO",base_dir=None,transactional=True,backend="NUMPY"):
        self.runtime=TAYRuntime(env=env,boundary=boundary,base_dir=base_dir,backend=backend)
        self.history=[]
        self.cell_count=0
        self.transactional=bool(transactional)

    def run_cell(self,source:str) -> CellResult:
        if not isinstance(source,str):
            raise TypeError("source must be a string")

        before=set(self.runtime.env)

        if self.transactional:
            saved_env=self.snapshot()
            saved_functions=dict(self.runtime.user_functions)
            saved_traces={k:list(v) for k,v in self.runtime.traces.items()}
            saved_boundary=self.runtime.boundary
            saved_backend=self.runtime.backend
            saved_pending=copy.deepcopy(self.runtime.pending)
            saved_engine_name=self.runtime.engine_name
            saved_engine=self.runtime.engine
            saved_engine_config=copy.deepcopy(self.runtime.engine_config)

        self.runtime.env.pop("_",None)

        try:
            self.runtime.run(source)
        except Exception:
            if self.transactional:
                self.runtime.env=saved_env
                self.runtime.user_functions=saved_functions
                self.runtime.traces=saved_traces
                self.runtime.boundary=saved_boundary
                self.runtime.backend=saved_backend
                from .backends import create_backend
                self.runtime.ops=create_backend(saved_backend)
                self.runtime.pending=saved_pending
                self.runtime.engine_name=saved_engine_name
                self.runtime.engine=saved_engine
                self.runtime.engine_config=saved_engine_config
            raise

        self.cell_count+=1
        self.history.append(source)
        after=set(self.runtime.env)
        new_names=sorted(
            n for n in (after-before)
            if not n.startswith("__") and n!="_"
        )
        visible=sorted(
            n for n in after
            if not n.startswith("__") and n!="_"
        )
        return CellResult(
            cell=self.cell_count,
            result=self.runtime.env.get("_"),
            new_names=new_names,
            visible_names=visible,
        )

    def vars(self):
        return {
            k:summarize_value(v)
            for k,v in sorted(self.runtime.env.items())
            if not k.startswith("__") and k!="_"
        }

    def snapshot(self):
        snap={}
        for k,v in self.runtime.env.items():
            if self.runtime.ops.is_array(v):
                snap[k]=self.runtime.ops.clone(v)
            elif isinstance(v,TAYTable):
                snap[k]=v.copy()
            else:
                try:
                    snap[k]=copy.deepcopy(v)
                except Exception:
                    snap[k]=v
        return snap

    def reset(self):
        base=self.runtime.base_dir
        boundary=self.runtime.boundary
        backend=self.runtime.backend
        engine_name=self.runtime.engine_name
        engine_config=copy.deepcopy(self.runtime.engine_config)
        self.runtime=TAYRuntime(boundary=boundary,base_dir=base,backend=backend)
        if engine_name is not None:
            self.runtime._set_engine(engine_name)
            self.runtime.engine_config=engine_config
        self.history.clear()
        self.cell_count=0

def split_notebook(text:str):
    """Split a lightweight .taynb file on lines beginning with %%."""
    cells=[]
    current=[]
    title=None

    def flush():
        nonlocal current,title
        src="\n".join(current).strip()
        if src:
            cells.append({"title":title,"source":src})
        current=[]
        title=None

    for raw in text.splitlines():
        if raw.lstrip().startswith("%%"):
            flush()
            label=raw.lstrip()[2:].strip()
            title=label or None
        else:
            current.append(raw)
    flush()
    return cells

def run_notebook(path,report_path=None,backend="NUMPY"):
    path=Path(path).resolve()
    cells=split_notebook(path.read_text(encoding="utf-8"))
    session=TAYSession(base_dir=path.parent,backend=backend)
    report={
        "notebook":str(path),
        "backend":str(backend).upper(),
        "cells":[],
        "status":"PASS",
    }
    for i,cell in enumerate(cells,1):
        try:
            result=session.run_cell(cell["source"])
            report["cells"].append({
                "index":i,
                "title":cell["title"],
                "status":"PASS",
                **result.as_dict(),
            })
        except Exception as e:
            report["status"]="FAIL"
            report["cells"].append({
                "index":i,
                "title":cell["title"],
                "status":"FAIL",
                "error":f"{type(e).__name__}: {e}",
            })
            break
    report["variables"]=session.vars()
    if report_path is not None:
        rp=Path(report_path)
        rp.parent.mkdir(parents=True,exist_ok=True)
        rp.write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report
