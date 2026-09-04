from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .version import __version__
from .core import run_tay, TAYError
from .session import TAYSession, run_notebook
from .table import TAYTable
from .backends import backend_status
from .engines import engine_status

HELLO_TAY = """# TAY Language Developer Preview
BACKEND NUMPY
PARAM n=8, steps=5
FIELD A = ZEROS(n,n,n)
A[2:6,2:6,2:6] = 1

REPEAT steps
    NEXT A = A + 0.1 * LAPLACE(A,1,1,1)
    COMMIT
    TRACE mass = SUM(A)
END

SAVE A TO "output/final.npy"
PLOT SLICE A Z 4 TO "output/slice.png"
PLOT TRACE mass TO "output/mass.png"
"""

NOTEBOOK_TEMPLATE = """%% Setup
PARAM n=8
FIELD A = ZEROS(n,n,n)
A[2:6,2:6,2:6] = 1

%% Inspect
SUM(A)

%% Evolve
REPEAT 3
    NEXT A = A + 0.1 * LAPLACE(A,1,1,1)
    COMMIT
END
SUM(A)
"""

def _summarize(v):
    try:
        import torch
        if torch.is_tensor(v):
            a=v.detach().cpu().numpy()
            if v.ndim==0:
                return v.item()
            return {
                "type":"torch.Tensor",
                "device":str(v.device),
                "shape":list(v.shape),
                "dtype":str(v.dtype),
                "min":float(a.min()) if a.size else None,
                "max":float(a.max()) if a.size else None,
            }
    except Exception:
        pass

    if isinstance(v,TAYTable):
        return {
            "type":"table",
            "rows":len(v),
            "columns":v.columns,
            "missing":v.missing_count(),
        }
    if isinstance(v,np.ndarray):
        if v.ndim==0:
            return v.item()
        return {
            "type":"numpy.ndarray",
            "shape":list(v.shape),
            "dtype":str(v.dtype),
            "min":float(v.min()) if v.size else None,
            "max":float(v.max()) if v.size else None,
        }
    if isinstance(v,np.generic):
        return v.item()
    if isinstance(v,dict):
        return {str(k):_summarize(value) for k,value in v.items()}
    if isinstance(v,(list,tuple)):
        return [_summarize(value) for value in v]
    if isinstance(v,(int,float,bool,str)) or v is None:
        return v
    return repr(v)

def _parse_bindings(items, arrays=False):
    env={}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"Binding must use name=value: {item}")
        name,value=item.split("=",1)
        name=name.strip()
        if not name.isidentifier():
            raise SystemExit(f"Invalid binding name: {name}")
        if arrays:
            env[name]=np.load(Path(value).expanduser().resolve(),allow_pickle=False)
        else:
            try:
                env[name]=float(value)
            except ValueError:
                raise SystemExit(f"Scalar binding must be numeric: {item}")
    return env

def cmd_run(args):
    program=Path(args.program).expanduser().resolve()
    if not program.exists():
        raise SystemExit(f"Program not found: {program}")

    env={}
    env.update(_parse_bindings(args.scalar))
    env.update(_parse_bindings(args.array,arrays=True))

    src=program.read_text(encoding="utf-8")
    try:
        out=run_tay(src,env=env,base_dir=program.parent,backend=args.backend)
    except Exception as e:
        print(f"{type(e).__name__}: {e}",file=sys.stderr)
        return 1

    if args.quiet:
        return 0

    visible={
        k:_summarize(v)
        for k,v in out.items()
        if not k.startswith("__")
    }
    print(json.dumps(visible,indent=2,ensure_ascii=False))
    return 0

def _block_balance(source):
    balance=0
    for raw in source.splitlines():
        line=raw.split("#",1)[0].strip()
        if not line:
            continue
        up=line.upper()
        if up=="END":
            balance-=1
        elif up.startswith(("FUNC ","IF ","REPEAT ")):
            balance+=1
        elif up.startswith("SOLVE "):
            if not ("=" in line and " MAX " in up):
                balance+=1
    return balance

def cmd_repl(args):
    try:
        session=TAYSession(base_dir=Path.cwd(),backend=args.backend)
    except Exception as e:
        print(f"{type(e).__name__}: {e}",file=sys.stderr)
        return 1

    print(f"TAY Language {__version__} REPL [{session.runtime.backend}/{session.runtime.ops.device}]")
    print("Commands: :vars  :history  :reset  :load FILE  :quit")
    buf=[]

    while True:
        prompt="... " if buf else "tay> "
        try:
            line=input(prompt)
        except (EOFError,KeyboardInterrupt):
            print()
            return 0

        if not buf and line.strip().startswith(":"):
            cmd=line.strip()
            if cmd in (":quit",":q",":exit"):
                return 0
            if cmd==":vars":
                for k,v in session.vars().items():
                    print(f"{k} = {v}")
                continue
            if cmd==":history":
                for i,src in enumerate(session.history,1):
                    print(f"[{i}] {src}")
                continue
            if cmd==":reset":
                session.reset()
                print("session reset")
                continue
            if cmd.startswith(":load "):
                path=Path(cmd[6:].strip()).expanduser().resolve()
                try:
                    r=session.run_cell(path.read_text(encoding="utf-8"))
                    if r.result is not None:
                        print(_summarize(r.result))
                except Exception as e:
                    print(f"{type(e).__name__}: {e}")
                continue
            print("unknown command")
            continue

        buf.append(line)
        source="\n".join(buf)
        if _block_balance(source)>0:
            continue
        try:
            r=session.run_cell(source)
            if r.result is not None:
                print(_summarize(r.result))
        except Exception as e:
            print(f"{type(e).__name__}: {e}")
        buf=[]

def cmd_notebook(args):
    nb=Path(args.notebook).expanduser().resolve()
    report=Path(args.report).expanduser().resolve() if args.report else nb.with_suffix(".report.json")
    try:
        result=run_notebook(nb,report,backend=args.backend)
    except Exception as e:
        print(f"{type(e).__name__}: {e}",file=sys.stderr)
        return 1
    print(json.dumps({
        "status":result["status"],
        "cells":len(result["cells"]),
        "backend":result.get("backend",args.backend),
        "report":str(report),
    },indent=2))
    return 0 if result["status"]=="PASS" else 1

def cmd_doctor(args):
    status=backend_status()
    payload={
        "tay_version":__version__,
        "python":sys.version.split()[0],
        "executable":sys.executable,
        "platform":sys.platform,
        "backends":status,
        "engines":engine_status(),
        "cwd":str(Path.cwd()),
    }
    print(json.dumps(payload,indent=2))
    return 0

def cmd_init(args):
    target=Path(args.directory).expanduser().resolve()
    target.mkdir(parents=True,exist_ok=True)
    (target/"output").mkdir(exist_ok=True)

    program=target/"hello.tay"
    notebook=target/"explore.taynb"
    if program.exists() and not args.force:
        raise SystemExit(f"Already exists: {program}. Use --force to overwrite.")
    program.write_text(HELLO_TAY,encoding="utf-8")
    notebook.write_text(NOTEBOOK_TEMPLATE,encoding="utf-8")

    print(f"Created TAY project: {target}")
    print(f"  {program.name}")
    print(f"  {notebook.name}")
    print("Run:")
    print(f'  tay run "{program.name}"')
    return 0

def build_parser():
    ap=argparse.ArgumentParser(
        prog="tay",
        description="TAY Language experimental scientific programming environment"
    )
    ap.add_argument("--version",action="version",version=f"%(prog)s {__version__}")
    sub=ap.add_subparsers(dest="command",required=True)

    p=sub.add_parser("run",help="run a .tay program")
    p.add_argument("program")
    p.add_argument("--backend",default="NUMPY",choices=["NUMPY","TORCH","GPU"])
    p.add_argument("--scalar",action="append",default=[],help="numeric binding name=value")
    p.add_argument("--array",action="append",default=[],help="NumPy .npy binding name=file.npy")
    p.add_argument("--quiet",action="store_true")
    p.set_defaults(func=cmd_run)

    p=sub.add_parser("repl",help="start the interactive TAY REPL")
    p.add_argument("--backend",default="NUMPY",choices=["NUMPY","TORCH","GPU"])
    p.set_defaults(func=cmd_repl)

    p=sub.add_parser("notebook",help="run a lightweight .taynb notebook")
    p.add_argument("notebook")
    p.add_argument("--backend",default="NUMPY",choices=["NUMPY","TORCH","GPU"])
    p.add_argument("--report",default=None)
    p.set_defaults(func=cmd_notebook)

    p=sub.add_parser("doctor",help="show runtime and backend availability")
    p.set_defaults(func=cmd_doctor)

    p=sub.add_parser("init",help="create a starter TAY project")
    p.add_argument("directory",nargs="?",default="tay-project")
    p.add_argument("--force",action="store_true")
    p.set_defaults(func=cmd_init)

    return ap

def main(argv=None):
    ap=build_parser()
    args=ap.parse_args(argv)
    code=args.func(args)
    return int(code or 0)

if __name__=="__main__":
    raise SystemExit(main())
