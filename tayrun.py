import argparse,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))

from taylang import run_tay, TAYTable

def summarize(v):
    try:
        import torch
        if torch.is_tensor(v):
            a=v.detach().cpu().numpy()
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
        return {
            "type":"numpy.ndarray",
            "shape":list(v.shape),
            "dtype":str(v.dtype),
            "min":float(np.min(v)) if v.size else None,
            "max":float(np.max(v)) if v.size else None,
        }
    if isinstance(v,np.generic):
        return v.item()
    if isinstance(v,(int,float,bool,str)) or v is None:
        return v
    return str(v)

def main():
    ap=argparse.ArgumentParser(description="TAY Language v0.8 runner")
    ap.add_argument("program",help=".tay program")
    ap.add_argument("--backend",default="NUMPY",choices=["NUMPY","TORCH","GPU"])
    ap.add_argument("--scalar",action="append",default=[],
                    help="scalar binding: name=value")
    ap.add_argument("--array",action="append",default=[],
                    help="array binding: name=file.npy")
    args=ap.parse_args()

    program=Path(args.program).resolve()
    env={}

    for item in args.scalar:
        name,value=item.split("=",1)
        env[name]=float(value)

    for item in args.array:
        name,path=item.split("=",1)
        env[name]=np.load(Path(path).resolve(),allow_pickle=False)

    src=program.read_text(encoding="utf-8")
    out=run_tay(src,env,base_dir=program.parent,backend=args.backend)
    visible={k:summarize(v) for k,v in out.items() if not k.startswith("__")}
    print(json.dumps(visible,indent=2))

if __name__=="__main__":
    main()
