from pathlib import Path
import argparse, json, sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from taylang import run_notebook

def main():
    ap=argparse.ArgumentParser(description="Run a lightweight TAY .taynb notebook")
    ap.add_argument("notebook")
    ap.add_argument("--report",default=None)
    ap.add_argument("--backend",default="NUMPY",choices=["NUMPY","TORCH","GPU"])
    args=ap.parse_args()

    nb=Path(args.notebook).resolve()
    report_path=Path(args.report).resolve() if args.report else nb.with_suffix(".report.json")
    report=run_notebook(nb,report_path,backend=args.backend)
    print(json.dumps({
        "status":report["status"],
        "cells":len(report["cells"]),
        "report":str(report_path)
    },indent=2))
    raise SystemExit(0 if report["status"]=="PASS" else 1)

if __name__=="__main__":
    main()
