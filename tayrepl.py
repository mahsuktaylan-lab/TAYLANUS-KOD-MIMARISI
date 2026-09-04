from pathlib import Path
import sys
import argparse

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))

from taylang import TAYSession, TAYError

def block_balance(source):
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
            # One-line state solver contains '=' and ' MAX ' and does not open a block.
            if not ("=" in line and " MAX " in up):
                balance+=1
    return balance

def print_result(result):
    data=result.as_dict()
    if data["result"] is not None:
        print(data["result"])

def main():
    ap=argparse.ArgumentParser(description="TAY Language v0.8 REPL")
    ap.add_argument("--backend",default="NUMPY",choices=["NUMPY","TORCH","GPU"])
    args=ap.parse_args()
    session=TAYSession(base_dir=Path.cwd(),backend=args.backend)
    print(f"TAY Language v0.8 REPL [{session.runtime.backend}/{session.runtime.ops.device}]")
    print("Commands: :vars  :history  :reset  :load FILE  :quit")

    buffer=[]
    while True:
        prompt="... " if buffer else "tay> "
        try:
            line=input(prompt)
        except (EOFError,KeyboardInterrupt):
            print()
            break

        if not buffer and line.strip().startswith(":"):
            cmd=line.strip()
            if cmd in (":quit",":q",":exit"):
                break
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
                    result=session.run_cell(path.read_text(encoding="utf-8"))
                    print_result(result)
                except Exception as e:
                    print(f"{type(e).__name__}: {e}")
                continue
            print("unknown command")
            continue

        buffer.append(line)
        source="\n".join(buffer)
        if block_balance(source)>0:
            continue

        try:
            result=session.run_cell(source)
            print_result(result)
        except Exception as e:
            print(f"{type(e).__name__}: {e}")
        buffer=[]

if __name__=="__main__":
    main()
