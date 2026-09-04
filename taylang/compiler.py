import ast
import re
from .core import _clean,_parse,TAYError

PRIMITIVES={
    "ACC6","ACC26","AVG6","AVG26","LAPLACE6","DX","DY","DZ","LAPLACE","GRAD","SCALE",
    "SUM","MEAN","MIN","MAX","ABS","SQRT","NORM","CLIP","LEN","ZEROS","ONES","FULL","LINSPACE","RANGE","CSV","COL","ROWS","NCOLS","MISSINGCOUNT"
}

class _Prefix(ast.NodeTransformer):
    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Name) and node.func.id in PRIMITIVES:
            node.func=ast.Attribute(
                value=ast.Name(id="__ops",ctx=ast.Load()),
                attr=node.func.id,
                ctx=ast.Load()
            )
        return node

def _expr(src):
    tree=ast.parse(src,mode="eval")
    tree=_Prefix().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree.body)

def compile_to_python(source, backend="NUMPY"):
    nodes,_,_=_parse(_clean(source))
    out=[
        "import numpy as np",
        "import matplotlib",
        'matplotlib.use("Agg")',
        "import matplotlib.pyplot as plt",
        "from pathlib import Path",
        "from taylang.backend_runtime import BackendOps",
        "from taylang.engines import create_engine as __create_engine",
        "from taylang.table import TAYTable",
        '__ops = BackendOps("' + str(backend).upper() + '","ZERO")',
        'BACKEND = "' + str(backend).upper() + '" ',
        "DEVICE = __ops.device",
        "__next = {}",
        "__traces = {}",
        "__engine = None",
        "__engine_config = {}",
        "__SOLVED = False",
    ]

    engine_selected=False

    def emit(nodes,indent=0):
        nonlocal engine_selected
        pad="    "*indent
        for node in nodes:
            kind=node[0]

            if kind=="line":
                line=node[1]; up=line.upper()

                if up.startswith("PARAM "):
                    body=line[6:].strip()
                    for item in body.split(","):
                        if "=" not in item:
                            raise TAYError("PARAM syntax: PARAM a=1, b=2")
                        name,expr=item.split("=",1)
                        name=name.strip()
                        if not name.isidentifier():
                            raise TAYError("Invalid PARAM name")
                        out.append(pad+f"{name} = {_expr(expr.strip())}")
                    continue

                if up.startswith("BACKEND "):
                    backend=line.split(None,1)[1].strip().upper()
                    out.append(pad+f'__ops.set_backend("{backend}")')
                    out.append(pad+f'BACKEND = "{backend}"')
                    out.append(pad+'DEVICE = __ops.device')
                    continue

                if up.startswith("ENGINE "):
                    engine=line.split(None,1)[1].strip().upper()
                    out.append(pad+f'__engine = __create_engine("{engine}")')
                    out.append(pad+f'ENGINE = "{engine}"')
                    out.append(pad+'__engine_config = {}')
                    engine_selected=True
                    continue

                engine_numeric={
                    "RESOLUTION":"resolution",
                    "DT":"dt",
                    "TEND":"tend",
                    "VISCOSITY":"viscosity",
                    "SLICE":"slice_index",
                    "PLANNER_REPS":"planner_reps",
                } if engine_selected else {}
                matched_engine_directive=False
                for directive,key in engine_numeric.items():
                    m=re.fullmatch(
                        rf"{directive}\s*(?:=\s*)?(.+)",line,re.I
                    )
                    if m:
                        out.append(
                            pad+f'__engine_config["{key}"] = '
                            +_expr(m.group(1).strip())
                        )
                        matched_engine_directive=True
                        break
                if matched_engine_directive:
                    continue

                m=re.fullmatch(r"MODE\s*(?:=\s*)?([A-Za-z_]+)",line,re.I)
                if engine_selected and m:
                    out.append(
                        pad+f'__engine_config["mode"] = "{m.group(1).upper()}"'
                    )
                    continue

                m=re.fullmatch(r"OUTPUT\s*(?:=\s*)?(.+)",line,re.I)
                if engine_selected and m:
                    out.append(
                        pad+'__engine_config["output"] = '+_expr(m.group(1).strip())
                    )
                    continue

                m=re.fullmatch(r"REFERENCE\s*(?:=\s*)?(.+)",line,re.I)
                if engine_selected and m:
                    raw=m.group(1).strip()
                    value="None" if raw.upper()=="NONE" else _expr(raw)
                    out.append(pad+'__engine_config["reference"] = '+value)
                    continue

                m=re.fullmatch(r"RUN\s+(TAYLANUS|ENGINE)",line,re.I)
                if m:
                    if not engine_selected:
                        raise TAYError(
                            "RUN TAYLANUS requires ENGINE TAYLANUS first."
                        )
                    out.append(
                        pad+'CFD_RESULT = __engine.run('
                        '__engine_config, base_dir=Path.cwd(), backend=BACKEND)'
                    )
                    out.append(
                        pad+'globals().update(__engine.environment_values(CFD_RESULT))'
                    )
                    continue

                m=re.fullmatch(r"REGION\s+([A-Za-z_]\w*)\[(.+)\]\s*=\s*(.+)",line,re.I)
                if m:
                    target,spec,rhs=m.group(1),m.group(2),m.group(3).strip()
                    parts=[x.strip() for x in spec.split(",")]
                    slices=[]
                    for part in parts:
                        if ":" not in part:
                            raise TAYError("REGION axes must use start:end.")
                        lo,hi=part.split(":",1)
                        lo_s="" if not lo.strip() else f"int({_expr(lo.strip())})"
                        hi_s="" if not hi.strip() else f"int({_expr(hi.strip())})"
                        slices.append(f"{lo_s}:{hi_s}")
                    out.append(pad+f"{target}[{', '.join(slices)}] = {_expr(rhs)}")
                    continue

                m=re.fullmatch(r"TRACE\s+([A-Za-z_]\w*)\s*=\s*(.+)",line,re.I)
                if m:
                    name,expr=m.group(1),m.group(2).strip()
                    out.append(pad+f'{name} = float(__ops.scalar({_expr(expr)}))')
                    out.append(pad+f'__traces.setdefault("{name}", []).append({name})')
                    out.append(pad+f'TRACE_{name} = np.asarray(__traces["{name}"], dtype=float)')
                    continue

                m=re.fullmatch(r'PLOT\s+SLICE\s+([A-Za-z_]\w*)\s+([XYZ])\s+(.+?)\s+TO\s+"([^"]+)"',line,re.I)
                if m:
                    target,axis,index,path=m.group(1),m.group(2).upper(),m.group(3).strip(),m.group(4)
                    ax={"X":0,"Y":1,"Z":2}[axis]
                    out.append(pad+f'__img = np.take(__ops.to_numpy({target}), int({_expr(index)}), axis={ax})')
                    out.append(pad+'__fig = plt.figure()')
                    out.append(pad+'__ax = __fig.add_subplot(111)')
                    out.append(pad+'__im = __ax.imshow(__img, origin="lower", aspect="auto")')
                    out.append(pad+'__fig.colorbar(__im, ax=__ax)')
                    out.append(pad+f'__ax.set_title("{target} {axis}=" + str(int({_expr(index)})))')
                    out.append(pad+'__fig.tight_layout()')
                    out.append(pad+f'__fig.savefig(r"{path}", dpi=140)')
                    out.append(pad+'plt.close(__fig)')
                    out.append(pad+f'LAST_PLOT = r"{path}"')
                    continue

                m=re.fullmatch(r'PLOT\s+TRACE\s+([A-Za-z_]\w*)\s+TO\s+"([^"]+)"',line,re.I)
                if m:
                    name,path=m.group(1),m.group(2)
                    out.append(pad+f'__y = np.asarray(__traces["{name}"], dtype=float)')
                    out.append(pad+'__x = np.arange(1, len(__y)+1)')
                    out.append(pad+'__fig = plt.figure()')
                    out.append(pad+'__ax = __fig.add_subplot(111)')
                    out.append(pad+'__ax.plot(__x, __y)')
                    out.append(pad+f'__ax.set_ylabel("{name}")')
                    out.append(pad+'__ax.set_xlabel("sample")')
                    out.append(pad+f'__ax.set_title("{name}")')
                    out.append(pad+'__fig.tight_layout()')
                    out.append(pad+f'__fig.savefig(r"{path}", dpi=140)')
                    out.append(pad+'plt.close(__fig)')
                    out.append(pad+f'LAST_PLOT = r"{path}"')
                    continue

                if up.startswith("BOUNDARY "):
                    mode=line.split(None,1)[1].upper()
                    out.append(pad+f'__ops.set_boundary("{mode}")')

                elif up=="COMMIT":
                    out.append(pad+"globals().update(__next)")
                    out.append(pad+"__next.clear()")

                elif up.startswith("NEXT "):
                    target,expr=line[5:].split("=",1)
                    out.append(pad+f'__next["{target.strip()}"] = {_expr(expr.strip())}')

                elif any(up.startswith(k+" ") for k in ("SCALAR","VECTOR","FIELD","TABLE")):
                    kind2=up.split(None,1)[0]
                    rest=line[len(kind2):].strip()
                    target,expr=rest.split("=",1)
                    target=target.strip()
                    out.append(pad+f"{target} = {_expr(expr.strip())}")
                    if kind2=="SCALAR":
                        out.append(pad+f'assert __ops.ndim({target}) == 0')
                    elif kind2=="VECTOR":
                        out.append(pad+f'assert __ops.ndim({target}) == 1')
                    elif kind2=="FIELD":
                        out.append(pad+f'assert __ops.ndim({target}) in (2,3)')
                    else:
                        out.append(pad+f'assert isinstance({target}, TAYTable)')

                else:
                    m=re.fullmatch(r"KEEP\s+([A-Za-z_]\w*)\s*=\s*(.+)",line,re.I)
                    if m:
                        name=m.group(1)
                        cols=[x.strip() for x in m.group(2).split(",") if x.strip()]
                        out.append(pad+f"{name} = {name}.keep({cols!r})")
                        continue

                    m=re.fullmatch(r"DROP\s+([A-Za-z_]\w*)\s*=\s*(.+)",line,re.I)
                    if m:
                        name=m.group(1)
                        cols=[x.strip() for x in m.group(2).split(",") if x.strip()]
                        out.append(pad+f"{name} = {name}.drop({cols!r})")
                        continue

                    m=re.fullmatch(r"FILTER\s+([A-Za-z_]\w*)\s+WHERE\s+(.+)",line,re.I)
                    if m:
                        name,expr=m.group(1),m.group(2).strip()
                        out.append(pad+f"{name} = {name}.filter({expr!r}, globals())")
                        continue

                    m=re.fullmatch(r"FILL\s+([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s*=\s*(.+)",line,re.I)
                    if m:
                        name,col,raw=m.group(1),m.group(2),m.group(3).strip()
                        upper_raw=raw.upper()
                        if upper_raw in {"MEAN","MEDIAN","MODE","ZERO","DROP"}:
                            arg=repr(upper_raw)
                        else:
                            arg=_expr(raw)
                        out.append(pad+f"{name} = {name}.fill({col!r}, {arg})")
                        continue

                    m=re.fullmatch(r"SORT\s+([A-Za-z_]\w*)\s+BY\s+([A-Za-z_]\w*)(?:\s+(ASC|DESC))?",line,re.I)
                    if m:
                        name,col,order=m.group(1),m.group(2),(m.group(3) or "ASC").upper()
                        out.append(pad+f"{name} = {name}.sort({col!r}, descending={order=='DESC'})")
                        continue

                    m=re.fullmatch(r"GROUP\s+([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\s+BY\s+(.+?)\s+SUMMARIZE\s+(.+)",line,re.I)
                    if m:
                        outname,src,byraw,sraw=m.group(1),m.group(2),m.group(3),m.group(4)
                        by=[x.strip() for x in byraw.split(",") if x.strip()]
                        specs=[]
                        for item in sraw.split(","):
                            col,agg=item.split(":",1)
                            specs.append((col.strip(),agg.strip()))
                        out.append(pad+f"{outname} = {src}.group({by!r}, {specs!r})")
                        continue

                    m=re.fullmatch(r'SAVE\s+([A-Za-z_]\w*)\s+TO\s+"([^"]+)"',line,re.I)
                    if m:
                        name,path=m.group(1),m.group(2)
                        if path.lower().endswith(".npy"):
                            out.append(pad+f'__ops.save({name}, r"{path}")')
                        elif path.lower().endswith(".csv"):
                            out.append(pad+f'__ops.save({name}, r"{path}")')
                        else:
                            raise TAYError("SAVE supports .npy/.csv")
                        continue

                    m=re.fullmatch(r'LOAD\s+([A-Za-z_]\w*)\s+FROM\s+"([^"]+)"',line,re.I)
                    if m:
                        name,path=m.group(1),m.group(2)
                        if path.lower().endswith(".npy"):
                            out.append(pad+f'{name} = __ops.load(r"{path}")')
                        elif path.lower().endswith(".csv"):
                            out.append(pad+f'{name} = __ops.load(r"{path}")')
                        else:
                            raise TAYError("LOAD supports .npy/.csv")
                        continue

                    m=re.fullmatch(r'USE\s+"([^"]+)"',line,re.I)
                    if m:
                        out.append(pad+f'# USE "{m.group(1)}" is interpreter-resolved')
                        out.append(pad+f'raise RuntimeError("USE is not transpiled")')
                        continue

                    m_assign=re.match(r"^(.+?)(?<![<>=!])=(?!=)(.+)$",line)
                    if m_assign:
                        target=m_assign.group(1).strip()
                        expr=m_assign.group(2).strip()
                        # Validate target using Python AST. Allow a name or direct subscript/slice.
                        target_ast=ast.parse(target,mode="eval").body
                        if not (
                            isinstance(target_ast,ast.Name)
                            or (isinstance(target_ast,ast.Subscript) and isinstance(target_ast.value,ast.Name))
                        ):
                            raise TAYError("Assignment target must be a name or direct slice/subscript.")
                        out.append(pad+f"{target} = {_expr(expr)}")
                    else:
                        out.append(pad+"_ = "+_expr(line))

            elif kind=="repeat":
                out.append(pad+f"for ITER in range(1, int({_expr(node[1])}) + 1):")
                emit(node[2],indent+1)

            elif kind=="if":
                out.append(pad+f"if {_expr(node[1])}:")
                emit(node[2],indent+1)
                if node[3]:
                    out.append(pad+"else:")
                    emit(node[3],indent+1)

            elif kind=="func":
                name,args,body=node[1],node[2],node[3]
                out.append(pad+f"def {name}({', '.join(args)}):")
                if not body:
                    out.append(pad+"    pass")
                else:
                    emit(body,indent+1)

            elif kind=="return":
                out.append(pad+("return "+_expr(node[1]) if node[1] else "return"))

            elif kind=="solve_block":
                max_expr,cond,body=node[1],node[2],node[3]
                out.append(pad+"SOLVED = False")
                out.append(pad+f"for ITER in range(1, int({_expr(max_expr)}) + 1):")
                emit(body,indent+1)
                out.append(pad+"    "+f"if {_expr(cond)}:")
                out.append(pad+"        SOLVED = True")
                out.append(pad+"        break")

            elif kind=="solve_state":
                target,expr,cond,max_expr=node[1],node[2],node[3],node[4]
                out.append(pad+"SOLVED = False")
                out.append(pad+f"for ITER in range(1, int({_expr(max_expr)}) + 1):")
                out.append(pad+f"    __old_{target} = np.array({target}, copy=True) if isinstance({target}, np.ndarray) else {target}")
                out.append(pad+f"    {target} = {_expr(expr)}")
                out.append(pad+f"    CHANGE = float(np.sqrt(np.sum((np.asarray({target},dtype=float)-np.asarray(__old_{target},dtype=float))**2)))")
                out.append(pad+f"    if {_expr(cond)}:")
                out.append(pad+"        SOLVED = True")
                out.append(pad+"        break")


            elif kind=="linear":
                out.append(pad+f"{node[1]} = np.linalg.solve(np.asarray({_expr(node[2])},dtype=float), np.asarray({_expr(node[3])},dtype=float))")

            elif kind=="fit":
                out.append(pad+f"{node[1]} = np.linalg.lstsq(np.asarray({_expr(node[2])},dtype=float), np.asarray({_expr(node[3])},dtype=float), rcond=None)[0]")

            elif kind=="smooth":
                target,src,wexpr=node[1],node[2],node[3]
                out.append(pad+f"__w = int({_expr(wexpr)})")
                out.append(pad+f"__r = __w//2")
                out.append(pad+f"{target} = np.convolve(np.pad(np.asarray({_expr(src)},dtype=float),(__r,__r),mode='edge'), np.ones(__w)/__w, mode='valid')")

            elif kind=="integrate":
                target,yexpr,xexpr=node[1],node[2],node[3]
                out.append(pad+f"{target} = float(np.trapezoid(np.asarray({_expr(yexpr)},dtype=float), np.asarray({_expr(xexpr)},dtype=float)) if hasattr(np,'trapezoid') else np.trapz(np.asarray({_expr(yexpr)},dtype=float), np.asarray({_expr(xexpr)},dtype=float)))")

            elif kind=="stats":
                target,src=node[1],node[2]
                out.append(pad+f"__s = np.asarray({_expr(src)},dtype=float).ravel()")
                out.append(pad+f"{target} = np.array([__s.size, float(np.mean(__s)), float(np.std(__s,ddof=1)) if __s.size>1 else 0.0, float(np.min(__s)), float(np.max(__s))], dtype=float)")

            elif kind=="ode":
                target,fn_name,t0,t1,h,method=node[1:]
                out.append(pad+f"__t = float({_expr(t0)}); __t1 = float({_expr(t1)}); __h = float({_expr(h)})")
                out.append(pad+f"__y = np.array({target},copy=True) if isinstance({target},np.ndarray) else float({target})")
                out.append(pad+"STEPS = 0")
                out.append(pad+"while __t < __t1 - 1e-15:")
                out.append(pad+"    __hs = min(__h, __t1-__t)")
                if method=="EULER":
                    out.append(pad+f"    __y = __y + __hs*{fn_name}(__y,__t)")
                else:
                    out.append(pad+f"    __k1 = {fn_name}(__y,__t)")
                    out.append(pad+f"    __k2 = {fn_name}(__y+0.5*__hs*__k1,__t+0.5*__hs)")
                    out.append(pad+f"    __k3 = {fn_name}(__y+0.5*__hs*__k2,__t+0.5*__hs)")
                    out.append(pad+f"    __k4 = {fn_name}(__y+__hs*__k3,__t+__hs)")
                    out.append(pad+"    __y = __y + (__hs/6.0)*(__k1+2*__k2+2*__k3+__k4)")
                out.append(pad+"    __t += __hs; STEPS += 1")
                out.append(pad+f"{target} = __y")
                out.append(pad+"TIME = __t")

            elif kind=="optimize":
                target,fn_name,lr,cond,maxexpr=node[1:]
                out.append(pad+f"__lr = float({_expr(lr)})")
                out.append(pad+"SOLVED = False")
                out.append(pad+f"for ITER in range(1, int({_expr(maxexpr)}) + 1):")
                out.append(pad+f"    __old = np.array({target},copy=True) if isinstance({target},np.ndarray) else float({target})")
                out.append(pad+f"    __xa = np.asarray({target},dtype=float)")
                out.append(pad+"    if __xa.ndim == 0:")
                out.append(pad+f"        __xv=float(__xa); __g=({fn_name}(__xv+1e-6)-{fn_name}(__xv-1e-6))/(2e-6)")
                out.append(pad+f"        {target}=__xv-__lr*__g")
                out.append(pad+"    else:")
                out.append(pad+"        __g=np.zeros_like(__xa,dtype=float)")
                out.append(pad+"        __it=np.nditer(__xa,flags=['multi_index'])")
                out.append(pad+"        while not __it.finished:")
                out.append(pad+"            __idx=__it.multi_index; __xp=__xa.copy(); __xm=__xa.copy()")
                out.append(pad+"            __xp[__idx]+=1e-6; __xm[__idx]-=1e-6")
                out.append(pad+f"            __g[__idx]=({fn_name}(__xp)-{fn_name}(__xm))/(2e-6); __it.iternext()")
                out.append(pad+f"        {target}=__xa-__lr*__g")
                out.append(pad+f"    CHANGE=float(np.sqrt(np.sum((np.asarray({target},dtype=float)-np.asarray(__old,dtype=float))**2)))")
                out.append(pad+f"    OBJECTIVE=float({fn_name}({target}))")
                out.append(pad+f"    if {_expr(cond)}:")
                out.append(pad+"        SOLVED=True")
                out.append(pad+"        break")

            else:
                raise TAYError(f"Unsupported node: {kind}")

    emit(nodes)
    out += [
        "if __next:",
        "    globals().update(__next)",
        "    __next.clear()",
    ]
    return "\n".join(out)+"\n"
