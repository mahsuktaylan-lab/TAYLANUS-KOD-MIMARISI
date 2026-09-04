from __future__ import annotations
import ast
import copy
import re
from pathlib import Path
import numpy as np
from .table import TAYTable, TableError
from .backends import create_backend, BackendError

class TAYError(Exception):
    pass

class _ReturnSignal(Exception):
    def __init__(self,value):
        self.value=value

DIR6=[(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]
DIR26=[(dx,dy,dz) for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1)
       if (dx,dy,dz)!=(0,0,0)]

def _pad(a, mode):
    mapping={"ZERO":"constant","WRAP":"wrap","EDGE":"edge"}
    if mode not in mapping:
        raise TAYError(f"Unknown boundary mode: {mode}")
    return np.pad(a,1,mode=mapping[mode])

def _acc(a, dirs, mode):
    a=np.asarray(a)
    if a.ndim!=3:
        raise TAYError("Neighborhood primitives currently require a 3D FIELD.")
    p=_pad(a,mode)
    nx,ny,nz=a.shape
    out=np.zeros_like(a,dtype=np.result_type(a.dtype,np.float64))
    for dx,dy,dz in dirs:
        out += p[1+dx:1+dx+nx,1+dy:1+dy+ny,1+dz:1+dz+nz]
    return out

def _grad(a,mode):
    a=np.asarray(a,dtype=float)
    if a.ndim!=3:
        raise TAYError("GRAD currently requires a 3D FIELD.")
    p=_pad(a,mode); nx,ny,nz=a.shape
    gx=.5*(p[2:2+nx,1:1+ny,1:1+nz]-p[0:nx,1:1+ny,1:1+nz])
    gy=.5*(p[1:1+nx,2:2+ny,1:1+nz]-p[1:1+nx,0:ny,1:1+nz])
    gz=.5*(p[1:1+nx,1:1+ny,2:2+nz]-p[1:1+nx,1:1+ny,0:nz])
    return np.stack((gx,gy,gz),axis=-1)

def _scale(a,k):
    a=np.asarray(a); k=int(k)
    if k<=0 or any(n%k for n in a.shape):
        raise TAYError("SCALE requires a positive factor dividing every dimension.")
    x,y,z=a.shape
    return a.reshape(x//k,k,y//k,k,z//k,k).mean(axis=(1,3,5))


def _linear_solve(a,b):
    return np.linalg.solve(np.asarray(a,dtype=float),np.asarray(b,dtype=float))

def _least_squares(x,y):
    beta,_,_,_=np.linalg.lstsq(np.asarray(x,dtype=float),np.asarray(y,dtype=float),rcond=None)
    return beta

def _smooth1d(x,window):
    x=np.asarray(x,dtype=float)
    w=int(window)
    if x.ndim!=1:
        raise TAYError("SMOOTH currently requires a 1D VECTOR.")
    if w<=0 or w%2==0:
        raise TAYError("SMOOTH WINDOW must be a positive odd integer.")
    r=w//2
    p=np.pad(x,(r,r),mode="edge")
    k=np.ones(w,dtype=float)/w
    return np.convolve(p,k,mode="valid")

def _integrate_xy(y,x):
    y=np.asarray(y,dtype=float)
    x=np.asarray(x,dtype=float)
    if y.ndim!=1 or x.ndim!=1 or len(y)!=len(x):
        raise TAYError("INTEGRATE requires equal-length 1D vectors.")
    if hasattr(np,"trapezoid"):
        return float(np.trapezoid(y,x))
    return float(np.trapz(y,x))

def _stats5(x):
    a=np.asarray(x,dtype=float).ravel()
    if a.size==0:
        raise TAYError("STATS requires at least one value.")
    std=float(np.std(a,ddof=1)) if a.size>1 else 0.0
    return np.array([a.size,float(np.mean(a)),std,float(np.min(a)),float(np.max(a))],dtype=float)

def _fd_gradient(fn,x,eps=1e-6):
    a=np.asarray(x,dtype=float)
    if a.ndim==0:
        xv=float(a)
        return float((fn(xv+eps)-fn(xv-eps))/(2*eps))
    g=np.zeros_like(a,dtype=float)
    it=np.nditer(a,flags=["multi_index"])
    while not it.finished:
        idx=it.multi_index
        xp=a.copy(); xm=a.copy()
        xp[idx]+=eps; xm[idx]-=eps
        g[idx]=(fn(xp)-fn(xm))/(2*eps)
        it.iternext()
    return g


def _daxis(a, axis, h, mode):
    a=np.asarray(a,dtype=float)
    if a.ndim!=3:
        raise TAYError("DX/DY/DZ currently require a 3D FIELD.")
    h=float(h)
    if h<=0:
        raise TAYError("Grid spacing must be positive.")
    p=_pad(a,mode)
    nx,ny,nz=a.shape
    if axis==0:
        return (p[2:2+nx,1:1+ny,1:1+nz]-p[0:nx,1:1+ny,1:1+nz])/(2*h)
    if axis==1:
        return (p[1:1+nx,2:2+ny,1:1+nz]-p[1:1+nx,0:ny,1:1+nz])/(2*h)
    return (p[1:1+nx,1:1+ny,2:2+nz]-p[1:1+nx,1:1+ny,0:nz])/(2*h)

def _laplace3d(a,hx,hy,hz,mode):
    a=np.asarray(a,dtype=float)
    if a.ndim!=3:
        raise TAYError("LAPLACE currently requires a 3D FIELD.")
    hx=float(hx); hy=float(hy); hz=float(hz)
    if min(hx,hy,hz)<=0:
        raise TAYError("Grid spacing must be positive.")
    p=_pad(a,mode); nx,ny,nz=a.shape
    c=p[1:1+nx,1:1+ny,1:1+nz]
    d2x=(p[2:2+nx,1:1+ny,1:1+nz]-2*c+p[0:nx,1:1+ny,1:1+nz])/(hx*hx)
    d2y=(p[1:1+nx,2:2+ny,1:1+nz]-2*c+p[1:1+nx,0:ny,1:1+nz])/(hy*hy)
    d2z=(p[1:1+nx,1:1+ny,2:2+nz]-2*c+p[1:1+nx,1:1+ny,0:nz])/(hz*hz)
    return d2x+d2y+d2z


def _zeros(*shape):
    dims=tuple(int(x) for x in shape)
    if not dims or any(x<=0 for x in dims):
        raise TAYError("ZEROS requires positive dimensions.")
    return np.zeros(dims,dtype=float)

def _ones(*shape):
    dims=tuple(int(x) for x in shape)
    if not dims or any(x<=0 for x in dims):
        raise TAYError("ONES requires positive dimensions.")
    return np.ones(dims,dtype=float)

def _full(value,*shape):
    dims=tuple(int(x) for x in shape)
    if not dims or any(x<=0 for x in dims):
        raise TAYError("FULL(value, dims...) requires positive dimensions.")
    return np.full(dims,float(value),dtype=float)

def _linspace(a,b,n):
    return np.linspace(float(a),float(b),int(n))

def _range(a,b=None,step=1):
    if b is None:
        return np.arange(0,float(a),float(step))
    return np.arange(float(a),float(b),float(step))

def _truth(value):
    if isinstance(value,np.ndarray):
        if value.ndim==0:
            return bool(value)
        raise TAYError("IF/SOLVE condition must reduce to one boolean.")
    return bool(value)

def _change(old,new):
    a=np.asarray(new,dtype=float)
    b=np.asarray(old,dtype=float)
    if a.shape!=b.shape:
        return float("inf")
    return float(np.sqrt(np.sum((a-b)**2)))

class _Eval(ast.NodeVisitor):
    def __init__(self,names):
        self.names=names

    def visit_Expression(self,n): return self.visit(n.body)

    def visit_Name(self,n):
        if n.id not in self.names:
            raise TAYError(f"Unknown name: {n.id}")
        return self.names[n.id]

    def visit_Constant(self,n):
        if isinstance(n.value,(int,float,bool,str)):
            return n.value
        raise TAYError("Constant type not allowed.")

    def visit_List(self,n):
        return np.asarray([self.visit(x) for x in n.elts])

    def visit_Tuple(self,n):
        return tuple(self.visit(x) for x in n.elts)

    def visit_BinOp(self,n):
        a,b=self.visit(n.left),self.visit(n.right)
        if isinstance(n.op,ast.Add): return a+b
        if isinstance(n.op,ast.Sub): return a-b
        if isinstance(n.op,ast.Mult): return a*b
        if isinstance(n.op,ast.Div): return a/b
        if isinstance(n.op,ast.Pow): return a**b
        if isinstance(n.op,ast.Mod): return a%b
        raise TAYError("Operator not allowed.")

    def visit_UnaryOp(self,n):
        x=self.visit(n.operand)
        if isinstance(n.op,ast.UAdd): return +x
        if isinstance(n.op,ast.USub): return -x
        if isinstance(n.op,ast.Not): return not _truth(x)
        raise TAYError("Unary operator not allowed.")

    def visit_Compare(self,n):
        left=self.visit(n.left)
        result=True
        for op,comp in zip(n.ops,n.comparators):
            right=self.visit(comp)
            if isinstance(op,ast.Lt): ok=left<right
            elif isinstance(op,ast.LtE): ok=left<=right
            elif isinstance(op,ast.Gt): ok=left>right
            elif isinstance(op,ast.GtE): ok=left>=right
            elif isinstance(op,ast.Eq): ok=left==right
            elif isinstance(op,ast.NotEq): ok=left!=right
            else: raise TAYError("Comparison operator not allowed.")
            result = result and _truth(ok)
            left=right
        return result

    def visit_BoolOp(self,n):
        if isinstance(n.op,ast.And):
            for x in n.values:
                if not _truth(self.visit(x)): return False
            return True
        if isinstance(n.op,ast.Or):
            for x in n.values:
                if _truth(self.visit(x)): return True
            return False
        raise TAYError("Boolean operator not allowed.")

    def _index(self,n):
        if isinstance(n,ast.Slice):
            lo=None if n.lower is None else self.visit(n.lower)
            hi=None if n.upper is None else self.visit(n.upper)
            st=None if n.step is None else self.visit(n.step)
            return slice(lo,hi,st)
        if isinstance(n,ast.Tuple):
            return tuple(self._index(x) for x in n.elts)
        if isinstance(n,ast.List):
            return np.asarray([self.visit(x) for x in n.elts],dtype=int)
        if isinstance(n,ast.Constant) and n.value is Ellipsis:
            return Ellipsis
        return self.visit(n)

    def visit_Subscript(self,n):
        value=self.visit(n.value)
        return value[self._index(n.slice)]

    def visit_Call(self,n):
        if not isinstance(n.func,ast.Name):
            raise TAYError("Only direct TAY calls are allowed.")
        fn=self.names.get(n.func.id)
        if not callable(fn):
            raise TAYError(f"Unknown function/primitive: {n.func.id}")
        if n.keywords:
            raise TAYError("Keyword arguments are not supported in v0.7.")
        return fn(*[self.visit(x) for x in n.args])

    def generic_visit(self,n):
        raise TAYError(f"Expression element not allowed: {type(n).__name__}")

def _clean(source):
    out=[]
    for raw in source.splitlines():
        line=raw.split("#",1)[0].strip()
        if line: out.append(line)
    return out

def _parse(lines,i=0,stops=None):
    stops=set(stops or [])
    nodes=[]

    while i<len(lines):
        line=lines[i]
        up=line.upper()

        if up in stops:
            return nodes,i,up

        if up=="END" or up=="ELSE":
            raise TAYError(f"Unexpected {up}")

        if up.startswith("REPEAT "):
            body,j,stop=_parse(lines,i+1,{"END"})
            if stop!="END": raise TAYError("REPEAT missing END")
            nodes.append(("repeat",line[7:].strip(),body))
            i=j+1
            continue

        if up.startswith("IF "):
            then_body,j,stop=_parse(lines,i+1,{"ELSE","END"})
            else_body=[]
            if stop=="ELSE":
                else_body,k,stop2=_parse(lines,j+1,{"END"})
                if stop2!="END": raise TAYError("IF missing END")
                i=k+1
            else:
                i=j+1
            nodes.append(("if",line[3:].strip(),then_body,else_body))
            continue

        if up.startswith("FUNC "):
            m=re.fullmatch(r"FUNC\s+([A-Za-z_]\w*)\s*\((.*?)\)",line,re.I)
            if not m:
                raise TAYError("FUNC syntax: FUNC name(a,b)")
            name=m.group(1)
            args=[x.strip() for x in m.group(2).split(",") if x.strip()]
            if len(set(args))!=len(args) or any(not x.isidentifier() for x in args):
                raise TAYError("Invalid function parameter list.")
            body,j,stop=_parse(lines,i+1,{"END"})
            if stop!="END": raise TAYError("FUNC missing END")
            nodes.append(("func",name,args,body))
            i=j+1
            continue

        # One-line ergonomic state solver:
        # SOLVE U = expr UNTIL CHANGE < eps MAX 100
        if up.startswith("SOLVE ") and "=" in line:
            m=re.fullmatch(
                r"SOLVE\s+([A-Za-z_]\w*)\s*=\s*(.+?)\s+UNTIL\s+(.+?)\s+MAX\s+(.+)",
                line,re.I
            )
            if m:
                nodes.append(("solve_state",m.group(1),m.group(2).strip(),
                              m.group(3).strip(),m.group(4).strip()))
                i+=1
                continue

        # Block solver:
        # SOLVE 100 UNTIL error < eps
        if up.startswith("SOLVE "):
            m=re.fullmatch(r"SOLVE\s+(.+?)\s+UNTIL\s+(.+)",line,re.I)
            if not m:
                raise TAYError("SOLVE syntax: SOLVE max UNTIL condition")
            body,j,stop=_parse(lines,i+1,{"END"})
            if stop!="END": raise TAYError("SOLVE missing END")
            nodes.append(("solve_block",m.group(1).strip(),m.group(2).strip(),body))
            i=j+1
            continue


        m=re.fullmatch(r"LINEAR\s+([A-Za-z_]\w*)\s*=\s*(.+?)\s*,\s*(.+)",line,re.I)
        if m:
            nodes.append(("linear",m.group(1),m.group(2).strip(),m.group(3).strip()))
            i+=1
            continue

        m=re.fullmatch(r"FIT\s+([A-Za-z_]\w*)\s*=\s*(.+?)\s*,\s*(.+)",line,re.I)
        if m:
            nodes.append(("fit",m.group(1),m.group(2).strip(),m.group(3).strip()))
            i+=1
            continue

        m=re.fullmatch(r"SMOOTH\s+([A-Za-z_]\w*)\s*=\s*(.+?)\s+WINDOW\s+(.+)",line,re.I)
        if m:
            nodes.append(("smooth",m.group(1),m.group(2).strip(),m.group(3).strip()))
            i+=1
            continue

        m=re.fullmatch(r"INTEGRATE\s+([A-Za-z_]\w*)\s*=\s*(.+?)\s+OVER\s+(.+)",line,re.I)
        if m:
            nodes.append(("integrate",m.group(1),m.group(2).strip(),m.group(3).strip()))
            i+=1
            continue

        m=re.fullmatch(r"STATS\s+([A-Za-z_]\w*)\s*=\s*(.+)",line,re.I)
        if m:
            nodes.append(("stats",m.group(1),m.group(2).strip()))
            i+=1
            continue

        m=re.fullmatch(
            r"ODE\s+([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\s*\(\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*\)"
            r"\s+FROM\s+(.+?)\s+TO\s+(.+?)\s+STEP\s+(.+?)\s+METHOD\s+(RK4|EULER)",
            line,re.I
        )
        if m:
            nodes.append(("ode",m.group(1),m.group(2),m.group(5).strip(),m.group(6).strip(),
                          m.group(7).strip(),m.group(8).upper()))
            i+=1
            continue

        m=re.fullmatch(
            r"OPTIMIZE\s+([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\s*\(\s*([A-Za-z_]\w*)\s*\)"
            r"\s+LR\s+(.+?)\s+UNTIL\s+(.+?)\s+MAX\s+(.+)",
            line,re.I
        )
        if m:
            nodes.append(("optimize",m.group(1),m.group(2),m.group(4).strip(),
                          m.group(5).strip(),m.group(6).strip()))
            i+=1
            continue

        if up.startswith("RETURN"):
            expr=line[6:].strip()
            nodes.append(("return",expr))
            i+=1
            continue

        nodes.append(("line",line))
        i+=1

    if stops:
        raise TAYError(f"Missing block terminator: {sorted(stops)}")
    return nodes,i,None

class TAYFunction:
    def __init__(self,runtime,name,args,body):
        self.runtime=runtime
        self.name=name
        self.args=args
        self.body=body

    def __call__(self,*values):
        if len(values)!=len(self.args):
            raise TAYError(f"{self.name} expects {len(self.args)} arguments.")
        child=TAYRuntime(
            env=dict(self.runtime.env),
            boundary=self.runtime.boundary,
            base_dir=self.runtime.base_dir,
            backend=self.runtime.backend
        )
        child.user_functions=self.runtime.user_functions
        child.engine_name=self.runtime.engine_name
        child.engine=self.runtime.engine
        child.engine_config=dict(self.runtime.engine_config)
        child.env.update(dict(zip(self.args,values)))
        try:
            child.block(self.body)
        except _ReturnSignal as r:
            return r.value
        return None

class TAYRuntime:
    def __init__(self,env=None,boundary="ZERO",base_dir=None,backend="NUMPY"):
        self.env=dict(env or {})
        self.boundary=boundary.upper()
        self.pending={}
        self.user_functions={}
        self.engine_name=None
        self.engine=None
        self.engine_config={}
        self.base_dir=Path(base_dir or ".").resolve()
        try:
            self.ops=create_backend(backend)
        except BackendError as e:
            raise TAYError(str(e)) from e
        self.backend=self.ops.name

        # Coerce externally supplied numerical arrays onto the selected backend.
        for _k,_v in list(self.env.items()):
            if isinstance(_v,TAYTable):
                continue
            if isinstance(_v,np.ndarray):
                self.env[_k]=self.ops.array(_v,dtype=None)
                continue
            try:
                import torch
                if torch.is_tensor(_v):
                    if self.backend=="NUMPY":
                        self.env[_k]=self.ops.array(_v.detach().cpu().numpy(),dtype=None)
                    else:
                        self.env[_k]=self.ops.array(_v,dtype=None)
            except Exception:
                pass

        self.env.setdefault("BACKEND",self.backend)
        self.env.setdefault("DEVICE",self.ops.device)
        self.traces={}

    def funcs(self):
        m=self.boundary
        f={
            "ACC6":lambda a:self.ops.ACC6(a,m),
            "ACC26":lambda a:self.ops.ACC26(a,m),
            "AVG6":lambda a:self.ops.AVG6(a,m),
            "AVG26":lambda a:self.ops.AVG26(a,m),
            "LAPLACE6":lambda a:self.ops.LAPLACE6(a,m),
            "DX":lambda a,h=1.0:self.ops.derivative(a,0,h,m),
            "DY":lambda a,h=1.0:self.ops.derivative(a,1,h,m),
            "DZ":lambda a,h=1.0:self.ops.derivative(a,2,h,m),
            "LAPLACE":lambda a,hx=1.0,hy=1.0,hz=1.0:self.ops.LAPLACE(a,hx,hy,hz,m),
            "GRAD":lambda a:self.ops.GRAD(a,m),
            "SCALE":lambda a,k:self.ops.SCALE(a,k),
            "SUM":lambda a:self.ops.SUM(a),
            "MEAN":lambda a:self.ops.MEAN(a),
            "MIN":lambda a:self.ops.MIN(a),
            "MAX":lambda a:self.ops.MAX(a),
            "ABS":lambda a:self.ops.ABS(a),
            "SQRT":lambda a:self.ops.SQRT(a),
            "NORM":lambda a:self.ops.NORM(a),
            "CLIP":lambda a,lo,hi:self.ops.CLIP(a,lo,hi),
            "LEN":lambda a:len(a),
            "ZEROS":lambda *shape:self.ops.ZEROS(*shape),
            "ONES":lambda *shape:self.ops.ONES(*shape),
            "FULL":lambda value,*shape:self.ops.FULL(value,*shape),
            "LINSPACE":lambda a,b,n:self.ops.LINSPACE(a,b,n),
            "RANGE":lambda a,b=None,step=1:self.ops.RANGE(a,b,step),
            "CSV":lambda p:TAYTable.from_csv((self.base_dir/str(p)).resolve()),
            "COL":lambda t,c:t.column(c) if isinstance(t,TAYTable) else (_ for _ in ()).throw(TAYError("COL requires TABLE.")),
            "ROWS":lambda t:len(t) if isinstance(t,TAYTable) else len(t),
            "NCOLS":lambda t:len(t.columns) if isinstance(t,TAYTable) else t.shape[1],
            "MISSINGCOUNT":lambda t,c=None:t.missing_count(c) if isinstance(t,TAYTable) else (_ for _ in ()).throw(TAYError("MISSINGCOUNT requires TABLE.")),
        }
        f.update(self.user_functions)
        return f

    def eval(self,s):
        names=dict(self.env); names.update(self.funcs())
        return _Eval(names).visit(ast.parse(s,mode="eval"))

    def commit(self):
        self.env.update(self.pending)
        self.pending.clear()

    def _has_numeric_array_state(self):
        for k,v in self.env.items():
            if k.startswith("TRACE_") or isinstance(v,TAYTable):
                continue
            if isinstance(v,np.ndarray):
                return True
            try:
                import torch
                if torch.is_tensor(v):
                    return True
            except Exception:
                pass
        return False

    def _set_backend(self,name):
        name=str(name).upper()
        if name==self.backend:
            self.env["BACKEND"]=self.backend
            self.env["DEVICE"]=self.ops.device
            return
        if self._has_numeric_array_state():
            raise TAYError("BACKEND must be selected before numerical VECTOR/FIELD state is created in v0.8.")
        try:
            ops=create_backend(name)
        except BackendError as e:
            raise TAYError(str(e)) from e
        self.backend=ops.name
        self.ops=ops
        self.env["BACKEND"]=ops.name
        self.env["DEVICE"]=ops.device

    def _set_engine(self,name):
        from .engines import EngineError, create_engine

        try:
            engine=create_engine(name)
        except EngineError as e:
            raise TAYError(str(e)) from e
        self.engine_name=engine.name
        self.engine=engine
        self.engine_config={}
        self.env["ENGINE"]=engine.name

    def _set_engine_config(self,key,value):
        if self.engine is None:
            raise TAYError(
                f"{key.upper()} is an engine setting; select ENGINE TAYLANUS first."
            )
        self.engine_config[str(key).lower()]=value

    def _run_engine(self,name):
        if self.engine is None:
            raise TAYError("RUN TAYLANUS requires ENGINE TAYLANUS first.")
        requested=str(name).strip().upper()
        if requested not in ("ENGINE",self.engine_name):
            raise TAYError(
                f"Selected engine is {self.engine_name}, not {requested}."
            )
        from .engines import EngineError

        try:
            result=self.engine.run(
                self.engine_config,
                base_dir=self.base_dir,
                backend=self.backend,
            )
        except EngineError as e:
            raise TAYError(str(e)) from e
        self.env.update(self.engine.environment_values(result))
        self.env["ENGINE"]=self.engine_name
        self.env["_"]=result

    def _state_change(self,old,new):
        a=self.ops.to_numpy(new).astype(float,copy=False)
        b=self.ops.to_numpy(old).astype(float,copy=False)
        if a.shape!=b.shape:
            return float("inf")
        return float(np.sqrt(np.sum((a-b)**2)))

    def _assignment_ast(self,target):
        try:
            node=ast.parse(target,mode="eval").body
        except SyntaxError as e:
            raise TAYError(f"Invalid assignment target: {target}") from e
        if isinstance(node,ast.Name):
            return node
        if isinstance(node,ast.Subscript) and isinstance(node.value,ast.Name):
            return node
        raise TAYError("Assignment target must be a name or direct slice/subscript such as A[1:4,:].")

    def _assign(self,target,value):
        node=self._assignment_ast(target)
        if isinstance(node,ast.Name):
            self.env[node.id]=value
            return
        name=node.value.id
        if name not in self.env:
            raise TAYError(f"Unknown assignment target: {name}")
        evaluator=_Eval(dict(self.env,**self.funcs()))
        key=evaluator._index(node.slice)
        obj=self.env[name]
        try:
            obj[key]=value
        except Exception as e:
            raise TAYError(f"Slice assignment failed for {target}: {e}") from e
        self.env[name]=obj

    def _assign_typed(self,kind,name,value):
        k=kind.upper()
        if k=="TABLE":
            if not isinstance(value,TAYTable):
                raise TAYError(f"{name} must be TABLE.")
            self.env[name]=value
            return

        try:
            nd=self.ops.ndim(value)
        except Exception as e:
            raise TAYError(f"Cannot determine type of {name}: {e}") from e

        if k=="SCALAR":
            if nd!=0:
                raise TAYError(f"{name} must be SCALAR.")
            value=self.ops.scalar(value)
        elif k=="VECTOR":
            if nd!=1:
                raise TAYError(f"{name} must be VECTOR (1D).")
            value=self.ops.array(value,dtype=float)
        elif k=="FIELD":
            if nd not in (2,3):
                raise TAYError(f"{name} must be FIELD (2D/3D).")
            value=self.ops.array(value,dtype=float)
        else:
            raise TAYError(f"Unknown type: {kind}")
        self.env[name]=value

    def _save(self,name,path_text):
        if name not in self.env:
            raise TAYError(f"Unknown name: {name}")
        path=(self.base_dir/path_text).resolve()
        path.parent.mkdir(parents=True,exist_ok=True)
        value=self.env[name]
        if isinstance(value,TAYTable):
            if path.suffix.lower()!=".csv":
                raise TAYError("TABLE SAVE currently supports only .csv.")
            value.to_csv(path)
            return
        a=self.ops.to_numpy(value)
        if path.suffix.lower()==".npy":
            np.save(path,a)
        elif path.suffix.lower()==".csv":
            if a.ndim>2:
                raise TAYError("CSV SAVE supports only scalar/vector/matrix values.")
            np.savetxt(path,np.atleast_2d(a),delimiter=",")
        else:
            raise TAYError("SAVE supports .npy or .csv.")

    def _load(self,name,path_text):
        path=(self.base_dir/path_text).resolve()
        if path.suffix.lower()==".npy":
            value=self.ops.array(np.load(path,allow_pickle=False),dtype=float)
        elif path.suffix.lower()==".csv":
            value=self.ops.array(np.loadtxt(path,delimiter=","),dtype=float)
        else:
            raise TAYError("LOAD supports .npy or .csv.")
        self.env[name]=value


    def _region_assign(self,name,spec,value):
        if name not in self.env:
            raise TAYError(f"Unknown REGION target: {name}")
        a=self.env[name]
        parts=[x.strip() for x in spec.split(",")]
        if len(parts)!=self.ops.ndim(a):
            raise TAYError(f"REGION for {name} needs exactly {self.ops.ndim(a)} axis ranges.")
        slices=[]
        for part in parts:
            if ":" not in part:
                raise TAYError("REGION axes must use start:end form.")
            lo,hi=part.split(":",1)
            lo_v=None if lo.strip()=="" else int(self.eval(lo.strip()))
            hi_v=None if hi.strip()=="" else int(self.eval(hi.strip()))
            slices.append(slice(lo_v,hi_v))
        a[tuple(slices)]=value
        self.env[name]=a

    def _plot_slice(self,name,axis,index,path_text):
        if name not in self.env:
            raise TAYError(f"Unknown plot field: {name}")
        a=self.ops.to_numpy(self.env[name])
        if a.ndim!=3:
            raise TAYError("PLOT SLICE requires a 3D FIELD.")
        axis=axis.upper()
        idx=int(self.eval(index))
        amap={"X":0,"Y":1,"Z":2}
        if axis not in amap:
            raise TAYError("PLOT SLICE axis must be X, Y, or Z.")
        ax=amap[axis]
        if idx<0 or idx>=a.shape[ax]:
            raise TAYError("PLOT SLICE index is out of range.")
        sl=[slice(None)]*3
        sl[ax]=idx
        img=a[tuple(sl)]
        path=(self.base_dir/path_text).resolve()
        path.parent.mkdir(parents=True,exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig=plt.figure()
        plot_ax=fig.add_subplot(111)
        im=plot_ax.imshow(img,origin="lower",aspect="auto")
        fig.colorbar(im,ax=plot_ax)
        plot_ax.set_title(f"{name} {axis}={idx}")
        fig.tight_layout()
        fig.savefig(path,dpi=140)
        plt.close(fig)
        self.env["LAST_PLOT"]=str(path)

    def _plot_trace(self,name,path_text):
        if name not in self.traces:
            raise TAYError(f"Unknown trace: {name}")
        y=np.asarray(self.traces[name],dtype=float)
        x=np.arange(1,len(y)+1)
        path=(self.base_dir/path_text).resolve()
        path.parent.mkdir(parents=True,exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig=plt.figure()
        plot_ax=fig.add_subplot(111)
        plot_ax.plot(x,y)
        plot_ax.set_xlabel("sample")
        plot_ax.set_ylabel(name)
        plot_ax.set_title(name)
        fig.tight_layout()
        fig.savefig(path,dpi=140)
        plt.close(fig)
        self.env["LAST_PLOT"]=str(path)

    def line(self,line):
        up=line.upper()

        if up.startswith("PARAM "):
            body=line[6:].strip()
            if not body:
                raise TAYError("PARAM requires name=value pairs.")
            for item in body.split(","):
                if "=" not in item:
                    raise TAYError("PARAM syntax: PARAM a=1, b=2")
                name,expr=item.split("=",1)
                name=name.strip()
                if not name.isidentifier():
                    raise TAYError(f"Invalid PARAM name: {name}")
                value=self.eval(expr.strip())
                if self.ops.ndim(value)!=0:
                    raise TAYError("PARAM values must be scalar.")
                self.env[name]=self.ops.scalar(value)
            return

        if up.startswith("BACKEND "):
            name=line.split(None,1)[1].strip().upper()
            self._set_backend(name)
            return

        if up.startswith("ENGINE "):
            name=line.split(None,1)[1].strip().upper()
            self._set_engine(name)
            return

        if self.engine is not None:
            for directive,key in (
                ("RESOLUTION","resolution"),
                ("DT","dt"),
                ("TEND","tend"),
                ("VISCOSITY","viscosity"),
                ("SLICE","slice_index"),
                ("PLANNER_REPS","planner_reps"),
            ):
                m=re.fullmatch(
                    rf"{directive}\s*(?:=\s*)?(.+)",line,re.I
                )
                if m:
                    self._set_engine_config(key,self.eval(m.group(1).strip()))
                    return

            m=re.fullmatch(r"MODE\s*(?:=\s*)?([A-Za-z_]+)",line,re.I)
            if m:
                self._set_engine_config("mode",m.group(1).upper())
                return

            m=re.fullmatch(r'OUTPUT\s*(?:=\s*)?(.+)',line,re.I)
            if m:
                value=self.eval(m.group(1).strip())
                if not isinstance(value,str):
                    raise TAYError("OUTPUT must be a quoted directory path.")
                self._set_engine_config("output",value)
                return

            m=re.fullmatch(r'REFERENCE\s*(?:=\s*)?(.+)',line,re.I)
            if m:
                raw=m.group(1).strip()
                value=None if raw.upper()=="NONE" else self.eval(raw)
                if value is not None and not isinstance(value,str):
                    raise TAYError("REFERENCE must be a quoted .npy path or NONE.")
                self._set_engine_config("reference",value)
                return

        m=re.fullmatch(r"RUN\s+(TAYLANUS|ENGINE)",line,re.I)
        if m:
            self._run_engine(m.group(1))
            return

        m=re.fullmatch(r"REGION\s+([A-Za-z_]\w*)\[(.+)\]\s*=\s*(.+)",line,re.I)
        if m:
            self._region_assign(m.group(1),m.group(2),self.eval(m.group(3).strip()))
            return

        m=re.fullmatch(r"TRACE\s+([A-Za-z_]\w*)\s*=\s*(.+)",line,re.I)
        if m:
            name=m.group(1)
            value=self.eval(m.group(2).strip())
            if self.ops.ndim(value)!=0:
                raise TAYError("TRACE currently requires a scalar expression.")
            val=float(self.ops.scalar(value))
            self.traces.setdefault(name,[]).append(val)
            self.env[name]=val
            self.env["TRACE_"+name]=np.asarray(self.traces[name],dtype=float)
            return

        m=re.fullmatch(r'PLOT\s+SLICE\s+([A-Za-z_]\w*)\s+([XYZ])\s+(.+?)\s+TO\s+"([^"]+)"',line,re.I)
        if m:
            self._plot_slice(m.group(1),m.group(2),m.group(3),m.group(4))
            return

        m=re.fullmatch(r'PLOT\s+TRACE\s+([A-Za-z_]\w*)\s+TO\s+"([^"]+)"',line,re.I)
        if m:
            self._plot_trace(m.group(1),m.group(2))
            return

        if up.startswith("BOUNDARY "):
            mode=line.split(None,1)[1].upper()
            if mode not in ("ZERO","WRAP","EDGE"):
                raise TAYError("BOUNDARY must be ZERO, WRAP, or EDGE.")
            self.boundary=mode
            return

        if up=="COMMIT":
            self.commit()
            return

        if up.startswith("NEXT "):
            rest=line[5:].strip()
            if "=" not in rest:
                raise TAYError("NEXT syntax: NEXT name = expression")
            name,expr=rest.split("=",1)
            name=name.strip()
            if not name.isidentifier():
                raise TAYError("Invalid NEXT target.")
            self.pending[name]=self.eval(expr.strip())
            return

        for kind in ("SCALAR","VECTOR","FIELD","TABLE"):
            if up.startswith(kind+" "):
                rest=line[len(kind):].strip()
                if "=" not in rest:
                    raise TAYError(f"{kind} syntax: {kind} name = expression")
                name,expr=rest.split("=",1)
                name=name.strip()
                if not name.isidentifier():
                    raise TAYError("Invalid typed target.")
                self._assign_typed(kind,name,self.eval(expr.strip()))
                return

        m=re.fullmatch(r'SAVE\s+([A-Za-z_]\w*)\s+TO\s+"([^"]+)"',line,re.I)
        if m:
            self._save(m.group(1),m.group(2))
            return

        m=re.fullmatch(r'LOAD\s+([A-Za-z_]\w*)\s+FROM\s+"([^"]+)"',line,re.I)
        if m:
            self._load(m.group(1),m.group(2))
            return

        m=re.fullmatch(r'USE\s+"([^"]+)"',line,re.I)
        if m:
            path=(self.base_dir/m.group(1)).resolve()
            src=path.read_text(encoding="utf-8")
            old_base=self.base_dir
            self.base_dir=path.parent
            try:
                nodes,_,_=_parse(_clean(src))
                self.block(nodes)
            finally:
                self.base_dir=old_base
            return

        m=re.fullmatch(r"KEEP\s+([A-Za-z_]\w*)\s*=\s*(.+)",line,re.I)
        if m:
            name=m.group(1)
            if name not in self.env or not isinstance(self.env[name],TAYTable):
                raise TAYError(f"KEEP requires TABLE target: {name}")
            cols=[x.strip() for x in m.group(2).split(",") if x.strip()]
            try:
                self.env[name]=self.env[name].keep(cols)
            except TableError as e:
                raise TAYError(str(e)) from e
            return

        m=re.fullmatch(r"DROP\s+([A-Za-z_]\w*)\s*=\s*(.+)",line,re.I)
        if m:
            name=m.group(1)
            if name not in self.env or not isinstance(self.env[name],TAYTable):
                raise TAYError(f"DROP requires TABLE target: {name}")
            cols=[x.strip() for x in m.group(2).split(",") if x.strip()]
            try:
                self.env[name]=self.env[name].drop(cols)
            except TableError as e:
                raise TAYError(str(e)) from e
            return

        m=re.fullmatch(r"FILTER\s+([A-Za-z_]\w*)\s+WHERE\s+(.+)",line,re.I)
        if m:
            name=m.group(1)
            if name not in self.env or not isinstance(self.env[name],TAYTable):
                raise TAYError(f"FILTER requires TABLE target: {name}")
            try:
                self.env[name]=self.env[name].filter(m.group(2).strip(),self.env)
            except TableError as e:
                raise TAYError(str(e)) from e
            return

        m=re.fullmatch(r"FILL\s+([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s*=\s*(.+)",line,re.I)
        if m:
            name,col,raw=m.group(1),m.group(2),m.group(3).strip()
            if name not in self.env or not isinstance(self.env[name],TAYTable):
                raise TAYError(f"FILL requires TABLE target: {name}")
            upper_raw=raw.upper()
            strategy=upper_raw if upper_raw in {"MEAN","MEDIAN","MODE","ZERO","DROP"} else self.eval(raw)
            try:
                self.env[name]=self.env[name].fill(col,strategy)
            except TableError as e:
                raise TAYError(str(e)) from e
            return

        m=re.fullmatch(r"SORT\s+([A-Za-z_]\w*)\s+BY\s+([A-Za-z_]\w*)(?:\s+(ASC|DESC))?",line,re.I)
        if m:
            name,col,order=m.group(1),m.group(2),(m.group(3) or "ASC").upper()
            if name not in self.env or not isinstance(self.env[name],TAYTable):
                raise TAYError(f"SORT requires TABLE target: {name}")
            try:
                self.env[name]=self.env[name].sort(col,descending=(order=="DESC"))
            except TableError as e:
                raise TAYError(str(e)) from e
            return

        m=re.fullmatch(r"GROUP\s+([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\s+BY\s+(.+?)\s+SUMMARIZE\s+(.+)",line,re.I)
        if m:
            outname,src,byraw,sraw=m.group(1),m.group(2),m.group(3),m.group(4)
            if src not in self.env or not isinstance(self.env[src],TAYTable):
                raise TAYError(f"GROUP requires TABLE source: {src}")
            by=[x.strip() for x in byraw.split(",") if x.strip()]
            specs=[]
            for item in sraw.split(","):
                if ":" not in item:
                    raise TAYError("SUMMARIZE items must use column:AGG.")
                col,agg=item.split(":",1)
                specs.append((col.strip(),agg.strip()))
            try:
                self.env[outname]=self.env[src].group(by,specs)
            except TableError as e:
                raise TAYError(str(e)) from e
            return

        # Generic assignment. Do not confuse comparisons (==, <=, >=, !=) with assignment.
        m=re.match(r"^(.+?)(?<![<>=!])=(?!=)(.+)$",line)
        if m:
            target=m.group(1).strip()
            expr=m.group(2).strip()
            self._assign(target,self.eval(expr))
            return

        value=self.eval(line)
        self.env["_"]=value
        return value

    def block(self,nodes):
        for node in nodes:
            kind=node[0]

            if kind=="line":
                self.line(node[1])

            elif kind=="repeat":
                count=int(self.eval(node[1]))
                if count<0: raise TAYError("REPEAT count must be non-negative.")
                for i in range(count):
                    self.env["ITER"]=i+1
                    self.block(node[2])

            elif kind=="if":
                branch=node[2] if _truth(self.eval(node[1])) else node[3]
                self.block(branch)

            elif kind=="func":
                fn=TAYFunction(self,node[1],node[2],node[3])
                self.user_functions[node[1]]=fn

            elif kind=="return":
                value=self.eval(node[1]) if node[1] else None
                raise _ReturnSignal(value)

            elif kind=="solve_block":
                max_iter=int(self.eval(node[1]))
                self.env["SOLVED"]=False
                for i in range(max_iter):
                    self.env["ITER"]=i+1
                    self.block(node[3])
                    if _truth(self.eval(node[2])):
                        self.env["SOLVED"]=True
                        break

            elif kind=="solve_state":
                target,expr,cond,max_expr=node[1],node[2],node[3],node[4]
                if target not in self.env:
                    raise TAYError(f"SOLVE target does not exist: {target}")
                max_iter=int(self.eval(max_expr))
                self.env["SOLVED"]=False
                for i in range(max_iter):
                    old=self.ops.clone(self.env[target]) if self.ops.is_array(self.env[target]) else copy.deepcopy(self.env[target])
                    new=self.eval(expr)
                    self.env[target]=new
                    self.env["CHANGE"]=self._state_change(old,new)
                    self.env["ITER"]=i+1
                    if _truth(self.eval(cond)):
                        self.env["SOLVED"]=True
                        break


            elif kind=="linear":
                self.env[node[1]]=_linear_solve(self.eval(node[2]),self.eval(node[3]))

            elif kind=="fit":
                self.env[node[1]]=_least_squares(self.eval(node[2]),self.eval(node[3]))

            elif kind=="smooth":
                self.env[node[1]]=_smooth1d(self.eval(node[2]),self.eval(node[3]))

            elif kind=="integrate":
                self.env[node[1]]=_integrate_xy(self.eval(node[2]),self.eval(node[3]))

            elif kind=="stats":
                self.env[node[1]]=_stats5(self.eval(node[2]))

            elif kind=="ode":
                target,fn_name,t0_expr,t1_expr,h_expr,method=node[1:]
                if target not in self.env:
                    raise TAYError(f"ODE target does not exist: {target}")
                fn=self.user_functions.get(fn_name)
                if fn is None:
                    candidate=self.funcs().get(fn_name)
                    if callable(candidate): fn=candidate
                if fn is None:
                    raise TAYError(f"ODE function not found: {fn_name}")
                t=float(self.eval(t0_expr)); t1=float(self.eval(t1_expr)); h=float(self.eval(h_expr))
                if h<=0 or t1<t:
                    raise TAYError("ODE requires STEP > 0 and TO >= FROM.")
                y=np.array(self.env[target],copy=True) if isinstance(self.env[target],np.ndarray) else float(self.env[target])
                steps=0
                while t < t1-1e-15:
                    hs=min(h,t1-t)
                    if method=="EULER":
                        y=y+hs*fn(y,t)
                    else:
                        k1=fn(y,t)
                        k2=fn(y+0.5*hs*k1,t+0.5*hs)
                        k3=fn(y+0.5*hs*k2,t+0.5*hs)
                        k4=fn(y+hs*k3,t+hs)
                        y=y+(hs/6.0)*(k1+2*k2+2*k3+k4)
                    t+=hs; steps+=1
                self.env[target]=y
                self.env["TIME"]=t
                self.env["STEPS"]=steps

            elif kind=="optimize":
                target,fn_name,lr_expr,cond,max_expr=node[1:]
                if target not in self.env:
                    raise TAYError(f"OPTIMIZE target does not exist: {target}")
                fn=self.user_functions.get(fn_name)
                if fn is None:
                    candidate=self.funcs().get(fn_name)
                    if callable(candidate): fn=candidate
                if fn is None:
                    raise TAYError(f"OPTIMIZE function not found: {fn_name}")
                lr=float(self.eval(lr_expr)); max_iter=int(self.eval(max_expr))
                if lr<=0:
                    raise TAYError("OPTIMIZE LR must be positive.")
                self.env["SOLVED"]=False
                x=np.array(self.env[target],dtype=float,copy=True) if isinstance(self.env[target],np.ndarray) else float(self.env[target])
                for i in range(max_iter):
                    old=np.array(x,copy=True) if isinstance(x,np.ndarray) else float(x)
                    g=_fd_gradient(fn,x)
                    x=x-lr*g
                    self.env[target]=x
                    self.env["CHANGE"]=self._state_change(old,x)
                    self.env["OBJECTIVE"]=float(fn(x))
                    self.env["ITER"]=i+1
                    if _truth(self.eval(cond)):
                        self.env["SOLVED"]=True
                        break

            else:
                raise TAYError(f"Unknown AST node: {kind}")

    def run(self,source):
        nodes,_,_=_parse(_clean(source))
        self.block(nodes)
        if self.pending:
            self.commit()
        return self.env

def run_tay(source,env=None,boundary="ZERO",base_dir=None,backend="NUMPY"):
    return TAYRuntime(env,boundary,base_dir,backend=backend).run(source)
