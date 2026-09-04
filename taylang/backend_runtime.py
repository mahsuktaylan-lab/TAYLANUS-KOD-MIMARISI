from __future__ import annotations
from pathlib import Path
import numpy as np

from .backends import create_backend, BackendError
from .table import TAYTable

class BackendOps:
    """Runtime dispatch object used by transpiled TAY programs."""

    def __init__(self,backend="NUMPY",boundary="ZERO"):
        self.boundary=str(boundary).upper()
        self.backend=create_backend(backend)

    @property
    def name(self): return self.backend.name
    @property
    def device(self): return self.backend.device

    def set_backend(self,name):
        self.backend=create_backend(name)

    def set_boundary(self,mode):
        mode=str(mode).upper()
        if mode not in {"ZERO","WRAP","EDGE"}:
            raise BackendError("BOUNDARY must be ZERO, WRAP, or EDGE.")
        self.boundary=mode

    def ndim(self,x): return self.backend.ndim(x)
    def scalar(self,x): return self.backend.scalar(x)
    def to_numpy(self,x): return self.backend.to_numpy(x)
    def array(self,x,dtype=None): return self.backend.array(x,dtype=dtype)

    def ACC6(self,a): return self.backend.ACC6(a,self.boundary)
    def ACC26(self,a): return self.backend.ACC26(a,self.boundary)
    def AVG6(self,a): return self.backend.AVG6(a,self.boundary)
    def AVG26(self,a): return self.backend.AVG26(a,self.boundary)
    def LAPLACE6(self,a): return self.backend.LAPLACE6(a,self.boundary)
    def DX(self,a,h=1.0): return self.backend.derivative(a,0,h,self.boundary)
    def DY(self,a,h=1.0): return self.backend.derivative(a,1,h,self.boundary)
    def DZ(self,a,h=1.0): return self.backend.derivative(a,2,h,self.boundary)
    def LAPLACE(self,a,hx=1.0,hy=1.0,hz=1.0): return self.backend.LAPLACE(a,hx,hy,hz,self.boundary)
    def GRAD(self,a): return self.backend.GRAD(a,self.boundary)
    def SCALE(self,a,k): return self.backend.SCALE(a,k)
    def SUM(self,a): return self.backend.SUM(a)
    def MEAN(self,a): return self.backend.MEAN(a)
    def MIN(self,a): return self.backend.MIN(a)
    def MAX(self,a): return self.backend.MAX(a)
    def ABS(self,a): return self.backend.ABS(a)
    def SQRT(self,a): return self.backend.SQRT(a)
    def NORM(self,a): return self.backend.NORM(a)
    def CLIP(self,a,lo,hi): return self.backend.CLIP(a,lo,hi)
    def LEN(self,a): return len(a)
    def ZEROS(self,*shape): return self.backend.ZEROS(*shape)
    def ONES(self,*shape): return self.backend.ONES(*shape)
    def FULL(self,value,*shape): return self.backend.FULL(value,*shape)
    def LINSPACE(self,a,b,n): return self.backend.LINSPACE(a,b,n)
    def RANGE(self,a,b=None,step=1): return self.backend.RANGE(a,b,step)

    # Structured-data operations remain CPU/pandas backed.
    def CSV(self,path): return TAYTable.from_csv(Path(path))
    def COL(self,t,c): return t.column(c)
    def ROWS(self,t): return len(t)
    def NCOLS(self,t): return len(t.columns) if isinstance(t,TAYTable) else self.ndim(t) and t.shape[1]
    def MISSINGCOUNT(self,t,c=None): return t.missing_count(c)

    def save(self,value,path):
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        if isinstance(value,TAYTable):
            if path.suffix.lower()!=".csv":
                raise BackendError("TABLE SAVE supports CSV.")
            value.to_csv(path)
            return
        a=self.to_numpy(value)
        if path.suffix.lower()==".npy":
            np.save(path,a)
        elif path.suffix.lower()==".csv":
            if a.ndim>2:
                raise BackendError("CSV SAVE supports scalar/vector/matrix values.")
            np.savetxt(path,np.atleast_2d(a),delimiter=",")
        else:
            raise BackendError("SAVE supports .npy or .csv.")

    def load(self,path):
        path=Path(path)
        if path.suffix.lower()==".npy":
            return self.array(np.load(path,allow_pickle=False),dtype=float)
        if path.suffix.lower()==".csv":
            return self.array(np.loadtxt(path,delimiter=","),dtype=float)
        raise BackendError("LOAD supports .npy or .csv.")
