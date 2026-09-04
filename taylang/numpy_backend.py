import numpy as np
from pathlib import Path
from .table import TAYTable
from .core import _acc,_grad,_scale,_daxis,_laplace3d,_zeros,_ones,_full,_linspace,_range,DIR6,DIR26,TAYError

class NumpyOps:
    def __init__(self,boundary="ZERO"):
        self.boundary=boundary.upper()
    def set_boundary(self,mode):
        mode=mode.upper()
        if mode not in ("ZERO","WRAP","EDGE"):
            raise TAYError("Bad boundary mode")
        self.boundary=mode
    def ACC6(self,a): return _acc(a,DIR6,self.boundary)
    def ACC26(self,a): return _acc(a,DIR26,self.boundary)
    def AVG6(self,a): return _acc(a,DIR6,self.boundary)/6.0
    def AVG26(self,a): return _acc(a,DIR26,self.boundary)/26.0
    def LAPLACE6(self,a): return _acc(a,DIR6,self.boundary)-6.0*np.asarray(a)
    def DX(self,a,h=1.0): return _daxis(a,0,h,self.boundary)
    def DY(self,a,h=1.0): return _daxis(a,1,h,self.boundary)
    def DZ(self,a,h=1.0): return _daxis(a,2,h,self.boundary)
    def LAPLACE(self,a,hx=1.0,hy=1.0,hz=1.0): return _laplace3d(a,hx,hy,hz,self.boundary)
    def GRAD(self,a): return _grad(a,self.boundary)
    def SCALE(self,a,k): return _scale(a,k)
    def SUM(self,a): return np.asarray(a).sum()
    def MEAN(self,a): return np.asarray(a).mean()
    def MIN(self,a): return np.asarray(a).min()
    def MAX(self,a): return np.asarray(a).max()
    def ABS(self,a): return np.abs(a)
    def SQRT(self,a): return np.sqrt(a)
    def NORM(self,a): return np.sqrt(np.sum(np.asarray(a,dtype=float)**2))
    def CLIP(self,a,lo,hi): return np.clip(a,lo,hi)
    def LEN(self,a): return len(a)
    def ZEROS(self,*shape): return _zeros(*shape)
    def ONES(self,*shape): return _ones(*shape)
    def FULL(self,value,*shape): return _full(value,*shape)
    def LINSPACE(self,a,b,count): return _linspace(a,b,count)
    def RANGE(self,a,b=None,step=1): return _range(a,b,step)
    def CSV(self,path): return TAYTable.from_csv(Path(path))
    def COL(self,t,name): return t.column(name)
    def ROWS(self,t): return len(t)
    def NCOLS(self,t): return len(t.columns) if isinstance(t,TAYTable) else np.asarray(t).shape[1]
    def MISSINGCOUNT(self,t,name=None): return t.missing_count(name)
