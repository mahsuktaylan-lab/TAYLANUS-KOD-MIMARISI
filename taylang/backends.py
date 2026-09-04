from __future__ import annotations
import numpy as np

class BackendError(Exception):
    pass

DIR6=[(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]
DIR26=[(dx,dy,dz) for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1)
       if (dx,dy,dz)!=(0,0,0)]

class NumPyBackend:
    name="NUMPY"
    device="cpu"

    def is_array(self,x): return isinstance(x,np.ndarray)
    def ndim(self,x): return np.asarray(x).ndim
    def scalar(self,x):
        a=np.asarray(x)
        if a.ndim!=0: raise BackendError("Expected scalar.")
        return a.item()
    def clone(self,x): return np.array(x,copy=True) if isinstance(x,np.ndarray) else x
    def to_numpy(self,x): return np.asarray(x)
    def array(self,x,dtype=None):
        return np.asarray(x,dtype=dtype)
    def _pad(self,a,mode):
        mapping={"ZERO":"constant","WRAP":"wrap","EDGE":"edge"}
        if mode not in mapping: raise BackendError(f"Unknown boundary mode: {mode}")
        return np.pad(np.asarray(a),1,mode=mapping[mode])
    def _acc(self,a,dirs,mode):
        a=np.asarray(a)
        if a.ndim!=3: raise BackendError("Neighborhood primitives require a 3D FIELD.")
        p=self._pad(a,mode); nx,ny,nz=a.shape
        out=np.zeros_like(a,dtype=np.result_type(a.dtype,np.float64))
        for dx,dy,dz in dirs:
            out += p[1+dx:1+dx+nx,1+dy:1+dy+ny,1+dz:1+dz+nz]
        return out
    def ACC6(self,a,mode): return self._acc(a,DIR6,mode)
    def ACC26(self,a,mode): return self._acc(a,DIR26,mode)
    def AVG6(self,a,mode): return self.ACC6(a,mode)/6.0
    def AVG26(self,a,mode): return self.ACC26(a,mode)/26.0
    def LAPLACE6(self,a,mode): return self.ACC6(a,mode)-6.0*np.asarray(a)
    def derivative(self,a,axis,h,mode):
        a=np.asarray(a,dtype=float)
        if a.ndim!=3: raise BackendError("DX/DY/DZ require a 3D FIELD.")
        h=float(h)
        if h<=0: raise BackendError("Grid spacing must be positive.")
        p=self._pad(a,mode); nx,ny,nz=a.shape
        if axis==0:
            return (p[2:2+nx,1:1+ny,1:1+nz]-p[0:nx,1:1+ny,1:1+nz])/(2*h)
        if axis==1:
            return (p[1:1+nx,2:2+ny,1:1+nz]-p[1:1+nx,0:ny,1:1+nz])/(2*h)
        return (p[1:1+nx,1:1+ny,2:2+nz]-p[1:1+nx,1:1+ny,0:nz])/(2*h)
    def LAPLACE(self,a,hx,hy,hz,mode):
        a=np.asarray(a,dtype=float)
        if a.ndim!=3: raise BackendError("LAPLACE requires a 3D FIELD.")
        hx,hy,hz=float(hx),float(hy),float(hz)
        if min(hx,hy,hz)<=0: raise BackendError("Grid spacing must be positive.")
        p=self._pad(a,mode); nx,ny,nz=a.shape
        c=p[1:1+nx,1:1+ny,1:1+nz]
        d2x=(p[2:2+nx,1:1+ny,1:1+nz]-2*c+p[0:nx,1:1+ny,1:1+nz])/(hx*hx)
        d2y=(p[1:1+nx,2:2+ny,1:1+nz]-2*c+p[1:1+nx,0:ny,1:1+nz])/(hy*hy)
        d2z=(p[1:1+nx,1:1+ny,2:2+nz]-2*c+p[1:1+nx,1:1+ny,0:nz])/(hz*hz)
        return d2x+d2y+d2z
    def GRAD(self,a,mode):
        return np.stack((self.derivative(a,0,1,mode),
                         self.derivative(a,1,1,mode),
                         self.derivative(a,2,1,mode)),axis=-1)
    def SCALE(self,a,k):
        a=np.asarray(a); k=int(k)
        if k<=0 or any(n%k for n in a.shape):
            raise BackendError("SCALE requires a positive factor dividing every dimension.")
        x,y,z=a.shape
        return a.reshape(x//k,k,y//k,k,z//k,k).mean(axis=(1,3,5))
    def SUM(self,a): return np.asarray(a).sum()
    def MEAN(self,a): return np.asarray(a).mean()
    def MIN(self,a): return np.asarray(a).min()
    def MAX(self,a): return np.asarray(a).max()
    def ABS(self,a): return np.abs(a)
    def SQRT(self,a): return np.sqrt(a)
    def NORM(self,a): return float(np.sqrt(np.sum(np.asarray(a,dtype=float)**2)))
    def CLIP(self,a,lo,hi): return np.clip(a,lo,hi)
    def ZEROS(self,*shape): return np.zeros(tuple(int(x) for x in shape),dtype=float)
    def ONES(self,*shape): return np.ones(tuple(int(x) for x in shape),dtype=float)
    def FULL(self,value,*shape): return np.full(tuple(int(x) for x in shape),float(value),dtype=float)
    def LINSPACE(self,a,b,n): return np.linspace(float(a),float(b),int(n))
    def RANGE(self,a,b=None,step=1):
        return np.arange(0,float(a),float(step)) if b is None else np.arange(float(a),float(b),float(step))

class TorchBackend:
    def __init__(self,device="cpu"):
        try:
            import torch
        except Exception as e:
            raise BackendError("PyTorch is not installed.") from e
        self.torch=torch
        self.device=str(device)
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise BackendError("CUDA GPU backend requested, but torch.cuda.is_available() is false.")
        self.name="GPU" if self.device.startswith("cuda") else "TORCH"

    def is_array(self,x): return self.torch.is_tensor(x)
    def ndim(self,x):
        if self.torch.is_tensor(x): return x.ndim
        return np.asarray(x).ndim
    def scalar(self,x):
        if self.torch.is_tensor(x):
            if x.ndim!=0: raise BackendError("Expected scalar.")
            return x.item()
        a=np.asarray(x)
        if a.ndim!=0: raise BackendError("Expected scalar.")
        return a.item()
    def clone(self,x): return x.clone() if self.torch.is_tensor(x) else x
    def to_numpy(self,x):
        if self.torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)
    def array(self,x,dtype=None):
        if self.torch.is_tensor(x):
            return x.to(self.device)
        tdtype=self.torch.float64 if dtype in (float,np.float64,"float64") else None
        return self.torch.as_tensor(x,dtype=tdtype,device=self.device)
    def _float(self,a):
        if self.torch.is_tensor(a):
            return a.to(device=self.device,dtype=self.torch.float64)
        return self.torch.as_tensor(a,dtype=self.torch.float64,device=self.device)
    def _pad(self,a,mode):
        torch=self.torch
        import torch.nn.functional as F
        a=self._float(a)
        if a.ndim!=3: raise BackendError("Padding requires a 3D FIELD.")
        x=a.unsqueeze(0).unsqueeze(0)
        if mode=="ZERO":
            y=F.pad(x,(1,1,1,1,1,1),mode="constant",value=0.0)
        elif mode=="WRAP":
            y=F.pad(x,(1,1,1,1,1,1),mode="circular")
        elif mode=="EDGE":
            y=F.pad(x,(1,1,1,1,1,1),mode="replicate")
        else:
            raise BackendError(f"Unknown boundary mode: {mode}")
        return y[0,0]
    def _acc(self,a,dirs,mode):
        torch=self.torch
        a=self._float(a)
        if a.ndim!=3: raise BackendError("Neighborhood primitives require a 3D FIELD.")
        p=self._pad(a,mode); nx,ny,nz=a.shape
        out=torch.zeros_like(a,dtype=torch.float64,device=self.device)
        for dx,dy,dz in dirs:
            out=out+p[1+dx:1+dx+nx,1+dy:1+dy+ny,1+dz:1+dz+nz]
        return out
    def ACC6(self,a,mode): return self._acc(a,DIR6,mode)
    def ACC26(self,a,mode): return self._acc(a,DIR26,mode)
    def AVG6(self,a,mode): return self.ACC6(a,mode)/6.0
    def AVG26(self,a,mode): return self.ACC26(a,mode)/26.0
    def LAPLACE6(self,a,mode): return self.ACC6(a,mode)-6.0*self._float(a)
    def derivative(self,a,axis,h,mode):
        a=self._float(a)
        if a.ndim!=3: raise BackendError("DX/DY/DZ require a 3D FIELD.")
        h=float(h)
        if h<=0: raise BackendError("Grid spacing must be positive.")
        p=self._pad(a,mode); nx,ny,nz=a.shape
        if axis==0:
            return (p[2:2+nx,1:1+ny,1:1+nz]-p[0:nx,1:1+ny,1:1+nz])/(2*h)
        if axis==1:
            return (p[1:1+nx,2:2+ny,1:1+nz]-p[1:1+nx,0:ny,1:1+nz])/(2*h)
        return (p[1:1+nx,1:1+ny,2:2+nz]-p[1:1+nx,1:1+ny,0:nz])/(2*h)
    def LAPLACE(self,a,hx,hy,hz,mode):
        a=self._float(a)
        if a.ndim!=3: raise BackendError("LAPLACE requires a 3D FIELD.")
        hx,hy,hz=float(hx),float(hy),float(hz)
        if min(hx,hy,hz)<=0: raise BackendError("Grid spacing must be positive.")
        p=self._pad(a,mode); nx,ny,nz=a.shape
        c=p[1:1+nx,1:1+ny,1:1+nz]
        d2x=(p[2:2+nx,1:1+ny,1:1+nz]-2*c+p[0:nx,1:1+ny,1:1+nz])/(hx*hx)
        d2y=(p[1:1+nx,2:2+ny,1:1+nz]-2*c+p[1:1+nx,0:ny,1:1+nz])/(hy*hy)
        d2z=(p[1:1+nx,1:1+ny,2:2+nz]-2*c+p[1:1+nx,1:1+ny,0:nz])/(hz*hz)
        return d2x+d2y+d2z
    def GRAD(self,a,mode):
        return self.torch.stack((self.derivative(a,0,1,mode),
                                 self.derivative(a,1,1,mode),
                                 self.derivative(a,2,1,mode)),dim=-1)
    def SCALE(self,a,k):
        a=self._float(a); k=int(k)
        if k<=0 or any(int(n)%k for n in a.shape):
            raise BackendError("SCALE requires a positive factor dividing every dimension.")
        x,y,z=map(int,a.shape)
        return a.reshape(x//k,k,y//k,k,z//k,k).mean(dim=(1,3,5))
    def SUM(self,a): return self._float(a).sum()
    def MEAN(self,a): return self._float(a).mean()
    def MIN(self,a): return self._float(a).min()
    def MAX(self,a): return self._float(a).max()
    def ABS(self,a): return self.torch.abs(self._float(a))
    def SQRT(self,a): return self.torch.sqrt(self._float(a))
    def NORM(self,a): return float(self.torch.linalg.vector_norm(self._float(a)).item())
    def CLIP(self,a,lo,hi): return self.torch.clamp(self._float(a),float(lo),float(hi))
    def ZEROS(self,*shape): return self.torch.zeros(tuple(int(x) for x in shape),dtype=self.torch.float64,device=self.device)
    def ONES(self,*shape): return self.torch.ones(tuple(int(x) for x in shape),dtype=self.torch.float64,device=self.device)
    def FULL(self,value,*shape): return self.torch.full(tuple(int(x) for x in shape),float(value),dtype=self.torch.float64,device=self.device)
    def LINSPACE(self,a,b,n): return self.torch.linspace(float(a),float(b),int(n),dtype=self.torch.float64,device=self.device)
    def RANGE(self,a,b=None,step=1):
        start,end=(0,float(a)) if b is None else (float(a),float(b))
        return self.torch.arange(start,end,float(step),dtype=self.torch.float64,device=self.device)

def create_backend(name):
    name=str(name).upper()
    if name=="NUMPY":
        return NumPyBackend()
    if name=="TORCH":
        return TorchBackend("cpu")
    if name=="GPU":
        return TorchBackend("cuda")
    raise BackendError(f"Unknown backend: {name}")

def backend_status():
    status={"NUMPY":{"available":True,"device":"cpu"}}
    try:
        import torch
        status["TORCH"]={"available":True,"version":torch.__version__,"device":"cpu"}
        status["GPU"]={
            "available":bool(torch.cuda.is_available()),
            "torch_cuda_available":bool(torch.cuda.is_available()),
            "device_count":int(torch.cuda.device_count()),
            "device_name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as e:
        status["TORCH"]={"available":False,"error":repr(e)}
        status["GPU"]={"available":False,"error":repr(e)}
    return status
