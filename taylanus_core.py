"""
TAYLANUS v3.0 Stable Research Core

Research implementation consolidating the validated v2.16 sparse-subface
and v2.22 streaming-local-modal branches under the v2.23 unified compiler.

This is a research CFD prototype. It is not a production solver and does
not claim equivalence or superiority to ANSYS or other production CFD.
The generated GPU backend has NOT been executed or validated.
"""
from __future__ import annotations

import numpy as np, pandas as pd, time, math, json
from pathlib import Path
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import factorized
from numba import njit

# ============================================================
# TAYLANUS v2.22 — Streaming / Local Modal Compiler
# Standalone reconstruction after v2.21 stress test.
#
# Goal:
#   - NO global modal P/R matrices
#   - NO global modal Q/G/L composed matrices
#   - local per-geometric-FACE modal basis only
#   - retain scalable v2.16 sparse subface operator as execution substrate
#   - pressure projection operates only on geometric FACE mean modes
# ============================================================

L = 2*np.pi
N0 = 8
nu = 1/300
MAXD, MAXL, MAXA = 4, 13, 8

def localized_velocity(x,y,z):
    x=np.asarray(x); y=np.asarray(y); z=np.asarray(z)
    out=np.zeros(np.broadcast(x,y,z).shape+(3,),dtype=float)
    vortices=[
        (+0.22,5.0,1.6,2.6,2.8),
        (-0.11,7.0,4.7,1.7,4.1),
    ]
    for A,k,x0,y0,z0 in vortices:
        e=np.exp(k*(np.cos(x-x0)+np.cos(y-y0)+np.cos(z-z0)-3.0))
        out[...,0] += -A*k*np.sin(y-y0)*e
        out[...,1] += +A*k*np.sin(x-x0)*e
    return out

def make_context(NF):
    max_level=int(round(math.log2(NF/N0)))
    hf=L/NF
    return NF,max_level,hf,hf*hf

def h_level(level):
    return L/(N0*(2**level))

def center_of(k):
    lev,ix,iy,iz=k
    h=h_level(lev)
    return np.array([(ix+.5)*h,(iy+.5)*h,(iz+.5)*h],float)

def vol_level(level):
    return h_level(level)**3

def local_indicator_cell(key):
    # Analytic local variation proxy sampled at child centers.
    lev,ix,iy,iz=key
    if lev>=CURRENT_MAX_LEVEL:
        return 0.0
    h=h_level(lev)
    c=center_of(key)
    vals=[]
    for dx in (-.25,.25):
        for dy in (-.25,.25):
            for dz in (-.25,.25):
                p=c+h*np.array([dx,dy,dz])
                vals.append(localized_velocity(p[0],p[1],p[2]))
    vals=np.asarray(vals)
    return float(np.max(np.linalg.norm(vals-vals.mean(axis=0),axis=1)))

def build_initial_topology(NF, target_leaves=(2500,6000)):
    global CURRENT_NF, CURRENT_MAX_LEVEL, CURRENT_HF, CURRENT_FACE_AREA
    CURRENT_NF,CURRENT_MAX_LEVEL,CURRENT_HF,CURRENT_FACE_AREA=make_context(NF)

    leaves={(0,i,j,k) for i in range(N0) for j in range(N0) for k in range(N0)}
    # Level-specific thresholds: concentrate refinement around localized vortices.
    base_thresholds={32:[0.020,0.012],64:[0.020,0.012,0.0065]}
    ths=base_thresholds.get(NF,[0.020]*(CURRENT_MAX_LEVEL))
    for lev in range(CURRENT_MAX_LEVEL):
        candidates=[]
        for key in list(leaves):
            if key[0]!=lev: continue
            score=local_indicator_cell(key)
            if score>ths[min(lev,len(ths)-1)]:
                candidates.append((score,key))
        # cap only to avoid pathological global growth
        candidates.sort(reverse=True)
        cap=max(1,int(.28*len(leaves)))
        for _,key in candidates[:cap]:
            if key not in leaves: continue
            leaves.remove(key)
            l,ix,iy,iz=key
            for dx in (0,1):
                for dy in (0,1):
                    for dz in (0,1):
                        leaves.add((l+1,2*ix+dx,2*iy+dy,2*iz+dz))
    return leaves

class Geometry:
    def __init__(self,NF,keys):
        self.NF=NF
        self.max_level=int(round(math.log2(NF/N0)))
        self.hf=L/NF
        self.face_area=self.hf**2
        self.keys=list(keys)
        self.levels=np.array([k[0] for k in self.keys],dtype=np.int16)
        self.centers=np.array([center_of(k) for k in self.keys],float)
        self.volumes=np.array([vol_level(k[0]) for k in self.keys],float)

        owner=np.empty((NF,NF,NF),dtype=np.int32)
        for idx,k in enumerate(self.keys):
            lev,ix,iy,iz=k
            span=2**(self.max_level-lev)
            owner[ix*span:(ix+1)*span,iy*span:(iy+1)*span,iz*span:(iz+1)*span]=idx
        self.owner=owner

        self.faces=[]
        for axis in range(3):
            A=owner
            B=np.roll(owner,-1,axis=axis)
            mask=A!=B
            coords=np.array(np.nonzero(mask)).T.astype(np.int16)
            ia=A[mask].astype(np.int32)
            ib=B[mask].astype(np.int32)
            d=self.centers[ib,axis]-self.centers[ia,axis]
            d=(d+L/2)%L-L/2
            dist=np.maximum(np.abs(d),1e-14)
            self.faces.append({"axis":axis,"coords":coords,"ia":ia,"ib":ib,"dist":dist})

        # subface pressure projection
        rows=[]; cols=[]; data=[]
        for f in self.faces:
            g=self.face_area/f["dist"]
            rows.extend([f["ia"],f["ia"],f["ib"],f["ib"]])
            cols.extend([f["ia"],f["ib"],f["ib"],f["ia"]])
            data.extend([-g,+g,-g,+g])
        rows=np.concatenate(rows); cols=np.concatenate(cols); data=np.concatenate(data)
        A=coo_matrix((data,(rows,cols)),shape=(len(self.keys),len(self.keys))).tolil()
        A[0,:]=0; A[:,0]=0; A[0,0]=1
        self.subface_pressure_solve=factorized(A.tocsc())

def analytic_face_state(geom):
    face=[]
    for f in geom.faces:
        a=f["axis"]; C=f["coords"].astype(float)
        pos=(C+.5)*geom.hf
        pos[:,a]=(C[:,a]+1.0)*geom.hf
        V=localized_velocity(pos[:,0],pos[:,1],pos[:,2])
        face.append(V[:,a].astype(float))
    return face

def project_subface(face,geom,dt):
    divnum=np.zeros(len(geom.keys))
    for u,f in zip(face,geom.faces):
        flux=u*geom.face_area
        np.add.at(divnum,f["ia"],+flux)
        np.add.at(divnum,f["ib"],-flux)
    rhs=divnum/dt; rhs[0]=0
    p=geom.subface_pressure_solve(rhs)
    out=[]
    for u,f in zip(face,geom.faces):
        gp=(p[f["ib"]]-p[f["ia"]])/f["dist"]
        out.append(u-dt*gp)
    return out

def subface_divergence(face,geom):
    divnum=np.zeros(len(geom.keys))
    for u,f in zip(face,geom.faces):
        flux=u*geom.face_area
        np.add.at(divnum,f["ia"],+flux)
        np.add.at(divnum,f["ib"],-flux)
    div=divnum/geom.volumes
    rms=float(np.sqrt(np.sum(div*div*geom.volumes)/np.sum(geom.volumes)))
    return rms,float(np.max(np.abs(div)))

# ---------- v2.16 sparse line compiler ----------
def build_sparse_lines(geom,comp):
    C=geom.faces[comp]["coords"].astype(np.int32)
    tang=[a for a in range(3) if a!=comp]
    line_key=C[:,tang[0]]*geom.NF+C[:,tang[1]]
    pos=C[:,comp]
    dof=np.arange(len(C),dtype=np.int32)
    order=np.lexsort((pos,line_key))
    ks=line_key[order].astype(np.int32)
    ps=pos[order].astype(np.int16)
    ds=dof[order].astype(np.int32)
    uniq,first=np.unique(ks,return_index=True)
    offsets=np.empty(len(uniq)+1,dtype=np.int32)
    offsets[:-1]=first.astype(np.int32); offsets[-1]=len(ks)
    return (uniq.astype(np.int32),offsets,ps,ds,np.array(tang,dtype=np.int8))

@njit
def lb(arr,v):
    lo=0; hi=arr.shape[0]
    while lo<hi:
        m=(lo+hi)//2
        if arr[m]<v: lo=m+1
        else: hi=m
    return lo

@njit
def support_sparse(q0,q1,q2,comp,t0,t1,keys,offs,pos,dofs,N):
    q0%=N; q1%=N; q2%=N
    a0=q0 if t0==0 else (q1 if t0==1 else q2)
    a1=q0 if t1==0 else (q1 if t1==1 else q2)
    key=a0*N+a1
    li=lb(keys,key)
    if li>=keys.shape[0] or keys[li]!=key:
        return -1,-1,1.0,0.0
    s=offs[li]; e=offs[li+1]
    qp=q0 if comp==0 else (q1 if comp==1 else q2)
    lo=s; hi=e
    while lo<hi:
        m=(lo+hi)//2
        if pos[m]<qp: lo=m+1
        else: hi=m
    if lo<e and pos[lo]==qp:
        d=dofs[lo]; return d,d,1.0,0.0
    ni=lo if lo<e else s
    pi=lo-1 if lo>s else e-1
    p0=int(pos[pi]); p1=int(pos[ni])
    span=(p1-p0)%N; ss=(qp-p0)%N
    if span==0:
        d=dofs[pi]; return d,d,1.0,0.0
    a=ss/span
    return int(dofs[pi]),int(dofs[ni]),1-a,a

@njit
def addterm(ids,ws,c,sid,w,maxn):
    if sid<0 or abs(w)<1e-15: return c
    for j in range(c):
        if ids[j]==sid:
            ws[j]+=w; return c
    if c<maxn:
        ids[c]=sid; ws[c]=w; return c+1
    return c

@njit
def addsupport(ids,ws,c,q0,q1,q2,comp,t0,t1,keys,offs,pos,dofs,factor,maxn,N):
    i0,i1,w0,w1=support_sparse(q0,q1,q2,comp,t0,t1,keys,offs,pos,dofs,N)
    c=addterm(ids,ws,c,i0,factor*w0,maxn)
    c=addterm(ids,ws,c,i1,factor*w1,maxn)
    return c

@njit
def compile_D_L(T,comp,t0,t1,keys,offs,pos,dofs,h,N):
    nt=T.shape[0]
    Didx=np.zeros((3,nt,MAXD),np.int32); Dw=np.zeros((3,nt,MAXD))
    Lidx=np.zeros((nt,MAXL),np.int32); Lw=np.zeros((nt,MAXL))
    ih=1/(2*h); ih2=1/(h*h)
    for r in range(nt):
        q0,q1,q2=int(T[r,0]),int(T[r,1]),int(T[r,2])
        li=np.zeros(MAXL,np.int32); lw=np.zeros(MAXL); lc=0
        lc=addsupport(li,lw,lc,q0,q1,q2,comp,t0,t1,keys,offs,pos,dofs,-6*ih2,MAXL,N)
        for ax in range(3):
            p0,p1,p2=q0,q1,q2; m0,m1,m2=q0,q1,q2
            if ax==0: p0=(q0+1)%N; m0=(q0-1)%N
            elif ax==1: p1=(q1+1)%N; m1=(q1-1)%N
            else: p2=(q2+1)%N; m2=(q2-1)%N
            di=np.zeros(MAXD,np.int32); dw=np.zeros(MAXD); dc=0
            dc=addsupport(di,dw,dc,p0,p1,p2,comp,t0,t1,keys,offs,pos,dofs,+ih,MAXD,N)
            dc=addsupport(di,dw,dc,m0,m1,m2,comp,t0,t1,keys,offs,pos,dofs,-ih,MAXD,N)
            Didx[ax,r]=di; Dw[ax,r]=dw
            lc=addsupport(li,lw,lc,p0,p1,p2,comp,t0,t1,keys,offs,pos,dofs,+ih2,MAXL,N)
            lc=addsupport(li,lw,lc,m0,m1,m2,comp,t0,t1,keys,offs,pos,dofs,+ih2,MAXL,N)
        Lidx[r]=li; Lw[r]=lw
    return Didx,Dw,Lidx,Lw

@njit
def compile_A(T,a,b,t0,t1,keys,offs,pos,dofs,N):
    nt=T.shape[0]
    Ai=np.zeros((nt,MAXA),np.int32); Aw=np.zeros((nt,MAXA))
    for r in range(nt):
        q0,q1,q2=int(T[r,0]),int(T[r,1]),int(T[r,2])
        ids=np.zeros(MAXA,np.int32); ws=np.zeros(MAXA); c=0
        if a==b:
            c=addsupport(ids,ws,c,q0,q1,q2,b,t0,t1,keys,offs,pos,dofs,1.0,MAXA,N)
        else:
            c=addsupport(ids,ws,c,q0,q1,q2,b,t0,t1,keys,offs,pos,dofs,.25,MAXA,N)
            a0,a1,a2=q0,q1,q2
            if a==0:a0=(a0-1)%N
            elif a==1:a1=(a1-1)%N
            else:a2=(a2-1)%N
            c=addsupport(ids,ws,c,a0,a1,a2,b,t0,t1,keys,offs,pos,dofs,.25,MAXA,N)
            b0,b1,b2=q0,q1,q2
            if b==0:b0=(b0+1)%N
            elif b==1:b1=(b1+1)%N
            else:b2=(b2+1)%N
            c=addsupport(ids,ws,c,b0,b1,b2,b,t0,t1,keys,offs,pos,dofs,.25,MAXA,N)
            c0,c1,c2=a0,a1,a2
            if b==0:c0=(c0+1)%N
            elif b==1:c1=(c1+1)%N
            else:c2=(c2+1)%N
            c=addsupport(ids,ws,c,c0,c1,c2,b,t0,t1,keys,offs,pos,dofs,.25,MAXA,N)
        Ai[r]=ids; Aw[r]=ws
    return Ai,Aw

def padded_to_csr(idx,wt,ncol):
    n=idx.shape[0]; rows0=np.arange(n,dtype=np.int32)
    mask=np.abs(wt)>0
    rows=np.broadcast_to(rows0[:,None],idx.shape)[mask]
    return csr_matrix((wt[mask],(rows,idx[mask])),shape=(n,ncol))

def compile_subface_ir(geom):
    lines=[build_sparse_lines(geom,c) for c in range(3)]
    out={"D":[[None]*3 for _ in range(3)],"LAP":[None]*3,"ADV":[[None]*3 for _ in range(3)],"lines":lines}
    for a in range(3):
        T=geom.faces[a]["coords"].astype(np.int32)
        lk,of,ps,ds,tang=lines[a]
        Di,Dw,Li,Lw=compile_D_L(T,a,int(tang[0]),int(tang[1]),lk,of,ps,ds,geom.hf,geom.NF)
        for ax in range(3): out["D"][a][ax]=padded_to_csr(Di[ax],Dw[ax],len(T))
        out["LAP"][a]=padded_to_csr(Li,Lw,len(T))
        for b in range(3):
            lk,of,ps,ds,tang=lines[b]
            Ai,Aw=compile_A(T,a,b,int(tang[0]),int(tang[1]),lk,of,ps,ds,geom.NF)
            out["ADV"][a][b]=padded_to_csr(Ai,Aw,len(geom.faces[b]["coords"]))
    return out

def subface_rhs(face,ir,nu):
    out=[]
    for a in range(3):
        d=[ir["D"][a][j].dot(face[a]) for j in range(3)]
        av=[ir["ADV"][a][b].dot(face[b]) for b in range(3)]
        lap=ir["LAP"][a].dot(face[a])
        out.append(-(av[0]*d[0]+av[1]*d[1]+av[2]*d[2])+nu*lap)
    return out

def add_state(a,b,scale):
    return [x+scale*y for x,y in zip(a,b)]
def rk2_combine(base,k1,k2,dt):
    return [u+.5*dt*(r1+r2) for u,r1,r2 in zip(base,k1,k2)]

# ---------- local modal maps: no global P/R ----------
def build_hier_faces(geom):
    H=[]
    for a,f in enumerate(geom.faces):
        ia,ib=f["ia"],f["ib"]
        order=np.lexsort((ib,ia))
        io,bo=ia[order],ib[order]
        starts=np.r_[0,1+np.flatnonzero((io[1:]!=io[:-1])|(bo[1:]!=bo[:-1]))]
        ends=np.r_[starts[1:],len(order)]
        sub_to_h=np.empty(len(ia),np.int32)
        rec=[]
        for g,(s,e) in enumerate(zip(starts,ends)):
            idx=order[s:e]
            sub_to_h[idx]=g
            rec.append((int(ia[idx[0]]),int(ib[idx[0]]),len(idx)*geom.face_area,float(np.mean(f["dist"][idx])),idx))
        H.append({"ia":np.array([r[0] for r in rec],np.int32),
                  "ib":np.array([r[1] for r in rec],np.int32),
                  "area":np.array([r[2] for r in rec],float),
                  "dist":np.array([r[3] for r in rec],float),
                  "sub_to_h":sub_to_h,
                  "groups":[r[4] for r in rec]})
    return H

def build_local_modal_maps(geom,H):
    maps=[]; total_modes=0
    for a,h in enumerate(H):
        coords=geom.faces[a]["coords"].astype(float)
        tang=[x for x in range(3) if x!=a]
        nsub=len(coords)
        ids=np.zeros((nsub,3),np.int32)
        Pw=np.zeros((nsub,3),float)
        Rw=np.zeros((nsub,3),float)
        mean_ids=[]
        face_modes=[]
        cursor=0
        for g,idx in enumerate(h["groups"]):
            pts=coords[idx][:,tang]
            c=pts.mean(axis=0); span=np.ptp(pts,axis=0)
            use1=span[0]>0; use2=span[1]>0
            nm=1+int(use1)+int(use2)
            B=np.ones((len(idx),nm))
            j=1
            if use1:
                B[:,j]=(pts[:,0]-c[0])/max(span[0]/2,1.0); j+=1
            if use2:
                B[:,j]=(pts[:,1]-c[1])/max(span[1]/2,1.0)
            pinv=np.linalg.pinv(B)
            for rr,sidx in enumerate(idx):
                for m in range(nm):
                    ids[sidx,m]=cursor+m
                    Pw[sidx,m]=B[rr,m]
                    Rw[sidx,m]=pinv[m,rr]
            mean_ids.append(cursor)
            face_modes.append(nm)
            cursor+=nm
        maps.append({"ids":ids,"P":Pw,"R":Rw,"mean_ids":np.array(mean_ids,np.int32),
                     "nmodes":cursor,"face_modes":np.array(face_modes,np.int8)})
        total_modes+=cursor
    return maps,total_modes

@njit
def modal_to_sub_jit(modal,ids,w):
    n=ids.shape[0]; out=np.zeros(n)
    for r in range(n):
        out[r]=w[r,0]*modal[ids[r,0]]+w[r,1]*modal[ids[r,1]]+w[r,2]*modal[ids[r,2]]
    return out

@njit
def sub_to_modal_jit(sub,ids,w,nm):
    out=np.zeros(nm)
    for r in range(ids.shape[0]):
        for j in range(3):
            out[ids[r,j]] += w[r,j]*sub[r]
    return out

def modal_to_sub(modal,maps):
    return [modal_to_sub_jit(modal[a],maps[a]["ids"],maps[a]["P"]) for a in range(3)]
def sub_to_modal(sub,maps):
    return [sub_to_modal_jit(sub[a],maps[a]["ids"],maps[a]["R"],maps[a]["nmodes"]) for a in range(3)]

def build_modal_pressure(geom,H,maps):
    ncell=len(geom.keys)
    offsets=np.cumsum([0]+[len(h["ia"]) for h in H]).astype(np.int32)
    rowsB=[]; colsB=[]; dataB=[]; rowsG=[]; colsG=[]; dataG=[]
    for a,h in enumerate(H):
        ids=np.arange(offsets[a],offsets[a+1],dtype=np.int32)
        rowsB.extend([h["ia"],h["ib"]]); colsB.extend([ids,ids]); dataB.extend([h["area"],-h["area"]])
        inv=1/h["dist"]
        rowsG.extend([ids,ids]); colsG.extend([h["ia"],h["ib"]]); dataG.extend([-inv,+inv])
    B=coo_matrix((np.concatenate(dataB),(np.concatenate(rowsB),np.concatenate(colsB))),
                 shape=(ncell,offsets[-1])).tocsr()
    G=coo_matrix((np.concatenate(dataG),(np.concatenate(rowsG),np.concatenate(colsG))),
                 shape=(offsets[-1],ncell)).tocsr()
    Lp=(B@G).tolil()
    Lp[0,:]=0; Lp[:,0]=0; Lp[0,0]=1
    solve=factorized(Lp.tocsc())
    return {"B":B,"G":G,"solve":solve,"offsets":offsets}

def modal_project(modal,geom,H,maps,pback,dt):
    means=np.concatenate([modal[a][maps[a]["mean_ids"]] for a in range(3)])
    divnum=pback["B"].dot(means)
    rhs=divnum/dt; rhs[0]=0
    p=pback["solve"](rhs)
    means=means-dt*pback["G"].dot(p)
    out=[x.copy() for x in modal]
    o=pback["offsets"]
    for a in range(3):
        out[a][maps[a]["mean_ids"]]=means[o[a]:o[a+1]]
    return out

def modal_div(modal,geom,H,maps,pback):
    means=np.concatenate([modal[a][maps[a]["mean_ids"]] for a in range(3)])
    div=pback["B"].dot(means)/geom.volumes
    return float(np.sqrt(np.sum(div*div*geom.volumes)/np.sum(geom.volumes)))

# ---------- dense validation only ----------
def fill_support_grid(face,geom,comp,ir):
    # reconstruct all component face samples via sparse line search
    grid=np.empty((geom.NF,geom.NF,geom.NF),float)
    lk,of,ps,ds,tang=ir["lines"][comp]
    for i in range(geom.NF):
        for j in range(geom.NF):
            for k in range(geom.NF):
                d0,d1,w0,w1=support_sparse(i,j,k,comp,int(tang[0]),int(tang[1]),lk,of,ps,ds,geom.NF)
                grid[i,j,k]=w0*face[comp][d0]+w1*face[comp][d1]
    return grid

def cell_velocity(face,geom,ir):
    U=np.empty((geom.NF,geom.NF,geom.NF,3),float)
    for a in range(3):
        G=fill_support_grid(face,geom,a,ir)
        U[...,a]=.5*(G+np.roll(G,1,axis=a))
    return U

# ---------- spectral reference ----------
def spectral_geometry(N):
    kv=np.fft.fftfreq(N,d=1/N)
    kx,ky,kz=np.meshgrid(kv,kv,kv,indexing="ij")
    K2=kx*kx+ky*ky+kz*kz; nz=K2>0; cut=N//3
    mask=(np.abs(kx)<=cut)&(np.abs(ky)<=cut)&(np.abs(kz)<=cut)
    return kx,ky,kz,K2,nz,mask

def spectral_project_hat(Uh,kx,ky,kz,K2,nz):
    dot=kx*Uh[...,0]+ky*Uh[...,1]+kz*Uh[...,2]
    out=Uh.copy()
    out[...,0][nz]-=kx[nz]*dot[nz]/K2[nz]
    out[...,1][nz]-=ky[nz]*dot[nz]/K2[nz]
    out[...,2][nz]-=kz[nz]*dot[nz]/K2[nz]
    return out

def spectral_rhs(U,nu):
    N=U.shape[0]; kx,ky,kz,K2,nz,mask=spectral_geometry(N)
    Uh=np.stack([np.fft.fftn(U[...,c]) for c in range(3)],axis=-1)
    Uh*=mask[...,None]
    ks=[kx,ky,kz]
    deriv=np.empty((3,3,N,N,N))
    for c in range(3):
        for a,k in enumerate(ks):
            deriv[c,a]=np.fft.ifftn(1j*k*Uh[...,c]).real
    adv=np.empty_like(U)
    for c in range(3):
        adv[...,c]=U[...,0]*deriv[c,0]+U[...,1]*deriv[c,1]+U[...,2]*deriv[c,2]
    Rh=np.stack([-np.fft.fftn(adv[...,c])-nu*K2*Uh[...,c] for c in range(3)],axis=-1)
    Rh*=mask[...,None]
    Rh=spectral_project_hat(Rh,kx,ky,kz,K2,nz)
    return np.stack([np.fft.ifftn(Rh[...,c]).real for c in range(3)],axis=-1)

def spectral_reference(N,dt,tend):
    x=(np.arange(N)+.5)*L/N; X,Y,Z=np.meshgrid(x,x,x,indexing="ij")
    U=localized_velocity(X,Y,Z)
    kx,ky,kz,K2,nz,mask=spectral_geometry(N)
    Uh=np.stack([np.fft.fftn(U[...,c]) for c in range(3)],axis=-1)*mask[...,None]
    Uh=spectral_project_hat(Uh,kx,ky,kz,K2,nz)
    U=np.stack([np.fft.ifftn(Uh[...,c]).real for c in range(3)],axis=-1)
    for _ in range(int(round(tend/dt))):
        k1=spectral_rhs(U,nu); U1=U+dt*k1; k2=spectral_rhs(U1,nu); U=U+.5*dt*(k1+k2)
    return U


import numpy as np, pandas as pd, time, json, hashlib
from dataclasses import dataclass
from pathlib import Path

# ============================================================
# TAYLANUS v2.23 — Unified Backend-Ready IR / Runtime
#
# One compiler:
#   topology -> shared geometry -> shared sparse subface IR
#            -> optional streaming-local modal layer
#            -> common pressure projection
#            -> backend planner
#
# Representations:
#   SUBFACE_SPARSE  : v2.16 scalable execution baseline
#   MODAL_STREAM    : v2.22 compressed-state branch
#
# No new physics in v2.23; this is consolidation + planning.
# ============================================================

def topology_signature(keys, NF):
    raw = f"NF={NF};" + ";".join(
        f"{a},{b},{c},{d}" for a,b,c,d in sorted(keys)
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

@dataclass
class CandidateMetrics:
    representation: str
    state_dofs: int
    state_bytes: int
    extra_setup_s: float
    step_median_s: float
    compression_vs_subface: float

class UnifiedCompileCache:
    def __init__(self):
        self.geometry = {}
        self.subface_ir = {}
        self.modal = {}
        self.hits = {"geometry":0,"subface":0,"modal":0}
        self.misses = {"geometry":0,"subface":0,"modal":0}

    def get_geometry(self, NF, keys):
        sig=topology_signature(keys,NF)
        if sig in self.geometry:
            self.hits["geometry"] += 1
            return sig,self.geometry[sig],0.0
        self.misses["geometry"] += 1
        s=time.perf_counter()
        geom=Geometry(NF,keys)
        dt=time.perf_counter()-s
        self.geometry[sig]=geom
        return sig,geom,dt

    def get_subface(self, sig, geom):
        if sig in self.subface_ir:
            self.hits["subface"] += 1
            return self.subface_ir[sig],0.0
        self.misses["subface"] += 1
        s=time.perf_counter()
        ir=compile_subface_ir(geom)
        dt=time.perf_counter()-s
        self.subface_ir[sig]=ir
        return ir,dt

    def get_modal(self, sig, geom):
        if sig in self.modal:
            self.hits["modal"] += 1
            return self.modal[sig],0.0
        self.misses["modal"] += 1
        s=time.perf_counter()
        H=build_hier_faces(geom)
        maps,nmodal=build_local_modal_maps(geom,H)
        pback=build_modal_pressure(geom,H,maps)
        dt=time.perf_counter()-s
        bundle={"H":H,"maps":maps,"nmodal":nmodal,"pback":pback}
        self.modal[sig]=bundle
        return bundle,dt

class UnifiedTaylanusIR:
    def __init__(self,NF,keys,cache=None):
        self.NF=NF
        self.keys=set(keys)
        self.cache=cache or UnifiedCompileCache()
        self.sig,self.geom,self.geometry_compile_s=self.cache.get_geometry(NF,self.keys)
        self.sub_ir,self.subface_compile_s=self.cache.get_subface(self.sig,self.geom)
        self.modal_bundle=None
        self.modal_setup_s=None

    def ensure_modal(self):
        if self.modal_bundle is None:
            self.modal_bundle,self.modal_setup_s=self.cache.get_modal(self.sig,self.geom)
        return self.modal_bundle

    def make_initial_subface(self,dt):
        return project_subface(analytic_face_state(self.geom),self.geom,dt)

    def subface_to_modal(self,face,dt):
        b=self.ensure_modal()
        modal=sub_to_modal(face,b["maps"])
        return modal_project(modal,self.geom,b["H"],b["maps"],b["pback"],dt)

    def modal_to_subface(self,modal):
        b=self.ensure_modal()
        return modal_to_sub(modal,b["maps"])

    def step_subface(self,state,dt):
        k1=subface_rhs(state,self.sub_ir,nu)
        pred=project_subface(add_state(state,k1,dt),self.geom,dt)
        k2=subface_rhs(pred,self.sub_ir,nu)
        return project_subface(rk2_combine(state,k1,k2,dt),self.geom,dt)

    def step_modal(self,state,dt):
        b=self.ensure_modal()
        k1=streaming_modal_rhs(state,b["maps"],self.sub_ir,nu)
        pred=modal_project(add_state(state,k1,dt),self.geom,b["H"],b["maps"],b["pback"],dt)
        k2=streaming_modal_rhs(pred,b["maps"],self.sub_ir,nu)
        return modal_project(rk2_combine(state,k1,k2,dt),self.geom,b["H"],b["maps"],b["pback"],dt)

    def benchmark_candidates(self,dt,reps=5):
        face=self.make_initial_subface(dt)
        sub_times=[]
        sstate=[x.copy() for x in face]
        # warm + repeats; use one-step-from-same-state copies to avoid drift
        self.step_subface([x.copy() for x in sstate],dt)
        for _ in range(reps):
            x=[v.copy() for v in sstate]
            s=time.perf_counter(); self.step_subface(x,dt); sub_times.append(time.perf_counter()-s)

        modal_bundle=self.ensure_modal()
        modal=self.subface_to_modal(face,dt)
        mod_times=[]
        self.step_modal([x.copy() for x in modal],dt)
        for _ in range(reps):
            x=[v.copy() for v in modal]
            s=time.perf_counter(); self.step_modal(x,dt); mod_times.append(time.perf_counter()-s)

        subd=sum(len(x) for x in face)
        modd=sum(len(x) for x in modal)

        return {
            "SUBFACE_SPARSE":CandidateMetrics(
                "SUBFACE_SPARSE",subd,subd*8,0.0,float(np.median(sub_times)),1.0
            ),
            "MODAL_STREAM":CandidateMetrics(
                "MODAL_STREAM",modd,modd*8,float(self.modal_setup_s or 0.0),
                float(np.median(mod_times)),subd/modd
            )
        }

    def plan(self,dt,nsteps,objective="AUTO",reps=5):
        metrics=self.benchmark_candidates(dt,reps=reps)
        if objective=="COMPACT":
            chosen=min(metrics.values(),key=lambda x:x.state_bytes)
        elif objective=="FAST":
            chosen=min(metrics.values(),key=lambda x:x.step_median_s)
        else:
            # Cold-run total estimate:
            # shared geometry/subface compile is common and omitted from comparison.
            def predicted(c):
                return c.extra_setup_s+nsteps*c.step_median_s
            chosen=min(metrics.values(),key=predicted)

        rows=[]
        for c in metrics.values():
            rows.append({
                "Representation":c.representation,
                "State DOFs":c.state_dofs,
                "State MiB":c.state_bytes/1024**2,
                "Compression vs subface":c.compression_vs_subface,
                "Extra modal setup s":c.extra_setup_s,
                "Median RK2 step ms":c.step_median_s*1000,
                "Predicted cold runtime contribution s":c.extra_setup_s+nsteps*c.step_median_s,
                "Selected":c.representation==chosen.representation,
                "Objective":objective
            })
        return chosen.representation,pd.DataFrame(rows),metrics



def select_representation(
    objective,
    subface_runtime_s,
    modal_runtime_s,
    subface_state_bytes,
    modal_state_bytes,
    tie_fraction=0.05,
    memory_budget_bytes=None,
):
    """Stable v3.0 representation policy."""
    objective = str(objective).upper()
    if objective == "COMPACT":
        return "MODAL_STREAM"
    if memory_budget_bytes is not None and subface_state_bytes > memory_budget_bytes:
        return "MODAL_STREAM"
    if modal_runtime_s < (1.0 - tie_fraction) * subface_runtime_s:
        return "MODAL_STREAM"
    return "SUBFACE_SPARSE"


# Validation helpers
@njit
def fill_grid_jit(facevals,comp,t0,t1,keys,offs,pos,dofs,N):
    G=np.empty((N,N,N),dtype=np.float64)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                d0,d1,w0,w1=support_sparse(i,j,k,comp,t0,t1,keys,offs,pos,dofs,N)
                G[i,j,k]=w0*facevals[d0]+w1*facevals[d1]
    return G

def cell_velocity_fast(face,geom,ir):
    U=np.empty((geom.NF,geom.NF,geom.NF,3),float)
    for a in range(3):
        lk,of,ps,ds,tang=ir["lines"][a]
        G=fill_grid_jit(face[a],a,int(tang[0]),int(tang[1]),lk,of,ps,ds,geom.NF)
        U[...,a]=.5*(G+np.roll(G,1,axis=a))
    return U

def rel_l2_face(face,geom,ir,ref):
    U=cell_velocity_fast(face,geom,ir)
    return float(np.linalg.norm(U-ref)/np.linalg.norm(ref))

def energy_face(face,geom,ir):
    U=cell_velocity_fast(face,geom,ir)
    return float(.5*np.mean(np.sum(U*U,axis=-1)))

def streaming_modal_rhs(modal,maps,ir,nu):
    sub=modal_to_sub(modal,maps)
    rhs=subface_rhs(sub,ir,nu)
    return sub_to_modal(rhs,maps)

def run_fixed_branch(NF,tend,dt,modal_branch=True):
    global CURRENT_NF,CURRENT_MAX_LEVEL,CURRENT_HF,CURRENT_FACE_AREA
    CURRENT_NF,CURRENT_MAX_LEVEL,CURRENT_HF,CURRENT_FACE_AREA=make_context(NF)
    keys=build_initial_topology(NF)
    geom=Geometry(NF,keys)

    t0=time.perf_counter()
    ir=compile_subface_ir(geom)
    compile_sub=time.perf_counter()-t0

    face=analytic_face_state(geom)
    face=project_subface(face,geom,dt)

    if not modal_branch:
        t0=time.perf_counter()
        for _ in range(int(round(tend/dt))):
            k1=subface_rhs(face,ir,nu)
            pred=add_state(face,k1,dt); pred=project_subface(pred,geom,dt)
            k2=subface_rhs(pred,ir,nu)
            face=rk2_combine(face,k1,k2,dt); face=project_subface(face,geom,dt)
        runtime=time.perf_counter()-t0
        return {"geom":geom,"ir":ir,"face":face,"runtime":runtime,"compile":compile_sub,
                "cells":len(keys),"dofs":sum(len(x) for x in face),"modal_dofs":None}

    H=build_hier_faces(geom)
    maps,nmodal=build_local_modal_maps(geom,H)
    modal=sub_to_modal(face,maps)
    pback=build_modal_pressure(geom,H,maps)
    modal=modal_project(modal,geom,H,maps,pback,dt)

    # local modal metadata only
    modal_meta_bytes=sum(
        m["ids"].nbytes+m["P"].nbytes+m["R"].nbytes+m["mean_ids"].nbytes+m["face_modes"].nbytes
        for m in maps
    )

    t0=time.perf_counter()
    for _ in range(int(round(tend/dt))):
        k1=streaming_modal_rhs(modal,maps,ir,nu)
        pred=add_state(modal,k1,dt); pred=modal_project(pred,geom,H,maps,pback,dt)
        k2=streaming_modal_rhs(pred,maps,ir,nu)
        modal=rk2_combine(modal,k1,k2,dt); modal=modal_project(modal,geom,H,maps,pback,dt)
    runtime=time.perf_counter()-t0
    face=modal_to_sub(modal,maps)
    return {"geom":geom,"ir":ir,"face":face,"modal":modal,"maps":maps,"H":H,"pback":pback,
            "runtime":runtime,"compile":compile_sub,"cells":len(keys),
            "dofs":sum(len(x) for x in face),"modal_dofs":nmodal,
            "modal_meta_bytes":modal_meta_bytes}

