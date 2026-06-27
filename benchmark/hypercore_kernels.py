"""Fused Triton kernels for HyperCore: core_fused, lift_t, lres_f."""
import torch
import triton
import triton.language as tl

BLK = 32

@triton.jit
def _p1(KS,V,KTV,SP,ns,B,N,H,D:tl.constexpr,BD:tl.constexpr,BLK:tl.constexpr):
    pid=tl.program_id(0); blk=tl.program_id(1); b=pid//H; h=pid%H
    od=tl.arange(0,BD); mk=od<D; inv=1.0/(tl.abs(ns)+1e-6)
    rr=tl.arange(0,BD)[:,None]; cc=tl.arange(0,BD)[None,:]; m2=(rr<D)&(cc<D)
    n0=blk*BLK; ktv=tl.zeros((BD,BD),dtype=tl.float32); sp=tl.zeros((BD,),dtype=tl.float32)
    for i in range(BLK):
        n=n0+i
        if n<N:
            bs=((b*N+n)*H+h)*D
            ks=tl.load(KS+bs+od,mask=mk,other=0.0).to(tl.float32); vv=tl.load(V+bs+od,mask=mk,other=0.0).to(tl.float32)
            rk=tl.maximum(ks,0.0)+1e-6; rk=rk*inv; xp=rk*rk
            nx=tl.sqrt(tl.sum(rk*rk,axis=0)); nxp=tl.sqrt(tl.sum(xp*xp,axis=0)); pk=(nx/(nxp+1e-12))*xp
            sp+=pk; ktv+=pk[:,None]*vv[None,:]
    kb=((b*H+h)*D)*D
    tl.atomic_add(KTV+kb+rr*D+cc,ktv,mask=m2); tl.atomic_add(SP+(b*H+h)*D+od,sp,mask=mk)

@triton.jit
def _p2(QS,KTV,SP,OUT,ns,B,N,H,D:tl.constexpr,BD:tl.constexpr,BLK:tl.constexpr):
    pid=tl.program_id(0); blk=tl.program_id(1); b=pid//H; h=pid%H
    od=tl.arange(0,BD); mk=od<D; inv=1.0/(tl.abs(ns)+1e-6)
    rr=tl.arange(0,BD)[:,None]; cc=tl.arange(0,BD)[None,:]; m2=(rr<D)&(cc<D)
    kb=((b*H+h)*D)*D
    ktv=tl.load(KTV+kb+rr*D+cc,mask=m2,other=0.0).to(tl.float32); sp=tl.load(SP+(b*H+h)*D+od,mask=mk,other=0.0).to(tl.float32)
    n0=blk*BLK
    for i in range(BLK):
        n=n0+i
        if n<N:
            bs=((b*N+n)*H+h)*D
            qs=tl.load(QS+bs+od,mask=mk,other=0.0).to(tl.float32)
            rq=tl.maximum(qs,0.0)+1e-6; rq=rq*inv; xp=rq*rq
            nx=tl.sqrt(tl.sum(rq*rq,axis=0)); nxp=tl.sqrt(tl.sum(xp*xp,axis=0)); pq=(nx/(nxp+1e-12))*xp
            num=tl.sum(pq[:,None]*ktv,axis=0); den=tl.sum(pq*sp,axis=0)
            tl.store(OUT+bs+od,(num/(den+1e-6)).to(OUT.dtype.element_ty),mask=mk)

@triton.jit
def _phg(x,g,inv,BD:tl.constexpr):
    rel=tl.maximum(x,0.0)+1e-6; rq=rel*inv; xp=rq*rq
    nx=tl.sqrt(tl.sum(rq*rq,axis=0)); nxp=tl.sqrt(tl.sum(xp*xp,axis=0)); cf=nx/(nxp+1e-12)
    gdx=tl.sum(g*xp,axis=0); dc=(rq/(nx+1e-12)*nxp-nx*2.0*rq*xp/(nxp+1e-12))/(nxp*nxp+1e-12)
    return tl.where(x>0.0,(gdx*dc+cf*2.0*rq*g)*inv,0.0)

@triton.jit
def _bA(QS,KTV,SP,DO,GKTV,GSP,GPQ,ns,B,N,H,D:tl.constexpr,BD:tl.constexpr,BLK:tl.constexpr):
    pid=tl.program_id(0); blk=tl.program_id(1); b=pid//H; h=pid%H
    od=tl.arange(0,BD); mk=od<D; inv=1.0/(tl.abs(ns)+1e-6)
    rr=tl.arange(0,BD)[:,None]; cc=tl.arange(0,BD)[None,:]; m2=(rr<D)&(cc<D)
    kb=((b*H+h)*D)*D
    ktv=tl.load(KTV+kb+rr*D+cc,mask=m2,other=0.0).to(tl.float32); sp=tl.load(SP+(b*H+h)*D+od,mask=mk,other=0.0).to(tl.float32)
    gktv=tl.zeros((BD,BD),dtype=tl.float32); gsp=tl.zeros((BD,),dtype=tl.float32); n0=blk*BLK
    for i in range(BLK):
        n=n0+i
        if n<N:
            bs=((b*N+n)*H+h)*D
            qs=tl.load(QS+bs+od,mask=mk,other=0.0).to(tl.float32); g=tl.load(DO+bs+od,mask=mk,other=0.0).to(tl.float32)
            rq=tl.maximum(qs,0.0)+1e-6; rq=rq*inv; xp=rq*rq
            nx=tl.sqrt(tl.sum(rq*rq,axis=0)); nxp=tl.sqrt(tl.sum(xp*xp,axis=0)); pq=(nx/(nxp+1e-12))*xp
            num=tl.sum(pq[:,None]*ktv,axis=0); den=tl.sum(pq*sp,axis=0); de=den+1e-6
            gn=g/de; gd=-tl.sum(g*num,axis=0)/(de*de)
            gpq=tl.sum(gn[None,:]*ktv,axis=1)+gd*sp
            gktv+=pq[:,None]*gn[None,:]; gsp+=gd*pq
            tl.store(GPQ+bs+od,gpq.to(GPQ.dtype.element_ty),mask=mk)
    tl.atomic_add(GKTV+kb+rr*D+cc,gktv,mask=m2); tl.atomic_add(GSP+(b*H+h)*D+od,gsp,mask=mk)

@triton.jit
def _bB(QS,KS,V,KTV,GKTV,GSP,GPQ,GQS,GKS,GV,ns,B,N,H,D:tl.constexpr,BD:tl.constexpr,BLK:tl.constexpr):
    pid=tl.program_id(0); blk=tl.program_id(1); b=pid//H; h=pid%H
    od=tl.arange(0,BD); mk=od<D; inv=1.0/(tl.abs(ns)+1e-6)
    rr=tl.arange(0,BD)[:,None]; cc=tl.arange(0,BD)[None,:]; m2=(rr<D)&(cc<D)
    kb=((b*H+h)*D)*D
    gktv=tl.load(GKTV+kb+rr*D+cc,mask=m2,other=0.0).to(tl.float32); gsp=tl.load(GSP+(b*H+h)*D+od,mask=mk,other=0.0).to(tl.float32)
    n0=blk*BLK
    for i in range(BLK):
        j=n0+i
        if j<N:
            bs=((b*N+j)*H+h)*D
            ks=tl.load(KS+bs+od,mask=mk,other=0.0).to(tl.float32); vv=tl.load(V+bs+od,mask=mk,other=0.0).to(tl.float32)
            rk=tl.maximum(ks,0.0)+1e-6; rkn=rk*inv; xp=rkn*rkn
            nx=tl.sqrt(tl.sum(rkn*rkn,axis=0)); nxp=tl.sqrt(tl.sum(xp*xp,axis=0)); pk=(nx/(nxp+1e-12))*xp
            gv=tl.sum(pk[:,None]*gktv,axis=0); gpk=tl.sum(gktv*vv[None,:],axis=1)+gsp
            gks=_phg(ks,gpk,inv,BD)
            tl.store(GV+bs+od,gv.to(GV.dtype.element_ty),mask=mk); tl.store(GKS+bs+od,gks.to(GKS.dtype.element_ty),mask=mk)
            gpq=tl.load(GPQ+bs+od,mask=mk,other=0.0).to(tl.float32); qs=tl.load(QS+bs+od,mask=mk,other=0.0).to(tl.float32)
            gqs=_phg(qs,gpq,inv,BD); tl.store(GQS+bs+od,gqs.to(GQS.dtype.element_ty),mask=mk)

class _Core(torch.autograd.Function):
    @staticmethod
    def forward(ctx,qs,ks,v,ns):
        B,N,H,D=qs.shape; bd=triton.next_power_of_2(D)
        qs=qs.contiguous(); ks=ks.contiguous(); v=v.contiguous()
        ktv=torch.zeros(B,H,D,D,device=qs.device,dtype=torch.float32)
        sp=torch.zeros(B,H,D,device=qs.device,dtype=torch.float32)
        out=torch.empty_like(qs); nb=triton.cdiv(N,BLK)
        _p1[(B*H,nb)](ks,v,ktv,sp,float(ns),B,N,H,D,BD=bd,BLK=BLK,num_warps=4)
        _p2[(B*H,nb)](qs,ktv,sp,out,float(ns),B,N,H,D,BD=bd,BLK=BLK,num_warps=4)
        ctx.save_for_backward(qs,ks,v,ktv,sp); ctx.ns=float(ns); ctx.dims=(B,N,H,D)
        return out
    @staticmethod
    def backward(ctx,do):
        qs,ks,v,ktv,sp=ctx.saved_tensors; B,N,H,D=ctx.dims
        bd=triton.next_power_of_2(D); ns=ctx.ns; do=do.contiguous()
        gktv=torch.zeros(B,H,D,D,device=qs.device,dtype=torch.float32)
        gsp=torch.zeros(B,H,D,device=qs.device,dtype=torch.float32)
        gpq=torch.empty_like(qs); gqs=torch.empty_like(qs); gks=torch.empty_like(ks); gv=torch.empty_like(v)
        nb=triton.cdiv(N,BLK)
        _bA[(B*H,nb)](qs,ktv,sp,do,gktv,gsp,gpq,float(ns),B,N,H,D,BD=bd,BLK=BLK,num_warps=4)
        _bB[(B*H,nb)](qs,ks,v,ktv,gktv,gsp,gpq,gqs,gks,gv,float(ns),B,N,H,D,BD=bd,BLK=BLK,num_warps=4)
        return gqs,gks,gv,None

def core_fused(qs,ks,v,ns): return _Core.apply(qs,ks,v,ns)

@triton.jit
def _lf(X,OUT,c,eps,sc,M,D:tl.constexpr,BD:tl.constexpr):
    r=tl.program_id(0); od=tl.arange(0,BD); m=od<D
    x=tl.load(X+r*D+od,mask=m,other=0.0).to(tl.float32)
    ss=tl.sum(x*x,axis=0)+c; t=tl.sqrt(tl.maximum(ss,eps))*sc
    tl.store(OUT+r*(D+1),t); tl.store(OUT+r*(D+1)+1+od,(x*sc).to(OUT.dtype.element_ty),mask=m)

@triton.jit
def _lb(X,DO,DX,c,eps,sc,M,D:tl.constexpr,BD:tl.constexpr):
    r=tl.program_id(0); od=tl.arange(0,BD); m=od<D
    x=tl.load(X+r*D+od,mask=m,other=0.0).to(tl.float32)
    ss=tl.sum(x*x,axis=0)+c; tr=tl.sqrt(tl.maximum(ss,eps))
    dt=tl.load(DO+r*(D+1)).to(tl.float32); ds=tl.load(DO+r*(D+1)+1+od,mask=m,other=0.0).to(tl.float32)
    u=ss>eps; ddx=tl.where(u,sc*x/tr,0.0); dx=ds*sc+dt*ddx
    tl.store(DX+r*D+od,dx.to(DX.dtype.element_ty),mask=m)

class _Lift(torch.autograd.Function):
    @staticmethod
    def forward(ctx,xs,c,eps,sc):
        sh=xs.shape; D=sh[-1]; x2=xs.reshape(-1,D).contiguous(); M=x2.shape[0]
        out=torch.empty(M,D+1,device=xs.device,dtype=xs.dtype); bd=triton.next_power_of_2(D)
        _lf[(M,)](x2,out,float(c),float(eps),float(sc),M,D,BD=bd,num_warps=4)
        ctx.save_for_backward(x2); ctx.c=float(c); ctx.eps=float(eps); ctx.sc=float(sc); ctx.sh=sh
        return out.reshape(*sh[:-1],D+1)
    @staticmethod
    def backward(ctx,do):
        (x2,)=ctx.saved_tensors; D=x2.shape[-1]; M=x2.shape[0]
        do=do.reshape(-1,D+1).contiguous(); dx=torch.empty_like(x2); bd=triton.next_power_of_2(D)
        _lb[(M,)](x2,do,dx,ctx.c,ctx.eps,ctx.sc,M,D,BD=bd,num_warps=4)
        return dx.reshape(ctx.sh),None,None,None

def lift_t(xs,c,eps,sc): return _Lift.apply(xs,c,eps,float(sc))

@triton.jit
def _rf(X,Y,OUT,wy,sc,c,eps,M,D:tl.constexpr,BD:tl.constexpr):
    r=tl.program_id(0); od=tl.arange(0,BD); m=od<D
    xt=tl.load(X+r*(D+1)).to(tl.float32); yt=tl.load(Y+r*(D+1)).to(tl.float32)
    xs=tl.load(X+r*(D+1)+1+od,mask=m,other=0.0).to(tl.float32); ys=tl.load(Y+r*(D+1)+1+od,mask=m,other=0.0).to(tl.float32)
    at=xt+yt*wy; as_=xs+ys*wy
    inr=-at*at+tl.sum(as_*as_,axis=0); dn=tl.sqrt(tl.maximum(tl.abs(inr),eps)); k=tl.sqrt(c)/dn
    ps=k*as_; xsp=sc*ps; xt2=tl.sqrt(tl.maximum(tl.sum(xsp*xsp,axis=0)+c,eps))
    tl.store(OUT+r*(D+1),xt2); tl.store(OUT+r*(D+1)+1+od,xsp.to(OUT.dtype.element_ty),mask=m)

@triton.jit
def _rb(X,Y,DO,DX,DY,wy,sc,c,eps,M,D:tl.constexpr,BD:tl.constexpr):
    r=tl.program_id(0); od=tl.arange(0,BD); m=od<D
    xt=tl.load(X+r*(D+1)).to(tl.float32); yt=tl.load(Y+r*(D+1)).to(tl.float32)
    xs=tl.load(X+r*(D+1)+1+od,mask=m,other=0.0).to(tl.float32); ys=tl.load(Y+r*(D+1)+1+od,mask=m,other=0.0).to(tl.float32)
    at=xt+yt*wy; as_=xs+ys*wy
    ss=tl.sum(as_*as_,axis=0); inr=-at*at+ss; ai=tl.abs(inr)
    dn=tl.sqrt(tl.maximum(ai,eps)); rc=tl.sqrt(c); k=rc/dn
    ps=k*as_; xsp=sc*ps; sq=tl.sum(xsp*xsp,axis=0)+c; xt2=tl.sqrt(tl.maximum(sq,eps))
    gt=tl.load(DO+r*(D+1)).to(tl.float32); gs=tl.load(DO+r*(D+1)+1+od,mask=m,other=0.0).to(tl.float32)
    ut=sq>eps; dtdx=tl.where(ut,xsp/xt2,0.0); gxs=gs+gt*dtdx; gps=gxs*sc
    ui=ai>eps; dkd=-rc/(dn*dn); dd=tl.where(ui,tl.where(inr>=0,1.0,-1.0)/(2.0*dn),0.0); dkdi=dkd*dd
    gpa=tl.sum(gps*as_,axis=0); gi=gpa*dkdi; gas=gps*k+gi*(2.0*as_); gat=gi*(-2.0*at)
    tl.store(DX+r*(D+1),gat); tl.store(DX+r*(D+1)+1+od,gas.to(DX.dtype.element_ty),mask=m)
    tl.store(DY+r*(D+1),gat*wy); tl.store(DY+r*(D+1)+1+od,(gas*wy).to(DY.dtype.element_ty),mask=m)

class _LRes(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,y,wy,sc,c,eps):
        sh=x.shape; D=sh[-1]-1; x2=x.reshape(-1,D+1).contiguous(); y2=y.reshape(-1,D+1).contiguous(); M=x2.shape[0]
        out=torch.empty_like(x2); bd=triton.next_power_of_2(D)
        _rf[(M,)](x2,y2,out,float(wy),float(sc),float(c),float(eps),M,D,BD=bd,num_warps=4)
        ctx.save_for_backward(x2,y2); ctx.wy=float(wy); ctx.sc=float(sc); ctx.c=float(c); ctx.eps=float(eps); ctx.sh=sh
        return out.reshape(sh)
    @staticmethod
    def backward(ctx,do):
        x2,y2=ctx.saved_tensors; D=x2.shape[-1]-1; M=x2.shape[0]
        do=do.reshape(-1,D+1).contiguous(); dx=torch.empty_like(x2); dy=torch.empty_like(y2); bd=triton.next_power_of_2(D)
        _rb[(M,)](x2,y2,do,dx,dy,ctx.wy,ctx.sc,ctx.c,ctx.eps,M,D,BD=bd,num_warps=4)
        return dx.reshape(ctx.sh),dy.reshape(ctx.sh),None,None,None,None

def lres_f(x,y,wy,sc,c,eps): return _LRes.apply(x,y,float(wy),float(sc),float(c),float(eps))
