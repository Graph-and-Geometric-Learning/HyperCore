"""CIFAR-10 paired training: PyTorch baseline vs fused Triton kernels."""
SEED   = 0
EPOCHS = 20
BATCH  = 128
LR     = 1e-3
WD     = 1e-4
WARM   = 1.0
LSM    = 0.1
CLIP   = 1.0
ROOT   = '..'
DATA   = './data'

import os, sys, json, math, time, statistics, urllib.request, tarfile, pickle
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hypercore_patches import _ON, load_hc, apply

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
MEAN = (0.4914,0.4822,0.4465); STD = (0.2470,0.2435,0.2616)

class _Raw(torch.utils.data.Dataset):
    def __init__(s,x,y,tr): s.x=torch.from_numpy(x); s.y=torch.from_numpy(y).long(); s.tr=tr; s.m=torch.tensor(MEAN).view(3,1,1); s.s=torch.tensor(STD).view(3,1,1)
    def __len__(s): return s.x.shape[0]
    def __getitem__(s,i):
        im=s.x[i].float()/255.0
        if s.tr:
            im=F.pad(im.unsqueeze(0),(4,4,4,4),mode='reflect')[0]
            a=torch.randint(0,9,(1,)).item(); b=torch.randint(0,9,(1,)).item(); im=im[:,a:a+32,b:b+32]
            if torch.rand(1).item()<0.5: im=torch.flip(im,dims=[2])
        return (im-s.m)/s.s, s.y[i]

def _man():
    os.makedirs(DATA,exist_ok=True); tgz=os.path.join(DATA,'cifar10.tar.gz')
    if not os.path.exists(tgz): urllib.request.urlretrieve('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',tgz)
    tf=tarfile.open(tgz)
    def rd(n): d=pickle.load(tf.extractfile(n),encoding='bytes'); return np.array(d[b'data']),list(d[b'labels'])
    tx=[]; ty=[]
    for k in range(1,6): a,b=rd(f'cifar-10-batches-py/data_batch_{k}'); tx.append(a); ty+=b
    ex,ey=rd('cifar-10-batches-py/test_batch')
    return _Raw(np.concatenate(tx).reshape(-1,3,32,32),np.array(ty),True), _Raw(ex.reshape(-1,3,32,32),np.array(ey),False)

def _ldrs(seed):
    try:
        import torchvision; from torchvision import transforms as T, datasets as D
        tr=T.Compose([T.RandomCrop(32,padding=4),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize(MEAN,STD)])
        te=T.Compose([T.ToTensor(),T.Normalize(MEAN,STD)])
        trs=D.CIFAR10(DATA,True,tr,download=True); tes=D.CIFAR10(DATA,False,te,download=True)
    except Exception: trs,tes=_man()
    g=torch.Generator(); g.manual_seed(seed)
    tl=torch.utils.data.DataLoader(trs,batch_size=BATCH,shuffle=True,num_workers=2,pin_memory=True,drop_last=True,generator=g)
    vl=torch.utils.data.DataLoader(tes,batch_size=256,shuffle=False,num_workers=2,pin_memory=True)
    return tl,vl

def _mk(img,seed,hc):
    Lz,LViT,LMA = hc['Lz'],hc['LViT'],hc['LMA']
    mi,mh,mo = Lz(),Lz(),Lz()
    torch.manual_seed(seed)
    m=LViT(mi,mh,mo,image_size=img,patch_size=4,num_layers=6,in_channel=3,hidden_channel=33,out_channel=10,mlp_hidden_size=33*4,num_heads=8,dropout=0.0).to(dev)
    for blk in m.encoder.blocks:
        a=blk.attention
        na=LMA(mh,a.in_channels,a.out_channels,a.num_heads,attention_type='linear_focused',trans_heads_concat=True).to(dev)
        oc=a.out_channels; nh=a.num_heads; r=oc-2
        na.v_map_mlp=nn.Linear(r,r,bias=True).to(dev); na.final_linear=nn.Linear(nh*r,nh*oc-1,bias=True).to(dev)
        blk.attention=na
    return m

def _tms(fn,it=20,wu=8):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(it):
        a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return statistics.median(ts)

hc=load_hc(ROOT); apply(hc); print("patched")

m=_mk(32,0,hc); x=torch.randn(8,3,32,32,device=dev); y=torch.randint(0,10,(8,),device=dev)
_ON[0]=False; m.zero_grad(); F.cross_entropy(m(x),y).backward()
g0={n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None}
_ON[0]=True; m.zero_grad(); F.cross_entropy(m(x),y).backward()
g1={n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None}
cm=set(g0)&set(g1); ge=max((g0[n]-g1[n]).abs().max().item() for n in cm)
_ON[0]=False
with torch.no_grad(): r0=m(x)
_ON[0]=True
with torch.no_grad(): r1=m(x)
print(f"correctness: out {(r0-r1).abs().max().item():.2e} | grads {ge:.2e}")
del m,x,y,g0,g1,r0,r1; torch.cuda.empty_cache()

torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED)
tl,vl=_ldrs(SEED); steps=len(tl)
print(f"seed={SEED} train={steps} test={len(vl)}")

mA=_mk(32,SEED,hc); mB=_mk(32,SEED,hc); mB.load_state_dict(mA.state_dict())
di=max((pa-pb).abs().max().item() for pa,pb in zip(mA.parameters(),mB.parameters()))
print(f"init max|A-B|={di:.2e}")

FZ=('w_y','norm_scale'); nf=0
for mdl in (mA,mB):
    for n,p in mdl.named_parameters():
        if any(k in n for k in FZ): p.requires_grad_(False); nf+=1
print(f"frozen {nf}")

try: from geoopt.optim import RiemannianAdam as OPT; on_='RiemannianAdam'
except Exception: OPT=torch.optim.AdamW; on_='AdamW'
oA=OPT([p for p in mA.parameters() if p.requires_grad],lr=LR,weight_decay=WD)
oB=OPT([p for p in mB.parameters() if p.requires_grad],lr=LR,weight_decay=WD)
print(f"opt={on_}")

tot=EPOCHS*steps; warm=int(WARM*steps)
def lr_at(s):
    if s<warm: return LR*(s+1)/max(1,warm)
    p=(s-warm)/max(1,tot-warm); return 0.5*LR*(1+math.cos(math.pi*min(1.0,p)))

crit=nn.CrossEntropyLoss(label_smoothing=LSM)
@torch.no_grad()
def ev(mdl,on):
    mdl.eval(); _ON[0]=on; c=t=0
    for xb,yb in vl:
        xb=xb.to(dev,non_blocking=True); yb=yb.to(dev,non_blocking=True)
        c+=(mdl(xb).argmax(1)==yb).sum().item(); t+=yb.numel()
    mdl.train(); return 100.0*c/t

xb,yb=next(iter(tl)); xb=xb.to(dev); yb=yb.to(dev)
_ON[0]=False; _=mA(xb); _ON[0]=True; _=mB(xb)

print("ep |  lr   | lossA  lossB |  accA   accB |  Δ  | t")
hist=[]; gs=0
for ep in range(1,EPOCHS+1):
    mA.train(); mB.train(); la=lb=0.0; t0=time.time()
    for xb,yb in tl:
        xb=xb.to(dev,non_blocking=True); yb=yb.to(dev,non_blocking=True); lr=lr_at(gs)
        for o in (oA,oB): o.param_groups[0]['lr']=lr
        _ON[0]=False; oA.zero_grad(set_to_none=True); lA=crit(mA(xb),yb); lA.backward()
        nn.utils.clip_grad_norm_([p for p in mA.parameters() if p.requires_grad],CLIP); oA.step()
        _ON[0]=True;  oB.zero_grad(set_to_none=True); lB=crit(mB(xb),yb); lB.backward()
        nn.utils.clip_grad_norm_([p for p in mB.parameters() if p.requires_grad],CLIP); oB.step()
        la+=lA.item(); lb+=lB.item(); gs+=1
    la/=steps; lb/=steps; aA=ev(mA,False); aB=ev(mB,True); dt=time.time()-t0
    hist.append([ep,la,lb,aA,aB])
    print(f"{ep:2d} | {lr:.1e} | {la:.3f}  {lb:.3f} | {aA:5.2f}  {aB:5.2f} | {aB-aA:+.2f} | {dt:.0f}")
    if not (math.isfinite(la) and math.isfinite(lb)): print("non-finite"); break

fA,fB=hist[-1][3],hist[-1][4]; bA=max(h[3] for h in hist); bB=max(h[4] for h in hist)
print(f"seed={SEED} base={fA:.2f} ker={fB:.2f} Δ={fB-fA:+.2f} (best {bA:.2f}/{bB:.2f})")

xb,yb=next(iter(tl)); xb=xb.to(dev); yb=yb.to(dev)
def stp(on): _ON[0]=on; mB.zero_grad(); F.cross_entropy(mB(xb),yb).backward()
toff=_tms(lambda:stp(False)); ton=_tms(lambda:stp(True))
print(f"step: base {toff:.1f}ms ker {ton:.1f}ms {toff/ton:.2f}x")

with open(f'parity_seed{SEED}.json','w') as f:
    json.dump({'seed':SEED,'hist':hist,'final':[fA,fB],'best':[bA,bB],'step_ms':{'base':toff,'ker':ton,'speedup':toff/ton}},f)
print(f"saved parity_seed{SEED}.json")
