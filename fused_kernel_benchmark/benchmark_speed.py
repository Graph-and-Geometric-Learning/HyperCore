"""Sustained training-loop speed benchmark: PyTorch baseline vs fused kernels on real CIFAR-10."""
EPOCHS  = 10
BATCH   = 128
LR      = 1e-3
WD      = 1e-4
ROOT    = '..'
DATA    = './data'
OUTPUT  = 'speed_bench.json'

import os, sys, json, math, time, statistics
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hypercore_patches import _ON, load_hc, apply

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
MEAN = (0.4914,0.4822,0.4465); STD = (0.2470,0.2435,0.2616)

import torchvision; from torchvision import transforms as T, datasets as D
tr = T.Compose([T.RandomCrop(32,padding=4),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize(MEAN,STD)])
ds = D.CIFAR10(DATA,True,tr,download=True)
ld = torch.utils.data.DataLoader(ds,batch_size=BATCH,shuffle=True,num_workers=2,pin_memory=True,drop_last=True)
steps = len(ld); print(f"data: {steps} steps/ep, {EPOCHS} ep")

hc = load_hc(ROOT); apply(hc); print("patched")

def _mk(img,seed,hc):
    Lz,LViT,LMA = hc['Lz'],hc['LViT'],hc['LMA']
    mi,mh,mo = Lz(),Lz(),Lz()
    torch.manual_seed(seed)
    m = LViT(mi,mh,mo,image_size=img,patch_size=4,num_layers=6,in_channel=3,hidden_channel=33,out_channel=10,mlp_hidden_size=33*4,num_heads=8,dropout=0.0).to(dev)
    for blk in m.encoder.blocks:
        a = blk.attention
        na = LMA(mh,a.in_channels,a.out_channels,a.num_heads,attention_type='linear_focused',trans_heads_concat=True).to(dev)
        oc = a.out_channels; nh = a.num_heads; r = oc-2
        na.v_map_mlp = nn.Linear(r,r,bias=True).to(dev); na.final_linear = nn.Linear(nh*r,nh*oc-1,bias=True).to(dev)
        blk.attention = na
    return m

try: from geoopt.optim import RiemannianAdam as OPT; on_ = 'RiemannianAdam'
except Exception: OPT = torch.optim.AdamW; on_ = 'AdamW'
crit = nn.CrossEntropyLoss(label_smoothing=0.1)

def run(on):
    m = _mk(32,0,hc)
    for n,p in m.named_parameters():
        if any(k in n for k in ('w_y','norm_scale')): p.requires_grad_(False)
    opt = OPT([p for p in m.parameters() if p.requires_grad],lr=LR,weight_decay=WD)
    m.train(); _ON[0] = on
    wu = 0
    for xb,yb in ld:
        xb = xb.to(dev,non_blocking=True); yb = yb.to(dev,non_blocking=True)
        opt.zero_grad(set_to_none=True); crit(m(xb),yb).backward()
        nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.0); opt.step()
        wu += 1
        if wu >= 10: break
    torch.cuda.synchronize()
    ep_med = []; all_ms = []; wall = 0.0
    for ep in range(EPOCHS):
        t0 = time.time(); e = []
        for xb,yb in ld:
            xb = xb.to(dev,non_blocking=True); yb = yb.to(dev,non_blocking=True)
            a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
            a.record()
            opt.zero_grad(set_to_none=True); crit(m(xb),yb).backward()
            nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.0); opt.step()
            b.record(); torch.cuda.synchronize()
            e.append(a.elapsed_time(b))
        torch.cuda.synchronize(); wall += time.time()-t0
        ep_med.append(statistics.median(e)); all_ms += e
    del m,opt; torch.cuda.empty_cache()
    return all_ms, wall, ep_med

print(f"opt={on_} B={BATCH}")
print("baseline..."); s_off,w_off,e_off = run(False)
print("kernels..."); s_on,w_on,e_on = run(True)

c_off = sum(s_off)/1000.0; c_on = sum(s_on)/1000.0
m_off = statistics.median(s_off); m_on = statistics.median(s_on)
half = EPOCHS//2
ss_off = statistics.median(e_off[half:]); ss_on = statistics.median(e_on[half:])

print(f"\nep | base   | ker   | x")
for i in range(EPOCHS):
    print(f"{i+1:2d} | {e_off[i]:6.1f} | {e_on[i]:5.1f} | {e_off[i]/e_on[i]:.3f}")
print(f"\ncompute  base {c_off:.1f}s ker {c_on:.1f}s {c_off/c_on:.3f}x")
print(f"median   base {m_off:.1f}ms ker {m_on:.1f}ms {m_off/m_on:.3f}x")
print(f"steady   base {ss_off:.1f}ms ker {ss_on:.1f}ms {ss_off/ss_on:.3f}x")
print(f"wall     base {w_off:.1f}s ker {w_on:.1f}s {w_off/w_on:.3f}x")

with open(OUTPUT,'w') as f:
    json.dump({'ep_med':{'base':e_off,'ker':e_on},
               'compute_s':{'base':c_off,'ker':c_on,'x':c_off/c_on},
               'median_ms':{'base':m_off,'ker':m_on,'x':m_off/m_on},
               'steady_ms':{'base':ss_off,'ker':ss_on,'x':ss_off/ss_on},
               'wall_s':{'base':w_off,'ker':w_on,'x':w_off/w_on},
               'epochs':EPOCHS,'batch':BATCH,'steps_per_epoch':steps},f)
print(f"saved {OUTPUT}")
