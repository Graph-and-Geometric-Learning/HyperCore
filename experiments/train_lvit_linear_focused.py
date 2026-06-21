"""
train_lvit_linear_focused.py

Reproducible setup: real HyperCore LViT with linear_focused attention + our v3
Triton phi kernel, trained on CIFAR-10.

NOTE: this bypasses torch_scatter via isolated imports (no wheel for torch
2.10+cu128). The runtime dim-patch (patch_dims / lf_fixed) is a WORKAROUND for
the known linear_focused dim bug (see report sec 8), not a proper fix. Model,
training, and profiling are all on the real LViT.

run: python train_lvit_linear_focused.py
"""
import sys, os, types, importlib, importlib.util, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

ROOT = os.environ.get('HYPERCORE_ROOT', '/kaggle/working/HyperCore')
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- isolated loader (skip torch_scatter chain) ----
def _stub(name, path):
    if name in sys.modules: return sys.modules[name]
    m = types.ModuleType(name); m.__path__ = [path]; m.__package__ = name
    sys.modules[name] = m; return m

def _load(name, fp, pkg):
    sp = importlib.util.spec_from_file_location(name, fp)
    m = importlib.util.module_from_spec(sp); m.__package__ = pkg
    sys.modules[name] = m; sp.loader.exec_module(m); return m

def _fill(pkg, mod):
    p = sys.modules[pkg]
    for s in dir(mod):
        if not s.startswith('_'): setattr(p, s, getattr(mod, s))

def load_hypercore():
    sp = importlib.util.spec_from_file_location(
        'hypercore.manifolds', f'{ROOT}/hypercore/manifolds/__init__.py',
        submodule_search_locations=[f'{ROOT}/hypercore/manifolds'])
    man = importlib.util.module_from_spec(sp); man.__package__ = 'hypercore.manifolds'
    _stub('hypercore', f'{ROOT}/hypercore')
    sys.modules['hypercore.manifolds'] = man; sp.loader.exec_module(man)
    _stub('hypercore.nn', f'{ROOT}/hypercore/nn')
    for s in ['linear', 'conv', 'attention']:
        _stub(f'hypercore.nn.{s}', f'{ROOT}/hypercore/nn/{s}')
    for nm, fp in [
        ('hypercore.nn.linear.lorentz_linear', 'nn/linear/lorentz_linear.py'),
        ('hypercore.nn.linear.lorentz_CLS', 'nn/linear/lorentz_CLS.py'),
        ('hypercore.nn.conv.conv_util_layers', 'nn/conv/conv_util_layers.py'),
        ('hypercore.nn.conv.lorentz_convolution', 'nn/conv/lorentz_convolution.py'),
        ('hypercore.nn.conv.lorentz_MLR', 'nn/conv/lorentz_MLR.py'),
    ]:
        pkg = nm.rsplit('.', 1)[0]
        _fill(pkg, _load(nm, f'{ROOT}/hypercore/{fp}', pkg))
    for m in ['hypercore.nn.linear.lorentz_linear', 'hypercore.nn.linear.lorentz_CLS',
              'hypercore.nn.conv.conv_util_layers', 'hypercore.nn.conv.lorentz_MLR']:
        _fill('hypercore.nn', sys.modules[m])
    _patch_phi()  # inject v3 phi before loading attention
    lf3 = _load('hypercore.nn.attention.linear_focus_attention_v3',
                f'{ROOT}/hypercore/nn/attention/linear_focus_attention_v3.py',
                'hypercore.nn.attention')
    setattr(sys.modules['hypercore.nn.attention'], 'linear_focus_attention_v3', lf3)
    for fn in ['flash_lorentz_attention','flash_lorentz_attention_v2',
               'flash_lorentz_dispatch','patch_embedding']:
        try:
            m = _load(f'hypercore.nn.attention.{fn}',
                      f'{ROOT}/hypercore/nn/attention/{fn}.py', 'hypercore.nn.attention')
            _fill('hypercore.nn.attention', m); _fill('hypercore.nn', m)
        except Exception as e:
            print(f'skip {fn}: {str(e)[:60]}')
    fc = _load('hypercore.nn.attention.lorentz_former_conv',
               f'{ROOT}/hypercore/nn/attention/lorentz_former_conv.py', 'hypercore.nn.attention')
    _fill('hypercore.nn.attention', fc); _fill('hypercore.nn', fc)
    _stub('hypercore.models', f'{ROOT}/hypercore/models')
    LViT = _load('hypercore.models.LViT', f'{ROOT}/hypercore/models/LViT.py', 'hypercore.models').LViT
    return LViT, sys.modules['hypercore.manifolds'].Lorentz, lf3, fc

def _patch_phi():
    fp = f'{ROOT}/hypercore/nn/attention/lorentz_former_conv.py'
    s = open(fp).read()
    if '_phi_v3' in s: return
    s = s.replace('from .flash_lorentz_dispatch import flash_attention_core',
                  'from .flash_lorentz_dispatch import flash_attention_core\n'
                  'from .linear_focus_attention_v3 import phi as _phi_v3')
    old = (
        "phi_qs = (F.relu(qs) + 1e-6) / (self.norm_scale.abs() + 1e-6)  # [B, N, H, D]\n"
        "phi_ks = (F.relu(ks) + 1e-6) / (self.norm_scale.abs() + 1e-6)  # [B, N, H, D]\n\n"
        "phi_qs = self.fp(phi_qs, p=self.power_k)  # [B, N, H, D]\n"
        "phi_ks = self.fp(phi_ks, p=self.power_k)  # [B, N, H, D]"
    )
    new = (
        "phi_qs = _phi_v3(qs, self.norm_scale, p=self.power_k)  # [B, N, H, D]\n"
        "phi_ks = _phi_v3(ks, self.norm_scale, p=self.power_k)  # [B, N, H, D]"
    )
    s = s.replace(old, new); open(fp, 'w').write(s)

# ---- linear_focused swap + dim workaround ----
def swap_lf(model, man):
    LMA = sys.modules['hypercore.nn'].LorentzMultiheadAttention
    for blk in model.encoder.blocks:
        a = blk.attention
        blk.attention = LMA(man, a.in_channels, a.out_channels, a.num_heads, attention_type='linear_focused', trans_heads_concat=True).to(dev)
    return model

def _lf_fixed(self, hq, hk, hv, output_attentions=False, mask=None):
    qs, ks, v = hq[..., 1:], hk[..., 1:], hv[..., 1:]
    pq = _FC._phi_v3(qs, self.norm_scale, p=self.power_k)
    pk = _FC._phi_v3(ks, self.norm_scale, p=self.power_k)
    ktv = torch.einsum('bnhm,bnhd->bhmd', pk, v)
    num = torch.einsum('bnhm,bhmd->bnhd', pq, ktv)
    den = torch.einsum('bnhd,bhd->bnh', pq, torch.einsum('bnhd->bhd', pk)).unsqueeze(-1)
    a = num / (den + 1e-6) + self.v_map_mlp(v)
    B, N, H, D = a.shape
    a = self.final_linear(a.reshape(B, N, H * D)) if self.trans_heads_concat else a.mean(1)
    t = (a.pow(2).sum(-1, keepdim=True) + self.manifold.c).sqrt()
    return torch.cat([t, a], dim=-1)

def patch_dims(model):
    for blk in model.encoder.blocks:
        a = blk.attention
        oc, nh = a.out_channels, a.num_heads
        r = oc - 2
        a.v_map_mlp = nn.Linear(r, r, bias=True).to(dev)
        a.final_linear = nn.Linear(nh * r, nh * oc - 1, bias=True).to(dev)
        a.linear_focus_attention = types.MethodType(_lf_fixed, a)
    return model

# ---- model build ----
def build(LViT, man_in, man_h, man_out, use_kernel):
    torch.manual_seed(0)
    m = LViT(
        man_in, man_h, man_out, image_size=32, patch_size=4, num_layers=6,
        in_channel=3, hidden_channel=33, out_channel=10, mlp_hidden_size=33 * 4,
        num_heads=8, dropout=0.1
    ).to(dev)
    m = patch_dims(swap_lf(m, man_h))
    _LF3._HAS_TRITON = use_kernel
    return m

# ---- data ----
def loaders(bs=128):
    tt = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    tv = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    root = os.environ.get('DATA_ROOT', '/kaggle/working/cifar')
    tr = datasets.CIFAR10(root, train=True, download=True, transform=tt)
    va = datasets.CIFAR10(root, train=False, download=True, transform=tv)
    return (DataLoader(tr, bs, shuffle=True, num_workers=4, pin_memory=True, drop_last=True),
            DataLoader(va, 256, shuffle=False, num_workers=4, pin_memory=True))

@torch.no_grad()
def evaluate(m, vl, uk):
    _LF3._HAS_TRITON = uk; m.eval()
    c = n = 0
    for x, y in vl:
        x, y = x.to(dev), y.to(dev)
        c += (m(x).argmax(-1) == y).sum().item(); n += x.size(0)
    return c/n

def train(LViT, mans, tl, vl, uk, epochs=20, lr=1e-3, warmup=2):
    m = build(LViT, *mans, uk)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.05)
    fn = lambda e: (e + 1) / warmup if e < warmup else 0.5 * (1 + math.cos(math.pi * (e - warmup) / (epochs - warmup)))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, fn)
    hist = []
    for ep in range(epochs):
        _LF3._HAS_TRITON = uk; m.train()
        el = c = n = 0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            lo = m(x); loss = F.cross_entropy(lo, y)
            opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item() * x.size(0); c += (lo.argmax(-1) == y).sum().item(); n += x.size(0)
        sch.step(); va = evaluate(m, vl, uk); hist.append((el / n, c / n, va))
        print(f'Epoch{ep + 1}: Loss={el/n:.3f} Acc={c/n:.3f} Val={va:.3f}')
    return m, hist

_LF3 = None
_FC = None

if __name__ == '__main__':
    LViT, Lorentz, _LF3, _FC = load_hypercore()
    mans = (Lorentz(), Lorentz(), Lorentz())
    tl, vl = loaders()
    _, hk = train(LViT, mans, tl, vl, True)
    _, hp = train(LViT, mans, tl, vl, False)
    print(f'best val_acc={max(h[2] for h in hp):.3f}')
    print(f'\ntriton {max(h[2] for h in hk):.3f} vs pytorch {max(h[2] for h in hp):.3f}')
