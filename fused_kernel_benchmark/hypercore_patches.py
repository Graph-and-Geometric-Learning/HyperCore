"""HyperCore isolated loader + monkey-patches gated by _ON[0]."""
import sys, types, importlib, importlib.util
import torch, torch.nn as nn, torch.nn.functional as F
from hypercore_kernels import core_fused, lift_t, lres_f

_ON = [False]
_P  = [False]

def _stb(n,p):
    if n in sys.modules: return sys.modules[n]
    m=types.ModuleType(n); m.__path__=[p]; m.__package__=n; sys.modules[n]=m; return m

def _ld(n,fp,pkg):
    sp=importlib.util.spec_from_file_location(n,fp); m=importlib.util.module_from_spec(sp)
    m.__package__=pkg; sys.modules[n]=m; sp.loader.exec_module(m); return m

def _fl(pkg,mod):
    p=sys.modules[pkg]
    for s in dir(mod):
        if not s.startswith('_'): setattr(p,s,getattr(mod,s))

def load_hc(root):
    if root not in sys.path: sys.path.insert(0,root)
    msp=importlib.util.spec_from_file_location('hypercore.manifolds',f'{root}/hypercore/manifolds/__init__.py',submodule_search_locations=[f'{root}/hypercore/manifolds'])
    mm=importlib.util.module_from_spec(msp); mm.__package__='hypercore.manifolds'
    _stb('hypercore',root+'/hypercore'); sys.modules['hypercore.manifolds']=mm; msp.loader.exec_module(mm)
    Lz=mm.Lorentz
    _stb('hypercore.nn',root+'/hypercore/nn')
    for s in ['linear','conv','attention']: _stb(f'hypercore.nn.{s}',f'{root}/hypercore/nn/{s}')
    for nm,fp in [('hypercore.nn.linear.lorentz_linear','nn/linear/lorentz_linear.py'),('hypercore.nn.linear.lorentz_CLS','nn/linear/lorentz_CLS.py'),('hypercore.nn.conv.conv_util_layers','nn/conv/conv_util_layers.py'),('hypercore.nn.conv.lorentz_convolution','nn/conv/lorentz_convolution.py'),('hypercore.nn.conv.lorentz_MLR','nn/conv/lorentz_MLR.py')]:
        pkg=nm.rsplit('.',1)[0]; _fl(pkg,_ld(nm,f'{root}/hypercore/{fp}',pkg))
    for q in ['hypercore.nn.linear.lorentz_linear','hypercore.nn.linear.lorentz_CLS','hypercore.nn.conv.conv_util_layers','hypercore.nn.conv.lorentz_MLR']:
        _fl('hypercore.nn',sys.modules[q])
    for fn in ['flash_lorentz_attention','flash_lorentz_attention_v2','flash_lorentz_dispatch','linear_focus_attention_v3','patch_embedding']:
        try:
            x=_ld(f'hypercore.nn.attention.{fn}',f'{root}/hypercore/nn/attention/{fn}.py','hypercore.nn.attention')
            _fl('hypercore.nn.attention',x); _fl('hypercore.nn',x)
        except Exception: pass
    fm=_ld('hypercore.nn.attention.lorentz_former_conv',f'{root}/hypercore/nn/attention/lorentz_former_conv.py','hypercore.nn.attention')
    _fl('hypercore.nn.attention',fm); _fl('hypercore.nn',fm)
    _stb('hypercore.models',root+'/hypercore/models')
    LViT=_ld('hypercore.models.LViT',f'{root}/hypercore/models/LViT.py','hypercore.models').LViT
    return {'Lz':Lz,'LViT':LViT,
            'LMA':sys.modules['hypercore.nn'].LorentzMultiheadAttention,
            'LL':sys.modules['hypercore.nn'].LorentzLinear,
            'LN':sys.modules['hypercore.nn'].LorentzLayerNorm,
            'LR':sys.modules['hypercore.nn'].LResNet,
            'cup':f'{root}/hypercore/nn/conv/conv_util_layers.py'}

def apply(hc):
    if _P[0]: return
    _P[0]=True
    LMA,LL,LN,LR,cup = hc['LMA'],hc['LL'],hc['LN'],hc['LR'],hc['cup']

    def lf(self,hq,hk,hv,output_attentions=False,mask=None):
        qs=hq[...,1:]; ks=hk[...,1:]; v=hv[...,1:]
        if _ON[0]: at=core_fused(qs,ks,v,self.norm_scale)
        else:
            pq=(F.relu(qs)+1e-6)/(self.norm_scale.abs()+1e-6); pk=(F.relu(ks)+1e-6)/(self.norm_scale.abs()+1e-6)
            pq=self.fp(pq,p=self.power_k); pk=self.fp(pk,p=self.power_k)
            kv=torch.einsum('bnhm,bnhd->bhmd',pk,v); num=torch.einsum('bnhm,bhmd->bnhd',pq,kv)
            dn=torch.einsum('bnhd,bhd->bnh',pq,torch.einsum('bnhd->bhd',pk)).unsqueeze(-1); at=num/(dn+1e-6)
        at=at+self.v_map_mlp(v); B,N,H,Dr=at.shape
        at=self.final_linear(at.reshape(B,N,H*Dr)) if self.trans_heads_concat else at.mean(dim=1)
        t=((at**2).sum(-1,keepdims=True)+self.manifold.c)**0.5
        return torch.cat([t,at],dim=-1)
    LMA.linear_focus_attention=lf

    def lin(self,x,x_manifold='hyp',return_space=False):
        if x_manifold!='hyp':
            x=torch.cat([torch.ones_like(x)[...,0:1],x],dim=-1); x=self.manifold.expmap0(x)
        xs=self.linear(x)
        if self.num_heads>1:
            d=self.out_features//self.num_heads; xs=xs.reshape(xs.size(0),xs.size(1),self.num_heads,d)
        if return_space: return xs
        if _ON[0] and self.manifold_out is None: return lift_t(xs,float(self.c),1e-8,1.0)
        t=((xs**2).sum(-1,keepdims=True)+self.c).clamp_min(1e-8).sqrt(); return torch.cat([t,xs],-1)
    LL.forward=lin

    def ln(self,x,space_only=False,return_space=False):
        xs=x if space_only else x[...,1:]
        xs=self.layer(xs)
        if return_space: return xs
        if _ON[0] and self.manifold_out is None: return lift_t(xs,float(self.c),float(self.eps),1.0)
        t=((xs**2).sum(-1,keepdims=True)+self.c).clamp_min(self.eps).sqrt(); return torch.cat([t,xs],-1)
    LN.forward=ln

    sp=importlib.util.spec_from_file_location('_lro',cup)
    om=importlib.util.module_from_spec(sp); om.__package__='hypercore.nn.conv'; sp.loader.exec_module(om)
    LRO=om.LResNet.forward

    def lr(self,x,y,weight=None):
        if not (_ON[0] and weight is None): return LRO(self,x,y,weight)
        if (self.scale is None or getattr(self,'learned_scale',True) or self.manifold_out is not None or abs(float(self.scale)-27.5)>1e-6): return LRO(self,x,y,weight)
        if hasattr(self.w_y,'numel') and self.w_y.numel()!=1: return LRO(self,x,y,weight)
        if x.shape[-1]!=y.shape[-1]: return LRO(self,x,y,weight)
        try: return lres_f(x,y,float(self.w_y.detach()),float(self.scale),float(self.c),float(self.eps))
        except Exception: return LRO(self,x,y,weight)
    LR.forward=lr
