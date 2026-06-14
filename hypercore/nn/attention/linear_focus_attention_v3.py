"""Fused Triton ops for Lorentz linear-focus attention: phi + time-concat."""

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


def _fp_ref(x, p):
    n = x.norm(dim=-1, keepdim=True)
    xp = x.pow(p)
    np_ = xp.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return n / np_ * xp


def phi_ref(x, ns, p, eps=1e-6):
    x = (torch.relu(x) + eps) / (ns.abs() + eps)
    return _fp_ref(x, p)


def time_concat_ref(x, c, eps=1e-15):
    t = (x.pow(2).sum(-1, keepdim=True) + c).clamp_min(eps).sqrt()
    return torch.cat([t, x], dim=-1)


if _HAS_TRITON:

    @triton.jit
    def _phi_fwd(X, NS, O, NX, NXP,
                 M, D: tl.constexpr, p: tl.constexpr, eps: tl.constexpr,
                 BD: tl.constexpr):
        r = tl.program_id(0)
        od = tl.arange(0, BD)
        md = od < D
        b = r * D
        ns = tl.abs(tl.load(NS).to(tl.float32)) + eps
        x = tl.load(X + b + od, mask=md, other=0.0).to(tl.float32)
        x = (tl.where(x > 0, x, 0.0) + eps) / ns
        xp = tl.exp(p * tl.log(x))
        nx = tl.sqrt(tl.sum(x * x, axis=0))
        nxp = tl.maximum(tl.sqrt(tl.sum(xp * xp, axis=0)), 1e-6)
        tl.store(O + b + od, (nx / nxp * xp).to(O.dtype.element_ty), mask=md)
        tl.store(NX + r, nx)
        tl.store(NXP + r, nxp)

    @triton.jit
    def _phi_bwd(X, NS, NX, NXP, DO, DX, DNS,M, D: tl.constexpr,
    p:tl.constexpr,eps: tl.constexpr,BD: tl.constexpr):
        r = tl.program_id(0)
        od = tl.arange(0, BD)
        md = od < D
        b = r * D
        ns = tl.abs(tl.load(NS).to(tl.float32)) + eps
        x = tl.load(X + b + od, mask=md, other=0.0).to(tl.float32)
        do = tl.load(DO + b + od, mask=md, other=0.0).to(tl.float32)
        y = tl.where(x > 0, x, 0.0) + eps
        z = y / ns
        zp = tl.exp(p * tl.log(z))
        nz = tl.load(NX + r).to(tl.float32)
        nzp = tl.load(NXP + r).to(tl.float32)
        sdz = tl.sum(do * zp, axis=0)
        dzp = (nz / nzp) * do - (nz / (nzp * nzp)) * (zp / nzp) * sdz
        dnz = sdz / nzp
        dz = dzp * p * zp / z + dnz * z / nz
        dx = (dz / ns) * tl.where(x > 0, 1.0, 0.0)
        tl.store(DX + b + od, dx.to(DX.dtype.element_ty), mask=md)
        tl.store(DNS + r, tl.sum(dz * z, axis=0))

    @triton.jit
    def _tc_fwd(X, O, c,M, D: tl.constexpr, Dt: tl.constexpr, BD: tl.constexpr):
        r = tl.program_id(0)
        od = tl.arange(0, BD)
        md = od < D
        x = tl.load(X + r * D + od, mask=md, other=0.0).to(tl.float32)
        t = tl.sqrt(tl.maximum(tl.sum(x * x, axis=0) + c, 1e-15))
        tl.store(O + r * Dt, t.to(O.dtype.element_ty))
        tl.store(O + r * Dt + 1 + od, x.to(O.dtype.element_ty), mask=md)

    @triton.jit
    def _tc_bwd(X, DO, DX, c,
                M, D: tl.constexpr, Dt: tl.constexpr, BD: tl.constexpr):
        r = tl.program_id(0)
        od = tl.arange(0, BD)
        md = od < D
        x = tl.load(X + r * D + od, mask=md, other=0.0).to(tl.float32)
        dot = tl.load(DO + r * Dt).to(tl.float32)
        dox = tl.load(DO + r * Dt + 1 + od, mask=md, other=0.0).to(tl.float32)
        t = tl.sqrt(tl.maximum(tl.sum(x * x, axis=0) + c, 1e-15))
        tl.store(DX + r * D + od, (dox + dot * x / t).to(DX.dtype.element_ty), mask=md)

    def _phi_fwd_run(x, ns, p, eps):
        B, N, H, D = x.shape
        M = B * N * H
        xc = x.contiguous().view(M, D)
        o = torch.empty_like(xc)
        nx = torch.empty(M, device=x.device, dtype=torch.float32)
        nxp = torch.empty(M, device=x.device, dtype=torch.float32)
        bd = triton.next_power_of_2(D)
        _phi_fwd[(M,)](xc, ns, o, nx, nxp, M, D, float(p), float(eps), BD=bd,
                       num_warps=4, num_stages=2)
        return o.view(B, N, H, D), nx.view(B, N, H), nxp.view(B, N, H)

    def _phi_bwd_run(x, ns, nx, nxp, do, p, eps):
        B, N, H, D = x.shape
        M = B * N * H
        xc = x.contiguous().view(M, D)
        doc = do.contiguous().view(M, D)
        dx = torch.empty_like(xc)
        dns = torch.empty(M, device=x.device, dtype=torch.float32)
        bd = triton.next_power_of_2(D)
        _phi_bwd[(M,)](xc, ns, nx.contiguous().view(M), nxp.contiguous().view(M),
                       doc, dx, dns, M, D, float(p), float(eps), BD=bd,
                       num_warps=4, num_stages=2)
        ns_abs = ns.abs() + eps
        dns_s = -dns.sum() / ns_abs * torch.sign(ns).squeeze()
        return dx.view(B, N, H, D), dns_s.view_as(ns).to(ns.dtype)

    def _tc_fwd_run(x, c):
        sh = x.shape
        D = sh[-1]; Dt = D + 1
        M = x.numel() // D
        xc = x.contiguous().view(M, D)
        o = torch.empty(M, Dt, device=x.device, dtype=x.dtype)
        bd = triton.next_power_of_2(D)
        _tc_fwd[(M,)](xc, o, float(c), M, D, Dt, BD=bd, num_warps=4, num_stages=2)
        return o.view(*sh[:-1], Dt)

    def _tc_bwd_run(x, do, c):
        sh = x.shape
        D = sh[-1]; Dt = D + 1
        M = x.numel() // D
        xc = x.contiguous().view(M, D)
        doc = do.contiguous().view(M, Dt)
        dx = torch.empty(M, D, device=x.device, dtype=x.dtype)
        bd = triton.next_power_of_2(D)
        _tc_bwd[(M,)](xc, doc, dx, float(c), M, D, Dt, BD=bd, num_warps=4, num_stages=2)
        return dx.view(*sh)

    class _PhiFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, ns, p, eps):
            o, nx, nxp = _phi_fwd_run(x, ns, p, eps)
            ctx.save_for_backward(x, ns, nx, nxp)
            ctx.p = float(p); ctx.eps = float(eps)
            return o

        @staticmethod
        def backward(ctx, do):
            x, ns, nx, nxp = ctx.saved_tensors
            dx, dns = _phi_bwd_run(x, ns, nx, nxp, do, ctx.p, ctx.eps)
            return dx, dns, None, None

    class _TcFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, c):
            ctx.save_for_backward(x)
            ctx.c = float(c)
            return _tc_fwd_run(x, c)

        @staticmethod
        def backward(ctx, do):
            x, = ctx.saved_tensors
            return _tc_bwd_run(x, do, ctx.c), None


def phi(x, ns, p=2.0, eps=1e-6):
    if _HAS_TRITON and x.is_cuda:
        return _PhiFn.apply(x, ns, p, eps)
    return phi_ref(x, ns, p, eps)


def time_concat(x, c=1.0, eps=1e-15):
    if _HAS_TRITON and x.is_cuda:
        return _TcFn.apply(x, c)
    return time_concat_ref(x, c, eps)
