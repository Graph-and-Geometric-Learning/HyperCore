"""Flash (IO-aware) attention for the Lorentz full attention path. O(N) memory, same result as full_attention."""

import math
import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


def lorentz_self_inner(v):
    space = v[..., 1:]
    time = v[..., :1]
    return (space * space).sum(-1, keepdim=True) - time * time


def _reproject(ave, c, eps=1e-15):
    denom = lorentz_self_inner(ave).abs().clamp_min(eps).sqrt()
    return math.sqrt(c) * ave / denom


def flash_lorentz_torch(q, k, v, c, scale, bn=128, mask=None):
    B, H, N, Dt = q.shape
    isc = 2.0 / scale

    qn = q.clone()
    qn[..., 0] = -qn[..., 0]

    o = q.new_zeros(B, H, N, Dt)
    m = q.new_full((B, H, N, 1), float("-inf"))
    l = q.new_zeros(B, H, N, 1)
    mbool = mask is not None and mask.dtype == torch.bool

    for j in range(0, N, bn):
        je = min(j + bn, N)
        kb = k[:, :, j:je]
        vb = v[:, :, j:je]

        s = torch.matmul(qn, kb.transpose(-1, -2)) * isc
        if mask is not None:
            mb = mask[:, :, :, j:je]
            s = s.masked_fill(mb, float("-inf")) if mbool else s + mb

        m_new = torch.maximum(m, s.max(dim=-1, keepdim=True).values)
        p = torch.exp(s - m_new)
        a = torch.exp(m - m_new)
        l = a * l + p.sum(dim=-1, keepdim=True)
        o = a * o + torch.matmul(p, vb)
        m = m_new

    return _reproject(o / l.clamp_min(1e-20), c)


if _HAS_TRITON:

    @triton.jit
    def _flash_lorentz_fwd(
        Q, K, V, O, sm_scale,
        stride_qh, stride_qn, stride_qd,
        stride_kh, stride_kn, stride_kd,
        stride_vh, stride_vn, stride_vd,
        stride_oh, stride_on, stride_od,
        N,
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_h = tl.program_id(1)

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        mask_m = offs_m < N

        qs = tl.load(
            Q + off_h * stride_qh + offs_m[:, None] * stride_qn + (1 + offs_d)[None, :] * stride_qd,
            mask=mask_m[:, None] & mask_d[None, :], other=0.0,
        )
        qt = tl.load(Q + off_h * stride_qh + offs_m * stride_qn, mask=mask_m, other=0.0)

        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc_space = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
        acc_time = tl.zeros([BLOCK_M], tl.float32)

        for n0 in range(0, N, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N

            ks = tl.load(
                K + off_h * stride_kh + offs_n[:, None] * stride_kn + (1 + offs_d)[None, :] * stride_kd,
                mask=mask_n[:, None] & mask_d[None, :], other=0.0,
            )
            vs = tl.load(
                V + off_h * stride_vh + offs_n[:, None] * stride_vn + (1 + offs_d)[None, :] * stride_vd,
                mask=mask_n[:, None] & mask_d[None, :], other=0.0,
            )
            kt = tl.load(K + off_h * stride_kh + offs_n * stride_kn, mask=mask_n, other=0.0)
            vt = tl.load(V + off_h * stride_vh + offs_n * stride_vn, mask=mask_n, other=0.0)

            s = tl.dot(qs, tl.trans(ks)) - qt[:, None] * kt[None, :]
            s = s * sm_scale
            s = tl.where(mask_n[None, :], s, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            p = tl.exp(s - m_new[:, None])
            a = tl.exp(m_i - m_new)

            l_i = a * l_i + tl.sum(p, axis=1)
            acc_space = acc_space * a[:, None] + tl.dot(p.to(vs.dtype), vs)
            acc_time = acc_time * a + tl.sum(p * vt[None, :], axis=1)
            m_i = m_new

        acc_space = acc_space / l_i[:, None]
        acc_time = acc_time / l_i

        tl.store(
            O + off_h * stride_oh + offs_m[:, None] * stride_on + (1 + offs_d)[None, :] * stride_od,
            acc_space, mask=mask_m[:, None] & mask_d[None, :],
        )
        tl.store(O + off_h * stride_oh + offs_m * stride_on, acc_time, mask=mask_m)

    _CONFIGS = [
        dict(block_m=128, block_n=64, num_stages=2, num_warps=4),
        dict(block_m=64, block_n=64, num_stages=2, num_warps=4),
        dict(block_m=64, block_n=32, num_stages=2, num_warps=4),
        dict(block_m=64, block_n=32, num_stages=1, num_warps=4),
        dict(block_m=32, block_n=32, num_stages=1, num_warps=2),
    ]
    _best_config = {}

    def flash_lorentz_triton(q, k, v, c, scale):
        B, H, N, Dt = q.shape
        D = Dt - 1

        q2 = q.reshape(B * H, N, Dt).contiguous()
        k2 = k.reshape(B * H, N, Dt).contiguous()
        v2 = v.reshape(B * H, N, Dt).contiguous()
        out = torch.empty_like(q2)

        block_d = triton.next_power_of_2(D)
        sm_scale = 2.0 / scale

        configs = [_best_config[Dt]] if Dt in _best_config else _CONFIGS
        for cfg in configs:
            try:
                grid = (triton.cdiv(N, cfg["block_m"]), B * H)
                _flash_lorentz_fwd[grid](
                    q2, k2, v2, out, sm_scale,
                    q2.stride(0), q2.stride(1), q2.stride(2),
                    k2.stride(0), k2.stride(1), k2.stride(2),
                    v2.stride(0), v2.stride(1), v2.stride(2),
                    out.stride(0), out.stride(1), out.stride(2),
                    N, D,
                    BLOCK_M=cfg["block_m"], BLOCK_N=cfg["block_n"], BLOCK_D=block_d,
                    num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
                )
                _best_config[Dt] = cfg
                return _reproject(out.reshape(B, H, N, Dt), c)
            except triton.runtime.errors.OutOfResources:
                continue
        raise RuntimeError("no Triton block config fit in shared memory")


def flash_attention_core(q, k, v, c, scale, mask=None):
    if _HAS_TRITON and q.is_cuda and mask is None and not torch.is_grad_enabled():
        return flash_lorentz_triton(q, k, v, c, float(scale))
    return flash_lorentz_torch(q, k, v, c, scale, mask=mask)
