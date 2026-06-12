"""Flash (IO-aware) Lorentz attention. Triton forward + backward, O(N) memory."""

import math
import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


def lorentz_self_inner(v):
    # uses s^2 - t^2 convention (= +c on the hyperboloid)
    s = v[..., 1:]
    t = v[..., :1]
    return (s * s).sum(-1, keepdim=True) - t * t


def _reproject(ave, c, eps=1e-15):
    d = lorentz_self_inner(ave).abs().clamp_min(eps).sqrt()
    return math.sqrt(c) * ave / d


def flash_lorentz_torch(q, k, v, c, scale, bn=128, mask=None):
    # differentiable reference: CPU / masked / fallback
    B, H, N, Dt = q.shape
    isc = 2.0 / scale

    qn = q.clone()
    qn[..., 0] = -qn[..., 0]   # Lorentz inner via time-sign flip

    o = q.new_zeros(B, H, N, Dt)
    m = q.new_full((B, H, N, 1), float("-inf"))
    l = q.new_zeros(B, H, N, 1)
    mb = mask is not None and mask.dtype == torch.bool

    for j in range(0, N, bn):
        je = min(j + bn, N)
        kb = k[:, :, j:je]
        vb = v[:, :, j:je]
        s = torch.matmul(qn, kb.transpose(-1, -2)) * isc
        if mask is not None:
            mj = mask[:, :, :, j:je]
            s = s.masked_fill(mj, float("-inf")) if mb else s + mj
        m_new = torch.maximum(m, s.max(dim=-1, keepdim=True).values)
        p = torch.exp(s - m_new)
        a = torch.exp(m - m_new)
        l = a * l + p.sum(dim=-1, keepdim=True)
        o = a * o + torch.matmul(p, vb)
        m = m_new

    return _reproject(o / l.clamp_min(1e-20), c)


if _HAS_TRITON:

    @triton.jit
    def _fwd(Q, K, V, O, Lo, sc,
             sh, sn, sd,
             N, D: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
        # forward: also stores lse (logsumexp per row) for the backward
        i = tl.program_id(0)
        h = tl.program_id(1)
        offs_m = i * BM + tl.arange(0, BM)
        offs_d = tl.arange(0, BD)
        md = offs_d < D
        mm = offs_m < N
        base = h * sh

        qs = tl.load(Q + base + offs_m[:, None] * sn + (1 + offs_d)[None, :] * sd,
                     mask=mm[:, None] & md[None, :], other=0.0)
        qt = tl.load(Q + base + offs_m * sn, mask=mm, other=0.0)

        m_i = tl.full([BM], float("-inf"), tl.float32)
        l_i = tl.zeros([BM], tl.float32)
        accs = tl.zeros([BM, BD], tl.float32)
        acct = tl.zeros([BM], tl.float32)

        for n0 in range(0, N, BN):
            offs_n = n0 + tl.arange(0, BN)
            mn = offs_n < N
            ks = tl.load(K + base + offs_n[:, None] * sn + (1 + offs_d)[None, :] * sd,
                         mask=mn[:, None] & md[None, :], other=0.0)
            vs = tl.load(V + base + offs_n[:, None] * sn + (1 + offs_d)[None, :] * sd,
                         mask=mn[:, None] & md[None, :], other=0.0)
            kt = tl.load(K + base + offs_n * sn, mask=mn, other=0.0)
            vt = tl.load(V + base + offs_n * sn, mask=mn, other=0.0)

            # Lorentz inner via split: space dot - time product
            s = tl.dot(qs, tl.trans(ks)) - qt[:, None] * kt[None, :]
            s = s * sc
            s = tl.where(mn[None, :], s, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            p = tl.exp(s - m_new[:, None])
            a = tl.exp(m_i - m_new)
            l_i = a * l_i + tl.sum(p, axis=1)
            accs = accs * a[:, None] + tl.dot(p.to(vs.dtype), vs)
            acct = acct * a + tl.sum(p * vt[None, :], axis=1)
            m_i = m_new

        lse = m_i + tl.log(l_i)   # saved for backward
        accs = accs / l_i[:, None]
        acct = acct / l_i

        tl.store(O + base + offs_m[:, None] * sn + (1 + offs_d)[None, :] * sd,
                 accs, mask=mm[:, None] & md[None, :])
        tl.store(O + base + offs_m * sn, acct, mask=mm)
        tl.store(Lo + h * N + offs_m, lse, mask=mm)

    @triton.jit
    def _bwd_kv(QN, K, V, DO, Lv, Dv, DK, DV, sc,
                sh, sn, sd,
                N, Dt: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
        # block over keys, loop queries; recompute P = exp(S - L) from saved L
        j = tl.program_id(0)
        h = tl.program_id(1)
        offs_j = j * BN + tl.arange(0, BN)
        offs_d = tl.arange(0, BD)
        mj = offs_j < N
        md = offs_d < Dt
        base = h * sh

        kb = tl.load(K + base + offs_j[:, None] * sn + offs_d[None, :] * sd,
                     mask=mj[:, None] & md[None, :], other=0.0)
        vb = tl.load(V + base + offs_j[:, None] * sn + offs_d[None, :] * sd,
                     mask=mj[:, None] & md[None, :], other=0.0)
        dkb = tl.zeros([BN, BD], tl.float32)
        dvb = tl.zeros([BN, BD], tl.float32)

        for i0 in range(0, N, BM):
            offs_i = i0 + tl.arange(0, BM)
            mi = offs_i < N
            qb = tl.load(QN + base + offs_i[:, None] * sn + offs_d[None, :] * sd,
                         mask=mi[:, None] & md[None, :], other=0.0)
            dob = tl.load(DO + base + offs_i[:, None] * sn + offs_d[None, :] * sd,
                          mask=mi[:, None] & md[None, :], other=0.0)
            li = tl.load(Lv + h * N + offs_i, mask=mi, other=0.0)
            di = tl.load(Dv + h * N + offs_i, mask=mi, other=0.0)

            s = tl.dot(qb, tl.trans(kb)) * sc
            s = tl.where(mj[None, :], s, float("-inf"))
            p = tl.exp(s - li[:, None])
            dp = tl.dot(dob, tl.trans(vb))
            ds = p * (dp - di[:, None]) * sc   # softmax Jacobian
            dvb += tl.dot(tl.trans(p).to(dob.dtype), dob)
            dkb += tl.dot(tl.trans(ds).to(qb.dtype), qb)

        tl.store(DK + base + offs_j[:, None] * sn + offs_d[None, :] * sd,
                 dkb, mask=mj[:, None] & md[None, :])
        tl.store(DV + base + offs_j[:, None] * sn + offs_d[None, :] * sd,
                 dvb, mask=mj[:, None] & md[None, :])

    @triton.jit
    def _bwd_q(QN, K, V, DO, Lv, Dv, DQ, sc,
               sh, sn, sd,
               N, Dt: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
        # block over queries, loop keys
        i = tl.program_id(0)
        h = tl.program_id(1)
        offs_i = i * BM + tl.arange(0, BM)
        offs_d = tl.arange(0, BD)
        mi = offs_i < N
        md = offs_d < Dt
        base = h * sh

        qb = tl.load(QN + base + offs_i[:, None] * sn + offs_d[None, :] * sd,
                     mask=mi[:, None] & md[None, :], other=0.0)
        dob = tl.load(DO + base + offs_i[:, None] * sn + offs_d[None, :] * sd,
                      mask=mi[:, None] & md[None, :], other=0.0)
        li = tl.load(Lv + h * N + offs_i, mask=mi, other=0.0)
        di = tl.load(Dv + h * N + offs_i, mask=mi, other=0.0)
        dqb = tl.zeros([BM, BD], tl.float32)

        for j0 in range(0, N, BN):
            offs_j = j0 + tl.arange(0, BN)
            mj = offs_j < N
            kb = tl.load(K + base + offs_j[:, None] * sn + offs_d[None, :] * sd,
                         mask=mj[:, None] & md[None, :], other=0.0)
            vb = tl.load(V + base + offs_j[:, None] * sn + offs_d[None, :] * sd,
                         mask=mj[:, None] & md[None, :], other=0.0)
            s = tl.dot(qb, tl.trans(kb)) * sc
            s = tl.where(mj[None, :], s, float("-inf"))
            p = tl.exp(s - li[:, None])
            dp = tl.dot(dob, tl.trans(vb))
            ds = p * (dp - di[:, None]) * sc
            dqb += tl.dot(ds.to(kb.dtype), kb)

        tl.store(DQ + base + offs_i[:, None] * sn + offs_d[None, :] * sd,
                 dqb, mask=mi[:, None] & md[None, :])

    def _fwd_run(q, k, v, scale):
        B, H, N, Dt = q.shape
        D = Dt - 1
        q2 = q.reshape(B * H, N, Dt).contiguous()
        k2 = k.reshape(B * H, N, Dt).contiguous()
        v2 = v.reshape(B * H, N, Dt).contiguous()
        o = torch.empty_like(q2)
        lse = torch.empty(B * H, N, device=q.device, dtype=torch.float32)
        bd = triton.next_power_of_2(D)
        sc = 2.0 / scale
        BM = BN = 64
        grid = (triton.cdiv(N, BM), B * H)
        _fwd[grid](q2, k2, v2, o, lse, sc,
                   q2.stride(0), q2.stride(1), q2.stride(2),
                   N, D, BM=BM, BN=BN, BD=bd)
        return o.reshape(B, H, N, Dt), lse.reshape(B, H, N, 1)

    def _bwd_run(qn, k, v, do, lse, dvec, scale):
        B, H, N, Dt = qn.shape
        M = B * H
        qn2 = qn.reshape(M, N, Dt).contiguous()
        k2 = k.reshape(M, N, Dt).contiguous()
        v2 = v.reshape(M, N, Dt).contiguous()
        do2 = do.reshape(M, N, Dt).contiguous()
        l2 = lse.reshape(M, N).contiguous()
        d2 = dvec.reshape(M, N).contiguous()
        dk = torch.zeros_like(k2)
        dv = torch.zeros_like(v2)
        dq = torch.zeros_like(qn2)
        sc = 2.0 / scale
        bd = triton.next_power_of_2(Dt)
        BM = BN = 16   # small block to fit shared memory on T4
        _bwd_kv[(triton.cdiv(N, BN), M)](qn2, k2, v2, do2, l2, d2, dk, dv, sc,
                                         qn2.stride(0), qn2.stride(1), qn2.stride(2),
                                         N, Dt, BM=BM, BN=BN, BD=bd)
        _bwd_q[(triton.cdiv(N, BM), M)](qn2, k2, v2, do2, l2, d2, dq, sc,
                                        qn2.stride(0), qn2.stride(1), qn2.stride(2),
                                        N, Dt, BM=BM, BN=BN, BD=bd)
        return dq.reshape(B, H, N, Dt), dk.reshape(B, H, N, Dt), dv.reshape(B, H, N, Dt)

    class _FlashFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, scale):
            o, lse = _fwd_run(q, k, v, scale)
            ctx.save_for_backward(q, k, v, o, lse)
            ctx.scale = float(scale)
            return o

        @staticmethod
        def backward(ctx, do):
            q, k, v, o, lse = ctx.saved_tensors
            sc = ctx.scale
            dvec = (do * o).sum(-1, keepdim=True)
            qn = q.clone()
            qn[..., 0] = -qn[..., 0]
            dq, dk, dv = _bwd_run(qn, k, v, do, lse, dvec, sc)
            dq[..., 0] = -dq[..., 0]   # undo the time-sign flip on dq
            return dq, dk, dv, None

    def flash_lorentz_triton(q, k, v, c, scale):
        o = _FlashFn.apply(q, k, v, float(scale))
        return _reproject(o, c)


def flash_attention_core(q, k, v, c, scale, mask=None):
    # Triton for CUDA without mask; torch fallback otherwise
    if _HAS_TRITON and q.is_cuda and mask is None:
        return flash_lorentz_triton(q, k, v, c, scale)
    return flash_lorentz_torch(q, k, v, c, scale, mask=mask)
