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
    s = v[..., 1:]; t = v[..., :1]
    return (s * s).sum(-1, keepdim=True) - t * t


def _reproject(ave, c, eps=1e-15):
    d = lorentz_self_inner(ave).abs().clamp_min(eps).sqrt()
    return math.sqrt(c) * ave / d


def flash_lorentz_torch(q, k, v, c, scale, bn=128, mask=None):
    B, H, N, Dt = q.shape
    isc = 2.0 / scale
    qn = q.clone(); qn[..., 0] = -qn[..., 0]
    o = q.new_zeros(B, H, N, Dt)
    m = q.new_full((B, H, N, 1), float("-inf"))
    l = q.new_zeros(B, H, N, 1)
    mb = mask is not None and mask.dtype == torch.bool
    for j in range(0, N, bn):
        je = min(j + bn, N)
        kb = k[:, :, j:je]; vb = v[:, :, j:je]
        s = torch.matmul(qn, kb.transpose(-1, -2)) * isc
        if mask is not None:
            mj = mask[:, :, :, j:je]
            s = s.masked_fill(mj, float("-inf")) if mb else s + mj
        m_new = torch.maximum(m, s.max(dim=-1, keepdim=True).values)
        p = torch.exp(s - m_new); a = torch.exp(m - m_new)
        l = a * l + p.sum(dim=-1, keepdim=True)
        o = a * o + torch.matmul(p, vb); m = m_new
    return _reproject(o / l.clamp_min(1e-20), c)


if _HAS_TRITON:

    @triton.jit
    def _fwd(Q, K, V, O, Lo, sc,
             sh, sn, sd,
             N, D: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
        i = tl.program_id(0); h = tl.program_id(1)
        offs_m = i * BM + tl.arange(0, BM)
        offs_d = tl.arange(0, BD)
        md = offs_d < D; mm = offs_m < N
        base = h * sh
        qs = tl.load(Q + base + offs_m[:, None] * sn + (1 + offs_d)[None, :] * sd,
                     mask=mm[:, None] & md[None, :], other=0.0)
        qt = tl.load(Q + base + offs_m * sn, mask=mm, other=0.0)
        m_i = tl.full([BM], float("-inf"), tl.float32)
        l_i = tl.zeros([BM], tl.float32)
        accs = tl.zeros([BM, BD], tl.float32)
        acct = tl.zeros([BM], tl.float32)
        for n0 in range(0, N, BN):
            offs_n = n0 + tl.arange(0, BN); mn = offs_n < N
            ks = tl.load(K + base + offs_n[:, None] * sn + (1 + offs_d)[None, :] * sd,
                         mask=mn[:, None] & md[None, :], other=0.0)
            vs = tl.load(V + base + offs_n[:, None] * sn + (1 + offs_d)[None, :] * sd,
                         mask=mn[:, None] & md[None, :], other=0.0)
            kt = tl.load(K + base + offs_n * sn, mask=mn, other=0.0)
            vt = tl.load(V + base + offs_n * sn, mask=mn, other=0.0)
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
        lse = m_i + tl.log(l_i)
        accs = accs / l_i[:, None]; acct = acct / l_i
        tl.store(O + base + offs_m[:, None] * sn + (1 + offs_d)[None, :] * sd,
                 accs, mask=mm[:, None] & md[None, :])
        tl.store(O + base + offs_m * sn, acct, mask=mm)
        tl.store(Lo + h * N + offs_m, lse, mask=mm)

    @triton.jit
    def _bwd_kv_fused(QN, K, V, DO, Lv, Dv, DK, DV, P, sc,
                      sh, sn, sd, sp_h, sp_m, sp_n,
                      N, Dt: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
                      WRITE_P: tl.constexpr, P_FP16: tl.constexpr):
        j = tl.program_id(0); h = tl.program_id(1)
        offs_j = j * BN + tl.arange(0, BN)
        offs_d = tl.arange(0, BD)
        mj = offs_j < N; md = offs_d < Dt
        base = h * sh
        kb = tl.load(K + base + offs_j[:, None] * sn + offs_d[None, :] * sd,
                     mask=mj[:, None] & md[None, :], other=0.0)
        vb = tl.load(V + base + offs_j[:, None] * sn + offs_d[None, :] * sd,
                     mask=mj[:, None] & md[None, :], other=0.0)
        dkb = tl.zeros([BN, BD], tl.float32)
        dvb = tl.zeros([BN, BD], tl.float32)
        for i0 in range(0, N, BM):
            offs_i = i0 + tl.arange(0, BM); mi = offs_i < N
            qb = tl.load(QN + base + offs_i[:, None] * sn + offs_d[None, :] * sd,
                         mask=mi[:, None] & md[None, :], other=0.0)
            dob = tl.load(DO + base + offs_i[:, None] * sn + offs_d[None, :] * sd,
                          mask=mi[:, None] & md[None, :], other=0.0)
            li = tl.load(Lv + h * N + offs_i, mask=mi, other=0.0)
            di = tl.load(Dv + h * N + offs_i, mask=mi, other=0.0)
            s = tl.dot(qb, tl.trans(kb)) * sc
            s = tl.where(mj[None, :], s, float("-inf"))
            p = tl.exp(s - li[:, None])
            p = tl.where(mi[:, None] & mj[None, :], p, 0.0)
            if WRITE_P:
                if P_FP16:
                    tl.store(P + h * sp_h + offs_i[:, None] * sp_m + offs_j[None, :] * sp_n,
                             p.to(tl.float16), mask=mi[:, None] & mj[None, :])
                else:
                    tl.store(P + h * sp_h + offs_i[:, None] * sp_m + offs_j[None, :] * sp_n,
                             p, mask=mi[:, None] & mj[None, :])
            dp = tl.dot(dob, tl.trans(vb))
            ds = p * (dp - di[:, None]) * sc
            dvb += tl.dot(tl.trans(p).to(dob.dtype), dob)
            dkb += tl.dot(tl.trans(ds).to(qb.dtype), qb)
        tl.store(DK + base + offs_j[:, None] * sn + offs_d[None, :] * sd,
                 dkb, mask=mj[:, None] & md[None, :])
        tl.store(DV + base + offs_j[:, None] * sn + offs_d[None, :] * sd,
                 dvb, mask=mj[:, None] & md[None, :])

    @triton.jit
    def _bwd_q(QN, K, V, DO, Lv, Dv, DQ, P, sc,
               sh, sn, sd, sp_h, sp_m, sp_n,
               N, Dt: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
               USE_P: tl.constexpr, P_FP16: tl.constexpr):
        i = tl.program_id(0); h = tl.program_id(1)
        offs_i = i * BM + tl.arange(0, BM)
        offs_d = tl.arange(0, BD)
        mi = offs_i < N; md = offs_d < Dt
        base = h * sh
        qb = tl.load(QN + base + offs_i[:, None] * sn + offs_d[None, :] * sd,
                     mask=mi[:, None] & md[None, :], other=0.0)
        dob = tl.load(DO + base + offs_i[:, None] * sn + offs_d[None, :] * sd,
                      mask=mi[:, None] & md[None, :], other=0.0)
        di = tl.load(Dv + h * N + offs_i, mask=mi, other=0.0)
        dqb = tl.zeros([BM, BD], tl.float32)
        for j0 in range(0, N, BN):
            offs_j = j0 + tl.arange(0, BN); mj = offs_j < N
            kb = tl.load(K + base + offs_j[:, None] * sn + offs_d[None, :] * sd,
                         mask=mj[:, None] & md[None, :], other=0.0)
            vb = tl.load(V + base + offs_j[:, None] * sn + offs_d[None, :] * sd,
                         mask=mj[:, None] & md[None, :], other=0.0)
            if USE_P:
                if P_FP16:
                    p = tl.load(P + h * sp_h + offs_i[:, None] * sp_m + offs_j[None, :] * sp_n,
                                mask=mi[:, None] & mj[None, :], other=0.0).to(tl.float32)
                else:
                    p = tl.load(P + h * sp_h + offs_i[:, None] * sp_m + offs_j[None, :] * sp_n,
                                mask=mi[:, None] & mj[None, :], other=0.0)
            else:
                li = tl.load(Lv + h * N + offs_i, mask=mi, other=0.0)
                s = tl.dot(qb, tl.trans(kb)) * sc
                s = tl.where(mj[None, :], s, float("-inf"))
                p = tl.exp(s - li[:, None])
            dp = tl.dot(dob, tl.trans(vb))
            ds = p * (dp - di[:, None]) * sc
            dqb += tl.dot(ds.to(kb.dtype), kb)
        tl.store(DQ + base + offs_i[:, None] * sn + offs_d[None, :] * sd,
                 dqb, mask=mi[:, None] & md[None, :])

    def _fwd_run(q, k, v, scale):
        B, H, N, Dt = q.shape; D = Dt - 1
        q2 = q.reshape(B * H, N, Dt).contiguous()
        k2 = k.reshape(B * H, N, Dt).contiguous()
        v2 = v.reshape(B * H, N, Dt).contiguous()
        o = torch.empty_like(q2)
        lse = torch.empty(B * H, N, device=q.device, dtype=torch.float32)
        bd = triton.next_power_of_2(D); sc = 2.0 / scale
        BM = BN = 64
        grid = (triton.cdiv(N, BM), B * H)
        _fwd[grid](q2, k2, v2, o, lse, sc,
                   q2.stride(0), q2.stride(1), q2.stride(2),
                   N, D, BM=BM, BN=BN, BD=bd)
        return o.reshape(B, H, N, Dt), lse.reshape(B, H, N, 1)

    # mode: 'recomp' = no P buffer (smallest mem, V1-style); 'fused' = save P in _bwd_kv,
    #   reuse in _bwd_q (faster bwd, larger mem). _P_DTYPE picks fp16 (compact)
    #   or fp32 (V1-precision); fp16 costs ~1e-4 in dQ from cast round-off.
    _MODE = 'fused'
    _P_DTYPE = torch.float32

    _BWD_KV_CFGS = [(16, 32, 1, 4), (32, 16, 1, 4), (16, 16, 2, 4)]
    _BWD_Q_CFGS  = [(32, 16, 1, 4), (16, 16, 1, 4), (16, 16, 2, 4)]
    _BWD_KV_PICK = {}; _BWD_Q_PICK = {}
    _BWD_LAST_KV = None; _BWD_LAST_Q = None

    def _oor(e):
        s = str(e).lower()
        return ("out of resource" in s or "shared memory" in s
                or "outofresources" in type(e).__name__.lower())

    def _try_ladder(cfgs, pick, key, launch):
        order = list(range(len(cfgs)))
        if key in pick:
            j = pick[key]; order = [j] + [i for i in order if i != j]
        last = RuntimeError("no cfg fit shared memory")
        for ci in order:
            BM, BN, ns, nw = cfgs[ci]
            try:
                launch(BM, BN, ns, nw); pick[key] = ci
                return (BM, BN, ns, nw)
            except Exception as e:
                if _oor(e): last = e; continue
                raise
        raise last

    def _bwd_run(qn, k, v, do, lse, dvec, scale):
        global _BWD_LAST_KV, _BWD_LAST_Q
        B, H, N, Dt = qn.shape; M = B * H
        qn2 = qn.reshape(M, N, Dt).contiguous()
        k2 = k.reshape(M, N, Dt).contiguous()
        v2 = v.reshape(M, N, Dt).contiguous()
        do2 = do.reshape(M, N, Dt).contiguous()
        l2 = lse.reshape(M, N).contiguous()
        d2 = dvec.reshape(M, N).contiguous()
        sc = 2.0 / scale
        bd = triton.next_power_of_2(Dt)
        st0, st1, st2 = qn2.stride(0), qn2.stride(1), qn2.stride(2)
        key = (Dt, bd)

        use_p = (_MODE == 'fused')
        p_fp16 = use_p and (_P_DTYPE == torch.float16)
        if use_p:
            p_buf = torch.empty(M, N, N, device=qn.device, dtype=_P_DTYPE)
            sp_h, sp_m, sp_n = p_buf.stride(0), p_buf.stride(1), p_buf.stride(2)
        else:
            p_buf = qn2; sp_h = sp_m = sp_n = 0

        dk = torch.zeros_like(k2); dv = torch.zeros_like(v2)
        def launch_kv(BM, BN, ns, nw):
            _bwd_kv_fused[(triton.cdiv(N, BN), M)](
                qn2, k2, v2, do2, l2, d2, dk, dv, p_buf, sc,
                st0, st1, st2, sp_h, sp_m, sp_n,
                N, Dt, BM=BM, BN=BN, BD=bd, num_warps=nw, num_stages=ns,
                WRITE_P=use_p, P_FP16=p_fp16)
        _BWD_LAST_KV = _try_ladder(_BWD_KV_CFGS, _BWD_KV_PICK, key, launch_kv)

        dq = torch.zeros_like(qn2)
        def launch_q(BM, BN, ns, nw):
            _bwd_q[(triton.cdiv(N, BM), M)](qn2, k2, v2, do2, l2, d2, dq, p_buf, sc,
                                            st0, st1, st2, sp_h, sp_m, sp_n,
                                            N, Dt, BM=BM, BN=BN, BD=bd,
                                            num_warps=nw, num_stages=ns,
                                            USE_P=use_p, P_FP16=p_fp16)
        _BWD_LAST_Q = _try_ladder(_BWD_Q_CFGS, _BWD_Q_PICK, key, launch_q)

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
            qn = q.clone(); qn[..., 0] = -qn[..., 0]
            dq, dk, dv = _bwd_run(qn, k, v, do, lse, dvec, sc)
            dq[..., 0] = -dq[..., 0]
            return dq, dk, dv, None

    def flash_lorentz_triton(q, k, v, c, scale):
        o = _FlashFn.apply(q, k, v, float(scale))
        return _reproject(o, c)


def flash_attention_core(q, k, v, c, scale, mask=None):
    if _HAS_TRITON and q.is_cuda and mask is None:
        return flash_lorentz_triton(q, k, v, c, scale)
    return flash_lorentz_torch(q, k, v, c, scale, mask=mask)
