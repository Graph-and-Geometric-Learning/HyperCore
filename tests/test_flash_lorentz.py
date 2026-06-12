"""Tests for flash Lorentz attention. CUDA/Triton tests skip without a gpu."""
import math

import pytest
import torch

from hypercore.nn.attention.flash_lorentz_attention import (
    flash_lorentz_torch,
    flash_attention_core,
    lorentz_self_inner,
    _reproject,
    _HAS_TRITON,
)


def full_ref(q, k, v, c, scale):
    isc = 2.0 / scale
    qn = torch.cat([-q[..., :1], q[..., 1:]], dim=-1)
    s = torch.matmul(qn, k.transpose(-1, -2)) * isc
    p = torch.softmax(s, dim=-1)
    o = torch.matmul(p, v)
    return _reproject(o, c)


def rand_qkv(B, H, N, D, device, dtype, seed=0):
    torch.manual_seed(seed)
    Dt = D + 1
    q = torch.randn(B, H, N, Dt, device=device, dtype=dtype)
    k = torch.randn(B, H, N, Dt, device=device, dtype=dtype)
    v = torch.randn(B, H, N, Dt, device=device, dtype=dtype)
    return q, k, v


@pytest.mark.parametrize("N", [64, 100, 128, 150])
def test_torch_matches_full_cpu(N):
    q, k, v = rand_qkv(1, 2, N, 32, "cpu", torch.float64)
    scale = torch.tensor([math.sqrt(2 * 33)], dtype=torch.float64)
    out = flash_lorentz_torch(q, k, v, 1.0, scale, bn=32)
    ref = full_ref(q, k, v, 1.0, scale)
    assert (out - ref).abs().max() < 1e-9


def test_torch_tail_blocks():
    # smallest block size vs huge block: result must match
    q, k, v = rand_qkv(1, 2, 100, 32, "cpu", torch.float64)
    scale = torch.tensor([math.sqrt(2 * 33)], dtype=torch.float64)
    a = flash_lorentz_torch(q, k, v, 1.0, scale, bn=32)
    b = flash_lorentz_torch(q, k, v, 1.0, scale, bn=4096)
    assert (a - b).abs().max() < 1e-12


def test_torch_on_manifold():
    q, k, v = rand_qkv(1, 2, 96, 32, "cpu", torch.float64)
    scale = torch.tensor([math.sqrt(2 * 33)], dtype=torch.float64)
    out = flash_lorentz_torch(q, k, v, 1.0, scale)
    inner = lorentz_self_inner(out)
    # lorentz_self_inner uses s^2 - t^2 (= +c on the hyperboloid).
    # HyperCore's Manifold.inner uses -t^2 + s^2 = -c. Same surface, opposite sign.
    assert torch.allclose(inner, torch.full_like(inner, 1.0), atol=1e-4)


@pytest.mark.skipif(not (_HAS_TRITON and torch.cuda.is_available()), reason="needs triton and cuda gpu")
@pytest.mark.parametrize("N", [64, 128, 256])
def test_triton_fwd_matches_full(N):
    q, k, v = rand_qkv(1, 2, N, 64, "cuda", torch.float32)
    scale = math.sqrt(2 * 65)
    with torch.no_grad():
        out = flash_attention_core(q, k, v, 1.0, scale)
        ref = full_ref(q, k, v, 1.0, scale)
    assert (out - ref).abs().max() < 1e-3


@pytest.mark.skipif(not (_HAS_TRITON and torch.cuda.is_available()), reason="needs triton and cuda gpu")
@pytest.mark.parametrize("N", [64, 128])
def test_triton_bwd_matches_full(N):
    # gradients from Triton backward must match autograd on the full reference
    q, k, v = rand_qkv(1, 2, N, 64, "cuda", torch.float32)
    scale = math.sqrt(2 * 65)
    q.requires_grad_(True); k.requires_grad_(True); v.requires_grad_(True)

    out = flash_attention_core(q, k, v, 1.0, scale)
    g = torch.randn_like(out)
    out.backward(g)
    dq, dk, dv = q.grad.clone(), k.grad.clone(), v.grad.clone()

    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    ref = full_ref(q2, k2, v2, 1.0, scale)
    ref.backward(g)

    assert (dq - q2.grad).abs().max() < 1e-3
    assert (dk - k2.grad).abs().max() < 1e-3
    assert (dv - v2.grad).abs().max() < 1e-3
