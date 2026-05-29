"""
Tests for attention_type='flash' in LorentzMultiheadAttention.
Flash must match the exact 'full' path. Triton test skips without a cuda gpu.
"""
import math

import pytest
import torch

from hypercore.manifolds import Lorentz
from hypercore.nn.attention.lorentz_former_conv import LorentzMultiheadAttention
from hypercore.nn.attention.flash_lorentz_attention import (
    flash_lorentz_torch,
    lorentz_self_inner,
    _HAS_TRITON,
)


def make_layer(num_heads, dim, concat, normalize, device, dtype):
    torch.manual_seed(0)
    return LorentzMultiheadAttention(
        Lorentz(c=1.0), 16, dim + 1, num_heads,
        attention_type="full", trans_heads_concat=concat, normalize=normalize,
    ).to(device=device, dtype=dtype)


def rand_input(batch, seq_len, layer, device, dtype):
    torch.manual_seed(1)
    feat = layer.num_heads * layer.in_channels
    return torch.randn(batch, seq_len, feat, device=device, dtype=dtype)


def run_both(layer, x):
    with torch.no_grad():
        layer.attention_type = "full"
        out_full = layer(x, x)
        layer.attention_type = "flash"
        out_flash = layer(x, x)
    return out_full, out_flash


@pytest.mark.parametrize("concat", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_flash_matches_full_cpu(concat, normalize):
    layer = make_layer(4, 32, concat, normalize, "cpu", torch.float64)
    x = rand_input(2, 96, layer, "cpu", torch.float64)
    out_full, out_flash = run_both(layer, x)
    assert (out_full - out_flash).abs().max() < 1e-9


@pytest.mark.parametrize("seq_len", [64, 100, 128, 150])
def test_flash_handles_tail_blocks(seq_len):
    torch.manual_seed(0)
    H, D = 2, 32
    q = torch.randn(1, H, seq_len, D + 1, dtype=torch.float64)
    k = torch.randn(1, H, seq_len, D + 1, dtype=torch.float64)
    v = torch.randn(1, H, seq_len, D + 1, dtype=torch.float64)
    scale = torch.tensor([math.sqrt(H * (D + 1))], dtype=torch.float64)

    out_small = flash_lorentz_torch(q, k, v, c=1.0, scale=scale, bn=32)
    out_big = flash_lorentz_torch(q, k, v, c=1.0, scale=scale, bn=4096)
    assert (out_small - out_big).abs().max() < 1e-12


def test_flash_output_on_manifold():
    layer = make_layer(4, 32, True, False, "cpu", torch.float64)
    x = rand_input(2, 96, layer, "cpu", torch.float64)
    layer.attention_type = "flash"
    with torch.no_grad():
        out = layer(x, x)
    inner = lorentz_self_inner(out)
    assert torch.allclose(inner, torch.full_like(inner, -layer.manifold.c.item()), atol=1e-4)


def test_flash_gradients_match_full():
    layer = make_layer(4, 32, True, False, "cpu", torch.float64)
    x = rand_input(2, 64, layer, "cpu", torch.float64)

    layer.attention_type = "full"
    g_full = torch.autograd.grad(layer(x, x).sum(), layer.Wq.weight)[0]
    layer.attention_type = "flash"
    g_flash = torch.autograd.grad(layer(x, x).sum(), layer.Wq.weight)[0]
    assert (g_full - g_flash).abs().max() < 1e-8


def test_output_attentions_falls_back_to_full():
    layer = make_layer(4, 32, False, False, "cpu", torch.float64)
    x = rand_input(1, 48, layer, "cpu", torch.float64)
    layer.attention_type = "flash"
    with torch.no_grad():
        out, attn = layer(x, x, output_attentions=True)
    assert attn is not None
    assert torch.allclose(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)), atol=1e-5)


@pytest.mark.skipif(not (_HAS_TRITON and torch.cuda.is_available()), reason="needs triton and cuda gpu")
@pytest.mark.parametrize("concat", [False, True])
@pytest.mark.parametrize("seq_len", [256, 1000])
def test_flash_triton_matches_full_cuda(concat, seq_len):
    layer = make_layer(8, 64, concat, False, "cuda", torch.float32)
    x = rand_input(1, seq_len, layer, "cuda", torch.float32)
    out_full, out_flash = run_both(layer, x)
    assert (out_full - out_flash).abs().max() < 1e-3
