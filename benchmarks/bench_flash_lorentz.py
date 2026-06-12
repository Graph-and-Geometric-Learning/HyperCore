"""Benchmark train step (forward + backward): full vs flash on cuda."""
import statistics
import torch

from hypercore.nn.attention.flash_lorentz_attention import flash_attention_core, _reproject


def full_ref(q, k, v, scale):
    isc = 2.0 / scale
    qn = torch.cat([-q[..., :1], q[..., 1:]], dim=-1)
    s = torch.matmul(qn, k.transpose(-1, -2)) * isc
    p = torch.softmax(s, dim=-1)
    o = torch.matmul(p, v)
    return _reproject(o, 1.0)


def time_ms(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return statistics.median(ts)


def peak_mb(fn):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    try:
        fn(); torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1e6
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return -1.0
        raise


def step_fn(kind, q, k, v, scale):
    def step():
        for t in (q, k, v):
            if t.grad is not None:
                t.grad = None
        out = full_ref(q, k, v, scale) if kind == "full" else flash_attention_core(q, k, v, 1.0, scale)
        out.pow(2).sum().backward()
    return step


def main():
    assert torch.cuda.is_available(), "needs cuda gpu"
    print("device:", torch.cuda.get_device_name(0))
    H, D = 8, 64
    Dt = D + 1
    scale = float(H * Dt) ** 0.5
    print(f"\nH={H}, D={D}  (train step: fwd + bwd)")
    print("N|full ms|flash ms|full MB|flash MB|memx")

    for N in [512, 1024, 2048, 4096, 8192]:
        torch.manual_seed(0)
        q = torch.randn(1, H, N, Dt, device="cuda", requires_grad=True)
        k = torch.randn(1, H, N, Dt, device="cuda", requires_grad=True)
        v = torch.randn(1, H, N, Dt, device="cuda", requires_grad=True)

        mf = peak_mb(step_fn("full", q, k, v, scale))
        ms = peak_mb(step_fn("flash", q, k, v, scale))
        tf = time_ms(step_fn("full", q, k, v, scale)) if mf > 0 else -1
        ts = time_ms(step_fn("flash", q, k, v, scale)) if ms > 0 else -1

        def s(x): return f"{x:.1f}" if x > 0 else "OOM"
        memx = f"{mf / ms:.1f}x" if (mf > 0 and ms > 0) else "--"
        print(f"{N}|{s(tf)}|{s(ts)}|{s(mf)}|{s(ms)}|{memx}")
        del q, k, v
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
