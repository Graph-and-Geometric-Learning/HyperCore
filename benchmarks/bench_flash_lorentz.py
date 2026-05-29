"""
Benchmark flash/full Lorentz attention on cuda gpu
"""
import argparse
import statistics

import torch

from hypercore.manifolds import Lorentz
from hypercore.nn.attention.lorentz_former_conv import LorentzMultiheadAttention


def time_ms(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))

    samples.sort()
    return statistics.median(samples), samples[0], samples[-1]


def peak_memory_mb(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    try:
        fn()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1e6
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return -1.0
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--configs", type=str, default="8x64,12x64,4x32", help="heads x dim")
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "needs cuda gpu"
    device = "cuda"
    manifold = Lorentz(c=1.0)
    print("device:", torch.cuda.get_device_name(0))

    print("correctness:")
    for concat in (False, True):
        torch.manual_seed(0)
        layer = LorentzMultiheadAttention(
            manifold, 16, 65, 8, attention_type="full", trans_heads_concat=concat,
        ).to(device)
        x = torch.randn(2, 256, layer.num_heads * layer.in_channels, device=device)
        with torch.no_grad():
            out_full = layer(x, x)
            layer.attention_type = "flash"
            out_flash = layer(x, x)
        err = (out_full - out_flash).abs().max().item()
        print(f"concat={concat} max abs error = {err:.2e}")

    for spec in args.configs.split(","):
        heads, dim = (int(t) for t in spec.lower().split("x"))
        torch.manual_seed(0)
        layer = LorentzMultiheadAttention(manifold, 16, dim + 1, heads, attention_type="full", trans_heads_concat=True)
        layer = layer.to(device)
        feat = layer.num_heads * layer.in_channels

        print(f"\nH={heads}, D={dim}", "N|full ms|flash ms|speedup|full MB|flash MB|memx", sep='\n')

        for N in args.seq_lens:
            x = torch.randn(1, N, feat, device=device)

            def run_full():
                layer.attention_type = "full"
                with torch.no_grad():
                    return layer(x, x)

            def run_flash():
                layer.attention_type = "flash"
                with torch.no_grad():
                    return layer(x, x)

            mem_full = peak_memory_mb(run_full)
            mem_flash = peak_memory_mb(run_flash)

            if mem_full < 0:  # baseline OOM, flash still runs
                mf, lo, hi = time_ms(run_flash, iters=args.iters)
                print(f"{N}|OOM|{mf:.2f} ({lo:.2f}-{hi:.2f})|--|OOM|{mem_flash:.1f}|inf")
            else:
                mo, lo_o, hi_o = time_ms(run_full, iters=args.iters)
                mf, lo_f, hi_f = time_ms(run_flash, iters=args.iters)
                print(f"{N}|{mo:.2f} ({lo_o:.2f}-{hi_o:.2f})|{mf:.2f} ({lo_f:.2f}-{hi_f:.2f})|"
                      f"{mo / mf:.2f}x|{mem_full:.1f}|{mem_flash:.1f}|{mem_full / mem_flash:.1f}x")

            del x
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
