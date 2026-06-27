# HyperCore Fused Kernel Benchmark

Reproducible benchmark of fused Triton kernels (linear-focus attention core, Lorentz
lift, LResNet residual) against the PyTorch reference path inside HyperCore.

**On CIFAR-10 (LViT, n=3 seeds, T4):**
- 1.35× wall-clock speedup on a real training step (B=128, img=32)
- +0.86 ± 0.32 pp accuracy over PyTorch baseline (95% CI [+0.50, +1.22], excludes 0)
- Forward correctness 6.7e-6, gradient correctness 1.1e-5

Full report: https://jet-teller-d33.notion.site/HyperCore-Triton-Kernels-Progress-Report-38b2bf4e410d8079b5e7d418218c5038

## Files

- `hypercore_kernels.py` — Triton kernels and autograd Functions (`core_fused`, `lift_t`, `lres_f`)
- `hypercore_patches.py` — isolated HyperCore loader + monkey-patches gated by `_ON[0]`
- `benchmark_cifar10.py` — paired training driver, JSON dump per seed

## Environment

T4 (sm_75) or newer; Triton ≥ 3.4; PyTorch ≥ 2.4 with CUDA; Python ≥ 3.10.

```bash
pip install triton geoopt loguru
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.10.0+cu128.html
```

Adjust the wheel URL to match your torch/CUDA combination.

## Run

Knobs are at the top of `benchmark_cifar10.py`. Default config:
`SEED=0 EPOCHS=20 BATCH=128 LR=1e-3 ROOT='..' DATA='./data'`.

For three seeds change `SEED` and rerun:

```bash
cd benchmark
python benchmark_cifar10.py          # SEED=0
# edit benchmark_cifar10.py: SEED=1
python benchmark_cifar10.py
# edit: SEED=2
python benchmark_cifar10.py
```

Each run ~40 minutes on T4. Outputs `parity_seed{0,1,2}.json` with per-epoch history.

## Aggregate

```python
import json, statistics
runs = {s: json.load(open(f'parity_seed{s}.json')) for s in (0,1,2)}
fA = [runs[s]['final'][0] for s in runs]; fB = [runs[s]['final'][1] for s in runs]
d  = [b-a for a,b in zip(fA,fB)]
print(f"base {statistics.mean(fA):.2f}±{statistics.stdev(fA):.2f}")
print(f"ker  {statistics.mean(fB):.2f}±{statistics.stdev(fB):.2f}")
print(f"Δ    {statistics.mean(d):+.2f}±{statistics.stdev(d):.2f}pp")
```

## Notes

- `dropout=0.0` in the paired protocol so the two arms share stochasticity exactly,
  isolating kernel numerics. Trades some absolute accuracy for a clean delta.
- `w_y`, `norm_scale` frozen in both arms (the latter has unstable gradient in the
  reference path; unrelated to kernels).
- LResNet kernel fires only on block residuals with `scale=27.5`. The `add_pos`
  LResNet (`scale=1.0`) intentionally falls back to original — fusion there
  introduced a +0.2 error.
- `linear_focused` attention has a dim mismatch in its default config (porting
  artifact from the upstream hyperbolic-transformer repo); the driver patches
  `v_map_mlp` and `final_linear` inline in `_mk`.
