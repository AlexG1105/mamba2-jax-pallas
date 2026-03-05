#!/usr/bin/env python3
"""
benchmarks/bench_model_forward.py

Benchmark one full forward pass of the Mamba2 SSD pipeline:
  Pallas Mosaic GPU  vs  Triton (mamba_ssm)  vs  Naive JAX/XLA

Tests configs matching real models:
  - mamba2-130m:  H=24,  P=64, N=128, G=1, Q=256
  - mamba2-2.7b:  H=80,  P=64, N=128, G=1, Q=256
  - nemotron-h:   H=128, P=64, N=128, G=8, Q=128

Usage:
    python benchmarks/bench_model_forward.py
"""

import os
import sys
import time
import types
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import jax
import jax.numpy as jnp

from mamba2_jax.kernels.ssd_combined import mamba_chunk_scan_combined_fwd
from mamba2_jax.ops.ssd_naive import mamba_chunk_scan_combined_naive

# ---------------------------------------------------------------------------
# Optional Triton reference
# ---------------------------------------------------------------------------
_HAS_TRITON = False
try:
    import torch
    import importlib.util
    _spec = importlib.util.find_spec("mamba_ssm")
    if _spec is None:
        raise ImportError("mamba-ssm not installed")
    _pkg = types.ModuleType("mamba_ssm")
    _pkg.__path__ = list(_spec.submodule_search_locations)
    _pkg.__package__ = "mamba_ssm"
    sys.modules["mamba_ssm"] = _pkg
    from mamba_ssm.ops.triton.ssd_combined import (
        _mamba_chunk_scan_combined_fwd as _triton_combined_fwd,
    )
    _HAS_TRITON = True
except Exception as e:
    print(f"[info] Triton reference unavailable ({e}); skipping Triton column.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_torch(x):
    return torch.tensor(np.array(x), device="cuda", dtype=torch.float32)


def _bench_pallas(fn, args, N=50, warmup=10, rep=30):
    """Amortized JAX benchmark using fori_loop."""
    sample = fn(*args)
    dummy = jax.tree.map(jnp.zeros_like, sample)
    first_leaf = jax.tree.leaves(sample)[0]
    idx = (0,) * first_leaf.ndim

    @jax.jit
    def looped(*a):
        def body(i, acc):
            result = jax.lax.cond(
                acc > -1e30,
                lambda: fn(*a),
                lambda: dummy,
            )
            return acc + jax.tree.leaves(result)[0][idx]
        return jax.lax.fori_loop(0, N, body, 0.0)

    for _ in range(warmup):
        jax.block_until_ready(looped(*args))

    times = []
    for _ in range(rep):
        t0 = time.perf_counter()
        jax.block_until_ready(looped(*args))
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1e3 / N


def _bench_triton(fn, warmup=10, rep=50):
    """Benchmark a Triton/PyTorch function using do_bench."""
    from triton.testing import do_bench
    return do_bench(fn, warmup=warmup, rep=rep)


def _bench_naive(fn, args, N=20, warmup=5, rep=15):
    """Amortized JAX benchmark for naive impl (same method as Pallas)."""
    sample = fn(*args)
    dummy = jax.tree.map(jnp.zeros_like, sample)
    first_leaf = jax.tree.leaves(sample)[0]
    idx = (0,) * first_leaf.ndim

    @jax.jit
    def looped(*a):
        def body(i, acc):
            result = jax.lax.cond(
                acc > -1e30,
                lambda: fn(*a),
                lambda: dummy,
            )
            return acc + jax.tree.leaves(result)[0][idx]
        return jax.lax.fori_loop(0, N, body, 0.0)

    for _ in range(warmup):
        jax.block_until_ready(looped(*args))

    times = []
    for _ in range(rep):
        t0 = time.perf_counter()
        jax.block_until_ready(looped(*args))
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1e3 / N


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

CONFIGS = [
    {
        "name": "mamba2-130m",
        "batch": 1, "seqlen": 2048,
        "nheads": 24, "headdim": 64, "dstate": 128,
        "ngroups": 1, "chunk_size": 256,
    },
    {
        "name": "mamba2-2.7b",
        "batch": 1, "seqlen": 2048,
        "nheads": 80, "headdim": 64, "dstate": 128,
        "ngroups": 1, "chunk_size": 256,
    },
    {
        "name": "nemotron-h-4b (mamba2 layer)",
        "batch": 1, "seqlen": 2048,
        "nheads": 112, "headdim": 64, "dstate": 128,
        "ngroups": 8, "chunk_size": 256,
    },
    {
        "name": "nemotron-h-8b (mamba2 layer)",
        "batch": 1, "seqlen": 2048,
        "nheads": 128, "headdim": 64, "dstate": 128,
        "ngroups": 8, "chunk_size": 128,
    },
    {
        "name": "nemotron-h-8b (mamba2 layer) L=4096",
        "batch": 1, "seqlen": 4096,
        "nheads": 128, "headdim": 64, "dstate": 128,
        "ngroups": 8, "chunk_size": 128,
    },
    {
        "name": "nemotron-h-47b (mamba2 layer)",
        "batch": 1, "seqlen": 2048,
        "nheads": 256, "headdim": 64, "dstate": 256,
        "ngroups": 8, "chunk_size": 128,
    },
]


def benchmark_config(cfg):
    """Benchmark a single config. Returns dict of timings."""
    batch = cfg["batch"]
    seqlen = cfg["seqlen"]
    nheads = cfg["nheads"]
    headdim = cfg["headdim"]
    dstate = cfg["dstate"]
    ngroups = cfg["ngroups"]
    chunk_size = cfg["chunk_size"]

    key = jax.random.PRNGKey(7)
    keys = jax.random.split(key, 8)

    x_j = jax.random.normal(keys[0], (batch, seqlen, nheads, headdim)) * 0.1
    dt_j = jax.random.normal(keys[1], (batch, seqlen, nheads)) * 0.5
    A_j = -jax.random.uniform(keys[2], (nheads,), minval=0.01, maxval=0.1)
    B_j = jax.random.normal(keys[3], (batch, seqlen, ngroups, dstate)) * 0.1
    C_j = jax.random.normal(keys[4], (batch, seqlen, ngroups, dstate)) * 0.1
    dt_bias_j = jax.random.normal(keys[5], (nheads,)) * 0.1
    D_j = jax.random.normal(keys[6], (nheads,)) * 0.1
    z_j = jax.random.normal(keys[7], (batch, seqlen, nheads, headdim)) * 0.5
    dt_limit = (0.0, float("inf"))

    results = {"name": cfg["name"]}

    # --- Pallas (amortized) ---
    pallas_fn = partial(
        mamba_chunk_scan_combined_fwd,
        chunk_size=chunk_size,
        D=D_j, z=z_j, dt_bias=dt_bias_j,
        dt_softplus=True, dt_limit=dt_limit,
        return_final_states=True,
    )
    pallas_args = (x_j, dt_j, A_j, B_j, C_j)
    try:
        results["pallas_ms"] = _bench_pallas(pallas_fn, pallas_args)
    except Exception as e:
        results["pallas_ms"] = None
        print(f"    [warn] Pallas failed: {e}")

    # --- Naive JAX ---
    naive_fn = partial(
        mamba_chunk_scan_combined_naive,
        chunk_size=chunk_size,
        D=D_j, z=z_j, dt_bias=dt_bias_j,
        dt_softplus=True, dt_limit=dt_limit,
        return_final_states=True,
    )
    results["naive_ms"] = _bench_naive(naive_fn, pallas_args)

    # --- Triton ---
    results["triton_ms"] = None
    if _HAS_TRITON:
        x_t = _to_torch(x_j)
        dt_t = _to_torch(dt_j)
        A_t = _to_torch(A_j)
        B_t = _to_torch(B_j)
        C_t = _to_torch(C_j)
        dt_bias_t = _to_torch(dt_bias_j)
        D_t = _to_torch(D_j)
        z_t = _to_torch(z_j)

        def _triton_fn():
            _triton_combined_fwd(
                x_t, dt_t, A_t, B_t, C_t, chunk_size,
                D=D_t, z=z_t, dt_bias=dt_bias_t,
                dt_softplus=True, dt_limit=dt_limit,
            )

        results["triton_ms"] = _bench_triton(_triton_fn)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {jax.devices()[0]}")
    if _HAS_TRITON:
        print(f"Triton: available (mamba-ssm)")
    else:
        print(f"Triton: not available")
    print()

    print("=" * 80)
    print("SSD Combined Forward Pass — One Mamba2 Layer Benchmark")
    print("=" * 80)
    print()

    all_results = []
    for cfg in CONFIGS:
        print(f"Benchmarking: {cfg['name']}  "
              f"(B={cfg['batch']} L={cfg['seqlen']} H={cfg['nheads']} "
              f"P={cfg['headdim']} N={cfg['dstate']} G={cfg['ngroups']} "
              f"Q={cfg['chunk_size']})...")
        r = benchmark_config(cfg)
        all_results.append(r)

        pallas_str = f"{r['pallas_ms']:.3f} ms" if r['pallas_ms'] else "N/A"
        naive_str = f"{r['naive_ms']:.3f} ms"
        triton_str = f"{r['triton_ms']:.3f} ms" if r['triton_ms'] else "N/A"
        print(f"  Pallas Mosaic : {pallas_str}")
        print(f"  Naive JAX/XLA : {naive_str}")
        print(f"  Triton        : {triton_str}")

        if r['pallas_ms'] and r['triton_ms']:
            print(f"  Pallas/Triton : {r['pallas_ms']/r['triton_ms']:.2f}x")
        print()

    # Print markdown table
    print()
    print("Markdown table for README:")
    print()
    has_triton = any(r["triton_ms"] is not None for r in all_results)
    if has_triton:
        print("| Config | Pallas Mosaic | Triton (mamba-ssm) | Naive JAX/XLA | Pallas/Triton |")
        print("|--------|--------------|-------------------|--------------|--------------|")
        for r in all_results:
            p = f"{r['pallas_ms']:.2f} ms" if r['pallas_ms'] else "N/A"
            t = f"{r['triton_ms']:.2f} ms" if r['triton_ms'] else "N/A"
            n = f"{r['naive_ms']:.2f} ms"
            ratio = f"{r['pallas_ms']/r['triton_ms']:.2f}x" if r['pallas_ms'] and r['triton_ms'] else "—"
            print(f"| {r['name']} | {p} | {t} | {n} | {ratio} |")
    else:
        print("| Config | Pallas Mosaic | Naive JAX/XLA |")
        print("|--------|--------------|--------------|")
        for r in all_results:
            p = f"{r['pallas_ms']:.2f} ms" if r['pallas_ms'] else "N/A"
            n = f"{r['naive_ms']:.2f} ms"
            print(f"| {r['name']} | {p} | {n} |")


if __name__ == "__main__":
    main()
