# mamba2-jax-pallas

Mamba2 Selective State Space Model in JAX, with Pallas Mosaic GPU kernels optimized for NVIDIA Hopper (H100/H200).

Supports loading pretrained HuggingFace models for inference:
- **Mamba2** (`state-spaces/mamba2-130m` through `mamba2-2.7b`)
- **Nemotron-H** (`nvidia/Nemotron-H-8B-Base-8K`) — hybrid Mamba2 + Attention + MLP

## Installation

```bash
conda create -n mamba2-jax python=3.12 -y && conda activate mamba2-jax
pip install torch --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
```

Optional, for Triton comparison benchmarks:
```bash
pip install mamba-ssm causal-conv1d
```

## Benchmarks

Single Mamba2 SSD layer forward pass on H100 (batch=1, fp32, amortized GPU time):

| Config | Pallas Mosaic | Triton (mamba-ssm) | Naive JAX/XLA | Pallas/Triton |
|--------|--------------|-------------------|--------------|--------------|
| mamba2-130m (H=24, G=1, Q=256) | 0.11 ms | 0.35 ms | 0.42 ms | 0.33x |
| mamba2-2.7b (H=80, G=1, Q=256) | 0.26 ms | 0.34 ms | 1.23 ms | 0.77x |
| nemotron-h (H=128, G=8, Q=128) | 0.51 ms | 0.36 ms | 2.45 ms | 1.40x |
| nemotron-h L=4096 | 0.94 ms | 0.70 ms | 4.66 ms | 1.35x |

All configs use seqlen=2048 (except last), P=64 (headdim), N=128 (d_state). All timings are amortized (dispatch overhead excluded). Pallas/Triton < 1.0 means Pallas is faster.

Reproduce with:
```bash
python benchmarks/bench_model_forward.py
```

## Quick Start

### Load and generate with Mamba2

```bash
python scripts/load_and_generate.py --model mamba2-130m --prompt "The meaning of life is"
python scripts/load_and_generate.py --model mamba2-2.7b --dtype bfloat16
```

Available sizes: `mamba2-130m`, `mamba2-370m`, `mamba2-780m`, `mamba2-1.3b`, `mamba2-2.7b`

### Load and generate with Nemotron-H

```bash
python scripts/load_nemotron.py --model nemotron-h-4b --dtype bfloat16 --prompt "The meaning of life is"
python scripts/load_nemotron.py --model nemotron-h-8b --dtype bfloat16 --prompt "The meaning of life is"
python scripts/load_nemotron.py --model nemotron-h-8b-reasoning --dtype bfloat16
python scripts/load_nemotron.py --validate-only --dtype bfloat16
```

Available variants: `nemotron-h-4b`, `nemotron-h-8b`, `nemotron-h-8b-reasoning` (or pass any HuggingFace model ID directly)

### Use as a library

```python
import jax
import jax.numpy as jnp
from mamba2_jax.models.mamba2_lm import load_from_pretrained

model, params, config = load_from_pretrained("state-spaces/mamba2-130m")

input_ids = jnp.array([[1, 2, 3, 4, 5]])
logits = model(params, input_ids)  # (1, 5, vocab_size)

rng = jax.random.PRNGKey(0)
output_ids = model.generate(params, input_ids, max_new_tokens=50, temperature=0.7, rng_key=rng)
```

## Project Structure

```
src/mamba2_jax/
  kernels/              # Pallas Mosaic GPU kernels (Hopper-only)
    chunk_cumsum_fwd.py   # dt processing + dA cumulative sum
    chunk_state_fwd.py    # Per-chunk SSM state computation
    state_passing_fwd.py  # Inter-chunk state propagation (prefix scan)
    bmm_chunk_fwd.py      # CB = C @ B^T within each chunk
    chunk_scan_fwd.py     # Final output: intra-chunk + inter-chunk
    ssd_combined.py       # Full SSD pipeline composition

  ops/                  # Pure JAX operations (any GPU)
    ssd_naive.py          # Naive XLA implementation of all 5 SSD sub-kernels
    causal_conv1d.py      # Causal 1D depthwise convolution
    selective_state_update.py  # Single-step SSM recurrence
    rms_norm.py           # RMSNorm with optional gating

  modules/
    mamba2.py             # Mamba2 Flax module (matches mamba_ssm.modules.mamba2)

  models/
    mamba2_lm.py          # Full Mamba2 LM with JIT-compiled autoregressive decode
    nemotron_h.py         # Nemotron-H hybrid: Mamba2 + Attention + MLP

scripts/
  load_and_generate.py  # Load Mamba2 from HuggingFace, run generation
  load_nemotron.py      # Load Nemotron-H from HuggingFace, run validation/generation

benchmarks/
  bench_model_forward.py  # Pallas vs Triton vs Naive JAX benchmark

tests/
  test_ssd_combined.py        # End-to-end SSD combined kernel tests + benchmarks
  test_mamba2_ops.py          # Unit tests for ops (conv1d, SSM update, RMSNorm)
  test_mamba2_vs_torch.py     # Self-consistency tests (prefill vs decode)
```

## Architecture

The SSD (Structured State Space Duality) chunked scan decomposes the Mamba2 forward pass into 5 stages:

1. **chunk_cumsum** — Process dt (bias, softplus, clamp) and compute cumulative dA within chunks
2. **chunk_state** — Compute per-chunk SSM states via einsum with exponential decay
3. **state_passing** — Propagate states across chunks (weighted prefix sum)
4. **bmm_chunk** — Compute CB = C @ B^T within each chunk
5. **chunk_scan** — Final output combining intra-chunk (CB attention) and inter-chunk (state) contributions

The Pallas Mosaic kernels implement each stage using WGMMA (Warpgroup Matrix Multiply-Accumulate), SMEM pipelining, and tiled computation patterns specific to Hopper. The naive XLA fallback implements the same math using standard JAX operations (einsum, cumsum, exp, etc.).

On Hopper GPUs (H100/H200), Pallas Mosaic kernels are automatically used. On all other GPUs (RTX 4090, A100, etc.), the pure JAX/XLA fallback is used transparently.

## Supported Models

### Mamba2 (pure SSM)

| Model | Params | d_model | Layers | nheads | ngroups |
|-------|--------|---------|--------|--------|---------|
| mamba2-130m | 168M | 768 | 24 | 24 | 1 |
| mamba2-370m | 370M | 1024 | 48 | 32 | 1 |
| mamba2-780m | 780M | 1536 | 48 | 48 | 1 |
| mamba2-1.3b | 1.3B | 2048 | 48 | 64 | 1 |
| mamba2-2.7b | 2.7B | 2560 | 64 | 80 | 1 |

All use: d_state=128, headdim=64, d_conv=4, chunk_size=256, tied embeddings, vocab=50277.

### Nemotron-H (hybrid)

| Model | Params | d_model | Layers | Pattern |
|-------|--------|---------|--------|---------|
| Nemotron-H-8B | 8.1B | 4096 | 52 | 24 Mamba2 + 24 MLP + 4 Attention |

Mamba2 blocks use ngroups=8, chunk_size=128. Attention uses GQA (32Q/8KV heads). MLP uses squared ReLU. Vocab=131072, untied embeddings.

## Running Tests

```bash
python tests/test_mamba2_ops.py            # Ops unit tests (any GPU)
python tests/test_mamba2_vs_torch.py       # Prefill vs decode consistency
python tests/test_ssd_combined.py          # Pallas kernel correctness + benchmarks
```

## License

Apache 2.0
