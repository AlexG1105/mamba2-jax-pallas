# mamba2-jax-pallas

Mamba2 Selective State Space Model in JAX, with Pallas Mosaic GPU kernels optimized for NVIDIA Hopper (H100/H200).

Supports loading pretrained HuggingFace models for inference:
- **Mamba2** (`state-spaces/mamba2-130m` through `mamba2-2.7b`)
- **Nemotron-H** (`nvidia/Nemotron-H-8B-Base-8K`) — hybrid Mamba2 + Attention + MLP

## Features

- **Hopper-optimized kernels**: Pallas Mosaic GPU kernels using WGMMA for the full SSD chunked scan pipeline (5 sub-kernels)
- **Universal fallback**: Pure JAX/XLA naive implementation that runs on any GPU (RTX 4090, A100, etc.)
- **Automatic dispatch**: Tries Pallas Mosaic kernels first, transparently falls back to XLA on non-Hopper hardware
- **HuggingFace weight loading**: Load pretrained Mamba2 and Nemotron-H models directly from HuggingFace
- **Inference modes**: Prefill (chunked scan) and autoregressive decode (single-step SSM update)
- **Text generation**: Greedy and top-k sampling with temperature

## Environment Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (Hopper for Pallas kernels, any CUDA GPU for XLA fallback)
- Conda (recommended) or pip

### Option A: Conda environment (recommended)

```bash
# Create environment
conda create -n mamba2-jax python=3.12 -y
conda activate mamba2-jax

# Install JAX with CUDA 13 support (for Hopper / Pallas Mosaic)
pip install jax[cuda13]>=0.9.0

# Install PyTorch with CUDA 12.9 (for weight loading)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# Install remaining dependencies
pip install -r requirements.txt

# Install this package in development mode
pip install -e .
```

### Option B: pip only

```bash
pip install jax[cuda13]>=0.9.0
pip install torch --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
pip install -e .
```

### Optional: mamba-ssm (for cross-framework comparison tests)

```bash
# Requires matching CUDA toolkit + PyTorch
pip install mamba-ssm causal-conv1d
```

### Verified environment

This project has been tested with the following versions:

| Package | Version |
|---------|---------|
| Python | 3.12 |
| JAX | 0.9.0 |
| jaxlib | 0.9.0 |
| Flax | 0.12.4 |
| PyTorch | 2.8.0+cu129 |
| CUDA (JAX) | 13.1 |
| CUDA (PyTorch) | 12.9 |
| NumPy | 2.3 |
| transformers | 5.3 |
| safetensors | 0.7 |
| einops | 0.8 |
| triton | 3.4 |

Tested on:
- **RTX 4090** (sm_89, Ada Lovelace) — XLA fallback path
- **H100** (sm_90, Hopper) — Pallas Mosaic kernels

### Dependencies overview

**Core (always required):**
- `jax[cuda13]`, `jaxlib`, `flax`, `numpy`, `scipy`, `ml-dtypes`, `einops`, `optax`, `chex`, `absl-py`

**Weight loading (required for pretrained models):**
- `torch` — PyTorch checkpoint deserialization
- `huggingface-hub` — Model download from HuggingFace Hub
- `safetensors` — Efficient tensor serialization (Nemotron-H uses this format)
- `transformers` — Tokenizer for text generation

**Optional:**
- `mamba-ssm`, `causal-conv1d` — For cross-framework comparison tests against the PyTorch reference

## Quick Start

### Load and generate with Mamba2

```bash
# Default: mamba2-130m (~500MB download)
python scripts/load_and_generate.py

# Specify model size
python scripts/load_and_generate.py --model mamba2-370m

# Custom prompt, greedy decoding
python scripts/load_and_generate.py --model mamba2-130m --prompt "The meaning of life is" --temperature 0

# Validate loading only (no generation)
python scripts/load_and_generate.py --model mamba2-130m --validate-only

# Use bfloat16 for larger models
python scripts/load_and_generate.py --model mamba2-2.7b --dtype bfloat16
```

Available Mamba2 sizes: `mamba2-130m`, `mamba2-370m`, `mamba2-780m`, `mamba2-1.3b`, `mamba2-2.7b`

### Load and validate Nemotron-H

```bash
# Load Nemotron-H-8B (~16GB download, requires ~16GB VRAM in bf16)
# Recommended for H100; tight fit on RTX 4090
python scripts/load_nemotron.py --validate-only --dtype bfloat16

# Generate text (H100 recommended)
python scripts/load_nemotron.py --dtype bfloat16 --prompt "The meaning of life is"
```

### Use as a library

```python
import jax
import jax.numpy as jnp
from mamba2_jax.models.mamba2_lm import load_from_pretrained

# Load pretrained model
model, params, config = load_from_pretrained("state-spaces/mamba2-130m")

# Forward pass
input_ids = jnp.array([[1, 2, 3, 4, 5]])
logits = model(params, input_ids)  # (1, 5, vocab_size)

# Generate
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
    mamba2_lm.py          # Full Mamba2 LM: embedding -> [norm -> mamba2] x N -> norm -> lm_head
    nemotron_h.py         # Nemotron-H hybrid: Mamba2 + Attention + MLP

  distributed/
    tensor_parallel.py    # ColumnParallelLinear, RowParallelLinear (JAX collectives)

scripts/
  load_and_generate.py  # Load Mamba2 from HuggingFace, run generation
  load_nemotron.py      # Load Nemotron-H from HuggingFace, run validation/generation

tests/
  test_mamba2_ops.py          # Unit tests for ops (conv1d, SSM update, RMSNorm)
  test_mamba2_vs_torch.py     # Self-consistency tests (prefill vs step-by-step decode)
  test_ssd_combined.py        # End-to-end SSD combined kernel tests
  test_chunk_cumsum_and_state.py
  test_state_passing_and_bmm.py
  test_chunk_scan_and_varlen.py
```

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

Layer pattern: `M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-`
(M=Mamba2, -=MLP, \*=Attention)

## Hardware Requirements

| GPU | Mamba2-130m | Mamba2-2.7b | Nemotron-H-8B |
|-----|-------------|-------------|---------------|
| RTX 4090 (24GB) | fp32 | bf16 | bf16 (validate only, tight fit) |
| H100 (80GB) | fp32/bf16 | fp32/bf16 | fp32/bf16 |

On Hopper GPUs (H100/H200), the Pallas Mosaic kernels are automatically used for the SSD scan, providing optimized performance via WGMMA. On all other GPUs, the pure JAX/XLA fallback is used transparently.

## Running Tests

```bash
# Ops unit tests (any GPU, pure JAX)
python tests/test_mamba2_ops.py

# Self-consistency tests (prefill vs step-by-step decode)
python tests/test_mamba2_vs_torch.py

# Pallas kernel tests (Hopper GPU required)
python tests/test_ssd_combined.py
python tests/test_chunk_cumsum_and_state.py
python tests/test_state_passing_and_bmm.py
python tests/test_chunk_scan_and_varlen.py
```

## Architecture Overview

The SSD (Structured State Space Duality) chunked scan decomposes the Mamba2 forward pass into 5 stages:

1. **chunk_cumsum** — Process dt (bias, softplus, clamp) and compute cumulative dA within chunks
2. **chunk_state** — Compute per-chunk SSM states via einsum with exponential decay
3. **state_passing** — Propagate states across chunks (weighted prefix sum)
4. **bmm_chunk** — Compute CB = C @ B^T within each chunk
5. **chunk_scan** — Final output combining intra-chunk (CB attention) and inter-chunk (state) contributions

The Pallas Mosaic kernels implement each stage using WGMMA (Warpgroup Matrix Multiply-Accumulate), SMEM pipelining, and tiled computation patterns specific to Hopper. The naive XLA fallback implements the same math using standard JAX operations (einsum, cumsum, exp, etc.).

## License

Apache 2.0
