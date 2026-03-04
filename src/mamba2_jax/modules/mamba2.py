"""
mamba2_jax/modules/mamba2.py

JAX/Flax implementation of the Mamba2 module.

Mirrors ``mamba_ssm.modules.mamba2.Mamba2`` from the official PyTorch
implementation. Supports both:
  - **Chunked scan** mode (sequence processing via Pallas Mosaic GPU kernels)
  - **Step** mode (single-token autoregressive inference)

Key differences from the PyTorch original:
  - Uses Flax ``nn.Module`` instead of ``torch.nn.Module``.
  - Causal conv1d is implemented in pure JAX (no CUDA kernel).
  - ``selective_state_update`` is implemented in pure JAX.
  - No backward pass (forward / inference only for now).
  - The fused ``mamba_split_conv1d_scan_combined`` path is NOT available;
    we always use the unfused (modular) path.

Usage
-----
  from mamba2_jax.modules.mamba2 import Mamba2

  model = Mamba2(d_model=2048)
  variables = model.init(rng, jnp.zeros((1, 128, 2048)))
  out = model.apply(variables, x)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from mamba2_jax.ops.ssd_naive import mamba_chunk_scan_combined_naive
from mamba2_jax.ops.selective_state_update import selective_state_update
from mamba2_jax.ops.causal_conv1d import causal_conv1d, causal_conv1d_update
from mamba2_jax.ops.rms_norm import rms_norm_gated

# Try to import Pallas Mosaic kernels (Hopper-only). Fall back to naive XLA.
try:
    from mamba2_jax.kernels.ssd_combined import mamba_chunk_scan_combined_fwd as _pallas_fwd
    _HAS_PALLAS = True
except ImportError:
    _HAS_PALLAS = False


_PALLAS_FALLBACK_WARNED = False


def _ssd_combined_fwd(use_pallas: bool = True, **kwargs):
    """Dispatch to Pallas kernels or naive JAX fallback.

    On Hopper (H100/H200), Pallas Mosaic kernels are used for performance.
    On non-Hopper GPUs (RTX 4090, A100, etc.), falls back to naive XLA.
    """
    global _PALLAS_FALLBACK_WARNED

    if use_pallas and _HAS_PALLAS and not _PALLAS_FALLBACK_WARNED:
        try:
            return _pallas_fwd(**kwargs)
        except (NotImplementedError, RuntimeError, Exception) as e:
            # NotImplementedError: WGMMA swizzle mismatch (trace-time)
            # RuntimeError/JaxRuntimeError: WGMMA not supported on sm_XX (lowering)
            _PALLAS_FALLBACK_WARNED = True
            import warnings
            warnings.warn(
                f"Pallas Mosaic kernels failed ({type(e).__name__}). "
                "Falling back to naive JAX/XLA. This is expected on non-Hopper GPUs.",
                stacklevel=2,
            )
    return mamba_chunk_scan_combined_naive(**kwargs)


# ---------------------------------------------------------------------------
# Mamba2 module
# ---------------------------------------------------------------------------

class Mamba2(nn.Module):
    """
    Mamba2 Selective State Space Model block (Flax module).

    This is a single Mamba2 layer that can be stacked to form a full model.

    Attributes
    ----------
    d_model : int
        Input / output hidden dimension.
    d_state : int
        SSM state dimension (N). Default 128.
    d_conv : int
        Causal convolution kernel width. Default 4.
    conv_init : float or None
        If set, initialize conv weights uniformly in [-conv_init, conv_init].
    expand : int
        Expansion factor. ``d_inner = expand * d_model``. Default 2.
    headdim : int
        Per-head dimension. Default 64.
    d_ssm : int or None
        Number of dims that go through the SSM. If None, equals d_inner.
        When d_ssm < d_inner, the remaining dims form a gated MLP bypass.
    ngroups : int
        Number of groups for B and C (like grouped-query attention). Default 1.
    A_init_range : tuple
        Uniform range for A initialization (before log). Default (1, 16).
    D_has_hdim : bool
        If True, D has shape (d_ssm,); if False, D has shape (nheads,).
    rmsnorm : bool
        Apply RMSNormGated before output projection. Default True.
    norm_before_gate : bool
        RMSNorm gating order. Default False.
    dt_min : float
        Min of uniform range for dt initialization. Default 0.001.
    dt_max : float
        Max of uniform range for dt initialization. Default 0.1.
    dt_init_floor : float
        Floor clamp for dt before inverse-softplus. Default 1e-4.
    dt_limit : tuple
        Runtime dt clamping limits. Default (0.0, inf).
    bias : bool
        Bias for in_proj and out_proj. Default False.
    conv_bias : bool
        Bias for conv1d. Default True.
    chunk_size : int
        Chunk size for SSD scan. Default 256.
    use_mem_eff_path : bool
        Ignored (fused path not available in JAX). Kept for API compat.
    param_dtype : jnp.dtype
        Parameter storage dtype. Default float32.
    """
    d_model: int
    d_state: int = 128
    d_conv: int = 4
    conv_init: Optional[float] = None
    expand: int = 2
    headdim: int = 64
    d_ssm: Optional[int] = None
    ngroups: int = 1
    A_init_range: Tuple[float, float] = (1.0, 16.0)
    D_has_hdim: bool = False
    rmsnorm: bool = True
    norm_before_gate: bool = False
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    dt_limit: Tuple[float, float] = (0.0, float("inf"))
    bias: bool = False
    conv_bias: bool = True
    chunk_size: int = 256
    use_mem_eff_path: bool = True
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        """Compute derived dimensions (mirrors PyTorch __init__)."""
        self.d_inner = self.expand * self.d_model
        self._d_ssm = self.d_inner if self.d_ssm is None else self.d_ssm
        self.d_mlp = self.d_inner - self._d_ssm
        self.nheads = self._d_ssm // self.headdim
        assert self._d_ssm % self.headdim == 0, (
            f"d_ssm ({self._d_ssm}) must be divisible by headdim ({self.headdim})"
        )
        assert self.nheads % self.ngroups == 0, (
            f"nheads ({self.nheads}) must be divisible by ngroups ({self.ngroups})"
        )

        # Convolution channels: x + B + C concatenated
        self.conv_dim = self._d_ssm + 2 * self.ngroups * self.d_state

        # Input projection total output dim
        self.d_in_proj = (
            2 * self.d_inner
            + 2 * self.ngroups * self.d_state
            + self.nheads
        )

    @nn.compact
    def __call__(
        self,
        u,                          # (batch, seqlen, d_model)
        seq_idx=None,               # (batch, seqlen) int32, optional
        cu_seqlens=None,            # (num_seqs + 1,) int32, optional
        inference_params=None,      # dict with conv_state, ssm_state, seqlen_offset
    ):
        """
        Forward pass of the Mamba2 block.

        Parameters
        ----------
        u : jax.Array
            (batch, seqlen, d_model) — input hidden states.
        seq_idx : jax.Array, optional
            (batch, seqlen) — sequence boundary indices for packed sequences.
        cu_seqlens : jax.Array, optional
            (num_seqs + 1,) — cumulative sequence lengths for variable-length.
        inference_params : dict, optional
            If provided, should contain:
            - "conv_state": (batch, conv_dim, d_conv) — rolling conv buffer
            - "ssm_state": (batch, nheads, headdim, dstate) — SSM state
            - "seqlen_offset": int — 0 for prefill, >0 for decode steps
            Returns updated states alongside the output.

        Returns
        -------
        out : jax.Array
            (batch, seqlen, d_model)
        new_inference_params : dict or None
            Updated conv_state and ssm_state (only when inference_params given).
        """
        batch, seqlen, _ = u.shape

        # ---- Learnable parameters ----
        in_proj_kernel = self.param(
            "in_proj_kernel",
            nn.initializers.lecun_normal(),
            (self.d_model, self.d_in_proj),
            self.param_dtype,
        )
        in_proj_bias = None
        if self.bias:
            in_proj_bias = self.param(
                "in_proj_bias",
                nn.initializers.zeros,
                (self.d_in_proj,),
                self.param_dtype,
            )

        out_proj_kernel = self.param(
            "out_proj_kernel",
            nn.initializers.lecun_normal(),
            (self.d_inner, self.d_model),
            self.param_dtype,
        )
        out_proj_bias = None
        if self.bias:
            out_proj_bias = self.param(
                "out_proj_bias",
                nn.initializers.zeros,
                (self.d_model,),
                self.param_dtype,
            )

        # Conv1d: depthwise, weight shape (conv_dim, d_conv)
        if self.conv_init is not None:
            conv_init = nn.initializers.uniform(self.conv_init)
        else:
            conv_init = nn.initializers.lecun_normal()
        conv1d_weight = self.param(
            "conv1d_weight",
            conv_init,
            (self.conv_dim, self.d_conv),
            self.param_dtype,
        )
        conv1d_bias = None
        if self.conv_bias:
            conv1d_bias = self.param(
                "conv1d_bias",
                nn.initializers.zeros,
                (self.conv_dim,),
                self.param_dtype,
            )

        # dt_bias: initialized via inverse-softplus of log-uniform samples
        dt_bias = self.param(
            "dt_bias",
            _dt_bias_init(self.dt_min, self.dt_max, self.dt_init_floor),
            (self.nheads,),
            jnp.float32,
        )

        # A_log: log of uniform samples from A_init_range
        A_log = self.param(
            "A_log",
            _A_log_init(self.A_init_range),
            (self.nheads,),
            self.param_dtype,
        )

        # D: skip connection, init to ones
        D_shape = (self._d_ssm,) if self.D_has_hdim else (self.nheads,)
        D = self.param("D", nn.initializers.ones, D_shape, self.param_dtype)

        # RMSNorm weight
        if self.rmsnorm:
            norm_weight = self.param(
                "norm_weight",
                nn.initializers.ones,
                (self._d_ssm,),
                self.param_dtype,
            )

        # ---- Compute A (always negative) ----
        A = -jnp.exp(A_log.astype(jnp.float32))

        # ---- Check for inference step mode ----
        if inference_params is not None and inference_params.get("seqlen_offset", 0) > 0:
            return self._step(
                u, inference_params,
                in_proj_kernel, in_proj_bias,
                conv1d_weight, conv1d_bias,
                dt_bias, A, D,
                out_proj_kernel, out_proj_bias,
                norm_weight if self.rmsnorm else None,
            )

        # ================================================================
        # Chunked scan forward (prefill or full sequence processing)
        # ================================================================

        # 1. Input projection
        zxbcdt = u @ in_proj_kernel  # (batch, seqlen, d_in_proj)
        if in_proj_bias is not None:
            zxbcdt = zxbcdt + in_proj_bias

        # 2. Split: [z0, x0, z, xBC, dt]
        d_mlp = self.d_mlp
        d_ssm = self._d_ssm
        split_sizes = [
            d_mlp,                                  # z0 (MLP gate)
            d_mlp,                                  # x0 (MLP input)
            d_ssm,                                  # z  (SSM gate)
            d_ssm + 2 * self.ngroups * self.d_state,  # xBC (conv input)
            self.nheads,                            # dt
        ]
        z0, x0, z, xBC, dt = _split_last(zxbcdt, split_sizes)

        # 3. Update conv_state for inference cache (prefill)
        new_conv_state = None
        if inference_params is not None:
            # Store the last d_conv timesteps of xBC for future decode steps
            # xBC: (batch, seqlen, conv_dim) -> need (batch, conv_dim, d_conv)
            xBC_T = jnp.transpose(xBC, (0, 2, 1))  # (batch, conv_dim, seqlen)
            if seqlen >= self.d_conv:
                new_conv_state = xBC_T[:, :, -self.d_conv:]
            else:
                new_conv_state = jnp.pad(
                    xBC_T,
                    ((0, 0), (0, 0), (self.d_conv - seqlen, 0)),
                )

        # 4. Causal conv1d on xBC
        xBC_conv = causal_conv1d(
            jnp.transpose(xBC, (0, 2, 1)),   # (batch, conv_dim, seqlen)
            conv1d_weight,                     # (conv_dim, d_conv)
            bias=conv1d_bias,
            activation="silu",
            seq_idx=seq_idx,
        )
        xBC = jnp.transpose(xBC_conv, (0, 2, 1))  # (batch, seqlen, conv_dim)

        # 5. Split post-conv: [x, B, C]
        x = xBC[:, :, :d_ssm]
        B = xBC[:, :, d_ssm:d_ssm + self.ngroups * self.d_state]
        C = xBC[:, :, d_ssm + self.ngroups * self.d_state:]

        # Reshape for scan kernel
        x = x.reshape(batch, seqlen, self.nheads, self.headdim)
        B = B.reshape(batch, seqlen, self.ngroups, self.d_state)
        C = C.reshape(batch, seqlen, self.ngroups, self.d_state)

        # D reshaping
        if self.D_has_hdim:
            D_scan = D.reshape(self.nheads, self.headdim)
        else:
            D_scan = D

        # z handling: if rmsnorm, don't pass z to scan (gating is in norm);
        #             if no rmsnorm, pass z to scan for internal gating
        if self.rmsnorm:
            z_scan = None
        else:
            z_scan = z.reshape(batch, seqlen, self.nheads, self.headdim)

        # 6. SSD chunk scan (Pallas Mosaic on Hopper, naive XLA otherwise)
        return_final = inference_params is not None
        scan_result = _ssd_combined_fwd(
            x=x, dt=dt, A=A, B=B, C=C,
            chunk_size=self.chunk_size,
            D=D_scan,
            z=z_scan,
            dt_bias=dt_bias,
            dt_softplus=True,
            dt_limit=self.dt_limit,
            seq_idx=seq_idx,
            cu_seqlens=cu_seqlens,
            return_final_states=return_final,
            return_varlen_states=False,
        )

        if return_final:
            y, final_states = scan_result
        else:
            y = scan_result
            final_states = None

        # y: (batch, seqlen, nheads, headdim) -> (batch, seqlen, d_ssm)
        y = y.reshape(batch, seqlen, self._d_ssm)

        # 7. RMSNorm gating (if enabled)
        if self.rmsnorm:
            y = rms_norm_gated(
                y, norm_weight, z=z,
                group_size=self._d_ssm // self.ngroups,
                norm_before_gate=self.norm_before_gate,
            )

        # 8. Gated MLP bypass (if d_mlp > 0)
        if d_mlp > 0:
            y = jnp.concatenate([jax.nn.silu(z0) * x0, y], axis=-1)

        # 9. Output projection
        out = y @ out_proj_kernel
        if out_proj_bias is not None:
            out = out + out_proj_bias

        # Return
        if inference_params is not None:
            new_params = {
                "conv_state": new_conv_state,
                "ssm_state": final_states,
                "seqlen_offset": inference_params.get("seqlen_offset", 0) + seqlen,
            }
            return out, new_params
        return out

    def _step(
        self,
        u,                      # (batch, 1, d_model) — single token
        inference_params,       # dict with conv_state, ssm_state
        in_proj_kernel,
        in_proj_bias,
        conv1d_weight,
        conv1d_bias,
        dt_bias,
        A,                      # (nheads,) already negated
        D,
        out_proj_kernel,
        out_proj_bias,
        norm_weight,
    ):
        """
        Single-step inference (autoregressive decoding).

        Parameters
        ----------
        u : (batch, 1, d_model)
        inference_params : dict
            "conv_state": (batch, conv_dim, d_conv)
            "ssm_state":  (batch, nheads, headdim, dstate)

        Returns
        -------
        out : (batch, 1, d_model)
        new_inference_params : dict
        """
        conv_state = inference_params["conv_state"]
        ssm_state = inference_params["ssm_state"]

        d_ssm = self._d_ssm
        d_mlp = self.d_mlp

        # 1. Project (squeeze seq dim)
        hidden = u[:, 0, :]  # (batch, d_model)
        zxbcdt = hidden @ in_proj_kernel
        if in_proj_bias is not None:
            zxbcdt = zxbcdt + in_proj_bias
        # zxbcdt: (batch, d_in_proj)

        # 2. Split
        split_sizes = [
            d_mlp, d_mlp, d_ssm,
            d_ssm + 2 * self.ngroups * self.d_state,
            self.nheads,
        ]
        z0, x0, z, xBC, dt = _split_last(zxbcdt, split_sizes)

        # 3. Conv update (shift-register)
        new_conv_state, xBC = causal_conv1d_update(
            xBC,            # (batch, conv_dim)
            conv_state,     # (batch, conv_dim, d_conv)
            conv1d_weight,  # (conv_dim, d_conv)
            bias=conv1d_bias,
            activation="silu",
        )

        # 4. Split post-conv
        x = xBC[:, :d_ssm]                                         # (batch, d_ssm)
        B = xBC[:, d_ssm:d_ssm + self.ngroups * self.d_state]      # (batch, ngroups*dstate)
        C = xBC[:, d_ssm + self.ngroups * self.d_state:]            # (batch, ngroups*dstate)

        # Reshape for SSM update
        x = x.reshape(-1, self.nheads, self.headdim)                # (batch, nheads, headdim)
        B = B.reshape(-1, self.ngroups, self.d_state)               # (batch, ngroups, dstate)
        C = C.reshape(-1, self.ngroups, self.d_state)               # (batch, ngroups, dstate)

        # D reshaping for selective_state_update
        if self.D_has_hdim:
            D_step = D.reshape(self.nheads, self.headdim)
        else:
            D_step = D  # (nheads,)

        # z for gating (only used if not rmsnorm)
        if not self.rmsnorm:
            z_step = z.reshape(-1, self.nheads, self.headdim)
        else:
            z_step = None

        # 5. SSM state update
        new_ssm_state, y = selective_state_update(
            ssm_state,      # (batch, nheads, headdim, dstate)
            x,              # (batch, nheads, headdim)
            dt,             # (batch, nheads)
            A,              # (nheads,) — already negated
            B,              # (batch, ngroups, dstate)
            C,              # (batch, ngroups, dstate)
            D=D_step,
            z=z_step,
            dt_bias=dt_bias,
            dt_softplus=True,
        )
        # y: (batch, nheads, headdim)

        # 6. Flatten heads
        y = y.reshape(-1, d_ssm)  # (batch, d_ssm)

        # 7. RMSNorm gating
        if self.rmsnorm and norm_weight is not None:
            y = rms_norm_gated(
                y, norm_weight, z=z,
                group_size=d_ssm // self.ngroups,
                norm_before_gate=self.norm_before_gate,
            )

        # 8. Gated MLP bypass
        if d_mlp > 0:
            y = jnp.concatenate([jax.nn.silu(z0) * x0, y], axis=-1)

        # 9. Output projection
        out = y @ out_proj_kernel
        if out_proj_bias is not None:
            out = out + out_proj_bias

        out = out[:, None, :]  # (batch, 1, d_model)

        new_params = {
            "conv_state": new_conv_state,
            "ssm_state": new_ssm_state,
            "seqlen_offset": inference_params["seqlen_offset"] + 1,
        }
        return out, new_params


# ---------------------------------------------------------------------------
# Helper: allocate inference cache
# ---------------------------------------------------------------------------

def allocate_inference_cache(
    batch_size: int,
    d_model: int,
    d_state: int = 128,
    d_conv: int = 4,
    expand: int = 2,
    headdim: int = 64,
    d_ssm: Optional[int] = None,
    ngroups: int = 1,
    dtype=jnp.float32,
):
    """
    Allocate initial inference state buffers.

    Returns
    -------
    inference_params : dict
        "conv_state": (batch_size, conv_dim, d_conv) of zeros
        "ssm_state":  (batch_size, nheads, headdim, d_state) of zeros
        "seqlen_offset": 0
    """
    d_inner = expand * d_model
    if d_ssm is None:
        d_ssm = d_inner
    nheads = d_ssm // headdim
    conv_dim = d_ssm + 2 * ngroups * d_state

    conv_state = jnp.zeros((batch_size, conv_dim, d_conv), dtype=dtype)
    ssm_state = jnp.zeros((batch_size, nheads, headdim, d_state), dtype=dtype)

    return {
        "conv_state": conv_state,
        "ssm_state": ssm_state,
        "seqlen_offset": 0,
    }


# ---------------------------------------------------------------------------
# Parameter initializers (matching PyTorch Mamba2)
# ---------------------------------------------------------------------------

def _dt_bias_init(dt_min, dt_max, dt_init_floor):
    """
    Initialize dt_bias via inverse-softplus of log-uniform samples.

    In PyTorch:
        dt = exp(uniform(0,1) * (log(dt_max) - log(dt_min)) + log(dt_min))
        dt = clamp(dt, min=dt_init_floor)
        inv_dt = dt + log(-expm1(-dt))   # inverse softplus
    """
    def init(key, shape, dtype=jnp.float32):
        u = jax.random.uniform(key, shape, dtype=jnp.float32)
        dt = jnp.exp(
            u * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        dt = jnp.clip(dt, a_min=dt_init_floor)
        # Inverse softplus: softplus_inv(x) = x + log(1 - exp(-x))
        inv_dt = dt + jnp.log(-jnp.expm1(-dt))
        return inv_dt.astype(dtype)
    return init


def _A_log_init(A_init_range):
    """
    Initialize A_log: log of uniform samples from A_init_range.

    In PyTorch:
        A = uniform(A_init_range[0], A_init_range[1])
        A_log = log(A)
    """
    def init(key, shape, dtype=jnp.float32):
        A = jax.random.uniform(
            key, shape, dtype=jnp.float32,
            minval=A_init_range[0], maxval=A_init_range[1],
        )
        return jnp.log(A).astype(dtype)
    return init


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _split_last(x, sizes):
    """Split x along the last dimension into chunks of given sizes."""
    splits = []
    start = 0
    for s in sizes:
        if s > 0:
            splits.append(x[..., start:start + s])
        else:
            # Empty slice for d_mlp=0 case
            splits.append(x[..., :0])
        start += s
    return splits
