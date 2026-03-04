"""
mamba2_jax/ops/rms_norm.py

Pure JAX implementation of RMSNorm with optional gating (RMSNormGated).

Matches the semantics of ``mamba_ssm.ops.triton.layernorm_gated.RMSNorm``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def rms_norm_gated(
    x,              # (..., hidden_size)
    weight,         # (hidden_size,)
    z=None,         # (..., hidden_size) or None
    eps=1e-5,
    group_size=None,
    norm_before_gate=True,
):
    """
    RMSNorm with optional SiLU gating.

    Parameters
    ----------
    x : jax.Array
        (..., hidden_size) — input to normalize.
    weight : jax.Array
        (hidden_size,) — learnable scale parameter.
    z : jax.Array, optional
        (..., hidden_size) — gating value. If provided, output is gated
        with SiLU(z).
    eps : float
        Epsilon for numerical stability in the RMS computation.
    group_size : int, optional
        If provided, RMS is computed over groups of this size instead of
        the full hidden dimension. ``hidden_size`` must be divisible by
        ``group_size``. When None, equivalent to ``group_size = hidden_size``.
    norm_before_gate : bool
        Only relevant when ``z`` is not None.
        - True:  ``output = RMSNorm(x) * SiLU(z)``
        - False: ``output = RMSNorm(x * SiLU(z))``

    Returns
    -------
    out : jax.Array
        Same shape as x.
    """
    hidden_size = x.shape[-1]

    if group_size is None or group_size == hidden_size:
        # Standard RMSNorm over the full hidden dim
        if z is not None and not norm_before_gate:
            x = x * jax.nn.silu(z)
            z_for_gate = None
        else:
            z_for_gate = z

        rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
        out = x / rms * weight

        if z_for_gate is not None:
            out = out * jax.nn.silu(z_for_gate)
    else:
        # Grouped RMSNorm: reshape to (..., ngroups, group_size)
        assert hidden_size % group_size == 0
        ngroups = hidden_size // group_size
        orig_shape = x.shape

        x = x.reshape(*orig_shape[:-1], ngroups, group_size)

        if z is not None and not norm_before_gate:
            z_reshaped = z.reshape(*orig_shape[:-1], ngroups, group_size)
            x = x * jax.nn.silu(z_reshaped)
            z_for_gate = None
        else:
            z_for_gate = z

        rms = jnp.sqrt(
            jnp.mean(x * x, axis=-1, keepdims=True) + eps
        )  # (..., ngroups, 1)
        x_normed = x / rms
        x_normed = x_normed.reshape(orig_shape)

        out = x_normed * weight

        if z_for_gate is not None:
            out = out * jax.nn.silu(z_for_gate)

    return out
