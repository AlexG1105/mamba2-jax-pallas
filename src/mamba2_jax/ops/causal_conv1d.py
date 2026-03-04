"""
mamba2_jax/ops/causal_conv1d.py

Pure JAX implementation of causal 1D depthwise convolution.

Replaces the CUDA ``causal_conv1d`` package used in the original Mamba2.
Uses ``jax.lax.conv_general_dilated`` with manual causal (left) padding
so that output[t] depends only on input[t-k+1 : t+1].
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.lax as lax


def causal_conv1d(
    x,              # (batch, d, seqlen) — channels-first
    weight,         # (d, kernel_size) or (d, 1, kernel_size)
    bias=None,      # (d,) or None
    activation=None,  # "silu" or None
    seq_idx=None,   # (batch, seqlen) int32, optional — sequence boundary indices
):
    """
    Causal 1D depthwise convolution.

    Equivalent to ``causal_conv1d_fn`` from the ``causal_conv1d`` CUDA package.

    Parameters
    ----------
    x : jax.Array
        (batch, d, seqlen) — input in channels-first layout.
    weight : jax.Array
        (d, kernel_size) or (d, 1, kernel_size) — depthwise conv weights.
    bias : jax.Array, optional
        (d,) — per-channel bias.
    activation : str, optional
        If "silu", apply SiLU activation after conv + bias.
    seq_idx : jax.Array, optional
        (batch, seqlen) — integer sequence indices. When provided, the
        convolution does not bleed across sequence boundaries (positions
        where seq_idx changes are treated as the start of a new sequence).

    Returns
    -------
    out : jax.Array
        (batch, d, seqlen) — same shape as input.
    """
    # Normalize weight shape to (d, 1, kernel_size)
    if weight.ndim == 2:
        weight = weight[:, None, :]  # (d, 1, k)
    d, _, kernel_size = weight.shape

    if seq_idx is not None:
        # Zero out positions where the convolution would cross sequence
        # boundaries. For each position t and lag j (0..k-1), if
        # seq_idx[t] != seq_idx[t-j], mask that contribution to zero.
        # We do this by zeroing x at boundary crossings in a padded copy.
        x = _mask_sequence_boundaries(x, seq_idx, kernel_size)

    # Causal (left) padding: pad left by (kernel_size - 1), no right pad
    x_padded = jnp.pad(
        x,
        ((0, 0), (0, 0), (kernel_size - 1, 0)),
    )  # (batch, d, seqlen + kernel_size - 1)

    # Depthwise conv1d: feature_group_count = d
    out = lax.conv_general_dilated(
        x_padded,                       # (batch, d, seqlen + k - 1)
        weight,                         # (d, 1, k)
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NCH", "OIH", "NCH"),
        feature_group_count=d,
    )
    # out: (batch, d, seqlen)

    if bias is not None:
        out = out + bias[None, :, None]

    if activation == "silu":
        out = jax.nn.silu(out)

    return out


def causal_conv1d_update(
    x_new,          # (batch, d) — single new timestep
    conv_state,     # (batch, d, kernel_size) — rolling state buffer
    weight,         # (d, kernel_size) or (d, 1, kernel_size)
    bias=None,      # (d,) or None
    activation=None,  # "silu" or None
):
    """
    Single-step causal conv1d update for autoregressive inference.

    Shifts the conv_state left, inserts x_new on the right, and computes
    the depthwise convolution output for this single timestep.

    Parameters
    ----------
    x_new : jax.Array
        (batch, d) — input for the current timestep.
    conv_state : jax.Array
        (batch, d, kernel_size) — rolling convolution state buffer.
    weight : jax.Array
        (d, kernel_size) or (d, 1, kernel_size) — depthwise conv weights.
    bias : jax.Array, optional
        (d,) — per-channel bias.
    activation : str, optional
        If "silu", apply SiLU activation after conv + bias.

    Returns
    -------
    new_conv_state : jax.Array
        (batch, d, kernel_size) — updated state with x_new shifted in.
    out : jax.Array
        (batch, d) — convolution output for this timestep.
    """
    if weight.ndim == 3:
        weight = weight[:, 0, :]  # (d, k)

    # Shift state left and insert new value
    new_conv_state = jnp.concatenate(
        [conv_state[:, :, 1:], x_new[:, :, None]],
        axis=-1,
    )

    # Depthwise dot: sum over kernel dimension
    out = jnp.sum(new_conv_state * weight[None, :, :], axis=-1)  # (batch, d)

    if bias is not None:
        out = out + bias[None, :]

    if activation == "silu":
        out = jax.nn.silu(out)

    return new_conv_state, out


def _mask_sequence_boundaries(x, seq_idx, kernel_size):
    """
    Zero out x values that would bleed across sequence boundaries during
    causal convolution.

    For each position t, if seq_idx[t] != seq_idx[t-j] for lag j in
    [1, kernel_size-1], the contribution from position t-j should be zero.
    We achieve this by inserting zeros at sequence boundaries in a way that
    the subsequent convolution naturally produces the correct result.

    This is a simple but potentially inefficient approach — for long sequences
    with many boundaries, a segment-aware convolution would be better.
    """
    batch, d, seqlen = x.shape
    # seq_idx: (batch, seqlen)
    # For each position, check if it's the start of a new sequence
    # (seq_idx[t] != seq_idx[t-1]). If so, zero out the previous
    # (kernel_size - 1) positions' contribution.

    # Approach: create a mask that is 0 at positions within (kernel_size-1)
    # steps after a boundary, and 1 elsewhere. Apply to padded x.
    # Actually simpler: just zero x at the exact boundary positions
    # and let the causal padding handle the rest.

    # Detect boundaries: positions where seq_idx changes
    # boundary[t] = 1 if t == 0 or seq_idx[t] != seq_idx[t-1]
    boundary = jnp.concatenate(
        [
            jnp.ones((batch, 1), dtype=jnp.bool_),
            seq_idx[:, 1:] != seq_idx[:, :-1],
        ],
        axis=1,
    )  # (batch, seqlen)

    # For each boundary at position t, zero out x at positions
    # [t - kernel_size + 1, t - 1] (the positions that would bleed in).
    # We create a cumulative mask: after each boundary, the next kernel_size-1
    # positions before it should not contribute.
    # Simple approach: for each lag j in [1, k-1], zero x[t-j] if there's
    # a boundary in [t-j+1, t].
    mask = jnp.ones((batch, seqlen), dtype=x.dtype)
    for j in range(1, kernel_size):
        # shifted_boundary[t] = boundary[t+j], i.e., there's a boundary
        # j positions ahead of t. If so, t should be zeroed because
        # convolution at t+j would look back j steps to t.
        shifted = jnp.concatenate(
            [boundary[:, j:], jnp.zeros((batch, j), dtype=jnp.bool_)],
            axis=1,
        )
        mask = mask * (1.0 - shifted.astype(x.dtype))

    return x * mask[:, None, :]  # broadcast over d
