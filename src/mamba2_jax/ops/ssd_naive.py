"""
mamba2_jax/ops/ssd_naive.py

Pure JAX/XLA implementation of the Mamba2 SSD chunked scan forward pass.

This is a reference implementation that works on ANY GPU (including RTX 4090,
Ada Lovelace, etc.). It does NOT require Hopper (H100/H200) hardware.

For optimal performance on Hopper GPUs, use the Pallas Mosaic kernels in
``mamba2_jax.kernels`` instead.

Matches the semantics of ``ssd_chunk_scan_combined_ref`` from
``mamba_ssm.ops.triton.ssd_combined``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# 1. chunk_cumsum_fwd — dt processing + dA_cumsum
# ---------------------------------------------------------------------------

def chunk_cumsum_naive(
    dt,             # (batch, seqlen, nheads)
    A,              # (nheads,)
    chunk_size: int,
    dt_bias=None,   # (nheads,) or None
    dt_softplus: bool = False,
    dt_limit: tuple = (0.0, float("inf")),
):
    """
    Process dt and compute dA cumulative sum within each chunk.

    Returns
    -------
    dA_cumsum : (batch, nheads, nchunks, chunk_size) float32
    dt_out    : (batch, nheads, nchunks, chunk_size) float32
    """
    batch, seqlen, nheads = dt.shape
    nchunks = (seqlen + chunk_size - 1) // chunk_size

    # Pad seqlen to multiple of chunk_size
    if seqlen % chunk_size != 0:
        pad_len = nchunks * chunk_size - seqlen
        dt = jnp.pad(dt, ((0, 0), (0, pad_len), (0, 0)))

    # Reshape: (batch, seqlen, nheads) -> (batch, nheads, nchunks, chunk_size)
    dt = dt.reshape(batch, nchunks, chunk_size, nheads)
    dt = jnp.transpose(dt, (0, 3, 1, 2))  # (batch, nheads, nchunks, chunk_size)
    dt = dt.astype(jnp.float32)

    # Add bias
    if dt_bias is not None:
        dt = dt + dt_bias[None, :, None, None]

    # Softplus
    if dt_softplus:
        dt = jax.nn.softplus(dt)

    # Clamp
    dt = jnp.clip(dt, a_min=dt_limit[0], a_max=dt_limit[1])

    # dA = dt * A, cumsum along chunk positions
    dA = dt * A[None, :, None, None]
    dA_cumsum = jnp.cumsum(dA, axis=-1)

    return dA_cumsum, dt


# ---------------------------------------------------------------------------
# 2. chunk_state_naive — per-chunk SSM states
# ---------------------------------------------------------------------------

def chunk_state_naive(
    B,              # (batch, seqlen, ngroups, dstate)
    x,              # (batch, seqlen, nheads, headdim)
    dt,             # (batch, nheads, nchunks, chunk_size)
    dA_cumsum,      # (batch, nheads, nchunks, chunk_size)
):
    """
    Compute per-chunk SSM states.

    states[c] = sum_l decay(l->end) * dt[l] * outer(x[l], B[l])

    Returns
    -------
    states : (batch, nchunks, nheads, headdim, dstate)
    """
    batch, seqlen, nheads, headdim = x.shape
    ngroups = B.shape[2]
    dstate = B.shape[3]
    _, _, nchunks, chunk_size = dt.shape
    heads_per_group = nheads // ngroups

    # Expand B groups: (batch, seqlen, ngroups, dstate) -> (batch, seqlen, nheads, dstate)
    B = jnp.repeat(B, heads_per_group, axis=2)

    # Pad if needed
    total_len = nchunks * chunk_size
    if seqlen < total_len:
        x = jnp.pad(x, ((0, 0), (0, total_len - seqlen), (0, 0), (0, 0)))
        B = jnp.pad(B, ((0, 0), (0, total_len - seqlen), (0, 0), (0, 0)))

    # Reshape into chunks
    # x: (batch, nchunks, chunk_size, nheads, headdim)
    x = x.reshape(batch, nchunks, chunk_size, nheads, headdim)
    # B: (batch, nchunks, chunk_size, nheads, dstate)
    B = B.reshape(batch, nchunks, chunk_size, nheads, dstate)

    # Decay from position l to end of chunk: exp(dA_cumsum[:,-1] - dA_cumsum[:,l])
    # dA_cumsum: (batch, nheads, nchunks, chunk_size)
    decay = jnp.exp(dA_cumsum[:, :, :, -1:] - dA_cumsum)
    # decay: (batch, nheads, nchunks, chunk_size)

    # Einsum: bclhn, bhcl, bhcl, bclhp -> bchpn
    # B:      (b, c, l, h, n)
    # decay:  (b, h, c, l)
    # dt:     (b, h, c, l)
    # x:      (b, c, l, h, p)
    states = jnp.einsum(
        "bclhn, bhcl, bhcl, bclhp -> bchpn",
        B, decay, dt, x,
    )

    return states


# ---------------------------------------------------------------------------
# 3. state_passing_naive — inter-chunk state propagation
# ---------------------------------------------------------------------------

def state_passing_naive(
    states,             # (batch, nchunks, nheads, dim)
    dA_chunk_cumsum,    # (batch, nheads, nchunks)
    initial_states=None,  # (batch, nheads, dim)
):
    """
    Propagate states across chunks: weighted prefix sum with exponential decay.

    Returns
    -------
    out          : (batch, nchunks, nheads, dim)
    final_states : (batch, nheads, dim)
    """
    batch, nchunks, nheads, dim = states.shape

    if initial_states is None:
        initial_states = jnp.zeros((batch, nheads, dim), dtype=states.dtype)

    # Prepend initial_states as chunk 0
    # (batch, nchunks+1, nheads, dim)
    states = jnp.concatenate(
        [initial_states[:, None, :, :], states],
        axis=1,
    )

    # Pad and cumsum of dA
    # (batch, nheads, nchunks+1)
    dA_padded = jnp.pad(dA_chunk_cumsum, ((0, 0), (0, 0), (1, 0)))
    dA_cumsum = jnp.cumsum(dA_padded, axis=-1)
    nc1 = dA_cumsum.shape[-1]  # nchunks + 1

    # Pairwise decay: decay[z,c] = exp(cumsum[z] - cumsum[c])
    # Mask exponent before exp to avoid inf above the diagonal
    # (IEEE 754: 0 * inf = NaN, so masking after exp is unsafe)
    causal = jnp.tril(jnp.ones((nc1, nc1)))
    exponent = dA_cumsum[:, :, :, None] - dA_cumsum[:, :, None, :]
    exponent = jnp.where(causal[None, None, :, :], exponent, -jnp.inf)
    decay = jnp.exp(exponent)
    # (batch, nheads, nc1, nc1)

    # Weighted sum: out[z] = sum_{c<=z} decay[z,c] * states[c]
    # decay: (batch, nheads, nc1, nc1)
    # states: (batch, nc1, nheads, dim)
    out = jnp.einsum("bhzc, bchd -> bzhd", decay, states)

    # out[:, :-1] = propagated states for chunks 0..nchunks-1
    # out[:, -1]  = final state
    return out[:, :-1], out[:, -1]


# ---------------------------------------------------------------------------
# 4. bmm_chunk_naive — CB = C @ B^T per chunk
# ---------------------------------------------------------------------------

def bmm_chunk_naive(
    C,              # (batch, seqlen, ngroups, dstate)
    B,              # (batch, seqlen, ngroups, dstate)
    chunk_size: int,
):
    """
    Compute CB = C @ B^T within each chunk.

    Returns
    -------
    CB : (batch, nchunks, ngroups, chunk_size, chunk_size) float32
    """
    batch, seqlen, ngroups, dstate = C.shape
    nchunks = (seqlen + chunk_size - 1) // chunk_size
    total_len = nchunks * chunk_size

    if seqlen < total_len:
        C = jnp.pad(C, ((0, 0), (0, total_len - seqlen), (0, 0), (0, 0)))
        B = jnp.pad(B, ((0, 0), (0, total_len - seqlen), (0, 0), (0, 0)))

    # Reshape into chunks
    C_c = C.reshape(batch, nchunks, chunk_size, ngroups, dstate)
    B_c = B.reshape(batch, nchunks, chunk_size, ngroups, dstate)

    # CB[b,c,g,l,s] = sum_n C[b,c,l,g,n] * B[b,c,s,g,n]
    CB = jnp.einsum("bclgn, bcsgn -> bcgls", C_c, B_c)

    return CB.astype(jnp.float32)


# ---------------------------------------------------------------------------
# 5. chunk_scan_naive — final output
# ---------------------------------------------------------------------------

def chunk_scan_naive(
    CB,             # (batch, nchunks, ngroups, chunk_size, chunk_size)
    x,              # (batch, seqlen, nheads, headdim)
    dt,             # (batch, nheads, nchunks, chunk_size)
    dA_cumsum,      # (batch, nheads, nchunks, chunk_size)
    C,              # (batch, seqlen, ngroups, dstate)
    prev_states,    # (batch, nchunks, nheads, headdim, dstate)
    D=None,         # (nheads, headdim) or (nheads,)
    z=None,         # (batch, seqlen, nheads, headdim)
):
    """
    Compute final output combining intra-chunk scan and inter-chunk state.

    Returns
    -------
    out   : (batch, seqlen, nheads, headdim)
    out_x : (batch, seqlen, nheads, headdim) or None — pre-gated output if z given
    """
    batch, seqlen, nheads, headdim = x.shape
    ngroups = C.shape[2]
    dstate = C.shape[3]
    _, _, nchunks, chunk_size = dt.shape
    heads_per_group = nheads // ngroups
    total_len = nchunks * chunk_size

    # Expand C groups: (batch, seqlen, ngroups, dstate) -> (batch, seqlen, nheads, dstate)
    C_exp = jnp.repeat(C, heads_per_group, axis=2)

    # Expand CB groups: (batch, nchunks, ngroups, cs, cs) -> (batch, nchunks, nheads, cs, cs)
    CB_exp = jnp.repeat(CB, heads_per_group, axis=2)

    # Pad x and C if needed
    if seqlen < total_len:
        x = jnp.pad(x, ((0, 0), (0, total_len - seqlen), (0, 0), (0, 0)))
        C_exp = jnp.pad(C_exp, ((0, 0), (0, total_len - seqlen), (0, 0), (0, 0)))

    # --- Part A: Intra-chunk contribution ---
    # Pairwise decay: decay[l,s] = exp(dA_cumsum[l] - dA_cumsum[s])
    # Mask exponent before exp to avoid inf above the diagonal
    # (IEEE 754: 0 * inf = NaN, so masking after exp is unsafe)
    causal = jnp.tril(jnp.ones((chunk_size, chunk_size)))
    exponent = dA_cumsum[:, :, :, :, None] - dA_cumsum[:, :, :, None, :]
    exponent = jnp.where(
        causal[None, None, None, :, :], exponent, -jnp.inf
    )
    decay = jnp.exp(exponent)
    # (batch, nheads, nchunks, chunk_size, chunk_size)

    # scores = CB * decay (causal already baked into decay)
    scores = CB_exp * jnp.transpose(decay, (0, 2, 1, 3, 4))
    # scores: (batch, nchunks, nheads, chunk_size, chunk_size)

    # Reshape x into chunks: (batch, nchunks, chunk_size, nheads, headdim)
    x_c = x.reshape(batch, nchunks, chunk_size, nheads, headdim)

    # Intra-chunk output: out[b,c,l,h,p] = sum_s scores[b,c,h,l,s] * dt[b,h,c,s] * x[b,c,s,h,p]
    out_intra = jnp.einsum(
        "bchls, bhcs, bcshp -> bclhp",
        scores, dt, x_c,
    )

    # --- Part B: Inter-chunk contribution ---
    # state_decay: exp(dA_cumsum[l]) = decay from start of chunk to position l
    state_decay = jnp.exp(
        jnp.transpose(dA_cumsum, (0, 2, 3, 1))  # (batch, nchunks, chunk_size, nheads)
    )[:, :, :, :, None]  # (batch, nchunks, chunk_size, nheads, 1)

    # C_chunked: (batch, nchunks, chunk_size, nheads, dstate)
    C_c = C_exp.reshape(batch, nchunks, chunk_size, nheads, dstate)

    # out_inter = C @ prev_states * decay
    out_inter = jnp.einsum(
        "bclhn, bchpn -> bclhp",
        C_c, prev_states.astype(C_c.dtype),
    ) * state_decay

    out = out_intra + out_inter

    # Reshape back to (batch, seqlen, nheads, headdim)
    out = out.reshape(batch, total_len, nheads, headdim)[:, :seqlen]

    # --- Part C: Skip connection (D) ---
    x_orig = x[:, :seqlen]
    if D is not None:
        if D.ndim == 1:
            out = out + x_orig * D[None, None, :, None]
        else:
            out = out + x_orig * D[None, None, :, :]

    # --- Part D: z-gating ---
    out_x = None
    if z is not None:
        out_x = out
        out = out * jax.nn.silu(z)

    return out, out_x


# ---------------------------------------------------------------------------
# Combined forward — full SSD pipeline
# ---------------------------------------------------------------------------

def mamba_chunk_scan_combined_naive(
    x,                          # (batch, seqlen, nheads, headdim)
    dt,                         # (batch, seqlen, nheads)
    A,                          # (nheads,)
    B,                          # (batch, seqlen, ngroups, dstate)
    C,                          # (batch, seqlen, ngroups, dstate)
    chunk_size: int,
    D=None,                     # (nheads,) or (nheads, headdim)
    z=None,                     # (batch, seqlen, nheads, headdim)
    dt_bias=None,               # (nheads,)
    initial_states=None,        # (batch, nheads, headdim, dstate)
    dt_softplus: bool = False,
    dt_limit: tuple = (0.0, float("inf")),
    return_final_states: bool = False,
    # These are accepted but ignored in naive impl
    seq_idx=None,
    cu_seqlens=None,
    return_varlen_states: bool = False,
):
    """
    Pure JAX implementation of the full Mamba2 SSD combined forward.

    Works on any JAX-supported GPU (RTX 4090, H100, etc.) via XLA.
    For Hopper-optimized kernels, use ``mamba_chunk_scan_combined_fwd``
    from ``mamba2_jax.kernels.ssd_combined``.

    Parameters and return values match ``mamba_chunk_scan_combined_fwd``.
    """
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape

    # 1. chunk_cumsum
    dA_cumsum, dt_out = chunk_cumsum_naive(
        dt, A, chunk_size,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
    )
    nchunks = dA_cumsum.shape[2]

    # 2. chunk_state
    states = chunk_state_naive(B, x, dt_out, dA_cumsum)
    # states: (batch, nchunks, nheads, headdim, dstate)

    # 3. state_passing
    states_flat = states.reshape(batch, nchunks, nheads, headdim * dstate)

    init_flat = None
    if initial_states is not None:
        init_flat = initial_states.reshape(batch, nheads, headdim * dstate)

    states_passed, final_states_flat = state_passing_naive(
        states_flat,
        dA_cumsum[:, :, :, -1],  # (batch, nheads, nchunks) — decay at chunk end
        initial_states=init_flat,
    )

    states_passed = states_passed.reshape(batch, nchunks, nheads, headdim, dstate)
    final_states = final_states_flat.reshape(batch, nheads, headdim, dstate)

    # 4. bmm_chunk
    CB = bmm_chunk_naive(C, B, chunk_size)

    # 5. chunk_scan
    out, out_x = chunk_scan_naive(
        CB, x, dt_out, dA_cumsum, C, states_passed,
        D=D, z=z,
    )

    # Return
    if return_final_states:
        return out, final_states
    return out
