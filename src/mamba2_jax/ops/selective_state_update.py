"""
mamba2_jax/ops/selective_state_update.py

Pure JAX implementation of the single-step SSM state update for Mamba2 inference.

Matches the semantics of ``mamba_ssm.ops.triton.selective_state_update``
(the Triton kernel) using standard JAX ops that XLA can fuse.

Mathematical formula:
    dt' = softplus(dt + dt_bias)                    # optional bias + softplus
    state := exp(dt' * A) * state + (dt' * B) * x  # SSM recurrence
    out   = sum_n(state * C) + x * D                # readout + skip
    out   = out * silu(z)                           # optional gating
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def selective_state_update(
    state,          # (batch, nheads, headdim, dstate) — mutable SSM state
    x,              # (batch, nheads, headdim)
    dt,             # (batch, nheads) or (batch, nheads, headdim)
    A,              # (nheads,) or (nheads, headdim, dstate)
    B,              # (batch, ngroups, dstate)
    C,              # (batch, ngroups, dstate)
    D=None,         # (nheads,) or (nheads, headdim)
    z=None,         # (batch, nheads, headdim)
    dt_bias=None,   # (nheads,) or (nheads, headdim)
    dt_softplus=False,
):
    """
    Single-step SSM state update (inference / autoregressive decoding).

    Parameters
    ----------
    state : jax.Array
        (batch, nheads, headdim, dstate) — the SSM hidden state.
        A **new** updated state is returned (JAX is functional).
    x : jax.Array
        (batch, nheads, headdim) — input at current timestep.
    dt : jax.Array
        (batch, nheads) or (batch, nheads, headdim) — delta time.
    A : jax.Array
        (nheads,) or (nheads, headdim, dstate) — state transition (typically negative).
    B : jax.Array
        (batch, ngroups, dstate) — input projection.
    C : jax.Array
        (batch, ngroups, dstate) — output projection.
    D : jax.Array, optional
        (nheads,) or (nheads, headdim) — skip / feedthrough connection.
    z : jax.Array, optional
        (batch, nheads, headdim) — SiLU gating value.
    dt_bias : jax.Array, optional
        (nheads,) or (nheads, headdim) — bias added to dt before softplus.
    dt_softplus : bool
        If True, apply softplus to dt (after adding dt_bias).

    Returns
    -------
    new_state : jax.Array
        (batch, nheads, headdim, dstate) — updated SSM state.
    out : jax.Array
        (batch, nheads, headdim) — output for this timestep.
    """
    batch, nheads, headdim, dstate = state.shape
    ngroups = B.shape[1]
    heads_per_group = nheads // ngroups

    # --- dt preprocessing ---
    if dt_bias is not None:
        dt = dt + dt_bias
    if dt_softplus:
        dt = jax.nn.softplus(dt)

    # Expand dt to (batch, nheads, headdim) if needed
    if dt.ndim == 2:
        dt = dt[:, :, None]  # (batch, nheads, 1) — broadcast over headdim

    # --- Expand B and C across head groups ---
    # B: (batch, ngroups, dstate) -> (batch, nheads, dstate)
    B = jnp.repeat(B, heads_per_group, axis=1)
    # C: (batch, ngroups, dstate) -> (batch, nheads, dstate)
    C = jnp.repeat(C, heads_per_group, axis=1)

    # --- Discretize A ---
    if A.ndim == 1:
        # A: (nheads,) -> broadcast to (1, nheads, 1, 1)
        dA = jnp.exp(dt[:, :, :, None] * A[None, :, None, None])
    else:
        # A: (nheads, headdim, dstate)
        dA = jnp.exp(dt[:, :, :, None] * A[None, :, :, :])
    # dA: (batch, nheads, headdim, dstate)

    # --- Discretize B ---
    # dBx = dt * B * x -> (batch, nheads, headdim, dstate)
    dBx = (
        dt[:, :, :, None]           # (batch, nheads, headdim, 1)
        * B[:, :, None, :]          # (batch, nheads, 1, dstate)
        * x[:, :, :, None]          # (batch, nheads, headdim, 1)
    )

    # --- State update ---
    new_state = state * dA + dBx

    # --- Readout ---
    # out = sum_n(state * C): (batch, nheads, headdim)
    out = jnp.sum(new_state * C[:, :, None, :], axis=-1)

    # --- D skip connection ---
    if D is not None:
        if D.ndim == 1:
            out = out + D[None, :, None] * x
        else:
            out = out + D[None, :, :] * x

    # --- z gating ---
    if z is not None:
        out = out * jax.nn.silu(z)

    return new_state, out
