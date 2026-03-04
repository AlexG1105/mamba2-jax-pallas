"""
tests/test_mamba2_ops.py

Standalone tests for the pure JAX ops (no PyTorch / mamba-ssm needed).
Tests selective_state_update, causal_conv1d, and rms_norm_gated.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

# Force CPU for testing if no GPU
# os.environ.setdefault("JAX_PLATFORMS", "cpu")

from mamba2_jax.ops.selective_state_update import selective_state_update
from mamba2_jax.ops.causal_conv1d import causal_conv1d, causal_conv1d_update
from mamba2_jax.ops.rms_norm import rms_norm_gated


def test_selective_state_update_basic():
    """Test SSM step update matches manual computation."""
    print("=" * 60)
    print("Test: selective_state_update basic")
    print("=" * 60)

    key = jax.random.PRNGKey(42)
    batch, nheads, headdim, dstate = 2, 4, 8, 16
    ngroups = 2

    keys = jax.random.split(key, 7)
    state = jax.random.normal(keys[0], (batch, nheads, headdim, dstate))
    x = jax.random.normal(keys[1], (batch, nheads, headdim))
    dt = jax.random.uniform(keys[2], (batch, nheads), minval=0.01, maxval=0.1)
    A = -jnp.abs(jax.random.normal(keys[3], (nheads,)))
    B = jax.random.normal(keys[4], (batch, ngroups, dstate))
    C = jax.random.normal(keys[5], (batch, ngroups, dstate))

    new_state, out = selective_state_update(
        state, x, dt, A, B, C,
        dt_softplus=False,
    )

    # Manual reference computation
    heads_per_group = nheads // ngroups
    B_exp = jnp.repeat(B, heads_per_group, axis=1)  # (batch, nheads, dstate)
    C_exp = jnp.repeat(C, heads_per_group, axis=1)

    dA = jnp.exp(dt[:, :, None, None] * A[None, :, None, None])
    dBx = dt[:, :, None, None] * B_exp[:, :, None, :] * x[:, :, :, None]
    ref_state = state * dA + dBx
    ref_out = jnp.sum(ref_state * C_exp[:, :, None, :], axis=-1)

    state_err = jnp.max(jnp.abs(new_state - ref_state))
    out_err = jnp.max(jnp.abs(out - ref_out))

    print(f"  State max error: {state_err:.2e}")
    print(f"  Output max error: {out_err:.2e}")
    assert state_err < 1e-5, f"State error too large: {state_err}"
    assert out_err < 1e-5, f"Output error too large: {out_err}"
    print("  PASSED\n")


def test_selective_state_update_with_D_z_dtbias():
    """Test SSM step with D, z gating, and dt_bias + softplus."""
    print("=" * 60)
    print("Test: selective_state_update with D, z, dt_bias, softplus")
    print("=" * 60)

    key = jax.random.PRNGKey(123)
    batch, nheads, headdim, dstate = 2, 4, 8, 16
    ngroups = 1

    keys = jax.random.split(key, 10)
    state = jax.random.normal(keys[0], (batch, nheads, headdim, dstate))
    x = jax.random.normal(keys[1], (batch, nheads, headdim))
    dt = jax.random.normal(keys[2], (batch, nheads))  # pre-softplus
    A = -jnp.abs(jax.random.normal(keys[3], (nheads,)))
    B = jax.random.normal(keys[4], (batch, ngroups, dstate))
    C = jax.random.normal(keys[5], (batch, ngroups, dstate))
    D = jax.random.normal(keys[6], (nheads,))
    z = jax.random.normal(keys[7], (batch, nheads, headdim))
    dt_bias = jax.random.normal(keys[8], (nheads,))

    new_state, out = selective_state_update(
        state, x, dt, A, B, C,
        D=D, z=z, dt_bias=dt_bias, dt_softplus=True,
    )

    # Manual reference
    dt_ref = jax.nn.softplus(dt + dt_bias)
    B_exp = jnp.repeat(B, nheads // ngroups, axis=1)
    C_exp = jnp.repeat(C, nheads // ngroups, axis=1)
    dA = jnp.exp(dt_ref[:, :, None, None] * A[None, :, None, None])
    dBx = dt_ref[:, :, None, None] * B_exp[:, :, None, :] * x[:, :, :, None]
    ref_state = state * dA + dBx
    ref_out = jnp.sum(ref_state * C_exp[:, :, None, :], axis=-1)
    ref_out = ref_out + D[None, :, None] * x
    ref_out = ref_out * jax.nn.silu(z)

    state_err = jnp.max(jnp.abs(new_state - ref_state))
    out_err = jnp.max(jnp.abs(out - ref_out))

    print(f"  State max error: {state_err:.2e}")
    print(f"  Output max error: {out_err:.2e}")
    assert state_err < 1e-5, f"State error too large: {state_err}"
    assert out_err < 1e-4, f"Output error too large: {out_err}"
    print("  PASSED\n")


def test_causal_conv1d_basic():
    """Test causal conv1d matches manual reference."""
    print("=" * 60)
    print("Test: causal_conv1d basic")
    print("=" * 60)

    key = jax.random.PRNGKey(42)
    batch, d, seqlen, kernel_size = 2, 8, 16, 4

    keys = jax.random.split(key, 3)
    x = jax.random.normal(keys[0], (batch, d, seqlen))
    weight = jax.random.normal(keys[1], (d, kernel_size))
    bias = jax.random.normal(keys[2], (d,))

    out = causal_conv1d(x, weight, bias=bias, activation=None)

    # Manual reference: pad and convolve
    x_padded = jnp.pad(x, ((0, 0), (0, 0), (kernel_size - 1, 0)))
    ref_out = jnp.zeros_like(x)
    for t in range(seqlen):
        window = x_padded[:, :, t:t + kernel_size]  # (batch, d, kernel_size)
        ref_out = ref_out.at[:, :, t].set(
            jnp.sum(window * weight[None, :, :], axis=-1) + bias[None, :]
        )

    err = jnp.max(jnp.abs(out - ref_out))
    print(f"  Max error vs manual conv: {err:.2e}")
    assert err < 1e-5, f"Error too large: {err}"
    print("  PASSED\n")


def test_causal_conv1d_with_silu():
    """Test causal conv1d with SiLU activation."""
    print("=" * 60)
    print("Test: causal_conv1d with SiLU")
    print("=" * 60)

    key = jax.random.PRNGKey(7)
    batch, d, seqlen, kernel_size = 1, 4, 8, 3

    keys = jax.random.split(key, 2)
    x = jax.random.normal(keys[0], (batch, d, seqlen))
    weight = jax.random.normal(keys[1], (d, kernel_size))

    out_no_act = causal_conv1d(x, weight, activation=None)
    out_silu = causal_conv1d(x, weight, activation="silu")

    ref = jax.nn.silu(out_no_act)
    err = jnp.max(jnp.abs(out_silu - ref))
    print(f"  Max error (silu applied post-conv): {err:.2e}")
    assert err < 1e-6, f"Error too large: {err}"
    print("  PASSED\n")


def test_causal_conv1d_update():
    """Test single-step conv update matches full conv on last position."""
    print("=" * 60)
    print("Test: causal_conv1d_update consistency with full conv")
    print("=" * 60)

    key = jax.random.PRNGKey(99)
    batch, d, kernel_size = 2, 8, 4
    seqlen = 10

    keys = jax.random.split(key, 3)
    x_full = jax.random.normal(keys[0], (batch, d, seqlen))
    weight = jax.random.normal(keys[1], (d, kernel_size))
    bias = jax.random.normal(keys[2], (d,))

    # Full conv
    out_full = causal_conv1d(x_full, weight, bias=bias, activation="silu")

    # Step-by-step using conv_state
    conv_state = jnp.zeros((batch, d, kernel_size))

    for t in range(seqlen):
        x_t = x_full[:, :, t]  # (batch, d)
        conv_state, out_t = causal_conv1d_update(
            x_t, conv_state, weight, bias=bias, activation="silu",
        )
        # Check this step matches full conv at position t
        err = jnp.max(jnp.abs(out_t - out_full[:, :, t]))
        if err > 1e-5:
            print(f"  Step {t}: max error {err:.2e} FAIL")
            assert False, f"Step {t} error too large: {err}"

    print(f"  All {seqlen} steps match full conv (max err < 1e-5)")
    print("  PASSED\n")


def test_rms_norm_gated_basic():
    """Test RMSNorm without gating."""
    print("=" * 60)
    print("Test: rms_norm_gated basic (no gate)")
    print("=" * 60)

    key = jax.random.PRNGKey(42)
    batch, seqlen, hidden = 2, 8, 16

    keys = jax.random.split(key, 2)
    x = jax.random.normal(keys[0], (batch, seqlen, hidden))
    weight = jnp.ones(hidden)

    out = rms_norm_gated(x, weight, z=None)

    # Manual: rms = sqrt(mean(x^2) + eps); out = x / rms
    rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-5)
    ref = x / rms

    err = jnp.max(jnp.abs(out - ref))
    print(f"  Max error: {err:.2e}")
    assert err < 1e-6, f"Error too large: {err}"
    print("  PASSED\n")


def test_rms_norm_gated_with_gate():
    """Test RMSNorm with SiLU gating (norm_before_gate=True and False)."""
    print("=" * 60)
    print("Test: rms_norm_gated with gate")
    print("=" * 60)

    key = jax.random.PRNGKey(42)
    batch, hidden = 2, 16

    keys = jax.random.split(key, 3)
    x = jax.random.normal(keys[0], (batch, hidden))
    z = jax.random.normal(keys[1], (batch, hidden))
    weight = jax.random.normal(keys[2], (hidden,))

    # norm_before_gate=True: norm(x) * silu(z)
    out_nbg = rms_norm_gated(x, weight, z=z, norm_before_gate=True)
    rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-5)
    ref_nbg = (x / rms * weight) * jax.nn.silu(z)
    err1 = jnp.max(jnp.abs(out_nbg - ref_nbg))

    # norm_before_gate=False: norm(x * silu(z))
    out_nbf = rms_norm_gated(x, weight, z=z, norm_before_gate=False)
    xz = x * jax.nn.silu(z)
    rms2 = jnp.sqrt(jnp.mean(xz * xz, axis=-1, keepdims=True) + 1e-5)
    ref_nbf = xz / rms2 * weight
    err2 = jnp.max(jnp.abs(out_nbf - ref_nbf))

    print(f"  norm_before_gate=True  max error: {err1:.2e}")
    print(f"  norm_before_gate=False max error: {err2:.2e}")
    assert err1 < 1e-5, f"Error too large: {err1}"
    assert err2 < 1e-5, f"Error too large: {err2}"
    print("  PASSED\n")


def test_rms_norm_gated_grouped():
    """Test grouped RMSNorm (group_size < hidden_size)."""
    print("=" * 60)
    print("Test: rms_norm_gated grouped")
    print("=" * 60)

    key = jax.random.PRNGKey(42)
    batch, hidden = 2, 32
    group_size = 8
    ngroups = hidden // group_size

    keys = jax.random.split(key, 2)
    x = jax.random.normal(keys[0], (batch, hidden))
    weight = jnp.ones(hidden)

    out = rms_norm_gated(x, weight, group_size=group_size)

    # Manual: reshape to (batch, ngroups, group_size), norm each group
    x_g = x.reshape(batch, ngroups, group_size)
    rms = jnp.sqrt(jnp.mean(x_g * x_g, axis=-1, keepdims=True) + 1e-5)
    ref = (x_g / rms).reshape(batch, hidden)

    err = jnp.max(jnp.abs(out - ref))
    print(f"  Max error: {err:.2e}")
    assert err < 1e-6, f"Error too large: {err}"
    print("  PASSED\n")


if __name__ == "__main__":
    print("Testing mamba2_jax ops (pure JAX, no GPU required)")
    print("=" * 60)
    print()

    test_selective_state_update_basic()
    test_selective_state_update_with_D_z_dtbias()
    test_causal_conv1d_basic()
    test_causal_conv1d_with_silu()
    test_causal_conv1d_update()
    test_rms_norm_gated_basic()
    test_rms_norm_gated_with_gate()
    test_rms_norm_gated_grouped()

    print("=" * 60)
    print("ALL OPS TESTS PASSED")
    print("=" * 60)
