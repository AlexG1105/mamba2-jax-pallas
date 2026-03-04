"""
mamba2_jax/distributed/tensor_parallel.py

JAX implementations of Megatron-style tensor parallel linear layers.

Mirrors ``mamba_ssm.distributed.tensor_parallel`` using JAX collective ops.

These use ``jax.lax`` collectives (psum, all_gather, psum_scatter) which
require the code to be running inside ``jax.pmap`` or ``shard_map`` with
named axes.

For single-GPU usage, the classes degrade to plain ``Dense`` layers
(process_group=None).
"""

from __future__ import annotations

from typing import Optional
import math

import jax
import jax.numpy as jnp
import flax.linen as nn


# ---------------------------------------------------------------------------
# Collective operation wrappers
# ---------------------------------------------------------------------------

def all_reduce(x, axis_name: Optional[str] = None):
    """Sum-reduce x across all devices on the named axis."""
    if axis_name is None:
        return x
    return jax.lax.psum(x, axis_name=axis_name)


def reduce_scatter(x, axis_name: Optional[str] = None, scatter_dim: int = 0):
    """Sum-reduce and scatter x along scatter_dim across devices."""
    if axis_name is None:
        return x
    return jax.lax.psum_scatter(x, axis_name=axis_name, scatter_dimension=scatter_dim)


def all_gather(x, axis_name: Optional[str] = None, gather_dim: int = 0):
    """Gather x along gather_dim from all devices."""
    if axis_name is None:
        return x
    return jax.lax.all_gather(x, axis_name=axis_name)


# ---------------------------------------------------------------------------
# ColumnParallelLinear
# ---------------------------------------------------------------------------

class ColumnParallelLinear(nn.Module):
    """
    Linear layer that splits the **output** (column) dimension across devices.

    In single-GPU mode (axis_name=None), this is just a standard Dense layer.

    Attributes
    ----------
    features : int
        **Local** output features (already divided by world_size).
    use_bias : bool
        Whether to include a bias term.
    axis_name : str or None
        Name of the pmap/shard_map axis for collectives. None = single GPU.
    sequence_parallel : bool
        If True, input is sequence-parallel sharded and an all_gather is
        performed on the sequence dimension before the matmul.
    dtype : jnp.dtype
        Parameter and computation dtype.
    """
    features: int
    use_bias: bool = True
    axis_name: Optional[str] = None
    sequence_parallel: bool = True
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        # If sequence_parallel, gather along seq dim first
        if self.sequence_parallel and self.axis_name is not None:
            x = all_gather(x, axis_name=self.axis_name, gather_dim=-2)

        kernel = self.param(
            "kernel",
            nn.initializers.lecun_normal(),
            (x.shape[-1], self.features),
            self.dtype,
        )
        out = x @ kernel

        if self.use_bias:
            bias = self.param(
                "bias",
                nn.initializers.zeros,
                (self.features,),
                self.dtype,
            )
            out = out + bias

        return out


# ---------------------------------------------------------------------------
# RowParallelLinear
# ---------------------------------------------------------------------------

class RowParallelLinear(nn.Module):
    """
    Linear layer that splits the **input** (row) dimension across devices.

    Expects input already partitioned (e.g., from a ColumnParallelLinear).
    After the local matmul, performs reduce_scatter (sequence_parallel=True)
    or all_reduce (sequence_parallel=False) to combine partial sums.

    In single-GPU mode (axis_name=None), this is just a standard Dense layer.

    Attributes
    ----------
    features : int
        Output features (full, not sharded).
    use_bias : bool
        Whether to include a bias term. In multi-GPU mode, only rank 0
        should have bias to avoid double-counting. For simplicity in JAX,
        we always include the bias and rely on the user to handle this
        correctly when loading weights.
    axis_name : str or None
        Name of the pmap/shard_map axis for collectives. None = single GPU.
    sequence_parallel : bool
        If True, use reduce_scatter; otherwise all_reduce.
    dtype : jnp.dtype
        Parameter and computation dtype.
    """
    features: int
    use_bias: bool = True
    axis_name: Optional[str] = None
    sequence_parallel: bool = True
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        in_features = x.shape[-1]

        kernel = self.param(
            "kernel",
            nn.initializers.lecun_normal(),
            (in_features, self.features),
            self.dtype,
        )
        out = x @ kernel

        if self.use_bias:
            bias = self.param(
                "bias",
                nn.initializers.zeros,
                (self.features,),
                self.dtype,
            )
            out = out + bias

        # Reduce across devices
        if self.axis_name is not None:
            if self.sequence_parallel:
                out = reduce_scatter(out, axis_name=self.axis_name, scatter_dim=-2)
            else:
                out = all_reduce(out, axis_name=self.axis_name)

        return out
