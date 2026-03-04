"""
mamba2_jax.distributed — Tensor parallelism utilities for multi-GPU inference.
"""

from .tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    all_reduce,
    reduce_scatter,
    all_gather,
)

__all__ = [
    "ColumnParallelLinear",
    "RowParallelLinear",
    "all_reduce",
    "reduce_scatter",
    "all_gather",
]
