"""
mamba2_jax/models/mamba2_lm.py

Full Mamba2 Language Model in JAX/Flax.

Architecture (mirrors ``mamba_ssm.models.mixer_seq_simple.MambaLMHeadModel``):
    embedding → [norm → mamba2_block] × n_layer → final_norm → lm_head

Supports loading weights from HuggingFace ``state-spaces/mamba2-*`` models.

Available model configs:
    mamba2-130m:  d_model=768,  n_layer=24
    mamba2-370m:  d_model=1024, n_layer=48
    mamba2-780m:  d_model=1536, n_layer=48
    mamba2-1.3b:  d_model=2048, n_layer=48
    mamba2-2.7b:  d_model=2560, n_layer=64
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from mamba2_jax.ops.rms_norm import rms_norm_gated


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Mamba2Config:
    """Configuration for a full Mamba2 language model."""
    d_model: int = 2048
    n_layer: int = 48
    vocab_size: int = 50277
    pad_vocab_size_multiple: int = 16
    # SSM config (Mamba2 block params)
    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    chunk_size: int = 256
    # Flags
    rmsnorm: bool = True
    norm_before_gate: bool = False
    bias: bool = False
    conv_bias: bool = True
    D_has_hdim: bool = False
    tie_embeddings: bool = True
    residual_in_fp32: bool = True
    dt_limit: tuple = (0.0, float("inf"))

    @property
    def vocab_size_padded(self):
        """Vocab size rounded up to pad_vocab_size_multiple."""
        v = self.vocab_size
        m = self.pad_vocab_size_multiple
        return math.ceil(v / m) * m

    @property
    def d_inner(self):
        return self.expand * self.d_model

    @property
    def nheads(self):
        return self.d_inner // self.headdim

    @property
    def conv_dim(self):
        return self.d_inner + 2 * self.ngroups * self.d_state

    @property
    def d_in_proj(self):
        return 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "Mamba2Config":
        """Load config from a HuggingFace model repo or local path."""
        try:
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(model_name_or_path, "config.json")
        except Exception:
            import os
            config_path = os.path.join(model_name_or_path, "config.json")

        with open(config_path) as f:
            data = json.load(f)

        # Map HF config keys to our config
        ssm_cfg = data.get("ssm_cfg", {})
        return cls(
            d_model=data.get("d_model", 2048),
            n_layer=data.get("n_layer", 48),
            vocab_size=data.get("vocab_size", 50277),
            pad_vocab_size_multiple=data.get("pad_vocab_size_multiple", 16),
            d_state=ssm_cfg.get("d_state", 128),
            d_conv=ssm_cfg.get("d_conv", 4),
            expand=ssm_cfg.get("expand", 2),
            headdim=ssm_cfg.get("headdim", 64),
            ngroups=ssm_cfg.get("ngroups", 1),
            chunk_size=ssm_cfg.get("chunk_size", 256),
            rmsnorm=data.get("rms_norm", True),
            residual_in_fp32=data.get("residual_in_fp32", True),
            tie_embeddings=data.get("tie_embeddings", True),
        )


# ---------------------------------------------------------------------------
# Predefined configs
# ---------------------------------------------------------------------------

MAMBA2_CONFIGS = {
    "mamba2-130m": Mamba2Config(d_model=768, n_layer=24),
    "mamba2-370m": Mamba2Config(d_model=1024, n_layer=48),
    "mamba2-780m": Mamba2Config(d_model=1536, n_layer=48),
    "mamba2-1.3b": Mamba2Config(d_model=2048, n_layer=48),
    "mamba2-2.7b": Mamba2Config(d_model=2560, n_layer=64),
}


# ---------------------------------------------------------------------------
# Pure-functional Mamba2 LM (no Flax nn.Module — just functions + param dicts)
# ---------------------------------------------------------------------------
# We use explicit parameter dictionaries rather than Flax modules for the
# full model, since weight loading from PyTorch checkpoints is much simpler
# with plain dicts than Flax's nested parameter structure.

def rms_norm(x, weight, eps=1e-5):
    """Standard RMSNorm (no gating)."""
    rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    return x / rms * weight


class Mamba2LMHeadModel:
    """
    Full Mamba2 Language Model for inference.

    Uses explicit parameter dicts (not Flax nn.Module) for easy weight loading.

    Parameters are stored as a nested dict::

        params = {
            "embedding": {
                "weight": (vocab_size_padded, d_model),
            },
            "layers": [
                {
                    "norm_weight": (d_model,),
                    "mixer": {
                        "in_proj_kernel": (d_model, d_in_proj),
                        "conv1d_weight": (conv_dim, d_conv),
                        "conv1d_bias": (conv_dim,),
                        "dt_bias": (nheads,),
                        "A_log": (nheads,),
                        "D": (nheads,),
                        "norm_weight": (d_ssm,),
                        "out_proj_kernel": (d_inner, d_model),
                    },
                },
                ...
            ],
            "norm_f_weight": (d_model,),
            "lm_head_weight": (vocab_size_padded, d_model),  # tied to embedding
        }
    """

    def __init__(self, config: Mamba2Config):
        self.config = config

    def __call__(self, params, input_ids):
        """
        Forward pass (prefill / full sequence).

        Parameters
        ----------
        params : dict
            Model parameters (see class docstring for structure).
        input_ids : jax.Array
            (batch, seqlen) — integer token IDs.

        Returns
        -------
        logits : jax.Array
            (batch, seqlen, vocab_size_padded)
        """
        cfg = self.config

        # Embedding
        x = params["embedding"]["weight"][input_ids]  # (batch, seqlen, d_model)

        # Residual dtype
        dtype = jnp.float32 if cfg.residual_in_fp32 else x.dtype

        # Layers
        for i, layer_params in enumerate(params["layers"]):
            residual = x.astype(dtype)
            # Pre-norm
            x_normed = rms_norm(x.astype(jnp.float32), layer_params["norm_weight"])
            # Mamba2 block
            x_out = _mamba2_block_forward(x_normed, layer_params["mixer"], cfg)
            # Residual
            x = residual + x_out.astype(dtype)

        # Final norm
        x = rms_norm(x.astype(jnp.float32), params["norm_f_weight"])

        # LM head
        logits = x @ params["lm_head_weight"].T
        return logits

    def generate(
        self,
        params,
        input_ids,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        rng_key=None,
    ):
        """
        Autoregressive text generation.

        Parameters
        ----------
        params : dict
            Model parameters.
        input_ids : jax.Array
            (1, prompt_len) — prompt token IDs (batch=1 only).
        max_new_tokens : int
            Number of tokens to generate.
        temperature : float
            Sampling temperature.
        top_k : int
            Top-k filtering.
        rng_key : jax.random.PRNGKey
            Random key for sampling.

        Returns
        -------
        output_ids : jax.Array
            (1, prompt_len + max_new_tokens)
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        cfg = self.config
        batch = input_ids.shape[0]
        assert batch == 1, "Generation only supports batch=1"

        # Prefill (eager, runs once)
        logits, inference_state = self._forward_with_cache(
            params, input_ids, inference_state=None,
        )

        # Stack layer params and states for scan-based JIT decode
        stacked_layer_params = jax.tree.map(
            lambda *xs: jnp.stack(xs), *params["layers"]
        )
        non_layer_params = {
            "embedding": params["embedding"],
            "norm_f_weight": params["norm_f_weight"],
            "lm_head_weight": params["lm_head_weight"],
        }
        conv_states = jnp.stack([s[0] for s in inference_state])
        ssm_states = jnp.stack([s[1] for s in inference_state])

        # Build JIT-compiled decode step (cached on self to avoid recompilation)
        if not hasattr(self, '_decode_step_jit'):
            self._decode_step_jit = jax.jit(
                lambda slp, nlp, tok, cs, ss: _scan_decode_step(
                    slp, nlp, tok, cs, ss, cfg
                )
            )

        # Sample first token
        next_token_logits = logits[:, -1, :]  # (1, vocab)
        all_ids = [input_ids]

        for step in range(max_new_tokens):
            rng_key, sample_key = jax.random.split(rng_key)

            # Sample
            next_token = _sample_token(
                next_token_logits, sample_key, temperature, top_k
            )
            all_ids.append(next_token[:, None])

            # JIT-compiled decode step with scan over layers
            next_token_logits, conv_states, ssm_states = self._decode_step_jit(
                stacked_layer_params, non_layer_params,
                next_token[:, None], conv_states, ssm_states,
            )

        return jnp.concatenate(all_ids, axis=1)

    def _forward_with_cache(self, params, input_ids, inference_state=None):
        """
        Forward pass with KV-cache-style state tracking for generation.

        Returns (logits, inference_state) where inference_state is a list
        of per-layer (conv_state, ssm_state) tuples.
        """
        cfg = self.config
        batch, seqlen = input_ids.shape

        x = params["embedding"]["weight"][input_ids]
        dtype = jnp.float32 if cfg.residual_in_fp32 else x.dtype

        if inference_state is None:
            # Prefill mode
            new_states = []
            for i, layer_params in enumerate(params["layers"]):
                residual = x.astype(dtype)
                x_normed = rms_norm(x.astype(jnp.float32), layer_params["norm_weight"])
                x_out, layer_state = _mamba2_block_forward_with_cache(
                    x_normed, layer_params["mixer"], cfg,
                    conv_state=None, ssm_state=None,
                    is_prefill=True,
                )
                x = residual + x_out.astype(dtype)
                new_states.append(layer_state)
        else:
            # Decode step
            new_states = []
            for i, layer_params in enumerate(params["layers"]):
                residual = x.astype(dtype)
                x_normed = rms_norm(x.astype(jnp.float32), layer_params["norm_weight"])
                conv_state, ssm_state = inference_state[i]
                x_out, layer_state = _mamba2_block_forward_with_cache(
                    x_normed, layer_params["mixer"], cfg,
                    conv_state=conv_state, ssm_state=ssm_state,
                    is_prefill=False,
                )
                x = residual + x_out.astype(dtype)
                new_states.append(layer_state)

        x = rms_norm(x.astype(jnp.float32), params["norm_f_weight"])
        logits = x @ params["lm_head_weight"].T
        return logits, new_states


# ---------------------------------------------------------------------------
# Mamba2 block forward (functional, operates on param dicts)
# ---------------------------------------------------------------------------

def _mamba2_block_forward(x, mixer_params, cfg):
    """
    Single Mamba2 block forward pass (chunked scan mode).

    Parameters
    ----------
    x : (batch, seqlen, d_model) — input (already normed)
    mixer_params : dict with in_proj_kernel, conv1d_weight, etc.
    cfg : Mamba2Config
    """
    from mamba2_jax.ops.causal_conv1d import causal_conv1d
    from mamba2_jax.modules.mamba2 import _ssd_combined_fwd

    batch, seqlen, _ = x.shape
    d_inner = cfg.d_inner
    d_ssm = d_inner  # d_ssm == d_inner for all standard configs
    d_mlp = 0

    A = -jnp.exp(mixer_params["A_log"].astype(jnp.float32))
    D = mixer_params["D"]
    dt_bias = mixer_params["dt_bias"]

    # 1. Input projection
    zxbcdt = x @ mixer_params["in_proj_kernel"]

    # 2. Split: [z, xBC, dt] (d_mlp=0 so no z0/x0)
    ngs = cfg.ngroups * cfg.d_state
    z = zxbcdt[:, :, :d_ssm]
    xBC = zxbcdt[:, :, d_ssm:2 * d_ssm + 2 * ngs]
    dt = zxbcdt[:, :, 2 * d_ssm + 2 * ngs:]

    # 3. Causal conv1d
    xBC_conv = causal_conv1d(
        jnp.transpose(xBC, (0, 2, 1)),
        mixer_params["conv1d_weight"],
        bias=mixer_params.get("conv1d_bias"),
        activation="silu",
    )
    xBC = jnp.transpose(xBC_conv, (0, 2, 1))

    # 4. Split post-conv
    x_ssm = xBC[:, :, :d_ssm].reshape(batch, seqlen, cfg.nheads, cfg.headdim)
    B = xBC[:, :, d_ssm:d_ssm + ngs].reshape(batch, seqlen, cfg.ngroups, cfg.d_state)
    C = xBC[:, :, d_ssm + ngs:].reshape(batch, seqlen, cfg.ngroups, cfg.d_state)

    # 5. SSD chunk scan
    scan_result = _ssd_combined_fwd(
        x=x_ssm, dt=dt, A=A, B=B, C=C,
        chunk_size=cfg.chunk_size,
        D=D, z=None,  # z handled by rmsnorm below
        dt_bias=dt_bias,
        dt_softplus=True,
        dt_limit=cfg.dt_limit,
        return_final_states=False,
    )
    y = scan_result.reshape(batch, seqlen, d_ssm)

    # 6. RMSNorm gating
    if cfg.rmsnorm:
        y = rms_norm_gated(
            y, mixer_params["norm_weight"], z=z,
            group_size=d_ssm // cfg.ngroups,
            norm_before_gate=cfg.norm_before_gate,
        )

    # 7. Output projection
    out = y @ mixer_params["out_proj_kernel"]
    return out


def _mamba2_block_forward_with_cache(x, mixer_params, cfg, conv_state, ssm_state, is_prefill):
    """
    Mamba2 block forward with inference state tracking.

    Returns (output, (new_conv_state, new_ssm_state)).
    """
    from mamba2_jax.ops.causal_conv1d import causal_conv1d, causal_conv1d_update
    from mamba2_jax.ops.selective_state_update import selective_state_update
    from mamba2_jax.modules.mamba2 import _ssd_combined_fwd

    batch, seqlen, _ = x.shape
    d_inner = cfg.d_inner
    d_ssm = d_inner
    ngs = cfg.ngroups * cfg.d_state

    A = -jnp.exp(mixer_params["A_log"].astype(jnp.float32))
    D = mixer_params["D"]
    dt_bias = mixer_params["dt_bias"]

    if is_prefill:
        # --- Prefill: chunked scan ---
        zxbcdt = x @ mixer_params["in_proj_kernel"]
        z = zxbcdt[:, :, :d_ssm]
        xBC = zxbcdt[:, :, d_ssm:2 * d_ssm + 2 * ngs]
        dt = zxbcdt[:, :, 2 * d_ssm + 2 * ngs:]

        # Save conv state (last d_conv timesteps)
        xBC_T = jnp.transpose(xBC, (0, 2, 1))
        if seqlen >= cfg.d_conv:
            new_conv_state = xBC_T[:, :, -cfg.d_conv:]
        else:
            new_conv_state = jnp.pad(
                xBC_T, ((0, 0), (0, 0), (cfg.d_conv - seqlen, 0))
            )

        xBC_conv = causal_conv1d(
            xBC_T, mixer_params["conv1d_weight"],
            bias=mixer_params.get("conv1d_bias"), activation="silu",
        )
        xBC = jnp.transpose(xBC_conv, (0, 2, 1))

        x_ssm = xBC[:, :, :d_ssm].reshape(batch, seqlen, cfg.nheads, cfg.headdim)
        B = xBC[:, :, d_ssm:d_ssm + ngs].reshape(batch, seqlen, cfg.ngroups, cfg.d_state)
        C = xBC[:, :, d_ssm + ngs:].reshape(batch, seqlen, cfg.ngroups, cfg.d_state)

        scan_result = _ssd_combined_fwd(
            x=x_ssm, dt=dt, A=A, B=B, C=C,
            chunk_size=cfg.chunk_size,
            D=D, z=None, dt_bias=dt_bias,
            dt_softplus=True, dt_limit=cfg.dt_limit,
            return_final_states=True,
        )
        y, final_states = scan_result
        y = y.reshape(batch, seqlen, d_ssm)

        if cfg.rmsnorm:
            y = rms_norm_gated(
                y, mixer_params["norm_weight"], z=z,
                group_size=d_ssm // cfg.ngroups,
                norm_before_gate=cfg.norm_before_gate,
            )

        out = y @ mixer_params["out_proj_kernel"]
        return out, (new_conv_state, final_states)

    else:
        # --- Decode step: single token ---
        hidden = x[:, 0, :]  # (batch, d_model)
        zxbcdt = hidden @ mixer_params["in_proj_kernel"]

        z = zxbcdt[:, :d_ssm]
        xBC = zxbcdt[:, d_ssm:2 * d_ssm + 2 * ngs]
        dt = zxbcdt[:, 2 * d_ssm + 2 * ngs:]

        # Conv update
        new_conv_state, xBC = causal_conv1d_update(
            xBC, conv_state, mixer_params["conv1d_weight"],
            bias=mixer_params.get("conv1d_bias"), activation="silu",
        )

        x_ssm = xBC[:, :d_ssm].reshape(-1, cfg.nheads, cfg.headdim)
        B = xBC[:, d_ssm:d_ssm + ngs].reshape(-1, cfg.ngroups, cfg.d_state)
        C = xBC[:, d_ssm + ngs:].reshape(-1, cfg.ngroups, cfg.d_state)

        # SSM state update
        new_ssm_state, y = selective_state_update(
            ssm_state, x_ssm, dt, A, B, C,
            D=D, z=None, dt_bias=dt_bias, dt_softplus=True,
        )
        y = y.reshape(-1, d_ssm)

        if cfg.rmsnorm:
            y = rms_norm_gated(
                y, mixer_params["norm_weight"], z=z,
                group_size=d_ssm // cfg.ngroups,
                norm_before_gate=cfg.norm_before_gate,
            )

        out = y @ mixer_params["out_proj_kernel"]
        return out[:, None, :], (new_conv_state, new_ssm_state)


# ---------------------------------------------------------------------------
# Weight loading from PyTorch checkpoint
# ---------------------------------------------------------------------------

def load_from_torch_checkpoint(config: Mamba2Config, checkpoint_path: str, dtype=jnp.float32):
    """
    Load weights from a PyTorch ``pytorch_model.bin`` checkpoint.

    Parameters
    ----------
    config : Mamba2Config
    checkpoint_path : str
        Path to ``pytorch_model.bin``.
    dtype : jnp.dtype
        Target dtype for weights.

    Returns
    -------
    params : dict
        Parameter dict compatible with ``Mamba2LMHeadModel``.
    """
    import torch

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    params = {
        "embedding": {
            "weight": _to_jax(state_dict["backbone.embedding.weight"], dtype),
        },
        "layers": [],
        "norm_f_weight": _to_jax(state_dict["backbone.norm_f.weight"], dtype),
    }

    for i in range(config.n_layer):
        prefix = f"backbone.layers.{i}"
        layer = {
            "norm_weight": _to_jax(state_dict[f"{prefix}.norm.weight"], dtype),
            "mixer": {
                # PyTorch Linear: (out, in) -> transpose to (in, out)
                "in_proj_kernel": _to_jax(
                    state_dict[f"{prefix}.mixer.in_proj.weight"].T, dtype
                ),
                # Conv1d: (out_ch, in_ch/groups, k) -> squeeze to (out_ch, k)
                "conv1d_weight": _to_jax(
                    state_dict[f"{prefix}.mixer.conv1d.weight"].squeeze(1), dtype
                ),
                "conv1d_bias": _to_jax(
                    state_dict[f"{prefix}.mixer.conv1d.bias"], dtype
                ),
                "dt_bias": _to_jax(
                    state_dict[f"{prefix}.mixer.dt_bias"], jnp.float32
                ),
                "A_log": _to_jax(
                    state_dict[f"{prefix}.mixer.A_log"], dtype
                ),
                "D": _to_jax(
                    state_dict[f"{prefix}.mixer.D"], dtype
                ),
                "norm_weight": _to_jax(
                    state_dict[f"{prefix}.mixer.norm.weight"], dtype
                ),
                # PyTorch Linear: (out, in) -> transpose to (in, out)
                "out_proj_kernel": _to_jax(
                    state_dict[f"{prefix}.mixer.out_proj.weight"].T, dtype
                ),
            },
        }
        params["layers"].append(layer)

    # LM head (tied to embedding)
    if config.tie_embeddings:
        params["lm_head_weight"] = params["embedding"]["weight"]
    else:
        params["lm_head_weight"] = _to_jax(state_dict["lm_head.weight"], dtype)

    return params


def load_from_pretrained(model_name: str, dtype=jnp.float32):
    """
    Download and load a Mamba2 model from HuggingFace.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID (e.g., "state-spaces/mamba2-130m") or
        short name (e.g., "mamba2-130m").

    Returns
    -------
    model : Mamba2LMHeadModel
    params : dict
    config : Mamba2Config
    """
    from huggingface_hub import hf_hub_download

    # Handle short names
    if not "/" in model_name:
        model_name = f"state-spaces/{model_name}"

    # Download config
    config = Mamba2Config.from_pretrained(model_name)

    # Download weights
    checkpoint_path = hf_hub_download(model_name, "pytorch_model.bin")

    # Load
    params = load_from_torch_checkpoint(config, checkpoint_path, dtype=dtype)
    model = Mamba2LMHeadModel(config)

    return model, params, config


def _to_jax(tensor, dtype=jnp.float32):
    """Convert a PyTorch tensor to JAX array."""
    return jnp.array(tensor.detach().cpu().float().numpy(), dtype=dtype)


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------

def _sample_token(logits, rng_key, temperature=1.0, top_k=50):
    """Sample a token from logits with temperature and top-k."""
    if temperature == 0:
        return jnp.argmax(logits, axis=-1)

    logits = logits / temperature

    if top_k > 0:
        top_k_vals = jax.lax.top_k(logits, top_k)
        min_val = top_k_vals[0][:, -1:]
        logits = jnp.where(logits < min_val, -1e10, logits)

    return jax.random.categorical(rng_key, logits, axis=-1)


# ---------------------------------------------------------------------------
# JIT-compiled decode step using jax.lax.scan over layers
# ---------------------------------------------------------------------------

def _scan_decode_step(stacked_layer_params, non_layer_params, token, conv_states, ssm_states, cfg):
    """
    Single-token decode using jax.lax.scan over layers.

    All layer params and states have a leading (n_layer,) dimension.
    scan compiles one layer body and loops, giving fast compilation and execution.
    """
    from mamba2_jax.ops.causal_conv1d import causal_conv1d_update
    from mamba2_jax.ops.selective_state_update import selective_state_update

    d_inner = cfg.d_inner
    d_ssm = d_inner
    ngs = cfg.ngroups * cfg.d_state
    dtype = jnp.float32 if cfg.residual_in_fp32 else jnp.float32

    x = non_layer_params["embedding"]["weight"][token]  # (1, 1, d_model)

    def layer_fn(x, layer_inputs):
        layer_params, conv_state, ssm_state = layer_inputs
        mixer = layer_params["mixer"]

        residual = x.astype(dtype)
        x_normed = rms_norm(x.astype(jnp.float32), layer_params["norm_weight"])

        hidden = x_normed[:, 0, :]  # (batch, d_model)

        A = -jnp.exp(mixer["A_log"].astype(jnp.float32))
        D = mixer["D"]
        dt_bias = mixer["dt_bias"]

        zxbcdt = hidden @ mixer["in_proj_kernel"]
        z = zxbcdt[:, :d_ssm]
        xBC = zxbcdt[:, d_ssm:2 * d_ssm + 2 * ngs]
        dt = zxbcdt[:, 2 * d_ssm + 2 * ngs:]

        new_conv_state, xBC_out = causal_conv1d_update(
            xBC, conv_state, mixer["conv1d_weight"],
            bias=mixer["conv1d_bias"], activation="silu",
        )

        x_ssm = xBC_out[:, :d_ssm].reshape(-1, cfg.nheads, cfg.headdim)
        B = xBC_out[:, d_ssm:d_ssm + ngs].reshape(-1, cfg.ngroups, cfg.d_state)
        C = xBC_out[:, d_ssm + ngs:].reshape(-1, cfg.ngroups, cfg.d_state)

        new_ssm_state, y = selective_state_update(
            ssm_state, x_ssm, dt, A, B, C,
            D=D, z=None, dt_bias=dt_bias, dt_softplus=True,
        )
        y = y.reshape(-1, d_ssm)

        if cfg.rmsnorm:
            y = rms_norm_gated(
                y, mixer["norm_weight"], z=z,
                group_size=d_ssm // cfg.ngroups,
                norm_before_gate=cfg.norm_before_gate,
            )

        out = y @ mixer["out_proj_kernel"]
        x = residual + out[:, None, :].astype(dtype)
        return x, (new_conv_state, new_ssm_state)

    x, (new_conv_states, new_ssm_states) = jax.lax.scan(
        layer_fn, x, (stacked_layer_params, conv_states, ssm_states),
    )

    x = rms_norm(x.astype(jnp.float32), non_layer_params["norm_f_weight"])
    logits = x @ non_layer_params["lm_head_weight"].T
    next_token_logits = logits[:, 0, :]  # (1, vocab)

    return next_token_logits, new_conv_states, new_ssm_states
