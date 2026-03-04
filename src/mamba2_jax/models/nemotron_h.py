"""
mamba2_jax/models/nemotron_h.py

Nemotron-H hybrid language model in JAX.

Architecture:
    embedding → [norm → layer_i] × 52 → final_norm → lm_head

Where each layer_i is one of:
    'M' = Mamba2 block  (24 layers, ngroups=8, chunk_size=128)
    '-' = MLP block     (24 layers, squared ReLU)
    '*' = Attention block (4 layers, GQA 32Q/8KV heads)

Determined by ``hybrid_override_pattern`` in config.

Supports loading weights from HuggingFace ``nvidia/Nemotron-H-8B-Base-8K``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from mamba2_jax.ops.rms_norm import rms_norm_gated


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class NemotronHConfig:
    """Configuration for Nemotron-H hybrid model."""
    hidden_size: int = 4096
    num_hidden_layers: int = 52
    vocab_size: int = 131072

    # Mamba2 SSM params
    ssm_state_size: int = 128       # d_state
    mamba_head_dim: int = 64        # headdim
    mamba_num_heads: int = 128      # nheads → d_inner = 128 * 64 = 8192
    n_groups: int = 8               # ngroups
    expand: int = 2
    conv_kernel: int = 4            # d_conv
    chunk_size: int = 128
    mamba_proj_bias: bool = False
    use_conv_bias: bool = True
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 0.0001
    time_step_limit: tuple = (0.0, float("inf"))

    # Attention params
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    attention_head_dim: int = 128
    attention_bias: bool = False

    # MLP params
    intermediate_size: int = 21504
    mlp_bias: bool = False

    # General
    rms_norm_eps: float = 1e-5
    residual_in_fp32: bool = False
    tie_word_embeddings: bool = False
    hybrid_override_pattern: str = "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"

    # Derived
    @property
    def d_model(self):
        return self.hidden_size

    @property
    def d_inner(self):
        return self.mamba_num_heads * self.mamba_head_dim

    @property
    def d_state(self):
        return self.ssm_state_size

    @property
    def headdim(self):
        return self.mamba_head_dim

    @property
    def nheads(self):
        return self.mamba_num_heads

    @property
    def ngroups(self):
        return self.n_groups

    @property
    def d_conv(self):
        return self.conv_kernel

    @property
    def conv_dim(self):
        return self.d_inner + 2 * self.n_groups * self.ssm_state_size

    @property
    def d_in_proj(self):
        return 2 * self.d_inner + 2 * self.n_groups * self.ssm_state_size + self.mamba_num_heads

    @property
    def layer_types(self):
        """Parse hybrid_override_pattern into a list of layer types."""
        mapping = {'M': 'mamba2', '-': 'mlp', '*': 'attention'}
        return [mapping[c] for c in self.hybrid_override_pattern]

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "NemotronHConfig":
        """Load config from HuggingFace model repo or local path."""
        try:
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(model_name_or_path, "config.json")
        except Exception:
            import os
            config_path = os.path.join(model_name_or_path, "config.json")

        with open(config_path) as f:
            data = json.load(f)

        tsl = data.get("time_step_limit", [0.0, "Infinity"])
        tsl = (tsl[0], float("inf") if tsl[1] == "Infinity" else tsl[1])

        return cls(
            hidden_size=data.get("hidden_size", 4096),
            num_hidden_layers=data.get("num_hidden_layers", 52),
            vocab_size=data.get("vocab_size", 131072),
            ssm_state_size=data.get("ssm_state_size", 128),
            mamba_head_dim=data.get("mamba_head_dim", 64),
            mamba_num_heads=data.get("mamba_num_heads", 128),
            n_groups=data.get("n_groups", 8),
            expand=data.get("expand", 2),
            conv_kernel=data.get("conv_kernel", 4),
            chunk_size=data.get("chunk_size", 128),
            mamba_proj_bias=data.get("mamba_proj_bias", False),
            use_conv_bias=data.get("use_conv_bias", True),
            time_step_min=data.get("time_step_min", 0.001),
            time_step_max=data.get("time_step_max", 0.1),
            time_step_floor=data.get("time_step_floor", 0.0001),
            time_step_limit=tsl,
            num_attention_heads=data.get("num_attention_heads", 32),
            num_key_value_heads=data.get("num_key_value_heads", 8),
            attention_head_dim=data.get("attention_head_dim", 128),
            attention_bias=data.get("attention_bias", False),
            intermediate_size=data.get("intermediate_size", 21504),
            mlp_bias=data.get("mlp_bias", False),
            rms_norm_eps=data.get("rms_norm_eps", 1e-5),
            residual_in_fp32=data.get("residual_in_fp32", False),
            tie_word_embeddings=data.get("tie_word_embeddings", False),
            hybrid_override_pattern=data.get(
                "hybrid_override_pattern",
                "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-",
            ),
        )


# ---------------------------------------------------------------------------
# Standalone helper
# ---------------------------------------------------------------------------

def rms_norm(x, weight, eps=1e-5):
    """Standard RMSNorm (no gating)."""
    rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    return x / rms * weight


# ---------------------------------------------------------------------------
# Layer forward functions
# ---------------------------------------------------------------------------

def _mamba2_block_forward(x, mixer_params, cfg):
    """
    Mamba2 block forward (chunked scan). Same as mamba2_lm but uses
    NemotronHConfig fields.
    """
    from mamba2_jax.ops.causal_conv1d import causal_conv1d
    from mamba2_jax.modules.mamba2 import _ssd_combined_fwd

    batch, seqlen, _ = x.shape
    d_inner = cfg.d_inner
    d_ssm = d_inner
    ngs = cfg.ngroups * cfg.d_state

    A = -jnp.exp(mixer_params["A_log"].astype(jnp.float32))
    D = mixer_params["D"]
    dt_bias = mixer_params["dt_bias"]

    # 1. Input projection
    zxbcdt = x @ mixer_params["in_proj_kernel"]

    # 2. Split: [z, xBC, dt]
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
        D=D, z=None,
        dt_bias=dt_bias,
        dt_softplus=True,
        dt_limit=cfg.time_step_limit,
        return_final_states=False,
    )
    y = scan_result.reshape(batch, seqlen, d_ssm)

    # 6. RMSNorm gating
    y = rms_norm_gated(
        y, mixer_params["norm_weight"], z=z,
        group_size=d_ssm // cfg.ngroups,
        norm_before_gate=False,
        eps=cfg.rms_norm_eps,
    )

    # 7. Output projection
    out = y @ mixer_params["out_proj_kernel"]
    return out


def _attention_forward(x, attn_params, cfg):
    """
    Grouped-query attention forward pass.

    Uses separate Q/K/V projections. No positional encoding (Nemotron-H
    uses no explicit positional encoding for attention layers — they rely
    on the Mamba2 layers for position awareness).
    """
    batch, seqlen, _ = x.shape
    head_dim = cfg.attention_head_dim
    n_heads = cfg.num_attention_heads
    n_kv_heads = cfg.num_key_value_heads
    n_groups = n_heads // n_kv_heads  # heads per KV group

    # Project Q, K, V
    q = x @ attn_params["q_proj_kernel"]  # (B, L, n_heads * head_dim)
    k = x @ attn_params["k_proj_kernel"]  # (B, L, n_kv_heads * head_dim)
    v = x @ attn_params["v_proj_kernel"]  # (B, L, n_kv_heads * head_dim)

    # Reshape
    q = q.reshape(batch, seqlen, n_heads, head_dim)
    k = k.reshape(batch, seqlen, n_kv_heads, head_dim)
    v = v.reshape(batch, seqlen, n_kv_heads, head_dim)

    # Expand KV heads for GQA: (B, L, n_kv_heads, head_dim) -> (B, L, n_heads, head_dim)
    k = jnp.repeat(k, n_groups, axis=2)
    v = jnp.repeat(v, n_groups, axis=2)

    # Attention: (B, H, L, D) format
    q = jnp.transpose(q, (0, 2, 1, 3))  # (B, H, L, D)
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))

    # Scaled dot-product attention with causal mask
    scale = 1.0 / math.sqrt(head_dim)
    attn = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) * scale

    # Causal mask
    mask = jnp.tril(jnp.ones((seqlen, seqlen)))
    attn = jnp.where(mask[None, None, :, :], attn, -1e9)
    attn = jax.nn.softmax(attn, axis=-1)

    # Apply attention to values
    out = jnp.matmul(attn, v)  # (B, H, L, D)
    out = jnp.transpose(out, (0, 2, 1, 3))  # (B, L, H, D)
    out = out.reshape(batch, seqlen, n_heads * head_dim)

    # Output projection
    out = out @ attn_params["o_proj_kernel"]
    return out


def _mlp_forward(x, mlp_params, cfg):
    """
    MLP forward with squared ReLU activation.

    Nemotron-H uses: out = down_proj(relu(up_proj(x))^2)
    """
    h = x @ mlp_params["up_proj_kernel"]
    h = jax.nn.relu(h) ** 2  # squared ReLU
    out = h @ mlp_params["down_proj_kernel"]
    return out


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class NemotronHModel:
    """
    Nemotron-H hybrid language model for inference.

    Parameters are stored as a nested dict::

        params = {
            "embedding": {"weight": (vocab, d_model)},
            "layers": [
                {
                    "type": "mamba2" | "attention" | "mlp",
                    "norm_weight": (d_model,),
                    "mixer": { ... layer-type-specific params ... },
                },
                ...
            ],
            "norm_f_weight": (d_model,),
            "lm_head_weight": (vocab, d_model),
        }
    """

    def __init__(self, config: NemotronHConfig):
        self.config = config

    def __call__(self, params, input_ids):
        """
        Forward pass.

        Parameters
        ----------
        params : dict
        input_ids : (batch, seqlen) int

        Returns
        -------
        logits : (batch, seqlen, vocab_size)
        """
        cfg = self.config

        x = params["embedding"]["weight"][input_ids]
        dtype = jnp.float32 if cfg.residual_in_fp32 else x.dtype

        for i, layer_params in enumerate(params["layers"]):
            residual = x.astype(dtype)
            x_normed = rms_norm(
                x.astype(jnp.float32), layer_params["norm_weight"],
                eps=cfg.rms_norm_eps,
            )
            # Cast back to param dtype (e.g., bf16) for block computation
            x_normed = x_normed.astype(dtype)

            layer_type = layer_params["type"]
            if layer_type == "mamba2":
                x_out = _mamba2_block_forward(x_normed, layer_params["mixer"], cfg)
            elif layer_type == "attention":
                x_out = _attention_forward(x_normed, layer_params["mixer"], cfg)
            elif layer_type == "mlp":
                x_out = _mlp_forward(x_normed, layer_params["mixer"], cfg)
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

            x = residual + x_out.astype(dtype)

        x = rms_norm(
            x.astype(jnp.float32), params["norm_f_weight"],
            eps=cfg.rms_norm_eps,
        ).astype(dtype)
        # lm_head_weight is stored pre-transposed: (d_model, vocab)
        logits = x @ params["lm_head_weight"]
        return logits


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def _to_jax(tensor, dtype=jnp.float32):
    """Convert a PyTorch tensor to JAX array."""
    return jnp.array(tensor.detach().cpu().float().numpy(), dtype=dtype)


def load_from_torch_checkpoint(config: NemotronHConfig, checkpoint_path: str, dtype=jnp.float32):
    """
    Load Nemotron-H weights from a PyTorch checkpoint or safetensors.

    Parameters
    ----------
    config : NemotronHConfig
    checkpoint_path : str
        Path to model directory containing safetensors/bin files.
    dtype : jnp.dtype
        Target dtype.

    Returns
    -------
    params : dict
    """
    import torch
    import os
    import glob

    # Load state dict (handle safetensors or pytorch_model.bin)
    if os.path.isdir(checkpoint_path):
        safetensor_files = sorted(glob.glob(os.path.join(checkpoint_path, "*.safetensors")))
        bin_files = sorted(glob.glob(os.path.join(checkpoint_path, "pytorch_model*.bin")))

        if safetensor_files:
            from safetensors.torch import load_file
            state_dict = {}
            for f in safetensor_files:
                state_dict.update(load_file(f))
        elif bin_files:
            state_dict = {}
            for f in bin_files:
                state_dict.update(torch.load(f, map_location="cpu", weights_only=True))
        else:
            raise FileNotFoundError(f"No model files found in {checkpoint_path}")
    else:
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    layer_types = config.layer_types

    params = {
        "embedding": {
            "weight": _to_jax(state_dict["backbone.embeddings.weight"], dtype),
        },
        "layers": [],
        "norm_f_weight": _to_jax(state_dict["backbone.norm_f.weight"], dtype),
    }

    for i in range(config.num_hidden_layers):
        prefix = f"backbone.layers.{i}"
        ltype = layer_types[i]

        layer = {
            "type": ltype,
            "norm_weight": _to_jax(state_dict[f"{prefix}.norm.weight"], dtype),
            "mixer": {},
        }

        if ltype == "mamba2":
            m = layer["mixer"]
            m["in_proj_kernel"] = _to_jax(
                state_dict[f"{prefix}.mixer.in_proj.weight"].T, dtype
            )
            m["conv1d_weight"] = _to_jax(
                state_dict[f"{prefix}.mixer.conv1d.weight"].squeeze(1), dtype
            )
            m["conv1d_bias"] = _to_jax(
                state_dict[f"{prefix}.mixer.conv1d.bias"], dtype
            )
            m["dt_bias"] = _to_jax(
                state_dict[f"{prefix}.mixer.dt_bias"], jnp.float32
            )
            m["A_log"] = _to_jax(
                state_dict[f"{prefix}.mixer.A_log"], dtype
            )
            m["D"] = _to_jax(
                state_dict[f"{prefix}.mixer.D"], dtype
            )
            m["norm_weight"] = _to_jax(
                state_dict[f"{prefix}.mixer.norm.weight"], dtype
            )
            m["out_proj_kernel"] = _to_jax(
                state_dict[f"{prefix}.mixer.out_proj.weight"].T, dtype
            )

        elif ltype == "attention":
            m = layer["mixer"]
            # Separate Q/K/V/O projections, transpose (out, in) -> (in, out)
            m["q_proj_kernel"] = _to_jax(
                state_dict[f"{prefix}.mixer.q_proj.weight"].T, dtype
            )
            m["k_proj_kernel"] = _to_jax(
                state_dict[f"{prefix}.mixer.k_proj.weight"].T, dtype
            )
            m["v_proj_kernel"] = _to_jax(
                state_dict[f"{prefix}.mixer.v_proj.weight"].T, dtype
            )
            m["o_proj_kernel"] = _to_jax(
                state_dict[f"{prefix}.mixer.o_proj.weight"].T, dtype
            )

        elif ltype == "mlp":
            m = layer["mixer"]
            m["up_proj_kernel"] = _to_jax(
                state_dict[f"{prefix}.mixer.up_proj.weight"].T, dtype
            )
            m["down_proj_kernel"] = _to_jax(
                state_dict[f"{prefix}.mixer.down_proj.weight"].T, dtype
            )

        params["layers"].append(layer)

    # LM head (NOT tied in Nemotron-H)
    # Store pre-transposed: (d_model, vocab) to avoid runtime transpose of large matrix
    if config.tie_word_embeddings:
        params["lm_head_weight"] = params["embedding"]["weight"].T
    else:
        # lm_head.weight is (vocab, d_model), we want (d_model, vocab)
        params["lm_head_weight"] = _to_jax(state_dict["lm_head.weight"].T, dtype)

    return params


def load_from_pretrained(model_name: str, dtype=jnp.float32):
    """
    Download and load a Nemotron-H model from HuggingFace.

    Parameters
    ----------
    model_name : str
        e.g., "nvidia/Nemotron-H-8B-Base-8K"

    Returns
    -------
    model : NemotronHModel
    params : dict
    config : NemotronHConfig
    """
    from huggingface_hub import snapshot_download

    config = NemotronHConfig.from_pretrained(model_name)
    local_dir = snapshot_download(model_name)
    params = load_from_torch_checkpoint(config, local_dir, dtype=dtype)
    model = NemotronHModel(config)

    return model, params, config
