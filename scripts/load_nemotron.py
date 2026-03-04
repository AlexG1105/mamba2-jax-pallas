#!/usr/bin/env python3
"""
scripts/load_nemotron.py

Load a Nemotron-H model from HuggingFace and run validation / text generation.

Usage:
    # Default: nvidia/Nemotron-H-8B-Base-8K (requires ~16GB VRAM in bf16)
    python scripts/load_nemotron.py

    # Validate only (no generation, just check shapes + forward pass)
    python scripts/load_nemotron.py --validate-only

    # With bfloat16 (recommended for 8B model)
    python scripts/load_nemotron.py --dtype bfloat16

    # Custom prompt
    python scripts/load_nemotron.py --dtype bfloat16 --prompt "The meaning of life is"

Prerequisites:
    pip install huggingface-hub torch safetensors transformers
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jax
import jax.numpy as jnp
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Load Nemotron-H and validate/generate")
    parser.add_argument(
        "--model", type=str, default="nvidia/Nemotron-H-8B-Base-8K",
        help="HuggingFace model ID",
    )
    parser.add_argument("--prompt", type=str, default="The meaning of life is", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0=greedy)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--validate-only", action="store_true", help="Only validate loading, skip generation")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
    args = parser.parse_args()

    dtype_map = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}
    dtype = dtype_map[args.dtype]

    print(f"Device: {jax.devices()[0]}")
    print(f"Model:  {args.model}")
    print(f"dtype:  {args.dtype}")
    print()

    # ---- Step 1: Load model ----
    print("Loading model...")
    t0 = time.time()

    from mamba2_jax.models.nemotron_h import load_from_pretrained

    model, params, config = load_from_pretrained(args.model, dtype=dtype)

    t_load = time.time() - t0
    print(f"  Loaded in {t_load:.1f}s")
    print(f"  Config: d_model={config.d_model}, num_layers={config.num_hidden_layers}, "
          f"vocab={config.vocab_size}")
    print(f"  Mamba2: nheads={config.nheads}, d_state={config.d_state}, "
          f"ngroups={config.ngroups}, headdim={config.headdim}, chunk_size={config.chunk_size}")
    print(f"  Attention: {config.num_attention_heads}Q/{config.num_key_value_heads}KV heads, "
          f"head_dim={config.attention_head_dim}")
    print(f"  MLP: intermediate_size={config.intermediate_size}")

    # Layer type summary
    types = config.layer_types
    n_mamba = sum(1 for t in types if t == "mamba2")
    n_attn = sum(1 for t in types if t == "attention")
    n_mlp = sum(1 for t in types if t == "mlp")
    print(f"  Layers: {n_mamba} Mamba2, {n_attn} Attention, {n_mlp} MLP")

    # Count parameters
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params) if hasattr(x, 'size'))
    print(f"  Parameters: {n_params / 1e9:.2f}B")
    print()

    # ---- Step 2: Validate shapes ----
    print("Validating parameter shapes...")
    _validate_shapes(params, config)
    print("  All shapes OK")
    print()

    if args.validate_only:
        print("Running quick forward pass test...")
        test_ids = jnp.array([[1, 2, 3, 4, 5]])
        t0 = time.time()
        logits = model(params, test_ids)
        jax.block_until_ready(logits)
        t_fwd = time.time() - t0
        print(f"  Output shape: {logits.shape}")
        print(f"  Forward pass: {t_fwd:.3f}s")
        print(f"  Logits range: [{float(logits.min()):.4f}, {float(logits.max()):.4f}]")
        has_nan = bool(jnp.any(jnp.isnan(logits)))
        print(f"  Has NaN: {has_nan}")
        print(f"\nValidation {'PASSED' if not has_nan else 'FAILED'}")
        return

    # ---- Step 3: Tokenize ----
    print("Loading tokenizer...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except Exception:
        # Fallback to GPT-NeoX tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    input_ids = tokenizer.encode(args.prompt, return_tensors="np")
    input_ids_jax = jnp.array(input_ids)
    print(f"  Prompt: \"{args.prompt}\"")
    print(f"  Token IDs: {input_ids_jax.shape}")
    print()

    # ---- Step 4: Generate (simple greedy loop) ----
    print(f"Generating {args.max_tokens} tokens (temperature={args.temperature})...")
    rng = jax.random.PRNGKey(args.seed)

    t0 = time.time()
    all_ids = input_ids_jax

    for step in range(args.max_tokens):
        logits = model(params, all_ids)
        jax.block_until_ready(logits)
        next_logits = logits[:, -1, :]

        if args.temperature == 0:
            next_token = jnp.argmax(next_logits, axis=-1)
        else:
            rng, sample_key = jax.random.split(rng)
            next_token = jax.random.categorical(
                sample_key, next_logits / args.temperature, axis=-1
            )

        all_ids = jnp.concatenate([all_ids, next_token[:, None]], axis=1)

        if step % 10 == 0:
            print(f"  Step {step}/{args.max_tokens}...")

    t_gen = time.time() - t0
    output_text = tokenizer.decode(np.array(all_ids[0]), skip_special_tokens=True)
    new_tokens = all_ids.shape[1] - input_ids_jax.shape[1]

    print(f"\n{'=' * 60}")
    print(f"Generated text ({new_tokens} tokens in {t_gen:.2f}s = {new_tokens/t_gen:.1f} tok/s):")
    print(f"{'=' * 60}")
    print(output_text)
    print(f"{'=' * 60}")


def _validate_shapes(params, config):
    """Verify parameter shapes match expected dimensions."""
    cfg = config
    d_model = cfg.d_model
    d_inner = cfg.d_inner
    conv_dim = cfg.conv_dim
    d_in_proj = cfg.d_in_proj

    def check(name, actual, expected):
        assert actual == expected, f"Shape mismatch {name}: {actual} vs {expected}"

    check("embedding", params["embedding"]["weight"].shape, (cfg.vocab_size, d_model))
    check("norm_f", params["norm_f_weight"].shape, (d_model,))
    # lm_head stored pre-transposed: (d_model, vocab)
    check("lm_head", params["lm_head_weight"].shape, (d_model, cfg.vocab_size))

    for i, layer in enumerate(params["layers"]):
        p = f"layer.{i}"
        ltype = layer["type"]
        check(f"{p}.norm", layer["norm_weight"].shape, (d_model,))

        m = layer["mixer"]
        if ltype == "mamba2":
            check(f"{p}.in_proj", m["in_proj_kernel"].shape, (d_model, d_in_proj))
            check(f"{p}.conv1d_w", m["conv1d_weight"].shape, (conv_dim, cfg.d_conv))
            check(f"{p}.conv1d_b", m["conv1d_bias"].shape, (conv_dim,))
            check(f"{p}.dt_bias", m["dt_bias"].shape, (cfg.nheads,))
            check(f"{p}.A_log", m["A_log"].shape, (cfg.nheads,))
            check(f"{p}.D", m["D"].shape, (cfg.nheads,))
            check(f"{p}.norm_w", m["norm_weight"].shape, (d_inner,))
            check(f"{p}.out_proj", m["out_proj_kernel"].shape, (d_inner, d_model))

        elif ltype == "attention":
            n_heads = cfg.num_attention_heads
            n_kv = cfg.num_key_value_heads
            hd = cfg.attention_head_dim
            check(f"{p}.q_proj", m["q_proj_kernel"].shape, (d_model, n_heads * hd))
            check(f"{p}.k_proj", m["k_proj_kernel"].shape, (d_model, n_kv * hd))
            check(f"{p}.v_proj", m["v_proj_kernel"].shape, (d_model, n_kv * hd))
            check(f"{p}.o_proj", m["o_proj_kernel"].shape, (n_heads * hd, d_model))

        elif ltype == "mlp":
            check(f"{p}.up_proj", m["up_proj_kernel"].shape, (d_model, cfg.intermediate_size))
            check(f"{p}.down_proj", m["down_proj_kernel"].shape, (cfg.intermediate_size, d_model))


if __name__ == "__main__":
    main()
