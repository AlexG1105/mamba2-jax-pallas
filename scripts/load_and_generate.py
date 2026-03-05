#!/usr/bin/env python3
"""
scripts/load_and_generate.py

Load a Mamba2 model from HuggingFace and run text generation.

Usage:
    # Default: load mamba2-130m (smallest, ~500MB)
    python scripts/load_and_generate.py

    # Specify a model size
    python scripts/load_and_generate.py --model mamba2-370m

    # All available sizes
    python scripts/load_and_generate.py --model mamba2-130m
    python scripts/load_and_generate.py --model mamba2-370m
    python scripts/load_and_generate.py --model mamba2-780m
    python scripts/load_and_generate.py --model mamba2-1.3b
    python scripts/load_and_generate.py --model mamba2-2.7b

    # Custom prompt
    python scripts/load_and_generate.py --model mamba2-130m --prompt "The meaning of life is"

    # Greedy decoding
    python scripts/load_and_generate.py --model mamba2-130m --temperature 0

    # Just validate loading (no generation)
    python scripts/load_and_generate.py --model mamba2-130m --validate-only

Prerequisites:
    pip install huggingface-hub torch transformers  # for downloading + tokenizer
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
    parser = argparse.ArgumentParser(description="Load Mamba2 from HuggingFace and generate text")
    parser.add_argument(
        "--model", type=str, default="mamba2-130m",
        choices=["mamba2-130m", "mamba2-370m", "mamba2-780m", "mamba2-1.3b", "mamba2-2.7b"],
        help="Model size to load (default: mamba2-130m)",
    )
    parser.add_argument("--prompt", type=str, default="The meaning of life is", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0=greedy)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--validate-only", action="store_true", help="Only validate loading, skip generation")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"])
    args = parser.parse_args()

    model_name = f"state-spaces/{args.model}"
    dtype_map = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}
    dtype = dtype_map[args.dtype]

    print(f"Device: {jax.devices()[0]}")
    print(f"Model:  {model_name}")
    print(f"dtype:  {args.dtype}")
    print()

    # ---- Step 1: Load model ----
    print("Loading model...")
    t0 = time.time()

    from mamba2_jax.models.mamba2_lm import load_from_pretrained, MAMBA2_CONFIGS

    model, params, config = load_from_pretrained(model_name, dtype=dtype)

    t_load = time.time() - t0
    print(f"  Loaded in {t_load:.1f}s")
    print(f"  Config: d_model={config.d_model}, n_layer={config.n_layer}, "
          f"vocab={config.vocab_size_padded}")
    print(f"  Mamba2: nheads={config.nheads}, d_state={config.d_state}, "
          f"ngroups={config.ngroups}, headdim={config.headdim}")

    # Count parameters
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Parameters: {n_params / 1e6:.1f}M")
    print()

    # ---- Step 2: Validate shapes ----
    print("Validating parameter shapes...")
    _validate_shapes(params, config)
    print("  All shapes OK")
    print()

    if args.validate_only:
        # Quick forward pass test
        print("Running quick forward pass test...")
        test_ids = jnp.array([[1, 2, 3, 4, 5]])
        t0 = time.time()
        logits = model(params, test_ids)
        jax.block_until_ready(logits)
        t_fwd = time.time() - t0
        print(f"  Output shape: {logits.shape}")
        print(f"  Forward pass: {t_fwd:.3f}s")
        print(f"  Logits range: [{float(logits.min()):.4f}, {float(logits.max()):.4f}]")
        print("\nValidation PASSED")
        return

    # ---- Step 3: Tokenize ----
    print("Loading tokenizer...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    except ImportError:
        print("ERROR: `transformers` not installed. Install with: pip install transformers")
        print("       (needed for tokenizer; the model itself is loaded without transformers)")
        sys.exit(1)

    input_ids = tokenizer.encode(args.prompt, return_tensors="np")
    input_ids_jax = jnp.array(input_ids)
    print(f"  Prompt: \"{args.prompt}\"")
    print(f"  Token IDs: {input_ids_jax.shape}")
    print()

    # ---- Step 4: Warmup (JIT compilation) ----
    print("Compiling decode step (JIT + scan over layers)...")
    rng = jax.random.PRNGKey(args.seed)
    t0 = time.time()
    warmup_ids = model.generate(
        params, input_ids_jax,
        max_new_tokens=1,
        temperature=args.temperature,
        top_k=args.top_k,
        rng_key=rng,
    )
    jax.block_until_ready(warmup_ids)
    t_compile = time.time() - t0
    print(f"  Compiled in {t_compile:.1f}s")

    # ---- Step 5: Generate ----
    print(f"Generating {args.max_tokens} tokens (temperature={args.temperature}, top_k={args.top_k})...")
    rng = jax.random.PRNGKey(args.seed)

    t0 = time.time()
    output_ids = model.generate(
        params, input_ids_jax,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        rng_key=rng,
    )
    jax.block_until_ready(output_ids)
    t_gen = time.time() - t0

    output_text = tokenizer.decode(np.array(output_ids[0]), skip_special_tokens=True)
    new_tokens = output_ids.shape[1] - input_ids_jax.shape[1]

    print(f"\n{'=' * 60}")
    print(f"Generated text ({new_tokens} tokens in {t_gen:.2f}s = {new_tokens/t_gen:.1f} tok/s):")
    print(f"{'=' * 60}")
    print(output_text)
    print(f"{'=' * 60}")


def _validate_shapes(params, config):
    """Verify all parameter shapes match expected dimensions."""
    cfg = config
    d_model = cfg.d_model
    d_inner = cfg.d_inner
    nheads = cfg.nheads
    conv_dim = cfg.conv_dim
    d_in_proj = cfg.d_in_proj

    def check(name, actual_shape, expected_shape):
        assert actual_shape == expected_shape, (
            f"Shape mismatch for {name}: got {actual_shape}, expected {expected_shape}"
        )

    check("embedding.weight",
          params["embedding"]["weight"].shape,
          (cfg.vocab_size_padded, d_model))

    check("norm_f_weight",
          params["norm_f_weight"].shape,
          (d_model,))

    check("lm_head_weight",
          params["lm_head_weight"].shape,
          (cfg.vocab_size_padded, d_model))

    for i, layer in enumerate(params["layers"]):
        p = f"layer.{i}"
        check(f"{p}.norm_weight", layer["norm_weight"].shape, (d_model,))

        m = layer["mixer"]
        check(f"{p}.in_proj_kernel", m["in_proj_kernel"].shape, (d_model, d_in_proj))
        check(f"{p}.conv1d_weight", m["conv1d_weight"].shape, (conv_dim, cfg.d_conv))
        check(f"{p}.conv1d_bias", m["conv1d_bias"].shape, (conv_dim,))
        check(f"{p}.dt_bias", m["dt_bias"].shape, (nheads,))
        check(f"{p}.A_log", m["A_log"].shape, (nheads,))
        check(f"{p}.D", m["D"].shape, (nheads,))
        check(f"{p}.norm_weight", m["norm_weight"].shape, (d_inner,))
        check(f"{p}.out_proj_kernel", m["out_proj_kernel"].shape, (d_inner, d_model))


if __name__ == "__main__":
    main()
