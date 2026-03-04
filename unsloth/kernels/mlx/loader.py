# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import os
import torch
import mlx.core as mx
from .quantization import MLXQuantizedWeight
from .bridge import torch_to_mlx


# MLX Key Mapping for Llama/Mistral architectures (mlx-lm style)
MLX_KEY_MAPPING = {
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo",
    "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1",
    "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3",
    "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2",
}

# HuggingFace-style quantized keys used by mlx-community repos.
# These repos use the standard HF naming but with .scales/.biases siblings.
HF_QUANTIZED_PROJ_KEYS = [
    "model.layers.{}.self_attn.q_proj",
    "model.layers.{}.self_attn.k_proj",
    "model.layers.{}.self_attn.v_proj",
    "model.layers.{}.self_attn.o_proj",
    "model.layers.{}.mlp.gate_proj",
    "model.layers.{}.mlp.up_proj",
    "model.layers.{}.mlp.down_proj",
]


def _load_mlx_native_weights(model, mlx_weights):
    """
    Load weights in MLX-native format (layers.N.attention.wq style keys).
    Returns (count, True) if successful, (0, False) if keys not recognized.
    """
    # Detect prefix (some models use 'model.layers.N', others just 'layers.N')
    prefix = ""
    for key in mlx_weights.keys():
        if "layers.0" in key:
            if key.startswith("model.layers.0"):
                prefix = "model."
            break

    def get_mlx_val(base_key, suffix=""):
        full_key = f"{prefix}{base_key}{suffix}"
        return mlx_weights.get(full_key)

    num_layers = 0
    while True:
        test_key = f"{prefix}layers.{num_layers}.attention.wq"
        if f"{test_key}.weight" in mlx_weights or test_key in mlx_weights:
            num_layers += 1
        else:
            break

    if num_layers == 0:
        return 0, False

    count = 0
    for i in range(num_layers):
        for hf_temp, mlx_temp in MLX_KEY_MAPPING.items():
            hf_key = hf_temp.format(i)
            mlx_base = mlx_temp.format(i)

            param = None
            for name, p in model.named_parameters():
                if name == hf_key:
                    param = p
                    break

            if param is None:
                continue

            w = get_mlx_val(mlx_base, ".weight") or get_mlx_val(mlx_base)
            s = get_mlx_val(mlx_base, ".scales")
            b = get_mlx_val(mlx_base, ".biases")

            if w is not None and s is not None:
                bits = 4
                group_size = 64

                bits_arr = get_mlx_val(mlx_base, ".bits")
                if bits_arr is not None:
                    bits = bits_arr.item()

                gs_arr = get_mlx_val(mlx_base, ".group_size")
                if gs_arr is not None:
                    group_size = gs_arr.item()

                param._mlx_cache = MLXQuantizedWeight(
                    weight=w, scales=s, biases=b, group_size=group_size, bits=bits
                )
                count += 1

    return count, True


def _load_hf_quantized_weights(model, mlx_weights, config=None):
    """
    Load weights in HuggingFace format with quantized .scales/.biases siblings.
    This handles mlx-community repos that use HF key naming
    (e.g. model.layers.0.self_attn.q_proj.weight + .scales + .biases).

    Also loads non-quantized weights (embed_tokens, lm_head, layer norms)
    and caches them as MLX arrays for the bridge fast-path.

    Returns (quantized_count, non_quantized_count, True) if successful,
    (0, 0, False) if keys not recognized.
    """
    # Detect if this is a HF-style quantized file by checking for .scales keys
    has_hf_quantized = any(k.endswith(".scales") for k in mlx_weights.keys())
    if not has_hf_quantized:
        return 0, 0, False

    # Detect number of layers
    num_layers = 0
    while True:
        test_key = f"model.layers.{num_layers}.self_attn.q_proj.weight"
        if test_key in mlx_weights:
            num_layers += 1
        else:
            break

    if num_layers == 0:
        return 0, 0, False

    # Extract default quantization config (bits, group_size) from model config if available
    default_bits = 4
    default_group_size = 64
    if config is not None:
        qconfig = getattr(config, "quantization", None) or {}
        if isinstance(qconfig, dict):
            default_bits = qconfig.get("bits", default_bits)
            default_group_size = qconfig.get("group_size", default_group_size)

    # Build a dict of model parameter names -> parameter objects for fast lookup
    param_dict = dict(model.named_parameters())

    # --- 1. Load quantized projection layers ---
    quantized_count = 0
    for i in range(num_layers):
        for key_template in HF_QUANTIZED_PROJ_KEYS:
            base_key = key_template.format(i)
            w_key = f"{base_key}.weight"
            s_key = f"{base_key}.scales"
            b_key = f"{base_key}.biases"

            w = mlx_weights.get(w_key)
            s = mlx_weights.get(s_key)
            b = mlx_weights.get(b_key)

            if w is None or s is None:
                continue

            param = param_dict.get(w_key)
            if param is None:
                continue

            # Check for per-layer bits/group_size metadata
            bits = default_bits
            group_size = default_group_size

            bits_arr = mlx_weights.get(f"{base_key}.bits")
            if bits_arr is not None:
                bits = bits_arr.item()

            gs_arr = mlx_weights.get(f"{base_key}.group_size")
            if gs_arr is not None:
                group_size = gs_arr.item()

            param._mlx_cache = MLXQuantizedWeight(
                weight=w, scales=s, biases=b, group_size=group_size, bits=bits
            )
            quantized_count += 1

    # --- 2. Load non-quantized weights (embed_tokens, lm_head, layer norms) ---
    # These don't have .scales/.biases siblings - they are full-precision MLX arrays.
    # We load them into the model parameters AND cache them as MLX arrays for the bridge.
    non_quantized_count = 0

    # Collect all weight keys that are NOT part of quantized triplets
    quantized_base_keys = set()
    for i in range(num_layers):
        for key_template in HF_QUANTIZED_PROJ_KEYS:
            base_key = key_template.format(i)
            quantized_base_keys.add(f"{base_key}.weight")
            quantized_base_keys.add(f"{base_key}.scales")
            quantized_base_keys.add(f"{base_key}.biases")
            quantized_base_keys.add(f"{base_key}.bits")
            quantized_base_keys.add(f"{base_key}.group_size")

    for key, mlx_val in mlx_weights.items():
        if key in quantized_base_keys:
            continue
        # Skip metadata keys (bits, group_size, etc.)
        if key.endswith(".bits") or key.endswith(".group_size"):
            continue

        param = param_dict.get(key)
        if param is None:
            continue

        # Convert MLX array to PyTorch tensor and load into parameter
        try:
            import numpy as np
            np_arr = np.array(mlx_val, copy=False)

            # Map MLX dtypes to torch dtypes
            if mlx_val.dtype == mx.bfloat16:
                # bfloat16 needs special handling - convert via float32
                np_arr = np.array(mlx_val.astype(mx.float32), copy=False)
                torch_tensor = torch.from_numpy(np_arr).to(torch.bfloat16)
            elif mlx_val.dtype == mx.float16:
                torch_tensor = torch.from_numpy(np_arr).to(torch.float16)
            else:
                torch_tensor = torch.from_numpy(np_arr)
        except Exception:
            # Fallback: convert through float32
            import numpy as np
            np_arr = np.array(mlx_val.astype(mx.float32), copy=False)
            torch_tensor = torch.from_numpy(np_arr).to(param.dtype)

        # Assign to the model parameter
        with torch.no_grad():
            param.data.copy_(torch_tensor)

        # Cache as MLX array for bridge fast-path (non-quantized, full precision)
        param._mlx_cache = mlx_val
        non_quantized_count += 1

    return quantized_count, non_quantized_count, True


def _load_sharded_mlx_weights(model, index_path, config=None):
    """
    Load weights from sharded safetensors files using the index JSON.
    Returns True if weights were loaded, False otherwise.
    """
    import json

    weights_dir = os.path.dirname(index_path)
    with open(index_path, "r") as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})
    if not weight_map:
        return False

    # Group keys by shard file
    shard_files = {}
    for key, shard_file in weight_map.items():
        shard_files.setdefault(shard_file, []).append(key)

    # Load all shards and merge into a single dict
    all_weights = {}
    for shard_file in shard_files:
        shard_path = os.path.join(weights_dir, shard_file)
        if not os.path.exists(shard_path):
            import logging
            logging.getLogger("unsloth").warning(
                f"Shard file {shard_file} not found at {shard_path}"
            )
            continue
        shard_weights = mx.load(shard_path)
        all_weights.update(shard_weights)

    if not all_weights:
        return False

    print(
        f"Unsloth: Loaded {len(all_weights)} tensors from "
        f"{len(shard_files)} sharded files."
    )

    # Try MLX-native format first
    count, success = _load_mlx_native_weights(model, all_weights)
    if success:
        print(f"Unsloth: Successfully mapped {count} MLX-native weights to model.")
        return True

    # Try HuggingFace-style quantized format
    model_config = config if config is not None else getattr(model, "config", None)
    q_count, nq_count, success = _load_hf_quantized_weights(
        model, all_weights, config=model_config
    )
    if success:
        print(
            f"Unsloth: Loaded {q_count} quantized + {nq_count} non-quantized weights "
            f"from sharded HuggingFace-format MLX files."
        )
        return True

    return False


def load_mlx_weights(model, weights_path, config=None):
    """
    Loads pre-quantized weights from an MLX-format file (safetensors or npz)
    and populates model.param._mlx_cache for zero-copy optimized inference.

    Supports two formats:
    1. MLX-native format: keys like layers.N.attention.wq (from mlx-lm convert)
    2. HuggingFace format with quantization: keys like model.layers.N.self_attn.q_proj.weight
       with .scales/.biases siblings (from mlx-community repos)

    Also handles sharded safetensors (model.safetensors.index.json).

    For HF-format quantized files, also loads non-quantized weights (embed_tokens,
    lm_head, layer norms) directly into model parameters.

    Returns True if weights were loaded, False if format not recognized.
    """
    if not os.path.exists(weights_path):
        return False

    # Handle sharded safetensors index files
    if weights_path.endswith(".index.json"):
        return _load_sharded_mlx_weights(model, weights_path, config=config)

    print(
        f"Unsloth: Loading pre-quantized MLX weights from {os.path.basename(weights_path)}..."
    )

    # Load MLX weights
    mlx_weights = mx.load(weights_path)

    # Try MLX-native format first (layers.N.attention.wq style)
    count, success = _load_mlx_native_weights(model, mlx_weights)
    if success:
        print(f"Unsloth: Successfully mapped {count} MLX-native weights to model.")
        return True

    # Try HuggingFace-style quantized format (mlx-community repos)
    # This also loads non-quantized weights like embed_tokens, lm_head, norms
    model_config = config if config is not None else getattr(model, "config", None)
    q_count, nq_count, success = _load_hf_quantized_weights(
        model, mlx_weights, config=model_config
    )
    if success:
        print(
            f"Unsloth: Loaded {q_count} quantized + {nq_count} non-quantized weights "
            f"from HuggingFace-format MLX file."
        )
        return True

    # Neither format matched
    import logging
    logger = logging.getLogger("unsloth")
    logger.debug(
        f"Weights in {os.path.basename(weights_path)} not recognized as MLX-native "
        "or HuggingFace-quantized format. Will quantize on-the-fly."
    )
    return False
