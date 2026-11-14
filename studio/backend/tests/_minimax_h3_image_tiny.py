# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A tiny random MiniMax-H3-Image release in the NATIVE layout (MiniMaxH3DiTModel keys, per-head interleaved fused
qkv, [gate; value] fc1, stacked PDD heads, native VAE keys), built offline from a random diffusers-layout model, plus
a tiny Qwen3-VL conditioner (51 layers: the DiT reads hidden_states[50]) and an offline tokenizer / processor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TINY_TRANSFORMER_NATIVE = {
    "_class_name": "MiniMaxH3DiTModel",
    "hidden_size": 64,
    "num_layers": 2,
    "token_refiner_num_layers": 2,
    "num_attention_heads": 2,
    "attention_head_dim": 32,
    "ffn_hidden_size": 128,
    "latents_dim": 24,
    "audio_latents_dim": 32,
    "patch_size": [1, 2, 2],
    "text_dim": 48,
    "timestep_input_dim": 16,
    "time_embed_hidden_size": 64,
    "time_embed_dim": 32,
    "rope_inv_freq_len": 4,
    "norm_eps": 1e-05,
    "qk_norm_eps": 1e-05,
    "final_norm_eps": 1e-05,
    "audio_enabled": False,
}

TINY_VAE_NATIVE = {
    "_class_name": "MiniMaxH3ImageVAE",
    "latent_channels": 24,
    "latents_mean": [0.1 * (i % 5) for i in range(24)],
    "latents_std": [1.0 + 0.05 * i for i in range(24)],
    "vae_decoder_tiling": 1,
    "vae_encoder_tiling": 1,
    "vae_tile_overlap_min": 64,
    "vae_tile_size": 256,
    "vit_decoder_2d": True,
    "vit_decoder_kwargs": {
        "dim_head": 32,
        "heads": 2,
        "num_layers": 2,
        "rope_dim_ratio": 0.5,
        "rope_theta": 100.0,
    },
    # Not in the release (it uses the class defaults): shrinks the CNN encoder for CPU tests, 16x like the release.
    "block_out_channels": [32, 32, 32, 32],
    "layers_per_block": 1,
    "spatial_downsample_factors": [2, 2, 2, 2],
    "temporal_downsample_factors": [1, 2, 2, 1],
}

SPECIAL = ["<|endoftext|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"]


def interleave_qkv(q: Any, k: Any, v: Any, heads: int, head_dim: int) -> Any:
    """Inverse of the converter's reorder: [q_all; k_all; v_all] -> per-head [q k v]."""
    import torch

    parts = [t.reshape(heads, head_dim, *t.shape[1:]) for t in (q, k, v)]
    return torch.cat(parts, dim = 1).reshape(heads * 3 * head_dim, *q.shape[1:])


def to_native_transformer(state: dict, heads: int, head_dim: int) -> dict:
    """diffusers-layout DiT state -> the release's native keys (the inverse of the PR converter)."""
    import torch

    out = {}
    done = set()
    renames = [
        ("time_embedder.linear_1.", "time_embedder.proj_in."),
        ("time_embedder.linear_2.", "time_embedder.proj_out."),
        ("proj_in.", "video_patch_proj."),
        ("context_embedder.", "condition_proj."),
        ("norm_out.norm.", "final_layer.norm."),
        ("norm_out.linear.", "final_layer.adaln_proj.linear."),
        (".attn.norm_q.", ".attn.q_norm."),
        (".attn.norm_k.", ".attn.k_norm."),
        (".attn.to_out.0.", ".attn.out_proj."),
        (".ff.net.2.", ".mlp.fc2."),
    ]
    for key, tensor in state.items():
        if key in done or key.startswith("pdd_proj_out."):
            continue
        name = key
        if name.startswith("token_refiner.refiner_blocks."):
            name = "token_refiner.blocks." + name[len("token_refiner.refiner_blocks.") :]
        elif name.startswith("transformer_blocks."):
            name = "blocks." + name[len("transformer_blocks.") :]
        if key.endswith(".attn.to_q.weight"):
            prefix = key[: -len("to_q.weight")]
            q, k, v = (
                state[prefix + "to_q.weight"],
                state[prefix + "to_k.weight"],
                state[prefix + "to_v.weight"],
            )
            done.update({prefix + "to_k.weight", prefix + "to_v.weight"})
            native_prefix = name[: -len("to_q.weight")]
            out[native_prefix + "qkv_proj.weight"] = interleave_qkv(q, k, v, heads, head_dim)
            continue
        if key.endswith((".attn.to_k.weight", ".attn.to_v.weight")):
            continue
        if key.endswith(".ff.net.0.proj.weight"):
            value, gate = tensor.chunk(2, dim = 0)
            out[name.replace(".ff.net.0.proj.weight", ".mlp.fc1.weight")] = torch.cat(
                [gate, value]
            ).contiguous()
            continue
        for old, new in renames:
            if name.startswith(old) or old.startswith("."):
                name = name.replace(old, new) if old.startswith(".") else new + name[len(old) :]
        out[name] = tensor.contiguous()
    # The release ships its rope buffer and a base output head that the PDD heads replace.
    return out


def to_native_vae(state: dict, heads: int, head_dim: int) -> dict:
    import torch

    out = {}
    for key, tensor in state.items():
        if ".attn.to_k." in key or ".attn.to_v." in key:
            continue
        if ".attn.to_q." in key:
            prefix, suffix = key.split(".attn.to_q.")
            q = tensor
            k = state[f"{prefix}.attn.to_k.{suffix}"]
            v = state[f"{prefix}.attn.to_v.{suffix}"]
            out[f"{prefix}.attn.to_qkv.{suffix}"] = interleave_qkv(q, k, v, heads, head_dim)
            continue
        name = key
        if name.startswith("encoder.down_blocks."):
            level, rest = name[len("encoder.down_blocks.") :].split(".", 1)
            rest = rest.replace("resnets.", "block.", 1).replace(
                "conv_shortcut.", "nin_shortcut.", 1
            )
            rest = rest.replace("downsamplers.0.", "downsample.", 1)
            name = f"encoder.down.{level}.{rest}"
        name = name.replace("decoder.proj_in.", "decoder.x_embedder.")
        name = name.replace(".attn.to_out.0.", ".attn.to_out.")
        if ".ff.net.0.proj." in name:
            up, gate = tensor.chunk(2, dim = 0)
            tensor = torch.cat([gate, up]).contiguous()
        name = name.replace(".ff.net.0.proj.", ".ff.w1.")
        name = name.replace(".ff.net.2.", ".ff.w2.")
        out[name] = tensor.contiguous()
    return out


def randomize(
    module: Any,
    seed: int,
    std: float = 0.05,
) -> None:
    import torch
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator = g, dtype = torch.float32).to(p.dtype) * std)


def make_tokenizer_and_processor(out: Path) -> tuple[Any, Any]:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = [chr(c) for c in range(32, 127)] + ["Picture", ":", "the", "a", "cat", "<", ">"]
    vocab = {w: i for i, w in enumerate(dict.fromkeys(SPECIAL + words + ["[UNK]"]))}
    tok = Tokenizer(models.WordLevel(vocab = vocab, unk_token = "[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Split(pattern = " ", behavior = "isolated")
    fast = PreTrainedTokenizerFast(tokenizer_object = tok, unk_token = "[UNK]")
    fast.add_special_tokens({"additional_special_tokens": SPECIAL})
    (out / "tokenizer").mkdir(parents = True, exist_ok = True)
    fast.save_pretrained(str(out / "tokenizer"))
    processor = None
    try:
        from transformers import Qwen3VLProcessor
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

        image_processor = Qwen2VLImageProcessor(
            patch_size = 16,
            temporal_patch_size = 2,
            merge_size = 2,
            min_pixels = 32 * 32,
            max_pixels = 2048 * 2048,
        )
        try:
            from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
            video = Qwen3VLVideoProcessor()
        except Exception:  # noqa: BLE001 - older transformers
            video = None
        kwargs = {"image_processor": image_processor, "tokenizer": fast}
        if video is not None:
            kwargs["video_processor"] = video
        processor = Qwen3VLProcessor(**kwargs)
        (out / "processor").mkdir(parents = True, exist_ok = True)
        processor.save_pretrained(str(out / "processor"))
    except Exception:  # noqa: BLE001 - editing tests skip without it
        processor = None
    return fast, processor


def make_text_encoder(
    tokenizer: Any,
    text_dim: int = 48,
    layers: int = 51,
    seed: int = 3,
) -> Any:
    import torch
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    ids = {t: tokenizer.convert_tokens_to_ids(t) for t in SPECIAL}
    config = Qwen3VLConfig(
        text_config = {
            "hidden_size": text_dim,
            "intermediate_size": 64,
            "num_hidden_layers": layers,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 24,
            "vocab_size": len(tokenizer),
            "max_position_embeddings": 8192,
            "rope_scaling": {
                "rope_type": "default",
                "mrope_section": [4, 4, 4],
                "mrope_interleaved": True,
            },
        },
        vision_config = {
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": text_dim,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 64,
            "deepstack_visual_indexes": [0],
        },
        image_token_id = ids["<|image_pad|>"],
        video_token_id = ids["<|video_pad|>"],
        vision_start_token_id = ids["<|vision_start|>"],
        vision_end_token_id = ids["<|vision_end|>"],
    )
    torch.manual_seed(seed)
    model = Qwen3VLForConditionalGeneration(config).eval()
    randomize(model, seed, std = 0.05)
    return model


def make_tiny_release(out: Path, seed: int = 0) -> dict:
    """Writes the release layout under ``out``; returns the diffusers-layout reference modules."""
    import torch
    from safetensors.torch import save_file

    from core.inference import diffusion_minimax_h3_image as h3i

    out.mkdir(parents = True, exist_ok = True)
    cfg = h3i.transformer_config_from_native(TINY_TRANSFORMER_NATIVE)
    torch.manual_seed(seed)
    dense = h3i.transformer_class()(**cfg).eval()
    randomize(dense, seed)
    # Release precision: bf16 block stack, fp32 embedders / heads.
    state = {}
    for k, v in dense.state_dict().items():
        if k.startswith("rope."):
            continue
        state[k] = (
            v.to(torch.float32)
            if k.startswith(("proj_in.", "time_embedder.", "pdd_proj_out."))
            else v.to(torch.bfloat16)
        )
    dense.load_state_dict(state, strict = False, assign = True)
    native = to_native_transformer(state, cfg["num_attention_heads"], cfg["attention_head_dim"])
    native["rope.inv_freq"] = (
        dense.rope.inv_freq.clone() if hasattr(dense.rope, "inv_freq") else torch.zeros(4)
    )
    hidden, patch_dim = cfg["hidden_size"], 24 * 4
    native["final_layer.video_out.weight"] = torch.zeros(patch_dim, hidden)
    native["final_layer.video_out.bias"] = torch.zeros(patch_dim)
    tdir = out / "transformer"
    tdir.mkdir(exist_ok = True)
    keys = sorted(native)
    half = len(keys) // 2
    weight_map = {}
    for i, chunk in enumerate((keys[:half], keys[half:])):
        name = f"model-0000{i + 1}-of-00002.safetensors"
        save_file({k: native[k].contiguous() for k in chunk}, str(tdir / name))
        weight_map.update({k: name for k in chunk})
    (tdir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (tdir / "config.json").write_text(json.dumps(TINY_TRANSFORMER_NATIVE))
    heads = cfg["num_pdd_heads"]
    save_file(
        {
            "video_out.weight": torch.stack(
                [state[f"pdd_proj_out.{i}.weight"] for i in range(heads)]
            ).contiguous(),
            "video_out.bias": torch.stack(
                [state[f"pdd_proj_out.{i}.bias"] for i in range(heads)]
            ).contiguous(),
        },
        str(out / h3i.PDD_HEADS_FILE),
    )
    (out / h3i.PDD_CONFIG_FILE).write_text(
        json.dumps({"num_inference_steps": 6, "sigmas": list(h3i.H3_IMAGE_SIGMAS)})
    )
    (out / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "MiniMaxH3ImagePipeline",
                "_minimax_h3": {"pdd_fused_heads": h3i.PDD_HEADS_FILE, "tasks": ["t2i", "edit"]},
                "transformer": ["diffusers", "MiniMaxH3DiTModel"],
                "vae": ["diffusers", "MiniMaxH3ImageVAE"],
                "text_encoder": [
                    "transformers",
                    "Qwen3VLForConditionalGeneration",
                    {
                        "pretrained_model_name_or_path": "MiniMaxAI/MiniMax-H3",
                        "subfolder": "text_encoder",
                    },
                ],
            }
        )
    )
    vcfg = h3i.vae_config_from_native(TINY_VAE_NATIVE)
    vae = h3i.vae_class()(**vcfg).eval()
    randomize(vae, seed + 1, std = 0.02)
    vdir = out / "vae"
    vdir.mkdir(exist_ok = True)
    save_file(
        {k: v.contiguous() for k, v in to_native_vae(vae.state_dict(), 2, 32).items()},
        str(vdir / "diffusion_pytorch_model.safetensors"),
    )
    (vdir / "config.json").write_text(json.dumps(TINY_VAE_NATIVE))
    tokenizer, processor = make_tokenizer_and_processor(out)
    return {"transformer": dense, "vae": vae, "tokenizer": tokenizer, "processor": processor}
