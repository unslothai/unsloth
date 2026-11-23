"""Convert an original MiniMax-H3 checkpoint into the diffusers layout.

The transformer checkpoint is streamed shard by shard, so peak memory stays close to a single shard (~4.9 GiB) and
never approaches the 62 GiB of the full 33B DiT.

Every source key maps onto a diffusers module by renaming alone, except for three things:

* the fused QKV projection is split into `to_q` / `to_k` / `to_v`. Raw checkpoint shards store QKV per-head
  interleaved; the shard streamer first reorders them into the reference's in-memory `[q_all; k_all; v_all]` layout
  (`reorder_interleaved_qkv`, mirroring the reference's load-time transform), then `convert_transformer_key` splits
  contiguous thirds (`split_fused_qkv`),
* the gated FFN's fused `mlp.fc1` becomes `ff.net.0.proj` and its two halves are swapped, because diffusers'
  [`SwiGLU`] reads `[value; gate]` where the reference stores `[gate; value]` (the same transform the video VAE needs,
  see `convert_video_vae_key`); `mlp.fc2` becomes `ff.net.2`,
* `rope.inv_freq` is dropped: it is a pure function of `rope_theta` and `rope_freq_dim`, the port recomputes it in a
  non-persistent buffer, and the recomputed value is bitwise equal to the shipped one for both released variants.

There are no transposes anywhere.

The FL2VA and Ref2VA variants differ only in the transformer weights, so the variant is selected by pointing
`--checkpoint_path` at the corresponding folder. Both land in one repository, which carries a single
`modular_model_index.json`: MiniMax-H3 is integrated as Modular Diffusers blocks only, so no `model_index.json` is
written.

`--version image` converts a MiniMax-H3-Image release instead: the same transformer without the audio projections,
with the per-step output heads of `pdd_fused_heads.safetensors` folded in as `pdd_proj_out.{i}` in place of the base
head, the image VAE, a scheduler config and a `modular_model_index.json`. The release keeps its original `transformer/`
and `vae/` at the repository root, so the converted ones are written to `transformer_diffusers/` and `vae_diffusers/`
and can be published next to them. The release conditions on the MiniMax-H3 text encoder without carrying it, so
`--text_encoder_path` names a local copy that is copied in: the converted repository is self-contained.

Usage:

```bash
# Validate the key mapping without any weights present.
python scripts/convert_minimax_h3_to_diffusers.py \
    --checkpoint_path /path/to/MiniMax-H3/FL2VA --output_path /tmp/h3-diffusers --dry_run

# Convert, and point the component loading specs at the Hub id the result is published under.
python scripts/convert_minimax_h3_to_diffusers.py \
    --checkpoint_path /path/to/MiniMax-H3/FL2VA --output_path /tmp/h3-diffusers \
    --modular_repo_id MiniMaxAI/MiniMax-H3

# Convert a MiniMax-H3-Image release.
python scripts/convert_minimax_h3_to_diffusers.py --version image \
    --checkpoint_path /path/to/MiniMax-H3-Image --output_path /tmp/h3-image-diffusers \
    --text_encoder_path /path/to/MiniMax-H3/text_encoder --modular_repo_id MiniMaxAI/MiniMax-H3-Image
```
"""

import argparse
import glob
import json
import math
import os
import shutil
import struct
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import AutoencoderKLMiniMaxH3Audio
from diffusers.utils.constants import SAFE_WEIGHTS_INDEX_NAME


# `MiniMaxH3Transformer3DModel` argument names. The original config uses the sglang-native names listed in the
# comments; everything else in the original config (`adaln_out_features`, `final_adaln_out_features`) is derived.
MINIMAX_H3_TRANSFORMER_CONFIG = {
    "num_attention_heads": 56,
    "attention_head_dim": 128,
    "hidden_size": 5376,
    "num_layers": 50,
    "num_refiner_layers": 2,  # token_refiner_num_layers
    "ffn_dim": 14336,  # ffn_hidden_size
    "in_channels": 24,  # latents_dim
    "audio_in_channels": 32,  # audio_latents_dim
    "patch_size": [1, 2, 2],
    "text_dim": 5120,
    "freq_dim": 256,  # timestep_input_dim
    "time_embed_hidden_dim": 5376,  # time_embed_hidden_size
    "time_embed_dim": 2688,
    "rope_freq_dim": 16,  # rope_inv_freq_len
    "rope_theta": 10000.0,
    "norm_eps": 1e-05,
    "qk_norm_eps": 1e-05,
    "final_norm_eps": 1e-05,
}

# A tiny configuration with the checkpoint-tied dimensions left intact, for building fixtures.
MINIMAX_H3_TEST_TRANSFORMER_CONFIG = {
    **MINIMAX_H3_TRANSFORMER_CONFIG,
    "num_attention_heads": 2,
    "attention_head_dim": 32,
    "hidden_size": 64,
    "num_layers": 2,
    "num_refiner_layers": 2,
    "ffn_dim": 128,
    "text_dim": 48,
    "freq_dim": 16,
    "time_embed_hidden_dim": 64,
    "time_embed_dim": 32,
    "rope_freq_dim": 4,
}

# `MiniMaxH3ImageTransformer2DModel` argument names. MiniMax-H3-Image is the same DiT without the audio input
# projection and output head (`audio_enabled: false`), plus one output head per step of its 6-step Parallel Decoding
# Distillation (PDD) schedule (`pdd_fused_heads.safetensors`).
MINIMAX_H3_IMAGE_TRANSFORMER_CONFIG = {
    **{name: value for name, value in MINIMAX_H3_TRANSFORMER_CONFIG.items() if name != "audio_in_channels"},
    "num_pdd_heads": 6,  # pdd_config.json num_inference_steps
}

# MiniMax-H3 ships a mixed-precision checkpoint. These *original* keys are float32; everything else is bfloat16 —
# including the AdaLN projections.
MINIMAX_H3_FP32_SOURCE_PREFIXES = (
    "video_patch_proj.",
    "audio_patch_proj.",
    "time_embedder.",
    "final_layer.video_out.",
    "final_layer.audio_out.",
)

# `rope.inv_freq` is `1 / rope_theta ** (arange(0, 2 * rope_freq_dim, 2) / (2 * rope_freq_dim))`, which
# `MiniMaxH3RotaryPosEmbed` recomputes into a non-persistent buffer. The recomputed tensor is bitwise equal to the
# shipped one in both released variants, so the key is not carried into the diffusers checkpoint.
MINIMAX_H3_TRANSFORMER_DROPPED_KEYS = ("rope.inv_freq",)


def reorder_interleaved_qkv(weight: torch.Tensor, num_attention_heads: int, attention_head_dim: int) -> torch.Tensor:
    """Reorder a *raw-checkpoint* per-head-interleaved fused QKV weight into `[q_all; k_all; v_all]`.

    The original checkpoint shards store rows as `[head0: q(head_dim) k(head_dim) v(head_dim), head1: q, k, v, ...]`.
    The reference implementation applies exactly this reorder at load time (`_reorder_grouped_qkv_to_qkv` with one head
    per query group), so `[q_all; k_all; v_all]` is the reference's in-memory / state-dict layout. There is no
    transpose.
    """
    expected_rows = num_attention_heads * 3 * attention_head_dim
    if weight.shape[0] != expected_rows:
        raise ValueError(
            f"fused qkv weight has {weight.shape[0]} rows, expected "
            f"{expected_rows} = {num_attention_heads} heads * 3 * {attention_head_dim}."
        )
    grouped = weight.reshape(num_attention_heads, 3 * attention_head_dim, *weight.shape[1:])
    query, key, value = grouped.split(attention_head_dim, dim=1)
    return torch.cat(
        [
            tensor.reshape(num_attention_heads * attention_head_dim, *weight.shape[1:])
            for tensor in (query, key, value)
        ],
        dim=0,
    )


def split_fused_qkv(
    weight: torch.Tensor, num_attention_heads: int, attention_head_dim: int
) -> tuple[torch.Tensor, ...]:
    """Split a fused `[q_all; k_all; v_all]` QKV weight into separate `to_q` / `to_k` / `to_v` weights.

    The input is the *reference model* layout — what `MiniMaxH3DiTModel.state_dict()` holds after the reference's
    load-time reorder — i.e. the three logical projection matrices stacked contiguously, NOT the raw checkpoint's
    per-head interleave (see `reorder_interleaved_qkv`, which the shard streamer applies first).
    """
    inner_dim = num_attention_heads * attention_head_dim
    if weight.shape[0] != 3 * inner_dim:
        raise ValueError(
            f"fused qkv weight has {weight.shape[0]} rows, expected "
            f"{3 * inner_dim} = 3 * {num_attention_heads} heads * {attention_head_dim}."
        )
    query, key, value = weight.split(inner_dim, dim=0)
    return tuple(tensor.contiguous() for tensor in (query, key, value))


def get_transformer_key_plan(config: dict[str, Any]) -> dict[str, list[tuple[str, list[int]]]]:
    """Map every original transformer key to the diffusers key(s) it produces, with the resulting shapes.

    The plan is derived from the config alone, so it can be printed and checked without any weights present.
    """
    hidden_size = config["hidden_size"]
    heads = config["num_attention_heads"]
    head_dim = config["attention_head_dim"]
    inner_dim = heads * head_dim
    ffn_dim = config["ffn_dim"]
    time_embed_dim = config["time_embed_dim"]
    video_patch_dim = (
        config["in_channels"] * config["patch_size"][0] * config["patch_size"][1] * config["patch_size"][2]
    )

    plan: dict[str, list[tuple[str, list[int]]]] = {
        "video_patch_proj.weight": [("proj_in.weight", [hidden_size, video_patch_dim])],
        "video_patch_proj.bias": [("proj_in.bias", [hidden_size])],
        "condition_proj.weight": [("context_embedder.weight", [hidden_size, config["text_dim"]])],
        "condition_proj.bias": [("context_embedder.bias", [hidden_size])],
        # `Timesteps` + `TimestepEmbedding` reproduce the reference sinusoid and MLP exactly, so the timestep MLP is
        # renamed onto `TimestepEmbedding`'s `linear_1` / `linear_2`.
        "time_embedder.proj_in.weight": [
            ("time_embedder.linear_1.weight", [config["time_embed_hidden_dim"], config["freq_dim"]])
        ],
        "time_embedder.proj_in.bias": [("time_embedder.linear_1.bias", [config["time_embed_hidden_dim"]])],
        "time_embedder.proj_out.weight": [
            ("time_embedder.linear_2.weight", [time_embed_dim, config["time_embed_hidden_dim"]])
        ],
        "time_embedder.proj_out.bias": [("time_embedder.linear_2.bias", [time_embed_dim])],
        "token_refiner.final_norm.weight": [("token_refiner.final_norm.weight", [hidden_size])],
        "final_layer.norm.weight": [("norm_out.norm.weight", [hidden_size])],
        "final_layer.adaln_proj.linear.weight": [("norm_out.linear.weight", [2 * hidden_size, time_embed_dim])],
        "final_layer.adaln_proj.linear.bias": [("norm_out.linear.bias", [2 * hidden_size])],
        "final_layer.video_out.weight": [("proj_out.weight", [video_patch_dim, hidden_size])],
        "final_layer.video_out.bias": [("proj_out.bias", [video_patch_dim])],
    }
    if config.get("audio_in_channels") is not None:
        plan.update(
            {
                "audio_patch_proj.weight": [("audio_proj_in.weight", [hidden_size, config["audio_in_channels"]])],
                "audio_patch_proj.bias": [("audio_proj_in.bias", [hidden_size])],
                "final_layer.audio_out.weight": [
                    ("audio_proj_out.weight", [config["audio_in_channels"], hidden_size])
                ],
                "final_layer.audio_out.bias": [("audio_proj_out.bias", [config["audio_in_channels"]])],
            }
        )
    for key in MINIMAX_H3_TRANSFORMER_DROPPED_KEYS:
        plan[key] = []
    if config.get("num_pdd_heads") is not None:
        # A PDD release ships its per-step heads outside `transformer/` (see `convert_pdd_heads`) and runs every step
        # with one of them, so the checkpoint's own `video_out` head is never used.
        plan["final_layer.video_out.weight"] = []
        plan["final_layer.video_out.bias"] = []

    block_specs = [
        ("blocks", "transformer_blocks", config["num_layers"], True),
        ("token_refiner.blocks", "token_refiner.refiner_blocks", config["num_refiner_layers"], False),
    ]
    for source_prefix, target_prefix, num_layers, has_adaln in block_specs:
        for i in range(num_layers):
            source = f"{source_prefix}.{i}"
            target = f"{target_prefix}.{i}"
            plan[f"{source}.norm1.weight"] = [(f"{target}.norm1.weight", [hidden_size])]
            plan[f"{source}.norm2.weight"] = [(f"{target}.norm2.weight", [hidden_size])]
            plan[f"{source}.attn.qkv_proj.weight"] = [
                (f"{target}.attn.to_q.weight", [inner_dim, hidden_size]),
                (f"{target}.attn.to_k.weight", [inner_dim, hidden_size]),
                (f"{target}.attn.to_v.weight", [inner_dim, hidden_size]),
            ]
            plan[f"{source}.attn.q_norm.weight"] = [(f"{target}.attn.norm_q.weight", [head_dim])]
            plan[f"{source}.attn.k_norm.weight"] = [(f"{target}.attn.norm_k.weight", [head_dim])]
            plan[f"{source}.attn.out_proj.weight"] = [(f"{target}.attn.to_out.0.weight", [hidden_size, inner_dim])]
            # `fc1` stays fused, as diffusers' `SwiGLU` also fuses its two projections, but the halves are swapped
            # from `[gate; value]` to `[value; gate]` (see `convert_transformer_key`).
            plan[f"{source}.mlp.fc1.weight"] = [(f"{target}.ff.net.0.proj.weight", [2 * ffn_dim, hidden_size])]
            plan[f"{source}.mlp.fc2.weight"] = [(f"{target}.ff.net.2.weight", [hidden_size, ffn_dim])]
            if has_adaln:
                plan[f"{source}.adaln_proj.linear.weight"] = [
                    (f"{target}.adaln_proj.linear.weight", [6 * 3 * hidden_size, time_embed_dim])
                ]
                plan[f"{source}.adaln_proj.linear.bias"] = [
                    (f"{target}.adaln_proj.linear.bias", [6 * 3 * hidden_size])
                ]

    return plan


def convert_transformer_key(
    source_key: str, tensor: torch.Tensor, config: dict[str, Any]
) -> list[tuple[str, torch.Tensor]]:
    """Convert one original key/tensor pair into the diffusers key/tensor pair(s) it maps to."""
    if source_key in MINIMAX_H3_TRANSFORMER_DROPPED_KEYS:
        return []
    if config.get("num_pdd_heads") is not None and source_key.startswith("final_layer.video_out."):
        # Replaced by the per-step PDD heads (see `get_transformer_key_plan`).
        return []

    target_key = source_key
    if target_key.startswith("token_refiner.blocks."):
        target_key = target_key.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.", 1)
    elif target_key.startswith("blocks."):
        target_key = target_key.replace("blocks.", "transformer_blocks.", 1)
    target_key = target_key.replace("time_embedder.proj_in.", "time_embedder.linear_1.")
    target_key = target_key.replace("time_embedder.proj_out.", "time_embedder.linear_2.")
    target_key = target_key.replace("video_patch_proj.", "proj_in.")
    target_key = target_key.replace("audio_patch_proj.", "audio_proj_in.")
    target_key = target_key.replace("condition_proj.", "context_embedder.")
    target_key = target_key.replace("final_layer.norm.", "norm_out.norm.")
    target_key = target_key.replace("final_layer.adaln_proj.linear.", "norm_out.linear.")
    target_key = target_key.replace("final_layer.video_out.", "proj_out.")
    target_key = target_key.replace("final_layer.audio_out.", "audio_proj_out.")
    target_key = target_key.replace(".attn.q_norm.", ".attn.norm_q.")
    target_key = target_key.replace(".attn.k_norm.", ".attn.norm_k.")
    target_key = target_key.replace(".attn.out_proj.", ".attn.to_out.0.")

    if target_key.endswith(".attn.qkv_proj.weight"):
        # `convert_transformer_key` consumes tensors in the reference model's state-dict layout, where the fused QKV
        # rows are already `[q_all; k_all; v_all]`. Raw checkpoint shards are per-head interleaved instead; the shard
        # streamer (`convert_transformer`) normalizes them with `reorder_interleaved_qkv` before calling this.
        query, key, value = split_fused_qkv(tensor, config["num_attention_heads"], config["attention_head_dim"])
        prefix = target_key.removesuffix("qkv_proj.weight")
        return [(f"{prefix}to_q.weight", query), (f"{prefix}to_k.weight", key), (f"{prefix}to_v.weight", value)]

    if target_key.endswith(".mlp.fc1.weight"):
        # The reference computes `fc2(silu(gate) * value)` from a fused `[gate; value]`; diffusers' `SwiGLU` computes
        # `value * silu(gate)` from a fused `[value; gate]`, so the two halves swap places. Identical transform to the
        # video VAE's `ff.w1` (see `convert_video_vae_key`).
        gate, value = tensor.chunk(2, dim=0)
        target_key = target_key.replace(".mlp.fc1.weight", ".ff.net.0.proj.weight")
        return [(target_key, torch.cat([value, gate], dim=0).contiguous())]

    target_key = target_key.replace(".mlp.fc2.", ".ff.net.2.")
    return [(target_key, tensor)]


#
# ---------------------------------------------------------------------------------------------------------------
# Video VAE
# ---------------------------------------------------------------------------------------------------------------
#

# `AutoencoderKLMiniMaxH3` argument names. Field-for-field equal to `video_vae/source/config.json`, with the original
# names in the comments. The keys that only ever take one value in the release (`use_3d_conv`, `use_vit_decoder`,
# `causal_encoder`, `causal_decoder`, `use_t_isolated_gn`, `space_up` / `time_up`, `zq_ch_*`, `num_res_blocks_decoder`,
# `shift_factor` / `scaling_factor`) are baked into the port instead of being config knobs.
MINIMAX_H3_VIDEO_VAE_CONFIG = {
    "in_channels": 3,
    "out_channels": 3,  # out_ch
    "latent_channels": 24,  # z_channels == embed_dim
    "block_out_channels": [128, 256, 256, 512, 512, 1024],  # ch * ch_mult
    "layers_per_block": 2,  # num_res_blocks
    "spatial_downsample_factors": [2, 2, 2, 2, 1, 1],  # space_down
    "temporal_downsample_factors": [1, 2, 2, 1, 1, 1],  # time_down
    "norm_num_groups": 32,
    "norm_eps": 1e-06,
    "spatial_padding_mode": "reflect",  # padding_mode
    "decoder_num_layers": 36,  # vit_decoder_kwargs.num_layers
    "decoder_num_attention_heads": 32,  # vit_decoder_kwargs.heads
    "decoder_attention_head_dim": 64,  # vit_decoder_kwargs.dim_head
    "decoder_num_register_tokens": 4,  # ViT3DDecoder default
    "decoder_ffn_mult": 4,  # FeedForward default
    "decoder_rope_theta": 100.0,  # vit_decoder_kwargs.rope_theta
    "decoder_rope_dim_ratio": 0.75,  # vit_decoder_kwargs.rope_dim_ratio
    "decoder_norm_eps": 1e-05,  # ViT3DDecoder eps
    "clip_length": 17,  # video_vae/config.json vae_clip_length
    "token_drop": 3,  # video_vae/config.json vae_token_drop
}

# A tiny configuration with the checkpoint-tied dimensions (`latent_channels`, the temporal geometry and the rotary
# ratio) left intact, for building fixtures and for the CPU parity check.
MINIMAX_H3_TEST_VIDEO_VAE_CONFIG = {
    **MINIMAX_H3_VIDEO_VAE_CONFIG,
    "block_out_channels": [32, 64],
    "layers_per_block": 1,
    "spatial_downsample_factors": [2, 2],
    "temporal_downsample_factors": [2, 2],
    "decoder_num_layers": 4,
    "decoder_num_attention_heads": 4,
    "decoder_attention_head_dim": 32,
}

# `decoder.mask_token` is an all-zero buffer belonging to the masked-autoencoding training objective; the released
# decoder never reads it, so the port does not carry the module and the conversion drops the key.
MINIMAX_H3_VIDEO_VAE_DROPPED_KEYS = ("decoder.mask_token",)


def convert_video_vae_key(
    source_key: str, tensor: torch.Tensor, config: dict[str, Any]
) -> list[tuple[str, torch.Tensor]]:
    """Convert one original video-VAE key/tensor pair into the diffusers key/tensor pair(s) it maps to.

    `quant_conv` / `post_quant_conv`, the encoder's `conv_in` / `norm_out` / `conv_out` and the ViT decoder's
    `register_tokens` / `norm_out` / `proj_out` / `norm{1,2}` / `scale{1,2}` are pure pass-throughs. What moves:

    * the encoder's CNN levels are renamed from the original CompVis spelling onto the diffusers autoencoder idiom:
      `down.{i}.block.{j}` -> `down_blocks.{i}.resnets.{j}`, `nin_shortcut` -> `conv_shortcut`, and
      `down.{i}.downsample` -> `down_blocks.{i}.downsamplers.0`,
    * the ViT decoder's `x_embedder` becomes `proj_in`, the counterpart of the `proj_out` it already ships,
    * the fused per-head-interleaved `attn.to_qkv` is split into `attn.to_q` / `to_k` / `to_v`,
    * `attn.to_out` becomes `attn.to_out.0` (diffusers wraps the output projection in an `nn.ModuleList`),
    * the gated FFN's `w1` / `w2` become `ff.net.0.proj` / `ff.net.2`, and the two halves of `w1` are swapped because
      diffusers' `SwiGLU` reads `[up; gate]` where the reference stores `[gate; up]`.
    """
    if source_key in MINIMAX_H3_VIDEO_VAE_DROPPED_KEYS:
        return []

    if ".attn.to_qkv." in source_key:
        # Same per-head interleave as the DiT: `[head0: q k v, head1: q k v, ...]`.
        reordered = reorder_interleaved_qkv(
            tensor, config["decoder_num_attention_heads"], config["decoder_attention_head_dim"]
        )
        query, key, value = split_fused_qkv(
            reordered, config["decoder_num_attention_heads"], config["decoder_attention_head_dim"]
        )
        prefix, suffix = source_key.split(".attn.to_qkv.")
        return [
            (f"{prefix}.attn.to_q.{suffix}", query),
            (f"{prefix}.attn.to_k.{suffix}", key),
            (f"{prefix}.attn.to_v.{suffix}", value),
        ]

    target_key = _rename_video_vae_key(source_key)

    if ".ff.w1." in source_key:
        gate, up = tensor.chunk(2, dim=0)
        return [(target_key, torch.cat([up, gate], dim=0).contiguous())]

    return [(target_key, tensor)]


def _rename_video_vae_key(source_key: str) -> str:
    """Rename one original video-VAE key onto its diffusers module path (no tensor transform)."""
    target_key = source_key
    if target_key.startswith("encoder.down."):
        level, rest = target_key.removeprefix("encoder.down.").split(".", 1)
        rest = rest.replace("block.", "resnets.", 1).replace("nin_shortcut.", "conv_shortcut.", 1)
        rest = rest.replace("downsample.", "downsamplers.0.", 1)
        target_key = f"encoder.down_blocks.{level}.{rest}"
    target_key = target_key.replace("decoder.x_embedder.", "decoder.proj_in.")
    target_key = target_key.replace(".attn.to_out.", ".attn.to_out.0.")
    target_key = target_key.replace(".ff.w1.", ".ff.net.0.proj.")
    target_key = target_key.replace(".ff.w2.", ".ff.net.2.")
    return target_key


def get_video_vae_key_plan(config: dict[str, Any]) -> dict[str, list[str]]:
    """Map every original video-VAE key to the diffusers key(s) it produces, derived from the config alone."""
    block_out_channels = config["block_out_channels"]
    block_in_channels = [block_out_channels[0]] + block_out_channels[:-1]
    plan: dict[str, list[str]] = {}

    def renamed(*keys: str) -> None:
        """Register keys whose diffusers name is `_rename_video_vae_key(key)` and whose tensor is unchanged."""
        for key in keys:
            plan[key] = [_rename_video_vae_key(key)]

    renamed("quant_conv.weight", "quant_conv.bias", "post_quant_conv.weight", "post_quant_conv.bias")
    renamed("encoder.conv_in.weight", "encoder.conv_in.bias")
    for level, (in_channels, out_channels) in enumerate(zip(block_in_channels, block_out_channels)):
        for i in range(config["layers_per_block"]):
            prefix = f"encoder.down.{level}.block.{i}"
            for name in ("norm1", "conv1", "norm2", "conv2"):
                renamed(f"{prefix}.{name}.weight", f"{prefix}.{name}.bias")
            if (in_channels if i == 0 else out_channels) != out_channels:
                renamed(f"{prefix}.nin_shortcut.weight", f"{prefix}.nin_shortcut.bias")
        if config["spatial_downsample_factors"][level] * config["temporal_downsample_factors"][level] > 1:
            renamed(f"encoder.down.{level}.downsample.conv.weight", f"encoder.down.{level}.downsample.conv.bias")
    renamed("encoder.norm_out.weight", "encoder.norm_out.bias", "encoder.conv_out.weight", "encoder.conv_out.bias")

    renamed("decoder.x_embedder.weight", "decoder.x_embedder.bias", "decoder.register_tokens")
    renamed("decoder.norm_out.weight", "decoder.norm_out.bias", "decoder.proj_out.weight", "decoder.proj_out.bias")
    for i in range(config["decoder_num_layers"]):
        prefix = f"decoder.transformer_blocks.{i}"
        renamed(f"{prefix}.norm1.weight", f"{prefix}.norm2.weight", f"{prefix}.scale1", f"{prefix}.scale2")
        for suffix in ("weight", "bias"):
            plan[f"{prefix}.attn.to_qkv.{suffix}"] = [
                f"{prefix}.attn.to_q.{suffix}",
                f"{prefix}.attn.to_k.{suffix}",
                f"{prefix}.attn.to_v.{suffix}",
            ]
            plan[f"{prefix}.attn.to_out.{suffix}"] = [f"{prefix}.attn.to_out.0.{suffix}"]
            plan[f"{prefix}.ff.w1.{suffix}"] = [f"{prefix}.ff.net.0.proj.{suffix}"]
            plan[f"{prefix}.ff.w2.{suffix}"] = [f"{prefix}.ff.net.2.{suffix}"]
    for key in MINIMAX_H3_VIDEO_VAE_DROPPED_KEYS:
        plan[key] = []
    return plan


def convert_video_vae(
    checkpoint_path: str, output_path: str, config: dict[str, Any], diffusers_version: str, max_shard_size: int
) -> None:
    """Convert the video VAE and emit its config.

    The original weights live one level deeper than the rest of the checkpoint
    (`video_vae/source/model.safetensors`, resolved by a hook in the reference); the diffusers layout flattens that to
    `vae/`. `latents_mean` / `latents_std` and the tiling geometry come from `video_vae/config.json` — MiniMax-H3
    normalizes latents per channel instead of with a `scaling_factor`.
    """
    source_dir = os.path.join(checkpoint_path, "video_vae")
    with open(os.path.join(source_dir, "config.json")) as f:
        wrapper_config = json.load(f)
    for key in ("latents_mean", "latents_std"):
        if key not in wrapper_config:
            raise KeyError(f"{source_dir}/config.json does not carry `{key}`.")

    weights_path = os.path.join(source_dir, wrapper_config["source_path"], wrapper_config["source_safetensors_path"])
    plan = get_video_vae_key_plan(config)

    os.makedirs(output_path, exist_ok=True)
    weight_map: dict[str, str] = {}
    total_size = 0
    written: list[str] = []
    buffer: dict[str, torch.Tensor] = {}
    buffer_size = 0
    seen_source_keys: set[str] = set()

    def flush() -> None:
        nonlocal buffer, buffer_size
        if not buffer:
            return
        path = os.path.join(output_path, f".tmp-shard-{len(written):05d}.safetensors")
        save_file(buffer, path, metadata={"format": "pt"})
        for key in buffer:
            weight_map[key] = path
        written.append(path)
        buffer = {}
        buffer_size = 0

    # `safe_open` memory-maps the file, so only the tensor being read is materialized.
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        for source_key in f.keys():
            if source_key not in plan:
                raise KeyError(f"Unexpected key in {weights_path}: {source_key}")
            seen_source_keys.add(source_key)
            for target_key, tensor in convert_video_vae_key(source_key, f.get_tensor(source_key), config):
                if tensor.dtype != torch.float32:
                    raise ValueError(f"{source_key}: expected torch.float32, got {tensor.dtype}.")
                buffer[target_key] = tensor
                buffer_size += tensor.numel() * tensor.element_size()
                total_size += tensor.numel() * tensor.element_size()
            if buffer_size >= max_shard_size:
                flush()
    flush()

    missing = sorted(set(plan) - seen_source_keys)
    if missing:
        raise KeyError(f"{len(missing)} planned key(s) missing from {weights_path}, e.g. {missing[:5]}.")

    renames = {
        path: os.path.join(output_path, f"diffusion_pytorch_model-{i + 1:05d}-of-{len(written):05d}.safetensors")
        for i, path in enumerate(written)
    }
    for old, new in renames.items():
        os.rename(old, new)
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": {key: os.path.basename(renames[path]) for key, path in weight_map.items()},
    }
    with open(os.path.join(output_path, SAFE_WEIGHTS_INDEX_NAME), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(
            {
                "_class_name": "AutoencoderKLMiniMaxH3",
                "_diffusers_version": diffusers_version,
                **config,
                "latents_mean": wrapper_config["latents_mean"],
                "latents_std": wrapper_config["latents_std"],
            },
            f,
            indent=2,
        )

    print(
        f"video_vae: {len(seen_source_keys)} original keys -> {len(weight_map)} diffusers keys in "
        f"{len(written)} shard(s), {total_size / 1024**3:.2f} GiB "
        f"(latents_mean/latents_std: {len(wrapper_config['latents_mean'])}/"
        f"{len(wrapper_config['latents_std'])} channels; tiling {wrapper_config['vae_tile_size']}px / "
        f"{wrapper_config['vae_tile_overlap_min']}px min overlap)."
    )


# `AutoencoderKLMiniMaxH3Image` argument names. The encoder is the video VAE's; the decoder comes from
# `vit_decoder_kwargs` in the release's `vae/config.json` (names in the comments), which `convert_image_vae` checks.
MINIMAX_H3_IMAGE_VAE_CONFIG = {
    **{key: MINIMAX_H3_VIDEO_VAE_CONFIG[key] for key in MINIMAX_H3_VIDEO_VAE_CONFIG if not key.startswith("decoder_")},
    "decoder_num_layers": 24,  # vit_decoder_kwargs.num_layers
    "decoder_num_attention_heads": 16,  # vit_decoder_kwargs.heads
    "decoder_attention_head_dim": 64,  # vit_decoder_kwargs.dim_head
    "decoder_num_register_tokens": 4,  # ViT3DDecoder default
    "decoder_ffn_mult": 4,  # FeedForward default
    "decoder_rope_theta": 100.0,  # vit_decoder_kwargs.rope_theta
    "decoder_rope_dim_ratio": 0.5,  # vit_decoder_kwargs.rope_dim_ratio
    "decoder_norm_eps": 1e-05,  # ViT3DDecoder eps
}
# The video temporal chunking does not apply to a single frame.
del MINIMAX_H3_IMAGE_VAE_CONFIG["clip_length"], MINIMAX_H3_IMAGE_VAE_CONFIG["token_drop"]


def convert_image_vae(checkpoint_path: str, output_path: str, config: dict[str, Any], diffusers_version: str) -> None:
    """Convert the MiniMax-H3-Image VAE (`vae/`) and emit its config.

    The keys are the video VAE's, so they convert with `convert_video_vae_key`; only `post_quant_conv` is a 2D
    convolution here, which `AutoencoderKLMiniMaxH3Image` matches.
    """
    source_dir = os.path.join(checkpoint_path, "vae")
    with open(os.path.join(source_dir, "config.json")) as f:
        source_config = json.load(f)
    decoder_kwargs = source_config["vit_decoder_kwargs"]
    expected = {
        "vit_decoder_2d": (source_config.get("vit_decoder_2d"), True),
        "num_layers": (decoder_kwargs["num_layers"], config["decoder_num_layers"]),
        "heads": (decoder_kwargs["heads"], config["decoder_num_attention_heads"]),
        "dim_head": (decoder_kwargs["dim_head"], config["decoder_attention_head_dim"]),
        "rope_theta": (decoder_kwargs["rope_theta"], config["decoder_rope_theta"]),
        "rope_dim_ratio": (decoder_kwargs["rope_dim_ratio"], config["decoder_rope_dim_ratio"]),
        "latent_channels": (source_config["latent_channels"], config["latent_channels"]),
        # The tile geometry `AutoencoderKLMiniMaxH3Image` enables by default.
        "vae_encoder_tiling": (source_config["vae_encoder_tiling"], 1),
        "vae_decoder_tiling": (source_config["vae_decoder_tiling"], 1),
        "vae_tile_size": (source_config["vae_tile_size"], 256),
        "vae_tile_overlap_min": (source_config["vae_tile_overlap_min"], 64),
    }
    mismatches = {key: values for key, values in expected.items() if values[0] != values[1]}
    if mismatches:
        raise ValueError(
            f"{source_dir}/config.json disagrees with the image VAE config (source, expected): {mismatches}"
        )

    weights_path = os.path.join(source_dir, "diffusion_pytorch_model.safetensors")
    plan = get_video_vae_key_plan(config)
    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        source_keys = set(f.keys())
        unexpected = sorted(source_keys - set(plan))
        missing = sorted(set(plan) - source_keys)
        if unexpected or missing:
            raise KeyError(f"{weights_path}: unexpected keys {unexpected[:5]}, missing keys {missing[:5]}.")
        for source_key in sorted(source_keys):
            for target_key, tensor in convert_video_vae_key(source_key, f.get_tensor(source_key), config):
                if tensor.dtype != torch.float32:
                    raise ValueError(f"{source_key}: expected torch.float32, got {tensor.dtype}.")
                state_dict[target_key] = tensor

    os.makedirs(output_path, exist_ok=True)
    save_file(state_dict, os.path.join(output_path, "diffusion_pytorch_model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(
            {
                "_class_name": "AutoencoderKLMiniMaxH3Image",
                "_diffusers_version": diffusers_version,
                **config,
                "latents_mean": source_config["latents_mean"],
                "latents_std": source_config["latents_std"],
            },
            f,
            indent=2,
        )
    total_size = sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())
    print(
        f"vae: {len(source_keys)} original keys -> {len(state_dict)} diffusers keys, {total_size / 1024**3:.2f} GiB."
    )


# Not present in `audio_vae/metadata.json`: the reference implementation hardcodes these in its DAC audio VAE and its
# attention projection, keyed off the sample rate.
MINIMAX_H3_AUDIO_VAE_FIXED_CONFIG = {
    "num_attention_heads": 8,
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
}


def get_audio_vae_config(checkpoint_path: str) -> dict[str, Any]:
    """Build the `AutoencoderKLMiniMaxH3Audio` config from the original audio-VAE metadata.

    `audio_vae/metadata.json` carries the constructor kwargs the checkpoint was built with, and
    `audio_vae/config.json` carries the per-channel `latents_mean` / `latents_std` MiniMax-H3 normalizes with. The two
    are cross-checked here because they duplicate the latent width and sample rate.
    """
    source_dir = os.path.join(checkpoint_path, "audio_vae")
    with open(os.path.join(source_dir, "metadata.json")) as f:
        kwargs = json.load(f)["metadata"]["kwargs"]
    with open(os.path.join(source_dir, "config.json")) as f:
        wrapper_config = json.load(f)

    if kwargs["decoder_type"] != "bigvgan":
        raise ValueError(f"Only the BigVGAN decoder is supported, got {kwargs['decoder_type']!r}.")
    if not kwargs["attn_proj"]:
        raise ValueError("The audio VAE is expected to carry the causal-attention latent projection.")
    latent_channels = kwargs["vae_latent_channels"]
    if wrapper_config["latent_channels"] != latent_channels:
        raise ValueError(
            f"latent width disagreement: metadata.json says {latent_channels}, "
            f"config.json says {wrapper_config['latent_channels']}."
        )
    if wrapper_config["sample_rate"] != kwargs["sample_rate"]:
        raise ValueError(
            f"sample rate disagreement: metadata.json says {kwargs['sample_rate']}, "
            f"config.json says {wrapper_config['sample_rate']}."
        )
    for key in ("latents_mean", "latents_std"):
        if len(wrapper_config[key]) != latent_channels:
            raise KeyError(f"{source_dir}/config.json `{key}` does not have {latent_channels} entries.")

    return {
        "encoder_dim": kwargs["encoder_dim"],
        "encoder_rates": kwargs["encoder_rates"],
        "latent_dim": kwargs["latent_dim"],
        "latent_channels": latent_channels,
        "decoder_dim": kwargs["decoder_dim"],
        "decoder_rates": kwargs["decoder_rates"],
        # The reference's two hardcoded BigVGAN tables (16 kHz and 32 kHz) both pair rate `u` with kernel
        # `2u` for even `u` and `2u - 1` for odd `u`, i.e. [5, 5, 2, ...] -> [9, 9, 4, ...].
        "decoder_kernel_sizes": [2 * rate - (rate % 2) for rate in kwargs["decoder_rates"]],
        **MINIMAX_H3_AUDIO_VAE_FIXED_CONFIG,
        # Renamed from the original `sample_rate` to the diffusers audio convention.
        "sampling_rate": kwargs["sample_rate"],
        "latents_mean": wrapper_config["latents_mean"],
        "latents_std": wrapper_config["latents_std"],
    }


def convert_audio_vae(checkpoint_path: str, output_path: str, diffusers_version: str) -> None:
    """Convert the audio VAE and emit its config.

    The mapping is an identity: `AutoencoderKLMiniMaxH3Audio` reproduces the original module tree name for name,
    including `torch.nn.utils.weight_norm`'s `weight_g` / `weight_v` spelling and the Kaiser-window `filter` buffers of
    the anti-aliased activations. The keys are therefore only *validated* against a freshly built model, not renamed.
    """
    config = get_audio_vae_config(checkpoint_path)
    expected_keys = set(AutoencoderKLMiniMaxH3Audio(**config).state_dict())

    weights_path = os.path.join(checkpoint_path, "audio_vae", "model.safetensors")
    state_dict: dict[str, torch.Tensor] = {}
    total_size = 0
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key not in expected_keys:
                raise KeyError(f"Unexpected key in {weights_path}: {key}")
            tensor = f.get_tensor(key)
            if tensor.dtype != torch.float32:
                raise ValueError(f"{key}: expected torch.float32, got {tensor.dtype}.")
            state_dict[key] = tensor
            total_size += tensor.numel() * tensor.element_size()

    missing = sorted(expected_keys - set(state_dict))
    if missing:
        raise KeyError(f"{len(missing)} key(s) missing from {weights_path}, e.g. {missing[:5]}.")

    os.makedirs(output_path, exist_ok=True)
    save_file(
        state_dict,
        os.path.join(output_path, "diffusion_pytorch_model.safetensors"),
        metadata={"format": "pt"},
    )
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(
            {"_class_name": "AutoencoderKLMiniMaxH3Audio", "_diffusers_version": diffusers_version, **config},
            f,
            indent=2,
        )

    print(
        f"audio_vae: {len(state_dict)} keys carried over unchanged, {total_size / 1024**3:.2f} GiB "
        f"({config['sampling_rate'] // math.prod(config['encoder_rates'])} latents/s, "
        f"latents_mean/latents_std: {len(config['latents_mean'])}/{len(config['latents_std'])} channels)."
    )


def write_scheduler_configs(checkpoint_path: str, output_path: str, diffusers_version: str) -> None:
    """Emit the two `MiniMaxH3Scheduler` configs, one per modality.

    The source `model_index.json` leaves `scheduler` null and instead carries the schedule constants in its
    `_minimax_h3.sigma_shift_scales` block. The sigma shift is the only per-modality difference, so it becomes two
    scheduler folders holding the same class at different `shift` values.
    """
    with open(os.path.join(checkpoint_path, "model_index.json")) as f:
        shift_scales = json.load(f)["_minimax_h3"]["sigma_shift_scales"]

    for folder, modality in (("scheduler", "video"), ("audio_scheduler", "audio")):
        folder_path = os.path.join(output_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        with open(os.path.join(folder_path, "scheduler_config.json"), "w") as f:
            json.dump(
                {
                    "_class_name": "MiniMaxH3Scheduler",
                    "_diffusers_version": diffusers_version,
                    "shift": float(shift_scales[modality]),
                },
                f,
                indent=2,
            )
    print(f"scheduler: shift={shift_scales['video']} (video), audio_scheduler: shift={shift_scales['audio']} (audio).")


def read_safetensors_header(path: str) -> dict[str, Any]:
    """Read the metadata header of a safetensors file without touching the tensor payload."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    header.pop("__metadata__", None)
    return header


def dry_run(checkpoint_path: str, config: dict[str, Any]) -> None:
    plan = get_transformer_key_plan(config)

    transformer_dir = os.path.join(checkpoint_path, "transformer")
    shards = sorted(glob.glob(os.path.join(transformer_dir, "*.safetensors")))
    header: dict[str, Any] = {}
    for shard in shards:
        header.update(read_safetensors_header(shard))
    if shards:
        print(f"Read headers of {len(shards)} shard(s) in {transformer_dir}: {len(header)} keys present.\n")
    else:
        print(f"No shards found under {transformer_dir}; validating the plan against the config only.\n")

    print(f"{'original key':<48} {'->':^4} {'diffusers key':<52} {'shape':<24} dtype")
    print("-" * 150)
    num_target_keys = 0
    shape_mismatches: list[str] = []
    for source_key, targets in plan.items():
        present = source_key in header
        if not targets:
            print(f"{source_key:<48} {'-x':^4} {'(dropped, recomputed by the port)':<52}")
            continue
        for index, (target_key, shape) in enumerate(targets):
            num_target_keys += 1
            expected_dtype = "F32" if source_key.startswith(MINIMAX_H3_FP32_SOURCE_PREFIXES) else "BF16"
            if present:
                actual_dtype = header[source_key]["dtype"]
                actual_shape = header[source_key]["shape"]
                if index == 0 and len(targets) == 1 and actual_shape != shape:
                    shape_mismatches.append(f"{source_key}: header {actual_shape} != planned {shape}")
                if actual_dtype != expected_dtype:
                    shape_mismatches.append(f"{source_key}: header dtype {actual_dtype} != expected {expected_dtype}")
                marker = "->"
            else:
                marker = "->?"
            left = source_key if index == 0 else ""
            print(f"{left:<48} {marker:^4} {target_key:<52} {str(shape):<24} {expected_dtype}")

    missing = [key for key in plan if key not in header]
    unexpected = [key for key in header if key not in plan]

    print("\n" + "=" * 150)
    print(f"planned original keys : {len(plan)}")
    print(f"planned diffusers keys: {num_target_keys}")
    print(
        f"dropped original keys : {len(MINIMAX_H3_TRANSFORMER_DROPPED_KEYS)} {list(MINIMAX_H3_TRANSFORMER_DROPPED_KEYS)}"
    )
    print(f"fp32 diffusers keys   : {sum(1 for key in plan if key.startswith(MINIMAX_H3_FP32_SOURCE_PREFIXES))}")
    if shards:
        print(f"keys present in shards: {len(header)}")
        print(f"planned but absent    : {len(missing)}" + (" (shards still downloading?)" if missing else ""))
        print(f"present but unplanned : {len(unexpected)}")
        if unexpected:
            print(f"  {unexpected}")
        print(f"header disagreements  : {len(shape_mismatches)}")
        for line in shape_mismatches:
            print(f"  {line}")
    total_bytes = sum(
        (4 if source_key.startswith(MINIMAX_H3_FP32_SOURCE_PREFIXES) else 2) * torch.Size(shape).numel()
        for source_key, targets in plan.items()
        for _, shape in targets
    )
    print(f"total output bytes    : {total_bytes} ({total_bytes / 1024**3:.2f} GiB)")


def convert_pdd_heads(checkpoint_path: str, config: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Split the stacked PDD heads of a MiniMax-H3-Image release into one `pdd_proj_out.{i}` head per step.

    The release keeps them outside `transformer/`, in the file `_minimax_h3.pdd_fused_heads` of `model_index.json`
    names: `video_out.weight` of shape `(num_steps, out, in)` and `video_out.bias` of shape `(num_steps, out)`, float32,
    with `pdd_config.json` listing the step count and the sigma grid they were distilled on.
    """
    with open(os.path.join(checkpoint_path, "model_index.json")) as f:
        heads_path = os.path.join(checkpoint_path, json.load(f)["_minimax_h3"]["pdd_fused_heads"])
    with open(os.path.join(checkpoint_path, "pdd_config.json")) as f:
        num_steps = json.load(f)["num_inference_steps"]
    if num_steps != config["num_pdd_heads"]:
        raise ValueError(f"pdd_config.json has {num_steps} steps, the config {config['num_pdd_heads']} heads.")

    with safe_open(heads_path, framework="pt", device="cpu") as f:
        heads = {key: f.get_tensor(key) for key in f.keys()}
    if set(heads) != {"video_out.weight", "video_out.bias"}:
        raise KeyError(f"{heads_path} must hold exactly `video_out.weight` and `video_out.bias`, got {sorted(heads)}.")

    hidden_size = config["hidden_size"]
    video_patch_dim = config["in_channels"] * math.prod(config["patch_size"])
    expected_shapes = {
        "video_out.weight": (num_steps, video_patch_dim, hidden_size),
        "video_out.bias": (num_steps, video_patch_dim),
    }
    converted = {}
    for key, tensor in heads.items():
        if tuple(tensor.shape) != expected_shapes[key] or tensor.dtype != torch.float32:
            raise ValueError(
                f"{key}: expected float32 {expected_shapes[key]}, got {tensor.dtype} {tuple(tensor.shape)}."
            )
        suffix = key.removeprefix("video_out.")
        for step in range(num_steps):
            converted[f"pdd_proj_out.{step}.{suffix}"] = tensor[step].contiguous()
    return converted


def convert_transformer(checkpoint_path: str, output_path: str, config: dict[str, Any], max_shard_size: int) -> None:
    plan = get_transformer_key_plan(config)
    transformer_dir = os.path.join(checkpoint_path, "transformer")
    shards = sorted(glob.glob(os.path.join(transformer_dir, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"No `*.safetensors` shards found under {transformer_dir}.")

    os.makedirs(output_path, exist_ok=True)
    weight_map: dict[str, str] = {}
    total_size = 0
    written: list[str] = []
    buffer: dict[str, torch.Tensor] = {}
    buffer_size = 0
    seen_source_keys: set[str] = set()

    def flush() -> None:
        nonlocal buffer, buffer_size
        if not buffer:
            return
        path = os.path.join(output_path, f".tmp-shard-{len(written):05d}.safetensors")
        save_file(buffer, path, metadata={"format": "pt"})
        for key in buffer:
            weight_map[key] = path
        written.append(path)
        buffer = {}
        buffer_size = 0

    for shard in shards:
        # `safe_open` memory-maps the file, so only the tensor being read is materialized.
        with safe_open(shard, framework="pt", device="cpu") as f:
            for source_key in f.keys():
                if source_key not in plan:
                    raise KeyError(f"Unexpected key in {os.path.basename(shard)}: {source_key}")
                seen_source_keys.add(source_key)
                source_tensor = f.get_tensor(source_key)
                if source_key.endswith(".attn.qkv_proj.weight"):
                    # Raw shards store fused QKV per-head interleaved; normalize to the reference's
                    # `[q_all; k_all; v_all]` layout (the same transform the reference applies at load time) so
                    # `convert_transformer_key` sees its state-dict-layout contract. The composition is bit-identical
                    # to de-interleaving the raw tensor directly.
                    source_tensor = reorder_interleaved_qkv(
                        source_tensor, config["num_attention_heads"], config["attention_head_dim"]
                    )
                for target_key, tensor in convert_transformer_key(source_key, source_tensor, config):
                    expected_dtype = (
                        torch.float32 if source_key.startswith(MINIMAX_H3_FP32_SOURCE_PREFIXES) else torch.bfloat16
                    )
                    if tensor.dtype != expected_dtype:
                        raise ValueError(f"{source_key}: expected {expected_dtype}, got {tensor.dtype}.")
                    buffer[target_key] = tensor
                    buffer_size += tensor.numel() * tensor.element_size()
                    total_size += tensor.numel() * tensor.element_size()
                if buffer_size >= max_shard_size:
                    flush()
    if config.get("num_pdd_heads") is not None:
        for target_key, tensor in convert_pdd_heads(checkpoint_path, config).items():
            buffer[target_key] = tensor
            buffer_size += tensor.numel() * tensor.element_size()
            total_size += tensor.numel() * tensor.element_size()
    flush()

    missing = sorted(set(plan) - seen_source_keys)
    if missing:
        raise KeyError(f"{len(missing)} planned key(s) missing from the checkpoint, e.g. {missing[:5]}.")

    # The shard count is only known once every source shard has been streamed, so the files are written under
    # provisional names and renamed here.
    renames = {
        path: os.path.join(output_path, f"diffusion_pytorch_model-{i + 1:05d}-of-{len(written):05d}.safetensors")
        for i, path in enumerate(written)
    }
    for old, new in renames.items():
        os.rename(old, new)
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": {key: os.path.basename(renames[path]) for key, path in weight_map.items()},
    }
    with open(os.path.join(output_path, SAFE_WEIGHTS_INDEX_NAME), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    print(
        f"transformer: {len(seen_source_keys)} original keys -> {len(weight_map)} diffusers keys "
        f"in {len(written)} shard(s), {total_size / 1024**3:.2f} GiB."
    )


def write_transformer_config(
    output_path: str, config: dict[str, Any], diffusers_version: str, class_name: str = "MiniMaxH3Transformer3DModel"
) -> None:
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(
            {"_class_name": class_name, "_diffusers_version": diffusers_version, **config},
            f,
            indent=2,
        )


# The components a MiniMax-H3 repository holds, and the class each one loads as. `video_processor` is absent: the
# blocks create it from config rather than loading it.
MINIMAX_H3_COMPONENTS = {
    # The source names a checkpoint-local wrapper class (`MiniMaxH3Qwen3VLHFEncoder`); the conditioner is the
    # released Qwen3-VL, read at its 50th decoder layer with its language-model head unused.
    "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
    "tokenizer": ["transformers", "Qwen2TokenizerFast"],
    "processor": ["transformers", "Qwen3VLProcessor"],
    # Renamed to the diffusers audio/video VAE convention (see `LTX2Pipeline`).
    "vae": ["diffusers", "AutoencoderKLMiniMaxH3"],
    "audio_vae": ["diffusers", "AutoencoderKLMiniMaxH3Audio"],
    "transformer": ["diffusers", "MiniMaxH3Transformer3DModel"],
    # One repository holds both checkpoint partitions: `transformer/` serves `MiniMaxH3Blocks` (`t2va` / `fl2va`) and
    # `transformer_ref/` serves `MiniMaxH3Ref2VABlocks`, while every other component is shared and converted once.
    "transformer_ref": ["diffusers", "MiniMaxH3Transformer3DModel"],
    # The source leaves `scheduler` null. MiniMax-H3 samples with Euler at eta=0 over shifted flow-matching sigmas, at
    # a different shift per modality, so it needs two scheduler entries (see `write_scheduler_configs`).
    "scheduler": ["diffusers", "MiniMaxH3Scheduler"],
    "audio_scheduler": ["diffusers", "MiniMaxH3Scheduler"],
}


def write_model_index(output_path: str, repo_id: str, diffusers_version: str) -> None:
    """Emit `modular_model_index.json`, the only index a MiniMax-H3 repository carries.

    MiniMax-H3 is integrated as Modular Diffusers blocks only, so there is no `model_index.json`: a modular repository
    declares one entry per component with its full loading spec rather than just its class, and a blockset then fetches
    exactly the subfolders it declares. That is what lets one repository hold both transformer partitions, and the
    original checkpoint folders next to the converted ones, without either half pulling the rest down.

    `_class_name` and `_blocks_class_name` name the `t2va` / `fl2va` half, which is what
    `ModularPipeline.from_pretrained` resolves to. The `ref2va` half reads the very same file through
    `MiniMaxH3Ref2VABlocks().init_pipeline(repo_id)`.

    The component map is the static one above, so this needs no source checkpoint: an index can be regenerated for a
    repository that is already published.
    """
    modular_index = {
        "_class_name": "MiniMaxH3ModularPipeline",
        "_diffusers_version": diffusers_version,
        "_blocks_class_name": "MiniMaxH3Blocks",
    }
    for name, (library, class_name) in MINIMAX_H3_COMPONENTS.items():
        modular_index[name] = [
            library,
            class_name,
            {
                "type_hint": [library, class_name],
                "pretrained_model_name_or_path": repo_id,
                "subfolder": name,
                "variant": None,
                "revision": None,
            },
        ]
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "modular_model_index.json"), "w") as f:
        json.dump(modular_index, f, indent=2)
    print(f"modular_model_index.json: {len(MINIMAX_H3_COMPONENTS)} components load from {repo_id}.")


# The components a MiniMax-H3-Image repository names. The release does not carry its text encoder (its
# `model_index.json` points at `text_encoder/` of the MiniMax-H3 repository); the converted repository holds a copy.
MINIMAX_H3_IMAGE_COMPONENTS = {
    "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
    "tokenizer": ["transformers", "Qwen2TokenizerFast"],
    "processor": ["transformers", "Qwen3VLProcessor"],
    "vae": ["diffusers", "AutoencoderKLMiniMaxH3Image"],
    "transformer": ["diffusers", "MiniMaxH3ImageTransformer2DModel"],
    "scheduler": ["diffusers", "MiniMaxH3Scheduler"],
}

# The release already has `transformer/` and `vae/` folders in the original format, so the converted components live
# in folders of their own. Every other component uses its name as its folder.
MINIMAX_H3_IMAGE_SUBFOLDERS = {"transformer": "transformer_diffusers", "vae": "vae_diffusers"}


def write_image_scheduler_config(output_path: str, diffusers_version: str) -> None:
    """Emit the `MiniMaxH3Scheduler` config of a MiniMax-H3-Image release.

    The release samples on the explicit grid of its `pdd_config.json`, which the blocks pass to `set_timesteps`
    verbatim (`shift_provided_sigmas=False`), and evaluates the Euler update in the velocity form its heads were distilled with.
    """
    folder_path = os.path.join(output_path, "scheduler")
    os.makedirs(folder_path, exist_ok=True)
    with open(os.path.join(folder_path, "scheduler_config.json"), "w") as f:
        json.dump(
            {
                "_class_name": "MiniMaxH3Scheduler",
                "_diffusers_version": diffusers_version,
                "shift_provided_sigmas": False,
                "euler_update_form": "velocity",
            },
            f,
            indent=2,
        )
    print("scheduler: explicit sigmas taken verbatim, Euler update in the velocity form.")


def write_image_model_index(checkpoint_path: str, output_path: str, repo_id: str, diffusers_version: str) -> None:
    """Emit the `modular_model_index.json` of a MiniMax-H3-Image repository.

    Besides the components it carries `sample_sigmas`, the grid of the release's `pdd_config.json` that the per-step
    output heads were distilled on. The transformer and the VAE load from the folders of
    `MINIMAX_H3_IMAGE_SUBFOLDERS`.
    """
    with open(os.path.join(checkpoint_path, "pdd_config.json")) as f:
        pdd_config = json.load(f)
    sigmas = pdd_config["sigmas"]
    if len(sigmas) != pdd_config["num_inference_steps"] + 1 or sigmas[0] != 1.0 or sigmas[-1] != 0.0:
        raise ValueError(f"pdd_config.json must list `num_inference_steps + 1` sigmas from 1.0 to 0.0, got {sigmas}.")

    modular_index = {
        "_class_name": "MiniMaxH3ImageModularPipeline",
        "_diffusers_version": diffusers_version,
        "_blocks_class_name": "MiniMaxH3ImageBlocks",
        "sample_sigmas": sigmas,
    }
    for name, (library, class_name) in MINIMAX_H3_IMAGE_COMPONENTS.items():
        modular_index[name] = [
            library,
            class_name,
            {
                "type_hint": [library, class_name],
                "pretrained_model_name_or_path": repo_id,
                "subfolder": MINIMAX_H3_IMAGE_SUBFOLDERS.get(name, name),
                "variant": None,
                "revision": None,
            },
        ]
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "modular_model_index.json"), "w") as f:
        json.dump(modular_index, f, indent=2)
    print(
        f"modular_model_index.json: {len(MINIMAX_H3_IMAGE_COMPONENTS)} components load from {repo_id}; "
        f"{len(sigmas) - 1} sampling steps."
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Local path to an original MiniMax-H3 variant folder (the one holding `transformer/`, `audio_vae/`, ...).",
    )
    parser.add_argument("--output_path", type=str, required=True, help="Where the diffusers checkpoint is written.")
    parser.add_argument(
        "--modular_repo_id",
        type=str,
        default=None,
        help=(
            "Repository the component entries of `modular_model_index.json` point at. Defaults to `--output_path`, so "
            "pass the Hub id the checkpoint is published under. Every entry carries its own loading spec, so a "
            "blockset fetches exactly the subfolders it declares out of that repository."
        ),
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help=(
            "`--version image` only: local path to the `text_encoder/` folder of a MiniMax-H3 checkpoint, which the "
            "image release conditions on without carrying it. It is copied into the output as is."
        ),
    )
    parser.add_argument(
        "--version",
        type=str,
        default="h3",
        choices=["h3", "image", "test"],
        help=(
            "`image` converts a MiniMax-H3-Image release (transformer and VAE); `test` emits the tiny config used for "
            "fixtures."
        ),
    )
    parser.add_argument(
        "--max_shard_size",
        type=int,
        default=5 * 1024**3,
        help="Maximum size of an output safetensors shard, in bytes.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the full planned key mapping (and cross-check any shard headers already present) without writing.",
    )
    return parser.parse_args()


def main(args):
    from diffusers import __version__ as diffusers_version

    config = {
        "h3": MINIMAX_H3_TRANSFORMER_CONFIG,
        "image": MINIMAX_H3_IMAGE_TRANSFORMER_CONFIG,
        "test": MINIMAX_H3_TEST_TRANSFORMER_CONFIG,
    }[args.version]

    if args.dry_run:
        dry_run(args.checkpoint_path, config)
        return

    if args.version == "image":
        if args.text_encoder_path is None:
            raise ValueError(
                "`--version image` needs `--text_encoder_path`, the `text_encoder/` folder of MiniMax-H3."
            )
        transformer_path = os.path.join(args.output_path, MINIMAX_H3_IMAGE_SUBFOLDERS["transformer"])
        convert_transformer(args.checkpoint_path, transformer_path, config, args.max_shard_size)
        write_transformer_config(transformer_path, config, diffusers_version, "MiniMaxH3ImageTransformer2DModel")
        convert_image_vae(
            args.checkpoint_path,
            os.path.join(args.output_path, MINIMAX_H3_IMAGE_SUBFOLDERS["vae"]),
            MINIMAX_H3_IMAGE_VAE_CONFIG,
            diffusers_version,
        )
        # The tokenizer and the processor are already in the `transformers` layout.
        for name in ("tokenizer", "processor"):
            shutil.copytree(
                os.path.join(args.checkpoint_path, name), os.path.join(args.output_path, name), dirs_exist_ok=True
            )
        # A plain copy keeps the files identical to MiniMax-H3's, which the Hub deduplicates.
        shutil.copytree(args.text_encoder_path, os.path.join(args.output_path, "text_encoder"), dirs_exist_ok=True)
        write_image_scheduler_config(args.output_path, diffusers_version)
        write_image_model_index(
            args.checkpoint_path, args.output_path, args.modular_repo_id or args.output_path, diffusers_version
        )
        return

    transformer_path = os.path.join(args.output_path, "transformer")
    convert_transformer(args.checkpoint_path, transformer_path, config, args.max_shard_size)
    write_transformer_config(transformer_path, config, diffusers_version)
    video_vae_config = MINIMAX_H3_TEST_VIDEO_VAE_CONFIG if args.version == "test" else MINIMAX_H3_VIDEO_VAE_CONFIG
    convert_video_vae(
        args.checkpoint_path,
        os.path.join(args.output_path, "vae"),
        video_vae_config,
        diffusers_version,
        args.max_shard_size,
    )
    convert_audio_vae(args.checkpoint_path, os.path.join(args.output_path, "audio_vae"), diffusers_version)
    write_scheduler_configs(args.checkpoint_path, args.output_path, diffusers_version)
    write_model_index(args.output_path, args.modular_repo_id or args.output_path, diffusers_version)


if __name__ == "__main__":
    main(get_args())
