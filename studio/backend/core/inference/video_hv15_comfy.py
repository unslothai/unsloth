# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HunyuanVideo-1.5 original (ComfyUI) layout -> diffusers names, for ``load_comfy_quant_transformer`` (diffusers has
no single-file converter for it). Row-only reshapes keep int8 / fp8 codes and per-row scales exact: fused qkv
projections split by thirds, ``final_layer.adaLN_modulation.1`` swaps ``[shift; scale]`` to ``[scale; shift]``.
Torch-free."""

from __future__ import annotations

import re
from typing import Any

_FIXED = {
    "time_in.mlp.0": "time_embed.timestep_embedder.linear_1",
    "time_in.mlp.2": "time_embed.timestep_embedder.linear_2",
    "time_r_in.mlp.0": "time_embed.timestep_embedder_r.linear_1",
    "time_r_in.mlp.2": "time_embed.timestep_embedder_r.linear_2",
    "txt_in.t_embedder.mlp.0": "context_embedder.time_text_embed.timestep_embedder.linear_1",
    "txt_in.t_embedder.mlp.2": "context_embedder.time_text_embed.timestep_embedder.linear_2",
    "txt_in.c_embedder.linear_1": "context_embedder.time_text_embed.text_embedder.linear_1",
    "txt_in.c_embedder.linear_2": "context_embedder.time_text_embed.text_embedder.linear_2",
    "txt_in.input_embedder": "context_embedder.proj_in",
    "byt5_in.layernorm": "context_embedder_2.norm",
    "byt5_in.fc1": "context_embedder_2.linear_1",
    "byt5_in.fc2": "context_embedder_2.linear_2",
    "byt5_in.fc3": "context_embedder_2.linear_3",
    "vision_in.proj.0": "image_embedder.norm_in",
    "vision_in.proj.1": "image_embedder.linear_1",
    "vision_in.proj.3": "image_embedder.linear_2",
    "vision_in.proj.4": "image_embedder.norm_out",
    "img_in.proj": "x_embedder.proj",
    "cond_type_embedding": "cond_type_embed",
    "final_layer.linear": "proj_out",
    "final_layer.adaLN_modulation.1": "norm_out.linear",
}
_REFINER = {
    "norm1": "norm1",
    "norm2": "norm2",
    "self_attn_proj": "attn.to_out.0",
    "mlp.fc1": "ff.net.0.proj",
    "mlp.fc2": "ff.net.2",
    "adaLN_modulation.1": "norm_out.linear",
}
_DOUBLE = {
    "img_mod.linear": "norm1.linear",
    "txt_mod.linear": "norm1_context.linear",
    "img_attn_q": "attn.to_q",
    "img_attn_k": "attn.to_k",
    "img_attn_v": "attn.to_v",
    "img_attn_q_norm": "attn.norm_q",
    "img_attn_k_norm": "attn.norm_k",
    "img_attn_proj": "attn.to_out.0",
    "txt_attn_q": "attn.add_q_proj",
    "txt_attn_k": "attn.add_k_proj",
    "txt_attn_v": "attn.add_v_proj",
    "txt_attn_q_norm": "attn.norm_added_q",
    "txt_attn_k_norm": "attn.norm_added_k",
    "txt_attn_proj": "attn.to_add_out",
    "img_mlp.fc1": "ff.net.0.proj",
    "img_mlp.fc2": "ff.net.2",
    "txt_mlp.fc1": "ff_context.net.0.proj",
    "txt_mlp.fc2": "ff_context.net.2",
}
# fused projection -> its three diffusers parts, in row order
_FUSED_DOUBLE = {
    "img_attn_qkv": ("attn.to_q", "attn.to_k", "attn.to_v"),
    "txt_attn_qkv": ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"),
}
_DOUBLE_RE = re.compile(r"^double_blocks\.(\d+)\.(.+)\.(weight|bias)$")
_REFINER_RE = re.compile(r"^txt_in\.individual_token_refiner\.blocks\.(\d+)\.(.+)\.(weight|bias)$")
_PLAIN_RE = re.compile(r"^(.+)\.(weight|bias)$")


def _thirds(prefix: str, parts: tuple, leaf: str, rows: int) -> list:
    if rows % 3:
        raise ValueError(f"{prefix}: fused projection with {rows} rows is not three equal parts")
    third = rows // 3
    return [(f"{prefix}{part}.{leaf}", [(i * third, third)]) for i, part in enumerate(parts)]


_DIFFUSERS_PREFIXES = (
    "transformer_blocks.",
    "context_embedder.",
    "context_embedder_2.",
    "x_embedder.",
    "time_embed.",
    "image_embedder.",
    "cond_type_embed.",
    "norm_out.",
    "proj_out.",
)


def hv15_comfy_key_map(key: str, shape: Any) -> list:
    """``[(diffusers key, rows)]`` for one original-layout HunyuanVideo-1.5 key (see the module docstring). A key
    already in the diffusers layout passes through unchanged."""
    if key.startswith(_DIFFUSERS_PREFIXES):
        return [(key, None)]
    rows = int(shape[0]) if shape else 0
    m = _DOUBLE_RE.match(key)
    if m:
        index, name, leaf = m.groups()
        prefix = f"transformer_blocks.{index}."
        if name in _FUSED_DOUBLE:
            return _thirds(prefix, _FUSED_DOUBLE[name], leaf, rows)
        if name in _DOUBLE:
            return [(f"{prefix}{_DOUBLE[name]}.{leaf}", None)]
        raise ValueError(f"{key}: not a HunyuanVideo-1.5 double-block tensor")
    m = _REFINER_RE.match(key)
    if m:
        index, name, leaf = m.groups()
        prefix = f"context_embedder.token_refiner.refiner_blocks.{index}."
        if name == "self_attn_qkv":
            return _thirds(prefix, ("attn.to_q", "attn.to_k", "attn.to_v"), leaf, rows)
        if name in _REFINER:
            return [(f"{prefix}{_REFINER[name]}.{leaf}", None)]
        raise ValueError(f"{key}: not a HunyuanVideo-1.5 token-refiner tensor")
    m = _PLAIN_RE.match(key)
    if m and m.group(1) in _FIXED:
        name, leaf = m.groups()
        if name == "final_layer.adaLN_modulation.1":
            if rows % 2:
                raise ValueError(f"{key}: odd row count {rows} for a shift / scale pair")
            half = rows // 2
            return [(f"{_FIXED[name]}.{leaf}", [(half, half), (0, half)])]
        return [(f"{_FIXED[name]}.{leaf}", None)]
    raise ValueError(f"{key}: not a HunyuanVideo-1.5 transformer tensor")


def comfy_layout(_path: str) -> dict:
    """``diffusion_comfy_quant.original_layout`` hooks for HunyuanVideo-1.5."""
    return {"key_map": hv15_comfy_key_map}
