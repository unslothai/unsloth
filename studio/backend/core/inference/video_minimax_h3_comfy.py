# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 denoisers stored in ComfyUI's layout, for ``diffusion_comfy_quant.load_comfy_quant_transformer``.

ComfyUI ships the H3 DiT under the reference model's names (``blocks.N.attn.qkv_proj``, ``mlp.fc1``, ...) and
diffusers has no single-file converter for it, so this module supplies the mapping the loader needs:

- a rename onto ``MiniMaxH3Transformer3DModel``'s names,
- two row-only reshapes: the fused ``qkv_proj`` splits into ``to_q`` / ``to_k`` / ``to_v`` by contiguous thirds, and
  ``mlp.fc1`` (``[gate; value]``) becomes ``ff.net.0.proj`` (``[value; gate]``, what diffusers' SwiGLU reads). Rows
  move whole, so int8 / fp8 codes and their per-row scales stay exact,
- the pruned (curve-form) adaLN: ``adaln_t_table`` is the curve table Studio's hosted checkpoints call
  ``time_embedder.table``, and its shape gives the curve rank and grid ``apply_h3_adaln_curve`` needs,
- the float32 tensors the pruned model keeps (adaLN, patch projections, output heads), as the hosted files do.

Which workflow partition a file holds (fl2va keyframe / text, ref2va reference) is not visible in the tensors, so it
is read from the file name. Torch-free apart from what the loader passes in.
"""

from __future__ import annotations

import json
import os
import re
import struct
from typing import Any, Optional

H3_COMFY_TABLE_KEY = "adaln_t_table"
_DROPPED = ("rope.inv_freq",)
_RENAMES = (
    ("video_patch_proj.", "proj_in."),
    ("audio_patch_proj.", "audio_proj_in."),
    ("condition_proj.", "context_embedder."),
    ("final_layer.norm.", "norm_out.norm."),
    ("final_layer.adaln_proj.linear.", "norm_out.linear."),
    ("final_layer.video_out.", "proj_out."),
    ("final_layer.audio_out.", "audio_proj_out."),
    (".attn.q_norm.", ".attn.norm_q."),
    (".attn.k_norm.", ".attn.norm_k."),
    (".attn.out_proj.", ".attn.to_out.0."),
    (".mlp.fc2.", ".ff.net.2."),
)
# Kept float32 like the hosted curve-form checkpoints (ComfyUI stores some of them as float16: widening is exact).
_FP32_PREFIXES = (
    H3_COMFY_TABLE_KEY,
    "video_patch_proj.",
    "audio_patch_proj.",
    "final_layer.adaln_proj.",
    "final_layer.video_out.",
    "final_layer.audio_out.",
)
_FP32_BLOCK = re.compile(r"^blocks\.\d+\.adaln_proj\.linear\.")


def is_h3_comfy_name(filename: Optional[str]) -> bool:
    """A ``.safetensors`` name that can only be an H3 denoiser (not the Qwen conditioner or a VAE)."""
    name = os.path.basename(str(filename or "")).lower()
    if not name.endswith(".safetensors") or not re.search(r"minimax[_-]h3", name):
        return False
    if any(token in name for token in ("qwen", "vae", "text_encoder", "controlnet", "lora")):
        return False
    # The quantized formats Studio runs (int8_convrot, fp8_scaled, ours -INT8 / -FP8); a bf16 or packed w6a8 file is not one.
    return ("int8" in name or "fp8" in name) and not any(t in name for t in ("w6a8", "w4a", "nvfp4", "mxfp"))


def h3_comfy_task(filename: Optional[str]) -> str:
    """The workflow partition a ComfyUI H3 denoiser file serves, from its name: ``ref2va`` or ``fl2va``."""
    return "ref2va" if "ref2va" in os.path.basename(str(filename or "")).lower() else "fl2va"


def h3_comfy_key_map(key: str, shape: Any) -> list:
    """``[(diffusers key, rows)]`` for one ComfyUI H3 key (``rows``: None or ``[(first row, n rows), ...]``)."""
    if key in _DROPPED:
        return []
    if key == H3_COMFY_TABLE_KEY:
        return [("time_embedder.table", None)]
    name = key
    if name.startswith("token_refiner.blocks."):
        name = "token_refiner.refiner_blocks." + name[len("token_refiner.blocks.") :]
    elif name.startswith("blocks."):
        name = "transformer_blocks." + name[len("blocks.") :]
    for old, new in _RENAMES:
        name = name.replace(old, new)
    rows = int(shape[0]) if shape else 0
    if name.endswith(".attn.qkv_proj.weight"):
        if rows % 3:
            raise ValueError(f"{key}: fused qkv with {rows} rows is not three equal projections")
        third = rows // 3
        stem = name[: -len("qkv_proj.weight")]
        return [
            (f"{stem}{part}.weight", [(i * third, third)])
            for i, part in enumerate(("to_q", "to_k", "to_v"))
        ]
    if name.endswith(".mlp.fc1.weight"):
        if rows % 2:
            raise ValueError(f"{key}: gated fc1 with an odd row count {rows}")
        half = rows // 2
        return [(name.replace(".mlp.fc1.weight", ".ff.net.0.proj.weight"), [(half, half), (0, half)])]
    return [(name, None)]


def h3_comfy_keep_dtype(key: str) -> Optional[Any]:
    """float32 for the tensors the pruned H3 keeps at full precision, else None (compute dtype)."""
    if key.startswith(_FP32_PREFIXES) or _FP32_BLOCK.match(key):
        import torch

        return torch.float32
    return None


def h3_comfy_curve_metadata(path: str) -> Optional[dict]:
    """``apply_h3_adaln_curve`` metadata for a pruned (curve-form) ComfyUI H3 file, from its header; None for a
    dense one (no ``adaln_t_table``)."""
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    table = header.get(H3_COMFY_TABLE_KEY)
    if not isinstance(table, dict) or len(table.get("shape") or ()) != 2:
        return None
    grid, dim = (int(v) for v in table["shape"])
    from .video_minimax_h3_adaln import (
        ADALN_CURVE_FORM,
        ADALN_FORM_KEY,
        ADALN_OUT_DTYPE_KEY,
        CURVE_DIM_KEY,
        CURVE_GRID_KEY,
    )

    return {
        ADALN_FORM_KEY: ADALN_CURVE_FORM,
        CURVE_DIM_KEY: dim,
        CURVE_GRID_KEY: grid,
        # The pruned modulation feeds the bf16 block stack, as in the hosted checkpoints built from the same file.
        ADALN_OUT_DTYPE_KEY: "bfloat16",
    }


def load_h3_comfy_transformer(
    transformer_cls: Any,
    path: str,
    scan: Any,
    *,
    config: str,
    subfolder: str,
    dtype: Any,
    hf_token: Optional[str],
    cache_dir: Optional[str],
    local_files_only: bool,
    int8_backend: Optional[str],
    fp8_backend: Optional[str],
    target: Any = None,
    logger: Any = None,
) -> Any:
    """The H3 denoiser of a ComfyUI-quantized file, int8 / fp8 codes kept, curve adaLN installed."""
    from .diffusion_comfy_quant import load_comfy_quant_transformer
    from .video_minimax_h3_adaln import apply_h3_adaln_curve

    metadata = h3_comfy_curve_metadata(path)
    return load_comfy_quant_transformer(
        transformer_cls,
        path,
        scan,
        {
            "torch_dtype": dtype,
            "config": config,
            "subfolder": subfolder,
            "token": hf_token,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
        },
        int8_backend = int8_backend,
        fp8_backend = fp8_backend,
        family = "minimax-h3",
        target = target,
        logger = logger,
        key_map = h3_comfy_key_map,
        prepare_model = (
            (lambda model: apply_h3_adaln_curve(model, metadata, logger = logger))
            if metadata is not None
            else None
        ),
        keep_dtype = h3_comfy_keep_dtype,
    )
