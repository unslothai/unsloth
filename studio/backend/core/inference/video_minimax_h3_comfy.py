# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 denoisers in ComfyUI layout: fused qkv split by thirds; mlp.fc1 [gate; value] ->
SwiGLU [value; gate], rows moved whole so codes and scales stay exact; pruned adaLN curve sized
from adaln_t_table; float32 tensors kept; partition is not in the tensors: read it from the file
name."""

from __future__ import annotations

import functools
import os
import re
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
    # dense (non-pruned) files only: the timestep MLP the curve table replaces
    ("time_embedder.proj_in.", "time_embedder.linear_1."),
    ("time_embedder.proj_out.", "time_embedder.linear_2."),
)
# Kept float32 like the hosted curve-form checkpoints (ComfyUI stores some of them as float16: widening is exact).
_FP32_PREFIXES = (
    H3_COMFY_TABLE_KEY,
    "time_embedder.",
    "video_patch_proj.",
    "audio_patch_proj.",
    "final_layer.video_out.",
    "final_layer.audio_out.",
)
# Only the pruned (curve-form) adaLN projections are float32; a dense file's are bf16 like the block stack.
_FP32_PRUNED_PREFIXES = ("final_layer.adaln_proj.",)
_FP32_BLOCK = re.compile(r"^blocks\.\d+\.adaln_proj\.linear\.")


def is_h3_comfy_name(filename: Optional[str]) -> bool:
    """A ``.safetensors`` name that can only be an H3 denoiser (not the Qwen conditioner or a VAE)."""
    name = os.path.basename(str(filename or "")).lower()
    if not name.endswith(".safetensors") or not re.search(r"minimax[_-]h3", name):
        return False
    if any(token in name for token in ("qwen", "vae", "text_encoder", "controlnet", "lora")):
        return False
    # Packed W6A8 / W4A8 run as int8 ConvRot (diffusion_grouped_int8); nvfp4 / mxfp / W4A4 files have no H3 runtime.
    if any(t in name for t in ("w6a8", "w4a8")):
        return not any(t in name for t in ("nvfp4", "mxfp", "w4a4"))
    return any(t in name for t in ("int8", "fp8", "bf16", "fp16")) and not any(
        t in name for t in ("w4a", "nvfp4", "mxfp")
    )


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
        return [
            (name.replace(".mlp.fc1.weight", ".ff.net.0.proj.weight"), [(half, half), (0, half)])
        ]
    return [(name, None)]


def h3_comfy_keep_dtype(key: str, pruned: bool = True) -> Optional[Any]:
    """float32 for the tensors H3 keeps at full precision, else None (compute dtype)."""
    if key.startswith(_FP32_PREFIXES) or (
        pruned and (key.startswith(_FP32_PRUNED_PREFIXES) or _FP32_BLOCK.match(key))
    ):
        import torch
        return torch.float32
    return None


def h3_comfy_curve_metadata(path: str) -> Optional[dict]:
    """Curve-form apply_h3_adaln_curve metadata from a pruned file's header; None if dense."""
    from .diffusion_comfy_quant import _read_header

    header, _ = _read_header(path)
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


def comfy_layout(path: str) -> dict:
    """``diffusion_comfy_quant.original_layout`` hooks for MiniMax-H3: key map, float32 tensors and, for a pruned
    file, the curve adaLN installed on the freshly built model."""
    from .video_minimax_h3_pdd import h3_pdd_heads_in_file

    metadata = h3_comfy_curve_metadata(path)
    heads = h3_pdd_heads_in_file(path)
    return {
        "key_map": h3_comfy_key_map,
        "keep_dtype": functools.partial(h3_comfy_keep_dtype, pruned = metadata is not None),
        "prepare_model": _h3_prepare_model(metadata, heads),
    }


def _h3_prepare_model(
    metadata: Optional[dict],
    heads: int,
    logger: Any = None,
) -> Any:
    """The ``prepare_model`` hook for one H3 file: curve adaLN for a pruned file, the stacked PDD head bank for a
    ``with_heads`` one, None when the stock model already matches."""
    if metadata is None and heads <= 1:
        return None

    def prepare(model: Any) -> None:
        if metadata is not None:
            from .video_minimax_h3_adaln import apply_h3_adaln_curve
            if logger is not None:
                apply_h3_adaln_curve(model, metadata, logger = logger)
            else:
                apply_h3_adaln_curve(model, metadata)
        if heads > 1:
            from .video_minimax_h3_pdd import install_h3_pdd_heads
            install_h3_pdd_heads(model, heads)
            if logger is not None:
                logger.info("video.h3_pdd: %d-head PDD output bank per stream installed", heads)

    return prepare


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
    # fp8 layers no fp8 runtime takes stay 1-byte codes widened per layer (``h3_comfy_backends``); False dequantizes
    fp8_storage: Optional[bool] = False,
) -> Any:
    """The H3 denoiser of a ComfyUI-quantized file, int8 / fp8 codes kept, curve adaLN installed."""
    from .diffusion_comfy_quant import load_comfy_quant_transformer
    from .video_minimax_h3_pdd import h3_pdd_heads_in_file

    metadata = h3_comfy_curve_metadata(path)
    # Header only, before any tensor is read: a PDD trunk without its head bank is refused here.
    heads = h3_pdd_heads_in_file(path)
    prepare, fp32_buffers = _keep_fp32_init_buffers(
        transformer_cls, _h3_prepare_model(metadata, heads, logger = logger)
    )
    model = load_comfy_quant_transformer(
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
        fp8_storage = fp8_storage,
        family = "minimax-h3",
        target = target,
        logger = logger,
        key_map = h3_comfy_key_map,
        prepare_model = prepare,
        keep_dtype = functools.partial(h3_comfy_keep_dtype, pruned = metadata is not None),
    )
    _restore_fp32_init_buffers(model, fp32_buffers)
    return model


def _keep_fp32_init_buffers(transformer_cls: Any, prepare: Any) -> tuple:
    """``prepare_model`` that also snapshots the float32 ``__init__`` buffers of ``_keep_in_fp32_modules``: the loader
    casts unloaded buffers to the compute dtype, and a bf16 rope ``inv_freq`` drifts the clip off the hosted one's."""
    keep = getattr(transformer_cls, "_keep_in_fp32_modules", None) or []
    if isinstance(keep, str):
        keep = [keep]
    saved: dict = {}

    def hook(model: Any) -> None:
        import torch

        if prepare is not None:
            prepare(model)
        saved.clear()
        for name, module in model.named_modules():
            if not name or not any(k in name.split(".") for k in keep):
                continue
            for buffer_name, buffer in module._buffers.items():
                if (
                    buffer is not None
                    and buffer.dtype == torch.float32
                    and buffer.device.type != "meta"
                ):
                    saved[(name, buffer_name)] = buffer.detach().clone()

    return hook, saved


def _restore_fp32_init_buffers(model: Any, saved: dict) -> None:
    for (name, buffer_name), value in saved.items():
        module = model.get_submodule(name)
        current = module._buffers.get(buffer_name)
        if current is not None and current.dtype != value.dtype and current.shape == value.shape:
            module._buffers[buffer_name] = value.to(current.device)
