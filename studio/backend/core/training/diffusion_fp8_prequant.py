# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Loads an unsloth/*-FP8 denoiser for fp8 base; None on unexpected layout, so the caller uses bf16."""

from __future__ import annotations

import json
import os
import struct
from typing import Any, Optional

_FORMAT = "unsloth_prequant_transformer_state_dict_v1"
_QDATA = "._weight_qdata"
_SCALE = "._weight_scale"
# Root-level plain tensors (z-image's pad tokens) are stored under this prefix (prequant_safetensors).
_ROOT = "unsloth_root::"
_DISABLE_ENV = "UNSLOTH_DIFFUSION_TRAIN_FP8_PREQUANT"
_MAX_HEADER = 64 << 20
# Attribute a placeholder nn.Linear carries until _apply_fp8_training converts or dequantizes it.
PREQUANT_ATTR = "_studio_fp8_prequant"


def _disabled() -> bool:
    return os.environ.get(_DISABLE_ENV, "").strip().lower() in ("0", "false", "no", "off")


def _parse_header(raw: bytes) -> dict:
    header = json.loads(raw)
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not an object")
    return header


def _local_header(path: str) -> dict:
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        if n > _MAX_HEADER:
            raise ValueError("safetensors header too large")
        return _parse_header(fh.read(n))


def _remote_header(repo_id: str, filename: str, token: Optional[str]) -> dict:
    """Range reads (~40 KB), so a checkpoint baked from another base is refused before its multi-GB download."""
    from huggingface_hub import HfFileSystem
    with HfFileSystem(token = token).open(f"{repo_id}/{filename}", "rb", block_size = 1 << 16) as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        if n > _MAX_HEADER:
            raise ValueError("safetensors header too large")
        return _parse_header(fh.read(n))


def checkpoint_problem(header: dict, base_model: str) -> Optional[str]:
    """Why this safetensors header cannot seed fp8 training of ``base_model``, or None when it can."""
    meta = header.get("__metadata__") or {}
    if meta.get("unsloth_format") != _FORMAT:
        return f"format {meta.get('unsloth_format')!r} is not {_FORMAT}"
    try:
        info = json.loads(meta.get("unsloth_metadata") or "{}")
    except ValueError:
        return "unreadable unsloth_metadata"
    if info.get("scheme") != "fp8":
        return f"scheme {info.get('scheme')!r} is not fp8"
    if info.get("fp8_granularity") not in (None, "per_row"):
        return f"fp8 granularity {info.get('fp8_granularity')!r} is not per_row"
    ckpt_base = info.get("base_model_id")
    if not ckpt_base:
        return "no base_model_id"
    from core.inference.diffusion_prequant import _same_base_model

    if not _same_base_model(str(ckpt_base), str(base_model)):
        return f"baked from {ckpt_base!r}, not {base_model!r}"
    n_quant = 0
    for name, entry in header.items():
        if not name.endswith(_QDATA):
            continue
        stem = name[: -len(_QDATA)]
        scale = header.get(stem + _SCALE)
        shape = entry.get("shape") or []
        if (
            entry.get("dtype") != "F8_E4M3"
            or len(shape) != 2
            or scale is None
            or scale.get("dtype") != "F32"
            or scale.get("shape") != [shape[0], 1]
        ):
            return f"{stem}: not an e4m3 [out, in] weight with an fp32 [out, 1] scale"
        n_quant += 1
    if not n_quant:
        return "no fp8 weights"
    return None


def _hub_cache() -> Optional[str]:
    try:
        from utils.hf_cache_settings import active_hf_hub_cache
        return active_hf_hub_cache()
    except Exception:  # noqa: BLE001 -- unset: huggingface_hub's own default
        return None


def resolve_fp8_checkpoint(
    family: Optional[str],
    base_model: str,
    hf_token: Optional[str] = None,
    logger: Any = None,
) -> Optional[str]:
    """Validated fp8 denoiser path for base_model (downloaded if absent); None; never raises."""
    if _disabled() or not family or not base_model:
        return None
    try:
        from core.inference.diffusion_families import detect_family
        from core.inference.diffusion_prequant import (
            _download_checkpoint_name,
            _same_base_model,  # noqa: F401 -- imported here so a missing helper falls back early
            cached_checkpoint_path,
            candidate_filenames_of,
            is_comfy_prequant_filename,
            resolve_prequant_source,
        )

        fam = detect_family("", override = str(family).strip().lower())
        if fam is None or not fam.prequant_repos:
            return None
        src = resolve_prequant_source(fam, "fp8", base_repo = base_model)
        if src is None or src.kind != "repo":
            return None
        # Studio's own safetensors only: comfy twins, ConvRot rotations and pickles are other layouts.
        names = [
            n
            for n in candidate_filenames_of(src)
            if n.endswith(".safetensors")
            and not is_comfy_prequant_filename(n)
            and "convrot" not in n.lower()
        ]
        cache_dir = _hub_cache()
        for name in names:
            path = cached_checkpoint_path(src, cache_dir = cache_dir, names = [name], online = False)
            try:
                header = (
                    _local_header(path) if path else _remote_header(src.location, name, hf_token)
                )
            except Exception as exc:  # noqa: BLE001 -- 404 / offline: next name
                if logger is not None:
                    logger.info("fp8 prequant: %s/%s unreadable: %s", src.location, name, exc)
                continue
            problem = checkpoint_problem(header, base_model)
            if problem:
                if logger is not None:
                    logger.info("fp8 prequant: %s/%s refused: %s", src.location, name, problem)
                return None
            if path is None:
                path = _download_checkpoint_name(
                    src, name, hf_token, cache_dir, propagate_missing = False
                )
            # The downloaded header is the one we load, so recheck it (a republish between the reads).
            problem = checkpoint_problem(_local_header(path), base_model)
            if problem:
                if logger is not None:
                    logger.info("fp8 prequant: %s refused: %s", path, problem)
                return None
            return path
    except Exception as exc:  # noqa: BLE001 -- the dense path is always there
        if logger is not None:
            logger.warning("fp8 prequant: resolution failed, loading bf16: %s", exc)
    return None


def _placeholder_weight(out_f: int, in_f: int, device):
    import torch

    # Zero-stride, one element: PEFT reads only its device / dtype when it attaches LoRA.
    return torch.nn.Parameter(
        torch.zeros((), dtype = torch.bfloat16, device = device).expand(out_f, in_f),
        requires_grad = False,
    )


def load_fp8_transformer(
    transformer_cls,
    config_repo: str,
    path: str,
    device,
    hf_token = None,
):
    """Build ``transformer_cls`` from ``config_repo``'s transformer config and the fp8 checkpoint at ``path``. Raises on any key / shape mismatch."""
    import torch
    import torch.nn as nn
    from accelerate import init_empty_weights
    from safetensors import safe_open

    config = transformer_cls.load_config(config_repo, subfolder = "transformer", token = hf_token)
    with init_empty_weights():
        model = transformer_cls.from_config(config)
    expected = set(model.state_dict().keys())
    keep_fp32 = tuple(getattr(model, "_keep_in_fp32_modules", None) or ())
    dense: dict = {}
    quant: dict = {}
    dev = str(device)
    with safe_open(path, framework = "pt", device = dev) as f:
        keys = list(f.keys())
        for key in keys:
            if key.endswith(_SCALE):
                continue
            if key.endswith(_QDATA):
                stem = key[: -len(_QDATA)]
                quant[stem] = (f.get_tensor(key), f.get_tensor(stem + _SCALE).float())
                continue
            t = f.get_tensor(key)
            name = key[len(_ROOT) :] if key.startswith(_ROOT) else key
            fp32 = any(tok in name.split(".") for tok in keep_fp32)
            dense[name] = t.to(torch.float32 if fp32 else torch.bfloat16)
    got = set(dense) | {f"{s}.weight" for s in quant}
    if got != expected:
        missing = sorted(expected - got)[:5]
        extra = sorted(got - expected)[:5]
        raise ValueError(
            f"checkpoint keys differ from the model: missing {missing}, unexpected {extra}"
        )
    for stem, (q, s) in quant.items():
        mod = model.get_submodule(stem)
        if type(mod) is not nn.Linear or tuple(mod.weight.shape) != tuple(q.shape):
            raise ValueError(f"{stem}: checkpoint fp8 weight {tuple(q.shape)} does not fit {mod!r}")
        mod.weight = _placeholder_weight(q.shape[0], q.shape[1], device)
        setattr(mod, PREQUANT_ATTR, (q, s))
    model.load_state_dict(dense, strict = False, assign = True)
    leftover = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if leftover:
        raise ValueError(f"parameters left unloaded: {leftover[:5]}")
    # from_pretrained hands back an eval-mode module; match it (the trainer calls train() itself).
    return model.to(device).eval()


def dequantize_placeholders(model) -> int:
    """Give every still-pending fp8 Linear a real bf16 weight (q * scale): the layers fp8 training leaves bf16
    (outside the compiled blocks, excluded tokens, a failed conversion, a bf16 fallback). Returns how many."""
    import torch
    import torch.nn as nn

    n = 0
    for mod in model.modules():
        pending = getattr(mod, PREQUANT_ATTR, None)
        if pending is None:
            continue
        q, s = pending
        w = torch.empty(q.shape, dtype = torch.bfloat16, device = q.device)
        with torch.no_grad():
            for i in range(0, q.shape[0], 2048):
                w[i : i + 2048] = (q[i : i + 2048].float() * s[i : i + 2048].reshape(-1, 1)).to(
                    torch.bfloat16
                )
        mod.weight = nn.Parameter(w, requires_grad = False)
        delattr(mod, PREQUANT_ATTR)
        n += 1
    return n
