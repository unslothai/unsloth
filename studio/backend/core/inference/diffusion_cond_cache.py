# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Prompt-embedding cache, off unless UNSLOTH_DIFFUSION_COND_CACHE_DIR is set; bypassed while LoRA
adapters are attached; best-effort, any cache failure falls back to the real encode; torch
imported lazily"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from typing import Any, Optional

_ENV_DIR = "UNSLOTH_DIFFUSION_COND_CACHE_DIR"

# Never change the embeddings: device is placement only and no encode path draws RNG.
_KEY_EXCLUDED_ARGS = frozenset({"device", "generator"})


def cache_dir() -> Optional[str]:
    """The configured cache directory, or None when the cache is off. A blank
    value means off (matching the trainers' ``cond_cache_dir`` semantics)."""
    value = (os.environ.get(_ENV_DIR) or "").strip()
    return value or None


def _json_safe(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_json_safe(v) for v in value)
    return False


# Diffusers quant schemes that cannot host ControlNet cleanly (torchao tensor subclasses); gated off like LoRA, with GGUF-via-diffusers
_SLOT_NONE = -1
_SLOT_TENSOR = -2


def _flatten(result: Any) -> Optional[list]:
    """Flatten an encode_prompt result tuple of tensors, None and tensor-lists into [layout_tensor,
    *tensors] for the cache, or None if anything else is present (those calls are not cached)."""
    if not isinstance(result, tuple) or not result:
        return None
    layout: list[int] = []
    flat: list[Any] = []
    for item in result:
        if item is None:
            layout.append(_SLOT_NONE)
        elif isinstance(item, (list, tuple)):
            if not all(hasattr(t, "detach") for t in item):
                return None
            layout.append(len(item))
            flat.extend(item)
        elif hasattr(item, "detach"):
            layout.append(_SLOT_TENSOR)
            flat.append(item)
        else:
            return None
    import torch  # noqa: PLC0415

    return [torch.tensor(layout, dtype = torch.int64), *flat]


def _unflatten(stored: tuple, device: Any) -> tuple:
    """Rebuild the ``encode_prompt`` result from a cached ``_flatten`` record,
    moving every tensor to ``device`` (dtype is preserved as stored)."""
    layout = stored[0].tolist()
    tensors = list(stored[1:])
    out: list[Any] = []
    index = 0
    for code in layout:
        if code == _SLOT_NONE:
            out.append(None)
        elif code == _SLOT_TENSOR:
            out.append(tensors[index].to(device))
            index += 1
        else:
            out.append([t.to(device) for t in tensors[index : index + code]])
            index += code
    return tuple(out)


def install(
    pipe: Any,
    *,
    family: str,
    repo_id: str,
    dtype: Any,
    base_repo: Any = None,
    te_quant: Any = None,
    logger: Any = None,
) -> bool:
    """Instance-level wrapper of pipe.encode_prompt, so it dies with the pipe and never mutates the
    class."""
    root = cache_dir()
    if root is None:
        return False
    encode = getattr(pipe, "encode_prompt", None)
    if not callable(encode):
        return False
    try:
        # Lazy: core.training imports parts of core.inference, so a module-level import would be circular.
        from core.training.diffusion_train_extras import PersistentConditioningCache
        signature = inspect.signature(encode)
        cache = PersistentConditioningCache(root, family, 0)
    except Exception as exc:  # noqa: BLE001 - cache is best-effort
        if logger is not None:
            logger.warning("diffusion.cond_cache: install failed: %s", exc)
        return False

    # GGUF checkpoints take text encoders from the base repo, so it keys the cache too.
    base_ref = base_repo if base_repo else repo_id
    load_fp = {
        "repo": str(repo_id),
        "repo_rev": _source_revision(repo_id),
        "base": str(base_ref),
        "base_rev": _source_revision(base_ref),
        "dtype": str(dtype),
        "te_quant": str(te_quant) if te_quant is not None else "none",
        "diffusers": _diffusers_version(),
    }
    stats = {"hits": 0, "misses": 0}

    def cached_encode_prompt(*args: Any, **kwargs: Any) -> Any:
        try:
            # LoRA may target the text encoders; adapters attached -> always encode.
            if getattr(pipe, "_unsloth_loras", ()):
                return encode(*args, **kwargs)
            # Textual-inversion embeddings applied for this generation: the text alone does not key the result.
            if getattr(pipe, "__dict__", {}).get("_unsloth_embeddings"):
                return encode(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            keyed = {
                name: value
                for name, value in bound.arguments.items()
                if name not in _KEY_EXCLUDED_ARGS
            }
            if not all(_json_safe(v) for v in keyed.values()):
                return encode(*args, **kwargs)
            payload = json.dumps({"load": load_fp, "args": keyed}, sort_keys = True, default = str)
            key = cache.text_key(
                f"inference::{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"
            )
            hit = cache.get(key)
            if hit is not None:
                stats["hits"] += 1
                return _unflatten(hit, _target_device(pipe, bound))
        except Exception as exc:  # noqa: BLE001 - never fail a generation over the cache
            if logger is not None:
                logger.warning("diffusion.cond_cache: lookup failed: %s", exc)
            return encode(*args, **kwargs)
        result = encode(*args, **kwargs)
        try:
            flat = _flatten(result)
            if flat is not None:
                cache.put(key, flat)
                stats["misses"] += 1
        except Exception as exc:  # noqa: BLE001 - a failed write only skips reuse
            if logger is not None:
                logger.warning("diffusion.cond_cache: store failed: %s", exc)
        return result

    pipe.encode_prompt = cached_encode_prompt
    pipe._unsloth_cond_cache_stats = stats
    if logger is not None:
        logger.info(
            "diffusion.cond_cache: enabled at %s (family=%s); repeated prompts skip "
            "the text-encoder forward",
            root,
            family,
        )
    return True


def _target_device(pipe: Any, bound: inspect.BoundArguments) -> Any:
    """The device a hit's tensors must land on: the caller's explicit ``device``
    argument when given, else the pipeline's execution device, else CPU."""
    device = bound.arguments.get("device")
    if device is not None:
        return device
    device = getattr(pipe, "_execution_device", None)
    return device if device is not None else "cpu"


def _diffusers_version() -> Optional[str]:
    try:
        import diffusers  # noqa: PLC0415
        return str(getattr(diffusers, "__version__", None))
    except Exception:  # noqa: BLE001
        return None


def _source_revision(ref: Any) -> str:
    """Delegates to the trainer's source_revision so both caches agree on what counts as the same source."""
    try:
        from core.training.diffusion_train_extras import source_revision  # noqa: PLC0415
        return source_revision(ref)
    except Exception:  # noqa: BLE001 - best-effort, never block a load
        return "unresolved"
