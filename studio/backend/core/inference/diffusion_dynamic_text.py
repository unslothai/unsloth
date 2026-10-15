# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Mark a DiT's prompt-length inputs dynamic (``dynamic_sources``, scoped to its forward) so a new prompt length
never recompiles; blanket ``dynamic=True`` hits torchao CantSplit.

torch 2.13+ guards symbolic slice bounds on ``s is None`` (Qwen-Image-2.1 ``cache_write_slice``), recompiling per
prompt length; ``install`` restores 2.12's answer. Probe-gated; kill switch ``UNSLOTH_DIFFUSION_SLICE_IDENTITY_FIX=0``."""

from __future__ import annotations

import os
import threading
from typing import Any, Optional

SLICE_IDENTITY_ENV = "UNSLOTH_DIFFUSION_SLICE_IDENTITY_FIX"
_SLICE_LOCK = threading.Lock()
_SLICE_STATE: dict[str, Any] = {}

# segments[0][0] stays static: a symbol there trips torchao CantSplit.
_QWEN_IMAGE_21_SOURCES: tuple[str, ...] = (
    "L['hidden_states']",
    "L['rotary_emb']",
    "L['target_token_mask']",
    "L['key_valid']",
    "L['attention_mask']",
    "L['layer_cache'].k",
    "L['layer_cache'].v",
    "L['cache_write_slice'].stop",
    "L['segments'][0][1]",
    r"L\['segments'\]\[\d+\]\[1\]$",
    r"L\['segments'\]\[[1-9]\d*\]\[0\]$",
)

_QWEN_IMAGE_SOURCES: tuple[str, ...] = (
    "L['encoder_hidden_states']",
    "L['encoder_hidden_states_mask']",
    "L['image_rotary_emb'][1]",
    r".*\['encoder_hidden_states'\]$",
    r".*\['encoder_hidden_states_mask'\]$",
    r".*\['image_rotary_emb'\]\[1\]$",
)

_MINIMAX_H3_SOURCES: tuple[str, ...] = (
    "L['hidden_states']",
    "L['adaln_indices']",
    "L['rotary_emb'][0]",
    "L['rotary_emb'][1]",
    "L['temb']",
)

_FAMILY_SOURCES: dict[str, tuple[str, ...]] = {
    "QwenImage21Transformer2DModel": _QWEN_IMAGE_21_SOURCES,
    "QwenImageTransformer2DModel": _QWEN_IMAGE_SOURCES,
    "MiniMaxH3Transformer3DModel": _MINIMAX_H3_SOURCES,
}

# Unbacked: a backed symbol specialises size 1, and H3's temb has 1 row then 2.
_FAMILY_UNBACKED: dict[str, tuple[str, ...]] = {
    "MiniMaxH3Transformer3DModel": ("L['temb']",),
}


def _compiler_config() -> Any:
    try:
        import torch.compiler.config as cfg  # noqa: PLC0415
        from torch._dynamo.variables import builder  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - torch < 2.7 or no torch
        return None
    try:
        getattr(cfg, "dynamic_sources")
    except Exception:  # noqa: BLE001 - knob absent on this build
        return None
    # 2.8+ only: 2.7 caches its first read per process, so per-forward scoping leaks.
    if not callable(getattr(builder, "is_dynamic_source", None)):
        return None
    return cfg


def supported() -> bool:
    return _compiler_config() is not None


def unbacked_supported() -> bool:
    cfg = _compiler_config()
    if cfg is None:
        return False
    try:
        from torch._dynamo.variables import builder  # noqa: PLC0415
        getattr(cfg, "unbacked_sources")
    except Exception:  # noqa: BLE001 - knob absent on this build
        return False
    return callable(getattr(builder, "is_unbacked_source", None))


def unbacked_sources_for(transformer: Any) -> tuple[str, ...]:
    if not unbacked_supported():
        return ()
    return _FAMILY_UNBACKED.get(type(transformer).__name__, ())


def sources_for(transformer: Any) -> tuple[str, ...]:
    unbacked = set(unbacked_sources_for(transformer))
    return tuple(
        s for s in _FAMILY_SOURCES.get(type(transformer).__name__, ()) if s not in unbacked
    )


def fingerprint(transformer: Any, dynamic: Any) -> Optional[str]:
    if dynamic is True:
        unbacked = unbacked_sources_for(transformer)
        return "unbacked:" + ",".join(unbacked) if unbacked else None
    if dynamic is not None:
        return None
    sources = sources_for(transformer)
    if not sources or not supported():
        return None
    unbacked = unbacked_sources_for(transformer)
    return ",".join(sources) + (";unbacked:" + ",".join(unbacked) if unbacked else "")


def _slice_identity_specialises() -> bool:
    """Whether this torch reads (and guards) a traced slice's bounds to answer ``is``: 2.13+ yes, 2.11/2.12 no."""
    from torch._dynamo.variables import ConstantVariable, SliceVariable  # noqa: PLC0415
    from torch._dynamo.variables.base import NO_SUCH_SUBOBJ  # noqa: PLC0415

    if "get_real_python_backed_value" in vars(SliceVariable):
        return False
    if not callable(getattr(SliceVariable, "get_real_python_backed_value", None)):
        return False
    probe = SliceVariable([ConstantVariable.create(0), ConstantVariable.create(1)])
    return probe.get_real_python_backed_value() is not NO_SUCH_SUBOBJ


def install_slice_identity_fix(logger: Any = None) -> bool:
    """Idempotently keep symbolic slice bounds symbolic under ``is`` / ``is not``. True when the patch is active."""
    if (os.environ.get(SLICE_IDENTITY_ENV) or "").strip().lower() in ("0", "off", "false", "no"):
        return False
    with _SLICE_LOCK:
        if "original" in _SLICE_STATE:
            return True
        if _SLICE_STATE.get("not_needed"):
            return False
        try:
            if not _slice_identity_specialises():
                _SLICE_STATE["not_needed"] = True
                return False
            from torch._dynamo.variables import SliceVariable, SymNodeVariable  # noqa: PLC0415
            from torch._dynamo.variables.base import NO_SUCH_SUBOBJ  # noqa: PLC0415
        except Exception:  # noqa: BLE001 - internals moved: leave torch alone
            return False
        original = SliceVariable.get_real_python_backed_value

        def get_real_python_backed_value(self: Any) -> object:
            if any(isinstance(item, SymNodeVariable) for item in getattr(self, "items", ())):
                return NO_SUCH_SUBOBJ
            return original(self)

        get_real_python_backed_value._unsloth_slice_identity = True  # type: ignore[attr-defined]
        SliceVariable.get_real_python_backed_value = get_real_python_backed_value
        _SLICE_STATE["original"] = original
        _SLICE_STATE["cls"] = SliceVariable
    if logger is not None:
        logger.info(
            "diffusion.dynamic_text: symbolic slice bounds stay symbolic under `is None` (torch 2.13+ recompile fix)"
        )
    return True


def uninstall_slice_identity_fix() -> None:
    with _SLICE_LOCK:
        _SLICE_STATE.pop("not_needed", None)
        cls = _SLICE_STATE.pop("cls", None)
        if _SLICE_STATE.pop("original", None) is not None and cls is not None:
            try:
                del cls.get_real_python_backed_value
            except AttributeError:
                pass


def _arms_slice_bound(sources: tuple[str, ...]) -> bool:
    return any(s.endswith((".start", ".stop", ".step")) for s in sources)


def _merge(current: str, extra: tuple[str, ...]) -> str:
    parts = [p for p in (current or "").split(",") if p.strip()]
    for s in extra:
        if s not in parts:
            parts.append(s)
    return ",".join(parts)


def install(
    transformer: Any,
    logger: Any = None,
    *,
    dynamic: Any = None,
) -> bool:
    """Arm sources for a ``dynamic`` compile: None arms dynamic + unbacked, True only unbacked, False nothing."""
    if dynamic is False:
        return False
    if getattr(transformer, "_unsloth_dynamic_text", None) is not None:
        return True
    if _arms_slice_bound(sources_for(transformer)):
        try:
            install_slice_identity_fix(logger)
        except Exception as exc:  # noqa: BLE001 - optimisation only
            if logger is not None:
                logger.warning("diffusion.dynamic_text: slice identity fix failed: %s", exc)
    unbacked = unbacked_sources_for(transformer)
    sources = sources_for(transformer) if dynamic is None else ()
    cfg = _compiler_config()
    if not (sources or unbacked) or cfg is None:
        return False
    saved: list[tuple[Optional[str], Optional[str]]] = []

    def _enter(module: Any, args: Any) -> None:
        prev = cfg.dynamic_sources
        prev_unbacked = cfg.unbacked_sources if unbacked else None
        saved.append((prev, prev_unbacked))
        cfg.dynamic_sources = _merge(prev, sources)
        if unbacked:
            cfg.unbacked_sources = _merge(prev_unbacked, unbacked)

    def _exit(module: Any, args: Any, output: Any) -> None:
        if saved:
            prev, prev_unbacked = saved.pop()
            cfg.dynamic_sources = prev
            if unbacked:
                cfg.unbacked_sources = prev_unbacked

    try:
        pre = transformer.register_forward_pre_hook(_enter)
        post = transformer.register_forward_hook(_exit, always_call = True)
    except Exception as exc:  # noqa: BLE001 - optimisation only
        if logger is not None:
            logger.warning("diffusion.dynamic_text: install failed: %s", exc)
        return False
    transformer._unsloth_dynamic_text = (pre, post)
    if logger is not None:
        logger.info(
            "diffusion.dynamic_text: %s of %s compile %s from the first forward",
            "prompt-length dims" if sources else ",".join(unbacked),
            type(transformer).__name__,
            "dynamic" if sources else "unbacked",
        )
    return True


def uninstall(transformer: Any) -> None:
    handles = getattr(transformer, "_unsloth_dynamic_text", None)
    if not handles:
        return
    for h in handles:
        try:
            h.remove()
        except Exception:  # noqa: BLE001
            pass
    try:
        del transformer._unsloth_dynamic_text
    except Exception:  # noqa: BLE001
        pass
