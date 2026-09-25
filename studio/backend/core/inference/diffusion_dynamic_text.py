# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Mark a regionally compiled DiT's prompt-length inputs dynamic from its FIRST forward via
``torch.compiler.config.dynamic_sources``, so a new prompt length never recompiles. Only prompt-sized inputs are
named: a blanket ``dynamic=True`` hits torchao CantSplit. The allowlist is process-global, so it is scoped to the
forward by hooks; that needs torch 2.8+ (2.7 reads it once per process, so scoping is ignored or leaks).
"""

from __future__ import annotations

from typing import Any, Optional

# segments[0][0] (always 0) stays static: a symbol there trips torchao CantSplit (rows become ``s0 - s_start``).
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

# Suffix regexes cover FBCache hook paths: ``L['kwargs'][...]`` and, after a graph break, ``___stack0[1][...]``.
_QWEN_IMAGE_SOURCES: tuple[str, ...] = (
    "L['encoder_hidden_states']",
    "L['encoder_hidden_states_mask']",
    "L['image_rotary_emb'][1]",
    r".*\['encoder_hidden_states'\]$",
    r".*\['encoder_hidden_states_mask'\]$",
    r".*\['image_rotary_emb'\]\[1\]$",
)

# MiniMax-H3: packed length S (transformer block), caption length (refiner block), timestep count (temb; i2v adds one).
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
    # 2.8+ only (is_dynamic_source): 2.7 caches its first read per process, breaking per-forward scoping.
    if not callable(getattr(builder, "is_dynamic_source", None)):
        return None
    return cfg


def supported() -> bool:
    return _compiler_config() is not None


def sources_for(transformer: Any) -> tuple[str, ...]:
    return _FAMILY_SOURCES.get(type(transformer).__name__, ())


def fingerprint(transformer: Any, dynamic: Any) -> Optional[str]:
    if dynamic is not None:
        return None
    sources = sources_for(transformer)
    if not sources or not supported():
        return None
    return ",".join(sources)


def _merge(current: str, extra: tuple[str, ...]) -> str:
    parts = [p for p in (current or "").split(",") if p.strip()]
    for s in extra:
        if s not in parts:
            parts.append(s)
    return ",".join(parts)


def install(transformer: Any, logger: Any = None) -> bool:
    if getattr(transformer, "_unsloth_dynamic_text", None) is not None:
        return True
    sources = sources_for(transformer)
    cfg = _compiler_config()
    if not sources or cfg is None:
        return False
    try:
        import torch  # noqa: F401, PLC0415
    except Exception:  # noqa: BLE001
        return False
    saved: list[Optional[str]] = []

    def _enter(module: Any, args: Any) -> None:
        prev = cfg.dynamic_sources
        saved.append(prev)
        cfg.dynamic_sources = _merge(prev, sources)

    def _exit(module: Any, args: Any, output: Any) -> None:
        if saved:
            cfg.dynamic_sources = saved.pop()

    try:
        pre = transformer.register_forward_pre_hook(_enter)
        try:
            post = transformer.register_forward_hook(_exit, always_call = True)
        except TypeError:  # torch < 2.1 has no always_call
            post = transformer.register_forward_hook(_exit)
    except Exception as exc:  # noqa: BLE001 - optimisation only
        if logger is not None:
            logger.warning("diffusion.dynamic_text: install failed: %s", exc)
        return False
    transformer._unsloth_dynamic_text = (pre, post)
    if logger is not None:
        logger.info(
            "diffusion.dynamic_text: prompt-length dims of %s compile dynamic from the first forward",
            type(transformer).__name__,
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
