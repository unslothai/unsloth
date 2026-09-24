# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Compile the prompt-length dimensions of a regionally compiled DiT dynamic from its FIRST forward.

Under automatic dynamic (``dynamic=None``: the max tier, and torchao weights on the default tier) the first prompt
compiles static graphs, and the first prompt of a DIFFERENT token length recompiles every block graph into the
generalised form it keeps from then on. On Qwen-Image-2.1 that second compile is 14 s (int8, default tier) to 51 s
(int8, max tier), paid on the user's second prompt; Qwen-Image pays 52-57 s on the max tier.

torch's ``compiler.config.dynamic_sources`` names the inputs dynamo should treat as dynamic up front, so the first
compile already produces the generalised graphs and no later prompt length recompiles. Only the inputs whose size
tracks the prompt are named, which keeps the torchao CantSplit that a blanket ``dynamic=True`` hits out of reach.

The allowlist is process-global, so it is set only around the owning transformer's forward (hooks, restored even when
the forward raises). A torch without the knob keeps today's behaviour.
"""

from __future__ import annotations

from typing import Any, Optional

# Source names of QwenImage21TransformerBlock.forward inputs whose size follows the prompt token count. torch 2.7
# matches names exactly; 2.8+ also accepts regexes (re.match), which cover the extra segments of an edit prompt. The
# first segment start is always 0 and stays static: a symbol for it is what trips the torchao CantSplit (the target
# rows become ``s0 - s_start``).
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

# QwenImageTransformerBlock.forward (Qwen-Image, 2512, Edit): the text stream and its RoPE half. A step cache
# (FBCache, on by default for this family) calls the blocks through diffusers hooks, where the same inputs arrive as
# ``L['kwargs'][...]`` or, after a graph break, ``___stack0[1][...]``; the suffix patterns cover those (torch 2.8+).
_QWEN_IMAGE_SOURCES: tuple[str, ...] = (
    "L['encoder_hidden_states']",
    "L['encoder_hidden_states_mask']",
    "L['image_rotary_emb'][1]",
    r".*\['encoder_hidden_states'\]$",
    r".*\['encoder_hidden_states_mask'\]$",
    r".*\['image_rotary_emb'\]\[1\]$",
)

# Transformer class name -> dynamic sources of its repeated block.
_FAMILY_SOURCES: dict[str, tuple[str, ...]] = {
    "QwenImage21Transformer2DModel": _QWEN_IMAGE_21_SOURCES,
    "QwenImageTransformer2DModel": _QWEN_IMAGE_SOURCES,
}


def _compiler_config() -> Any:
    try:
        import torch.compiler.config as cfg  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - torch < 2.7 or no torch
        return None
    try:
        getattr(cfg, "dynamic_sources")
    except Exception:  # noqa: BLE001 - knob absent on this build
        return None
    return cfg


def supported() -> bool:
    """Whether this torch honours ``torch.compiler.config.dynamic_sources``."""
    return _compiler_config() is not None


def sources_for(transformer: Any) -> tuple[str, ...]:
    return _FAMILY_SOURCES.get(type(transformer).__name__, ())


def fingerprint(transformer: Any, dynamic: Any) -> Optional[str]:
    """The allowlist a compile of ``transformer`` runs under, for the compile-cache key; None when not armed."""
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
    """Arm the allowlist around ``transformer``'s forward. Idempotent; False when the family has no entry or torch
    lacks the knob, in which case compilation behaves exactly as before."""
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
