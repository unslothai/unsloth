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

# FLUX.3 Klein (distilled and base): the pipeline pads the prompt to a multiple of 80 tokens, so every new bucket was
# a new shape. The per-mode block runs the text stream dense and the image stream int8 (two graphs by weight type),
# and the joint block takes the text beside the image tokens; both recompiled on the first longer prompt (two graphs,
# ~12-20 s). Arming the block inputs makes the first compile cover every length. Same images bit for bit on a B200,
# no recompile from 5 to 138 words, no slower steady state.
_FLUX3_SOURCES: tuple[str, ...] = (
    "L['hidden_states']",
    "L['temb_mod']",
    "L['encoder_hidden_states']",
    "L['temb_mod_context']",
    "L['image_rotary_emb'][0]",
    "L['image_rotary_emb'][1]",
)

# FLUX.1: Studio pads T5 to 256 tokens, or 512 for a longer prompt (diffusion_text_length.flux_t5_sequence_length),
# so the first prompt past 256 tokens was a new shape. Double blocks see the text stream beside the image tokens,
# single blocks take both, and the rotary tables span text + image.
_FLUX1_SOURCES: tuple[str, ...] = (
    "L['hidden_states']",
    "L['encoder_hidden_states']",
    "L['image_rotary_emb'][0]",
    "L['image_rotary_emb'][1]",
    r".*\['hidden_states'\]$",
    r".*\['encoder_hidden_states'\]$",
    r".*\['image_rotary_emb'\]\[[01]\]$",
)

# Z-Image: the pipeline drops the padded caption tokens (pipeline_z_image.py, ``prompt_embeds[i][prompt_masks[i]]``)
# and the transformer pads the caption to a multiple of 32, so every new 32-token bucket was a new shape. The
# context refiner sees the caption alone, the main layers the unified caption + image sequence, both positionally
# as (x, attn_mask, freqs_cis, ...).
_ZIMAGE_SOURCES: tuple[str, ...] = (
    "L['x']",
    "L['attn_mask']",
    "L['freqs_cis']",
    r".*\['args'\]\[[0-2]\]$",
)

# Cosmos3 tokenizes the prompt unpadded (pipeline_cosmos3_omni.py ``_tokenize``): every prompt length is a new shape.
_COSMOS3_SOURCES: tuple[str, ...] = (
    "L['und_seq']",
    "L['rotary_emb'][0]",
    "L['rotary_emb'][1]",
)

# MOSS-Image: unpadded text (cond and uncond differ), so every prompt is a new length.
_MOSS_IMAGE_SOURCES: tuple[str, ...] = (
    "L['x']",
    "L['rope'][0]",
    "L['rope'][1]",
)
# Prompt-length policy for every denoiser class Studio loads (the ONE list; test_diffusion_prompt_length_policy checks
# it covers diffusion_families and video_families). A tuple: the denoiser's text length follows the prompt, and these
# are the repeated block's dynamo sources that carry it. FIXED_LENGTH: the pipeline pads every prompt to one length
# (padding="max_length" to max_sequence_length, masked), so a new prompt never changes a shape.
FIXED_LENGTH = "fixed"
PROMPT_LENGTH_POLICY: dict[str, Any] = {
    "FluxTransformer2DModel": _FLUX1_SOURCES,
    "Flux2Transformer2DModel": FIXED_LENGTH,  # Mistral / Qwen3 padded to max_sequence_length 512
    "Flux3Transformer2DModel": _FLUX3_SOURCES,
    "QwenImageTransformer2DModel": _QWEN_IMAGE_SOURCES,
    "QwenImage21Transformer2DModel": _QWEN_IMAGE_21_SOURCES,
    "ZImageTransformer2DModel": _ZIMAGE_SOURCES,
    "Krea2Transformer2DModel": FIXED_LENGTH,  # padded to max_sequence_length 512
    "Lumina2Transformer2DModel": FIXED_LENGTH,  # Gemma padded to max_sequence_length 256
    "HunyuanImageTransformer2DModel": FIXED_LENGTH,  # padded to the tokenizer max length
    "HiDreamImageTransformer2DModel": FIXED_LENGTH,  # T5 / Llama padded to max_sequence_length 128
    "Ideogram4Transformer2DModel": FIXED_LENGTH,  # token ids left-padded into max_sequence_length 2048
    "UNet2DConditionModel": FIXED_LENGTH,  # CLIP padded to 77
    "MiniMaxH3Transformer3DModel": _MINIMAX_H3_SOURCES,
    "MOSSImageTransformer": _MOSS_IMAGE_SOURCES,
    "LTX2VideoTransformer3DModel": FIXED_LENGTH,  # Gemma left-padded to max_sequence_length 1024
    "WanTransformer3DModel": FIXED_LENGTH,  # UMT5 padded to max_sequence_length 512
    "HunyuanVideo15Transformer3DModel": FIXED_LENGTH,  # padded to the tokenizer max length
    "Cosmos3OmniTransformer": _COSMOS3_SOURCES,
}


def prompt_length_varies(transformer: Any) -> bool:
    """Whether ``transformer``'s text length follows the prompt (a class or an instance), so every new prompt length is
    a new input shape. An unclassified class answers False: it keeps what it does today."""
    cls = transformer if isinstance(transformer, type) else type(transformer)
    return isinstance(PROMPT_LENGTH_POLICY.get(cls.__name__), tuple)


# What the compile arms. int8 / fp8 FLUX.1 and Z-Image (torchao weights compile with automatic dynamic) recompiled on
# new text lengths: FLUX.1 both blocks on the first prompt past 256 T5 tokens and again on the first new resolution (31 s
# and 60 s on a loaded B200); Z-Image's shared block on the caption, image and unified lengths (12 recompiles in the
# first render) and again on the first longer prompt. Armed, the first compile covers every text length and image size.
_FAMILY_SOURCES: dict[str, tuple[str, ...]] = {
    "FluxTransformer2DModel": _FLUX1_SOURCES,
    "ZImageTransformer2DModel": _ZIMAGE_SOURCES,
    "Flux3Transformer2DModel": _FLUX3_SOURCES,
    "QwenImage21Transformer2DModel": _QWEN_IMAGE_21_SOURCES,
    "QwenImageTransformer2DModel": _QWEN_IMAGE_SOURCES,
    "MiniMaxH3Transformer3DModel": _MINIMAX_H3_SOURCES,
    "MOSSImageTransformer": _MOSS_IMAGE_SOURCES,
}

_ARMED_WHEN_DYNAMIC: frozenset[str] = frozenset({"QwenImageTransformer2DModel"})


def _armed_when_dynamic(transformer: Any) -> bool:
    return type(transformer).__name__ in _ARMED_WHEN_DYNAMIC


# Unbacked: a backed symbol specialises size 1, and H3's temb has 1 row on step 1, 2 after (a second compile).
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
    # 2.8+ only (is_dynamic_source): 2.7 caches its first read per process, so per-forward scoping leaks.
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


def _torchao_float8_weights(transformer: Any) -> bool:
    """Whether a torchao float8 tensor holds any of ``transformer``'s weights (wrapped in a plain Parameter or not)."""
    try:
        from .diffusion_speed import torchao_weight_tensors
        for t in torchao_weight_tensors(transformer):
            if "float8" in type(t).__name__.lower():
                return True
            # torchao < 0.12: AffineQuantizedTensor + Float8Layout
            layout = getattr(t, "_layout", None)
            if layout is not None and "float8" in type(layout).__name__.lower():
                return True
    except Exception:  # noqa: BLE001 - unreadable: keep the family's choice
        return False
    return False


def unbacked_sources_for(transformer: Any) -> tuple[str, ...]:
    if not unbacked_supported():
        return ()
    unbacked = _FAMILY_UNBACKED.get(type(transformer).__name__, ())
    if unbacked and _torchao_float8_weights(transformer):
        # torchao float8 activation quant guards on the row count (``_get_reduction_params``): an unbacked
        # ``temb`` is a data-dependent error and the compile falls back to eager. Backed costs one respecialise.
        return ()
    return unbacked


def sources_for(transformer: Any) -> tuple[str, ...]:
    # A source is dynamic OR unbacked; without the unbacked knob it stays on the dynamic list.
    unbacked = set(unbacked_sources_for(transformer))
    return tuple(
        s for s in _FAMILY_SOURCES.get(type(transformer).__name__, ()) if s not in unbacked
    )


def fingerprint(transformer: Any, dynamic: Any) -> Optional[str]:
    if dynamic is True and not _armed_when_dynamic(transformer):
        # dynamic=True already makes every dim dynamic; only the unbacked sources are armed (see install).
        unbacked = unbacked_sources_for(transformer)
        return "unbacked:" + ",".join(unbacked) if unbacked else None
    if dynamic is not None and dynamic is not True:
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
        return False  # torch gives slices their own answer; trust it
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
            # Stock builds a fresh slice (identical to nothing): "no backing object" keeps every answer, minus the guard.
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
    """Arm sources for a ``dynamic`` compile: None arms dynamic + unbacked, True only unbacked (dynamic + unbacked
    for :data:`_ARMED_WHEN_DYNAMIC`), False nothing."""
    if dynamic is False:
        return False
    if getattr(transformer, "_unsloth_dynamic_text", None) is not None:
        return True
    # dynamic=True makes the slice bound symbolic too, so the fix keys on the family, not on what gets armed.
    if _arms_slice_bound(sources_for(transformer)):
        try:
            install_slice_identity_fix(logger)
        except Exception as exc:  # noqa: BLE001 - optimisation only
            if logger is not None:
                logger.warning("diffusion.dynamic_text: slice identity fix failed: %s", exc)
    unbacked = unbacked_sources_for(transformer)
    sources = (
        sources_for(transformer) if dynamic is None or _armed_when_dynamic(transformer) else ()
    )
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
