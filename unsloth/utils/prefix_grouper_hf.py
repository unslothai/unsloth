# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""PrefixGrouper on transformers-native attention modules (FastModel families).

With ``prefix_seg_info`` present, each wrapped attention module's interface lookup is pointed at
the shared-prefix flex kernel for that call; HF still computes q / k / v. The wrapper raises if the
kernel was not reached, so callers fall back instead of leaking across suffixes. MoE families also
need ``UNSLOTH_GRPO_PREFIX_GROUPER_MOE=1``.
"""

from __future__ import annotations

import functools
import os
from typing import Optional

PG_ATTN_IMPL = "unsloth_prefix_grouper"
# Eligibility is per module config, not attention_dispatch's class-level APPLIES_PREFIX_SEG_INFO.
_WRAPPED = "_unsloth_pg_hf_wrapper"
_ORIGINAL = "_unsloth_pg_hf_original_forward"


# Opt-in until a GRPO step-time A/B shows a win: UNSLOTH_GRPO_PREFIX_GROUPER_MOE=1.
MOE_ENV = "UNSLOTH_GRPO_PREFIX_GROUPER_MOE"
# Set once every decoder layer routes prefix_seg_info; lifts the GRPO gates' MoE exclusion.
MOE_OK_ATTR = "_unsloth_prefix_grouper_moe_ok"


def moe_opt_in() -> bool:
    return os.environ.get(MOE_ENV, "0").lower() not in ("0", "false", "no", "off", "")


def _routing_replay_requested() -> bool:
    # Routing replay keys on position_ids resets, which the shared stream lacks per suffix.
    return os.environ.get("UNSLOTH_GRPO_ROUTING_REPLAY", "0") == "1"


def _qwen3_moe_text_ok(module) -> bool:
    # Per-token MoE routing, so deduplicating the prompt does not change routing.
    config = getattr(module, "config", None)
    return (
        getattr(config, "model_type", None) == "qwen3_moe"
        and getattr(module, "is_causal", True)
        and not getattr(config, "attention_dropout", 0)
        and not getattr(config, "output_router_logits", False)
        and moe_opt_in()
        and not _routing_replay_requested()
    )


# Attention class name -> per-module eligibility. Verified: Qwen3-MoE text (transformers 4.57 / 5.x).
# Gemma-3 (head_dim 256) is correct but about 2x slower on L4, so it is not listed.
_FAMILIES = {
    "Qwen3MoeAttention": _qwen3_moe_text_ok,
}
_MOE_FAMILIES = frozenset(("Qwen3MoeAttention",))
_REQUIRED_ATTRS = ("q_proj", "k_proj", "v_proj", "o_proj", "scaling", "config", "layer_idx")


def _seg_max_span(seg) -> Optional[int]:
    """Longest prefix + suffix span in the table, from its host-side signature (no device sync)."""
    sig = getattr(seg, "signature", None)
    try:
        if sig[0] == "multi":
            return max(P + max(Rs) for P, Rs in sig[2])
        if sig[0] == "single":
            return sig[2] + max(e - s for s, e in sig[3])
    except Exception:
        pass
    return None


def prefix_grouper_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout = 0.0,
    scaling = None,
    sliding_window = None,
    softcap = None,
    **kwargs,
):
    """HF attention-interface entry: q [1, H, T, D], k / v [1, Hkv, T, D] -> ([1, T, H, D], None)."""
    seg = kwargs.get("prefix_seg_info", None)
    if seg is None or not hasattr(seg, "T_pad") or not hasattr(seg, "signature"):
        raise RuntimeError("PrefixGrouper: attention called without a prefix_seg_info table.")
    if dropout:
        raise RuntimeError("PrefixGrouper: attention dropout is not supported.")
    if softcap or getattr(module, "attn_logit_softcapping", None):
        raise RuntimeError("PrefixGrouper: attention logit softcapping is not supported.")
    if query.shape[0] != 1:
        raise RuntimeError(f"PrefixGrouper: expected one flat stream, got batch {query.shape[0]}.")
    if sliding_window is not None:
        span = _seg_max_span(seg)
        # The kernel is plain causal inside a segment, equal to the window only when the span fits.
        if span is None or span > int(sliding_window):
            raise RuntimeError(
                f"PrefixGrouper: segment span {span} exceeds sliding_window {sliding_window}."
            )
    if scaling is None:
        scaling = query.shape[-1] ** -0.5
    from .prefix_grouper_kernel import flex_shared_prefix_attention

    out = flex_shared_prefix_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        seg,
        scale = float(scaling),
    )
    module.__dict__["_unsloth_pg_hf_hit"] = True
    return out, None


class _ConfigView:
    """Read-through view of a config whose attention implementation is the PrefixGrouper kernel."""

    __slots__ = ("_unsloth_base",)

    def __init__(self, base):
        object.__setattr__(self, "_unsloth_base", base)

    @property
    def _attn_implementation(self):
        return PG_ATTN_IMPL

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_unsloth_base"), name)


def _wrap_forward(original):
    @functools.wraps(original)
    def forward(self, *args, **kwargs):
        seg = kwargs.get("prefix_seg_info", None)
        if seg is None:
            return original(self, *args, **kwargs)
        check = _FAMILIES.get(type(self).__name__)
        if check is None or not check(self):
            raise RuntimeError(
                f"PrefixGrouper: {type(self).__name__} with this config does not support prefix_seg_info."
            )
        if (
            kwargs.get("past_key_values", None) is not None
            or kwargs.get("past_key_value", None) is not None
        ):
            raise RuntimeError("PrefixGrouper: prefix_seg_info requires prefill with no KV cache.")
        base = self.__dict__.get("config", None)
        self.__dict__["config"] = _ConfigView(base)
        self.__dict__["_unsloth_pg_hf_hit"] = False
        try:
            out = original(self, *args, **kwargs)
        finally:
            self.__dict__["config"] = base
        if not self.__dict__.pop("_unsloth_pg_hf_hit", False):
            raise RuntimeError(
                f"PrefixGrouper: {type(self).__name__}.forward did not dispatch through the "
                "attention interface, so prefix_seg_info was not applied."
            )
        return out

    setattr(forward, _WRAPPED, True)
    setattr(forward, _ORIGINAL, original)
    return forward


def _register_interface() -> bool:
    try:
        from transformers.modeling_utils import AttentionInterface
    except Exception:
        return False
    try:
        AttentionInterface.register(PG_ATTN_IMPL, prefix_grouper_attention)
    except Exception:
        return False
    return True


# Composite configs verified text-only (the GRPO gate excludes pixel_values / token_type_ids).
_COMPOSITE_MODEL_TYPES = ()


def _text_config_of(config):
    """The config the decoder attention reads, or None for unsupported top-level configs."""
    if config is None or getattr(config, "is_encoder_decoder", False):
        return None
    get_text_config = getattr(config, "get_text_config", None)
    text_config = config
    if callable(get_text_config):
        try:
            text_config = get_text_config()
        except Exception:
            return None
    if text_config is config:
        return None if getattr(config, "vision_config", None) is not None else config
    if getattr(config, "model_type", None) in _COMPOSITE_MODEL_TYPES:
        return text_config
    return None


def enable_hf_prefix_seg_info(model, num_layers) -> bool:
    """Route prefix_seg_info through the flex kernel on every eligible HF attention module of model.

    True only when all ``num_layers`` decoder layers have an eligible attention module reading the
    model's text config (callers cap the segment span at that config's ``sliding_window``).
    Nothing is patched unless the whole model qualifies.
    """
    if not isinstance(num_layers, int) or num_layers <= 0:
        return False
    text_config = _text_config_of(getattr(model, "config", None))
    if text_config is None:
        return False
    found = []
    for module in model.modules():
        check = _FAMILIES.get(type(module).__name__)
        if check is None:
            continue
        if not all(hasattr(module, a) for a in _REQUIRED_ATTRS) or not check(module):
            return False
        if module.config is not text_config:
            return False
        forward = getattr(type(module), "forward", None)
        if not callable(forward) or getattr(forward, "_unsloth_gated", False):
            return False
        found.append(module)
    if len({id(m) for m in found}) < num_layers or len({m.layer_idx for m in found}) < num_layers:
        return False
    if not _register_interface():
        return False
    for cls in {type(m) for m in found}:
        forward = cls.__dict__.get("forward", None) or cls.forward
        if not getattr(forward, _WRAPPED, False):
            cls.forward = _wrap_forward(forward)
    if any(type(m).__name__ in _MOE_FAMILIES for m in found):
        try:
            setattr(model, MOE_OK_ATTR, True)
        except Exception:
            pass
    return True


__all__ = [
    "MOE_ENV",
    "MOE_OK_ATTR",
    "moe_opt_in",
    "PG_ATTN_IMPL",
    "enable_hf_prefix_seg_info",
    "prefix_grouper_attention",
]
