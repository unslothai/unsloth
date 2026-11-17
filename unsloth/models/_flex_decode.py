# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""One-token flex decode (zoo flex_decoding) for FastModel generate() over a StaticCache.

Per layer type, builds zoo's graph-refillable decode BlockMask when every layer's shape is supported, else
transformers' SDPA mask, so a model can mix both. Prefill, dynamic caches and custom masks are untouched.
UNSLOTH_FLEX_DECODE: auto (sm80+, MODEL_TYPES, CUDA-graphed decode only: eager flex decode is slower than SDPA),
1 (always), 0 (off)."""

import inspect
import os
import threading
import weakref

import torch

__all__ = [
    "ATTN_KEY",
    "ENV",
    "MODEL_TYPES",
    "available",
    "register",
    "wanted",
    "routable",
    "stats",
    "reset_stats",
]

ATTN_KEY = "unsloth_flex_decode_routed"
ENV = "UNSLOTH_FLEX_DECODE"
# Model types whose flex decode was checked against eager attention.
MODEL_TYPES = frozenset(
    (
        "gemma4",
        "gemma4_text",
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    )
)
# zoo owns these models' decode attention (gpt-oss: sinks and its own graph modes).
EXCLUDED_MODEL_TYPES = frozenset(("gpt_oss",))
_KINDS = {"full_attention": "full", "sliding_attention": "sliding"}

try:
    from unsloth_zoo.flex_attention.decode import (
        decode_block_mask,
        flex_decode_attention,
        flex_decode_supported,
    )
except Exception:
    decode_block_mask = flex_decode_attention = flex_decode_supported = None
try:
    from unsloth_zoo.flex_attention.decode import FlexDecodeUnavailable, dense_decode_mask
except Exception:
    FlexDecodeUnavailable = dense_decode_mask = None

try:
    from torch.nn.attention.flex_attention import BlockMask
except Exception:
    BlockMask = None


def _unwrap(config):
    """The config behind a forwarding proxy (zoo's Gemma 4 _Gemma4KVSharedSafeProxy keeps it in `_real`)."""
    for _ in range(4):
        try:
            real = object.__getattribute__(config, "_real")
        except Exception:
            return config
        if real is None or real is config:
            return config
        config = real
    return config


class _ConfigMap:
    """Keyed by id + weakref: configs are unhashable in 5.x and their __dict__ is saved by save_pretrained."""

    def __init__(self):
        self.items = {}

    def get(
        self,
        config,
        default = None,
    ):
        config = _unwrap(config)
        entry = self.items.get(id(config))
        if entry is None or entry[0]() is not config:
            return default
        return entry[1]

    def set(self, config, value):
        config = _unwrap(config)
        key = id(config)
        try:
            ref = weakref.ref(config, lambda _, key = key, items = self.items: items.pop(key, None))
        except TypeError:
            return value
        self.items[key] = (ref, value)
        return value

    def setdefault(self, config, value):
        current = self.get(config, None)
        return current if current is not None else self.set(config, value)


_LOCK = threading.Lock()
_REGISTERED = False
_LAYERS = _ConfigMap()
_MASK_REASONS = _ConfigMap()
_SUPPORT_CACHE = {}


def available():
    """zoo's flex decode API and a transformers mask interface taking q_length / q_offset (5.x)."""
    if decode_block_mask is None or BlockMask is None:
        return False
    try:
        from transformers.masking_utils import sdpa_mask
    except Exception:
        return False
    params = inspect.signature(sdpa_mask).parameters
    return "q_length" in params and "q_offset" in params


def register():
    """Registers the attention + mask interface pair under ATTN_KEY (idempotent). False when unavailable."""
    global _REGISTERED
    if _REGISTERED:
        return True
    if not available():
        return False
    with _LOCK:
        if _REGISTERED:
            return True
        try:
            from transformers.masking_utils import AttentionMaskInterface
            from transformers.modeling_utils import AttentionInterface
        except Exception:
            return False
        # generate() calls .contiguous() on masks; class-level so it is not a BlockMask field in the graph key.
        if not hasattr(BlockMask, "contiguous"):
            BlockMask.contiguous = lambda self, *args, **kwargs: self
        try:
            # Accelerate moves a split model's layer arguments with BlockMask.to(), which leaves mask_mod's
            # offsets / padding on the first card; zoo's transfer rebuilds the mask on the target card.
            from unsloth_zoo.flex_attention.decode import install_block_mask_transfer
            install_block_mask_transfer()
        except Exception:
            pass
        AttentionInterface.register(ATTN_KEY, _attention)
        AttentionMaskInterface.register(ATTN_KEY, _mask)
        _REGISTERED = True
    return True


def _mode():
    return os.environ.get(ENV, "auto").strip().lower()


def forced():
    """UNSLOTH_FLEX_DECODE=1: flex decode whether or not the call decodes through CUDA graphs, any batch."""
    return _mode() in ("1", "true", "on", "yes", "force")


MAX_BATCH_ENV = "UNSLOTH_FLEX_DECODE_MAX_BATCH"


def batch_ok(batch):
    """auto: batch <= UNSLOTH_FLEX_DECODE_MAX_BATCH (default 4); A100 graphed flex is slower at batch 8."""
    if forced() or batch is None:
        return True
    try:
        limit = int(os.environ.get(MAX_BATCH_ENV, "4"))
    except ValueError:
        limit = 4
    return limit <= 0 or int(batch) <= limit


def wanted(device):
    """Whether one-token steps on `device` should decode on flex (policy only; shapes are checked per layer)."""
    mode = _mode()
    if mode in ("0", "false", "off", "no"):
        return False
    if not register():
        return False
    if mode in ("1", "true", "on", "yes", "force"):
        return True
    try:
        device = torch.device(device)
    except Exception:
        return False
    # auto is calibrated on NVIDIA; ROCm reports gfx numbers as the capability, so it needs =1.
    if device.type != "cuda" or not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    try:
        major = torch.cuda.get_device_capability(device)[0]
    except Exception:
        return False
    return major >= 8


def _model_type(config):
    try:
        return str(getattr(config, "model_type", "") or "").lower()
    except Exception:
        return ""


def _layer_types(config):
    try:
        return list(getattr(config, "layer_types", None) or ())
    except Exception:
        return []


def routable(model, config):
    """Allowed model type, SDPA / eager / flex attention, no chunked layers. Records its layer shapes."""
    model_type = _model_type(config)
    if model_type in EXCLUDED_MODEL_TYPES or any(
        model_type.startswith(t) for t in EXCLUDED_MODEL_TYPES
    ):
        return False
    if _mode() not in ("1", "true", "on", "yes", "force") and model_type not in MODEL_TYPES:
        return False
    try:
        impl = getattr(config, "_attn_implementation", None)
    except Exception:
        impl = None
    if impl not in ("sdpa", "eager", "flex_attention", ATTN_KEY):
        return False
    if any(t not in _KINDS and t not in ("linear_attention",) for t in _layer_types(config)):
        return False
    layers = _describe(model, config)
    return bool(layers)


def _describe(model, config):
    """{kind: {(head_dim, n_q, n_kv)}} over the attention modules that read `config`."""
    cached = _LAYERS.get(config)
    if cached is not None:
        return cached
    layers = {}
    types = _layer_types(config)
    real = _unwrap(config)
    for module in model.modules():
        owner = getattr(module, "config", None)
        if owner is None or _unwrap(owner) is not real:
            continue
        head_dim = getattr(module, "head_dim", None)
        groups = getattr(module, "num_key_value_groups", None)
        layer_idx = getattr(module, "layer_idx", None)
        if not isinstance(head_dim, int) or not isinstance(groups, int) or groups <= 0:
            continue
        layer_type = getattr(module, "layer_type", None)
        if layer_type is None and isinstance(layer_idx, int) and layer_idx < len(types):
            layer_type = types[layer_idx]
        kind = _KINDS.get(layer_type or "full_attention")
        if kind is None:
            continue
        try:
            n_q = int(config.num_attention_heads)
        except Exception:
            continue
        layers.setdefault(kind, set()).add((head_dim, n_q, n_q // groups))
        module.__dict__["_unsloth_flex_decode_kind"] = kind
    _LAYERS.set(config, layers)
    return layers


def _supported(config, kind, dtype, device):
    layers = _LAYERS.get(config, {}).get(kind)
    if not layers:
        return False, f"no {kind} attention layers recorded"
    try:
        device = torch.device(device)
    except Exception:
        return False, "unknown device"
    for head_dim, n_q, n_kv in layers:
        key = (head_dim, n_q, n_kv, dtype, str(device))
        ok = _SUPPORT_CACHE.get(key)
        if ok is None:
            try:
                ok = bool(flex_decode_supported(head_dim, n_q, n_kv, dtype, device))
            except Exception:
                ok = False
            _SUPPORT_CACHE[key] = ok
        if not ok:
            return (
                False,
                f"flex decode unsupported for head_dim {head_dim}, {n_q} / {n_kv} heads, {dtype}",
            )
    return True, None


def _note(config, kind, reason):
    if config is None:
        return
    reasons = _MASK_REASONS.setdefault(config, {})
    reasons[kind] = reason


def _mask(
    batch_size,
    q_length = None,
    kv_length = None,
    q_offset = 0,
    kv_offset = 0,
    mask_function = None,
    attention_mask = None,
    local_size = None,
    config = None,
    use_vmap = False,
    device = "cpu",
    dtype = None,
    **kwargs,
):
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, causal_mask_function

    args = dict(
        batch_size = batch_size,
        q_length = q_length,
        kv_length = kv_length,
        q_offset = q_offset,
        kv_offset = kv_offset,
        mask_function = mask_function,
        attention_mask = attention_mask,
        config = config,
        use_vmap = use_vmap,
        device = device,
        dtype = dtype,
        **kwargs,
    )
    if local_size is not None:
        args["local_size"] = local_size
    kind = "sliding" if local_size else "full"
    reason = None
    if q_length != 1:
        reason = "not a one-token step"
    elif use_vmap:
        reason = "custom mask function"
    elif kind == "full" and mask_function is not causal_mask_function:
        reason = "not a causal mask"
    elif kwargs.get("allow_is_bidirectional_skip") is not None:
        reason = "bidirectional mask"
    else:
        ok, reason = _supported(config, kind, dtype, device)
    _note(config, kind, reason)
    if reason is not None:
        return ALL_MASK_ATTENTION_FUNCTIONS["sdpa"](**args)
    return decode_block_mask(
        batch_size,
        kv_length,
        q_offset,
        kv_offset,
        mask_function = mask_function,
        attention_mask = attention_mask,
        device = device,
    )


def _count(module, reason):
    if torch.compiler.is_compiling():
        # Python counters under trace become guards: every increment would recompile.
        return
    stats = module.__dict__.get("_unsloth_flex_decode_stats")
    if stats is None:
        stats = module.__dict__["_unsloth_flex_decode_stats"] = {"flex": 0, "fallback": {}}
    if reason is None:
        stats["flex"] += 1
    else:
        stats["fallback"][reason] = stats["fallback"].get(reason, 0) + 1


def _default_stance():
    """Compiles new flex decode shapes even under eager_on_recompile (eager flex_attention runs dense)."""
    try:
        from torch._dynamo.eval_frame import _stance
        if getattr(_stance, "stance", "default") == "eager_on_recompile" and hasattr(
            torch.compiler, "set_stance"
        ):
            return torch.compiler.set_stance("default")
    except Exception:
        pass
    return None


def _dense_attention(module, query, key, value, attention_mask, dropout, scaling, softcap, kwargs):
    """Dense fallback: SDPA, or eager attention when a logit softcap is set (SDPA silently drops it)."""
    if softcap is None:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        return sdpa_attention_forward(
            module, query, key, value, attention_mask, dropout = dropout, scaling = scaling, **kwargs
        )
    groups = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(groups, dim = 1)
    value = value.repeat_interleave(groups, dim = 1)
    scores = torch.matmul(query, key.transpose(2, 3)) * scaling
    scores = torch.tanh(scores / softcap) * softcap
    L, S = query.shape[-2], key.shape[-2]
    if (
        attention_mask is None
        and L > 1
        and kwargs.get("is_causal", getattr(module, "is_causal", True))
    ):
        # An elided causal mask: the upper-left causal mask SDPA's is_causal would apply (query i sees keys 0..i).
        attention_mask = torch.ones(L, S, dtype = torch.bool, device = query.device).tril()
    if attention_mask is not None:
        mask = attention_mask[..., : key.shape[-2]]
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            scores = scores + mask
    upcast = torch.float32 if scores.dtype in (torch.float16, torch.bfloat16) else scores.dtype
    probs = torch.softmax(scores, dim = -1, dtype = upcast).to(query.dtype)
    probs = torch.nn.functional.dropout(
        probs, p = dropout, training = getattr(module, "training", False)
    )
    return torch.matmul(probs, value).transpose(1, 2).contiguous(), None


def _attention(
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
    reason = None
    if not isinstance(attention_mask, BlockMask):
        config = getattr(module, "config", None)
        kind = module.__dict__.get("_unsloth_flex_decode_kind", "full")
        reason = _MASK_REASONS.get(config, {}).get(kind) if config is not None else None
        reason = reason or "dense mask"
    elif query.shape[-2] != 1:
        reason = "not a one-token step"
    elif dropout:
        reason = "dropout"
    elif kwargs.get("s_aux") is not None:
        reason = "attention sinks"
    if reason is not None:
        _count(module, reason)
        if isinstance(attention_mask, BlockMask):
            raise RuntimeError(f"Unsloth: flex decode got a BlockMask it cannot run ({reason}).")
        return _dense_attention(
            module, query, key, value, attention_mask, dropout, scaling, softcap, kwargs
        )
    if scaling is None:
        scaling = getattr(module, "scaling", None)
    if scaling is None:
        scaling = query.shape[-1] ** -0.5
    stance = None if torch.compiler.is_compiling() else _default_stance()
    try:
        if stance is None:
            out = flex_decode_attention(query, key, value, attention_mask, scaling, softcap = softcap)
        else:
            with stance:
                out = flex_decode_attention(
                    query, key, value, attention_mask, scaling, softcap = softcap
                )
    except Exception as error:
        if FlexDecodeUnavailable is None or not isinstance(error, FlexDecodeUnavailable):
            raise
        _count(module, "flex_decoding unavailable for this shape")
        dense = dense_decode_mask(attention_mask, key.shape[-2], query.device)
        return _dense_attention(module, query, key, value, dense, dropout, scaling, softcap, kwargs)
    _count(module, None)
    if out.dim() == 4 and out.shape[1] == query.shape[1] and out.shape[2] == query.shape[2]:
        out = out.transpose(1, 2)
    return out.contiguous(), None


def stats(model):
    """{layer_idx: {"kind", "flex", "fallback"}}; counts Python executions (graph replays run none)."""
    out = {}
    for name, module in model.named_modules():
        s = module.__dict__.get("_unsloth_flex_decode_stats")
        if s is None:
            continue
        idx = getattr(module, "layer_idx", name)
        out[idx] = {
            "kind": module.__dict__.get("_unsloth_flex_decode_kind"),
            "flex": s["flex"],
            "fallback": dict(s["fallback"]),
        }
    return out


def reset_stats(model):
    for module in model.modules():
        module.__dict__.pop("_unsloth_flex_decode_stats", None)
