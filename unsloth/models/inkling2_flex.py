# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""FlexAttention for Inkling 2's language attention (``inkling2`` / ``inkling2_loop``, transformers >= 5.24).

transformers ships Inkling 2 eager only (``_supports_sdpa / flash / flex = False``): every layer builds the full
``L x L`` scores, the ``L x L`` gathered relative bias and an ``L x L`` additive mask. Eager computes, per head,

    s[q, k]  = scale * <q_q, k_k> + bias[q, q - k] + mask[q, k]
    p[q, :]  = softmax([s[q, :], sink[q]])[:-1]            (the sink logit takes mass and has no value)
    out[q]   = sum_k p[q, k] v_k

with ``bias[q, d] = (rel_states[q] @ proj)[d]`` for ``0 <= d < rel_extent`` and 0 otherwise. q / k / v already carry
the scaled L2 clips, the global layers' log-position scaling and the short convolutions, so only the three lines
above are attention. Here:

- the mask is a BlockMask built from the same transformers mask function (causal, the ``kv > q - window`` sliding
  window on ``hybrid_sliding`` layers, packed-sequence and padding terms), so skipped tiles are never computed;
- ``score_mod`` adds ``rel_logits[b, q, h, q - kv]`` from the per-token ``[B, L, H, rel_extent]`` table when
  ``0 <= q - kv < rel_extent`` (0 beyond it, as eager), so the ``L x L`` bias is never built;
- the sink is merged exactly from flex's log-sum-exp: with ``Z = sum_k exp(s[q, k])``,
  ``softmax([s, sink])[:-1] = softmax(s) * Z / (Z + exp(sink)) = softmax(s) * sigmoid(lse - sink)``.

UNSLOTH_INKLING2_FLEX: ``0`` keeps transformers' eager attention, ``1`` uses flex on every CUDA device for prefill /
training forwards, ``auto`` (default) uses flex on Ampere and newer and keeps eager on Turing (T4), where flex was
measured 1.7x slower (see ``_auto_min_length``). Decode steps (one query against a cache) and every other call keep the
eager path.
"""

import contextvars
import copy
import functools
import inspect
import os
import sys
import types

import torch

__all__ = [
    "ATTN_KEY",
    "ENV",
    "available",
    "enable_inkling2_flex",
    "disable_inkling2_flex",
    "inkling2_flex_attention",
    "stats",
    "reset_stats",
]

ATTN_KEY = "unsloth_inkling2_flex"
ENV = "UNSLOTH_INKLING2_FLEX"
MIN_LEN_ENV = "UNSLOTH_INKLING2_FLEX_MIN_LENGTH"
_ATTENTION_CLASS_SUFFIXES = ("SlidingAttention", "FullAttention")

# Lines of the transformers 5.24 code this module re-implements; a different modeling file keeps eager.
_ATTENTION_SOURCE_MARKERS = (
    'query_states = self.q_proj(hidden_states, kwargs["position_ids"])',
    "key_states = self.k_proj(",
    "value_states = self.v_proj(",
    "cache.update(key_states, value_states, cache_layer_idx, kwargs=kwargs)",
    'position_bias = self.relative_position_bias(hidden_states, relative_distance, kwargs["position_ids"])',
    "sinks = self.sink_proj(hidden_states).transpose(1, 2).unsqueeze(-1)",
    "scaling=self.scaling,",
    "attn_output = self.o_proj(attn_output, hidden_states)",
)
_BIAS_SOURCE_MARKERS = (
    "relative_states = self.linear(hidden_states).view(*hidden_states.shape[:-1], self.num_heads, -1)",
    "relative_states = self.norm(relative_states, position_ids)",
    "rel_logits = (relative_states @ self.proj).transpose(1, 2)",
    "masked_fill((relative_distance < 0) | (relative_distance >= self.rel_extent), 0.0)",
)
_EAGER_SOURCE_MARKERS = (
    "attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling",
    "attn_weights = attn_weights + position_bias",
    "combined_logits = torch.cat([attn_weights, sinks], dim=-1).float()",
    "[..., :-1].to(query.dtype)",
)
_RELATIVE_DISTANCE_MARKERS = ("return q_positions[:, None] - kv_positions[None, :]",)

_STATS = {
    "flex": 0,
    "eager": 0,
    "block_masks": 0,
    "dense_masks": 0,
    "recomputed": 0,
    "distance_skipped": 0,
}
# Per text-model forward: which mask types were built as BlockMasks (so their L x L relative distances are not needed).
_FORWARD_STATE = contextvars.ContextVar("unsloth_inkling2_flex_forward", default = None)
_PATCHED = {}
_FAILED = {}
_COMPILED_FLEX = None


def _torch_version():
    try:
        return tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    except Exception:
        return (0, 0)


def _mode():
    value = os.environ.get(ENV, "auto").strip().lower()
    if value in ("0", "false", "off", "no"):
        return "0"
    if value in ("1", "true", "on", "yes", "force"):
        return "1"
    return "auto"


def _flex_symbols():
    try:
        from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
    except Exception:
        return None
    return BlockMask, create_block_mask, flex_attention


def available():
    """flex_attention with log-sum-exp output and gradients through captured buffers (torch >= 2.7)."""
    return _flex_symbols() is not None and _torch_version() >= (2, 7)


def _is_block_mask(mask):
    symbols = _flex_symbols()
    return symbols is not None and isinstance(mask, symbols[0])


def stats():
    return dict(_STATS)


def reset_stats():
    for key in list(_STATS):
        _STATS[key] = 0


def _count(key):
    if not torch.compiler.is_compiling():
        _STATS[key] += 1


# ---------------------------------------------------------------------------------------------------------------------
# Kernel


@functools.lru_cache(maxsize = None)
def _kernel_options_for(index):
    # The zoo's prefill options: smaller tiles on <= 24 GB cards (T4 / L4 shared memory) and FORCE_USE_FLEX_ATTENTION,
    # else short query blocks route to flex decoding, which has no config once q_len * GQA group > 128.
    try:
        from unsloth_zoo.flex_attention.utils import PREFILL_KERNEL_OPTIONS
        return tuple(PREFILL_KERNEL_OPTIONS.items())
    except Exception:
        pass
    options = {"FORCE_USE_FLEX_ATTENTION": True}
    try:
        total_gb = torch.cuda.get_device_properties(index).total_memory / 2**30
    except Exception:
        return tuple(options.items())
    if total_gb <= 16:
        options.update(BLOCK_M = 32, BLOCK_N = 32, BLOCK_M1 = 32, BLOCK_N1 = 32, BLOCK_M2 = 32, BLOCK_N2 = 32)
    elif total_gb <= 24:
        options.update(BLOCK_M = 64, BLOCK_N = 64, BLOCK_M1 = 32, BLOCK_N1 = 64, BLOCK_M2 = 64, BLOCK_N2 = 32)
    return tuple(options.items())


def _kernel_options(
    device,
    dtype = None,
    head_dim = None,
):
    if device.type != "cuda":
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    options = dict(_kernel_options_for(index))
    if dtype == torch.float32 and (head_dim or 0) >= 128:
        # float32 at head_dim 128: the backward template needs 69.6 KB of shared memory on T4 (64 KB) even at 32-wide
        # one-stage tiles, and 106 KB on L4 (99 KB) at the zoo's 64-wide ones. Only parity / UNSLOTH_FORCE_FLOAT32 runs get here.
        for key in ("BLOCK_M", "BLOCK_N", "BLOCK_M1", "BLOCK_N1", "BLOCK_M2", "BLOCK_N2"):
            options[key] = 16
        options["num_stages"] = 1
    return options


def _lse_kwargs():
    try:
        from torch.nn.attention.flex_attention import AuxRequest
        return {"return_aux": AuxRequest(lse = True)}
    except Exception:
        return {"return_lse": True}


def _split_lse(output, lse_kwargs):
    out, aux = output
    return out, (aux.lse if "return_aux" in lse_kwargs else aux)


def _flex_function(device):
    """torch.compile'd flex_attention on CUDA (the uncompiled HOP materializes the scores). CPU keeps the HOP: only
    the tests run it, through flex's reference math."""
    global _COMPILED_FLEX
    flex_attention = _flex_symbols()[2]
    if device.type != "cuda":
        return flex_attention
    if _COMPILED_FLEX is None:
        _COMPILED_FLEX = torch.compile(flex_attention, dynamic = None)
    return _COMPILED_FLEX


def _relative_bias_score_mod(rel_logits, rel_extent):
    """score + rel_logits[b, q, h, q - kv] for 0 <= q - kv < rel_extent, else score (eager's masked_fill(.., 0))."""
    last = rel_extent - 1

    def score_mod(score, b, h, q_idx, kv_idx):
        distance = q_idx - kv_idx
        bias = rel_logits[b, q_idx, h, torch.clamp(distance, 0, last)].to(score.dtype)
        return score + torch.where((distance >= 0) & (distance < rel_extent), bias, 0.0)

    return score_mod


def _flex_core(query, key, value, block_mask, rel_logits, sinks, scaling, rel_extent):
    lse_kwargs = _lse_kwargs()
    flex = _flex_function(query.device)
    output = flex(
        query,
        key,
        value,
        score_mod = _relative_bias_score_mod(rel_logits, rel_extent),
        block_mask = block_mask,
        scale = scaling,
        enable_gqa = query.shape[1] != key.shape[1],
        kernel_options = _kernel_options(query.device, query.dtype, query.shape[-1]),
        **lse_kwargs,
    )
    out, lse = _split_lse(output, lse_kwargs)
    # Rows with nothing visible: flex gives out = 0, lse = -inf, so the weight is 0, as eager's softmax gives there.
    acc = torch.promote_types(query.dtype, torch.float32)
    weight = torch.sigmoid(lse.to(acc) - sinks.transpose(1, 2).to(acc))
    out = (out.to(acc) * weight.unsqueeze(-1)).to(query.dtype)
    return out.transpose(1, 2)


class _RecomputedFlex(torch.autograd.Function):
    """flex_attention saved under torch.utils.checkpoint(use_reentrant = False) (transformers' default) recomputes twice
    and accumulates the score_mod buffer's gradient twice (the relative bias grads come out doubled or NaN). Here the
    forward keeps no flex graph; the backward recomputes it outside the checkpoint's saved-tensor hooks."""

    @staticmethod
    def forward(ctx, query, key, value, rel_logits, sinks, block_mask, scaling, rel_extent):
        ctx.save_for_backward(query, key, value, rel_logits, sinks)
        ctx.block_mask, ctx.scaling, ctx.rel_extent = block_mask, scaling, rel_extent
        with torch.no_grad():
            return _flex_core(query, key, value, block_mask, rel_logits, sinks, scaling, rel_extent)

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        inputs = [t.detach().requires_grad_(t.requires_grad) for t in saved]
        with torch.enable_grad():
            out = _flex_core(
                *inputs[:3], ctx.block_mask, inputs[3], inputs[4], ctx.scaling, ctx.rel_extent
            )
        wanted = [t for t in inputs if t.requires_grad]
        grads = (
            iter(torch.autograd.grad(out, wanted, grad_output, allow_unused = True))
            if wanted
            else iter(())
        )
        return (*[next(grads) if t.requires_grad else None for t in inputs], None, None, None)


def _saved_tensor_hooks_active():
    try:
        return torch._C._autograd._top_saved_tensors_default_hooks(False) is not None
    except Exception:
        return True


def inkling2_flex_attention(query, key, value, block_mask, rel_logits, sinks, scaling, rel_extent):
    """Eager Inkling 2 attention (relative bias, per-token sink) through flex_attention.

    query [B, H, L, D], key / value [B, H_kv, L, D], rel_logits [B, L, H, rel_extent] (``rel_states @ proj``),
    sinks [B, L, H] (``sink_proj(hidden_states)``). Returns [B, L, H, D] like eager_attention_forward."""
    needs_grad = torch.is_grad_enabled() and any(
        t.requires_grad for t in (query, key, value, rel_logits, sinks)
    )
    if needs_grad and _saved_tensor_hooks_active():
        _count("recomputed")
        return _RecomputedFlex.apply(
            query, key, value, rel_logits, sinks, block_mask, scaling, rel_extent
        )
    return _flex_core(query, key, value, block_mask, rel_logits, sinks, scaling, rel_extent)


# ---------------------------------------------------------------------------------------------------------------------
# When flex runs


def _capability(device):
    try:
        return torch.cuda.get_device_capability(device)
    except Exception:
        return None


def _auto_min_length(device):
    """Shortest sequence that takes flex under ``auto``; None keeps eager on that card.

    Inkling-2-Mini QLoRA, one training step, eager vs flex in one process (A/A repeats, PR description):
    - L4 (sm89, bf16), 1 x 1024 tokens: eager 5.83-5.92 s, flex 5.00-5.03 s, same peak memory. Ampere and newer take
      flex at every length.
    - 2x T4 (sm75, fp16), 1 x 1024 tokens: eager 9.21-9.54 s, flex 15.79-15.95 s (Triton emits no mma on Turing), peak
      0.16-0.20 GB lower per card. 4 x 1024 and 1 x 2048..4096 run out of memory in both (the fp32 logits and the
      expert scratch, not attention), so flex buys nothing there and T4 stays eager unless forced.
    UNSLOTH_INKLING2_FLEX_MIN_LENGTH=N overrides the threshold on every card."""
    override = os.environ.get(MIN_LEN_ENV)
    if override is not None:
        try:
            return int(override)
        except ValueError:
            pass
    capability = _capability(device)
    if capability is None:
        return None
    if tuple(capability) >= (8, 0):
        return 0
    return None


def _wants_flex(q_length, kv_length, q_offset, kv_offset, device):
    if not (isinstance(q_offset, int) and isinstance(kv_offset, int)):
        return False
    # Prefill and training only: the queries are the whole key range, so q - kv is the index difference.
    if q_offset != 0 or kv_offset != 0 or q_length != kv_length or q_length < 2:
        return False
    device = torch.device(device)
    if device.type != "cuda":
        return _ALLOW_CPU
    mode = _mode()
    if mode == "0":
        return False
    if mode == "1":
        return True
    min_length = _auto_min_length(device)
    return min_length is not None and q_length >= min_length


# Tests flip this to run flex's reference math on CPU.
_ALLOW_CPU = False


def _build_block_mask(mask_function, batch_size, q_length, kv_length, attention_mask, device):
    from transformers.masking_utils import and_masks, padding_mask_function, prepare_padding_mask

    create_block_mask = _flex_symbols()[1]
    if attention_mask is not None and not bool(attention_mask.all()):
        padding_mask = prepare_padding_mask(attention_mask, kv_length, 0)
        mask_function = and_masks(mask_function, padding_mask_function(padding_mask))
    device = torch.device(device)
    block = 128
    padded = -(-q_length // block) * block * (-(-kv_length // block) * block) * batch_size
    # The compiled builder keeps memory O(blocks); torch < 2.13 miscompiles it for one-block lengths (zoo #2509).
    compile_builder = device.type == "cuda" and (
        _torch_version() >= (2, 13)
        or (q_length > block and kv_length > block)
        or padded > (1 << 27)
    )
    return create_block_mask(
        mask_function,
        B = batch_size,
        H = None,
        Q_LEN = q_length,
        KV_LEN = kv_length,
        device = device,
        _compile = compile_builder,
    )


# Unsloth compiles transformers' mask builders (dynamic, not fullgraph); building the BlockMask inside that trace graph
# breaks on the padding check and hands create_block_mask to dynamo, so this builder always runs eagerly.
@torch.compiler.disable
def _mask_interface(
    batch_size,
    q_length,
    kv_length,
    q_offset = 0,
    kv_offset = 0,
    mask_function = None,
    attention_mask = None,
    device = "cpu",
    **kwargs,
):
    """Mask builder registered under ATTN_KEY: a BlockMask for prefill / training forwards, else eager's mask."""
    from transformers.masking_utils import causal_mask_function, eager_mask

    if mask_function is None:
        mask_function = causal_mask_function
    _STATS["mask_calls"] = _STATS.get("mask_calls", 0) + 1
    state = _FORWARD_STATE.get()
    kind = "sliding" if kwargs.get("local_size") is not None else "full"
    if _wants_flex(q_length, kv_length, q_offset, kv_offset, device):
        try:
            block_mask = _build_block_mask(
                mask_function, batch_size, q_length, kv_length, attention_mask, device
            )
        except Exception as error:
            _FAILED.setdefault("block_mask", repr(error))
        else:
            _count("block_masks")
            if state is not None:
                state[kind] = True
            return block_mask
    _count("dense_masks")
    if state is not None:
        state[kind] = False
    return eager_mask(
        batch_size = batch_size,
        q_length = q_length,
        kv_length = kv_length,
        q_offset = q_offset,
        kv_offset = kv_offset,
        mask_function = mask_function,
        attention_mask = attention_mask,
        device = device,
        **kwargs,
    )


def _attention_interface(module, query, key, value, attention_mask, **kwargs):
    """ATTN_KEY in the AttentionInterface: the language attention layers call their patched forward instead, so only
    the block / message attention residuals (learned queries, no mask) and unpatched callers land here: eager."""
    eager = getattr(sys.modules.get(type(module).__module__), "eager_attention_forward", None)
    if eager is None:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward as eager
    return eager(module, query, key, value, attention_mask, **kwargs)


# ---------------------------------------------------------------------------------------------------------------------
# Patched modeling code


def _move_closure(obj, device, memo):
    """Copy of a mask function whose captured tensors (padding mask, packed-sequence ids, position ids) live on
    ``device``: the BlockMask is built once on the embedding device, and layers placed on another card evaluate its
    mask_mod on theirs."""
    if id(obj) in memo:
        return memo[id(obj)]
    if isinstance(obj, torch.Tensor):
        out = obj.to(device)
    elif isinstance(obj, (tuple, list)):
        out = type(obj)(_move_closure(x, device, memo) for x in obj)
    elif isinstance(obj, types.FunctionType) and obj.__closure__:
        cells = tuple(
            types.CellType(_move_closure(c.cell_contents, device, memo)) for c in obj.__closure__
        )
        out = types.FunctionType(
            obj.__code__, obj.__globals__, obj.__name__, obj.__defaults__, cells
        )
        out.__kwdefaults__ = obj.__kwdefaults__
    else:
        out = obj
    memo[id(obj)] = out
    return out


def _closure_off_device(obj, device, seen):
    if id(obj) in seen:
        return False
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        return obj.device != device
    if isinstance(obj, (tuple, list)):
        return any(_closure_off_device(x, device, seen) for x in obj)
    if isinstance(obj, types.FunctionType) and obj.__closure__:
        return any(_closure_off_device(c.cell_contents, device, seen) for c in obj.__closure__)
    return False


def _block_mask_on(block_mask, device):
    """The BlockMask with its block tables and its mask_mod's captured tensors on ``device``. accelerate's device-map
    hook already moves a BlockMask argument with ``.to``, which leaves the mask_mod closure on the first card."""
    if block_mask.kv_num_blocks.device != device:
        block_mask = block_mask.to(device)
    mask_mod = block_mask.mask_mod
    if not _closure_off_device(mask_mod, device, set()):
        return block_mask
    cache = mask_mod.__dict__.setdefault("_unsloth_on_device", {})
    if device not in cache:
        cache[device] = _move_closure(mask_mod, device, {})
    moved = copy.copy(block_mask)
    moved.mask_mod = cache[device]
    return moved


def _flex_forward(
    self,
    hidden_states,
    key_hidden_states,
    value_hidden_states,
    attention_mask,
    conv_mask,
    cache,
    cache_offset,
    kwargs,
):
    """Mirror of Inkling2{,Loop}{Sliding,Full}Attention.forward (transformers 5.24) with the attention replaced."""
    cache_layer_idx = self.layer_idx + cache_offset
    input_shape = hidden_states.shape[:-1]
    position_ids = kwargs["position_ids"]

    query_states = self.q_proj(hidden_states, position_ids)
    key_states = self.k_proj(
        key_hidden_states, cache = cache, conv_mask = conv_mask, cache_offset = cache_offset, **kwargs
    )
    value_states = self.v_proj(
        value_hidden_states, cache = cache, conv_mask = conv_mask, cache_offset = cache_offset, **kwargs
    )
    if cache is not None:
        key_states, value_states = cache.update(
            key_states, value_states, cache_layer_idx, kwargs = kwargs
        )

    if key_states.dtype != query_states.dtype:
        key_states = key_states.to(query_states.dtype)
    if value_states.dtype != query_states.dtype:
        value_states = value_states.to(query_states.dtype)

    bias = self.relative_position_bias
    relative_states = bias.linear(hidden_states).view(*hidden_states.shape[:-1], bias.num_heads, -1)
    relative_states = bias.norm(relative_states, position_ids)
    rel_logits = relative_states @ bias.proj  # [B, L, H, rel_extent], never gathered to L x L
    sinks = self.sink_proj(hidden_states)  # [B, L, H]

    attention_mask = _block_mask_on(attention_mask, query_states.device)
    attn_output = inkling2_flex_attention(
        query_states,
        key_states,
        value_states,
        attention_mask,
        rel_logits,
        sinks,
        self.scaling,
        bias.rel_extent,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output, hidden_states)
    return attn_output, None


def _wrap_attention_forward(original):
    signature = inspect.signature(original)

    @functools.wraps(original)
    def forward(self, *args, **kwargs):
        if "past_key_values" in kwargs and "cache" not in kwargs:
            kwargs["cache"] = kwargs.pop("past_key_values")
        try:
            bound = signature.bind(self, *args, **kwargs)
        except TypeError:
            return original(self, *args, **kwargs)
        arguments = bound.arguments
        attention_mask = arguments.get("attention_mask")
        if not _is_block_mask(attention_mask):
            _count("eager")
            return original(self, *args, **kwargs)
        if self.training and (getattr(self, "attention_dropout", 0) or 0) != 0:
            raise RuntimeError(
                f"Unsloth: Inkling 2 flex attention does not apply attention dropout; set {ENV}=0 to train with it."
            )
        extra = dict(arguments.get("kwargs", {}))
        hidden_states = arguments["hidden_states"]
        key_hidden_states = arguments.get("key_hidden_states")
        value_hidden_states = arguments.get("value_hidden_states")
        _count("flex")
        return _flex_forward(
            self,
            hidden_states,
            hidden_states if key_hidden_states is None else key_hidden_states,
            hidden_states if value_hidden_states is None else value_hidden_states,
            attention_mask,
            arguments.get("conv_mask"),
            arguments.get("cache"),
            arguments.get("cache_offset", 0) or 0,
            extra,
        )

    forward._unsloth_inkling2_flex_original = original
    return forward


def _wrap_text_forward(original):
    @functools.wraps(original)
    def forward(self, *args, **kwargs):
        if getattr(self.config, "_attn_implementation", None) != ATTN_KEY:
            return original(self, *args, **kwargs)
        token = _FORWARD_STATE.set({})
        try:
            return original(self, *args, **kwargs)
        finally:
            _FORWARD_STATE.reset(token)

    forward._unsloth_inkling2_flex_original = original
    return forward


def _wrap_relative_distance(original):
    @functools.wraps(original)
    def get_relative_distance(inputs_embeds, cache, is_sliding):
        state = _FORWARD_STATE.get()
        if state is not None and state.get("sliding" if is_sliding else "full"):
            # Flex layers compute q - kv themselves; skip the L x L int64 matrix (512 MB at 8192 tokens). Empty, so
            # any eager use fails loudly instead of broadcasting.
            _count("distance_skipped")
            return torch.empty((0, 0), dtype = torch.long, device = inputs_embeds.device)
        return original(inputs_embeds, cache, is_sliding)

    get_relative_distance._unsloth_inkling2_flex_original = original
    return get_relative_distance


def _source(obj):
    out = []
    for candidate in (obj, inspect.unwrap(obj)):
        try:
            out.append(inspect.getsource(candidate))
        except Exception:
            pass
    return "\n".join(out)


def _effective_functions(cls, attr = "forward"):
    """The function a class attribute runs: Unsloth's compiler leaves ``forward`` as ``return Cls_forward(self, ...)``
    with the body in its compiled module, so follow that one hop."""
    function = getattr(cls, attr, None)
    function = getattr(function, "_unsloth_inkling2_flex_original", function)
    if function is None:
        return []
    out = [function]
    namespace = getattr(inspect.unwrap(function), "__globals__", {})
    delegate = namespace.get(f"{cls.__name__}_{attr}")
    if callable(delegate) and delegate is not function:
        out.append(delegate)
    return out


def _effective_source(cls, attr = "forward"):
    return "\n".join(_source(f) for f in _effective_functions(cls, attr))


def _namespaces(cls, attr = "forward"):
    return [
        getattr(inspect.unwrap(f), "__globals__", None) for f in _effective_functions(cls, attr)
    ]


def _has_all(source, markers):
    return all(marker in source for marker in markers)


def _check_attention_class(cls):
    if not _has_all(_effective_source(cls), _ATTENTION_SOURCE_MARKERS):
        return f"{cls.__name__}.forward changed"
    for namespace in _namespaces(cls):
        eager = (namespace or {}).get("eager_attention_forward")
        if eager is not None and not _has_all(_source(eager), _EAGER_SOURCE_MARKERS):
            return f"eager attention of {cls.__name__} changed"
    return None


def _check_text_class(cls):
    found = False
    for namespace in _namespaces(cls):
        distance = (namespace or {}).get("get_relative_distance")
        if distance is None:
            continue
        found = True
        distance = getattr(distance, "_unsloth_inkling2_flex_original", distance)
        if not _has_all(_source(distance), _RELATIVE_DISTANCE_MARKERS):
            return f"get_relative_distance of {cls.__name__} changed"
    return None if found else f"{cls.__name__} has no get_relative_distance"


def _patch_classes(model):
    """Wrap the attention, text model and relative distance code the loaded model really runs. Returns a reason when
    any of it is not the transformers 5.24 code this module mirrors (nothing is patched then)."""
    attention, text, bias = set(), set(), set()
    for module in model.modules():
        cls = type(module)
        name = cls.__name__
        if not name.startswith("Inkling2"):
            continue
        if (
            name.endswith(_ATTENTION_CLASS_SUFFIXES)
            and hasattr(module, "relative_position_bias")
            and hasattr(module, "sink_proj")
        ):
            attention.add(cls)
            bias.add(type(module.relative_position_bias))
        elif name.endswith("TextModel"):
            text.add(cls)
    if not attention or not text:
        return "no Inkling 2 language attention found"
    for cls in sorted(attention | text | bias, key = lambda c: c.__qualname__):
        key = f"{cls.__module__}.{cls.__qualname__}"
        if cls in attention:
            reason = _check_attention_class(cls)
        elif cls in text:
            reason = _check_text_class(cls)
        else:
            reason = (
                None
                if _has_all(_effective_source(cls), _BIAS_SOURCE_MARKERS)
                else f"{cls.__name__} changed"
            )
        if reason is not None:
            _FAILED[key] = reason
            return reason
    for cls in attention:
        if not hasattr(cls.forward, "_unsloth_inkling2_flex_original"):
            cls.forward = _wrap_attention_forward(cls.forward)
        _PATCHED[f"{cls.__module__}.{cls.__qualname__}"] = True
    for cls in text:
        if not hasattr(cls.forward, "_unsloth_inkling2_flex_original"):
            cls.forward = _wrap_text_forward(cls.forward)
        for namespace in _namespaces(cls):
            distance = (namespace or {}).get("get_relative_distance")
            if distance is not None and not hasattr(distance, "_unsloth_inkling2_flex_original"):
                namespace["get_relative_distance"] = _wrap_relative_distance(distance)
        _PATCHED[f"{cls.__module__}.{cls.__qualname__}"] = True
    for cls in bias:
        _PATCHED[f"{cls.__module__}.{cls.__qualname__}"] = True
    _rebind_hooked_instances(model, attention | text)
    return None


def _rebind_hooked_instances(model, classes):
    """accelerate's device-map hooks (a model split over GPUs) call the forward bound at load, ``_old_forward``, not the
    class attribute: point those at the wrapped forward, or the split model would keep running eager attention on
    flex masks."""
    rebound = 0
    for module in model.modules():
        cls = type(module)
        if cls not in classes:
            continue
        old = module.__dict__.get("_old_forward")
        original = getattr(cls.forward, "_unsloth_inkling2_flex_original", None)
        if old is not None and original is not None and getattr(old, "__func__", None) is original:
            module._old_forward = types.MethodType(cls.forward, module)
            rebound += 1
    _STATS["rebound_hooks"] = _STATS.get("rebound_hooks", 0) + rebound
    return rebound


def _register():
    from transformers.masking_utils import AttentionMaskInterface
    from transformers.modeling_utils import AttentionInterface

    AttentionInterface.register(ATTN_KEY, _attention_interface)
    AttentionMaskInterface.register(ATTN_KEY, _mask_interface)


def _text_configs(model):
    config = getattr(model, "config", None)
    seen, out = set(), []
    for cfg in (getattr(config, "text_config", None),):
        if cfg is not None and id(cfg) not in seen:
            seen.add(id(cfg))
            out.append(cfg)
    if config is not None and not out and str(getattr(config, "model_type", "")).endswith("_text"):
        out.append(config)
    return out


def _cuda_devices(model):
    devices = set()
    try:
        for parameter in model.parameters():
            if parameter.device.type == "cuda":
                devices.add(parameter.device)
    except Exception:
        pass
    return devices


def enable_inkling2_flex(model, verbose = True):
    """Route a loaded Inkling 2 model's language attention through flex. Returns True when enabled."""
    if _mode() == "0" or not available():
        return False
    model_types = {
        str(getattr(cfg, "model_type", "")).lower()
        for cfg in (
            getattr(model, "config", None),
            getattr(getattr(model, "config", None), "text_config", None),
        )
    }
    if not any(t.startswith("inkling2") for t in model_types):
        return False
    configs = _text_configs(model)
    if not configs:
        return False
    for cfg in configs:
        if (getattr(cfg, "attention_dropout", 0) or 0) != 0:
            _FAILED["dropout"] = "attention_dropout != 0"
            return False
        if getattr(cfg, "_attn_implementation", None) not in ("eager", None, ATTN_KEY):
            # Someone asked for another implementation explicitly.
            return False
    if not _ALLOW_CPU:
        devices = _cuda_devices(model)
        if not devices:
            return False
        if _mode() == "auto" and all(_auto_min_length(d) is None for d in devices):
            return False
    if _patch_classes(model) is not None:
        if verbose:
            print(f"Unsloth: Inkling 2 keeps eager attention ({_FAILED}).")
        return False
    _register()
    for cfg in configs:
        cfg._attn_implementation = ATTN_KEY
    if verbose:
        print(
            "Unsloth: Inkling 2 language attention uses FlexAttention (relative bias + sinks fused)."
        )
    return True


def disable_inkling2_flex(model):
    for cfg in _text_configs(model):
        if getattr(cfg, "_attn_implementation", None) == ATTN_KEY:
            cfg._attn_implementation = "eager"
