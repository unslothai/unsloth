# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Utilities for enabling packed (padding-free) batches across Unsloth."""

from __future__ import annotations

import copy
import inspect
import logging
import os
import sys
from collections import OrderedDict
from functools import wraps
from typing import Any, Iterable, Optional, Sequence, Tuple

import torch

try:
    from xformers.ops.fmha.attn_bias import (
        BlockDiagonalCausalMask as _XFormersBlockMask,
    )
except Exception:
    try:
        from xformers.attn_bias import BlockDiagonalCausalMask as _XFormersBlockMask
    except Exception:
        _XFormersBlockMask = None

try:
    from xformers.ops.fmha.attn_bias import BlockDiagonalMask as _XFormersBidirectionalMask
except Exception:
    try:
        from xformers.attn_bias import BlockDiagonalMask as _XFormersBidirectionalMask
    except Exception:
        _XFormersBidirectionalMask = None

_XFORMERS_MASK_CACHE_MAXSIZE = 32
_XFORMERS_MASK_CACHE: OrderedDict[Tuple[torch.device, Tuple[int, ...], int, bool], Any] = (
    OrderedDict()
)

# Cache per device for get_packed_info_from_kwargs to avoid repeated D2H sync across layers
_PACKED_INFO_CACHE: dict = {}

# Cache per device for build_sdpa_packed_attention_mask to avoid repeated D2H sync across layers
_SDPA_MASK_CACHE: dict = {}

_SEGMENT_LENGTHS_CACHE: dict = {}

# Cache per device for build_xformers_block_causal_mask to avoid repeated D2H sync across layers
_XFORMERS_BLOCK_MASK_CACHE: dict = {}

# Cache per device for cover_padded_cu_seqlens to avoid repeated D2H sync across layers
_PADDED_CU_SEQLENS_CACHE: dict = {}


def _window_cache_key(sliding_window: Optional[int]) -> int:
    if sliding_window is None or sliding_window <= 0:
        return 0
    return int(sliding_window)


def move_xformers_attention_bias(attn_bias: Any, device: torch.device):
    """Return an xFormers attention bias whose tensor metadata is on ``device``."""
    if attn_bias is None:
        return None

    device = torch.device(device)
    seqinfos = [
        (name, seqinfo)
        for name in ("q_seqinfo", "k_seqinfo")
        if (seqinfo := getattr(attn_bias, name, None)) is not None
    ]
    if seqinfos:
        if all(
            getattr(getattr(seqinfo, "seqstart", None), "device", None) == device
            for _, seqinfo in seqinfos
        ):
            return attn_bias

        # Move the device-bearing metadata instead of the top-level mask: older xFormers versions demote
        # causal masks in their inherited `to` method, and copies also keep later model shards from
        # rewriting masks retained for backward.
        moved_bias = copy.copy(attn_bias)
        moved_seqinfos = {}
        for name, seqinfo in seqinfos:
            source_id = id(seqinfo)
            if source_id not in moved_seqinfos:
                moved_seqinfo = copy.copy(seqinfo)
                move = getattr(moved_seqinfo, "to", None)
                if callable(move):
                    moved = move(device)
                    if moved is not None:
                        moved_seqinfo = moved
                moved_seqinfos[source_id] = moved_seqinfo
            setattr(moved_bias, name, moved_seqinfos[source_id])
        return moved_bias

    # Biases without sequence metadata can safely use their own move protocol.
    moved_bias = copy.copy(attn_bias)
    move = getattr(moved_bias, "to", None)
    if callable(move):
        moved = move(device)
        if moved is not None:
            moved_bias = moved
    return moved_bias


def _get_cached_block_mask(
    lengths: Tuple[int, ...],
    sliding_window: Optional[int],
    device: torch.device,
    is_causal: bool = True,
):
    mask_class = _XFormersBlockMask if is_causal else _XFormersBidirectionalMask
    if mask_class is None:
        return None

    device = torch.device(device)
    window_key = _window_cache_key(sliding_window)
    cache_key = (device, lengths, window_key, is_causal)
    cached = _XFORMERS_MASK_CACHE.get(cache_key)
    if cached is not None:
        _XFORMERS_MASK_CACHE.move_to_end(cache_key)
        return cached

    mask = mask_class.from_seqlens(list(lengths))
    if window_key and mask is not None and hasattr(mask, "make_local_attention"):
        mask = mask.make_local_attention(window_size = window_key)
    mask = move_xformers_attention_bias(mask, device)

    _XFORMERS_MASK_CACHE[cache_key] = mask
    if len(_XFORMERS_MASK_CACHE) > _XFORMERS_MASK_CACHE_MAXSIZE:
        _XFORMERS_MASK_CACHE.popitem(last = False)
    return mask


class _TrlPackingWarningFilter(logging.Filter):
    to_filter = (
        "attention implementation is not",
        "kernels-community",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(substring in message for substring in self.to_filter)


_TRL_FILTER_INSTALLED = False


def _ensure_trl_warning_filter():
    global _TRL_FILTER_INSTALLED
    if _TRL_FILTER_INSTALLED:
        return
    logging.getLogger("trl.trainer.sft_trainer").addFilter(_TrlPackingWarningFilter())
    _TRL_FILTER_INSTALLED = True


def mark_allow_overlength(module):
    """Mark a module hierarchy so padding-free batches can exceed max_seq_length."""
    if module is None:
        return
    if hasattr(module, "max_seq_length"):
        setattr(module, "_unsloth_allow_packed_overlength", True)
    children = getattr(module, "children", None)
    if children is None:
        return
    for child in children():
        mark_allow_overlength(child)


def configure_sample_packing(config):
    """Mutate an ``SFTConfig`` so TRL prepares packed batches."""
    _ensure_trl_warning_filter()
    setattr(config, "packing", True)
    setattr(config, "padding_free", True)
    setattr(config, "remove_unused_columns", False)


def configure_padding_free(config):
    """Mutate an ``SFTConfig`` so TRL enables padding-free batching without packing."""
    _ensure_trl_warning_filter()
    setattr(config, "padding_free", True)
    setattr(config, "remove_unused_columns", False)


def enable_sample_packing(
    model,
    trainer,
    *,
    sequence_lengths_key: str = "seq_lengths",
) -> None:
    """Enable runtime support for packed batches on an existing trainer."""
    if model is None or trainer is None:
        raise ValueError("model and trainer must not be None")

    mark_allow_overlength(model)

    if hasattr(trainer, "args") and hasattr(trainer.args, "remove_unused_columns"):
        trainer.args.remove_unused_columns = False

    collator = getattr(trainer, "data_collator", None)
    if collator is None or not hasattr(collator, "torch_call"):
        return
    if getattr(collator, "_unsloth_packing_wrapped", False):
        return

    if hasattr(collator, "padding_free"):
        collator.padding_free = True
    if hasattr(collator, "return_position_ids"):
        collator.return_position_ids = True

    original_torch_call = collator.torch_call

    def torch_call_with_lengths(examples: Sequence[dict]):
        batch = original_torch_call(examples)
        if examples and isinstance(examples[0], dict):
            seq_lengths: list[int] = []
            for example in examples:
                lengths = example.get(sequence_lengths_key)
                if isinstance(lengths, Iterable):
                    seq_lengths.extend(int(length) for length in lengths)
            # Fallback: infer lengths from tokenized inputs when metadata is absent.
            if not seq_lengths:
                for example in examples:
                    ids = example.get("input_ids")
                    if isinstance(ids, Iterable):
                        seq_lengths.append(len(ids))
            if seq_lengths:
                # Boundary labels are NOT masked here: unsloth_zoo's _unsloth_get_batch_samples counts
                # num_items_in_batch off this batch and discounts the N-1 boundary targets itself, idempotently,
                # zeroing those slots rather than subtracting a constant, so the count is unaffected by upstream
                # masking (TRL >= 0.24's labels[position_ids == 0] = -100, completion-only masking,
                # assistant_masks). Labels are left alone because the guard that needs these positions runs in the
                # forward, off packed_seq_lengths.
                batch["packed_seq_lengths"] = torch.tensor(seq_lengths, dtype = torch.int32)
                if "attention_mask" in batch:
                    batch.pop("attention_mask")
        return batch

    collator.torch_call = torch_call_with_lengths
    collator._unsloth_packing_wrapped = True


def enable_padding_free_metadata(model, trainer):
    """Inject seq-length metadata when padding-free batching is enabled without packing."""
    collator = getattr(trainer, "data_collator", None)
    if (
        collator is None
        or getattr(collator, "_unsloth_padding_free_lengths_wrapped", False)
        or not getattr(collator, "padding_free", False)
    ):
        return

    mark_allow_overlength(model)
    if hasattr(collator, "return_position_ids"):
        collator.return_position_ids = True
    if hasattr(trainer, "args") and hasattr(trainer.args, "remove_unused_columns"):
        trainer.args.remove_unused_columns = False

    original_torch_call = collator.torch_call

    def torch_call_with_padding_free_metadata(examples: Sequence[dict]):
        seq_lengths: list[int] = []
        collated = examples
        if examples and isinstance(examples[0], dict):
            for index, example in enumerate(examples):
                lengths = example.get("seq_lengths")
                if lengths is None:
                    ids = example.get("input_ids")
                    if ids is None:
                        continue
                    lengths = [len(ids)]
                    # TRL's collator keys seq_lengths off examples[0] and reads every row: pass a copy, not the caller's row.
                    if collated is examples:
                        collated = list(examples)
                    collated[index] = {**example, "seq_lengths": lengths}
                seq_lengths.extend(lengths)

        batch = original_torch_call(collated)
        if seq_lengths:
            # Labels left alone for the same reason as enable_sample_packing: num_items_in_batch is counted off
            # this batch and the zoo's discount of the boundary targets is idempotent.
            batch["packed_seq_lengths"] = torch.tensor(
                seq_lengths,
                dtype = torch.int32,
            )
        return batch

    collator.torch_call = torch_call_with_padding_free_metadata
    collator._unsloth_padding_free_lengths_wrapped = True


# Experimental packing for hybrid linear-attention (Qwen3.5 / Qwen3-Next gated-delta, Nemotron-H Mamba2):
# a flattened batch leaks recurrent + conv state across sequences unless seq_idx / cu_seqlens reach the
# accelerated kernels, so this is env-gated and fails closed on the pure-torch fallbacks.
_MAMBA2_FUSED_NAMES = (
    "mamba2_split_conv1d_scan_combined",
    "mamba_split_conv1d_scan_combined",
)
_HYBRID_PACKING_ENV_VAR = "UNSLOTH_EXPERIMENTAL_HYBRID_PACKING"
_HYBRID_LOGGER = logging.getLogger("unsloth.hybrid_packing")
_HYBRID_WARNED: set = set()


def _hybrid_packing_enabled() -> bool:
    # Read at call time so setting the flag after `import unsloth` still takes effect.
    return os.environ.get(_HYBRID_PACKING_ENV_VAR, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _hybrid_reject(reason: str) -> bool:
    # One deduped diagnostic explaining why hybrid packing stayed on the padded path.
    if reason not in _HYBRID_WARNED:
        _HYBRID_WARNED.add(reason)
        _HYBRID_LOGGER.warning(
            "Unsloth: hybrid linear-attention packing disabled (padded path): %s.",
            reason,
        )
    return False


def _names_param(
    module,
    param,
    methods = ("forward",),
) -> bool:
    # Decorated forwards can hide their signature; their kernel methods still name the parameter.
    for method in methods:
        fn = getattr(type(module), method, None)
        try:
            if fn is not None and param in inspect.signature(fn).parameters:
                return True
        except (TypeError, ValueError):
            continue
    return False


_QKV_CONVS = ("q_conv1d", "k_conv1d", "v_conv1d")


def _stateful_mixer_kind(module) -> Optional[str]:
    """How a token mixer carries state across positions: "gated_delta" / "ssd" (Mamba2-style
    chunked scan) / "short_conv" take boundary metadata, "unsupported" leaks with no kernel
    fix (Mamba1 selective scan, lightning attention, RWKV), None is stateless."""
    name = type(module).__name__
    if name.endswith("GatedDeltaNet"):
        if hasattr(module, "conv1d"):
            return "gated_delta"
        convs = [getattr(module, c, None) for c in _QKV_CONVS]
        if all(c is not None and _names_param(c, "cu_seqlens") for c in convs):
            return "gated_delta_qkv"
        return "unsupported"
    if hasattr(module, "A_log"):
        # Mamba2 / SSD mixers learn dt_bias; Mamba1 projects dt through dt_proj.
        if hasattr(module, "conv1d") and hasattr(module, "dt_bias"):
            return "ssd"
        return "unsupported"
    if "ShortConv" in name:
        methods = ("forward", "cuda_kernels_forward", "slow_forward")
        return "short_conv" if _names_param(module, "seq_idx", methods) else "unsupported"
    if "LightningAttention" in name or (name.startswith("Rwkv") and hasattr(module, "time_decay")):
        return "unsupported"
    return None


def _iter_stateful_mixers(model):
    """(module, kind) for every stateful mixer, outermost first; submodules of a gated-delta
    mixer (its own short convs) are owned by that mixer's shim."""
    found, owned = [], set()
    for module in model.modules():
        if id(module) in owned:
            continue
        kind = _stateful_mixer_kind(module)
        if kind is None:
            continue
        found.append((module, kind))
        owned.update(id(m) for m in module.modules())
    return found


def _iter_gated_delta_modules(model):
    return [
        m for m, kind in _iter_stateful_mixers(model) if kind in ("gated_delta", "gated_delta_qkv")
    ]


def _iter_mamba2_modules(model):
    return [m for m, kind in _iter_stateful_mixers(model) if kind == "ssd"]


def _iter_short_conv_modules(model):
    return [m for m, kind in _iter_stateful_mixers(model) if kind == "short_conv"]


def _callable_accepts_named_seq_idx(fn) -> Optional[str]:
    # A **kwargs-only hub stub does not count: it may drop seq_idx.
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return "kernel signature not introspectable"
    if "seq_idx" in params:
        return None
    return "mamba2 fused kernel does not accept seq_idx"


_MAMBA2_NAMESPACE_SUBSTR = ("unsloth_compiled", "mamba_ssm")


_MAMBA2_KERNEL_GLOBALS = (
    "is_fast_path_available",
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "selective_state_update",
    "mamba_chunk_scan_combined",
)


def _force_install_mamba2_fused(
    namespace,
    wrapped,
    source = None,
) -> None:
    # unsloth_compiled_cache imported the kernel globals before the mixer's __init__ loaded them, so sync
    # them from source; forcing is_fast_path_available on would call a missing decode kernel.
    if wrapped is None or not isinstance(namespace, dict):
        return
    for name in _MAMBA2_FUSED_NAMES:
        namespace[name] = wrapped
    if (
        isinstance(source, dict)
        and namespace is not source
        and "unsloth_compiled" in str(namespace.get("__name__", ""))
    ):
        for name in _MAMBA2_KERNEL_GLOBALS:
            if name in namespace and name in source:
                namespace[name] = source[name]
    for value in list(namespace.values()):
        inner = getattr(value, "__dict__", None)
        if not isinstance(inner, dict):
            continue
        for name in _MAMBA2_FUSED_NAMES:
            if name in inner:
                inner[name] = wrapped


def _iter_mamba2_install_namespaces(mamba2_modules):
    seen: set[int] = set()

    def take(ns):
        if isinstance(ns, dict) and id(ns) not in seen:
            seen.add(id(ns))
            return True
        return False

    for module in mamba2_modules:
        modeling = inspect.getmodule(type(module))
        ns = getattr(modeling, "__dict__", None)
        if take(ns):
            yield ns
    for name, mod in list(sys.modules.items()):
        lname = (name or "").lower()
        if not any(s in lname for s in _MAMBA2_NAMESPACE_SUBSTR):
            continue
        ns = getattr(mod, "__dict__", None)
        if take(ns):
            yield ns


def _mamba2_handshake_debug(modules) -> str:
    if not modules:
        return ""
    module = modules[0]
    cls = type(module)
    modeling = inspect.getmodule(cls)
    globs = getattr(modeling, "__dict__", {}) if modeling is not None else {}
    return (
        f"Mixer {cls.__name__} from {getattr(cls, '__module__', None)}: "
        f"is_fast_path_available={globs.get('is_fast_path_available', '<missing>')} "
        f"fused_hit={getattr(module, '_unsloth_varlen_fused_hit', None)} "
        f"conv_hit={getattr(module, '_unsloth_varlen_conv_hit', None)} "
        f"scan_hit={getattr(module, '_unsloth_varlen_scan_hit', None)}"
    )


def _call_as_packed_mamba2_prefill(fn, args, kwargs):
    # transformers <= 5.15 takes the fused kernel only with an all-ones mask and no cache.
    # Only binding sits in the try: a TypeError from inside fn must not re-run fn.
    try:
        sig = inspect.signature(fn)
        bound = sig.bind_partial(*args, **kwargs)
    except (TypeError, ValueError):
        return fn(*args, **{**kwargs, "attention_mask": None, "cache_params": None})
    for name in ("attention_mask", "cache_params"):
        if name in sig.parameters:
            bound.arguments[name] = None
    call_kwargs = bound.kwargs
    for name in ("attention_mask", "cache_params"):
        if name in call_kwargs:
            call_kwargs[name] = None
    return fn(*bound.args, **call_kwargs)


_MAMBA2_MASK_CLEAR_NAMES = ("cuda_kernels_forward",)


def _is_mamba2_mask_clear_name(name: str, classes = ()) -> bool:
    # Unsloth's compiler names hoisted methods "<Class>_<method>".
    if name in _MAMBA2_MASK_CLEAR_NAMES:
        return True
    return any(name in (f"{c}_cuda_kernels_forward", f"{c}_forward") for c in classes)


# Shared kernel wrappers live on module globals and outlive any one model, so they read the running
# packed batch and SSD mixer from here instead of closing over a model.
_PACKED: list = [None]
_ACTIVE_MIXER: list = [None]


def _active_mixer_varlen():
    mixer = _ACTIVE_MIXER[0]
    return None if mixer is None else getattr(mixer, "_unsloth_varlen", None)


def _wrap_clear_attention_mask(fn, varlen_getter):
    if fn is None or not callable(fn) or getattr(fn, "_unsloth_varlen_mask_cleared", False):
        return fn

    @wraps(fn)
    def wrapped(*args, **kwargs):
        if varlen_getter() is not None:
            return _call_as_packed_mamba2_prefill(fn, args, kwargs)
        return fn(*args, **kwargs)

    wrapped._unsloth_varlen_mask_cleared = True
    return wrapped


def _install_mamba2_mask_clear(
    namespace,
    varlen_getter,
    classes = (),
) -> None:
    if not isinstance(namespace, dict):
        return
    for name, value in list(namespace.items()):
        if not _is_mamba2_mask_clear_name(name, classes) or not callable(value):
            continue
        namespace[name] = _wrap_clear_attention_mask(value, varlen_getter)


_MAMBA2_FALLBACK_SEQ_IDX = {
    "mamba_chunk_scan_combined": "_unsloth_varlen_scan_hit",
    "causal_conv1d_fn": "_unsloth_varlen_conv_hit",
}


def _install_mamba2_seq_idx_fallbacks(namespace) -> None:
    if not isinstance(namespace, dict):
        return
    for name, hit_attr in _MAMBA2_FALLBACK_SEQ_IDX.items():
        fn = namespace.get(name)
        if not callable(fn) or getattr(fn, "_unsloth_varlen_seq_idx_wrapped", False):
            continue
        namespace[name] = _wrap_mamba2_seq_idx_call(fn, hit_attr = hit_attr)


def _resolve_mamba2_fused(module):
    orig = getattr(module, "_unsloth_varlen_orig_fused", None)
    if callable(orig):
        return orig, ("instance", None, None)
    for name in _MAMBA2_FUSED_NAMES:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn, ("instance", None, name)
    modeling = inspect.getmodule(type(module))
    if modeling is not None:
        modeling_dict = getattr(modeling, "__dict__", None)
        if isinstance(modeling_dict, dict):
            for name in _MAMBA2_FUSED_NAMES:
                fn = modeling_dict.get(name)
                if callable(fn):
                    return fn, ("modeling", modeling, name)
    try:
        from mamba_ssm.ops.triton.ssd_combined import (  # type: ignore
            mamba_split_conv1d_scan_combined as fn,
        )
    except Exception:
        return None, None
    return fn, ("ssm", None, "mamba_split_conv1d_scan_combined")


def _hybrid_varlen_kernels_available(gated_delta_modules) -> Optional[str]:
    """None if every module can use the accelerated varlen path, else a short
    reason string. All modules are validated before any are mutated; signatures
    are read off the captured originals when already wrapped.

    Dispatch (the mixer actually calling self.causal_conv1d_fn /
    self.chunk_gated_delta_rule) is verified at RUNTIME by the forward-wrapper
    handshake, not statically: Unsloth's compile-disable shim hides it from
    inspect.getsource, and every supported transformers release dispatches
    through the instance attribute."""
    if not gated_delta_modules:
        return "no gated-delta modules found"
    for module in gated_delta_modules:
        # Split q/k/v short convs were checked for cu_seqlens by _stateful_mixer_kind.
        split_convs = _stateful_mixer_kind(module) == "gated_delta_qkv"
        conv = getattr(module, "_unsloth_varlen_orig_conv", None) or getattr(
            module,
            "causal_conv1d_fn",
            None,
        )
        if split_convs:
            conv = getattr(module, "q_conv1d").forward
        scan = getattr(module, "_unsloth_varlen_orig_scan", None) or getattr(
            module,
            "chunk_gated_delta_rule",
            None,
        )
        if conv is None or scan is None:
            return "accelerated kernels missing (install causal_conv1d and fla)"
        if getattr(scan, "__name__", "").startswith("torch_") or getattr(
            conv,
            "__name__",
            "",
        ).startswith("torch_"):
            return "pure-torch kernel fallback in use"
        try:
            if not split_convs and "seq_idx" not in inspect.signature(conv).parameters:
                return "conv kernel does not accept seq_idx"
            if "cu_seqlens" not in inspect.signature(scan).parameters:
                return "scan kernel does not accept cu_seqlens"
        except (TypeError, ValueError):
            return "kernel signature not introspectable"
    return None


def _mamba2_varlen_kernels_available(mamba2_modules) -> Optional[str]:
    if not mamba2_modules:
        return "no mamba2 modules found"
    for module in mamba2_modules:
        fn, _loc = _resolve_mamba2_fused(module)
        if fn is None:
            return "mamba2 fused kernel missing (install mamba_ssm)"
        if getattr(fn, "__name__", "").startswith("torch_"):
            return "pure-torch kernel fallback in use"
        reason = _callable_accepts_named_seq_idx(fn)
        if reason is not None:
            return reason
        if _mamba2_kernel_globals(module).get("is_fast_path_available", True) is False:
            return "mamba2 fast path unavailable (install mamba_ssm and causal_conv1d)"
    return None


def _mamba2_kernel_globals(module) -> dict:
    # The mixer's __init__ assigns the lazily loaded kernels into its own module globals.
    globs = getattr(type(module).__init__, "__globals__", None)
    return globs if isinstance(globs, dict) else {}


def _hybrid_varlen_dispatched(module) -> bool:
    if getattr(module, "_unsloth_varlen_kwargs_mode", False):
        return bool(getattr(module, "_unsloth_varlen_kwargs_hit", False))
    kind = _stateful_mixer_kind(module)
    if kind == "short_conv":
        return bool(getattr(module, "_unsloth_varlen_conv_hit", False))
    if kind == "ssd":
        if getattr(module, "_unsloth_varlen_fused_hit", False):
            return True
        return bool(
            getattr(module, "_unsloth_varlen_conv_hit", False)
            and getattr(module, "_unsloth_varlen_scan_hit", False)
        )
    return bool(
        getattr(module, "_unsloth_varlen_conv_hit", False)
        and getattr(module, "_unsloth_varlen_scan_hit", False)
    )


def _varlen_seq_idx_applies(seq_idx, args, kwargs) -> bool:
    # generate() bypasses model.forward, so a stale training seq_idx must not reach a prompt.
    if seq_idx is None:
        return False
    total = seq_idx.shape[-1]
    tensor = None
    for name in ("x", "hidden_states", "zxbcdt"):
        candidate = kwargs.get(name)
        if isinstance(candidate, torch.Tensor):
            tensor = candidate
            break
    if tensor is None:
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
    if tensor is None or tensor.dim() < 2 or seq_idx.shape[0] != tensor.shape[0]:
        return False
    # causal_conv1d_fn takes (batch, dim, seqlen), the others (batch, seqlen, ...).
    return total in (tensor.shape[1], tensor.shape[-1])


def _wrap_mamba2_seq_idx_call(orig, *, hit_attr = "_unsloth_varlen_fused_hit"):
    # Wrap once: Nemotron-H mixers share one kernel name, nesting per layer hits RecursionError.
    if getattr(orig, "_unsloth_varlen_seq_idx_wrapped", False):
        return orig

    @wraps(orig)
    def wrapped(*args, **kwargs):
        # The calling mixer's own stash, so recompute and a second patched model both resolve correctly.
        mixer, varlen = _ACTIVE_MIXER[0], _active_mixer_varlen()
        if varlen is not None:
            if kwargs.get("seq_idx") is None and _varlen_seq_idx_applies(varlen[1], args, kwargs):
                kwargs["seq_idx"] = varlen[1]
            if kwargs.get("seq_idx") is not None:
                setattr(mixer, hit_attr, True)
        return orig(*args, **kwargs)

    wrapped._unsloth_varlen_seq_idx_wrapped = True
    return wrapped


def _rebind_mamba2_fused_aliases(orig, wrapped) -> None:
    # LOAD_GLOBAL resolves through the module dict at call time; misses are caught by the handshake.
    if orig is None or wrapped is orig:
        return
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        namespace = getattr(mod, "__dict__", None)
        if not isinstance(namespace, dict):
            continue
        for key, value in list(namespace.items()):
            if value is orig:
                namespace[key] = wrapped


def _wrap_mamba2_mixer_forward(module):
    if getattr(module, "_unsloth_mamba2_forward_wrapped", False):
        return
    forward_orig = module.forward
    cuda_orig = getattr(module, "cuda_kernels_forward", None)
    # A **kwargs-only forward may splat into a kernel that also gets seq_idx (duplicate keyword).
    forward_names_seq_idx = False
    try:
        forward_names_seq_idx = "seq_idx" in inspect.signature(forward_orig).parameters
    except (TypeError, ValueError):
        pass

    def _packed():
        return getattr(module, "_unsloth_varlen", None)

    @wraps(forward_orig)
    def mixer_forward(*args, **kwargs):
        varlen = _packed()
        outer, _ACTIVE_MIXER[0] = _ACTIVE_MIXER[0], module
        try:
            if varlen is None:
                return forward_orig(*args, **kwargs)
            if (
                forward_names_seq_idx
                and kwargs.get("seq_idx") is None
                and _varlen_seq_idx_applies(varlen[1], args, kwargs)
            ):
                kwargs["seq_idx"] = varlen[1]
            return _call_as_packed_mamba2_prefill(forward_orig, args, kwargs)
        finally:
            _ACTIVE_MIXER[0] = outer

    module.forward = mixer_forward
    if callable(cuda_orig) and not getattr(cuda_orig, "_unsloth_varlen_mask_cleared", False):

        @wraps(cuda_orig)
        def cuda_kernels_forward(*args, **kwargs):
            if _packed() is not None:
                return _call_as_packed_mamba2_prefill(cuda_orig, args, kwargs)
            return cuda_orig(*args, **kwargs)

        cuda_kernels_forward._unsloth_varlen_mask_cleared = True
        module.cuda_kernels_forward = cuda_kernels_forward
    module._unsloth_mamba2_forward_wrapped = True


# transformers >= 5.16 mixers splat **kwargs into module-level kernels wrapped by
# use_kernel_func_from_hub_with_fallback; each must bind an accelerated kernel taking this boundary.
_KWARGS_KERNELS = {
    "causal_conv1d_fn": "seq_idx",
    "mamba2_split_conv1d_scan_combined": "seq_idx",
    "mamba2_chunk_scan": "seq_idx",
    "torch_chunk_gated_delta_rule": "cu_seqlens",
}
_KWARGS_TRACE: list = [None]


def _hub_closure(fn) -> dict:
    fn = getattr(fn, "_unsloth_varlen_probe_of", fn)
    try:
        return inspect.getclosurevars(fn).nonlocals
    except (TypeError, ValueError):
        return {}


def _is_kwargs_mixer(module) -> bool:
    if hasattr(module, "cuda_kernels_forward") or "causal_conv1d_fn" in vars(module):
        return False
    try:
        params = inspect.signature(module.forward).parameters.values()
    except (TypeError, ValueError):
        return False
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return False
    globs = getattr(inspect.getmodule(type(module)), "__dict__", {})
    return "implementation" in _hub_closure(globs.get("causal_conv1d_fn"))


def _kwargs_kernels_available(modules, recurrent_ids = ()) -> Optional[str]:
    for module in modules:
        globs = getattr(inspect.getmodule(type(module)), "__dict__", {})
        kernels = {n: globs[n] for n in _KWARGS_KERNELS if callable(globs.get(n))}
        if "causal_conv1d_fn" not in kernels:
            return f"{type(module).__name__}: no module-level conv kernel"
        if id(module) in recurrent_ids and len(kernels) < 2:
            return f"{type(module).__name__}: no module-level recurrent kernel"
        for name, fn in kernels.items():
            closure = _hub_closure(fn)
            # The hub wrapper drops kwargs its bound implementation does not name, silently.
            impl = closure.get("implementation")
            if impl is None or impl is closure.get("torch_function"):
                return f"{name} has no accelerated kernel"
            if _KWARGS_KERNELS[name] not in closure.get("applicable_params", ()):
                return f"{name} kernel does not accept {_KWARGS_KERNELS[name]}"
    return None


def _install_kwargs_probes(namespace) -> None:
    if not isinstance(namespace, dict):
        return
    for name, param in _KWARGS_KERNELS.items():
        fn = namespace.get(name)
        if not callable(fn) or getattr(fn, "_unsloth_varlen_probe_of", None) is not None:
            continue

        @wraps(fn)
        def probe(
            *args,
            _fn = fn,
            _name = name,
            _param = param,
            **kwargs,
        ):
            if _KWARGS_TRACE[0] is not None:
                _KWARGS_TRACE[0].append((_name, kwargs.get(_param) is not None))
            return _fn(*args, **kwargs)

        probe._unsloth_varlen_probe_of = fn
        namespace[name] = probe


def _wrap_kwargs_mixer_forward(module) -> None:
    forward_orig = module.forward

    @wraps(forward_orig)
    def forward(*args, **kwargs):
        varlen = getattr(module, "_unsloth_varlen", None)
        if varlen is None or not _varlen_seq_idx_applies(varlen[1], args, kwargs):
            return forward_orig(*args, **kwargs)
        if kwargs.get("seq_idx") is None:
            kwargs["seq_idx"] = varlen[1]
        if kwargs.get("cu_seq_lens_q") is None:
            kwargs["cu_seq_lens_q"] = varlen[0]
        outer, _KWARGS_TRACE[0] = _KWARGS_TRACE[0], []
        try:
            out = forward_orig(*args, **kwargs)
        finally:
            trace, _KWARGS_TRACE[0] = _KWARGS_TRACE[0], outer
        # Every boundary-capable kernel this mixer ran, a recurrent one included, must have received it.
        if all(ok for _, ok in trace) and any(n != "causal_conv1d_fn" for n, _ in trace):
            module._unsloth_varlen_kwargs_hit = True
        return out

    module.forward = forward
    module._unsloth_varlen_kwargs_mode = True


_MASK_BUILDERS = ("create_causal_mask", "create_sliding_window_causal_mask")


def _packed_position_ids(varlen):
    cu_seqlens, seq_idx = varlen
    if seq_idx.shape[0] != 1:
        return None
    seg = seq_idx[0].long()
    pos = torch.arange(seg.numel(), device = seg.device) - cu_seqlens.to(seg.device).long()[seg]
    return pos[None]


def _install_packed_mask_positions(namespace, varlen_getter) -> None:
    """Some hybrids (GraniteMoeHybrid) build the attention mask without position_ids, so attention
    would span packed sequences; hand the builder per-sequence positions while packed."""
    if not isinstance(namespace, dict):
        return
    for name in _MASK_BUILDERS:
        fn = namespace.get(name)
        if not callable(fn) or getattr(fn, "_unsloth_packed_positions", False):
            continue

        @wraps(fn)
        def build(
            *args,
            _fn = fn,
            **kwargs,
        ):
            varlen = varlen_getter()
            if varlen is not None and kwargs.get("position_ids") is None:
                kwargs["position_ids"] = _packed_position_ids(varlen)
            return _fn(*args, **kwargs)

        build._unsloth_packed_positions = True
        namespace[name] = build


def _wrap_qkv_conv_forward(conv, owner) -> None:
    # fla ShortConvolution resets at cu_seqlens; the owning mixer holds the packed boundaries.
    forward_orig = conv.forward

    @wraps(forward_orig)
    def forward(*args, **kwargs):
        varlen = getattr(owner, "_unsloth_varlen", None)
        if varlen is not None and kwargs.get("cu_seqlens") is None:
            if _varlen_seq_idx_applies(varlen[1], args, kwargs):
                kwargs["cu_seqlens"] = varlen[0]
        if kwargs.get("cu_seqlens") is not None:
            owner._unsloth_varlen_conv_hit = True
        return forward_orig(*args, **kwargs)

    conv.forward = forward


def _wrap_short_conv_forward(module) -> None:
    forward_orig = module.forward

    @wraps(forward_orig)
    def forward(*args, **kwargs):
        varlen = getattr(module, "_unsloth_varlen", None)
        if (
            varlen is not None
            and kwargs.get("seq_idx") is None
            and _varlen_seq_idx_applies(varlen[1], args, kwargs)
        ):
            kwargs["seq_idx"] = varlen[1]
        outer, _KWARGS_TRACE[0] = _KWARGS_TRACE[0], []
        try:
            out = forward_orig(*args, **kwargs)
        finally:
            trace, _KWARGS_TRACE[0] = _KWARGS_TRACE[0], outer
        # A probed (hub) conv kernel must itself have received seq_idx; the torch fallback drops it.
        if kwargs.get("seq_idx") is not None and all(ok for _, ok in trace):
            module._unsloth_varlen_conv_hit = True
        return out

    module.forward = forward


def _uses_hub_conv(module) -> bool:
    globs = getattr(inspect.getmodule(type(module)), "__dict__", {})
    return "implementation" in _hub_closure(globs.get("causal_conv1d_fn"))


def _varlen_from_position_ids(position_ids):
    """(cu_seqlens int32[n+1], seq_idx int32[1,T]) for a flattened padding-free
    batch, else None. Padding-free position_ids reset to 0 at each sequence start;
    accepts only a validated single-row pack (normal batch or single sequence ->
    None). Fallback used only when packed_seq_lengths is absent: it assumes
    right-packed reset position_ids and would mis-segment a left-padded row, which
    is why packed_seq_lengths is always preferred."""
    if position_ids is None:
        return None
    pos = position_ids
    if pos.dim() == 3:  # MRoPE [n_planes, 1, T] -> text plane is index 0
        pos = pos[0]
    if pos.dim() != 2 or pos.shape[0] != 1:
        return None
    row = pos[0]
    total = row.shape[0]
    starts = (row == 0).nonzero(as_tuple = False).flatten()
    if starts.numel() <= 1 or int(starts[0].item()) != 0:
        return None
    cu_seqlens = torch.cat(
        [
            starts.to(torch.int32),
            torch.tensor([total], dtype = torch.int32, device = row.device),
        ]
    )
    return _seq_idx_from_cu_seqlens(cu_seqlens, total)


def _seq_idx_from_cu_seqlens(cu_seqlens, total):
    """(cu_seqlens int32[n+1], seq_idx int32[1,total]) partitioning [0, total),
    else None. Appends a trailing segment for pad_to_multiple_of zero tokens so the
    boundaries always cover the full flattened length the kernels see."""
    if cu_seqlens is None or cu_seqlens.numel() < 2 or int(cu_seqlens[0].item()) != 0:
        return None
    boundaries = cu_seqlens.to(torch.int32)
    last = int(boundaries[-1].item())
    if last > total:
        return None
    if last < total:  # trailing pad tokens -> one final segment
        boundaries = torch.cat(
            [
                boundaries,
                torch.tensor([total], dtype = torch.int32, device = boundaries.device),
            ]
        )
    lengths = boundaries[1:] - boundaries[:-1]
    if not bool((lengths > 0).all()):
        return None
    seq_idx = torch.repeat_interleave(
        torch.arange(lengths.numel(), dtype = torch.int32, device = boundaries.device),
        lengths.to(torch.int64),
    ).unsqueeze(0)
    return boundaries, seq_idx


def _hybrid_varlen_metadata(kwargs):
    """Boundary metadata (cu_seqlens, seq_idx) for one flattened packed forward,
    else None. Prefers the authoritative packed_seq_lengths, falls back to
    reset-style position_ids. Returns None for cached forwards and non-packed
    batches so decode / eval / normal batches are a strict no-op."""
    if kwargs.get("use_cache"):
        return None
    if kwargs.get("past_key_values") is not None or kwargs.get("cache_params") is not None:
        return None
    total, device = None, None
    for key in ("input_ids", "inputs_embeds", "position_ids"):
        tensor = kwargs.get(key)
        if tensor is not None and hasattr(tensor, "shape"):
            total = tensor.shape[1] if key == "inputs_embeds" else tensor.shape[-1]
            device = tensor.device
            break
    if total is None:
        return None
    psl = kwargs.get("packed_seq_lengths")
    if psl is not None and getattr(psl, "numel", lambda: 1)() > 0:
        info = get_packed_info_from_kwargs(kwargs, device)
        if info is not None:
            _, cu_seqlens, _ = info
            built = _seq_idx_from_cu_seqlens(cu_seqlens, total)
            if built is not None:
                return built
    return _varlen_from_position_ids(kwargs.get("position_ids"))


def patch_hybrid_linear_attention_varlen(model) -> bool:
    """Feed seq_idx / cu_seqlens to hybrid mixers so packing resets state at sequence boundaries.
    Gated by UNSLOTH_EXPERIMENTAL_HYBRID_PACKING (short-conv-only models pack without it), fail-closed,
    idempotent. True when active."""
    mixers = _iter_stateful_mixers(model)
    # Short-conv-only models (LFM2) reset at seq_idx on both kernel and torch paths, so they pack by default.
    short_conv_only = bool(mixers) and all(kind == "short_conv" for _, kind in mixers)
    if not (_hybrid_packing_enabled() or short_conv_only):
        return False
    unsupported = sorted({type(m).__name__ for m, kind in mixers if kind == "unsupported"})
    if unsupported:
        return _hybrid_reject(f"no varlen kernel path for {unsupported}")
    gated_delta_modules = [m for m, kind in mixers if kind in ("gated_delta", "gated_delta_qkv")]
    mamba2_modules = [m for m, kind in mixers if kind == "ssd"]
    short_conv_modules = [m for m, kind in mixers if kind == "short_conv"]
    hybrid_modules = gated_delta_modules + mamba2_modules + short_conv_modules
    if not hybrid_modules:
        return _hybrid_reject("no stateful mixers found")
    kwargs_modules = [
        m
        for m in gated_delta_modules + mamba2_modules
        if getattr(m, "_unsloth_varlen_kwargs_mode", False) or _is_kwargs_mixer(m)
    ]
    kwargs_ids = {id(m) for m in kwargs_modules}
    gated_delta_modules = [m for m in gated_delta_modules if id(m) not in kwargs_ids]
    mamba2_modules = [m for m in mamba2_modules if id(m) not in kwargs_ids]

    # Idempotency: an already fully-patched model stays active without re-validation.
    if getattr(model, "_unsloth_varlen_forward_wrapped", False) and all(
        getattr(m, "_unsloth_varlen_wrapped", False) for m in hybrid_modules
    ):
        return True

    hub_short_convs = [m for m in short_conv_modules if _uses_hub_conv(m)]
    if kwargs_modules or hub_short_convs:
        reason = _kwargs_kernels_available(
            kwargs_modules + hub_short_convs, recurrent_ids = {id(m) for m in kwargs_modules}
        )
        if reason is not None:
            return _hybrid_reject(reason)
    if gated_delta_modules:
        reason = _hybrid_varlen_kernels_available(gated_delta_modules)
        if reason is not None:
            return _hybrid_reject(reason)
    if mamba2_modules:
        reason = _mamba2_varlen_kernels_available(mamba2_modules)
        if reason is not None:
            return _hybrid_reject(reason)

    # Transactional: every module validated above, now wrap each and stash originals.
    for module in gated_delta_modules:
        if getattr(module, "_unsloth_varlen_wrapped", False):
            continue
        scan_orig = module.chunk_gated_delta_rule
        module._unsloth_varlen_orig_scan = scan_orig
        if _stateful_mixer_kind(module) == "gated_delta_qkv":
            for conv_name in _QKV_CONVS:
                _wrap_qkv_conv_forward(getattr(module, conv_name), module)
            conv_orig = None
        else:
            conv_orig = module.causal_conv1d_fn
            module._unsloth_varlen_orig_conv = conv_orig

        @wraps(conv_orig)
        def conv_fn(
            *args,
            _orig = conv_orig,
            _module = module,
            **kwargs,
        ):
            varlen = getattr(_module, "_unsloth_varlen", None)
            if varlen is not None:
                _module._unsloth_varlen_conv_hit = True  # runtime dispatch handshake
                if kwargs.get("seq_idx") is None:
                    kwargs["seq_idx"] = varlen[1]
            return _orig(*args, **kwargs)

        @wraps(scan_orig)
        def scan_fn(
            *args,
            _orig = scan_orig,
            _module = module,
            **kwargs,
        ):
            varlen = getattr(_module, "_unsloth_varlen", None)
            if varlen is not None:
                _module._unsloth_varlen_scan_hit = True
                if kwargs.get("cu_seqlens") is None:
                    kwargs["cu_seqlens"] = varlen[0]
            return _orig(*args, **kwargs)

        if conv_orig is not None:
            module.causal_conv1d_fn = conv_fn
        module.chunk_gated_delta_rule = scan_fn
        module._unsloth_varlen = None
        module._unsloth_varlen_wrapped = True

    for ns in _iter_mamba2_install_namespaces(hybrid_modules):
        _install_packed_mask_positions(ns, lambda: _PACKED[0])
    for ns in _iter_mamba2_install_namespaces(kwargs_modules + hub_short_convs):
        _install_kwargs_probes(ns)
    for module in kwargs_modules:
        if not getattr(module, "_unsloth_varlen_wrapped", False):
            _wrap_kwargs_mixer_forward(module)
            module._unsloth_varlen = None
            module._unsloth_varlen_wrapped = True
    for module in short_conv_modules:
        if not getattr(module, "_unsloth_varlen_wrapped", False):
            _wrap_short_conv_forward(module)
            module._unsloth_varlen = None
            module._unsloth_varlen_wrapped = True

    wrapped_fused: dict[int, Any] = {}

    def _ensure_mamba2_fused_wrapped(fn):
        if fn is None:
            return None
        if getattr(fn, "_unsloth_varlen_seq_idx_wrapped", False):
            return fn
        wrapped = wrapped_fused.get(id(fn))
        if wrapped is None:
            wrapped = _wrap_mamba2_seq_idx_call(fn)
            wrapped_fused[id(fn)] = wrapped
            _rebind_mamba2_fused_aliases(fn, wrapped)
        return wrapped

    wrapped_real = None
    for module in mamba2_modules:
        if getattr(module, "_unsloth_varlen_wrapped", False):
            continue
        fn, loc = _resolve_mamba2_fused(module)
        wrapped = _ensure_mamba2_fused_wrapped(fn)
        wrapped_real = wrapped_real or wrapped
        kind = loc[0] if loc is not None else None
        if kind == "instance" and wrapped is not None:
            module._unsloth_varlen_orig_fused = fn
            setattr(module, loc[2] or _MAMBA2_FUSED_NAMES[0], wrapped)
        elif kind == "modeling" and wrapped is not None:
            setattr(loc[1], loc[2], wrapped)
        elif kind == "ssm":
            module._unsloth_varlen_orig_fused = fn
        _wrap_mamba2_mixer_forward(module)
        module._unsloth_varlen = None
        module._unsloth_varlen_wrapped = True
    if mamba2_modules:
        if wrapped_real is None:
            try:
                from mamba_ssm.ops.triton.ssd_combined import (  # type: ignore
                    mamba_split_conv1d_scan_combined as _ssm_fused,
                )
            except Exception:
                _ssm_fused = None
            wrapped_real = _ensure_mamba2_fused_wrapped(_ssm_fused)
        source = _mamba2_kernel_globals(mamba2_modules[0])

        classes = {type(m).__name__ for m in mamba2_modules}

        for ns in _iter_mamba2_install_namespaces(mamba2_modules):
            _force_install_mamba2_fused(ns, wrapped_real, source)
            _install_mamba2_mask_clear(ns, _active_mixer_varlen, classes)
            _install_mamba2_seq_idx_fallbacks(ns)

    # Refresh the boundary stash on the outermost forward, once per step and outside gradient-checkpoint
    # recompute so it stays valid for recomputed inner forwards.
    if not getattr(model, "_unsloth_varlen_forward_wrapped", False):
        forward_orig = model.forward
        try:
            forward_sig = inspect.signature(forward_orig)
        except (TypeError, ValueError):
            forward_sig = None

        @wraps(forward_orig)
        def forward_with_varlen(*args, **kwargs):
            try:
                bound = dict(kwargs)
                if forward_sig is not None and args:
                    bound.update(forward_sig.bind_partial(*args).arguments)
                varlen = _hybrid_varlen_metadata(bound)
            except Exception:
                varlen = None
            first_pack = varlen is not None and not getattr(
                model,
                "_unsloth_varlen_handshake_done",
                False,
            )
            for module in hybrid_modules:
                module._unsloth_varlen = varlen
                if first_pack:
                    module._unsloth_varlen_conv_hit = False
                    module._unsloth_varlen_scan_hit = False
                    module._unsloth_varlen_fused_hit = False
                    module._unsloth_varlen_kwargs_hit = False
            outer, _PACKED[0] = _PACKED[0], varlen
            try:
                out = forward_orig(*args, **kwargs)
            finally:
                _PACKED[0] = outer
            # Handshake: on the first packed forward every module must have hit its boundary kernels.
            if first_pack:
                model._unsloth_varlen_handshake_done = True
                missing = [
                    type(m).__name__ for m in hybrid_modules if not _hybrid_varlen_dispatched(m)
                ]
                if missing:
                    for m in hybrid_modules:
                        m._unsloth_varlen = None
                    _hybrid_reject("varlen kernels not dispatched (dispatch changed?)")
                    raise RuntimeError(
                        "Unsloth: experimental hybrid packing cannot continue because the "
                        "varlen boundary kernels were not invoked for "
                        f"{sorted(set(missing))}. Unset UNSLOTH_EXPERIMENTAL_HYBRID_PACKING "
                        "to train these models on the padded path.\n"
                        + _mamba2_handshake_debug(
                            [m for m in hybrid_modules if _stateful_mixer_kind(m) == "ssd"]
                            or hybrid_modules
                        )
                    )
            return out

        model.forward = forward_with_varlen
        model._unsloth_varlen_forward_wrapped = True

    _wrap_generate_clears_varlen(model, hybrid_modules)
    return True


def _wrap_generate_clears_varlen(model, hybrid_modules) -> None:
    generate_orig = getattr(model, "generate", None)
    if not callable(generate_orig) or getattr(model, "_unsloth_varlen_generate_wrapped", False):
        return

    @wraps(generate_orig)
    def generate_without_varlen(*args, **kwargs):
        # Restore after: gradient-checkpoint recompute in a pending backward still needs the stash.
        saved = [module._unsloth_varlen for module in hybrid_modules]
        for module in hybrid_modules:
            module._unsloth_varlen = None
        try:
            return generate_orig(*args, **kwargs)
        finally:
            for module, varlen in zip(hybrid_modules, saved):
                module._unsloth_varlen = varlen

    model.generate = generate_without_varlen
    model._unsloth_varlen_generate_wrapped = True


def get_packed_info_from_kwargs(
    kwargs: dict, device: torch.device
) -> Optional[Tuple[torch.Tensor, torch.Tensor, int]]:
    """Return packed sequence metadata expected by the attention kernels."""

    seq_lengths = kwargs.get("packed_seq_lengths")
    if seq_lengths is None:
        return None

    entry = _PACKED_INFO_CACHE.get(device)
    if entry is not None and entry["seq_lengths"] is seq_lengths:
        return entry["result"]

    lengths = seq_lengths.to(device = device, dtype = torch.int32, non_blocking = True)
    cu_seqlens = torch.zeros(lengths.numel() + 1, dtype = torch.int32, device = device)
    torch.cumsum(lengths, dim = 0, dtype = torch.int32, out = cu_seqlens[1:])

    max_seqlen = int(lengths.max().item())
    result = (lengths, cu_seqlens, max_seqlen)
    _PACKED_INFO_CACHE[device] = {"seq_lengths": seq_lengths, "result": result}
    return result


def _with_padding_segment(lengths: Tuple[int, ...], total_tokens: Optional[int]) -> Tuple[int, ...]:
    # TRL pads the flattened padding-free row to pad_to_multiple_of after the lengths are
    # taken; the tail gets its own block, since a row outside every block softmaxes to NaN.
    if total_tokens is None:
        return lengths
    padding = total_tokens - sum(lengths)
    if padding <= 0:
        return lengths
    return lengths + (padding,)


def cover_padded_cu_seqlens(
    seq_info: Tuple[torch.Tensor, torch.Tensor, int], total_tokens: int
) -> Tuple[torch.Tensor, int]:
    """Flash varlen leaves rows past cu_seqlens[-1] unwritten, so the pad tail gets a segment."""
    _, cu_seqlens, max_seqlen = seq_info
    device = cu_seqlens.device
    entry = _PADDED_CU_SEQLENS_CACHE.get(device)
    if entry is not None and entry["cu_seqlens"] is cu_seqlens and entry["total"] == total_tokens:
        return entry["result"]

    padding = total_tokens - int(cu_seqlens[-1].item())
    result = (cu_seqlens, max_seqlen)
    if padding > 0:
        tail = torch.tensor([total_tokens], dtype = cu_seqlens.dtype, device = device)
        result = (torch.cat([cu_seqlens, tail]), max(max_seqlen, padding))
    _PADDED_CU_SEQLENS_CACHE[device] = {
        "cu_seqlens": cu_seqlens,
        "total": total_tokens,
        "result": result,
    }
    return result


def build_xformers_block_causal_mask(
    seq_info: Optional[Tuple[torch.Tensor, torch.Tensor, int]],
    *,
    sliding_window: Optional[int] = None,
    base_mask: Optional[Any] = None,
    total_tokens: Optional[int] = None,
    is_causal: bool = True,
):
    mask_class = _XFormersBlockMask if is_causal else _XFormersBidirectionalMask
    if mask_class is None:
        return None
    if seq_info is not None:
        seq_lengths, _, _ = seq_info
        # Cache the mask to avoid repeated D2H sync across layers
        device = seq_lengths.device
        params = (sliding_window, total_tokens, is_causal)
        entry = _XFORMERS_BLOCK_MASK_CACHE.get(device)
        if entry is not None and entry["seq_lengths"] is seq_lengths and entry["params"] == params:
            return entry["mask"]

        lengths_tensor = seq_lengths.to("cpu", torch.int32)
        if lengths_tensor.numel() == 0:
            return None
        lengths = tuple(int(x) for x in lengths_tensor.tolist())
        lengths = _with_padding_segment(lengths, total_tokens)
        mask = _get_cached_block_mask(lengths, sliding_window, device, is_causal = is_causal)

        _XFORMERS_BLOCK_MASK_CACHE[device] = {
            "seq_lengths": seq_lengths,
            "params": params,
            "mask": mask,
        }
    else:
        mask = base_mask

        if (
            sliding_window is not None
            and sliding_window > 0
            and mask is not None
            and hasattr(mask, "make_local_attention")
        ):
            mask = mask.make_local_attention(window_size = sliding_window)
    return mask


def packed_block_mask(
    length: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    sliding_window: Optional[int] = None,
    is_causal: bool = True,
) -> torch.Tensor:
    """Additive (length, length) mask of one packed segment: causal and / or sliding window."""
    block = torch.zeros((length, length), dtype = dtype, device = device)
    if is_causal:
        upper = torch.triu(torch.ones((length, length), device = device), diagonal = 1).bool()
        block = block.masked_fill(upper, float("-inf"))
    if sliding_window is not None and sliding_window > 0 and length > sliding_window:
        idx = torch.arange(length, device = device)
        dist = idx.unsqueeze(1) - idx.unsqueeze(0)
        block = block.masked_fill(dist >= sliding_window, float("-inf"))
    return block


def packed_segment_lengths(
    seq_info: Tuple[torch.Tensor, torch.Tensor, int], total_tokens: Optional[int] = None
) -> Tuple[int, ...]:
    """Segment lengths of a packed row, the pad tail as its own segment; one D2H sync per batch."""
    seq_lengths = seq_info[0]
    entry = _SEGMENT_LENGTHS_CACHE.get(seq_lengths.device)
    if entry is not None and entry[0] is seq_lengths and entry[1] == total_tokens:
        return entry[2]
    lengths = _with_padding_segment(
        tuple(int(length) for length in seq_lengths.tolist()), total_tokens
    )
    _SEGMENT_LENGTHS_CACHE[seq_lengths.device] = (seq_lengths, total_tokens, lengths)
    return lengths


def build_sdpa_packed_attention_mask(
    seq_info: Tuple[torch.Tensor, torch.Tensor, int],
    *,
    dtype: torch.dtype,
    device: torch.device,
    sliding_window: Optional[int] = None,
    total_tokens: Optional[int] = None,
    is_causal: bool = True,
) -> torch.Tensor:
    seq_lengths, _, _ = seq_info

    params = (dtype, sliding_window, total_tokens, is_causal)
    entry = _SDPA_MASK_CACHE.get(device)
    if entry is not None and entry["seq_lengths"] is seq_lengths and entry["params"] == params:
        return entry["mask"]

    lengths = _with_padding_segment(
        tuple(int(length) for length in seq_lengths.tolist()),
        total_tokens,
    )
    total_tokens = sum(lengths)
    mask = torch.full(
        (total_tokens, total_tokens),
        float("-inf"),
        dtype = dtype,
        device = device,
    )
    offset = 0
    for length in lengths:
        if length <= 0:
            continue
        mask[offset : offset + length, offset : offset + length] = packed_block_mask(
            length,
            dtype = dtype,
            device = device,
            sliding_window = sliding_window,
            is_causal = is_causal,
        )
        offset += length

    result = mask.unsqueeze(0).unsqueeze(0)
    _SDPA_MASK_CACHE[device] = {
        "seq_lengths": seq_lengths,
        "params": params,
        "mask": result,
    }
    return result


def _normalize_packed_lengths(seq_lengths: Any, *, device: torch.device) -> Optional[torch.Tensor]:
    if seq_lengths is None:
        return None
    if isinstance(seq_lengths, torch.Tensor):
        lengths = seq_lengths.to(device = device, dtype = torch.int64)
    else:
        lengths = torch.tensor(seq_lengths, device = device, dtype = torch.int64)
    if lengths.ndim != 1:
        lengths = lengths.reshape(-1)
    if lengths.numel() == 0:
        return None
    return lengths


def mask_packed_sequence_boundaries(
    shift_labels: torch.Tensor,
    seq_lengths: Any,
    *,
    ignore_index: int = -100,
) -> bool:
    """Mark final token of every packed sample so CE ignores boundary predictions."""
    lengths = _normalize_packed_lengths(seq_lengths, device = shift_labels.device)
    if lengths is None:
        return False

    flat = shift_labels.reshape(-1)
    total_tokens = flat.shape[0]
    boundary_positions = torch.cumsum(lengths, dim = 0) - 1
    valid = boundary_positions < total_tokens
    if not torch.all(valid):
        boundary_positions = boundary_positions[valid]
    if boundary_positions.numel() == 0:
        return False
    flat[boundary_positions] = ignore_index
    return True


def mask_packed_boundary_labels(
    labels: Optional[torch.Tensor],
    seq_lengths: Any,
    *,
    ignore_index: int = -100,
) -> Optional[torch.Tensor]:
    """Same guard as :func:`mask_packed_sequence_boundaries`, but on RAW (unshifted)
    labels and out-of-place, for fused cross-entropy paths that shift internally.

    The shift maps target slot ``i`` to ``labels[i + 1]``, so masking shift slot
    ``cumsum - 1`` is exactly masking ``labels[cumsum]``, the first token of each
    following document.

    Returns ``labels`` unchanged when ``seq_lengths`` is absent or empty, else a NEW
    tensor; the caller's batch is never mutated. Idempotent, and a no-op on TRL's
    padding-free collator output (``labels[position_ids == 0] = -100``).

    Contract: ``sum(seq_lengths) <= labels.numel()``. Out-of-range entries (the final
    cumsum, or malformed lengths) redirect to index 0, which the shift discards and so
    is never a CE target; this avoids device syncs and data-dependent shapes in the
    compiled fused-CE path.
    """
    if labels is None or not isinstance(labels, torch.Tensor):
        return labels
    lengths = _normalize_packed_lengths(seq_lengths, device = labels.device)
    if lengths is None:
        return labels

    total_tokens = labels.numel()
    if total_tokens == 0:
        return labels

    positions = torch.cumsum(lengths, dim = 0)
    positions = torch.where(
        positions < total_tokens,
        positions,
        torch.zeros_like(positions),
    )
    flat = labels.reshape(-1).index_fill(0, positions, ignore_index)
    return flat.view(labels.shape)


def clear_packed_caches():
    """Release cached masks/metadata to free device memory."""
    _XFORMERS_MASK_CACHE.clear()
    _PACKED_INFO_CACHE.clear()
    _SDPA_MASK_CACHE.clear()
    _SEGMENT_LENGTHS_CACHE.clear()
    _XFORMERS_BLOCK_MASK_CACHE.clear()
    _PADDED_CU_SEQLENS_CACHE.clear()


__all__ = [
    "configure_sample_packing",
    "configure_padding_free",
    "enable_sample_packing",
    "enable_padding_free_metadata",
    "move_xformers_attention_bias",
    "mark_allow_overlength",
    "get_packed_info_from_kwargs",
    "build_xformers_block_causal_mask",
    "build_sdpa_packed_attention_mask",
    "cover_padded_cu_seqlens",
    "mask_packed_sequence_boundaries",
    "mask_packed_boundary_labels",
    "clear_packed_caches",
]
