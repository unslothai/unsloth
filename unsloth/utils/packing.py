# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
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

_XFORMERS_MASK_CACHE_MAXSIZE = 32
_XFORMERS_MASK_CACHE: OrderedDict[Tuple[torch.device, Tuple[int, ...], int], Any] = OrderedDict()

# Cache per device for get_packed_info_from_kwargs to avoid repeated D2H sync across layers
_PACKED_INFO_CACHE: dict = {}

# Cache per device for build_sdpa_packed_attention_mask to avoid repeated D2H sync across layers
_SDPA_MASK_CACHE: dict = {}

# Cache per device for build_xformers_block_causal_mask to avoid repeated D2H sync across layers
_XFORMERS_BLOCK_MASK_CACHE: dict = {}


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

        # Move the device-bearing metadata instead of the top-level mask. Older
        # xFormers versions demote causal masks in their inherited ``to`` method.
        # Copies also keep later model shards from rewriting masks retained for
        # backward by earlier shards.
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
    lengths: Tuple[int, ...], sliding_window: Optional[int], device: torch.device
):
    if _XFormersBlockMask is None:
        return None

    device = torch.device(device)
    window_key = _window_cache_key(sliding_window)
    cache_key = (device, lengths, window_key)
    cached = _XFORMERS_MASK_CACHE.get(cache_key)
    if cached is not None:
        _XFORMERS_MASK_CACHE.move_to_end(cache_key)
        return cached

    mask = _XFormersBlockMask.from_seqlens(list(lengths))
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
            # Fallback: infer lengths from tokenized inputs when metadata is absent
            if not seq_lengths:
                for example in examples:
                    ids = example.get("input_ids")
                    if isinstance(ids, Iterable):
                        seq_lengths.append(len(ids))
            if seq_lengths:
                # Boundary labels are NOT masked here. unsloth_zoo's
                # _unsloth_get_batch_samples counts num_items_in_batch off this batch and
                # discounts the N-1 boundary targets itself, idempotently: it zeroes those
                # slots rather than subtracting a constant, so the count is unaffected by
                # upstream masking (TRL >= 0.24's labels[position_ids == 0] = -100,
                # completion-only masking, assistant_masks). Masking here would be harmless
                # to the count; labels are left alone because the guard that needs these
                # positions runs in the forward, off packed_seq_lengths.
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
        if examples and isinstance(examples[0], dict):
            for example in examples:
                lengths = example.get("seq_lengths")
                if lengths is None:
                    ids = example.get("input_ids")
                    if ids is None:
                        continue
                    lengths = [len(ids)]
                    example["seq_lengths"] = lengths
                seq_lengths.extend(lengths)

        batch = original_torch_call(examples)
        if seq_lengths:
            # Labels left alone for the same reason as enable_sample_packing:
            # num_items_in_batch is counted off this batch, and the zoo's discount of the
            # boundary targets is idempotent, so masked slots do not change the count.
            batch["packed_seq_lengths"] = torch.tensor(
                seq_lengths,
                dtype = torch.int32,
            )
        return batch

    collator.torch_call = torch_call_with_padding_free_metadata
    collator._unsloth_padding_free_lengths_wrapped = True


# --- Experimental: correct packing / padding-free for hybrid linear-attention ---
# Qwen3.5 / Qwen3-Next mix a gated-delta recurrence with a causal conv1d. Nemotron-H
# mixes Mamba2 (fused conv1d+scan) with attention. Packing flattens the batch, and
# those ops leak state across sequence boundaries unless we pass seq_idx (conv /
# Mamba2 fused kernel) and cu_seqlens (gated-delta scan). Only the accelerated
# kernels accept these, so we fail closed on the pure-torch fallbacks. Gated
# behind an env flag.
#
# Gated-delta: override per-module prefill kernels (causal_conv1d_fn /
# chunk_gated_delta_rule). Mamba2: inject seq_idx into mixer.forward kwargs,
# force packed prefill (drop attention_mask + cache_params) so transformers
# takes the fused mem-eff path, wrap the fused kernel, and also wrap the
# chunk-scan fallback. Decode / cached forwards stay untouched.
# Feature-detect (never version-detect), fail closed, idempotent, one deduped
# diagnostic when it declines to activate.
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


def _iter_gated_delta_modules(model):
    modules, seen = [], set()
    for module in model.modules():
        if id(module) in seen:
            continue
        seen.add(id(module))
        if type(module).__name__.endswith("GatedDeltaNet") and hasattr(module, "conv1d"):
            modules.append(module)
    return modules


def _iter_mamba2_modules(model):
    modules, seen = [], set()
    for module in model.modules():
        if id(module) in seen:
            continue
        seen.add(id(module))
        name = type(module).__name__
        if name.endswith("Mamba2Mixer") and hasattr(module, "conv1d") and hasattr(module, "A_log"):
            modules.append(module)
    return modules


def _callable_accepts_named_seq_idx(fn) -> Optional[str]:
    """None if ``seq_idx`` is a named parameter. ``**kwargs``-only stubs are
    rejected so transformers' unused hub fallback is not treated as varlen-ready."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return "kernel signature not introspectable"
    if "seq_idx" in params:
        return None
    return "mamba2 fused kernel does not accept seq_idx"


_MAMBA2_NAMESPACE_SUBSTR = (
    "unsloth_compiled",
    "nemotron",
    "mamba_ssm",
    "modeling_nemotron",
)


def _force_install_mamba2_fused(namespace, wrapped) -> None:
    """Overwrite fused names in a module/function dict, including stale compile-time imports.

    Unsloth's compiler does ``from modeling import mamba_split_conv1d_scan_combined``
    when it writes ``unsloth_compiled_cache``. That binding is whatever the
    modeling module had *at compile import*, often ``None`` or a hub stub, while
    mixer ``__init__`` later stores the real kernel only on the modeling module.
    Compiled ``cuda_kernels_forward`` still LOAD_GLOBALs the stale copy.
    """
    if wrapped is None or not isinstance(namespace, dict):
        return
    for name in _MAMBA2_FUSED_NAMES:
        namespace[name] = wrapped
    if "is_fast_path_available" in namespace:
        namespace["is_fast_path_available"] = True
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
    """One-line summary of why the packed dispatch handshake failed."""
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
    """Invoke ``fn`` with ``attention_mask=None`` and ``cache_params=None``.

    transformers 5.5 only calls ``mamba_split_conv1d_scan_combined`` when the
    mask is all-ones *and* ``cache_params is None``. Packed 8k batches still
    carry pad zeros, and Nemotron-H layers pass an empty ``Cache`` into the
    mixer, so the fused wrapper never runs.
    """
    kwargs = dict(kwargs)
    try:
        sig = inspect.signature(fn)
        names = sig.parameters
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in names.values())
        if not has_var_kw and "attention_mask" not in names and "cache_params" not in names:
            return fn(*args, **kwargs)
        bound = sig.bind_partial(*args, **kwargs)
        if "attention_mask" in names or has_var_kw:
            bound.arguments["attention_mask"] = None
        if "cache_params" in names or has_var_kw:
            bound.arguments["cache_params"] = None
        return fn(*bound.args, **bound.kwargs)
    except (TypeError, ValueError):
        kwargs["attention_mask"] = None
        kwargs["cache_params"] = None
        return fn(*args, **kwargs)


_MAMBA2_MASK_CLEAR_NAMES = ("cuda_kernels_forward",)


def _is_mamba2_mask_clear_name(name: str) -> bool:
    if name in _MAMBA2_MASK_CLEAR_NAMES:
        return True
    if name.endswith("Mamba2Mixer_cuda_kernels_forward"):
        return True
    if name.endswith("Mamba2Mixer_forward") and not name.endswith("torch_forward"):
        return True
    return False


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


def _install_mamba2_mask_clear(namespace, varlen_getter) -> None:
    """Packed pad tokens keep 0s in ``attention_mask``, and Nemotron-H passes
    an empty ``Cache`` into the mixer. transformers 5.5 only calls
    ``mamba_split_conv1d_scan_combined`` when the mask is all-ones *and*
    ``cache_params is None``.
    """
    if not isinstance(namespace, dict):
        return
    for name, value in list(namespace.items()):
        if not _is_mamba2_mask_clear_name(name) or not callable(value):
            continue
        namespace[name] = _wrap_clear_attention_mask(value, varlen_getter)


_MAMBA2_FALLBACK_SEQ_IDX = {
    "mamba_chunk_scan_combined": "_unsloth_varlen_scan_hit",
    "causal_conv1d_fn": "_unsloth_varlen_conv_hit",
}


def _install_mamba2_seq_idx_fallbacks(namespace, mixers, varlen_slot) -> None:
    """Wrap the non-fused cuda path so packed seq_idx still resets state."""
    if not isinstance(namespace, dict):
        return
    for name, hit_attr in _MAMBA2_FALLBACK_SEQ_IDX.items():
        fn = namespace.get(name)
        if not callable(fn) or getattr(fn, "_unsloth_varlen_seq_idx_wrapped", False):
            continue
        namespace[name] = _wrap_mamba2_seq_idx_call(
            fn,
            mixers,
            varlen_slot = varlen_slot,
            hit_attr = hit_attr,
        )


def _resolve_mamba2_fused(module):
    """Locate the fused conv1d+scan kernel this mixer will call.

    Prefers an instance attribute (tests / some vendor copies), then the
    mixer's owning module, then ``mamba_ssm``.
    """
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
        conv = getattr(module, "_unsloth_varlen_orig_conv", None) or getattr(
            module,
            "causal_conv1d_fn",
            None,
        )
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
            if "seq_idx" not in inspect.signature(conv).parameters:
                return "conv kernel does not accept seq_idx"
            if "cu_seqlens" not in inspect.signature(scan).parameters:
                return "scan kernel does not accept cu_seqlens"
        except (TypeError, ValueError):
            return "kernel signature not introspectable"
    return None


def _mamba2_varlen_kernels_available(mamba2_modules) -> Optional[str]:
    """None if every Mamba2 mixer can take ``seq_idx`` on its fused kernel."""
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
    return None


def _hybrid_varlen_dispatched(module) -> bool:
    if type(module).__name__.endswith("Mamba2Mixer"):
        if getattr(module, "_unsloth_varlen_fused_hit", False):
            return True
        # transformers' non-fused cuda path: causal_conv1d_fn + mamba_chunk_scan_combined
        return bool(
            getattr(module, "_unsloth_varlen_conv_hit", False)
            and getattr(module, "_unsloth_varlen_scan_hit", False)
        )
    return bool(
        getattr(module, "_unsloth_varlen_conv_hit", False)
        and getattr(module, "_unsloth_varlen_scan_hit", False)
    )


def _varlen_seq_idx_applies(seq_idx, args, kwargs) -> bool:
    """True when ``seq_idx`` covers exactly the tokens this kernel call sees.

    ``generate()`` bypasses the wrapped ``model.forward``, so mixers can still
    hold the last training step's boundaries while the kernel is handed a
    prompt of a different length. Injecting then trips the kernel's own
    ``seq_idx must have shape (batch_size, seqlen)`` check.
    """
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
    if tensor is None or tensor.dim() < 2:
        return False
    # Fused/scan kernels take (batch, seqlen, ...); causal_conv1d_fn takes
    # (batch, dim, seqlen).
    return total in (tensor.shape[1], tensor.shape[-1])


def _wrap_mamba2_seq_idx_call(
    orig,
    mixers,
    *,
    varlen_slot = None,
    hit_attr = "_unsloth_varlen_fused_hit",
):
    """Inject packed ``seq_idx`` into a Mamba2 conv/scan kernel and mark dispatch.

    Returns ``orig`` unchanged when it is already a wrapper: Nemotron-H has one
    mixer per layer sharing a single kernel name, so re-wrapping per module
    would nest ~one frame per layer and blow the recursion limit on the first
    packed forward.
    """
    if getattr(orig, "_unsloth_varlen_seq_idx_wrapped", False):
        return orig

    @wraps(orig)
    def wrapped(*args, **kwargs):
        varlen = varlen_slot[0] if varlen_slot else None
        if varlen is None:
            # Gradient-checkpoint recompute runs during backward, outside the
            # model.forward wrapper, so the per-mixer stash is the only source.
            donors = [m for m in mixers if getattr(m, "_unsloth_varlen", None) is not None]
            if donors:
                varlen = donors[0]._unsloth_varlen
        if varlen is not None:
            if kwargs.get("seq_idx") is None and _varlen_seq_idx_applies(varlen[1], args, kwargs):
                kwargs["seq_idx"] = varlen[1]
            if kwargs.get("seq_idx") is not None:
                for mixer in mixers:
                    setattr(mixer, hit_attr, True)
        return orig(*args, **kwargs)

    wrapped._unsloth_varlen_seq_idx_wrapped = True
    return wrapped


def _wrap_mamba2_fused_call(
    orig,
    mixers,
    *,
    on_module = None,
    attr_name = None,
    varlen_slot = None,
):
    """Inject packed ``seq_idx`` into the fused conv1d+scan kernel.

    ``mixers`` is the list of Mamba2 modules sharing this kernel. On a packed
    forward they all stash the same boundary metadata. ``varlen_slot`` is a
    1-element list set by the outer model.forward wrapper so injection still
    works if PEFT/compile replaced mixer instances after patch time.
    """
    fused_fn = _wrap_mamba2_seq_idx_call(
        orig,
        mixers,
        varlen_slot = varlen_slot,
        hit_attr = "_unsloth_varlen_fused_hit",
    )
    if on_module is not None and attr_name is not None:
        setattr(on_module, attr_name, fused_fn)
    return fused_fn


def _rebind_mamba2_fused_aliases(orig, wrapped) -> None:
    """Point every imported fused-kernel name at ``wrapped``.

    Unsloth's Fast Nemotron-H compile copies the transformers mixer source into
    ``unsloth_compiled_cache``, which imports ``mamba_split_conv1d_scan_combined``
    into its own module globals. Wrapping only the transformers modeling import
    leaves that copy bound to the original, so packed forwards never reach the
    varlen wrapper. A LOAD_GLOBAL resolves through the module dict at call time,
    so reassigning the name here is enough; anything this misses is caught by the
    dispatch handshake rather than silently training unpacked.
    """
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


def _wrap_mamba2_mixer_forward(module, varlen_getter = None):
    if getattr(module, "_unsloth_mamba2_forward_wrapped", False):
        return
    forward_orig = module.forward
    cuda_orig = getattr(module, "cuda_kernels_forward", None)
    # Only pass seq_idx through mixer.forward when it names the parameter. A
    # forward that merely collects **kwargs may splat them into the kernel
    # while also passing seq_idx itself, which is a duplicate-keyword
    # TypeError. The kernel wrappers inject it in that case.
    forward_names_seq_idx = False
    try:
        forward_names_seq_idx = "seq_idx" in inspect.signature(forward_orig).parameters
    except (TypeError, ValueError):
        pass

    def _packed():
        if varlen_getter is not None:
            varlen = varlen_getter()
            if varlen is not None:
                return varlen
        return getattr(module, "_unsloth_varlen", None)

    @wraps(forward_orig)
    def mixer_forward(*args, **kwargs):
        varlen = _packed()
        if varlen is not None:
            if (
                forward_names_seq_idx
                and kwargs.get("seq_idx") is None
                and _varlen_seq_idx_applies(varlen[1], args, kwargs)
            ):
                kwargs["seq_idx"] = varlen[1]
            return _call_as_packed_mamba2_prefill(forward_orig, args, kwargs)
        return forward_orig(*args, **kwargs)

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
    if psl is not None and getattr(psl, "numel", lambda: 1)() > 0:  # skip empty (no max())
        info = get_packed_info_from_kwargs(kwargs, device)
        if info is not None:
            _, cu_seqlens, _ = info
            built = _seq_idx_from_cu_seqlens(cu_seqlens, total)
            if built is not None:
                return built
    return _varlen_from_position_ids(kwargs.get("position_ids"))


def patch_hybrid_linear_attention_varlen(model) -> bool:
    """Feed seq_idx / cu_seqlens to hybrid mixers so packing resets state.

    Gated-delta: wrap ``causal_conv1d_fn`` + ``chunk_gated_delta_rule``.
    Mamba2: wrap ``mamba2_split_conv1d_scan_combined`` and inject ``seq_idx``
    into mixer.forward kwargs (transformers already forwards ``**kwargs`` into
    the fused kernel). Gated by ``UNSLOTH_EXPERIMENTAL_HYBRID_PACKING`` and
    fail-closed. Returns True when the varlen path is active.
    Idempotent: repeat calls on an already-patched model return True.
    """
    if not _hybrid_packing_enabled():
        return False
    gated_delta_modules = _iter_gated_delta_modules(model)
    mamba2_modules = _iter_mamba2_modules(model)
    hybrid_modules = gated_delta_modules + mamba2_modules
    if not hybrid_modules:
        return _hybrid_reject("no gated-delta or mamba2 modules found")

    # Idempotency: an already fully-patched model stays active without re-validation.
    if getattr(model, "_unsloth_varlen_forward_wrapped", False) and all(
        getattr(m, "_unsloth_varlen_wrapped", False) for m in hybrid_modules
    ):
        return True

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
        conv_orig, scan_orig = module.causal_conv1d_fn, module.chunk_gated_delta_rule
        module._unsloth_varlen_orig_conv = conv_orig
        module._unsloth_varlen_orig_scan = scan_orig

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

        module.causal_conv1d_fn = conv_fn
        module.chunk_gated_delta_rule = scan_fn
        module._unsloth_varlen = None
        module._unsloth_varlen_wrapped = True

    wrapped_fused: dict[int, Any] = {}
    varlen_slot: list = [None]

    def _ensure_mamba2_fused_wrapped(fn):
        if fn is None:
            return None
        if getattr(fn, "_unsloth_varlen_seq_idx_wrapped", False):
            return fn
        wrapped = wrapped_fused.get(id(fn))
        if wrapped is None:
            wrapped = _wrap_mamba2_fused_call(fn, mamba2_modules, varlen_slot = varlen_slot)
            wrapped_fused[id(fn)] = wrapped
            _rebind_mamba2_fused_aliases(fn, wrapped)
        return wrapped

    for module in mamba2_modules:
        if getattr(module, "_unsloth_varlen_wrapped", False):
            continue
        fn, loc = _resolve_mamba2_fused(module)
        wrapped = _ensure_mamba2_fused_wrapped(fn)
        kind = loc[0] if loc is not None else None
        if kind == "instance" and wrapped is not None:
            name = loc[2]
            module._unsloth_varlen_orig_fused = fn
            if name is not None:
                setattr(module, name, wrapped)
            else:
                module.mamba2_split_conv1d_scan_combined = wrapped
        elif kind == "modeling" and wrapped is not None:
            modeling, name = loc[1], loc[2]
            setattr(modeling, name, wrapped)
            modeling._unsloth_mamba2_fused_wrapped = True
        elif kind == "ssm":
            module._unsloth_varlen_orig_fused = fn
        elif kind == "globals" and wrapped is not None:
            globs, name = loc[1], loc[2]
            globs[name] = wrapped
        _wrap_mamba2_mixer_forward(module, varlen_getter = lambda: varlen_slot[0])
        module._unsloth_varlen = None
        module._unsloth_varlen_wrapped = True
    if mamba2_modules:
        try:
            from mamba_ssm.ops.triton.ssd_combined import (  # type: ignore
                mamba_split_conv1d_scan_combined as _ssm_fused,
            )
        except Exception:
            _ssm_fused = None
        wrapped_real = _ensure_mamba2_fused_wrapped(_ssm_fused)
        if wrapped_real is None:
            wrapped_real = wrapped

        def packed():
            return varlen_slot[0]

        for ns in _iter_mamba2_install_namespaces(mamba2_modules):
            _force_install_mamba2_fused(ns, wrapped_real)
            _install_mamba2_mask_clear(ns, packed)
            _install_mamba2_seq_idx_fallbacks(ns, mamba2_modules, varlen_slot)

    # Refresh the boundary stash on the outermost forward (once per step, outside
    # gradient-checkpoint recompute, so it stays valid for recomputed inner
    # forwards). Read from both positional and keyword args via the bound signature.
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
            varlen_slot[0] = varlen
            try:
                out = forward_orig(*args, **kwargs)
            finally:
                varlen_slot[0] = None
            # Runtime dispatch handshake: on the first packed forward, confirm the
            # load-bearing kernels ran. Gated-delta needs conv+scan; Mamba2 needs
            # the fused conv1d+scan. Partial dispatch leaves cross-sequence
            # contamination on a flattened batch with no padded recovery.
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
                        "varlen conv/scan wrappers were not both invoked for "
                        f"{sorted(set(missing))}. Unset UNSLOTH_EXPERIMENTAL_HYBRID_PACKING "
                        "to train these models on the padded path.\n"
                        + _mamba2_handshake_debug(
                            [m for m in hybrid_modules if type(m).__name__.endswith("Mamba2Mixer")]
                            or hybrid_modules
                        )
                    )
            return out

        model.forward = forward_with_varlen
        model._unsloth_varlen_forward_wrapped = True

    _wrap_generate_clears_varlen(model, hybrid_modules)
    return True


def _wrap_generate_clears_varlen(model, hybrid_modules) -> None:
    """Drop the packed boundaries for the duration of ``generate()``.

    ``generate()`` reaches the decoder without going through the wrapped
    ``model.forward``, so the per-mixer stash would otherwise still hold the
    last training step's ``seq_idx`` while the kernels see a prompt.
    """
    generate_orig = getattr(model, "generate", None)
    if not callable(generate_orig) or getattr(model, "_unsloth_varlen_generate_wrapped", False):
        return

    @wraps(generate_orig)
    def generate_without_varlen(*args, **kwargs):
        for module in hybrid_modules:
            module._unsloth_varlen = None
        return generate_orig(*args, **kwargs)

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


def build_xformers_block_causal_mask(
    seq_info: Optional[Tuple[torch.Tensor, torch.Tensor, int]],
    *,
    sliding_window: Optional[int] = None,
    base_mask: Optional[Any] = None,
):
    if _XFormersBlockMask is None:
        return None
    if seq_info is not None:
        seq_lengths, _, _ = seq_info
        # Cache the mask to avoid repeated D2H sync across layers
        device = seq_lengths.device
        params = (sliding_window,)
        entry = _XFORMERS_BLOCK_MASK_CACHE.get(device)
        if entry is not None and entry["seq_lengths"] is seq_lengths and entry["params"] == params:
            return entry["mask"]

        lengths_tensor = seq_lengths.to("cpu", torch.int32)
        if lengths_tensor.numel() == 0:
            return None
        lengths = tuple(int(x) for x in lengths_tensor.tolist())
        mask = _get_cached_block_mask(lengths, sliding_window, device)

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


def build_sdpa_packed_attention_mask(
    seq_info: Tuple[torch.Tensor, torch.Tensor, int],
    *,
    dtype: torch.dtype,
    device: torch.device,
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    seq_lengths, _, _ = seq_info

    params = (dtype, sliding_window)
    entry = _SDPA_MASK_CACHE.get(device)
    if entry is not None and entry["seq_lengths"] is seq_lengths and entry["params"] == params:
        return entry["mask"]

    total_tokens = int(seq_lengths.sum().item())
    mask = torch.full(
        (total_tokens, total_tokens),
        float("-inf"),
        dtype = dtype,
        device = device,
    )
    offset = 0
    for length in seq_lengths.tolist():
        length = int(length)
        if length <= 0:
            continue
        block = torch.zeros((length, length), dtype = dtype, device = device)
        upper = torch.triu(torch.ones((length, length), device = device), diagonal = 1).bool()
        block = block.masked_fill(upper, float("-inf"))
        if sliding_window is not None and sliding_window > 0 and length > sliding_window:
            idx = torch.arange(length, device = device)
            dist = idx.unsqueeze(1) - idx.unsqueeze(0)
            window_mask = dist >= sliding_window
            block = block.masked_fill(window_mask, float("-inf"))
        mask[offset : offset + length, offset : offset + length] = block
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
    _XFORMERS_BLOCK_MASK_CACHE.clear()


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
    "mask_packed_sequence_boundaries",
    "mask_packed_boundary_labels",
    "clear_packed_caches",
]
