# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Capture-safe forwards for denoisers whose stock forward cannot be recorded into a CUDA graph.

``diffusion_cuda_graph.GraphedForward`` records ONE whole denoiser forward. A forward that makes the
host wait on the device cannot be recorded: CUDA refuses the call with
``cudaErrorStreamCaptureUnsupported`` and ``capture_end`` then reports
``cudaErrorStreamCaptureInvalidated``, so the graph layer poisons the wrapper and the load runs eager.

HunyuanImage-2.1 is such a forward. ``HunyuanImageTransformer2DModel.forward`` merges its two text
streams (Qwen2.5-VL mllm and ByT5 glyph) per batch element with boolean-mask indexing::

    new_encoder_hidden_states.append(
        torch.cat(
            [
                text_2[text_mask_2],  # valid byt5
                text[text_mask],  # valid mllm
                text_2[~text_mask_2],  # invalid byt5
                text[~text_mask],  # invalid mllm
            ],
            dim=0,
        )
    )

``tensor[bool_mask]`` runs ``nonzero`` and copies the count to the host to size its output: a
device-to-host sync on every one of the eight index calls, every forward. Identical in diffusers
0.36.0 through 0.41.

The merge is a PERMUTATION, not a data-dependent shape: the four pieces always add back up to the
two streams' full lengths. Concatenating ``[byt5 ; mllm]`` and stable-sorting on "is padding" puts
the valid byt5 tokens first, then the valid mllm tokens, then the padded byt5, then the padded mllm,
each in its original order, which is exactly the order above. ``argsort`` and ``gather`` stay on the
device, so the rewritten forward records, and because it only moves rows it is bit-identical to the
stock one. Only the graph wrapper uses it; an ungraphed load keeps the stock forward.

The rewrite takes the INSTALLED forward's own source and swaps just that block, so everything else
(LoRA scaling, which moved from inline code in 0.36 to a decorator in 0.37, checkpointing, the
unpatchify) stays whatever the installed diffusers ships. The block is matched line for line; if it
changed, nothing is rewritten and the graph layer declines the load with the reason.

Kill switch: ``UNSLOTH_DIFFUSION_CAPTURE_SAFE=0`` (the graph is then declined for these classes).
"""

from __future__ import annotations

import inspect
import linecache
import os
import sys
import textwrap
import threading
from typing import Any, Callable, Optional

CAPTURE_SAFE_ENV = "UNSLOTH_DIFFUSION_CAPTURE_SAFE"

_HUNYUANIMAGE_CLS = "HunyuanImageTransformer2DModel"

# The stock merge block, stripped line by line (blank lines dropped). Order of the four pieces is
# what the permutation reproduces, so the match is exact rather than a needle search.
_HUNYUANIMAGE_MERGE_BLOCK: tuple[str, ...] = (
    "# reorder and combine text tokens: combine valid tokens first, then padding",
    "new_encoder_hidden_states = []",
    "new_encoder_attention_mask = []",
    "for text, text_mask, text_2, text_mask_2 in zip(",
    "encoder_hidden_states, encoder_attention_mask, encoder_hidden_states_2, encoder_attention_mask_2",
    "):",
    "# Concatenate: [valid_mllm, valid_byt5, invalid_mllm, invalid_byt5]",
    "new_encoder_hidden_states.append(",
    "torch.cat(",
    "[",
    "text_2[text_mask_2],  # valid byt5",
    "text[text_mask],  # valid mllm",
    "text_2[~text_mask_2],  # invalid byt5",
    "text[~text_mask],  # invalid mllm",
    "],",
    "dim=0,",
    ")",
    ")",
    "# Apply same reordering to attention masks",
    "new_encoder_attention_mask.append(",
    "torch.cat(",
    "[",
    "text_mask_2[text_mask_2],",
    "text_mask[text_mask],",
    "text_mask_2[~text_mask_2],",
    "text_mask[~text_mask],",
    "],",
    "dim=0,",
    ")",
    ")",
    "encoder_hidden_states = torch.stack(new_encoder_hidden_states)",
    "encoder_attention_mask = torch.stack(new_encoder_attention_mask)",
)

_MERGE_HELPER_NAME = "_unsloth_merge_text_streams"
_MERGE_CALL = (
    f"encoder_hidden_states, encoder_attention_mask = {_MERGE_HELPER_NAME}("
    "encoder_hidden_states, encoder_attention_mask, encoder_hidden_states_2, encoder_attention_mask_2)"
)


def capture_safe_disabled() -> bool:
    return (os.environ.get(CAPTURE_SAFE_ENV) or "").strip().lower() in ("0", "off", "false", "no")


def merge_text_streams(
    encoder_hidden_states: Any,
    encoder_attention_mask: Any,
    encoder_hidden_states_2: Any,
    encoder_attention_mask_2: Any,
) -> tuple[Any, Any]:
    """HunyuanImage's text-stream merge without a host sync; bit-identical to the stock loop.

    Takes the embedded mllm stream ``[B, S1, D]`` with its bool mask ``[B, S1]`` and the embedded
    byt5 stream ``[B, S2, D]`` with its bool mask ``[B, S2]``; returns ``[B, S2 + S1, D]`` ordered
    [valid byt5, valid mllm, padded byt5, padded mllm] and the matching mask (all True, then all
    False), per batch element. ``torch.cat`` promotes dtypes exactly as the stock per-row cat does.
    """
    import torch

    states = torch.cat([encoder_hidden_states_2, encoder_hidden_states], dim = 1)
    mask = torch.cat([encoder_attention_mask_2, encoder_attention_mask], dim = 1)
    # Key 0 for a valid token, 1 for padding; a STABLE sort keeps byt5 ahead of mllm and each
    # stream's own order within both groups.
    order = torch.argsort((~mask).to(torch.uint8), dim = 1, stable = True)
    states = torch.gather(states, 1, order.unsqueeze(-1).expand(-1, -1, states.shape[-1]))
    mask = torch.gather(mask, 1, order)
    return states, mask


def _find_block(lines: list[str], block: tuple[str, ...]) -> Optional[tuple[int, int]]:
    """``(start, end)`` line indices (end exclusive) of ``block`` in ``lines``, blank lines ignored.

    None unless it occurs exactly once."""
    starts = [i for i, line in enumerate(lines) if line.strip() == block[0]]
    found: list[tuple[int, int]] = []
    for start in starts:
        want = 0
        i = start
        while i < len(lines) and want < len(block):
            text = lines[i].strip()
            if text:
                if text != block[want]:
                    break
                want += 1
            i += 1
        if want == len(block):
            found.append((start, i))
    return found[0] if len(found) == 1 else None


def _signature_key(fn: Callable) -> Optional[tuple]:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return None
    return tuple((p.name, p.kind, repr(p.default)) for p in params)


def _rewrite_hunyuanimage(forward: Callable) -> Callable:
    """Rebuild ``forward`` from its own source with the merge block swapped for the helper call.

    Raises ``RuntimeError`` naming what did not match, so the reason reaches the status payload."""
    target = inspect.unwrap(forward)
    module = sys.modules.get(getattr(target, "__module__", "") or "")
    if module is None:
        raise RuntimeError("its defining module is not imported")
    try:
        source = textwrap.dedent(inspect.getsource(target))
    except (OSError, TypeError) as exc:
        raise RuntimeError(f"its source is unreadable ({exc})") from None
    lines = source.splitlines()
    span = _find_block(lines, _HUNYUANIMAGE_MERGE_BLOCK)
    if span is None:
        raise RuntimeError("the boolean-mask text merge block changed in this diffusers version")
    start, end = span
    indent = lines[start][: len(lines[start]) - len(lines[start].lstrip())]
    new_lines = lines[:start] + [indent + _MERGE_CALL] + lines[end:]
    # A source-level decorator (``@apply_lora_scale`` from 0.37) re-applies when the def executes,
    # so the rebuilt function carries the installed version's LoRA handling unchanged.
    new_source = "\n".join(new_lines) + "\n"
    filename = f"<unsloth capture-safe {getattr(target, '__qualname__', 'forward')}>"
    namespace = dict(vars(module))
    namespace[_MERGE_HELPER_NAME] = merge_text_streams
    code = compile(new_source, filename, "exec")
    exec(code, namespace)  # noqa: S102 - the installed diffusers' own source with one block replaced
    rebuilt = namespace.get(target.__name__)
    if not callable(rebuilt):
        raise RuntimeError("the rebuilt source did not define the forward")
    # Tracebacks and inspect.getsource resolve through linecache.
    linecache.cache[filename] = (len(new_source), None, new_source.splitlines(True), filename)
    if _signature_key(rebuilt) != _signature_key(forward):
        raise RuntimeError("the rebuilt forward's signature differs from the stock one")
    rebuilt.__unsloth_capture_safe__ = True
    return rebuilt


# class name -> (why the stock forward cannot be captured, rewriter)
_REWRITES: dict[str, tuple[str, Callable[[Callable], Callable]]] = {
    _HUNYUANIMAGE_CLS: (
        "boolean-mask indexing in its text-stream merge syncs the host",
        _rewrite_hunyuanimage,
    ),
}

# class name -> why its stock call cannot be captured at all, with no rewrite to offer. Declined at
# load, so status says "off" with the reason instead of arming a wrapper that refuses every step.
_UNCAPTURABLE: dict[str, str] = {
    # The pipeline passes its prefix KV cache (a QwenImage21KVCache) on every step, and the forward
    # syncs the host (repeat_interleave, boolean indexing, nonzero, .tolist) before its first block.
    "QwenImage21Transformer2DModel": "its pipeline passes the prefix KV cache as a Python object "
    "on every step and its forward syncs the host",
}

_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()


def _defining_class(cls: type) -> Optional[type]:
    """The class in ``cls``'s MRO that defines the ``forward`` instances of ``cls`` run."""
    for klass in getattr(cls, "__mro__", ()):
        if "forward" in vars(klass):
            return klass
    return None


def resolve(cls: type) -> tuple[Optional[Callable], Optional[str]]:
    """``(forward, None)``: an unbound capture-safe replacement for ``cls.forward``.
    ``(None, None)``: nothing known to be unsafe, capture the stock forward.
    ``(None, reason)``: the stock forward is known not to capture and no rewrite applied.

    Never raises. Cheap for an unknown class: a name lookup, no torch import."""
    owner = _defining_class(cls)
    if owner is not None and owner.__name__ in _UNCAPTURABLE and owner.__name__ not in _REWRITES:
        return None, f"{owner.__name__} forward is not capture-safe ({_UNCAPTURABLE[owner.__name__]})"
    if owner is None or owner.__name__ not in _REWRITES:
        return None, None
    why, rewrite = _REWRITES[owner.__name__]
    if capture_safe_disabled():
        return (
            None,
            f"{owner.__name__} forward is not capture-safe ({why}); rewrite disabled by {CAPTURE_SAFE_ENV}",
        )
    forward = vars(owner)["forward"]
    key = (owner, forward)
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
    if cached is not None:
        return cached
    try:
        result: tuple[Optional[Callable], Optional[str]] = (rewrite(forward), None)
    except Exception as exc:  # noqa: BLE001 - an unrewritable forward declines the graph, never fails a load
        result = (
            None,
            f"{owner.__name__} forward is not capture-safe ({why}) and was not rewritten: {exc}",
        )
    with _CACHE_LOCK:
        _CACHE[key] = result
    return result
