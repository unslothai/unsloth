# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Krea 2 denoises against its valid text rows, not 512 padded ones.

``Krea2Pipeline.get_text_hidden_states`` pads every prompt to ``max_sequence_length`` (512) in the middle of its
template (``[prompt | PAD | suffix]``), and the transformer carries all 512 rows through the text fusion and every
block, beside 4096 image rows at 1024x1024. The padded rows are only ever masked keys: the text rows all sit at the
rotary origin (``prepare_position_ids``), the layerwise text blocks run per token, the refiner and the main blocks mask
them out as keys, and their own outputs are sliced away. Dropping them leaves every kept row's math unchanged.

A forward pre-hook keeps the columns valid for any batch row, padded with masked columns up to a multiple of
``BUCKET`` (a few input shapes per load for the CUDA graphs and the compiled blocks, and the same length for a CFG
pair in the common case), and the matching ``position_ids`` rows. The columns are read once per prompt (one host
read of the mask); every step reuses the trimmed tensors. Not bit-identical to the padded call (attention reduces
over fewer keys), mathematically the same. ``UNSLOTH_DIFFUSION_KREA2_TEXT_TRIM=0`` keeps the padded rows."""

from __future__ import annotations

import os
import weakref
from typing import Any, Optional

TEXT_TRIM_ENV = "UNSLOTH_DIFFUSION_KREA2_TEXT_TRIM"
CLASS_NAME = "Krea2Transformer2DModel"
BUCKET = 128
_HOOK_ATTR = "_unsloth_krea2_text_trim_hook"
_MEMO_ATTR = "_unsloth_krea2_text_trim_memo"
_MEMO_MAX = 4  # the CFG branches of one render, and the last render's
_OFF = ("0", "off", "false", "no")
# forwards that ran on trimmed text, and the text rows they skipped
_CALLS = [0]
_ROWS_SKIPPED = [0]


def enabled() -> bool:
    return (os.environ.get(TEXT_TRIM_ENV) or "").strip().lower() not in _OFF


def call_count() -> int:
    return _CALLS[0]


def rows_skipped() -> int:
    return _ROWS_SKIPPED[0]


def bucket_length(valid: int, total: int) -> int:
    """Rows kept for ``valid`` valid columns of ``total``: the next multiple of ``BUCKET``, never more than ``total``."""
    return min(int(total), max(BUCKET, -(-int(valid) // BUCKET) * BUCKET))


def _version(t: Any) -> Any:
    # Inference tensors have no version counter: keyed on identity alone.
    return "inference" if t.is_inference() else t._version


def _lookup(memo: list, srcs: tuple) -> Optional[Any]:
    versions = tuple(_version(t) for t in srcs)
    for refs, held_versions, value in memo:
        if held_versions == versions and all(r() is t for r, t in zip(refs, srcs)):
            return value
    return None


def _store(memo: list, srcs: tuple, value: Any) -> Any:
    memo.insert(0, (tuple(weakref.ref(t) for t in srcs), tuple(_version(t) for t in srcs), value))
    del memo[_MEMO_MAX:]
    return value


def _columns(mask: Any, text_len: int) -> Optional[tuple]:
    """(column index, every kept column valid for every row) or None when nothing would be dropped. One host read."""
    import torch

    mb = mask.bool()
    keep = mb.any(dim = 0)
    valid = int(keep.sum().item())
    rows = bucket_length(valid, text_len)
    if valid == 0 or rows >= text_len:
        return None
    index = keep.nonzero().squeeze(1)
    if rows > valid:
        # masked columns fill the bucket: masked keys, outputs sliced away, the same as the padding they replace
        index = torch.cat((index, (~keep).nonzero().squeeze(1)[: rows - valid]))
    index = index.sort().values
    return index, bool(mb.index_select(1, index).all().item())


def _trim(
    module: Any, states: Any, mask: Any, position_ids: Any, image_len: int
) -> Optional[tuple]:
    """(states, mask or None, position_ids) on the kept rows, or None to run the call as given."""
    memo = module.__dict__.setdefault(_MEMO_ATTR, [])
    srcs = (states, mask, position_ids)
    hit = _lookup(memo, srcs)
    if hit is not None:
        return hit or None
    text_len = int(states.shape[1])
    plan = _columns(mask, text_len)
    if plan is not None and not bool((position_ids[:text_len] == 0).all().item()):
        plan = None  # text rows off the rotary origin: dropping one would move the others' phases
    if plan is None:
        _store(memo, srcs, ())
        return None
    index, all_valid = plan
    rows = int(index.shape[0])
    pos_key = (position_ids,)
    pos_memo = module.__dict__.setdefault(_MEMO_ATTR + "_pos", [])
    held = _lookup(pos_memo, pos_key)
    pos = held.get(rows) if isinstance(held, dict) else None
    if pos is None:
        import torch

        pos = torch.cat((position_ids[:rows], position_ids[text_len:]))
        if not isinstance(held, dict):
            held = _store(pos_memo, pos_key, {})
        # one tensor per length: a CFG pair passes the same object, which the CFG router batches without a compare
        held[rows] = pos
    out = (
        states.index_select(1, index),
        None if all_valid else mask.index_select(1, index),
        pos,
        text_len - rows,
    )
    return _store(memo, srcs, out)


def _pre_hook(module: Any, args: tuple, kwargs: dict) -> tuple:
    """Keyword calls only (the pipeline's): positional text would collide with the swapped keywords."""
    try:
        import torch

        if args or not enabled():
            return args, kwargs
        states = kwargs.get("encoder_hidden_states")
        mask = kwargs.get("encoder_attention_mask")
        pos = kwargs.get("position_ids")
        hidden = kwargs.get("hidden_states")
        if not all(torch.is_tensor(t) for t in (states, mask, pos, hidden)):
            return args, kwargs
        if (
            states.dim() != 4
            or mask.dim() != 2
            or pos.dim() != 2
            or tuple(mask.shape) != tuple(states.shape[:2])
            or pos.shape[0] != states.shape[1] + hidden.shape[1]
        ):
            return args, kwargs
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return args, kwargs
        out = _trim(module, states, mask, pos, int(hidden.shape[1]))
        if out is None:
            return args, kwargs
        kwargs = dict(kwargs)
        (
            kwargs["encoder_hidden_states"],
            kwargs["encoder_attention_mask"],
            kwargs["position_ids"],
            skipped,
        ) = out
        _CALLS[0] += 1
        _ROWS_SKIPPED[0] += int(skipped)
        return args, kwargs
    except Exception:  # noqa: BLE001 - optimisation only: the padded call still runs
        return args, kwargs


def installed(transformer: Any) -> bool:
    return getattr(transformer, _HOOK_ATTR, None) is not None


def uninstall(transformer: Any) -> None:
    handle = getattr(transformer, _HOOK_ATTR, None)
    if handle is not None:
        handle.remove()
        try:
            delattr(transformer, _HOOK_ATTR)
        except AttributeError:
            pass
    if transformer is not None:
        transformer.__dict__.pop(_MEMO_ATTR, None)
        transformer.__dict__.pop(_MEMO_ATTR + "_pos", None)


def install_for_pipe(
    pipe: Any,
    logger: Any = None,
    explicit_off: bool = False,
) -> bool:
    """Per transformer instance (dies with the pipe). An explicit speed ``off`` load runs the padded rows."""
    transformer = getattr(pipe, "transformer", None)
    if type(transformer).__name__ != CLASS_NAME:
        return False
    if explicit_off or not enabled():
        uninstall(transformer)
        return False
    if installed(transformer):
        return True
    try:
        handle = transformer.register_forward_pre_hook(_pre_hook, with_kwargs = True)
    except Exception as exc:  # noqa: BLE001 - optimisation only
        if logger is not None:
            logger.warning("diffusion.krea2: text trim unavailable: %s", exc)
        return False
    object.__setattr__(transformer, _HOOK_ATTR, handle)
    if logger is not None:
        logger.info(
            "diffusion.krea2: text rows trimmed to the valid prompt (multiples of %d of 512)",
            BUCKET,
        )
    return True
