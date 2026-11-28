# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Counters for how image requests share pipeline calls, reported on status as ``batch_trace``.

Always on and O(1) per request: admissions, the idle direct dispatch, joins, realized call sizes, why a request
rendered alone, and the rows of every pipe call. Denoiser-level counts (invocations, rows, block executions) need
forward hooks, so they run only with ``UNSLOTH_STUDIO_IMAGE_BATCH_TRACE=1``: a pipeline that loops over the rows of
one call shows up there as invocations and block executions that grow with the rows."""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Any, Iterator

TRACE_ENV = "UNSLOTH_STUDIO_IMAGE_BATCH_TRACE"

_LOCK = threading.Lock()
_COUNTS: dict[str, int] = {}
_CALL_SIZES: dict[int, int] = {}
_PIPE_ROWS: dict[int, int] = {}
_FALLBACKS: dict[str, int] = {}
_DENOISER: dict[str, int] = {}


def hooks_enabled() -> bool:
    return (os.environ.get(TRACE_ENV) or "").strip().lower() in ("1", "true", "yes", "on")


def count(name: str, by: int = 1) -> None:
    with _LOCK:
        _COUNTS[name] = _COUNTS.get(name, 0) + int(by)


def fallback(reason: str) -> None:
    """A request that could have shared a call rendered alone, and why."""
    with _LOCK:
        _FALLBACKS[reason] = _FALLBACKS.get(reason, 0) + 1


def coalesced_call(requests: int, images: int) -> None:
    with _LOCK:
        _COUNTS["coalesced_calls"] = _COUNTS.get("coalesced_calls", 0) + 1
        _COUNTS["coalesced_requests"] = _COUNTS.get("coalesced_requests", 0) + int(requests)
        _CALL_SIZES[int(images)] = _CALL_SIZES.get(int(images), 0) + 1


def pipe_call(rows: int) -> None:
    with _LOCK:
        _COUNTS["pipe_calls"] = _COUNTS.get("pipe_calls", 0) + 1
        _COUNTS["pipe_rows"] = _COUNTS.get("pipe_rows", 0) + int(rows)
        _PIPE_ROWS[int(rows)] = _PIPE_ROWS.get(int(rows), 0) + 1


def snapshot() -> dict[str, Any]:
    with _LOCK:
        return {
            "counts": dict(_COUNTS),
            "call_sizes": {str(k): v for k, v in sorted(_CALL_SIZES.items())},
            "pipe_rows": {str(k): v for k, v in sorted(_PIPE_ROWS.items())},
            "fallbacks": dict(_FALLBACKS),
            "denoiser": dict(_DENOISER),
            "hooks": hooks_enabled(),
        }


def reset() -> None:
    with _LOCK:
        for table in (_COUNTS, _CALL_SIZES, _PIPE_ROWS, _FALLBACKS, _DENOISER):
            table.clear()


def _rows_of(args: tuple, kwargs: dict) -> int:
    for value in (kwargs.get("hidden_states"), kwargs.get("sample"), kwargs.get("x"), *args[:1]):
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) > 0:
            return int(shape[0])
    return 0


def _block_lists(module: Any) -> list:
    """The repeated-block containers of a denoiser (``transformer_blocks``, ``single_transformer_blocks``, ...)."""
    try:
        import torch.nn as nn
    except Exception:  # noqa: BLE001 - no torch: nothing to hook
        return []
    found = []
    for name, child in getattr(module, "_modules", {}).items():
        if isinstance(child, nn.ModuleList) and len(child) and "block" in name:
            found.append(child)
    return found


@contextmanager
def denoiser_calls(pipe: Any, rows: int) -> Iterator[None]:
    """Count this pipe call (always) and, with the trace env set, its denoiser invocations / rows / block runs."""
    pipe_call(rows)
    if not hooks_enabled():
        yield
        return
    handles = []
    local = {"invocations": 0, "rows": 0, "block_runs": 0}

    def on_denoiser(_module: Any, args: tuple, kwargs: dict) -> None:
        local["invocations"] += 1
        local["rows"] += _rows_of(args, kwargs)

    def on_block(_module: Any, _args: tuple) -> None:
        local["block_runs"] += 1

    for name in ("transformer", "transformer_2", "unet"):
        module = getattr(pipe, name, None)
        if module is None or not hasattr(module, "register_forward_pre_hook"):
            continue
        try:
            handles.append(module.register_forward_pre_hook(on_denoiser, with_kwargs = True))
            for blocks in _block_lists(module):
                for block in blocks:
                    handles.append(block.register_forward_pre_hook(on_block))
        except Exception:  # noqa: BLE001 - a module that refuses hooks is simply not traced
            continue
    try:
        yield
    finally:
        for handle in handles:
            try:
                handle.remove()
            except Exception:  # noqa: BLE001
                pass
        with _LOCK:
            _DENOISER["pipe_calls"] = _DENOISER.get("pipe_calls", 0) + 1
            _DENOISER["requested_rows"] = _DENOISER.get("requested_rows", 0) + int(rows)
            for key, value in local.items():
                _DENOISER[key] = _DENOISER.get(key, 0) + value
            last = f"last_call_rows_{int(rows)}"
            _DENOISER[last + "_invocations"] = local["invocations"]
            _DENOISER[last + "_block_runs"] = local["block_runs"]
