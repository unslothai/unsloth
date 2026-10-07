# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Decision runtime dispatch, keeping legacy Laya and the owned Clef worker exclusive."""

from __future__ import annotations

import threading
from typing import Any

from . import catalog, laya_runtime

Unavailable = laya_runtime.Unavailable
_gate = threading.Lock()
_admission = threading.BoundedSemaphore(8)
_fallback_reason = None


def _clef():
    from . import owned_runtime
    return owned_runtime


def select_checkpoint(
    checkpoint,
    *,
    images = False,
    questions = None,
    preference = None,
):
    from utils.systemone_settings import get_backend
    from .native_worker import native_availability

    preference = preference or get_backend()
    if preference == "pytorch":
        return checkpoint, None
    schema_gap = any(
        q.get("instructions") == ""
        or (q.get("type") == "score" and len(q.get("criteria") or []) < 2)
        for q in (questions or {}).values()
    )
    if images or schema_gap:
        reason = "Unsloth's native Clef path supports text with nonempty instructions and multi-level scores only."
        if preference == "llama.cpp":
            raise Unavailable(400, "api_usage_error", reason + " Select Auto or PyTorch.")
    else:
        native = native_availability()
        if native["available"]:
            return catalog.NATIVE_CHECKPOINTS[checkpoint.name], None
        reason = native["reason"]
        if preference == "llama.cpp":
            raise Unavailable(503, "model_unavailable", reason)
    return checkpoint, reason


def backend_info(checkpoint):
    from utils.systemone_settings import get_backend

    info = {"backend": get_backend(), "effective_backend": None, "fallback_reason": None}
    if isinstance(checkpoint, catalog.ClefCheckpoint):
        try:
            selected, info["fallback_reason"] = select_checkpoint(checkpoint)
            info["effective_backend"] = "llama.cpp" if _clef().is_native(selected) else "pytorch"
        except Unavailable as exc:
            info["fallback_reason"] = exc.message
    return info


def accepts_images(checkpoint):
    from utils.systemone_settings import get_backend, runtime_unavailable_reason
    return (
        isinstance(checkpoint, catalog.ClefCheckpoint)
        and get_backend() != "llama.cpp"
        and runtime_unavailable_reason() is None
    )


def _enter() -> None:
    if not _gate.acquire(timeout = 30):
        raise Unavailable(529, "overloaded", "The Decision API is busy; retry shortly", 1)


def decide(
    checkpoint: catalog.Checkpoint, state: Any, questions: dict, images: list[bytes]
) -> dict:
    global _fallback_reason
    if not _admission.acquire(blocking = False):
        raise Unavailable(529, "overloaded", "The Decision API is busy; retry shortly", 1)
    held = False
    try:
        _enter()
        held = True
        if isinstance(checkpoint, catalog.ClefCheckpoint):
            checkpoint, _fallback_reason = select_checkpoint(
                checkpoint, images = bool(images), questions = questions
            )
            laya_runtime.ensure_can_unload()
            if laya_runtime.status()["loaded_model"]:
                laya_runtime.unload()
            from utils.systemone_settings import get_device

            if get_device() == "gpu":
                from core.inference.gpu_arbiter import DECISIONS, GpuOwnerBusyError, acquire_for
                try:
                    # Atomically register startup without evicting another model.
                    acquire_for(DECISIONS, lambda: _clef().prepare(checkpoint), allow_evict = False)
                except GpuOwnerBusyError:
                    raise Unavailable(
                        409,
                        "gpu_busy",
                        "Unload the resident chat, image or video model before using Clef on GPU.",
                        1,
                    ) from None
            result = _clef().decide(checkpoint, state, questions, images)
            return {
                **result,
                "_backend": "llama.cpp" if _clef().is_native(checkpoint) else "pytorch",
            }
        _clef().ensure_can_unload()
        if _clef().status()["loaded_model"] or _clef().status()["loading_model"]:
            _clef().unload()
        return laya_runtime.decide(checkpoint, state, questions)
    finally:
        if held:
            _gate.release()
        _admission.release()


def status() -> dict:
    native = _clef().status()
    if native["loaded_model"] or native["loading_model"] or native["error"]:
        return {**native, "fallback_reason": _fallback_reason}
    return laya_runtime.status()


def ensure_can_unload() -> None:
    laya_runtime.ensure_can_unload()
    _clef().ensure_can_unload()


def unload() -> bool:
    _enter()
    try:
        ensure_can_unload()
        result = _clef().unload()
        return laya_runtime.unload() or result
    finally:
        _gate.release()


def shutdown() -> None:
    # Also retire a still-starting child; Laya is in-process.
    _clef().shutdown()


def download_plan(checkpoint: catalog.Checkpoint, *, preference = None) -> dict:
    if isinstance(checkpoint, catalog.ClefCheckpoint):
        selected, _ = select_checkpoint(checkpoint, preference = preference)
        return _clef().download_plan(selected)
    return laya_runtime.download_plan(checkpoint)


def loading_repo_ids() -> tuple[str, ...]:
    return laya_runtime.loading_repo_ids() + _clef().loading_repo_ids()
