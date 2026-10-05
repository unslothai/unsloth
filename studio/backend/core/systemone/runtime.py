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


def _clef():
    from . import clef_runtime
    return clef_runtime


def _enter() -> None:
    if not _gate.acquire(timeout = 30):
        raise Unavailable(529, "overloaded", "The Decision API is busy; retry shortly", 1)


def decide(checkpoint: catalog.Checkpoint, state: Any, questions: dict, images: list[bytes]) -> dict:
    if not _admission.acquire(blocking = False):
        raise Unavailable(529, "overloaded", "The Decision API is busy; retry shortly", 1)
    held = False
    try:
        _enter()
        held = True
        if isinstance(checkpoint, catalog.ClefCheckpoint):
            laya_runtime.ensure_can_unload()
            if laya_runtime.status()["loaded_model"]:
                laya_runtime.unload()
            from utils.systemone_settings import get_device
            if get_device() == "gpu":
                from core.inference.gpu_arbiter import DECISIONS, GpuOwnerBusyError, acquire_for
                try:
                    # Register before another consumer can evict this worker's startup.
                    # Explicitly refuse to evict an existing chat/image/video model.
                    acquire_for(DECISIONS, lambda: _clef().prepare(checkpoint), allow_evict = False)
                except GpuOwnerBusyError:
                    raise Unavailable(
                        409, "gpu_busy", "Unload the resident chat, image or video model before using Clef on GPU.", 1
                    ) from None
            return _clef().decide(checkpoint, state, questions, images)
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
        return native
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
    # Native teardown must retire a still-starting process too; Laya lives in this process.
    _clef().shutdown()


def download_plan(checkpoint: catalog.Checkpoint) -> dict:
    if isinstance(checkpoint, catalog.ClefCheckpoint):
        return _clef().download_plan(checkpoint)
    return laya_runtime.download_plan(checkpoint)


def loading_repo_ids() -> tuple[str, ...]:
    return laya_runtime.loading_repo_ids() + _clef().loading_repo_ids()
