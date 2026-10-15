# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in model auto-switch for the image and video generation APIs.

The chat twin lives in ``local_model_resolver`` + ``routes.inference``: a ``/v1`` request
naming a downloaded GGUF loads it before serving. Media had no equivalent, so
``POST /v1/images/generations`` answered 503 unless someone had already picked a model on the
Images page, and ``model`` was documented as informational. This resolves that name against
the downloaded image/video models, drains what the backend is doing, and runs the load the
picker would run.

Off by default (``media_api_auto_switch_model``), so existing clients see no change.

Only downloaded models resolve, and an unknown name is refused rather than answered by
whatever is resident. Nothing here starts a download: the media equivalent of the chat
auto-download setting would let one API key spend tens of GB, which is its own decision.

Both waits are bounded, because Unsloth's secure-mode tunnel caps an origin response near 100
seconds. Exceeding a bound leaves the work running and asks the caller to retry, the contract
``begin_load`` already gives the UI.

This module is the orchestration. The pieces it drives live next to it: ``media_model_index``
resolves a name and recognises the resident model, ``media_locality`` proves a pick is already
downloaded, ``media_switch_backends`` waits out work a switch would interrupt,
``media_switch_locks`` serializes switches, and ``media_switch_errors`` holds the refusals.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import time
from typing import Any, Callable, Optional

from core.inference.gpu_arbiter import DIFFUSION, VIDEO
from core.inference.media_locality import is_edit_only, missing_download_bytes
from core.inference.media_model_index import (
    IMAGE_TASK,
    VIDEO_TASK,
    MediaModelPick,
    available_media_model_ids,
    expected_partition,
    invalidate_index,
    partition_matches,
    resident_is_gguf,
    resident_is_pick,
    resolve_local_media_model,
    same_identity,
    satisfied_by,
)
from core.inference.media_switch_backends import (
    POLL_S,
    backend_for,
    drain,
    load_takes_the_gpu,
)
from core.inference.media_switch_errors import (
    EDIT_ONLY_MSG,
    LOADING_MSG,
    RETRY_AFTER_S,
    UNVERIFIED_MSG,
    bounded,
    busy,
    format_available,
    incomplete_message,
    refuse,
)
from core.inference.media_switch_locks import (
    gpu_switch_lock,
    note_switcher,
    note_waiter,
    switch_lock,
)
from loggers import get_logger

logger = get_logger(__name__)

# One end-to-end budget for the whole switch, under the ~100s tunnel window
_SWITCH_BUDGET_S = 90.0

_DRAIN_WAIT_S = 30.0

# how long the gates are kept for a load that has not reached begin_load yet
_SETUP_GRACE_S = 120.0


def _resident_answers_exactly(resident: dict[str, Any], name: str) -> bool:
    """Never true for a resident GGUF: a bare repo id means the preferred quant, which this cannot see."""
    return (
        bool(resident.get("loaded"))
        and not resident_is_gguf(resident)
        and partition_matches(resident)
        and same_identity(name, str(resident.get("repo_id") or ""))
    )


def resident_answers_media_request(
    resident: dict[str, Any], requested_model: Optional[str], *, owner: str
) -> bool:
    """Whether one exact resident state still answers an auto-switch request."""
    if not isinstance(requested_model, str) or not requested_model.strip():
        return False
    name = requested_model.strip()
    if _resident_answers_exactly(resident, name):
        return True
    task = IMAGE_TASK if owner == DIFFUSION else VIDEO_TASK
    pick = resolve_local_media_model(name, task = task)
    return pick is not None and satisfied_by(resident, name, pick)


async def _require_local(
    owner: str,
    pick: MediaModelPick,
    deadline: float,
    *,
    kind: str,
    openai_errors: bool,
    hf_token: Optional[str],
) -> None:
    """Refuses unless the pick is fully downloaded; no side effects, so a stalled planner can drop locks."""
    missing = await bounded(
        asyncio.to_thread(missing_download_bytes, owner, pick, hf_token),
        deadline,
        kind = kind,
        openai_errors = openai_errors,
    )
    if missing is None:
        raise refuse(
            UNVERIFIED_MSG.format(model = pick.model_id, kind = kind),
            status_code = 409,
            openai_errors = openai_errors,
            code = "model_not_downloaded",
        )
    if missing:
        raise refuse(
            incomplete_message(pick.model_id, missing, kind),
            status_code = 409,
            openai_errors = openai_errors,
            code = "model_not_downloaded",
        )


async def _acquire_all(locks: list, deadline: float, *, kind: str, openai_errors: bool) -> None:
    """Takes every lock within the budget or releases them all, so a queued switch cannot blow the
    window."""
    acquired: list = []
    try:
        for held in locks:
            await bounded(held.acquire(), deadline, kind = kind, openai_errors = openai_errors)
            acquired.append(held)
    except BaseException:
        for held in reversed(acquired):
            held.release()
        raise


def _consume_detached_error(task: "asyncio.Task") -> None:
    """Retrieves a detached task's exception so a handled refusal is not logged as unretrieved."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.debug("Media auto-switch: setup finished after the caller stopped waiting: %s", exc)


async def _await_loaded(
    backend: Any,
    name: str,
    pick: MediaModelPick,
    deadline: float,
    *,
    kind: str,
    openai_errors: bool,
) -> bool:
    """Checks the requested model, not any load; probes are bounded since load_progress walks cache dirs."""
    probe = functools.partial(bounded, deadline = deadline, kind = kind, openai_errors = openai_errors)
    while True:
        progress = await probe(asyncio.to_thread(backend.load_progress)) or {}
        phase = progress.get("phase")
        if phase == "error":
            raise RuntimeError(progress.get("error") or "The model failed to load.")
        if phase in (None, "ready"):
            status = await probe(asyncio.to_thread(backend.status))
            if resident_is_pick(status, name, pick):
                return True
            raise RuntimeError(f"'{pick.model_id}' was replaced by another load before it served.")
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(POLL_S)


async def _start_load(
    owner: str,
    pick: MediaModelPick,
    current_subject: str,
    hf_token: Optional[str] = None,
) -> None:
    """Run the load its own route would run, as an API load rather than a user one."""
    partition = expected_partition(pick)
    if owner == DIFFUSION:
        from models.inference import DiffusionLoadRequest
        from routes.inference import load_diffusion_model_gated
        await load_diffusion_model_gated(
            DiffusionLoadRequest(
                model_path = pick.model_path,
                display_repo_id = pick.model_id,
                gguf_filename = pick.gguf_filename,
                model_kind = pick.model_kind,
                hf_token = hf_token,
            ),
            current_subject,
            user_initiated = False,
        )
    else:
        from models.inference import VideoLoadRequest
        from routes.video import load_video_model_gated
        await load_video_model_gated(
            VideoLoadRequest(
                model_path = pick.model_path,
                display_repo_id = pick.model_id,
                gguf_filename = pick.gguf_filename,
                model_kind = pick.model_kind,
                h3_task = partition,
                hf_token = hf_token,
            ),
            current_subject,
            user_initiated = False,
        )
    logger.info("Media auto-switch: loading %s on the %s backend", pick.model_id, owner)


async def _gated_start_load(
    owner: str,
    name: str,
    pick: MediaModelPick,
    current_subject: str,
    locks: list,
    deadline: float,
    *,
    kind: str,
    openai_errors: bool,
    hf_token: Optional[str],
    takes_the_gpu: bool,
) -> bool:
    """Chat's gate is taken first, so a request parked on media cannot falsely 409 an idle switch."""
    from fastapi import HTTPException
    from core.inference.media_keepwarm import admission_gate
    from core.inference.llama_keepwarm import inference_lifecycle_gate

    needed = (
        (inference_lifecycle_gate(), admission_gate(DIFFUSION), admission_gate(VIDEO))
        if takes_the_gpu
        else (admission_gate(owner),)
    )
    try:
        async with contextlib.AsyncExitStack() as gates:
            for gate in needed:
                await bounded(
                    gates.enter_async_context(gate),
                    deadline,
                    kind = kind,
                    openai_errors = openai_errors,
                )
            backend = backend_for(owner)
            if satisfied_by(await asyncio.to_thread(backend.status), name, pick):
                return True
            if not await drain(
                owner,
                backend,
                time.monotonic(),
                count_pending = False,
                probe_deadline = deadline,
                kind = kind,
                openai_errors = openai_errors,
            ):
                raise busy(kind, openai_errors)
            await _require_local(
                owner,
                pick,
                deadline,
                kind = kind,
                openai_errors = openai_errors,
                hf_token = hf_token,
            )
            # Own task with a cap: a first-run native install can hold the gates for minutes
            setup = asyncio.ensure_future(_start_load(owner, pick, current_subject, hf_token))
            setup.add_done_callback(_consume_detached_error)
            with contextlib.suppress(asyncio.TimeoutError):
                try:
                    await asyncio.wait_for(asyncio.shield(setup), _SETUP_GRACE_S)
                except HTTPException as exc:
                    if isinstance(exc.detail, dict) and exc.detail.get("error") == "gpu_busy":
                        raise busy(
                            kind,
                            openai_errors,
                            retry_after = int(exc.detail["retry_after"]),
                        ) from exc
                    raise
            return False
    finally:
        for held in reversed(locks):
            held.release()


async def maybe_auto_switch_media_model(
    requested_model: Optional[str],
    *,
    owner: str,
    current_subject: str,
    openai_errors: bool,
    hf_token: Optional[str] = None,
    before_switch: Optional[Callable[[MediaModelPick], None]] = None,
) -> None:
    """No-op unless the setting is on; then refuses a name resolving to no downloaded model."""
    from utils.openai_auto_switch_settings import get_media_auto_switch_enabled

    if not isinstance(requested_model, str) or not requested_model.strip():
        return
    if not get_media_auto_switch_enabled():
        return

    deadline = time.monotonic() + _SWITCH_BUDGET_S
    name = requested_model.strip()
    task = IMAGE_TASK if owner == DIFFUSION else VIDEO_TASK
    kind = "image" if owner == DIFFUSION else "video"

    if _resident_answers_exactly(await asyncio.to_thread(backend_for(owner).status), name):
        return

    pick = await bounded(
        asyncio.to_thread(resolve_local_media_model, name, task = task),
        deadline,
        kind = kind,
        openai_errors = openai_errors,
    )
    if pick is None:
        available = format_available(
            await bounded(
                asyncio.to_thread(available_media_model_ids, task),
                deadline,
                kind = kind,
                openai_errors = openai_errors,
            )
        )
        raise refuse(
            f"No downloaded {kind} model matches '{name}'."
            + (f" Downloaded {kind} models: {available}." if available else ""),
            status_code = 404,
            openai_errors = openai_errors,
            code = "model_not_found",
        )

    # before anything is evicted: the load would otherwise finish and be refused for lacking txt2img
    if owner == DIFFUSION and await asyncio.to_thread(is_edit_only, pick):
        raise refuse(
            EDIT_ONLY_MSG.format(model = pick.model_id),
            status_code = 400,
            openai_errors = openai_errors,
            code = "invalid_value",
        )

    # Re-read: the index build can take the whole budget and an idle unload can land in it
    if satisfied_by(await asyncio.to_thread(backend_for(owner).status), name, pick):
        return

    if before_switch is not None:
        await bounded(
            asyncio.to_thread(before_switch, pick),
            deadline,
            kind = kind,
            openai_errors = openai_errors,
        )

    lock = switch_lock(owner)
    takes_the_gpu = await asyncio.to_thread(load_takes_the_gpu)
    gpu_lock = gpu_switch_lock() if takes_the_gpu else None
    locks = [held for held in (gpu_lock, lock) if held is not None]
    with note_switcher(owner):
        with note_waiter(owner):
            await _acquire_all(locks, deadline, kind = kind, openai_errors = openai_errors)
        handed_over = False
        try:
            backend = backend_for(owner)
            if satisfied_by(await asyncio.to_thread(backend.status), name, pick):
                return
            await _require_local(
                owner,
                pick,
                deadline,
                kind = kind,
                openai_errors = openai_errors,
                hf_token = hf_token,
            )
            if not await drain(
                owner,
                backend,
                min(deadline, time.monotonic() + _DRAIN_WAIT_S),
                probe_deadline = deadline,
                kind = kind,
                openai_errors = openai_errors,
            ):
                raise busy(kind, openai_errors)
            setup = asyncio.ensure_future(
                _gated_start_load(
                    owner,
                    name,
                    pick,
                    current_subject,
                    locks,
                    deadline,
                    kind = kind,
                    openai_errors = openai_errors,
                    hf_token = hf_token,
                    takes_the_gpu = takes_the_gpu,
                )
            )
            setup.add_done_callback(_consume_detached_error)
            handed_over = True
            if await bounded(
                asyncio.shield(setup), deadline, kind = kind, openai_errors = openai_errors
            ):
                return
        finally:
            if not handed_over:
                for held in reversed(locks):
                    held.release()

    try:
        ready = await _await_loaded(
            backend_for(owner), name, pick, deadline, kind = kind, openai_errors = openai_errors
        )
    except RuntimeError as exc:
        # the loader already redacts this text; a bare raise would 500 with it
        raise refuse(
            f"'{pick.model_id}' could not be loaded: {exc}",
            status_code = 503,
            openai_errors = openai_errors,
            code = "model_load_failed",
        )
    if not ready:
        raise refuse(
            LOADING_MSG.format(model = pick.model_id),
            status_code = 503,
            openai_errors = openai_errors,
            code = "model_loading",
            retry_after = RETRY_AFTER_S,
        )


__all__ = [
    "IMAGE_TASK",
    "VIDEO_TASK",
    "MediaModelPick",
    "available_media_model_ids",
    "invalidate_index",
    "maybe_auto_switch_media_model",
    "resident_answers_media_request",
    "resolve_local_media_model",
]
