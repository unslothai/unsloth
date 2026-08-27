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

# one end-to-end budget for the whole switch, under the ~100s tunnel window
_SWITCH_BUDGET_S = 90.0

# a generation the caller cannot see is yielded to rather than cut short, capped inside the budget
_DRAIN_WAIT_S = 30.0

# how long the gates are kept for a load that has not reached begin_load yet
_SETUP_GRACE_S = 120.0


def _resident_answers_exactly(resident: dict[str, Any], name: str) -> bool:
    """Whether the resident model is this exact name, needing no discovery at all.

    A scan that failed or skipped an entry would otherwise 404 the very model that is loaded,
    for as long as the empty index stays cached. Never true for a resident GGUF: a bare repo id
    means the preferred quant, which this comparison cannot see, so it would serve whichever
    quant happens to be up.
    """
    return (
        bool(resident.get("loaded"))
        and not resident_is_gguf(resident)
        and partition_matches(resident)
        and same_identity(name, str(resident.get("repo_id") or ""))
    )


async def _require_local(
    owner: str,
    pick: MediaModelPick,
    deadline: float,
    *,
    kind: str,
    openai_errors: bool,
    hf_token: Optional[str],
) -> None:
    """Refuse unless *pick* is provably downloaded in full.

    Bounded inside the switch budget, and free of side effects, so a planner that stalls can
    safely give back whatever locks and gates the caller is holding while it runs.
    """
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
    """Take every lock within the budget, releasing what was taken if one cannot be had.

    A request that spent most of its budget resolving would otherwise queue behind another full
    switch and blow past the response window before any of the inner waits could notice.
    """
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
    """Retrieve a handed-over task's exception, since the caller may have stopped awaiting it.

    ``_gated_start_load`` refuses on ordinary paths (a backend still busy at the in-gate drain,
    a cache deletion during it), and once the budget expires nothing awaits the task again. An
    unretrieved exception is reported by the loop at collection time, so a routine slow switch
    would log a traceback for a refusal that was handled correctly.
    """
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
    """Poll the background load until the REQUESTED model is resident; False if still going.

    Checked against the pick, not merely "something is loaded": a user load accepted between
    two polls supersedes this one, and returning success there would generate on the
    replacement while reporting the requested model.

    The probes are bounded like every other wait here: ``load_progress`` walks cache directories
    to count bytes, so on a slow or stalled filesystem a single poll can outlive the budget that
    the check at the bottom of the loop is meant to enforce.
    """
    probe = functools.partial(bounded, deadline = deadline, kind = kind, openai_errors = openai_errors)
    while True:
        progress = await probe(asyncio.to_thread(backend.load_progress)) or {}
        phase = progress.get("phase")
        if phase == "error":
            raise RuntimeError(progress.get("error") or "The model failed to load.")
        if phase in (None, "ready"):
            status = await probe(asyncio.to_thread(backend.status))
            # the landed check, not the skip check: this load is ours, so ambiguity is settled
            if resident_is_pick(status, name, pick):
                return True
            # loaded, but not this pick: a load that landed after ours replaced it
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
    """Run the final checks and start the load, owning the gates and *locks* throughout.

    Returns True when the resident model already answers the request, so the caller can stop.

    Ownership is the point. The caller shields this and may stop waiting on it, and the work
    from the last drain observation through ``begin_load`` must not be interruptible: engine
    activation unloads the resident pipeline on its way, so anything admitted before
    registration would be cut short by a load that no longer has a request behind it.

    Ownership is bounded all the same: a load that has not registered within ``_SETUP_GRACE_S``
    gives the gates back and carries on without them, since an installer running for minutes
    behind them costs more than the race they close.

    The gates held are the ones this load could evict behind, entered in a fixed order so two
    switches cannot deadlock, and one at a time under the budget: a stalled holder elsewhere
    would otherwise pin this task, and with it the switch lock, indefinitely. Cancelling during
    that acquisition is free, and the stack releases whatever was already entered; nothing past
    it may be interrupted.

    Chat's lifecycle gate is the FIRST of them, not the last. Every media generation route is
    counted on chat's in-flight counter as well as its own, and the middleware takes chat's gate
    and releases it before it parks on the media one. With the media gates taken first, a request
    arriving in between passed the still-open chat gate, incremented chat's ``_inflight``, and
    only then blocked on the held media gate: the in-gate drain discounts it on the media side
    (``count_pending=False``) but ``chat_busy(count_pending=False)`` still read it as running chat
    work, and an otherwise idle switch answered 409 without loading anything. Taking chat's gate
    first parks such a request in ``_note_pending`` instead, where both counters ignore it, and
    the middleware never holds a media gate while it waits for chat's, so the order is safe.

    A load that does not take the GPU holds its own backend's gate only. It cannot evict chat or
    the other media backend, so waiting on their gates would let an unrelated chat teardown time
    the switch out, and holding them would block new chat and video requests for as long as the
    re-plan and the load registration take.
    """
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
            # re-resolved under the gate: a concurrent load can activate the other image engine
            backend = backend_for(owner)
            # what the drain waited out may have been the very load this request wanted
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
            # re-planned because a cache deletion during the drain sees no load to guard against
            await _require_local(
                owner,
                pick,
                deadline,
                kind = kind,
                openai_errors = openai_errors,
                hf_token = hf_token,
            )
            # given its own task and waited on with a cap: a first-run native install runs for
            # minutes before begin_load, and holding both media gates and chat's that long
            # blocks every unrelated request. On expiry the load keeps going without them.
            setup = asyncio.ensure_future(_start_load(owner, pick, current_subject, hf_token))
            setup.add_done_callback(_consume_detached_error)
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(setup), _SETUP_GRACE_S)
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
    """Load the image or video model a generation request names, if it is not resident.

    No-op when the setting is off or nothing was named, so ``model`` keeps its old
    informational meaning for every existing client. With the setting on, a name that resolves
    to no downloaded model is refused: answering it would return one model's output under
    another's name.

    ``before_switch`` is the caller's last say on the resolved pick, run only when a switch is
    actually going to happen. It exists so a request the target model cannot serve is refused
    while the resident one is still loaded, rather than after a multi-minute load; a request the
    resident model already answers skips it, since the generate route judges that one anyway.
    """
    from utils.openai_auto_switch_settings import get_media_auto_switch_enabled

    if not isinstance(requested_model, str) or not requested_model.strip():
        return
    if not get_media_auto_switch_enabled():
        return

    # started before resolution: the cold scan is part of the wait the caller experiences
    deadline = time.monotonic() + _SWITCH_BUDGET_S
    name = requested_model.strip()
    task = IMAGE_TASK if owner == DIFFUSION else VIDEO_TASK
    kind = "image" if owner == DIFFUSION else "video"

    if _resident_answers_exactly(await asyncio.to_thread(backend_for(owner).status), name):
        return

    # off the loop: a cold index walks the model roots and reads gguf headers
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

    # re-read: the index build can run for the whole budget, and an idle unload can land in it
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
    # held only when the load takes the gpu, since a cpu-only switch takes it from nobody
    takes_the_gpu = await asyncio.to_thread(load_takes_the_gpu)
    gpu_lock = gpu_switch_lock() if takes_the_gpu else None
    locks = [held for held in (gpu_lock, lock) if held is not None]
    with note_switcher(owner):
        # the marker covers only the wait: once this request holds the lock it is real work
        with note_waiter(owner):
            await _acquire_all(locks, deadline, kind = kind, openai_errors = openai_errors)
        handed_over = False
        try:
            backend = backend_for(owner)
            # re-read under the lock: a concurrent request may have just loaded this model
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
                # probes answer to the switch budget: only a spent budget is the slow-switch 503
                probe_deadline = deadline,
                kind = kind,
                openai_errors = openai_errors,
            ):
                raise busy(kind, openai_errors)
            # its own task, so a timeout below frees the caller without unwinding gate or lock
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
        # re-resolved: an engine switch (diffusers <-> sd.cpp) replaces the object
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
    "resolve_local_media_model",
]
