# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in media auto-switch (media_api_auto_switch_model); never downloads; refuses unknown names;
waits are bounded near the ~100 s tunnel cap, and exceeding one leaves work running and asks
retry."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import time
from typing import Any, Callable, Optional

from core.inference.gpu_arbiter import DIFFUSION, MULTI_MODEL_REFUSAL_HEADERS, VIDEO
from core.inference import media_locality
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


def _resolve_pick(owner: str, name: str, task: str) -> Optional[MediaModelPick]:
    """The downloaded model *name* refers to. A retired FLUX.3 full copy that is not downloaded itself answers with
    what replaced it when that is downloaded: the base it copies, which Studio stages in that base's -FP8 mirror."""
    pick = resolve_local_media_model(name, task = task)
    if pick is not None or owner != DIFFUSION:
        return pick
    from core.inference.diffusion_families import mirror_repo, retired_full_copy_base

    base = retired_full_copy_base(name)
    if base is None:
        return None
    if any(
        c and resolve_local_media_model(c, task = task) is not None for c in (base, mirror_repo(base))
    ):
        # Under the base's identity, as the load route publishes it: locality is planned for the base (its cached
        # mirror), and a base the Images page already loaded answers the request without a reload.
        return MediaModelPick(base, base, None, "pipeline")
    return None


def _retired_copy_as_base(pick: MediaModelPick) -> MediaModelPick:
    """A downloaded retired FLUX.3 full copy as the load will read it: the pick of the base it copies, which
    ``DiffusionLoadRequest`` redirects it to. So locality is proven for the files that load fetches (the -FP8 mirror's),
    not the copy's snapshot, and the resident match compares the id the status publishes. A copy complete on disk stays
    itself, as the load route keeps it (``retired_copy_loads_from_disk``)."""
    from core.inference.diffusion_families import (
        retired_copy_loads_from_disk,
        retired_full_copy_base,
    )

    base = retired_full_copy_base(pick.model_id)
    if base is None:
        return pick
    single = (pick.gguf_filename or "").replace("\\", "/")
    if single and single != "model.safetensors":
        return pick
    if retired_copy_loads_from_disk(pick.model_path):
        return pick
    return MediaModelPick(base, base, None, "pipeline")


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
        from core.inference.diffusion_families import retired_copy_loads_from_disk
        from core.inference.media_locality import flux3_seeded_cache_repo, seeded_family_override
        from models.inference import DiffusionLoadRequest
        from routes.inference import load_diffusion_model_gated

        # A seeded FLUX.3 snapshot loads by its repo id (an API load is cache only), so the load plans the
        # cached seed the locality check approved instead of opening dense shards the snapshot never held.
        model_path = pick.model_path
        family_override = None
        seeded_repo = None if pick.gguf_filename else flux3_seeded_cache_repo(pick.model_path)
        # By id only the active cache is searched: a complete retired copy in an earlier Models folder keeps its path.
        if seeded_repo and retired_copy_loads_from_disk(pick.model_path):
            seeded_repo = None
        if seeded_repo:
            model_path = seeded_repo
            # The listing named the family from the snapshot's modular_model_index.json; an opaquely named repo id
            # carries no such evidence, so the load is told the family the manifest declared.
            family_override = seeded_family_override(pick.model_path, seeded_repo)
        await load_diffusion_model_gated(
            DiffusionLoadRequest(
                model_path = model_path,
                display_repo_id = pick.model_id,
                gguf_filename = pick.gguf_filename,
                model_kind = pick.model_kind,
                family_override = family_override,
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
    """Chat gate first, then media gates in fixed order; the load must not be interruptible"""
    from fastapi import HTTPException
    from core.inference import gpu_arbiter, gpu_residency
    from core.inference.media_keepwarm import admission_gate
    from core.inference.llama_keepwarm import inference_lifecycle_gate

    try:
        # Under Always a chat load's GPUs aren't known yet, so it overlaps every card: refuse now
        # (retryable) instead of queueing on chat's gate behind it. Off and fit keep the wait, and
        # so does a CPU chat load (no GPU claim) or one on known GPUs (the arbiter checks overlap).
        if (
            takes_the_gpu
            and gpu_residency.chat_load_in_flight()
            and gpu_arbiter.residents().get(gpu_arbiter.CHAT, ...) is None
            and await asyncio.to_thread(gpu_residency.current_policy) == "always"
        ):
            raise gpu_residency.chat_loading_http_exception(openai_errors = openai_errors)
        needed = (
            (inference_lifecycle_gate(), admission_gate(DIFFUSION), admission_gate(VIDEO))
            if takes_the_gpu
            else (admission_gate(owner),)
        )
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
            # Ranked once: the check below and the load it starts both use this card.
            planned = media_locality.pin_planned_gpu_ordinal(
                await bounded(
                    asyncio.to_thread(media_locality.plan_gpu_ordinal),
                    deadline,
                    kind = kind,
                    openai_errors = openai_errors,
                )
            )
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
            media_locality._PLANNED_ORDINAL.reset(planned)  # the task copied the context
            setup.add_done_callback(_consume_detached_error)
            with contextlib.suppress(asyncio.TimeoutError):
                try:
                    await asyncio.wait_for(asyncio.shield(setup), _SETUP_GRACE_S)
                except HTTPException as exc:
                    # The final arbiter check also covers a generation registered while load preparation was off the loop.
                    if isinstance(exc.detail, dict) and exc.detail.get("error") == "gpu_busy":
                        raise busy(
                            kind,
                            openai_errors,
                            retry_after = int(exc.detail["retry_after"]),
                        ) from exc
                    headers = {k.lower(): v for k, v in (exc.headers or {}).items()}
                    if (
                        headers.get("x-unsloth-refusal")
                        == MULTI_MODEL_REFUSAL_HEADERS["X-Unsloth-Refusal"]
                    ):
                        # Keep multiple models loaded is set to Always: waiting does not help, so no Retry-After.
                        refusal = refuse(
                            exc.detail if isinstance(exc.detail, str) else str(exc.detail),
                            status_code = 409,
                            openai_errors = openai_errors,
                            code = "multi_model_always",
                        )
                        refusal.headers = dict(MULTI_MODEL_REFUSAL_HEADERS)
                        raise refusal from exc
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

    # started before resolution: the cold scan is part of the wait the caller experiences
    deadline = time.monotonic() + _SWITCH_BUDGET_S
    name = requested_model.strip()
    task = IMAGE_TASK if owner == DIFFUSION else VIDEO_TASK
    kind = "image" if owner == DIFFUSION else "video"

    if _resident_answers_exactly(await asyncio.to_thread(backend_for(owner).status), name):
        return

    # off the loop: a cold index walks the model roots and reads gguf headers
    pick = await bounded(
        asyncio.to_thread(_resolve_pick, owner, name, task),
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

    if owner == DIFFUSION:
        pick = _retired_copy_as_base(pick)

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
    "resident_answers_media_request",
    "resolve_local_media_model",
]
