# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Reaching the media backends, and waiting until nothing a switch would interrupt is running.

Loading a media model is not a private act. The load route takes the GPU through the arbiter,
whose cross-owner handoff unloads whichever owner holds it, so a switch can cancel a video
generation or terminate a streaming chat completion that has nothing to do with the request that
asked for it. The drain here waits all of that out first, and refuses rather than interrupting.
"""

from __future__ import annotations

import asyncio
import functools
import time
from typing import Any, Optional

from core.inference.gpu_arbiter import DIFFUSION, VIDEO
from core.inference.media_switch_errors import probe
from core.inference.media_switch_locks import switcher_count, waiter_count

POLL_S = 0.2


def backend_for(owner: str) -> Any:
    """The live backend object for *owner*, resolved on each call rather than cached."""
    if owner == DIFFUSION:
        from core.inference.diffusion_engine_router import get_active_diffusion_engine
        return get_active_diffusion_engine()
    from core.inference.video import get_video_backend

    return get_video_backend()


def other_owner(owner: str) -> str:
    """The media backend this one would take the GPU from."""
    return VIDEO if owner == DIFFUSION else DIFFUSION


def load_takes_the_gpu() -> bool:
    """Whether this load will go through the arbiter and evict the current owner.

    A CPU-only diffusion device releases ownership instead of acquiring it, so such a switch
    interrupts nothing and must not wait on chat or the other media backend.
    """
    try:
        from core.inference.diffusion_device import resolve_diffusion_device_target
        return resolve_diffusion_device_target().device != "cpu"
    except Exception:  # noqa: BLE001 -- assume the handoff, which is the careful direction
        return True


def chat_busy(count_pending: bool = True) -> bool:
    """Whether a chat request or load is in flight, so the GPU handoff would interrupt it.

    The arbiter evicts chat unconditionally for the current owner, terminating a streaming
    completion that has nothing to do with this switch.

    ``count_pending`` is False once the lifecycle gate is held: a request blocked in the
    middleware behind that gate has not started inference and cannot be interrupted, while one
    admitted just before the gate was taken is already running and still can be.
    """
    try:
        from core.inference.llama_keepwarm import other_inference_request_count
    except Exception:  # noqa: BLE001 -- no chat stack means no chat work
        return False
    try:
        # chat's counter covers media requests too, and none of those is using chat
        parked = switcher_count()
        counted = other_inference_request_count(
            current_request_counted = True, include_pending = count_pending
        )
        # counted once, since a request parked on a switch lock is a waiter inside its own switch
        return max(0, counted - max(0, parked - 1)) > 0
    except Exception:  # noqa: BLE001
        return False


def backend_busy(backend: Any) -> bool:
    """One off-loop read of whether a load or generation is running. Mirrors media_keepwarm."""
    if backend.loading_repo_ids():
        return True
    return bool((backend.generate_progress() or {}).get("active"))


def other_backend_busy(owner: str) -> bool:
    """Whether the other media backend is loading or generating, off the loop.

    Guarded and lazy: an Unsloth that never opened the other page has no backend to ask, and
    importing one just to find that out would drag torch in for nothing.
    """
    import sys

    other = other_owner(owner)
    wanted = (
        {"core.inference.video"}
        if other == VIDEO
        else {"core.inference.diffusion", "core.inference.sd_cpp_backend"}
    )
    if not wanted & set(sys.modules):
        return False
    try:
        return backend_busy(backend_for(other))
    except Exception:  # noqa: BLE001 -- an unavailable backend is not busy work
        return False


async def drain(
    owner: str,
    backend: Any,
    deadline: float,
    *,
    count_pending: bool = True,
    probe_deadline: Optional[float] = None,
    check_chat: bool = True,
    kind: str = "image",
    openai_errors: bool = True,
) -> bool:
    """Wait out other tracked requests and any in-flight load or generation.

    A request queued on this backend's switch lock is counted by the middleware but is not
    doing any work, so it is discounted here: two concurrent requests for the same absent
    model would otherwise each wait the other out and both return 409. Mirrors the chat
    switch, which excludes its own waiters from ``_wait_for_model_switch_idle``.

    The other media backend counts too, because the arbiter's cross-owner handoff unloads
    whatever holds the GPU. So does chat, whether or not it is streaming. Both are skipped
    entirely when this load does not take the GPU at all.

    ``count_pending`` is False for the check made while holding the admission gate. A request
    arriving then is counted pending and immediately blocks on that gate, so counting it would
    abort a switch over a newcomer that cannot be touching the backend.

    ``probe_deadline`` bounds the busy probes themselves, and is the switch budget rather than
    this loop's deadline: the in-gate check evaluates the condition once with no time to wait,
    and reusing that as the probe bound would report every backend busy. The probes need a bound
    at all because ``loading_repo_ids`` takes the backend lock, which the loader holds across
    pipeline assembly, so an unbounded probe outlives the response window.

    ``check_chat`` stays on for the in-gate check, where ``count_pending`` is what makes it safe:
    a chat request blocked behind the held lifecycle gate has not started inference and must not
    abort the switch, but one admitted between the outer drain's last probe and this gate being
    taken is already running and would be terminated by the handoff.
    """
    from core.inference.media_keepwarm import other_request_count

    # device configuration is resolved once, not on every poll: that cost ~150 round-trips a switch
    cross_owner = await asyncio.to_thread(load_takes_the_gpu)
    while True:
        # this request is itself tracked and itself a waiter, so it counts as neither
        others = other_request_count(
            owner, current_request_counted = True, count_pending = count_pending
        )
        others -= waiter_count(owner)
        if cross_owner:
            other = other_owner(owner)
            others += max(
                0,
                other_request_count(other, count_pending = count_pending) - switcher_count(other),
            )
        probe_by = deadline if probe_deadline is None else probe_deadline
        bounded_probe = functools.partial(probe, kind = kind, openai_errors = openai_errors)
        if (
            others <= 0
            and not await bounded_probe(backend_busy, backend, probe_by)
            and not (cross_owner and await bounded_probe(other_backend_busy, owner, probe_by))
            and not (
                cross_owner
                and check_chat
                and await bounded_probe(functools.partial(chat_busy, count_pending), None, probe_by)
            )
        ):
            return True
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(POLL_S)


__all__ = [
    "POLL_S",
    "backend_busy",
    "backend_for",
    "chat_busy",
    "drain",
    "load_takes_the_gpu",
    "other_backend_busy",
    "other_owner",
]
