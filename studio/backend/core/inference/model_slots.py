# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Models kept loaded alongside the primary one, each with its own llama-server and orchestrator.
A task naming one sets ``routed_slot``, so the usual backend getters answer with that slot."""

import asyncio
import atexit
import contextvars
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from core.inference.orchestrator import routed_slot
from hub.services.models import account_access
from loggers import get_logger
from state import active_generations
from utils.account_context import current_account_id

logger = get_logger(__name__)


@dataclass(eq = False)
class ExtraSlot:
    llama: Any
    orchestrator: Any
    account: str
    request: Any = None
    last_used: float = 0.0
    generations: set = field(default_factory = set)
    refs: int = 0


slots: list[ExtraSlot] = []
lock = threading.Lock()
# (slot, model_path) while a load fills a slot of its own.
loading: Optional[tuple[ExtraSlot, str]] = None
# Out of routing, but their server would not stop: still priced for VRAM and retried by every drop.
stuck: list[ExtraSlot] = []


def in_slot(slot: Optional[ExtraSlot], fn: Callable):
    context = contextvars.copy_context()
    context.run(routed_slot.set, slot)
    return context.run(fn)


def filling_model() -> Optional[str]:
    current = loading
    return current[1] if current else None


def any_loading() -> bool:
    return loading is not None or any(
        getattr(slot.orchestrator, "loading_models", None) for slot in list(slots)
    )


def in_use(slot: ExtraSlot) -> bool:
    return bool(slot.llama.is_active or slot.orchestrator.active_model_name)


def busy() -> bool:
    return any_loading() or any(in_use(slot) for slot in list(slots))


def resident() -> list[ExtraSlot]:
    """Every kept slot that may still hold a model: the routed ones and the stuck ones."""
    return [*slots, *stuck]


def _worker_alive(orchestrator) -> bool:
    worker_alive = getattr(orchestrator, "is_worker_alive", None)
    return bool(callable(worker_alive) and worker_alive())


def holds_vram() -> bool:
    """A kept model on the GPU or one loading; a llama-server started with no GPU layers is not."""
    return any_loading() or any(
        slot.orchestrator.active_model_name
        or (slot in stuck and _worker_alive(slot.orchestrator))
        or (slot.llama.is_active and getattr(slot.llama, "_gpu_offload_active", None) is not False)
        for slot in [*slots, *stuck]
    )


def visible() -> list[ExtraSlot]:
    if not account_access.managed_account():
        return list(slots)
    return [slot for slot in slots if slot.account == current_account_id()]


def visible_loading() -> Optional[tuple[ExtraSlot, str]]:
    current = loading
    return current if current and current[0] in visible() else None


def serving_slot(requested: str, candidates: list[ExtraSlot], satisfies) -> Optional[ExtraSlot]:
    """The slot among ``candidates`` serving ``requested``; None when the primary does (it wins a tie)."""
    for slot in (None, *candidates):
        if in_slot(slot, lambda: satisfies(requested)):
            return slot
    return None


async def route(requested: Optional[str], satisfies) -> Optional[ExtraSlot]:
    """Route this task to the slot serving ``requested``, if any, and hold it until the task ends."""
    routed_slot.set(None)
    if not (slots and isinstance(requested, str) and requested):
        return None
    candidates = visible()
    if not candidates:
        return None
    slot = await asyncio.to_thread(serving_slot, requested, candidates, satisfies)
    if slot is None or not reserve(slot):
        return None
    routed_slot.set(slot)
    slot.last_used = time.monotonic()
    return slot


def reserve(slot: ExtraSlot) -> bool:
    """Hold ``slot`` against eviction until the current task ends; False once it is gone.
    The task, not the request: a durable chat run inherits the POST that queued it, which has
    already answered 202 by the time the run routes."""
    with lock:
        if slot not in slots:
            return False
        slot.refs += 1

    def release(_task = None):
        with lock:
            slot.refs -= 1

    task = asyncio.current_task()
    release() if task is None else task.add_done_callback(release)
    return True


def eviction_victims(
    exclude,
    short_mib: int,
    gpu_indices = None,
) -> list[ExtraSlot]:
    """Least recently used idle slots, as many as it takes to free ``short_mib`` on ``gpu_indices``;
    one at a time when a footprint is unknown, since the retry prices the rest. A slot still
    answering is never a victim."""
    candidates = [s for s in visible() if s is not exclude and in_use(s)]
    victims, freed = [], 0
    for slot in sorted(candidates, key = lambda s: s.last_used):
        planned = getattr(slot.llama, "_planned_vram_mib", {})
        held = sum(mib for idx, mib in planned.items() if gpu_indices is None or idx in gpu_indices)
        if slot.generations or slot.refs or (planned and not held):
            continue
        victims.append(slot)
        freed += held
        if not held or freed >= short_mib:
            return victims
    return []


def claim_victim(slot: ExtraSlot) -> bool:
    """Take ``slot`` out of routing unless a request reached it since it was picked."""
    with lock:
        if slot.generations or slot.refs or slot not in slots:
            return False
        slots.remove(slot)
        return True


def evict(
    exclude,
    short_mib: int,
    gpu_indices = None,
) -> list[ExtraSlot]:
    """Unload the idle slots ``eviction_victims`` picks."""
    return evict_these(eviction_victims(exclude, short_mib, gpu_indices))


def evict_these(victims) -> list[ExtraSlot]:
    """Unload ``victims``, skipping any a request reached since they were picked."""
    dropped = []
    for victim in victims:
        if claim_victim(victim):
            drop(victim)
            dropped.append(victim)
    return dropped


def all_generations() -> set:
    return {event for slot in list(slots) for event in list(slot.generations)}


def routed_generation_count() -> int:
    """Generations on the model this task is routed to; each model decodes on its own server."""
    slot = routed_slot.get()
    if slot is not None:
        return len(slot.generations)
    elsewhere = all_generations()
    return active_generations.count(None, elsewhere) if elsewhere else active_generations.count()


def drop(slot: ExtraSlot) -> None:
    from core.inference.llama_cpp import unregister_serving_backend

    with suppress(ValueError):
        slots.remove(slot)
    try:
        slot.llama.unload_model()
    except BaseException:
        if slot not in stuck:
            stuck.append(slot)
        slot.orchestrator._cleanup()
        raise
    slot.orchestrator._cleanup()
    # _cleanup does not report a worker that outlived its kill.
    if _worker_alive(slot.orchestrator):
        if slot not in stuck:
            stuck.append(slot)
        raise RuntimeError("An inference worker kept alongside is still alive")
    with suppress(ValueError):
        stuck.remove(slot)
    unregister_serving_backend(slot.llama)
    atexit.unregister(slot.llama._cleanup)
    atexit.unregister(slot.orchestrator._cleanup)


def stop_orchestrator_workers() -> None:
    """Drop every slot running an orchestrator worker, for the transformers sidecar swap. Raises
    when one survives: it would lazy-import from the swapped package tree."""

    doomed = [
        s for s in resident() if s.orchestrator.active_model_name or _worker_alive(s.orchestrator)
    ]
    for slot in doomed:
        with suppress(Exception):
            drop(slot)
    if any(_worker_alive(slot.orchestrator) for slot in doomed):
        raise RuntimeError(
            "An inference worker kept alongside is still alive before the transformers swap"
        )


def _drop_where(predicate, strict: bool = False) -> int:
    """``strict`` raises once every slot was tried if any teardown failed: a GPU handoff must not
    proceed past a server that may still hold VRAM."""
    filling = loading[0] if loading else None
    doomed = [slot for slot in list(slots) if predicate(slot, slot is filling)]
    retried = list(stuck)
    failed = []
    for slot in doomed:
        try:
            drop(slot)
        except Exception as exc:
            logger.warning("Could not unload an extra model: %s", exc)
            failed.append(exc)
    for slot in retried:
        try:
            drop(slot)
        except Exception as exc:
            failed.append(exc)
    if strict and failed:
        raise RuntimeError(f"Could not unload {len(failed)} model(s) kept alongside") from failed[0]
    return len(doomed)


def unload_extra_models(
    keep = None,
    spare_filling: bool = False,
    strict: bool = False,
) -> int:
    """Drop every slot, or with ``keep`` only the loaded ones it does not spare. Returns how many."""

    def doomed(slot, filling):
        loaded = slot.llama.is_loaded or slot.orchestrator.active_model_name
        return not (spare_filling and filling) and (
            keep is None or bool(loaded and not keep(slot.llama))
        )

    return _drop_where(doomed, strict)


def unload_idle() -> int:
    """Drop every kept model no request is using or loading."""
    filling = loading[0] if loading else None
    idle = [s for s in list(slots) if s is not filling and not s.generations and not s.refs]
    return len(evict_these(idle))


def audio_cpp_model_names(orchestrator) -> list[str]:
    """The audio.cpp models an orchestrator holds or is loading: their servers run from the managed tree."""
    from core.inference.audio_cpp_models import AUDIO_CPP_AUDIO_TYPES, looks_like_audio_cpp

    # Snapshot the loads first: one that publishes between the two reads moves from loading to held.
    loading = list(getattr(orchestrator, "loading_models", None) or ())
    held = [
        name
        for name, entry in list((getattr(orchestrator, "models", None) or {}).items())
        if isinstance(entry, dict) and entry.get("audio_type") in AUDIO_CPP_AUDIO_TYPES
    ]
    return held + [name for name in loading if name not in held and looks_like_audio_cpp(name)]


def unload_audio_cpp_slots(strict: bool = False) -> int:
    """Drop every slot holding or loading an audio.cpp model, for the audio.cpp runtime swap."""
    doomed = [slot for slot in list(slots) if audio_cpp_model_names(slot.orchestrator)]
    for slot in doomed:
        # drop() keeps the loading marker, so a load still ahead of its spawn would start a worker after it.
        for name in list(slot.orchestrator.loading_models):
            slot.orchestrator.cancel_load(name)
    return _drop_where(lambda slot, filling: any(slot is held for held in doomed), strict)


def unload_llama_slots(strict: bool = False) -> int:
    """Drop every slot running or starting a llama-server, a GGUF load still filling one included."""
    return _drop_where(
        lambda slot, filling: slot.llama.is_active
        or slot.llama.is_loaded
        or (filling and not getattr(slot.orchestrator, "loading_models", None)),
        strict,
    )
