# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Batches waiting image requests from any account; a shared outcome reaches only the leader's account."""

from __future__ import annotations

import asyncio
import os
import secrets
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, ContextManager, Hashable, Optional

from loggers import get_logger

from . import diffusion_batch_trace as trace
from .diffusion_batched import SEED_MASK

logger = get_logger(__name__)

BATCH_MAX_DEFAULT = 8
POLL_S = 0.05
# Derived from every prompt in the call: never shown across accounts.
_PROMPT_DERIVED_FIELDS = ("embeddings", "embeddings_skipped")


class _RunAlone(Exception):
    """Retry this request individually."""


# A member a fired batch could not take (no admitted size): it may coalesce again, behind that batch.
_LEFT_OUT_MSG = "left out"
_LEFT_OUT = object()
REJOIN_ROUNDS = 3


@dataclass
class _Member:
    prompt: str
    seeds: list[int]
    future: "asyncio.Future[dict[str, Any]]"
    owner: Hashable = None


@dataclass
class _Batch:
    key: Hashable
    members: list[_Member] = field(default_factory = list)
    images: int = 0
    fired: bool = False


_OPEN: dict[Hashable, _Batch] = {}


def batch_max() -> int:
    try:
        return max(1, int(os.environ.get("UNSLOTH_STUDIO_IMAGE_BATCH_MAX", BATCH_MAX_DEFAULT)))
    except ValueError:
        return BATCH_MAX_DEFAULT


def collecting() -> bool:
    return bool(_OPEN)


def seeded_join_enabled() -> bool:
    return os.environ.get("UNSLOTH_STUDIO_IMAGE_BATCH_SEEDED", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def request_seeds(seed: Optional[int], n: int) -> list[int]:
    base = seed if seed is not None else secrets.randbelow(SEED_MASK + 1)
    return [(base + i) & SEED_MASK for i in range(n)]


def _split(
    result: dict[str, Any],
    members: list[_Member],
    *,
    prefix: bool = False,
) -> list[dict[str, Any]]:
    """Each member's slice; ``prefix``: only leading members whose images all finished get a part."""
    images = list(result.get("images") or [])
    seeds = list(result.get("seeds") or [])
    wanted = sum(len(member.seeds) for member in members)
    if prefix:
        covered, total = [], 0
        for member in members:
            if total + len(member.seeds) > len(images):
                break
            total += len(member.seeds)
            covered.append(member)
        members, wanted = covered, total
        images, seeds = images[:total], seeds[:total]
    if len(images) != wanted or (seeds and len(seeds) != wanted):
        raise ValueError(f"a batch of {wanted} images returned {len(images)}")
    if len({member.owner for member in members}) > 1:
        result = {k: v for k, v in result.items() if k not in _PROMPT_DERIVED_FIELDS}
    parts = []
    start = 0
    call_size = sum(len(member.seeds) for member in members)
    for member in members:
        end = start + len(member.seeds)
        parts.append(
            {
                **result,
                "images": images[start:end],
                "seeds": seeds[start:end] if seeds else list(member.seeds),
                "seed": member.seeds[0],
                "coalesced_batch_size": call_size,
                "coalesced_index": start,
            }
        )
        start = end
    return parts


def _share(exc: BaseException, members: list[_Member], owner: Hashable) -> None:
    for member in members:
        if member.owner == owner and not member.future.done():
            member.future.set_exception(type(exc)(*exc.args))


def _fitting_prefix(members: list[_Member], fits: Callable[[int], bool]) -> list[_Member]:
    best, total = 0, 0
    for index, member in enumerate(members):
        total += len(member.seeds)
        if fits(total):
            best = index + 1
    return members[:best]


async def _render_group(
    group: list[_Member],
    run_batch: Callable[..., Awaitable[dict[str, Any]]],
    shared_outcome: Callable[[BaseException], bool],
    splittable: Optional[Callable[[BaseException], bool]],
    salvage: Optional[Callable[[BaseException], Optional[dict[str, Any]]]],
    fits: Optional[Callable[[int], bool]] = None,
    adapter: Any = None,
) -> dict[int, dict[str, Any]]:
    """One call for ``group``; on failure keep finished images and split the rest by member. Sizes respect ``fits``.
    ``adapter`` (diffusion_batch_adapters.BatchAdapter) runs the call and splits it; None = the list call."""
    group = [member for member in group if not member.future.done()]
    if fits is not None:
        group = _fitting_prefix(group, fits)
    if len(group) < 2:
        return {}
    prompts = [member.prompt for member in group for _ in member.seeds]
    seeds = [s for member in group for s in member.seeds]
    owners = [member.owner for member in group]
    try:
        if adapter is not None:
            result = await adapter.denoise_batched(run_batch, prompts, seeds, owners)
            parts = adapter.split(result, group)
        else:
            result = await run_batch(prompts, seeds, owners = owners)
            parts = _split(result, group)
        trace.coalesced_call(len(group), len(seeds))
        return {id(m): part for m, part in zip(group, parts)}
    except Exception as exc:  # noqa: BLE001 -- keep what finished, split what can be split
        if shared_outcome(exc):
            raise
        done: dict[int, dict[str, Any]] = {}
        finished = salvage(exc) if salvage is not None else None
        if finished:
            try:
                parts = _split(finished, group, prefix = True)
            except ValueError:  # noqa: PERF203 -- a malformed salvage keeps nothing
                parts = []
            done.update((id(m), part) for m, part in zip(group, parts))
        rest = [member for member in group if id(member) not in done]
        logger.info(
            "images.batch_failed: %s; %d of %d requests kept their finished images",
            exc,
            len(done),
            len(group),
        )
        split = len(rest) > 1 and splittable is not None and splittable(exc)
        # The traceback pins the failed call's frames and every tensor they hold (an OOM's activations): drop it
        # before the halves run, or they start with that memory still allocated.
        _drop_tracebacks(exc)
        del exc
    if not split:
        return done
    middle = (len(rest) + 1) // 2
    for half in (rest[:middle], rest[middle:]):
        done.update(
            await _render_group(half, run_batch, shared_outcome, splittable, salvage, fits, adapter)
        )
    return done


def _drop_tracebacks(exc: BaseException) -> None:
    seen: set = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        cur.__traceback__ = None
        cur = cur.__cause__ or cur.__context__


def _close(batch: _Batch) -> None:
    if _OPEN.get(batch.key) is batch:
        del _OPEN[batch.key]


def _leave(batch: _Batch, member: _Member) -> None:
    if not batch.fired and member in batch.members:
        batch.members.remove(member)
        batch.images -= len(member.seeds)
    if not member.future.done():
        member.future.cancel()


async def _await_member(
    batch: _Batch, member: _Member, check: Optional[Callable[[], None]]
) -> dict[str, Any]:
    try:
        while True:
            # A Stop never hands back images.
            if check is not None:
                check()
            if member.future.done():
                return member.future.result()
            await asyncio.wait({member.future}, timeout = POLL_S)
    except _RunAlone:
        raise
    except BaseException:
        _leave(batch, member)
        raise


async def generate(
    key: Hashable,
    prompt: str,
    n: int,
    seed: Optional[int],
    *,
    on_rejoin: Optional[Callable[[], None]] = None,
    **kwargs: Any,
) -> Optional[dict[str, Any]]:
    """None = render alone. A member that a fired batch left out (its size was not admitted) coalesces again behind
    that batch, at most ``REJOIN_ROUNDS`` times; ``on_rejoin`` runs first (the caller's busy signal then waits for the
    generation running now). Keyword arguments as ``_generate_once``."""
    for _ in range(REJOIN_ROUNDS + 1):
        result = await _generate_once(key, prompt, n, seed, **kwargs)
        if result is not _LEFT_OUT:
            return result
        trace.count("rejoined")
        if on_rejoin is not None:
            on_rejoin()
    trace.fallback("left out of the call")
    return None


async def _generate_once(
    key: Hashable,
    prompt: str,
    n: int,
    seed: Optional[int],
    *,
    busy: Callable[[], bool],
    run_batch: Callable[[list[str], list[int]], Awaitable[dict[str, Any]]],
    shared_outcome: Callable[[BaseException], bool] = lambda exc: False,
    owner: Hashable = None,
    check: Optional[Callable[[], None]] = None,
    ride: Optional[Callable[[], ContextManager[Any]]] = None,
    fits: Optional[Callable[[int], bool]] = None,
    splittable: Optional[Callable[[BaseException], bool]] = None,
    salvage: Optional[Callable[[BaseException], Optional[dict[str, Any]]]] = None,
    on_lead: Optional[Callable[[], None]] = None,
    wait: Optional[Callable[[], Awaitable[None]]] = None,
    seeded_ok: Optional[Callable[[], bool]] = None,
    adapter: Any = None,
) -> Optional[dict[str, Any]]:
    """None = render alone. ``check`` raises when a joined member must leave. ``seeded_ok``: an explicitly seeded
    request may join (a measured neighbour-exact path); the env switch still opts every seeded request in."""
    cap = batch_max()
    if cap <= 1 or n >= cap:
        trace.fallback("over the batch cap")
        return None
    if (
        seed is not None
        and not seeded_join_enabled()
        and not (seeded_ok is not None and seeded_ok())
    ):
        trace.fallback("explicit seed")
        return None
    loop = asyncio.get_running_loop()
    batch = _OPEN.get(key)
    if batch is not None:
        if batch.images + n <= cap:
            member = _Member(prompt, request_seeds(seed, n), loop.create_future(), owner)
            batch.members.append(member)
            batch.images += n
            trace.count("joined")
            if batch.images >= cap:
                _close(batch)
            try:
                with ride() if ride is not None else nullcontext():
                    return await _await_member(batch, member, check)
            except _RunAlone as exc:
                if exc.args == (_LEFT_OUT_MSG,):
                    return _LEFT_OUT
                trace.fallback("render failed, retried alone")
                return None
        _close(batch)
    if not busy():
        trace.fallback("slot free")
        return None

    batch = _OPEN[key] = _Batch(key)
    batch.members.append(_Member(prompt, request_seeds(seed, n), loop.create_future(), owner))
    batch.images = n
    trace.count("led")
    left_out: set = set()
    try:
        try:
            # Reserve the slot turn now so later arrivals cannot overtake the leader.
            if on_lead is not None:
                on_lead()
            while busy():
                if wait is not None:
                    await wait()
                else:
                    await asyncio.sleep(POLL_S)
        except Exception as exc:
            if shared_outcome(exc):
                _share(exc, batch.members[1:], owner)
            raise
        _close(batch)
        batch.fired = True
        leader = batch.members[0]
        # A blank prompt fails the whole call: render it alone.
        if not leader.prompt.strip():
            trace.fallback("blank prompt")
            return None
        members = [member for member in batch.members if member.prompt.strip()]
        if fits is not None:
            members = _fitting_prefix(members, fits)
        taken = {id(member) for member in members} if len(members) > 1 else set()
        left_out = {
            id(member)
            for member in batch.members[1:]
            if id(member) not in taken and member.prompt.strip()
        }
        # Told now, not after the call: they coalesce again behind it while it runs.
        for member in batch.members[1:]:
            if id(member) in left_out and not member.future.done():
                member.future.set_exception(_RunAlone(_LEFT_OUT_MSG))
        if len(members) <= 1:
            trace.fallback("no admitted batch size" if len(batch.members) > 1 else "nobody joined")
            return None
        logger.info(
            "images.batch: %d requests from %d accounts render as one call of %d images",
            len(members),
            len({member.owner for member in members}),
            sum(len(member.seeds) for member in members),
        )
        try:
            done = await _render_group(
                members, run_batch, shared_outcome, splittable, salvage, fits, adapter
            )
        except Exception as exc:
            _share(exc, members[1:], owner)
            raise
        for member in members[1:]:
            part = done.get(id(member))
            if part is not None and not member.future.done():
                member.future.set_result(part)
        return done.get(id(leader))
    finally:
        _close(batch)
        batch.fired = True
        for member in batch.members[1:]:
            if not member.future.done():
                member.future.set_exception(
                    _RunAlone(_LEFT_OUT_MSG) if id(member) in left_out else _RunAlone()
                )
