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

from .diffusion_batched import SEED_MASK

logger = get_logger(__name__)

BATCH_MAX_DEFAULT = 8
POLL_S = 0.05
# Derived from every prompt in the call: never shown across accounts.
_PROMPT_DERIVED_FIELDS = ("embeddings", "embeddings_skipped")


class _RunAlone(Exception):
    """Retry this request individually."""


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
) -> dict[int, dict[str, Any]]:
    """One call for ``group``; on failure keep finished images and split the rest by member. Sizes respect ``fits``."""
    group = [member for member in group if not member.future.done()]
    if fits is not None:
        group = _fitting_prefix(group, fits)
    if len(group) < 2:
        return {}
    prompts = [member.prompt for member in group for _ in member.seeds]
    seeds = [s for member in group for s in member.seeds]
    try:
        result = await run_batch(prompts, seeds, owners = [member.owner for member in group])
        return {id(m): part for m, part in zip(group, _split(result, group))}
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
        if len(rest) > 1 and splittable is not None and splittable(exc):
            middle = (len(rest) + 1) // 2
            for half in (rest[:middle], rest[middle:]):
                done.update(
                    await _render_group(half, run_batch, shared_outcome, splittable, salvage, fits)
                )
        return done


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
) -> Optional[dict[str, Any]]:
    """None = render alone. ``check`` raises when a joined member must leave."""
    cap = batch_max()
    if cap <= 1 or n >= cap or (seed is not None and not seeded_join_enabled()):
        return None
    loop = asyncio.get_running_loop()
    batch = _OPEN.get(key)
    if batch is not None:
        if batch.images + n <= cap:
            member = _Member(prompt, request_seeds(seed, n), loop.create_future(), owner)
            batch.members.append(member)
            batch.images += n
            if batch.images >= cap:
                _close(batch)
            try:
                with ride() if ride is not None else nullcontext():
                    return await _await_member(batch, member, check)
            except _RunAlone:
                return None
        _close(batch)
    if not busy():
        return None

    batch = _OPEN[key] = _Batch(key)
    batch.members.append(_Member(prompt, request_seeds(seed, n), loop.create_future(), owner))
    batch.images = n
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
            return None
        members = [member for member in batch.members if member.prompt.strip()]
        if fits is not None:
            members = _fitting_prefix(members, fits)
        if len(members) <= 1:
            return None
        logger.info(
            "images.batch: %d requests from %d accounts render as one call of %d images",
            len(members),
            len({member.owner for member in members}),
            sum(len(member.seeds) for member in members),
        )
        try:
            done = await _render_group(
                members, run_batch, shared_outcome, splittable, salvage, fits
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
                member.future.set_exception(_RunAlone())
