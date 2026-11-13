# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Batches waiting /v1/images/generations requests; pixels vary with batch size (diffusion_batched)."""

from __future__ import annotations

import asyncio
import os
import secrets
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Hashable, Optional

from loggers import get_logger

from .diffusion_batched import SEED_MASK

logger = get_logger(__name__)

BATCH_MAX_DEFAULT = 4
POLL_S = 0.05


class _RunAlone(Exception):
    """Retry this request individually."""


@dataclass
class _Member:
    prompt: str
    seeds: list[int]
    future: "asyncio.Future[dict[str, Any]]"


@dataclass
class _Batch:
    key: Hashable
    members: list[_Member] = field(default_factory = list)
    images: int = 0


_OPEN: dict[Hashable, _Batch] = {}


def batch_max() -> int:
    try:
        return max(1, int(os.environ.get("UNSLOTH_STUDIO_IMAGE_BATCH_MAX", BATCH_MAX_DEFAULT)))
    except ValueError:
        return BATCH_MAX_DEFAULT


def request_seeds(seed: Optional[int], n: int) -> list[int]:
    base = seed if seed is not None else secrets.randbelow(SEED_MASK + 1)
    return [(base + i) & SEED_MASK for i in range(n)]


def _split(result: dict[str, Any], members: list[_Member]) -> list[dict[str, Any]]:
    images = list(result.get("images") or [])
    seeds = list(result.get("seeds") or [])
    wanted = sum(len(member.seeds) for member in members)
    if len(images) != wanted or (seeds and len(seeds) != wanted):
        raise ValueError(f"a batch of {wanted} images returned {len(images)}")
    parts = []
    start = 0
    for member in members:
        end = start + len(member.seeds)
        parts.append(
            {
                **result,
                "images": images[start:end],
                "seeds": seeds[start:end] if seeds else list(member.seeds),
                "seed": member.seeds[0],
            }
        )
        start = end
    return parts


def _share(exc: BaseException, members: list[_Member]) -> None:
    for member in members:
        if not member.future.done():
            member.future.set_exception(type(exc)(*exc.args))


def _close(batch: _Batch) -> None:
    if _OPEN.get(batch.key) is batch:
        del _OPEN[batch.key]


async def generate(
    key: Hashable,
    prompt: str,
    n: int,
    seed: Optional[int],
    *,
    busy: Callable[[], bool],
    run_batch: Callable[[list[str], list[int]], Awaitable[dict[str, Any]]],
    shared_outcome: Callable[[BaseException], bool] = lambda exc: False,
) -> Optional[dict[str, Any]]:
    cap = batch_max()
    # An explicit seed renders alone: a batch's pixels differ slightly from a lone render's.
    if cap <= 1 or n >= cap or seed is not None:
        return None
    loop = asyncio.get_running_loop()
    batch = _OPEN.get(key)
    if batch is not None:
        if batch.images + n > cap:
            return None
        member = _Member(prompt, request_seeds(seed, n), loop.create_future())
        batch.members.append(member)
        batch.images += n
        try:
            return await member.future
        except _RunAlone:
            return None
    if not busy():
        return None

    batch = _OPEN[key] = _Batch(key)
    batch.members.append(_Member(prompt, request_seeds(seed, n), loop.create_future()))
    batch.images = n
    try:
        try:
            while busy():
                await asyncio.sleep(POLL_S)
        except Exception as exc:
            if shared_outcome(exc):
                _share(exc, batch.members[1:])
            raise
        _close(batch)
        members = list(batch.members)
        if len(members) == 1:
            return None
        prompts = [member.prompt for member in members for _ in member.seeds]
        seeds = [s for member in members for s in member.seeds]
        logger.info(
            "openai_images.batch: %d requests render as one call of %d images",
            len(members),
            len(seeds),
        )
        try:
            parts = _split(await run_batch(prompts, seeds), members)
        except Exception as exc:  # noqa: BLE001 -- fall back unless the outcome applies to every request
            if shared_outcome(exc):
                _share(exc, members[1:])
                raise
            logger.info("openai_images.batch_failed: %s; each request runs alone", exc)
            return None
        for member, part in zip(members[1:], parts[1:]):
            if not member.future.done():
                member.future.set_result(part)
        return parts[0]
    finally:
        _close(batch)
        for member in batch.members[1:]:
            if not member.future.done():
                member.future.set_exception(_RunAlone())
