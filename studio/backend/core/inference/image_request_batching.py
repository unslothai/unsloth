# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Compatible ``POST /v1/images/generations`` requests that wait for the GPU together render as one batch.

An image engine renders one request at a time, so a request that arrives while another renders waits. The
first compatible request to wait collects the ones that arrive behind it, and once the engine is free it runs
them as one call over a list of prompts and seeds, then hands each request back its own images. Each image
keeps its own generator, so its seed reproduces it as it would alone, up to batch-size-dependent kernel
numerics (see ``diffusion_batched``).

Nothing waits for more requests: one that finds the engine idle runs at once, exactly as before. A request
that does not fit the batch runs alone. If the shared call fails for any reason, every request in it runs
alone instead, so a refusal reads exactly as it would have without batching.

Pure asyncio and torch-free: the caller supplies the busy check and the call that renders a batch.
"""

from __future__ import annotations

import asyncio
import os
import secrets
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Hashable, Optional

from loggers import get_logger

from .diffusion_batched import SEED_MASK

logger = get_logger(__name__)

# Images in one batched call. A single request can already ask for 10 in one forward (n), so 4 stays well
# inside what the memory planner and the OOM backoff are built for.
BATCH_MAX_DEFAULT = 4
POLL_S = 0.05


class _RunAlone(Exception):
    """The batch this request joined did not render; the request runs on its own instead."""


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


# Batches still collecting, keyed by what their requests share. Read and written on the event loop only.
_OPEN: dict[Hashable, _Batch] = {}


def batch_max() -> int:
    try:
        return max(1, int(os.environ.get("UNSLOTH_STUDIO_IMAGE_BATCH_MAX", BATCH_MAX_DEFAULT)))
    except ValueError:
        return BATCH_MAX_DEFAULT


def request_seeds(seed: Optional[int], n: int) -> list[int]:
    """Per-image seeds as a lone request derives them: base, base+1, ...; a random base when none is sent."""
    base = seed if seed is not None else secrets.randbelow(SEED_MASK + 1)
    return [(base + i) & SEED_MASK for i in range(n)]


def _split(result: dict[str, Any], members: list[_Member]) -> list[dict[str, Any]]:
    """Each member's own images and seeds, with its base seed as the batch seed a lone request reports."""
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
) -> Optional[dict[str, Any]]:
    """This request's result from a shared batch, or None for the caller to render it alone: the engine is
    idle, the request does not fit, it led a batch nobody joined, or the shared call failed."""
    cap = batch_max()
    if cap <= 1 or n >= cap:
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
        while busy():
            await asyncio.sleep(POLL_S)
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
        except Exception as exc:  # noqa: BLE001 -- each request then runs alone and reports its own outcome
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
