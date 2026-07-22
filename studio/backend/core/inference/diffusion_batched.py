# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Batched multi-image planning for the local diffusion backend.

One generation call can produce N images three ways: a prompt LIST (one image per
prompt), one prompt x a seed LIST, or the legacy single prompt + ``batch_size``
(whose per-image seeds derive as base..base+batch_size-1, matching the native
sd.cpp engine and the gallery recipe replay). These helpers turn the request into
an explicit per-image ``(prompt, seed)`` job list, chunk it into per-forward
batches, and support OOM backoff by splitting a failed chunk in half.

Measured on the 32-image eval suites (diffusers 0.39 / torch 2.10): one batched
forward with per-image ``torch.Generator``s is numerics-safe (LPIPS deltas within
0.002 of serial) and 10-22x faster end-to-end than serial per-image engines --
batch 32 fits 4-step 12B-class models on one GPU, batch 8 fits a 20B model at
1024px with CFG batching. Per-image generators keep every image individually
reproducible: same-seed images within the same batch shape are bit-identical
once the compiled graph is settled (the very first generation during an
in-flight deferred compile can deviate transiently by a few ulps);
regenerating an image alone with its recorded seed reproduces it up to
batch-size-dependent kernel numerics (measured mean abs pixel delta ~2.5/255,
LPIPS delta under 0.002), not bit-exactly.

Pure and torch-free so the CPU unit tests stay light; the engine owns the
torch.Generator construction and the actual pipeline calls.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

# Upper bound on images per generation call (mirrors the route's batch_size cap):
# a prompt/seed list beyond this is a client error, not an OOM to back off from.
MAX_BATCH_IMAGES = 32

# Seeds stay in JS's safe-integer range so they round-trip through the JSON
# gallery recipes and reproduce the image (a raw 64-bit seed loses precision).
SEED_MASK = (1 << 53) - 1


def resolve_batch_jobs(
    *,
    prompt: str,
    prompts: Optional[list[str]],
    seed: Optional[int],
    seeds: Optional[list[int]],
    batch_size: int,
    draw_seed: Callable[[], int],
) -> tuple[list[tuple[str, int]], int]:
    """The per-image ``(prompt, seed)`` jobs plus the base seed for this call.

    - ``prompts`` (list): one image per prompt. With ``seeds`` too, lengths must
      match (seed i drives prompt i); without, seeds derive from the base.
    - ``seeds`` (list) alone: one image per seed, all with ``prompt``.
    - neither: ``batch_size`` images of ``prompt`` with derived seeds
      base..base+batch_size-1 (each masked JSON-safe).

    ``draw_seed`` supplies a fresh random base when the caller sent none (the
    engine passes a ``torch.Generator`` draw). Raises ``ValueError`` on empty /
    oversized lists, a length mismatch, or an out-of-range seed."""
    if prompts is not None:
        if not prompts or not all(isinstance(p, str) and p.strip() for p in prompts):
            raise ValueError("prompts must be a non-empty list of non-empty strings")
        if len(prompts) > MAX_BATCH_IMAGES:
            raise ValueError(f"prompts supports at most {MAX_BATCH_IMAGES} entries per call")
    if seeds is not None:
        if not seeds:
            raise ValueError("seeds must be a non-empty list of integers")
        if len(seeds) > MAX_BATCH_IMAGES:
            raise ValueError(f"seeds supports at most {MAX_BATCH_IMAGES} entries per call")
        seeds = [int(s) for s in seeds]
        if any(s < 0 or s > SEED_MASK for s in seeds):
            raise ValueError("every seed must be between 0 and 2**53 - 1 (JSON-safe)")
        if prompts is not None and len(seeds) != len(prompts):
            raise ValueError(
                f"prompts and seeds must have the same length "
                f"(got {len(prompts)} prompts, {len(seeds)} seeds)"
            )

    if prompts is not None:
        count = len(prompts)
    elif seeds is not None:
        count = len(seeds)
    else:
        count = max(1, int(batch_size))

    if seeds is not None:
        job_seeds = seeds
        base_seed = seeds[0]
    else:
        base_seed = int(seed) if seed is not None else int(draw_seed()) & SEED_MASK
        job_seeds = [(base_seed + i) & SEED_MASK for i in range(count)]

    job_prompts = prompts if prompts is not None else [prompt] * count
    return list(zip(job_prompts, job_seeds)), base_seed


def chunk_jobs(
    jobs: list[tuple[str, int]], batch_size: int
) -> list[list[tuple[str, int]]]:
    """Split the jobs into per-forward chunks.

    ``batch_size`` doubles as the per-forward cap when a prompt/seed list drives
    the image count: an explicit ``batch_size > 1`` bounds each forward, while
    the untouched default (1) lets the whole list run as ONE forward -- the
    measured sweet spot (batch 32 on 4-step models) -- with OOM backoff as the
    safety net rather than a serial default."""
    if not jobs:
        return []
    per_forward = len(jobs) if batch_size <= 1 else min(int(batch_size), len(jobs))
    return [jobs[i : i + per_forward] for i in range(0, len(jobs), per_forward)]


def split_chunk(
    chunk: list[tuple[str, int]],
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Halve a chunk for OOM backoff (first half never smaller than the second,
    so repeated splits terminate at singletons). Raises on an unsplittable chunk."""
    if len(chunk) < 2:
        raise ValueError("cannot split a chunk of fewer than 2 jobs")
    mid = (len(chunk) + 1) // 2
    return chunk[:mid], chunk[mid:]


def uniform_prompt(chunk: list[tuple[str, int]]) -> Optional[str]:
    """The chunk's single shared prompt, or None when prompts differ.

    A uniform chunk encodes its prompt ONCE (``num_images_per_prompt`` fans it
    out); a mixed chunk passes the prompt list with one image per prompt."""
    first = chunk[0][0]
    return first if all(p == first for p, _ in chunk) else None


def is_oom_error(exc: BaseException) -> bool:
    """Whether an exception is a CUDA/accelerator out-of-memory, worth a smaller
    retry. Matched structurally (class name across torch versions / devices) and
    by message, so the caller needn't import torch to classify."""
    for klass in type(exc).__mro__:
        if klass.__name__ == "OutOfMemoryError":
            return True
    return "out of memory" in str(exc).lower()
