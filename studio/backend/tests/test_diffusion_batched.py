# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the batched multi-image planning helpers (``diffusion_batched.py``).

Pure and torch-free: job resolution (prompt lists / seed lists / legacy batch_size),
chunking, OOM split, and the OOM classifier."""

from __future__ import annotations

import pytest

from core.inference.diffusion_batched import (
    MAX_BATCH_IMAGES,
    SEED_MASK,
    chunk_jobs,
    is_oom_error,
    resolve_batch_jobs,
    split_chunk,
    uniform_prompt,
)


def _draw():
    raise AssertionError("draw_seed must not be called when seed material was supplied")


# --------------------------------------------------------------------------- job resolution
def test_legacy_batch_derives_sequential_seeds():
    jobs, base = resolve_batch_jobs(
        prompt = "p", prompts = None, seed = 7, seeds = None, batch_size = 3, draw_seed = _draw
    )
    assert jobs == [("p", 7), ("p", 8), ("p", 9)]
    assert base == 7


def test_single_image_draws_a_masked_random_seed():
    jobs, base = resolve_batch_jobs(
        prompt = "p",
        prompts = None,
        seed = None,
        seeds = None,
        batch_size = 1,
        draw_seed = lambda: (1 << 60) + 5,  # over JS's safe range: must be masked
    )
    assert base == ((1 << 60) + 5) & SEED_MASK
    assert jobs == [("p", base)]


def test_prompt_list_one_job_per_prompt():
    jobs, base = resolve_batch_jobs(
        prompt = "unused",
        prompts = ["a", "b"],
        seed = 100,
        seeds = None,
        batch_size = 1,
        draw_seed = _draw,
    )
    assert jobs == [("a", 100), ("b", 101)]
    assert base == 100


def test_seed_list_one_job_per_seed():
    jobs, base = resolve_batch_jobs(
        prompt = "p", prompts = None, seed = None, seeds = [5, 6, 7], batch_size = 1, draw_seed = _draw
    )
    assert jobs == [("p", 5), ("p", 6), ("p", 7)]
    assert base == 5


def test_prompt_and_seed_lists_pair_elementwise():
    jobs, base = resolve_batch_jobs(
        prompt = "unused",
        prompts = ["a", "b"],
        seed = None,
        seeds = [9, 3],
        batch_size = 1,
        draw_seed = _draw,
    )
    assert jobs == [("a", 9), ("b", 3)]
    assert base == 9  # base seed = first per-image seed


def test_derived_seeds_stay_json_safe_at_the_cap():
    jobs, _ = resolve_batch_jobs(
        prompt = "p",
        prompts = None,
        seed = SEED_MASK,
        seeds = None,
        batch_size = 2,
        draw_seed = _draw,
    )
    assert all(0 <= s <= SEED_MASK for _, s in jobs)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(prompts = []), "non-empty"),
        (dict(prompts = ["ok", "  "]), "non-empty"),
        (dict(prompts = ["p"] * (MAX_BATCH_IMAGES + 1)), "at most"),
        (dict(seeds = []), "non-empty"),
        (dict(seeds = [1] * (MAX_BATCH_IMAGES + 1)), "at most"),
        (dict(seeds = [-1]), "between 0"),
        (dict(seeds = [SEED_MASK + 1]), "between 0"),
        (dict(prompts = ["a", "b"], seeds = [1]), "same length"),
    ],
)
def test_invalid_lists_rejected(kwargs, match):
    base = dict(prompt = "p", prompts = None, seed = None, seeds = None, batch_size = 1)
    base.update(kwargs)
    with pytest.raises(ValueError, match = match):
        resolve_batch_jobs(draw_seed = lambda: 0, **base)


# --------------------------------------------------------------------------------- chunking
def test_default_batch_size_runs_everything_in_one_forward():
    jobs = [("p", i) for i in range(8)]
    assert chunk_jobs(jobs, 1) == [jobs]


def test_explicit_batch_size_caps_each_chunk():
    jobs = [("p", i) for i in range(5)]
    chunks = chunk_jobs(jobs, 2)
    assert [len(c) for c in chunks] == [2, 2, 1]
    assert [s for c in chunks for _, s in c] == list(range(5))  # order preserved


def test_chunk_jobs_empty():
    assert chunk_jobs([], 4) == []


def test_split_chunk_halves_and_terminates():
    chunk = [("p", i) for i in range(5)]
    first, second = split_chunk(chunk)
    assert first + second == chunk
    assert len(first) == 3 and len(second) == 2  # first never smaller: splits terminate
    with pytest.raises(ValueError):
        split_chunk([("p", 0)])


def test_uniform_prompt():
    assert uniform_prompt([("a", 1), ("a", 2)]) == "a"
    assert uniform_prompt([("a", 1), ("b", 2)]) is None


# ---------------------------------------------------------------------------- OOM classifier
def test_is_oom_error_matches_class_name_and_message():
    oom_cls = type("OutOfMemoryError", (RuntimeError,), {})
    assert is_oom_error(oom_cls("boom"))
    assert is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    assert not is_oom_error(RuntimeError("shape mismatch"))
    assert not is_oom_error(ValueError("bad prompt"))
