# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Coalescing review fixes: per-account resources, Stop, previews, compiled shapes, partial failures, FIFO
follow-ups, recipes and per-family prompt grouping."""

from __future__ import annotations

import asyncio
import sys
import threading
import types
from pathlib import Path

import pytest

from core.inference import image_request_batching as batching

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_image_concurrency import (  # noqa: E402,F401  (fixture)
    _Backend,
    _batched_calls,
    _create,
    _openai,
    _post_while_rendering,
    _until,
    _waiting,
    studio,
)
from utils.account_context import current_account_id  # noqa: E402


def _real_plan(backend, family):
    """Bind the real engine's coalesce_plan onto the fake, over a stand-in load state."""
    from core.inference.diffusion import DiffusionBackend

    backend._state = types.SimpleNamespace(
        family = family, pipe = types.SimpleNamespace(tokenizer = _words)
    )
    backend.coalesce_plan = types.MethodType(DiffusionBackend.coalesce_plan, backend)


def test_requests_with_loras_batch_only_with_their_own_account(studio):
    app, backend, _saved = studio
    lora = {"loras": [{"id": "style-lora", "weight": 0.8}]}
    posts = [
        ("/v1/images/generations", "alice", _openai("a fox", **lora)),
        ("/v1/images/generations", "bob", _openai("a cat", **lora)),
        ("/v1/images/generations", "alice", _openai("an owl", **lora)),
    ]
    responses = _post_while_rendering(app, backend, posts, waiting = 3)
    assert all(r.status_code == 200 for r in responses), [r.text for r in responses]
    [call] = _batched_calls(backend)
    assert call["account"] == "alice" and sorted(call["prompts"]) == ["a fox", "an owl"]
    [alone] = [c for c in backend.calls if c["prompts"] is None]
    assert alone["account"] == "bob"


def test_an_account_with_embeddings_never_shares_a_call_with_another_account(studio, monkeypatch):
    import core.inference.diffusion_embeddings as embeddings

    app, backend, _saved = studio
    _real_plan(backend, types.SimpleNamespace(name = "sdxl", textual_inversion = "sdxl"))
    owned = {"bob": ["secret-style"]}
    monkeypatch.setattr(
        embeddings, "list_embeddings", lambda kind = None: owned.get(current_account_id(), [])
    )
    posts = [
        ("/v1/images/generations", "alice", _openai("a fox")),
        ("/v1/images/generations", "bob", _openai("a cat in secret-style")),
        ("/v1/images/generations", "carol", _openai("an owl")),
    ]
    responses = _post_while_rendering(app, backend, posts, waiting = 3)
    assert all(r.status_code == 200 for r in responses)
    [call] = _batched_calls(backend)
    assert sorted(call["prompts"]) == ["a fox", "an owl"]
    assert [c["account"] for c in backend.calls if c["prompts"] is None] == ["bob"]


def test_the_engine_plan_marks_loras_and_owned_embeddings_private(monkeypatch):
    import core.inference.diffusion_embeddings as embeddings
    from core.inference.diffusion import DiffusionBackend

    engine = DiffusionBackend()
    assert engine.coalesce_plan("a fox", False) is None  # nothing loaded
    engine._state = types.SimpleNamespace(family = types.SimpleNamespace(name = "z-image"), pipe = None)
    assert engine.coalesce_plan("a fox", False) == ((), False)
    assert engine.coalesce_plan("a fox", True) == ((), True)
    engine._state = types.SimpleNamespace(
        family = types.SimpleNamespace(name = "sdxl", textual_inversion = "sdxl"),
        pipe = types.SimpleNamespace(tokenizer = _words),
    )
    monkeypatch.setattr(embeddings, "list_embeddings", lambda kind = None: [])
    assert engine.coalesce_plan("a fox", False)[1] is False
    monkeypatch.setattr(embeddings, "list_embeddings", lambda kind = None: ["x"])
    assert engine.coalesce_plan("a fox", False)[1] is True


def _words(text, **_kwargs):
    return {"input_ids": list(range(len(text.split())))}


def _qwen21_pipe():
    transformer = type("QwenImage21Transformer2DModel", (), {})()
    return types.SimpleNamespace(
        transformer = transformer, prompt_template_t2i = "{}", tokenizer = _words
    )


def test_minimax_h3_image_never_batches():
    from core.inference.diffusion import coalesce_prompt_key
    family = types.SimpleNamespace(name = "minimax-h3-image")
    assert coalesce_prompt_key(family, None, "a fox") is None


def test_qwen_image_21_groups_only_equal_prompt_token_lengths():
    from core.inference.diffusion import coalesce_prompt_key

    family = types.SimpleNamespace(name = "qwen-image-2.1")
    pipe = _qwen21_pipe()
    short, same, longer = ("a red fox", "a blue car", "a red fox in deep snow")
    assert coalesce_prompt_key(family, pipe, short) == coalesce_prompt_key(family, pipe, same)
    assert coalesce_prompt_key(family, pipe, short) != coalesce_prompt_key(family, pipe, longer)


def test_flux1_groups_only_members_in_the_same_t5_bucket():
    from core.inference.diffusion import coalesce_prompt_key
    for name in ("flux.1", "flux.1-kontext"):
        family = types.SimpleNamespace(name = name)
        pipe = types.SimpleNamespace(tokenizer_2 = _words)
        short = coalesce_prompt_key(family, pipe, "a fox")
        assert short == coalesce_prompt_key(family, pipe, "a cat on a mat")
        assert short != coalesce_prompt_key(family, pipe, " ".join(["word"] * 300))


def test_sdxl_groups_only_matching_weighting_and_chunk_count():
    from core.inference.diffusion import coalesce_prompt_key

    family = types.SimpleNamespace(name = "sdxl", textual_inversion = "sdxl")
    pipe = types.SimpleNamespace(tokenizer = _words)
    plain = coalesce_prompt_key(family, pipe, "a red fox")
    assert plain == coalesce_prompt_key(family, pipe, "a blue car")
    assert plain != coalesce_prompt_key(family, pipe, "a (red:1.3) fox")
    assert plain != coalesce_prompt_key(family, pipe, " ".join(["word"] * 80))


def test_qwen_image_21_members_of_different_lengths_render_apart_through_the_route(studio):
    app, backend, _saved = studio
    _real_plan(backend, types.SimpleNamespace(name = "qwen-image-2.1"))
    backend._state.pipe = _qwen21_pipe()
    posts = [
        ("/v1/images/generations", "alice", _openai("a red fox")),
        ("/v1/images/generations", "bob", _openai("a red fox in deep snow")),
        ("/v1/images/generations", "carol", _openai("a blue car")),
    ]
    responses = _post_while_rendering(app, backend, posts, waiting = 3)
    assert all(r.status_code == 200 for r in responses)
    [call] = _batched_calls(backend)
    assert sorted(call["prompts"]) == ["a blue car", "a red fox"]
    assert [c["prompt"] for c in backend.calls if c["prompts"] is None] == [
        "a red fox in deep snow"
    ]
