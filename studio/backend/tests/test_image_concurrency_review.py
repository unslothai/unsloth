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

    backend._state = types.SimpleNamespace(family = family, pipe = None)
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
        family = types.SimpleNamespace(name = "sdxl", textual_inversion = "sdxl"), pipe = None
    )
    monkeypatch.setattr(embeddings, "list_embeddings", lambda kind = None: [])
    assert engine.coalesce_plan("a fox", False)[1] is False
    monkeypatch.setattr(embeddings, "list_embeddings", lambda kind = None: ["x"])
    assert engine.coalesce_plan("a fox", False)[1] is True
