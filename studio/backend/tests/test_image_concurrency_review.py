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


def _as(name, func):
    from utils.account_context import ROLE_OWNER, AccountContext, bind_account, reset_account
    token = bind_account(AccountContext(name, name, ROLE_OWNER))
    try:
        return func()
    finally:
        reset_account(token)


def test_a_member_stop_through_the_route_reaches_its_batched_images_under_managed_accounts(
    studio, monkeypatch
):
    import httpx

    from hub.services.models import account_access

    app, backend, saved = studio

    def stop_waiting_work(account = None):
        mine = [cancel for cancel, owner in backend.riders if account is None or owner == account]
        for cancel in mine:
            cancel.set()
        return bool(mine)

    backend.stop_waiting_work = stop_waiting_work
    # Managed accounts: alice's call holds the slot, so the generation is foreign to bob.
    monkeypatch.setattr(account_access, "tracked_generation_account", current_account_id)
    monkeypatch.setattr(
        account_access, "generation_is_foreign", lambda modality: current_account_id() != "alice"
    )
    posts = [
        ("/v1/images/generations", "alice", _openai("a fox")),
        ("/v1/images/generations", "bob", _openai("a cat")),
        ("/v1/images/generations", "carol", _openai("an owl")),
    ]
    answers = []

    async def bob_stops():
        await _until(lambda: len(backend.riders) == 2)
        transport = httpx.ASGITransport(app = app)
        async with httpx.AsyncClient(transport = transport, base_url = "http://studio") as client:
            answer = await client.post(
                "/api/inference/images/generate/cancel", headers = {"x-test-account": "bob"}
            )
        answers.append(answer.json())
        await _until(lambda: _waiting() == 2)

    responses = _post_while_rendering(app, backend, posts, waiting = 3, then = bob_stops)
    assert answers == [{"cancelled": True}]
    assert [r.status_code == 200 for r in responses] == [True, False, True]
    assert sorted(record["account"] for record in saved) == ["alice", "carol"]


def test_one_stop_rule_ends_the_accounts_render_and_its_waiting_batched_work():
    from core.inference.diffusion import DiffusionBackend

    engine = DiffusionBackend()
    render, waiting, riding, other = (threading.Event() for _ in range(4))
    import contextlib

    with contextlib.ExitStack() as stack:
        _as("alice", lambda: stack.enter_context(engine._generation_slot(render)))
        _as("alice", lambda: stack.enter_context(engine.generation_waiter(waiting)))
        _as("alice", lambda: stack.enter_context(engine.generation_rider(riding)))
        _as("bob", lambda: stack.enter_context(engine.generation_waiter(other)))
        assert engine.cancel_generate(expected_account = "alice") is True
        assert render.is_set() and waiting.is_set() and riding.is_set()
        assert not other.is_set()


def test_a_stop_during_a_teardown_reaches_only_the_callers_queued_requests():
    from core.inference.diffusion import DiffusionBackend
    from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG

    engine = DiffusionBackend()
    events = {"alice": threading.Event(), "bob": threading.Event()}
    outcome: dict = {}

    def queued(name):
        def run():
            try:
                with engine._generation_slot(events[name]):
                    outcome[name] = "served"
            except RuntimeError as exc:
                outcome[name] = str(exc)

        return lambda: _as(name, run)

    with engine._model_transition_slot():
        threads = [threading.Thread(target = queued(name), daemon = True) for name in events]
        for thread in threads:
            thread.start()
        deadline = __import__("time").monotonic() + 5
        while len(engine._queued_generate_cancels) < 2:
            assert __import__("time").monotonic() < deadline
            threading.Event().wait(0.01)
        assert engine.cancel_generate(expected_account = "bob") is True
        assert events["bob"].is_set() and not events["alice"].is_set()
    for thread in threads:
        thread.join(timeout = 5)
    assert outcome == {"alice": "served", "bob": DIFFUSION_CANCELLED_MSG}


def test_a_batch_shared_across_accounts_shows_no_live_preview(studio):
    app, backend, _saved = studio
    posts = [
        ("/api/inference/images/generate", "alice", _create("a fox", live_preview = True)),
        ("/api/inference/images/generate", "bob", _create("a cat", live_preview = True)),
    ]
    assert all(r.status_code == 200 for r in _post_while_rendering(app, backend, posts, waiting = 2))
    [call] = _batched_calls(backend)
    assert call["kwargs"]["live_preview"] is False

    backend.calls.clear()
    posts = [
        ("/api/inference/images/generate", "alice", _create("a fox", live_preview = True)),
        ("/api/inference/images/generate", "alice", _create("a cat", live_preview = True)),
    ]
    assert all(r.status_code == 200 for r in _post_while_rendering(app, backend, posts, waiting = 2))
    [call] = _batched_calls(backend)
    assert call["kwargs"]["live_preview"] is True


def test_a_static_shape_load_batches_only_at_sizes_it_already_compiled(studio):
    app, backend, _saved = studio
    asked = []

    def sizes(width, height):
        asked.append((width, height))
        return frozenset({1, 2})

    backend.coalesce_batch_sizes = sizes
    posts = [
        ("/v1/images/generations", "alice", _openai("a fox")),
        ("/v1/images/generations", "bob", _openai("a cat")),
        ("/v1/images/generations", "carol", _openai("an owl")),
    ]
    responses = _post_while_rendering(app, backend, posts, waiting = 3)
    assert all(r.status_code == 200 for r in responses)
    [call] = _batched_calls(backend)
    assert call["prompts"] == ["a fox", "a cat"] and call["batch_size"] == 2
    assert [c["prompt"] for c in backend.calls if c["prompts"] is None] == ["an owl"]
    assert set(asked) == {(256, 256)}


def test_the_engine_offers_only_compiled_batch_sizes_on_a_static_load(monkeypatch):
    import core.inference.diffusion as diffusion_module
    from core.inference.diffusion import DiffusionBackend

    engine = DiffusionBackend()
    engine._state = types.SimpleNamespace(
        speed_optims = ("compiled",),
        speed_mode = "default",
        pipe = None,
        cuda_graphs = (),
        compile_cache_ctx = types.SimpleNamespace(
            shapes = {(512, 512, 1), (512, 512, 4), (768, 768, 2)}
        ),
    )
    monkeypatch.setattr(diffusion_module, "compiled_shapes_are_static", lambda pipe, mode: True)
    assert engine.coalesce_batch_sizes(512, 512) == frozenset({1, 4})
    assert engine.coalesce_batch_sizes(1024, 1024) == frozenset()
    monkeypatch.setattr(diffusion_module, "compiled_shapes_are_static", lambda pipe, mode: False)
    assert engine.coalesce_batch_sizes(512, 512) is None
    engine._state.cuda_graphs = (object(),)
    assert engine.coalesce_batch_sizes(512, 512) == frozenset({1, 4})


def test_a_batch_keeps_the_longest_arrival_prefix_that_fits(monkeypatch):
    rendering = {"on": True}
    batches = []

    async def run_batch(
        prompts,
        seeds,
        owners = None,
    ):
        batches.append(list(prompts))
        return {"images": list(prompts), "seeds": list(seeds), "seed": seeds[0]}

    async def run():
        def request(prompt, n):
            return asyncio.ensure_future(
                batching.generate(
                    "key",
                    prompt,
                    n,
                    None,
                    busy = lambda: rendering["on"],
                    run_batch = run_batch,
                    fits = lambda total: total in (1, 4),
                )
            )

        tasks = [request("a", 1), request("b", 2), request("c", 1), request("d", 2)]
        await _until(lambda: _waiting() == 4)
        rendering["on"] = False
        return await asyncio.gather(*tasks)

    results = asyncio.run(run())
    assert batches == [["a", "b", "b", "c"]]
    assert [r["images"] if r else None for r in results] == [["a"], ["b", "b"], ["c"], None]


def _queue_members(specs, run_batch, **extra):
    """Starts one request per (prompt, n) behind a running render, frees it, returns each result."""
    rendering = {"on": True}

    async def run():
        tasks = [
            asyncio.ensure_future(
                batching.generate(
                    "key",
                    prompt,
                    n,
                    None,
                    busy = lambda: rendering["on"],
                    run_batch = run_batch,
                    **extra,
                )
            )
            for prompt, n in specs
        ]
        await _until(lambda: _waiting() == len(specs))
        rendering["on"] = False
        return await asyncio.gather(*tasks)

    return asyncio.run(run())


def _echo(prompts, seeds):
    return {"images": list(prompts), "seeds": list(seeds), "seed": seeds[0]}


def test_a_failed_call_keeps_the_images_that_finished():
    calls = []

    class Failed(RuntimeError):
        pass

    async def run_batch(
        prompts,
        seeds,
        owners = None,
    ):
        calls.append(list(prompts))
        exc = Failed("CUDA error: device lost")
        exc.finished = _echo(prompts[:2], seeds[:2])
        raise exc

    results = _queue_members(
        [("a", 1), ("b", 1), ("c", 1)],
        run_batch,
        splittable = lambda exc: False,
        salvage = lambda exc: getattr(exc, "finished", None),
    )
    assert calls == [["a", "b", "c"]]
    assert [r["images"] if r else None for r in results] == [["a"], ["b"], None]


def test_one_members_value_error_never_fails_the_others():
    calls = []

    async def run_batch(
        prompts,
        seeds,
        owners = None,
    ):
        calls.append(list(prompts))
        if "bad" in prompts:
            raise ValueError("bad is not a valid prompt for this model")
        return _echo(prompts, seeds)

    results = _queue_members(
        [("a", 1), ("bad", 1), ("c", 1), ("d", 1)],
        run_batch,
        splittable = lambda exc: isinstance(exc, ValueError),
    )
    assert calls == [["a", "bad", "c", "d"], ["a", "bad"], ["c", "d"]]
    # c and d keep a shared call; a and the bad request render alone, so only the bad one sees its error.
    assert [r["images"] if r else None for r in results] == [None, None, ["c"], ["d"]]


def test_a_blank_prompt_renders_alone_and_the_rest_still_batch():
    calls = []

    async def run_batch(
        prompts,
        seeds,
        owners = None,
    ):
        calls.append(list(prompts))
        return _echo(prompts, seeds)

    results = _queue_members([("a", 1), ("   ", 1), ("c", 1)], run_batch)
    assert calls == [["a", "c"]]
    assert [r["images"] if r else None for r in results] == [["a"], None, ["c"]]


def test_the_route_keeps_finished_images_from_a_failed_call(studio):
    app, backend, saved = studio
    real_generate = backend.generate

    def generate(**kwargs):
        if kwargs.get("prompts") is None:
            return real_generate(**kwargs)
        done = real_generate(
            **{**kwargs, "prompts": kwargs["prompts"][:2], "seeds": kwargs["seeds"][:2]}
        )
        exc = RuntimeError("CUDA error: an illegal memory access was encountered")
        exc._unsloth_finished_images = {"images": done["images"], "seeds": done["seeds"]}
        raise exc

    backend.generate = generate
    posts = [
        ("/v1/images/generations", "alice", _openai("a fox")),
        ("/v1/images/generations", "bob", _openai("a cat")),
        ("/v1/images/generations", "carol", _openai("an owl")),
    ]
    responses = _post_while_rendering(app, backend, posts, waiting = 3)
    assert all(r.status_code == 200 for r in responses)
    # One partial batched call, then only the unfinished member renders again.
    assert [c["prompts"] for c in backend.calls] == [["a fox", "a cat"], None]
    assert backend.calls[1]["prompt"] == "an owl"
    assert sorted(record["account"] for record in saved) == ["alice", "bob", "carol"]
