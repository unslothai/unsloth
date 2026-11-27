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

    # A load streaming its denoiser from host: only those coalesce across requests.
    backend._state = types.SimpleNamespace(
        family = family,
        pipe = _clip_pipe(),
        offload_policy = "sequential",
        cuda_graphs = (),
        speed_optims = (),
    )
    backend.coalesce_plan = types.MethodType(DiffusionBackend.coalesce_plan, backend)
    backend.request_batching = types.MethodType(DiffusionBackend.request_batching, backend)


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
    engine._state = types.SimpleNamespace(
        family = types.SimpleNamespace(name = "z-image"), pipe = None, speed_optims = ()
    )
    assert (
        engine.coalesce_plan("a fox", False) is None
    )  # resident weights serve one request at a time
    engine._state.offload_policy = "sequential"
    assert engine.coalesce_plan("a fox", False) == ((), False)
    assert engine.coalesce_plan("a fox", True) == ((), True)
    engine._state = types.SimpleNamespace(
        family = types.SimpleNamespace(name = "sdxl", textual_inversion = "sdxl"),
        pipe = _clip_pipe(),
        offload_policy = "sequential",
        speed_optims = (),
    )
    monkeypatch.setattr(embeddings, "list_embeddings", lambda kind = None: [])
    assert engine.coalesce_plan("a fox", False)[1] is False
    monkeypatch.setattr(embeddings, "list_embeddings", lambda kind = None: ["x"])
    assert engine.coalesce_plan("a fox", False)[1] is True


def _words(text, **_kwargs):
    return {"input_ids": list(range(len(text.split())))}


class _ClipWords:
    """A CLIP-shaped tokenizer: one id per word, a comma id, bos / eos."""

    bos_token_id, eos_token_id, pad_token_id = 0, 1, None

    def get_vocab(self):
        return {",</w>": 2}

    def __call__(self, text, **_kwargs):
        return {
            "input_ids": [
                2 if word == "," else 10 + len(word) for word in text.replace(",", " , ").split()
            ]
        }


def _clip_pipe(**extra):
    return types.SimpleNamespace(tokenizer = _ClipWords(), text_encoder = object(), **extra)


def _qwen21_pipe():
    transformer = type("QwenImage21Transformer2DModel", (), {})()
    return types.SimpleNamespace(
        transformer = transformer, prompt_template_t2i = "{}", tokenizer = _words
    )


def test_minimax_h3_image_never_batches():
    from core.inference.diffusion import coalesce_prompt_key
    family = types.SimpleNamespace(name = "minimax-h3-image")
    assert coalesce_prompt_key(family, None, "a fox") is None


def test_moss_image_never_batches():
    """Without a load that runs layer-major on a card with a measured verdict, MOSS renders alone."""
    from core.inference.diffusion import coalesce_prompt_key

    family = types.SimpleNamespace(name = "moss-image")
    assert coalesce_prompt_key(family, None, "a fox") is None


def test_moss_image_renders_alone_when_its_module_cannot_import(monkeypatch):
    """coalesce_prompt_key never raises when the MOSS module (diffusers at import) is unavailable."""
    import sys

    from core.inference.diffusion import coalesce_prompt_key

    monkeypatch.delitem(sys.modules, "core.inference.diffusion_moss_image", raising = False)
    monkeypatch.setitem(sys.modules, "diffusers.configuration_utils", None)
    family = types.SimpleNamespace(name = "moss-image")
    pipe = types.SimpleNamespace(transformer = object())
    assert coalesce_prompt_key(family, pipe, "a fox") is None


def test_qwen_image_21_never_batches():
    """A batch gains about 3% (A100, INT8) while a member lands about 35 dB from its solo render."""
    from core.inference.diffusion import coalesce_prompt_key

    family = types.SimpleNamespace(name = "qwen-image-2.1")
    pipe = _qwen21_pipe()
    assert coalesce_prompt_key(family, pipe, "a red fox") is None
    assert coalesce_prompt_key(family, pipe, "a red fox in deep snow") is None


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
    pipe = _clip_pipe()
    plain = coalesce_prompt_key(family, pipe, "a red fox")
    assert plain == coalesce_prompt_key(family, pipe, "a blue car")
    assert plain != coalesce_prompt_key(family, pipe, "a (red:1.3) fox")
    assert plain != coalesce_prompt_key(family, pipe, " ".join(["word"] * 80))


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
        _wait_until(lambda: len(engine._queued_generate_cancels) >= 2)
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
        offload_policy = "sequential",
        compile_cache_ctx = types.SimpleNamespace(
            shapes = {(512, 512, 1), (512, 512, 4), (768, 768, 2)}
        ),
    )
    monkeypatch.setattr(diffusion_module, "compiled_shapes_are_static", lambda pipe, mode: True)
    assert engine.coalesce_batch_sizes(512, 512) == frozenset({1, 4})
    assert engine.coalesce_batch_sizes(1024, 1024) == frozenset()
    monkeypatch.setattr(diffusion_module, "compiled_shapes_are_static", lambda pipe, mode: False)
    assert engine.coalesce_batch_sizes(512, 512) is None
    # CUDA graphs serve serially, compiled sizes or not.
    engine._state.cuda_graphs = (object(),)
    assert engine.coalesce_batch_sizes(512, 512) == frozenset()


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


def _wait_until(predicate, polls = 1000):
    """Poll until predicate holds, failing after a fixed number of 5 ms waits (no clock comparison)."""
    for _ in range(polls):
        if predicate():
            return
        threading.Event().wait(0.005)
    assert predicate(), "condition never held"


def _hold_slot(engine):
    """A render holding the slot in a thread until the returned Event is set."""
    held, release = threading.Event(), threading.Event()

    def run():
        with engine._generation_slot(threading.Event()):
            held.set()
            release.wait(10)

    threading.Thread(target = run, daemon = True).start()
    held.wait(5)
    return release


def _queued_render(
    engine,
    served,
    name,
    context = None,
):
    import contextlib

    def run():
        with context() if context is not None else contextlib.nullcontext():
            with engine._generation_slot(threading.Event()):
                served.append(name)

    import contextvars

    # Like the image executor: the thread runs in a copy of the caller's context, reserved turn included.
    thread = threading.Thread(target = contextvars.copy_context().run, args = (run,), daemon = True)
    thread.start()
    return thread


def test_a_batch_keeps_the_place_in_line_it_took_when_it_opened():
    from core.inference.diffusion import DiffusionBackend, reserved_slot_turn

    engine = DiffusionBackend()
    served: list = []
    release = _hold_slot(engine)
    turn = engine.reserve_slot_turn()  # the batch opens behind the running render
    later = _queued_render(engine, served, "arrived later")
    _wait_until(lambda: len(engine._slot_turns) == 2)
    assert not engine.slot_turn_ready(turn)
    woken = []
    remove = engine.add_slot_listener(lambda: woken.append(1))
    release.set()
    _wait_until(lambda: engine.slot_turn_ready(turn))
    assert woken, "the slot release woke the batch"
    remove()
    with reserved_slot_turn(turn):
        batch = _queued_render(engine, served, "batch")
    batch.join(5)
    later.join(5)
    assert served == ["batch", "arrived later"]


def test_follow_up_passes_go_before_requests_waiting_in_line(monkeypatch):
    import core.inference.diffusion_engine_router as engine_router
    from core.inference import diffusion_detailer
    from core.inference.diffusion import DiffusionBackend, slot_follow_up

    engine = DiffusionBackend()
    served: list = []
    release = _hold_slot(engine)
    waiting = _queued_render(engine, served, "waiting")
    _wait_until(lambda: len(engine._slot_turns) == 1)
    follow_up = _queued_render(engine, served, "x/y cell 2", context = slot_follow_up)
    _wait_until(lambda: len(engine._slot_turns) == 2)
    release.set()
    follow_up.join(5)
    waiting.join(5)
    assert served == ["x/y cell 2", "waiting"]

    # A Detailer crop takes the lock directly; it too goes before a waiting request.
    served.clear()
    monkeypatch.setattr(engine_router, "get_active_diffusion_engine", lambda: engine)
    release = _hold_slot(engine)
    waiting = _queued_render(engine, served, "waiting")
    _wait_until(lambda: len(engine._slot_turns) == 1)
    got = []

    def detailer():
        lock = diffusion_detailer._take_generation_slot()
        served.append("detailer crop")
        lock.release()
        engine._notify_slot_listeners()
        got.append(lock)

    crop = threading.Thread(target = detailer, daemon = True)
    crop.start()
    _wait_until(lambda: len(engine._slot_turns) == 2)
    release.set()
    crop.join(5)
    waiting.join(5)
    assert served == ["detailer crop", "waiting"]


def test_a_waiting_batch_wakes_from_the_slot_instead_of_polling():
    calls = {"wait": 0}
    rendering = {"on": True}

    async def wait():
        calls["wait"] += 1
        await asyncio.sleep(0.05)
        rendering["on"] = False

    async def run_batch(
        prompts,
        seeds,
        owners = None,
    ):
        return _echo(prompts, seeds)

    async def run():
        tasks = [
            asyncio.ensure_future(
                batching.generate(
                    "key", p, 1, None, busy = lambda: rendering["on"], run_batch = run_batch, wait = wait
                )
            )
            for p in ("a", "b")
        ]
        return await asyncio.gather(*tasks)

    results = asyncio.run(run())
    assert [r["images"] for r in results] == [["a"], ["b"]]
    assert calls["wait"] == 1


def test_recipes_record_the_shared_call_size_and_position(studio):
    app, backend, saved = studio
    posts = [
        ("/api/inference/images/generate", "alice", _create("a fox", batch_size = 2)),
        ("/api/inference/images/generate", "bob", _create("a cat")),
        ("/v1/images/generations", "carol", _openai("an owl")),
    ]
    responses = _post_while_rendering(app, backend, posts[:2], waiting = 2)
    assert all(r.status_code == 200 for r in responses)
    recipes = sorted((r["account"], r["coalesced_batch_size"], r["coalesced_index"]) for r in saved)
    assert recipes == [("alice", 3, 0), ("alice", 3, 1), ("bob", 3, 2)]
    assert [image["coalesced_batch_size"] for image in responses[1].json()["images"]] == [3]

    saved.clear()
    responses = _post_while_rendering(
        app, backend, [posts[2], ("/v1/images/generations", "dave", _openai("a hen"))], waiting = 2
    )
    assert sorted((r["account"], r["coalesced_index"]) for r in saved) == [
        ("carol", 0),
        ("dave", 1),
    ]


def test_a_lone_render_records_no_shared_call():
    import routes.inference as inference_routes
    assert inference_routes._coalesced_recipe({"images": ["x"]}, 0) == {}


def test_a_coalesced_call_is_always_pinned_to_the_model_its_settings_were_read_for(studio):
    from core.inference.diffusion_families import load_identity

    app, backend, _saved = studio
    # Every field explicit and no adapters: the route itself pins no load for this request.
    posts = [
        ("/api/inference/images/generate", "alice", _create("a fox", loras = [])),
        ("/api/inference/images/generate", "bob", _create("a cat", loras = [])),
    ]
    assert all(r.status_code == 200 for r in _post_while_rendering(app, backend, posts, waiting = 2))
    [call] = _batched_calls(backend)
    status = backend.status()
    assert call["kwargs"]["expected_load"] == load_identity(
        status["repo_id"], status["base_repo"], status["family"]
    )


def test_sdxl_groups_by_the_chunks_the_weighted_encode_actually_builds():
    from core.inference.diffusion import coalesce_prompt_key

    family = types.SimpleNamespace(name = "sdxl", textual_inversion = "sdxl")
    pipe = _clip_pipe(tokenizer_2 = _ClipWords(), text_encoder_2 = object())
    # Same raw token count and both weighted, but BREAK closes a chunk: two conditioning chunks against one.
    assert coalesce_prompt_key(family, pipe, "a BREAK cat") != coalesce_prompt_key(
        family, pipe, "a (cat)"
    )
    assert coalesce_prompt_key(family, pipe, "a (cat)") == coalesce_prompt_key(
        family, pipe, "a (dog)"
    )


def test_a_random_subseed_variation_request_never_joins_a_batch(studio):
    import httpx

    app, backend, _saved = studio

    async def run():
        transport = httpx.ASGITransport(app = app)
        async with httpx.AsyncClient(transport = transport, base_url = "http://studio") as client:
            backend.rendering = True
            first = asyncio.ensure_future(
                client.post(
                    "/api/inference/images/generate",
                    json = _create("a fox"),
                    headers = {"x-test-account": "alice"},
                )
            )
            await _until(lambda: _waiting() == 1)
            variation = asyncio.ensure_future(
                client.post(
                    "/api/inference/images/generate",
                    json = _create("a cat", batch_size = 2, variation_strength = 0.3),
                    headers = {"x-test-account": "bob"},
                )
            )
            await asyncio.sleep(0.3)
            assert _waiting() == 1
            backend.rendering = False
            return await asyncio.gather(first, variation)

    responses = asyncio.run(run())
    assert all(r.status_code == 200 for r in responses), [r.text[:200] for r in responses]
    assert _batched_calls(backend) == []
    [variation_call] = [c for c in backend.calls if c["prompt"] == "a cat"]
    assert variation_call["batch_size"] == 2 and variation_call["seeds"] is None


class _SlotBackend(_Backend):
    """The fake engine over the real engine's slot line, so queued renders block in ``_generation_slot``."""

    def __init__(self):
        from core.inference.diffusion import DiffusionBackend

        super().__init__()
        self.engine = DiffusionBackend()
        for name in (
            "generation_holder",
            "generation_waiter",
            "generation_rider",
            "reserve_slot_turn",
            "release_slot_turn",
            "slot_turn_ready",
            "add_slot_listener",
            "stop_waiting_work",
        ):
            setattr(self, name, getattr(self.engine, name))

    def generate(self, **kwargs):
        with self.engine._generation_slot(threading.Event()):
            return super().generate(**kwargs)


def test_a_reserved_batch_runs_even_when_queued_requests_hold_every_generate_worker(
    studio, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor

    import httpx

    import core.inference.diffusion as diffusion_module
    import core.inference.diffusion_engine_router as engine_router
    import routes.inference as inference_routes

    app, _fake, saved = studio
    backend = _SlotBackend()
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    monkeypatch.setattr(engine_router, "get_active_diffusion_engine", lambda: backend)
    pool = ThreadPoolExecutor(max_workers = 2, thread_name_prefix = "test-generate")
    monkeypatch.setattr(inference_routes, "_IMAGE_GENERATE_EXECUTOR", pool)
    engine = backend.engine
    release = _hold_slot(engine)

    async def run():
        transport = httpx.ASGITransport(app = app)
        async with httpx.AsyncClient(transport = transport, base_url = "http://studio") as client:

            def post(account, body):
                return asyncio.ensure_future(
                    client.post(
                        "/v1/images/generations", json = body, headers = {"x-test-account": account}
                    )
                )

            batch = [post("alice", _openai("a fox"))]
            await _until(lambda: _waiting() == 1 and len(engine._slot_turns) == 1)
            batch.append(post("bob", _openai("a cat")))
            await _until(lambda: _waiting() == 2)
            # Seeded requests never join: they take both generate workers and queue behind the batch's turn.
            seeded = [
                post("carol", _openai("an owl", seed = 5)),
                post("dave", _openai("a hen", seed = 6)),
            ]
            await _until(lambda: len(engine._slot_turns) == 3)
            release.set()
            return await asyncio.wait_for(asyncio.gather(*batch, *seeded), timeout = 15)

    try:
        responses = asyncio.run(run())
    finally:
        release.set()
        # Unstick any waiter a failing run left behind the reserved turn.
        with engine._slot_turn_changed:
            engine._slot_turns.clear()
            engine._slot_turn_changed.notify_all()
        pool.shutdown(wait = False, cancel_futures = True)
    assert all(r.status_code == 200 for r in responses)
    calls = [c["prompts"] or c["prompt"] for c in backend.calls]
    assert calls[0] == ["a fox", "a cat"]
    assert sorted(calls[1:]) == ["a hen", "an owl"]
    assert len(saved) == 4


def test_an_unclaimed_reservation_cannot_stall_waiters_that_hold_every_worker():
    """The recheck's schedule: one pool runs everything, and both of its workers wait behind a reservation whose
    owner then has no worker. The waiters step past the idle reservation and the queue drains."""
    from concurrent.futures import ThreadPoolExecutor

    from core.inference.diffusion import DiffusionBackend, reserved_slot_turn

    engine = DiffusionBackend()
    started, release = threading.Event(), threading.Event()
    entered = []

    def active():
        with engine._generation_slot(threading.Event()):
            started.set()
            release.wait(5)

    def later(name):
        with engine._generation_slot(threading.Event()):
            entered.append(name)

    def earlier(turn):
        with reserved_slot_turn(turn), engine._generation_slot(threading.Event()):
            entered.append("reserved")

    with ThreadPoolExecutor(max_workers = 2) as pool:
        holder = pool.submit(active)
        assert started.wait(5)
        turn = engine.reserve_slot_turn()
        first = pool.submit(later, "later 1")
        _wait_until(lambda: len(engine._slot_turns) == 2)
        release.set()
        holder.result(timeout = 5)
        second = pool.submit(later, "later 2")
        _wait_until(lambda: len(engine._slot_turns) == 3)
        reserved = pool.submit(earlier, turn)
        for future in (first, second, reserved):
            future.result(timeout = 10)
    assert sorted(entered) == ["later 1", "later 2", "reserved"]
    assert not engine._slot_turns and not engine._unclaimed_turns


def test_sdxl_keys_break_prompts_apart_in_both_tokenizers():
    from core.inference.diffusion import coalesce_prompt_key

    family = types.SimpleNamespace(name = "sdxl", textual_inversion = "sdxl")
    pipe = _clip_pipe(tokenizer_2 = _ClipWords(), text_encoder_2 = object())
    assert coalesce_prompt_key(family, pipe, "(red) BREAK blue") != coalesce_prompt_key(
        family, pipe, "(red) blue"
    )
    # A pipe with a tokenizer but no encoder to count for never batches.
    tokenizer_only = types.SimpleNamespace(tokenizer = _ClipWords())
    assert coalesce_prompt_key(family, tokenizer_only, "(red) blue") is None


def test_failure_recovery_renders_only_admitted_sizes():
    attempts = []

    async def run_batch(
        prompts,
        seeds,
        owners = None,
    ):
        attempts.append(len(seeds))
        if len(seeds) == 4:
            raise ValueError("controlled batch failure")
        return _echo(prompts, seeds)

    results = _queue_members(
        [("0", 1), ("1", 1), ("2", 1), ("3", 1)],
        run_batch,
        fits = lambda size: size == 4,
        splittable = lambda exc: isinstance(exc, ValueError),
    )
    # The halves (size 2) are not admitted: every member falls back to rendering alone.
    assert attempts == [4]
    assert results == [None, None, None, None]


def test_status_tells_how_concurrent_requests_are_served(monkeypatch):
    import core.inference.diffusion as diffusion_module
    from core.inference.diffusion import DiffusionBackend

    engine = DiffusionBackend()
    assert engine.request_batching() is None
    engine._state = types.SimpleNamespace(
        speed_optims = (), speed_mode = "off", pipe = None, cuda_graphs = (), compile_cache_ctx = None
    )
    policy = engine.request_batching()
    assert policy["mode"] == "serial" and policy["reason"].startswith("resident weights")
    assert engine.coalesce_batch_sizes(512, 512) == frozenset()
    engine._state.offload_policy = "sequential"
    assert engine.request_batching()["mode"] == "any"
    assert engine.coalesce_batch_sizes(512, 512) is None
    engine._state.speed_optims = ("compiled",)
    monkeypatch.setattr(diffusion_module, "compiled_shapes_are_static", lambda pipe, mode: True)
    assert engine.request_batching()["mode"] == "compiled_sizes"
    engine._state.cuda_graphs = (object(),)
    policy = engine.request_batching()
    assert policy["mode"] == "serial" and "CUDA graphs" in policy["reason"]
    assert engine.coalesce_batch_sizes(512, 512) == frozenset()


def test_the_status_model_carries_request_batching():
    from models.inference import DiffusionStatusResponse
    status = DiffusionStatusResponse(request_batching = {"mode": "serial", "reason": "CUDA graphs"})
    assert status.model_dump()["request_batching"] == {"mode": "serial", "reason": "CUDA graphs"}


def test_a_leader_stop_acknowledged_after_the_engine_finished_returns_no_images(studio):
    """A Stop landing after the engine's last cancel check, while the leader is still a registered waiter, is
    answered as cancelled; the leader must then hand back no images, like its members."""
    import contextlib

    app, backend, saved = studio
    registered: list = []

    @contextlib.contextmanager
    def generation_waiter(cancel):
        entry = (cancel, current_account_id())
        registered.append(entry)
        try:
            yield
        finally:
            registered.remove(entry)

    backend.generation_waiter = generation_waiter
    real_generate = backend.generate

    def generate(**kwargs):
        result = real_generate(**kwargs)
        # What stop_waiting_work("alice") does once the render is past its final check.
        for cancel, owner in list(registered):
            if owner == "alice":
                cancel.set()
        return result

    backend.generate = generate
    posts = [
        ("/v1/images/generations", "alice", _openai("a fox")),
        ("/v1/images/generations", "bob", _openai("a cat")),
    ]
    responses = _post_while_rendering(app, backend, posts, waiting = 2)
    assert [r.status_code == 200 for r in responses] == [False, True]
    assert [record["account"] for record in saved] == ["bob"]
