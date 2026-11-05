# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


from __future__ import annotations

import asyncio
import contextlib

import httpx
import pytest
from fastapi import FastAPI

import core.inference.diffusion as diffusion_module
import core.inference.diffusion_engine_router as engine_router
import core.inference.image_gallery as gallery_module
from auth.authentication import get_current_subject
from core.inference import image_request_batching as batching
from core.inference.diffusion_batched import SEED_MASK
from routes.inference import router
from utils.api_errors import install_api_error_handlers


class _Backend:
    def __init__(self) -> None:
        self.rendering = False
        self.fail_lists = False
        self.cancel_lists = False
        self.calls: list[dict] = []
        self.waiters: list = []
        self.before_generate = None
        self.is_loaded = True
        self.holder = object()
        self.active_cancel = None

    def status(self) -> dict:
        return {
            "loaded": True,
            "repo_id": "unsloth/Z-Image-Turbo-GGUF",
            "family": "z-image",
            "base_repo": None,
        }

    def generation_holder(self):
        return self.holder if self.rendering else None

    @contextlib.contextmanager
    def generation_waiter(self, cancel):
        self.waiters.append(cancel)
        try:
            yield
        finally:
            self.waiters.remove(cancel)

    def stop_during_a_model_change(self) -> None:
        for cancel in list(self.waiters):
            cancel.set()

    def generate(
        self,
        *,
        prompt,
        width,
        height,
        steps,
        guidance,
        batch_size = 1,
        expected_load = None,
        seed = None,
        negative_prompt = None,
        prompts = None,
        seeds = None,
    ) -> dict:
        if self.before_generate is not None:
            self.before_generate()
        import threading

        from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG
        from core.inference.generation_ticket import bind_generation_cancel

        cancel = self.active_cancel = threading.Event()
        bind_generation_cancel(cancel)
        if cancel.is_set():
            raise RuntimeError(DIFFUSION_CANCELLED_MSG)
        self.calls.append(
            dict(prompt = prompt, prompts = prompts, seeds = seeds, seed = seed, batch_size = batch_size)
        )
        if prompts is not None and self.fail_lists:
            raise RuntimeError("CUDA out of memory")
        if prompts is not None and self.cancel_lists:
            from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG
            raise RuntimeError(DIFFUSION_CANCELLED_MSG)
        if seeds is not None:
            base, image_seeds = seeds[0], list(seeds)
        else:
            base = 4242 if seed is None else seed
            image_seeds = [base + i for i in range(batch_size)]
        labels = prompts if prompts is not None else [prompt] * len(image_seeds)
        return {
            "images": [f"{label}#{s}" for label, s in zip(labels, image_seeds)],
            "seed": base,
            "seeds": image_seeds,
            "repo_id": "unsloth/Z-Image-Turbo-GGUF",
            "negative_prompt": negative_prompt,
        }


@pytest.fixture
def studio(monkeypatch):
    backend = _Backend()
    saved: list[dict] = []

    def _save(image, meta):
        record = {**meta, "id": f"img{len(saved)}", "image": image}
        saved.append(record)
        return record

    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    monkeypatch.setattr(gallery_module, "save", _save)
    monkeypatch.setattr(gallery_module, "image_b64", lambda image_id: "QUJD")
    app = FastAPI()
    install_api_error_handlers(app)
    app.include_router(router, prefix = "/v1")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return app, backend, saved


async def _until(predicate, timeout = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition never held")
        await asyncio.sleep(0.01)


def _waiting() -> int:
    return sum(len(batch.members) for batch in batching._OPEN.values())


def _post_while_rendering(app, backend, bodies: list[dict], waiting: int) -> list[httpx.Response]:
    async def run():
        transport = httpx.ASGITransport(app = app)
        async with httpx.AsyncClient(transport = transport, base_url = "http://studio") as client:
            backend.rendering = True
            sent = [
                asyncio.ensure_future(client.post("/v1/images/generations", json = body))
                for body in bodies
            ]
            await _until(lambda: _waiting() >= waiting)
            backend.rendering = False
            return await asyncio.gather(*sent)

    return asyncio.run(run())


def _b64(body: dict) -> dict:
    return {"size": "256x256", "response_format": "b64_json", **body}


def test_waiting_compatible_requests_render_as_one_call_and_each_gets_its_own_images(studio):
    app, backend, saved = studio
    bodies = [
        _b64({"prompt": "a fox", "n": 2}),
        _b64({"prompt": "a cat"}),
        _b64({"prompt": "an owl"}),
    ]
    responses = _post_while_rendering(app, backend, bodies, waiting = 3)
    assert [len(r.json()["data"]) for r in responses] == [2, 1, 1]
    [call] = backend.calls
    assert call["batch_size"] == 4 and call["seed"] is None
    assert call["prompt"] == call["prompts"][0]
    assert sorted(call["prompts"]) == ["a cat", "a fox", "a fox", "an owl"]
    seeds = {}
    for prompt, seed in zip(call["prompts"], call["seeds"]):
        seeds.setdefault(prompt, []).append(seed)
    assert seeds["a fox"][1] == (seeds["a fox"][0] + 1) & SEED_MASK
    by_prompt = {}
    for record in saved:
        by_prompt.setdefault(record["prompt"], []).append(
            (
                record["image"],
                record["seed"],
                record["batch_seed"],
                record["batch_size"],
                record["batch_index"],
            )
        )
    fox, cat, owl = seeds["a fox"], seeds["a cat"][0], seeds["an owl"][0]
    assert by_prompt == {
        "a fox": [
            (f"a fox#{fox[0]}", fox[0], fox[0], 2, 0),
            (f"a fox#{fox[1]}", fox[1], fox[0], 2, 1),
        ],
        "a cat": [(f"a cat#{cat}", cat, cat, 1, 0)],
        "an owl": [(f"an owl#{owl}", owl, owl, 1, 0)],
    }


def test_a_request_with_a_seed_never_joins_a_batch(studio):
    app, backend, _saved = studio

    async def run():
        transport = httpx.ASGITransport(app = app)
        async with httpx.AsyncClient(transport = transport, base_url = "http://studio") as client:
            backend.rendering = True
            sent = [
                asyncio.ensure_future(client.post("/v1/images/generations", json = _b64(body)))
                for body in ({"prompt": "a fox", "seed": 5}, {"prompt": "a cat", "seed": 9})
            ]
            await asyncio.sleep(0.2)
            assert _waiting() == 0
            backend.rendering = False
            return await asyncio.gather(*sent)

    assert all(r.status_code == 200 for r in asyncio.run(run()))
    assert [call["prompts"] for call in backend.calls] == [None, None]
    assert sorted(call["seed"] for call in backend.calls) == [5, 9]


def test_requests_that_differ_in_more_than_prompt_seed_and_n_render_apart(studio):
    app, backend, _saved = studio
    bodies = [_b64({"prompt": "a fox"}), _b64({"prompt": "a cat", "size": "512x512"})]
    responses = _post_while_rendering(app, backend, bodies, waiting = 2)
    assert all(r.status_code == 200 for r in responses)
    assert len(backend.calls) == 2
    assert all(call["prompts"] is None for call in backend.calls)


def test_a_failed_shared_call_sends_every_request_back_to_render_alone(studio):
    app, backend, _saved = studio
    backend.fail_lists = True
    bodies = [_b64({"prompt": "a fox"}), _b64({"prompt": "a cat"})]
    responses = _post_while_rendering(app, backend, bodies, waiting = 2)
    assert [r.status_code for r in responses] == [200, 200]
    assert sorted(backend.calls[0]["prompts"]) == ["a cat", "a fox"]
    assert sorted(call["prompt"] for call in backend.calls[1:]) == ["a cat", "a fox"]
    assert [call["prompts"] for call in backend.calls[1:]] == [None, None]


def test_a_cancelled_batch_ends_every_request_without_rendering_again(studio):
    app, backend, saved = studio
    backend.cancel_lists = True
    bodies = [_b64({"prompt": "a fox"}), _b64({"prompt": "a cat"})]
    responses = _post_while_rendering(app, backend, bodies, waiting = 2)
    assert len(backend.calls) == 1 and backend.calls[0]["prompts"] is not None
    assert saved == []
    assert all(r.status_code >= 400 for r in responses)
    assert responses[0].status_code == responses[1].status_code


def test_a_stop_while_a_batch_waits_cancels_every_request_in_it(studio):
    app, backend, saved = studio

    async def run():
        transport = httpx.ASGITransport(app = app)
        async with httpx.AsyncClient(transport = transport, base_url = "http://studio") as client:
            backend.rendering = True
            sent = [
                asyncio.ensure_future(
                    client.post("/v1/images/generations", json = _b64({"prompt": p}))
                )
                for p in ("a fox", "a cat")
            ]
            await _until(lambda: _waiting() >= 2 and backend.waiters)
            backend.stop_during_a_model_change()
            return await asyncio.gather(*sent)

    responses = asyncio.run(run())
    assert backend.calls == [] and saved == []
    assert all(r.status_code >= 400 for r in responses)
    assert responses[0].status_code == responses[1].status_code
    assert backend.waiters == []


def test_a_stop_after_the_batch_stops_waiting_still_cancels_its_generation(studio):
    app, backend, saved = studio
    backend.before_generate = backend.stop_during_a_model_change
    bodies = [_b64({"prompt": "a fox"}), _b64({"prompt": "a cat"})]
    responses = _post_while_rendering(app, backend, bodies, waiting = 2)
    assert backend.calls == [] and saved == []
    assert all(r.status_code >= 400 for r in responses)
    assert responses[0].status_code == responses[1].status_code


def test_a_stop_during_a_real_model_change_reaches_a_generation_that_starts_after_it():
    from core.inference.diffusion import DiffusionBackend
    from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG
    from core.inference.generation_ticket import (
        GenerationTicket,
        TicketStopEvent,
        generation_ticket,
    )

    engine = DiffusionBackend()
    ticket = GenerationTicket()
    stopped = TicketStopEvent(ticket)
    with engine.generation_waiter(stopped):
        with engine._model_transition_slot():
            assert engine.cancel_generate() is True
        assert stopped.is_set()
        with generation_ticket(ticket):
            with pytest.raises(RuntimeError, match = DIFFUSION_CANCELLED_MSG):
                engine.generate(prompt = "a fox", width = 256, height = 256)
    with generation_ticket(GenerationTicket()):
        with pytest.raises(RuntimeError) as exc:
            engine.generate(prompt = "a fox", width = 256, height = 256)
    assert str(exc.value) != DIFFUSION_CANCELLED_MSG


def test_a_stop_while_a_failed_batch_unwinds_keeps_its_requests_from_rendering_alone(studio):
    app, backend, saved = studio
    backend.fail_lists = True
    real_generate = backend.generate

    def _fail_then_stop(**kwargs):
        try:
            return real_generate(**kwargs)
        finally:
            if kwargs.get("prompts") is not None:
                backend.stop_during_a_model_change()

    backend.generate = _fail_then_stop
    bodies = [_b64({"prompt": "a fox"}), _b64({"prompt": "a cat"})]
    responses = _post_while_rendering(app, backend, bodies, waiting = 2)
    assert [call["prompts"] is not None for call in backend.calls] == [True]
    assert saved == []
    assert all(r.status_code >= 400 for r in responses)
    assert responses[0].status_code == responses[1].status_code


def test_an_engine_stop_acknowledged_as_a_shared_call_fails_keeps_every_request_from_retrying(
    studio,
):
    app, backend, saved = studio
    backend.fail_lists = True
    real_generate = backend.generate

    def _fail_as_the_engine_cancels(**kwargs):
        try:
            return real_generate(**kwargs)
        finally:
            if kwargs.get("prompts") is not None:
                backend.active_cancel.set()

    backend.generate = _fail_as_the_engine_cancels
    bodies = [_b64({"prompt": "a fox"}), _b64({"prompt": "a cat"})]
    responses = _post_while_rendering(app, backend, bodies, waiting = 2)
    assert [call["prompts"] is not None for call in backend.calls] == [True]
    assert saved == []
    assert all(r.status_code >= 400 for r in responses)
    assert responses[0].json() == responses[1].json()


def test_a_batch_stops_waiting_when_the_generation_it_queued_behind_ends(studio):
    app, backend, saved = studio

    async def run():
        transport = httpx.ASGITransport(app = app)
        async with httpx.AsyncClient(transport = transport, base_url = "http://studio") as client:
            backend.rendering = True
            sent = [
                asyncio.ensure_future(
                    client.post("/v1/images/generations", json = _b64({"prompt": p}))
                )
                for p in ("a fox", "a cat")
            ]
            await _until(lambda: _waiting() >= 2)
            backend.holder = object()
            await _until(lambda: backend.calls, timeout = 2.0)
            backend.rendering = False
            return await asyncio.gather(*sent)

    responses = asyncio.run(run())
    assert all(r.status_code == 200 for r in responses)
    [call] = backend.calls
    assert sorted(call["prompts"]) == ["a cat", "a fox"]
    assert len(saved) == 2


def test_a_request_that_finds_the_engine_idle_renders_at_once_alone(studio):
    app, backend, _saved = studio

    async def run():
        transport = httpx.ASGITransport(app = app)
        async with httpx.AsyncClient(transport = transport, base_url = "http://studio") as client:
            return await client.post(
                "/v1/images/generations", json = _b64({"prompt": "a fox", "n": 2})
            )

    assert asyncio.run(run()).status_code == 200
    assert backend.calls == [
        dict(prompt = "a fox", prompts = None, seeds = None, seed = None, batch_size = 2)
    ]


@pytest.mark.parametrize("engine", ["sd_cpp", "mlx"])
def test_engines_without_prompt_lists_never_batch(studio, monkeypatch, engine):
    app, backend, _saved = studio
    monkeypatch.setattr(engine_router, "active_engine_name", lambda: engine)

    async def run():
        transport = httpx.ASGITransport(app = app)
        async with httpx.AsyncClient(transport = transport, base_url = "http://studio") as client:
            backend.rendering = True
            sent = [
                asyncio.ensure_future(
                    client.post("/v1/images/generations", json = _b64({"prompt": p}))
                )
                for p in ("a fox", "a cat")
            ]
            await asyncio.sleep(0.2)
            assert _waiting() == 0
            backend.rendering = False
            return await asyncio.gather(*sent)

    assert all(r.status_code == 200 for r in asyncio.run(run()))
    assert [call["prompts"] for call in backend.calls] == [None, None]


def test_request_seeds_derive_like_a_lone_request():
    assert batching.request_seeds(5, 3) == [5, 6, 7]
    assert batching.request_seeds(SEED_MASK, 2) == [SEED_MASK, 0]
    drawn = batching.request_seeds(None, 2)
    assert drawn[1] == (drawn[0] + 1) & SEED_MASK


def test_a_batch_stops_collecting_at_the_cap_and_the_rest_render_alone(monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_IMAGE_BATCH_MAX", "4")
    rendering = {"on": True}
    batches: list[list[str]] = []

    async def run_batch(prompts, seeds):
        batches.append(prompts)
        return {"images": list(prompts), "seeds": list(seeds), "seed": seeds[0]}

    async def run():
        def request(prompt, n):
            return asyncio.ensure_future(
                batching.generate(
                    "key", prompt, n, None, busy = lambda: rendering["on"], run_batch = run_batch
                )
            )

        lead, joined, overflow, too_big = (
            request("a", 2),
            request("b", 2),
            request("c", 1),
            request("d", 4),
        )
        await _until(lambda: overflow.done() and too_big.done())
        assert overflow.result() is None and too_big.result() is None
        rendering["on"] = False
        return await lead, await joined

    lead, joined = asyncio.run(run())
    assert batches == [["a", "a", "b", "b"]]
    assert lead["images"] == ["a", "a"] and joined["images"] == ["b", "b"]


def test_a_cap_of_one_turns_batching_off(monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_IMAGE_BATCH_MAX", "1")

    async def run():
        return await batching.generate(
            "key",
            "a",
            1,
            None,
            busy = lambda: True,
            run_batch = None,  # type: ignore[arg-type]
        )

    assert asyncio.run(run()) is None
    assert batching._OPEN == {}


def test_the_diffusers_engine_names_each_slot_holder_apart():
    import threading

    from core.inference.diffusion import DiffusionBackend

    engine = DiffusionBackend()
    assert engine.generation_holder() is None
    first, second = threading.Event(), threading.Event()
    with engine._generation_slot(first):
        assert engine.generation_holder() is first
    with engine._model_transition_slot():
        assert engine.generation_holder() not in (None, first)
    with engine._generation_slot(second):
        assert engine.generation_holder() is second
    assert engine.generation_holder() is None
