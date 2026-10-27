# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/v1/images/generations requests that wait for the GPU together render as one batch; the backend is faked."""

from __future__ import annotations

import asyncio

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
    """A diffusers-like engine whose render the test holds open with ``rendering``."""

    def __init__(self) -> None:
        self.rendering = False
        self.fail_lists = False
        self.calls: list[dict] = []
        self.is_loaded = True

    def status(self) -> dict:
        return {
            "loaded": True,
            "repo_id": "unsloth/Z-Image-Turbo-GGUF",
            "family": "z-image",
            "base_repo": None,
        }

    def generation_running(self) -> bool:
        return self.rendering

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
        self.calls.append(
            dict(prompt = prompt, prompts = prompts, seeds = seeds, seed = seed, batch_size = batch_size)
        )
        if prompts is not None and self.fail_lists:
            raise RuntimeError("CUDA out of memory")
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
    """Send ``bodies`` at once while a render holds the engine, release it once ``waiting`` requests wait."""

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
        _b64({"prompt": "a fox", "seed": 5, "n": 2}),
        _b64({"prompt": "a cat", "seed": 9}),
        _b64({"prompt": "an owl", "seed": 20}),
    ]
    responses = _post_while_rendering(app, backend, bodies, waiting = 3)
    assert [len(r.json()["data"]) for r in responses] == [2, 1, 1]
    # One call; requests join in the order they reach the batch, each keeping its images together.
    [call] = backend.calls
    assert call["batch_size"] == 4 and call["seed"] is None
    assert call["prompt"] == call["prompts"][0]
    assert sorted(zip(call["prompts"], call["seeds"])) == [
        ("a cat", 9),
        ("a fox", 5),
        ("a fox", 6),
        ("an owl", 20),
    ]
    # Each request's gallery entries carry its own prompt, n, seeds and base seed, as a lone request's would.
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
    assert by_prompt == {
        "a fox": [("a fox#5", 5, 5, 2, 0), ("a fox#6", 6, 5, 2, 1)],
        "a cat": [("a cat#9", 9, 9, 1, 0)],
        "an owl": [("an owl#20", 20, 20, 1, 0)],
    }


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
    bodies = [_b64({"prompt": "a fox", "seed": 1}), _b64({"prompt": "a cat", "seed": 2})]
    responses = _post_while_rendering(app, backend, bodies, waiting = 2)
    assert [r.status_code for r in responses] == [200, 200]
    assert sorted(backend.calls[0]["prompts"]) == ["a cat", "a fox"]
    assert [call["prompts"] for call in backend.calls[1:]] == [None, None]
    assert sorted(call["seed"] for call in backend.calls[1:]) == [1, 2]


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
                    "key", prompt, n, 1, busy = lambda: rendering["on"], run_batch = run_batch
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


def test_the_diffusers_engine_reports_running_only_while_a_generation_holds_its_slot():
    import threading

    from core.inference.diffusion import DiffusionBackend

    engine = DiffusionBackend()
    assert engine.generation_running() is False
    with engine._generation_slot(threading.Event()):
        assert engine.generation_running() is True
    assert engine.generation_running() is False
