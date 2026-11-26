# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Concurrent image users: requests from different accounts share one batched call, each gets only its own images,
and one member's Stop never ends another account's request."""

from __future__ import annotations

import asyncio
import contextlib
import threading

import httpx
import pytest
from fastapi import FastAPI

import core.inference.diffusion as diffusion_module
import core.inference.diffusion_engine_router as engine_router
import core.inference.gpu_arbiter as gpu_arbiter
import core.inference.image_gallery as gallery_module
from auth.authentication import authenticated_via_api_key, get_current_subject
from core.inference import image_request_batching as batching
from routes.inference import router, studio_router
from utils.account_context import (
    ROLE_OWNER,
    AccountContext,
    bind_account,
    current_account_id,
    reset_account,
)
from utils.api_errors import install_api_error_handlers


class _Backend:
    """Diffusers-shaped fake: one slot, per-account Stop, batched calls labelled by prompt and seed."""

    def __init__(self) -> None:
        self.rendering = False
        self.holder = object()
        self.calls: list[dict] = []
        self.waiters: list = []
        self.riders: list[tuple[threading.Event, str]] = []
        self.active_cancel = None
        self.active_account = None
        self.is_loaded = True
        self.during_batch = None

    def status(self) -> dict:
        return {
            "loaded": True,
            "repo_id": "unsloth/Z-Image-Turbo-GGUF",
            "family": "z-image",
            "base_repo": None,
        }

    def load_progress(self) -> dict:
        return {"phase": "ready"}

    def generation_holder(self):
        return self.holder if self.rendering else None

    @contextlib.contextmanager
    def generation_waiter(self, cancel):
        self.waiters.append(cancel)
        try:
            yield
        finally:
            self.waiters.remove(cancel)

    @contextlib.contextmanager
    def generation_rider(self, cancel):
        entry = (cancel, current_account_id())
        self.riders.append(entry)
        try:
            yield
        finally:
            self.riders.remove(entry)

    def cancel_generate(self, expected_account = None) -> bool:
        if self.active_cancel is not None and expected_account in (None, self.active_account):
            self.active_cancel.set()
            return True
        mine = [cancel for cancel, owner in self.riders if owner == expected_account]
        for cancel in mine:
            cancel.set()
        return bool(mine)

    def generate(
        self,
        *,
        prompt,
        seed = None,
        batch_size = 1,
        prompts = None,
        seeds = None,
        sd_controls = None,
        live_preview = None,
        **kwargs,
    ):
        kwargs.update(sd_controls = sd_controls, live_preview = live_preview)
        from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG
        from core.inference.generation_ticket import bind_generation_cancel

        cancel = threading.Event()
        bind_generation_cancel(cancel)
        self.active_cancel, self.active_account = cancel, current_account_id()
        self.calls.append(
            dict(
                account = current_account_id(),
                prompt = prompt,
                prompts = prompts,
                seeds = seeds,
                seed = seed,
                batch_size = batch_size,
                kwargs = kwargs,
            )
        )
        try:
            if prompts is not None and self.during_batch is not None:
                self.during_batch()
            if cancel.is_set():
                raise RuntimeError(DIFFUSION_CANCELLED_MSG)
        finally:
            self.active_cancel = self.active_account = None
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
            "negative_prompt": kwargs.get("negative_prompt"),
            # Derived from every prompt of the call, like textual-inversion names.
            "embeddings": sorted(set(labels)),
            "workflow": "txt2img",
        }


def _accounts(app):
    """Binds the X-Test-Account header as the acting account (owner role, so no managed-account checks)."""

    async def asgi(scope, receive, send):
        account = dict(scope.get("headers") or ()).get(b"x-test-account")
        if scope["type"] != "http" or account is None:
            return await app(scope, receive, send)
        name = account.decode()
        token = bind_account(AccountContext(name, name, ROLE_OWNER))
        try:
            await app(scope, receive, send)
        finally:
            reset_account(token)

    return asgi


@pytest.fixture
def studio(monkeypatch):
    backend = _Backend()
    saved: list[dict] = []

    def _save(image, meta):
        image_id = f"img{len(saved)}"
        record = {
            **meta,
            "id": image_id,
            "url": f"/api/inference/images/gallery/{image_id}/file",
            "image": image,
            "account": current_account_id(),
        }
        saved.append(record)
        return record

    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    monkeypatch.setattr(engine_router, "get_active_diffusion_engine", lambda: backend)
    monkeypatch.setattr(engine_router, "_active_engine_name", "diffusers")
    monkeypatch.setattr(gpu_arbiter, "_owner", None)
    monkeypatch.setattr(gallery_module, "save", _save)
    monkeypatch.setattr(gallery_module, "image_b64", lambda image_id: "QUJD")
    app = FastAPI()
    install_api_error_handlers(app)
    app.include_router(router, prefix = "/v1")
    app.include_router(studio_router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    return _accounts(app), backend, saved


async def _until(predicate, timeout = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition never held")
        await asyncio.sleep(0.01)


def _waiting() -> int:
    return sum(len(batch.members) for batch in batching._OPEN.values())


def _openai(prompt: str, **extra) -> dict:
    return {"prompt": prompt, "size": "256x256", "response_format": "b64_json", **extra}


def _post_while_rendering(
    app,
    backend,
    posts,
    waiting,
    *,
    then = None,
):
    """``posts``: (path, account, body). Holds the slot until ``waiting`` requests queue, then frees it."""

    async def run():
        transport = httpx.ASGITransport(app = app)
        async with httpx.AsyncClient(transport = transport, base_url = "http://studio") as client:
            backend.rendering = True
            sent = []
            # One at a time, so the first post leads its batch.
            for path, account, body in posts:
                sent.append(
                    asyncio.ensure_future(
                        client.post(path, json = body, headers = {"x-test-account": account})
                    )
                )
                count = len(sent)
                await _until(lambda: _waiting() >= min(count, waiting))
            await _until(lambda: _waiting() >= waiting)
            if then is not None:
                await then()
            backend.rendering = False
            return await asyncio.gather(*sent)

    return asyncio.run(run())


def _batched_calls(backend) -> list[dict]:
    return [call for call in backend.calls if call["prompts"] is not None]


def test_two_accounts_with_identical_settings_share_one_call_and_each_gets_only_its_own_images(
    studio,
):
    app, backend, saved = studio
    posts = [
        ("/v1/images/generations", "alice", _openai("a fox", n = 2)),
        ("/v1/images/generations", "bob", _openai("a secret cat")),
    ]
    responses = _post_while_rendering(app, backend, posts, waiting = 2)
    assert [r.status_code for r in responses] == [200, 200]
    assert [len(r.json()["data"]) for r in responses] == [2, 1]
    [call] = backend.calls
    assert sorted(call["prompts"]) == ["a fox", "a fox", "a secret cat"]
    by_account: dict[str, list] = {}
    for record in saved:
        by_account.setdefault(record["account"], []).append((record["prompt"], record["image"]))
    assert {prompt for prompt, _ in by_account["alice"]} == {"a fox"}
    assert all(image.startswith("a fox#") for _, image in by_account["alice"])
    assert by_account["bob"] == [("a secret cat", by_account["bob"][0][1])]
    assert by_account["bob"][0][1].startswith("a secret cat#")


def test_prompt_derived_result_fields_never_cross_accounts():
    from core.inference.image_request_batching import _Member, _split
    loop = asyncio.new_event_loop()
    try:
        members = [
            _Member("a fox", [1], loop.create_future(), "alice"),
            _Member("a secret cat", [2], loop.create_future(), "bob"),
        ]
        result = {"images": ["fox", "cat"], "seeds": [1, 2], "seed": 1, "embeddings": ["cat"]}
        parts = _split(result, members)
        assert [part["images"] for part in parts] == [["fox"], ["cat"]]
        assert all("embeddings" not in part for part in parts)
        same = [
            _Member("a", [1], loop.create_future(), "alice"),
            _Member("b", [2], loop.create_future(), "alice"),
        ]
        assert _split(result, same)[0]["embeddings"] == ["cat"]
    finally:
        loop.close()


@pytest.mark.parametrize(
    "other",
    [
        {"size": "512x512"},
        {"negative_prompt": "blurry"},
        {"steps": 4},
        {"guidance": 2.5},
        {"loras": [{"id": "style-lora", "weight": 0.5}]},
    ],
)
def test_requests_with_different_settings_never_share_a_call(studio, other):
    app, backend, _saved = studio
    posts = [
        ("/v1/images/generations", "alice", _openai("a fox")),
        ("/v1/images/generations", "bob", _openai("a cat", **other)),
    ]
    responses = _post_while_rendering(app, backend, posts, waiting = 2)
    assert all(r.status_code == 200 for r in responses), [r.text for r in responses]
    assert len(backend.calls) == 2
    assert _batched_calls(backend) == []


def test_mixed_settings_group_only_the_matching_accounts(studio):
    app, backend, _saved = studio
    posts = [
        ("/v1/images/generations", "alice", _openai("a fox")),
        ("/v1/images/generations", "bob", _openai("a cat")),
        ("/v1/images/generations", "carol", _openai("an owl", steps = 3)),
    ]
    responses = _post_while_rendering(app, backend, posts, waiting = 3)
    assert all(r.status_code == 200 for r in responses)
    [call] = _batched_calls(backend)
    assert sorted(call["prompts"]) == ["a cat", "a fox"]
    assert [c["prompt"] for c in backend.calls if c["prompts"] is None] == ["an owl"]


def test_a_member_stop_leaves_the_batch_and_the_other_accounts_still_render(studio):
    app, backend, saved = studio
    posts = [
        ("/v1/images/generations", "alice", _openai("a fox")),
        ("/v1/images/generations", "bob", _openai("a cat")),
        ("/v1/images/generations", "carol", _openai("an owl")),
    ]

    async def bob_stops():
        await _until(lambda: len(backend.riders) == 2)
        assert backend.cancel_generate(expected_account = "bob") is True
        await _until(lambda: _waiting() == 2)

    responses = _post_while_rendering(app, backend, posts, waiting = 3, then = bob_stops)
    assert [r.status_code == 200 for r in responses] == [True, False, True]
    [call] = backend.calls
    assert sorted(call["prompts"]) == ["a fox", "an owl"]
    assert sorted(record["account"] for record in saved) == ["alice", "carol"]


def test_the_leaders_stop_ends_its_own_account_and_other_accounts_render_alone(studio):
    app, backend, saved = studio
    posts = [
        ("/v1/images/generations", "alice", _openai("a fox")),
        ("/v1/images/generations", "alice", _openai("a second fox")),
        ("/v1/images/generations", "bob", _openai("a cat")),
    ]
    # Alice's Stop lands on the shared call while it renders.
    backend.during_batch = lambda: backend.cancel_generate(expected_account = "alice")
    responses = _post_while_rendering(app, backend, posts, waiting = 3)
    assert [r.status_code == 200 for r in responses] == [False, False, True]
    assert len(_batched_calls(backend)) == 1
    [alone] = [call for call in backend.calls if call["prompts"] is None]
    assert alone["prompt"] == "a cat" and alone["account"] == "bob"
    assert [record["account"] for record in saved] == ["bob"]


def test_a_stop_from_another_account_does_not_reach_a_render_it_does_not_ride_in():
    """The real engine: Stop from bob signals only bob's riders, never alice's running generation."""
    from core.inference.diffusion import DiffusionBackend

    engine = DiffusionBackend()
    alice_render, bob_ride, carol_ride = threading.Event(), threading.Event(), threading.Event()

    def as_account(name, func):
        token = bind_account(AccountContext(name, name, ROLE_OWNER))
        try:
            return func()
        finally:
            reset_account(token)

    with contextlib.ExitStack() as stack:
        as_account("alice", lambda: stack.enter_context(engine._generation_slot(alice_render)))
        as_account("bob", lambda: stack.enter_context(engine.generation_rider(bob_ride)))
        as_account("carol", lambda: stack.enter_context(engine.generation_rider(carol_ride)))
        assert engine.cancel_generate(expected_account = "bob") is True
        assert bob_ride.is_set() and not alice_render.is_set() and not carol_ride.is_set()
        assert engine.cancel_generate(expected_account = "dave") is False
        assert engine.cancel_generate(expected_account = "alice") is True
        assert alice_render.is_set() and not carol_ride.is_set()
    assert engine._generate_riders == {}


def _png_b64() -> str:
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buffer, format = "PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _create(prompt: str, **extra) -> dict:
    return {"prompt": prompt, "width": 256, "height": 256, "steps": 4, "guidance": 0.0, **extra}


def test_images_generate_plain_txt2img_from_two_accounts_shares_one_call(studio):
    app, backend, saved = studio
    posts = [
        ("/api/inference/images/generate", "alice", _create("a fox", batch_size = 2)),
        ("/api/inference/images/generate", "bob", _create("a secret cat")),
    ]
    responses = _post_while_rendering(app, backend, posts, waiting = 2)
    assert [r.status_code for r in responses] == [200, 200], [r.text for r in responses]
    [call] = backend.calls
    assert sorted(call["prompts"]) == ["a fox", "a fox", "a secret cat"]
    assert call["kwargs"]["init_image"] is None
    alice, bob = (r.json()["images"] for r in responses)
    assert [image["prompt"] for image in alice] == ["a fox", "a fox"]
    assert [image["prompt"] for image in bob] == ["a secret cat"]
    by_account = {}
    for record in saved:
        by_account.setdefault(record["account"], []).append(record["image"])
    assert all(image.startswith("a fox#") for image in by_account["alice"])
    assert [image.split("#")[0] for image in by_account["bob"]] == ["a secret cat"]
    # The batch seeds derive like a lone request's: base, base + 1.
    assert alice[1]["seed"] == alice[0]["seed"] + 1 and alice[1]["batch_seed"] == alice[0]["seed"]


def test_images_generate_with_an_init_image_never_joins_a_batch(studio):
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
            edits = [
                asyncio.ensure_future(
                    client.post(
                        "/api/inference/images/generate",
                        json = _create("a cat", **extra),
                        headers = {"x-test-account": "bob"},
                    )
                )
                for extra in (
                    {"init_image": _png_b64(), "strength": 0.5},
                    {"variation_seed": 7, "variation_strength": 0.3},
                    {"seeds": [1, 2]},
                )
            ]
            await asyncio.sleep(0.3)
            assert _waiting() == 1
            backend.rendering = False
            return await asyncio.gather(first, *edits)

    responses = asyncio.run(run())
    assert all(r.status_code == 200 for r in responses), [
        (r.status_code, r.text[:300]) for r in responses if r.status_code != 200
    ]
    # Four calls, none batched: the plain Create rendered alone too, as nothing could join it.
    assert len(backend.calls) == 4
    assert _batched_calls(backend) == []
    assert sorted(call["prompt"] for call in backend.calls) == ["a cat", "a cat", "a cat", "a fox"]


def test_a_lone_images_generate_request_renders_at_once_without_a_batch(studio):
    app, backend, _saved = studio

    async def run():
        transport = httpx.ASGITransport(app = app)
        async with httpx.AsyncClient(transport = transport, base_url = "http://studio") as client:
            return await client.post(
                "/api/inference/images/generate",
                json = _create("a fox"),
                headers = {"x-test-account": "alice"},
            )

    response = asyncio.run(run())
    assert response.status_code == 200, response.text
    [call] = backend.calls
    assert call["prompts"] is None and call["prompt"] == "a fox"


@pytest.mark.parametrize("path", ["/v1/images/generations", "/api/inference/images/generate"])
def test_seeded_requests_join_a_batch_only_when_opted_in(studio, monkeypatch, path):
    app, backend, saved = studio
    body = _openai if path.startswith("/v1") else _create

    async def run():
        transport = httpx.ASGITransport(app = app)
        async with httpx.AsyncClient(transport = transport, base_url = "http://studio") as client:
            backend.rendering = True
            sent = []
            for account, prompt, seed in (("alice", "a fox", 11), ("bob", "a cat", 22)):
                sent.append(
                    asyncio.ensure_future(
                        client.post(
                            path,
                            json = body(
                                prompt,
                                seed = seed,
                                **({"n": 2} if path.startswith("/v1") else {"batch_size": 2}),
                            ),
                            headers = {"x-test-account": account},
                        )
                    )
                )
                await asyncio.sleep(0.2)
            backend.rendering = False
            return await asyncio.gather(*sent)

    monkeypatch.delenv("UNSLOTH_STUDIO_IMAGE_BATCH_SEEDED", raising = False)
    assert all(r.status_code == 200 for r in asyncio.run(run()))
    assert _batched_calls(backend) == []
    assert sorted(call["seed"] for call in backend.calls) == [11, 22]

    backend.calls.clear()
    saved.clear()
    monkeypatch.setenv("UNSLOTH_STUDIO_IMAGE_BATCH_SEEDED", "1")
    assert all(r.status_code == 200 for r in asyncio.run(run()))
    [call] = backend.calls
    # Each member keeps the seeds a lone request would use: base, base + 1.
    assert call["seeds"] == [11, 12, 22, 23]
    assert call["prompts"] == ["a fox", "a fox", "a cat", "a cat"]
    images = sorted((record["account"], record["seed"], record["image"]) for record in saved)
    assert images == [
        ("alice", 11, "a fox#11"),
        ("alice", 12, "a fox#12"),
        ("bob", 22, "a cat#22"),
        ("bob", 23, "a cat#23"),
    ]


def _queued(engine) -> int:
    with engine._generation_cancel_lock:
        return len(engine._queued_generate_cancels)


def _wait_for(predicate, timeout = 5.0) -> None:
    import time
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition never held")
        time.sleep(0.005)


def test_the_generation_slot_serves_waiters_in_arrival_order():
    from core.inference.diffusion import DiffusionBackend

    engine = DiffusionBackend()
    served: list[int] = []
    threads = []
    with engine._generation_slot(threading.Event()):
        for index in range(8):

            def run(index = index):
                with engine._generation_slot(threading.Event()):
                    served.append(index)

            thread = threading.Thread(target = run, daemon = True)
            thread.start()
            threads.append(thread)
            _wait_for(lambda: _queued(engine) == index + 1)
            # Each waiter has polled the lock a few times before the next one arrives.
            threading.Event().wait(0.25)
    for thread in threads:
        thread.join(timeout = 10)
    assert served == list(range(8))


def test_a_cancelled_waiter_leaves_the_queue_and_the_next_one_is_served():
    from core.inference.diffusion import DiffusionBackend
    from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG

    engine = DiffusionBackend()
    first_cancel = threading.Event()
    outcome: dict[str, object] = {}

    def first():
        try:
            with engine._generation_slot(first_cancel):
                outcome["first"] = "served"
        except RuntimeError as exc:
            outcome["first"] = str(exc)

    def second():
        with engine._generation_slot(threading.Event()):
            outcome["second"] = "served"

    with engine._generation_slot(threading.Event()):
        a = threading.Thread(target = first, daemon = True)
        a.start()
        _wait_for(lambda: _queued(engine) == 1)
        b = threading.Thread(target = second, daemon = True)
        b.start()
        _wait_for(lambda: _queued(engine) == 2)
        first_cancel.set()
        a.join(timeout = 2)
        assert outcome == {"first": DIFFUSION_CANCELLED_MSG}
        assert "second" not in outcome
    b.join(timeout = 5)
    assert outcome["second"] == "served"
    assert not engine._generate_lock.locked()
