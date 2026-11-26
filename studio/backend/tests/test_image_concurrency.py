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
        **kwargs,
    ):
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
        record = {**meta, "id": f"img{len(saved)}", "image": image, "account": current_account_id()}
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
