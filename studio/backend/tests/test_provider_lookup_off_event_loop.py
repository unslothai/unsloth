# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resolving a saved provider must not read providers from the event loop thread.

Every chat routed to a saved external provider looks the row up, so on a stalled store
that read parks the loop and the server stops answering anything, /api/liveness included.

Asserts which thread the read ran on rather than timing it.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import routes.inference as inference_routes
import routes.provider_credentials as provider_credentials
import routes.providers as provider_routes


def _request():
    async def is_disconnected():
        return False

    return SimpleNamespace(
        headers = {},
        state = SimpleNamespace(skip_api_monitor = True),
        is_disconnected = is_disconnected,
    )


def _payload(external_model: str = "gpt-5.4"):
    from models.inference import ChatCompletionRequest
    return ChatCompletionRequest(
        messages = [{"role": "user", "content": "what is 2+2?"}],
        provider_id = "saved-1",
        external_model = external_model,
        stream = True,
    )


def test_the_saved_provider_row_is_read_off_the_event_loop_thread(monkeypatch):
    threads: list[int] = []

    def _get_provider(_provider_id):
        threads.append(threading.get_ident())
        return None

    monkeypatch.setattr(inference_routes.providers_db, "get_provider", _get_provider)

    # run_until_complete drives the loop on this thread.
    loop_thread = threading.get_ident()
    with pytest.raises(HTTPException) as excinfo:
        asyncio.new_event_loop().run_until_complete(
            inference_routes._proxy_to_external_provider(
                _payload(), _request(), current_subject = "t"
            )
        )

    assert excinfo.value.status_code == 404
    assert threads, "the proxy never looked the saved provider up"
    assert loop_thread not in threads, "the provider row was read on the event loop thread"


def test_the_saved_provider_target_and_key_are_one_snapshot(monkeypatch):
    state = {
        "row": {
            "id": "saved-1",
            "provider_type": "openai",
            "display_name": "Saved",
            "base_url": "https://old.example/v1",
            "is_enabled": True,
        },
        "key": "old-secret",
    }
    row_read = threading.Event()
    update_done = threading.Event()
    observed = []

    def _get_provider(_provider_id):
        row = dict(state["row"])
        if not row_read.is_set():
            row_read.set()
            assert update_done.wait(2), "the concurrent update never completed"
        return row

    def _resolve_key(*_args, **_kwargs):
        observed.append(state["key"])
        return state["key"]

    monkeypatch.setattr(inference_routes.providers_db, "get_provider", _get_provider)
    monkeypatch.setattr(inference_routes, "resolve_provider_api_key_or_400", _resolve_key)

    async def _update():
        assert await asyncio.to_thread(row_read.wait, 2), "the saved row was never read"
        async with provider_credentials.provider_config_guard("saved-1"):
            state["row"]["base_url"] = "https://new.example/v1"
            state["key"] = "new-secret"
        update_done.set()

    async def _drive():
        proxy = asyncio.create_task(
            inference_routes._proxy_to_external_provider(
                _payload(external_model = "default"), _request(), current_subject = "t"
            )
        )
        update = asyncio.create_task(_update())
        with pytest.raises(HTTPException) as excinfo:
            await proxy
        assert excinfo.value.status_code == 409
        await update

    asyncio.run(_drive())
    assert observed == []
    assert state["row"]["base_url"] == "https://new.example/v1"
    assert state["key"] == "new-secret"


@pytest.mark.parametrize(
    "handler",
    [
        # update_provider_config splits its serialized mutation into
        # _apply_provider_update so the catalog proof can run outside the lock.
        provider_routes._apply_provider_update,
        provider_routes.migrate_provider_api_key,
        provider_routes.delete_provider_config,
    ],
)
def test_provider_mutations_share_the_saved_snapshot_guard(handler):
    assert getattr(handler, "_provider_config_serialized", False)


def _container_body():
    from models.inference import OpenAIContainerRequest
    return OpenAIContainerRequest(provider_id = "saved-1")


def test_the_container_resolver_reads_on_the_event_loop_thread(monkeypatch):
    """The container routes resolve the row and the credential as one snapshot.

    _resolve_openai_cloud_client reads the provider row for the base URL and then reads the
    saved key. Run in a worker, an edit landing between the two pairs the old base URL with
    the new key, so the routes call it on the loop where nothing interleaves.
    """
    threads: list[int] = []

    def _get_provider(_provider_id):
        threads.append(threading.get_ident())
        return None

    monkeypatch.setattr(inference_routes.providers_db, "get_provider", _get_provider)

    loop_thread = threading.get_ident()
    with pytest.raises(HTTPException) as excinfo:
        asyncio.new_event_loop().run_until_complete(
            inference_routes.list_openai_containers(
                _container_body(), _request(), current_subject = "t"
            )
        )

    assert excinfo.value.status_code == 404
    assert threads, "the container route never looked the saved provider up"
    assert threads[0] == loop_thread, "the container resolver ran off the event loop thread"
