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


def _request():
    async def is_disconnected():
        return False

    return SimpleNamespace(
        headers = {},
        state = SimpleNamespace(skip_api_monitor = True),
        is_disconnected = is_disconnected,
    )


def _payload():
    from models.inference import ChatCompletionRequest
    return ChatCompletionRequest(
        messages = [{"role": "user", "content": "what is 2+2?"}],
        provider_id = "saved-1",
        external_model = "gpt-5.4",
        stream = True,
    )


def test_the_saved_provider_row_is_read_off_the_event_loop_thread(monkeypatch):
    threads: list[int] = []

    def _get_provider(_provider_id):
        threads.append(threading.get_ident())
        return None  # a missing row 404s right after the read, which is all this needs

    monkeypatch.setattr(inference_routes.providers_db, "get_provider", _get_provider)

    # run_until_complete drives the loop on this thread, so this ident is the loop's.
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
