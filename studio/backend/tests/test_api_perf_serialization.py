# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""_model_json_response produces the same body as JSONResponse(model.model_dump())."""

import asyncio
import json
from typing import Optional

from fastapi.responses import JSONResponse
from pydantic import BaseModel

import routes.inference as inference_route
from core.inference import llama_http


class _Usage(BaseModel):
    prompt_tokens: int = 3
    completion_tokens: int = 5
    details: Optional[dict] = None


class _Choice(BaseModel):
    index: int = 0
    text: str = "hello"
    logprobs: Optional[dict] = None


class _Resp(BaseModel):
    id: str = "chatcmpl-abc"
    object: str = "chat.completion"
    created: int = 1700000000
    model: str = "unsloth/SmolLM2-135M-Instruct-GGUF"
    choices: list[_Choice] = [_Choice()]
    usage: _Usage = _Usage()
    system_fingerprint: Optional[str] = None


def _old_body(model) -> bytes:
    # What the previous code emitted: dict -> Starlette json.dumps.
    return JSONResponse(content = model.model_dump()).body


def test_body_matches_old_jsonresponse():
    model = _Resp()
    resp = inference_route._model_json_response(model)
    # Same decoded JSON (key order is irrelevant once parsed), nulls preserved.
    assert json.loads(resp.body) == json.loads(_old_body(model))
    assert json.loads(resp.body)["system_fingerprint"] is None  # null kept, not dropped


def test_media_type_and_status():
    resp = inference_route._model_json_response(_Resp(), status_code = 200)
    assert resp.media_type == "application/json"
    assert resp.status_code == 200
    err = inference_route._model_json_response(_Resp(), status_code = 503)
    assert err.status_code == 503


def test_pooled_client_disables_proxy_env():
    async def _scenario():
        client = llama_http.nonstreaming_client()
        assert client.trust_env is False
        await llama_http.aclose()

    asyncio.run(_scenario())


def test_pooled_client_reused_within_loop_and_recreated_after_close():
    async def _scenario():
        a = llama_http.nonstreaming_client()
        b = llama_http.nonstreaming_client()
        assert a is b  # reused within one loop
        await llama_http.aclose()
        assert a.is_closed
        c = llama_http.nonstreaming_client()  # must not return the closed client
        assert c is not a and not c.is_closed
        await llama_http.aclose()

    asyncio.run(_scenario())


def test_pooled_client_is_per_event_loop():
    clients = []
    # Each asyncio.run uses a fresh loop; the pooled client must not leak across.
    for _ in range(2):

        async def _grab():
            clients.append(llama_http.nonstreaming_client())
            await llama_http.aclose()

        asyncio.run(_grab())
    assert clients[0] is not clients[1]
