# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""_model_json_response produces the same body as JSONResponse(model.model_dump())."""

import json
from typing import Optional

from fastapi.responses import JSONResponse
from pydantic import BaseModel

import routes.inference as inference_route


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
    return JSONResponse(content=model.model_dump()).body


def test_body_matches_old_jsonresponse():
    model = _Resp()
    resp = inference_route._model_json_response(model)
    # Same decoded JSON (key order is irrelevant once parsed), nulls preserved.
    assert json.loads(resp.body) == json.loads(_old_body(model))
    assert json.loads(resp.body)["system_fingerprint"] is None  # null kept, not dropped


def test_media_type_and_status():
    resp = inference_route._model_json_response(_Resp(), status_code=200)
    assert resp.media_type == "application/json"
    assert resp.status_code == 200
    err = inference_route._model_json_response(_Resp(), status_code=503)
    assert err.status_code == 503
