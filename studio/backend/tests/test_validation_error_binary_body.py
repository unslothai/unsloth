# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A binary body on a JSON route must yield 422, not 500 plus a megabyte of log.

POSTing a multipart WAV upload to /api/inference/audio/transcribe (which takes a
JSON body with base64 audio) failed request validation. The handler then ran
jsonable_encoder over exc.errors(), whose "input" was the whole raw body, and
FastAPI encodes bytes with o.decode(): UnicodeDecodeError. The 422 became a 500
whose traceback embedded the escaped payload, so one 531 KB upload wrote a single
2.2 MB line into the server log.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.encoders import jsonable_encoder  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from utils.api_errors import (  # noqa: E402
    install_api_error_handlers,
    safe_validation_errors,
)

# Non-UTF-8: 0x80 is an invalid start byte, exactly like a RIFF/WAV payload.
_BINARY = b"RIFF\x00\x00\x80\xff\xfe\xfd" * 64


class _Body(BaseModel):
    audio: str


def _client() -> TestClient:
    app = FastAPI()
    install_api_error_handlers(app)

    @app.post("/api/thing")
    async def _thing(payload: _Body):  # pragma: no cover - never reached here
        return {"ok": True}

    return TestClient(app)


def _post_multipart_binary(client: TestClient):
    # A multipart upload to a JSON-body route: the shape that produced the 500.
    # FastAPI reads the body, fails to coerce it to the model, and puts the raw
    # bytes in the error's "input".
    return client.post("/api/thing", files = {"file": ("audio.wav", _BINARY, "audio/wav")})


def test_binary_body_returns_422_not_500():
    resp = _post_multipart_binary(_client())
    assert resp.status_code == 422, resp.text


def test_response_does_not_echo_the_body():
    resp = _post_multipart_binary(_client())
    # The response stays small: the payload is summarized, never echoed back.
    assert len(resp.text) < 2000, len(resp.text)
    assert "RIFF" not in resp.text


def test_valid_body_still_passes():
    resp = _client().post("/api/thing", json = {"audio": "abc"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_a_normal_validation_error_still_names_the_field():
    resp = _client().post("/api/thing", json = {"wrong": 1})
    assert resp.status_code == 422
    body = resp.json()
    detail = body["detail"]
    assert any("audio" in str(item.get("loc", "")) for item in detail), detail


@pytest.mark.parametrize(
    "value, expected_fragment",
    [
        (b"\x80\xff", "2 bytes of binary data"),
        (bytearray(b"\x00" * 5), "5 bytes of binary data"),
        ("x" * 5000, "truncated, 5000 chars"),
    ],
)
def test_summarizer_makes_inputs_encodable(value, expected_fragment):
    errors = [{"type": "x", "loc": ("body",), "msg": "bad", "input": value}]
    safe = safe_validation_errors(errors)
    assert expected_fragment in str(safe[0]["input"])
    jsonable_encoder(safe)  # must not raise


def test_short_inputs_are_left_alone():
    errors = [{"type": "x", "loc": ("body",), "msg": "bad", "input": {"a": 1}}]
    assert safe_validation_errors(errors)[0]["input"] == {"a": 1}


def test_nested_binary_inside_a_dict_is_summarized():
    errors = [{"type": "x", "loc": ("body",), "msg": "bad", "input": {"f": b"\x80\x81"}}]
    safe = safe_validation_errors(errors)
    jsonable_encoder(safe)
    assert "2 bytes of binary data" in str(safe[0]["input"]["f"])
