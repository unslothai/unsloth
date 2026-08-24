# SPDX-License-Identifier: AGPL-3.0-only
"""The multipart transcription route must not materialize an unbounded upload."""

import asyncio

import pytest
from fastapi import HTTPException
from starlette.requests import Request

import routes.inference as inference_route


def _request():
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "server": ("testserver", 80),
            "path": "/v1/audio/transcriptions",
            "raw_path": b"/v1/audio/transcriptions",
            "query_string": b"",
            "root_path": "",
            "headers": [],
        }
    )


def test_openai_transcription_reads_only_one_byte_past_limit(monkeypatch):
    sizes = []

    class _Upload:
        filename = "clip.wav"

        async def read(self, size = -1):
            sizes.append(size)
            return b"x" * size

    monkeypatch.setattr(inference_route, "_MAX_AUDIO_RAW_BYTES", 4)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            inference_route.openai_audio_transcriptions(
                request = _request(),
                file = _Upload(),
                model = None,
                language = None,
                response_format = "json",
                current_subject = "tester",
            )
        )

    assert sizes == [5]
    assert exc.value.status_code == 413
