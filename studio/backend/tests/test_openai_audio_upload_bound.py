# SPDX-License-Identifier: AGPL-3.0-only
"""The multipart transcription route must not materialize an unbounded upload."""

import asyncio

import pytest
from fastapi import HTTPException

import routes.inference as inference_route


def test_openai_transcription_reads_only_one_byte_past_limit(monkeypatch):
    sizes = []

    class _Upload:
        async def read(self, size = -1):
            sizes.append(size)
            return b"x" * size

    monkeypatch.setattr(inference_route, "_MAX_AUDIO_RAW_BYTES", 4)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            inference_route.openai_audio_transcriptions(
                request = None,
                file = _Upload(),
                model = None,
                language = None,
                response_format = "json",
                current_subject = "tester",
            )
        )

    assert sizes == [5]
    assert exc.value.status_code == 413
