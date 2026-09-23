# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json

from core.inference import transcript_stream


def test_progress_and_saved_result(monkeypatch):
    async def scenario():
        async def transcribe(progress):
            progress({"text": "hello"})
            await asyncio.sleep(0.01)
            return {"text": "hello world", "model": "tiny", "duration": 2}

        monkeypatch.setattr(
            transcript_stream.transcript_gallery,
            "save",
            lambda result, title: {"id": "saved", **result},
        )
        events = [
            json.loads(line)
            async for line in transcript_stream.stream_transcript(transcribe, "clip")
        ]
        assert any(event.get("text") == "hello" for event in events)
        assert events[-1]["record"]["id"] == "saved"
        assert events[-1]["text"] == "hello world"

    asyncio.run(scenario())


def test_save_failure_returns_complete_text(monkeypatch):
    async def transcribe(progress):
        return {"text": "keep this", "model": "tiny"}

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(transcript_stream.transcript_gallery, "save", fail)

    async def scenario():
        events = [
            json.loads(line)
            async for line in transcript_stream.stream_transcript(transcribe, "clip")
        ]
        assert events[-1]["text"] == "keep this"
        assert events[-1]["record"] is None

    asyncio.run(scenario())


def test_closing_stream_cancels_inference_without_saving(monkeypatch):
    async def scenario():
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def transcribe(progress):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        monkeypatch.setattr(
            transcript_stream.transcript_gallery,
            "save",
            lambda *args: (_ for _ in ()).throw(AssertionError("saved cancelled request")),
        )
        stream = transcript_stream.stream_transcript(transcribe, "clip")
        await anext(stream)
        await started.wait()
        await stream.aclose()
        assert cancelled.is_set()

    asyncio.run(scenario())


def test_transformers_progress_uses_existing_audio_windows(monkeypatch):
    import numpy as np
    from core.inference.stt_sidecar import WhisperSttSidecar

    class Worker:
        generation_config = None
        calls = 0

        def transcribe_window(self, pcm, kwargs, cancel):
            self.calls += 1
            return f"part {self.calls}"

    worker = Worker()
    sidecar = WhisperSttSidecar()
    monkeypatch.setattr(sidecar, "load", lambda model: worker)
    updates = []
    result = sidecar._transcribe_decoded(
        "tiny", np.zeros(35 * 16000), {}, on_progress=updates.append
    )
    assert result == "part 1 part 2"
    assert [event["processed_seconds"] for event in updates] == [30, 35]
    assert updates[-1]["text"] == result


def test_mtmd_stream_cleans_metadata_and_requires_completion(monkeypatch):
    import io
    import pytest
    from core.inference import stt_mtmd_sidecar as mtmd

    marker = mtmd.MTMD_STT_MODELS["qwen3-asr-0.6b"].transcript_marker
    wire = [
        {"choices": [{"delta": {"content": f"language English{marker}hello"}}]},
        {"choices": [{"delta": {"content": " world"}}]},
    ]
    payloads = []

    class Response(io.BytesIO):
        status = 200

    class Connection:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, *args, **kwargs):
            payloads.append(json.loads(kwargs["body"]))

        def getresponse(self):
            return Response(
                b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in wire)
            )

        def close(self):
            pass

    monkeypatch.setattr(mtmd.http.client, "HTTPConnection", Connection)
    updates = []
    sidecar = mtmd.MtmdSttSidecar()
    with pytest.raises(RuntimeError, match="before finishing"):
        sidecar._post_transcribe(1234, "qwen3-asr-0.6b", b"wav", on_progress=updates.append)
    assert updates[-1]["text"] == "hello world"
    wire.append({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    assert (
        sidecar._post_transcribe(1234, "qwen3-asr-0.6b", b"wav", on_progress=updates.append)
        == "hello world"
    )
    assert payloads[-1]["stream"] is True


def test_diffusion_status_retains_the_exact_checkpoint_filename():
    from models.inference import DiffusionStatusResponse

    status = DiffusionStatusResponse(
        loaded=True, model_kind="gguf", gguf_filename="model-Q8_0.gguf"
    ).model_dump()
    assert status["gguf_filename"] == "model-Q8_0.gguf"
