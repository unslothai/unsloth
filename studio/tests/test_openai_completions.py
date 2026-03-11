# SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
# Copyright © 2025 Unsloth AI

"""
Tests for the OpenAI-compatible /chat/completions endpoint.

Validates:
  - Streaming: SSE chunk format matches OpenAI spec
  - Non-streaming: single JSON ChatCompletion response
  - System prompt extraction from messages array
  - Request validation (no messages, missing model, etc.)
  - Response headers for proxy compatibility

All tests mock the inference backend and bypass auth.
"""

import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ── Path setup ────────────────────────────────────────────────────
_backend_root = Path(__file__).resolve().parent.parent / "backend"
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

from fastapi.testclient import TestClient
from main import app


# ── Fixtures ──────────────────────────────────────────────────────


def _make_mock_backend(
    *, tokens: list[str] | None = None, active_model: str = "test-model"
):
    """Build a mock InferenceBackend that yields preset tokens."""
    backend = MagicMock()
    backend.active_model_name = active_model
    backend.models = {active_model: {"is_vision": False}}

    def fake_generate(**kwargs):
        for t in tokens or ["Hello", "Hello world", "Hello world!"]:
            yield t

    backend.generate_chat_response = MagicMock(side_effect = fake_generate)
    backend.reset_generation_state = MagicMock()
    return backend


def _parse_sse_data(raw: str) -> list[dict | str]:
    """Extract `data:` payloads from raw SSE text. Returns dicts or raw strings."""
    results = []
    for line in raw.split("\n"):
        if line.startswith("data: "):
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                results.append("[DONE]")
            else:
                try:
                    results.append(json.loads(payload))
                except json.JSONDecodeError:
                    results.append(payload)
    return results


@pytest.fixture()
def client():
    yield TestClient(app)


# =====================================================================
# Streaming tests
# =====================================================================


class TestStreamingChunkFormat:
    """Each SSE chunk must match the OpenAI chat.completion.chunk schema."""

    def test_chunks_have_required_fields(self, client: TestClient):
        mock_backend = _make_mock_backend(tokens = ["Hi"])
        with patch("routes.inference.get_inference_backend", return_value = mock_backend):
            resp = client.post(
                "/api/inference/chat/completions",
                json = {
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        chunks = _parse_sse_data(resp.text)

        # Filter to actual chunk dicts (not [DONE])
        json_chunks = [c for c in chunks if isinstance(c, dict) and "choices" in c]
        assert len(json_chunks) >= 2  # role chunk + content chunk(s) + final

        for chunk in json_chunks:
            assert "id" in chunk
            assert chunk["object"] == "chat.completion.chunk"
            assert "created" in chunk
            assert "model" in chunk
            assert len(chunk["choices"]) == 1
            assert "delta" in chunk["choices"][0]

    def test_first_chunk_has_role(self, client: TestClient):
        mock_backend = _make_mock_backend(tokens = ["Hi"])
        with patch("routes.inference.get_inference_backend", return_value = mock_backend):
            resp = client.post(
                "/api/inference/chat/completions",
                json = {"messages": [{"role": "user", "content": "Hello"}]},
            )

        chunks = [
            c
            for c in _parse_sse_data(resp.text)
            if isinstance(c, dict) and "choices" in c
        ]
        first = chunks[0]
        assert first["choices"][0]["delta"].get("role") == "assistant"

    def test_last_chunk_has_stop_finish_reason(self, client: TestClient):
        mock_backend = _make_mock_backend(tokens = ["Done"])
        with patch("routes.inference.get_inference_backend", return_value = mock_backend):
            resp = client.post(
                "/api/inference/chat/completions",
                json = {"messages": [{"role": "user", "content": "Hello"}]},
            )

        chunks = [
            c
            for c in _parse_sse_data(resp.text)
            if isinstance(c, dict) and "choices" in c
        ]
        last = chunks[-1]
        assert last["choices"][0]["finish_reason"] == "stop"
        # Delta should be empty on the final chunk
        assert last["choices"][0]["delta"].get("content") is None

    def test_stream_ends_with_done(self, client: TestClient):
        mock_backend = _make_mock_backend(tokens = ["x"])
        with patch("routes.inference.get_inference_backend", return_value = mock_backend):
            resp = client.post(
                "/api/inference/chat/completions",
                json = {"messages": [{"role": "user", "content": "Hello"}]},
            )

        all_data = _parse_sse_data(resp.text)
        assert all_data[-1] == "[DONE]"

    def test_consistent_id_across_chunks(self, client: TestClient):
        mock_backend = _make_mock_backend(tokens = ["a", "b", "c"])
        with patch("routes.inference.get_inference_backend", return_value = mock_backend):
            resp = client.post(
                "/api/inference/chat/completions",
                json = {"messages": [{"role": "user", "content": "Hello"}]},
            )

        chunks = [
            c
            for c in _parse_sse_data(resp.text)
            if isinstance(c, dict) and "choices" in c
        ]
        ids = set(c["id"] for c in chunks)
        assert len(ids) == 1, "All chunks should share the same completion ID"


class TestStreamingHeaders:
    """Verify response headers for SSE proxy compatibility."""

    def test_headers(self, client: TestClient):
        mock_backend = _make_mock_backend(tokens = ["x"])
        with patch("routes.inference.get_inference_backend", return_value = mock_backend):
            resp = client.post(
                "/api/inference/chat/completions",
                json = {"messages": [{"role": "user", "content": "Hello"}]},
            )

        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.headers.get("cache-control") == "no-cache"
        assert resp.headers.get("x-accel-buffering") == "no"


# =====================================================================
# Non-streaming tests
# =====================================================================


class TestNonStreaming:
    """When stream=false, return a single ChatCompletion JSON object."""

    def test_returns_json_object(self, client: TestClient):
        mock_backend = _make_mock_backend(tokens = ["Full response text"])
        with patch("routes.inference.get_inference_backend", return_value = mock_backend):
            resp = client.post(
                "/api/inference/chat/completions",
                json = {
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": False,
                },
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["choices"][0]["message"]["content"] == "Full response text"
        assert body["choices"][0]["finish_reason"] == "stop"

    def test_non_streaming_has_model(self, client: TestClient):
        mock_backend = _make_mock_backend(tokens = ["x"], active_model = "my-model")
        with patch("routes.inference.get_inference_backend", return_value = mock_backend):
            resp = client.post(
                "/api/inference/chat/completions",
                json = {
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": False,
                },
            )

        body = resp.json()
        assert body["model"] == "my-model"


# =====================================================================
# System prompt extraction
# =====================================================================


class TestSystemPromptExtraction:
    """System messages should be extracted and passed as system_prompt."""

    def test_system_message_extracted(self, client: TestClient):
        mock_backend = _make_mock_backend(tokens = ["ok"])
        with patch("routes.inference.get_inference_backend", return_value = mock_backend):
            client.post(
                "/api/inference/chat/completions",
                json = {
                    "messages": [
                        {"role": "system", "content": "You are a pirate."},
                        {"role": "user", "content": "Hello"},
                    ],
                    "stream": False,
                },
            )

        # Check that generate_chat_response was called with the correct system_prompt
        call_kwargs = mock_backend.generate_chat_response.call_args[1]
        assert call_kwargs["system_prompt"] == "You are a pirate."
        # System message should NOT be in the chat_messages list
        assert all(m["role"] != "system" for m in call_kwargs["messages"])

    def test_default_system_prompt_when_none(self, client: TestClient):
        mock_backend = _make_mock_backend(tokens = ["ok"])
        with patch("routes.inference.get_inference_backend", return_value = mock_backend):
            client.post(
                "/api/inference/chat/completions",
                json = {
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": False,
                },
            )

        call_kwargs = mock_backend.generate_chat_response.call_args[1]
        assert call_kwargs["system_prompt"] == "You are a helpful AI assistant."


# =====================================================================
# Error handling
# =====================================================================


class TestErrorHandling:
    """Validate error responses for bad requests."""

    def test_no_model_loaded(self, client: TestClient):
        mock_backend = _make_mock_backend()
        mock_backend.active_model_name = None
        with patch("routes.inference.get_inference_backend", return_value = mock_backend):
            resp = client.post(
                "/api/inference/chat/completions",
                json = {"messages": [{"role": "user", "content": "Hi"}]},
            )

        assert resp.status_code == 400
        assert "No model loaded" in resp.json()["detail"]

    def test_only_system_messages_rejected(self, client: TestClient):
        mock_backend = _make_mock_backend()
        with patch("routes.inference.get_inference_backend", return_value = mock_backend):
            resp = client.post(
                "/api/inference/chat/completions",
                json = {
                    "messages": [{"role": "system", "content": "You are a bot."}],
                },
            )

        assert resp.status_code == 400
        assert "non-system message" in resp.json()["detail"]
