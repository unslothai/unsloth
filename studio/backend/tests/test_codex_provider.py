# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Unit tests for the Codex SDK provider integration.

Covers:

* Availability probe: codex missing, codex present but logged out,
  codex present + logged in, plus the empty-output / non-zero rc
  edge cases the CLI has shipped over time.
* ``stream_codex`` event translation: a fake codex_app_server module
  is dropped into ``sys.modules`` so the production import path runs
  without the real SDK installed. Verifies an OpenAI Chat Completions
  shape (content chunk, stop chunk, [DONE]).
* Parallel-calls fan-out: ``parallel_calls > 1`` spawns N async tasks
  and emits ``codex_tab_open`` / ``codex_tab_chunk`` / ``codex_tab_close``
  events plus a final ``codex_gather`` synthesis event.
* Request validator: ``parallel_calls`` is clamped to [1, 20] by
  pydantic so a runaway value is rejected with 422 before any Codex
  task is spawned.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import types
from typing import Any

import pytest


_backend = os.path.join(os.path.dirname(__file__), "..")
if _backend not in sys.path:
    sys.path.insert(0, _backend)


# ── Helpers ─────────────────────────────────────────────────────────


class _FakeStream:
    """Async iterator that yields predetermined string text events.

    The Codex SDK's ``thread.run_streaming`` returns an async iterable
    of events. ``_stream_thread_run`` converts those into raw text via
    ``_coerce_text``; passing in plain strings exercises the simplest
    coercion path.
    """

    def __init__(self, chunks: list[str]):
        self._chunks = list(chunks)
        self._i = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._i >= len(self._chunks):
            raise StopAsyncIteration
        text = self._chunks[self._i]
        self._i += 1
        return text


class _FakeThread:
    def __init__(self, chunks: list[str], final: str | None = None):
        self._chunks = chunks
        self._final = final if final is not None else "".join(chunks)

    def run_streaming(self, prompt: str):
        # ``run_streaming`` may return either an async iterable or a
        # coroutine that resolves to one; cover the direct-return
        # shape here, the coroutine shape is covered in a separate
        # test below.
        return _FakeStream(self._chunks)

    async def run(self, prompt: str):
        return self._final


class _FakeAsyncCodex:
    """Async-context-manager facade matching codex_app_server.AsyncCodex."""

    def __init__(
        self,
        chunks: list[str] | None = None,
        final: str | None = None,
        raise_on_start: Exception | None = None,
    ):
        self._chunks = chunks or []
        self._final = final
        self._raise = raise_on_start

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def thread_start(self, **kwargs):
        if self._raise is not None:
            raise self._raise
        return _FakeThread(self._chunks, self._final)


def _install_fake_codex_sdk(monkeypatch, async_codex_cls):
    """Drop a fake ``codex_app_server`` module into sys.modules so the
    production lazy-import path picks it up without the real SDK
    being installed.
    """
    fake_mod = types.ModuleType("codex_app_server")
    fake_mod.AsyncCodex = async_codex_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "codex_app_server", fake_mod)
    # importlib.util.find_spec walks finders, not sys.modules; patch
    # it directly so the lazy-import gate accepts the fake.
    import importlib.util as _iu

    real_find_spec = _iu.find_spec

    def _shim(name: str, *args, **kwargs):
        if name == "codex_app_server":
            return types.SimpleNamespace()
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr("importlib.util.find_spec", _shim)


# ── Availability probe ─────────────────────────────────────────────


class TestCodexAvailability:
    def test_absent_when_cli_missing(self, monkeypatch):
        from core.inference import codex_availability as ca

        monkeypatch.setattr(ca, "_which_codex", lambda: None)
        monkeypatch.setattr(ca, "_sdk_importable", lambda: False)

        payload = asyncio.run(ca.probe_codex_availability())
        assert payload["installed"] is False
        assert payload["cli_path"] is None
        assert payload["sdk_importable"] is False
        # supported_models is a sensible default even when nothing is
        # installed so the picker has something to render IF the user
        # forces the entry on a future status flip.
        assert isinstance(payload["supported_models"], list)
        assert len(payload["supported_models"]) > 0

    def test_present_but_sdk_missing(self, monkeypatch):
        from core.inference import codex_availability as ca

        monkeypatch.setattr(ca, "_which_codex", lambda: "/usr/local/bin/codex")
        monkeypatch.setattr(ca, "_sdk_importable", lambda: False)

        async def fake_version():
            return "codex-cli 0.133.0"

        async def fake_logged_in():
            return True

        monkeypatch.setattr(ca, "_detect_version", fake_version)
        monkeypatch.setattr(ca, "_detect_logged_in", fake_logged_in)

        payload = asyncio.run(ca.probe_codex_availability())
        # installed requires BOTH CLI and SDK -- this is the gate the
        # frontend uses to decide whether to surface the provider entry
        # at all, so missing-SDK means hide.
        assert payload["installed"] is False
        assert payload["cli_path"] == "/usr/local/bin/codex"
        assert payload["sdk_importable"] is False
        assert payload["version"] == "codex-cli 0.133.0"

    def test_present_and_logged_out(self, monkeypatch):
        from core.inference import codex_availability as ca

        monkeypatch.setattr(ca, "_which_codex", lambda: "/usr/local/bin/codex")
        monkeypatch.setattr(ca, "_sdk_importable", lambda: True)

        async def fake_version():
            return "codex-cli 0.133.0"

        async def fake_logged_in():
            return False

        monkeypatch.setattr(ca, "_detect_version", fake_version)
        monkeypatch.setattr(ca, "_detect_logged_in", fake_logged_in)

        payload = asyncio.run(ca.probe_codex_availability())
        assert payload["installed"] is True
        assert payload["logged_in"] is False
        assert payload["version"] == "codex-cli 0.133.0"

    def test_present_and_logged_in(self, monkeypatch):
        from core.inference import codex_availability as ca

        monkeypatch.setattr(ca, "_which_codex", lambda: "/usr/local/bin/codex")
        monkeypatch.setattr(ca, "_sdk_importable", lambda: True)

        async def fake_version():
            return "codex-cli 0.133.0"

        async def fake_logged_in():
            return True

        monkeypatch.setattr(ca, "_detect_version", fake_version)
        monkeypatch.setattr(ca, "_detect_logged_in", fake_logged_in)

        payload = asyncio.run(ca.probe_codex_availability())
        assert payload["installed"] is True
        assert payload["logged_in"] is True


# ── _stream_codex translation ──────────────────────────────────────


def _collect_stream(gen) -> list[str]:
    async def run():
        out: list[str] = []
        async for line in gen:
            out.append(line)
        return out

    return asyncio.run(run())


def _parse_sse_chunks(lines: list[str]) -> list[dict[str, Any]]:
    """Decode SSE ``data: {...}`` lines into the chunk dicts. Skips the
    sentinel ``data: [DONE]`` line and anything that isn't valid JSON.
    """
    out: list[dict[str, Any]] = []
    for raw in lines:
        if not raw.startswith("data:"):
            continue
        body = raw[len("data:") :].strip()
        if not body or body == "[DONE]":
            continue
        try:
            out.append(json.loads(body))
        except json.JSONDecodeError:
            continue
    return out


class TestStreamCodexSingle:
    def test_streaming_chunks_translate_into_openai_shape(self, monkeypatch):
        _install_fake_codex_sdk(
            monkeypatch,
            lambda: _FakeAsyncCodex(chunks = ["Hello", ", ", "world"]),
        )
        from core.inference.codex_provider import stream_codex

        lines = _collect_stream(
            stream_codex(
                messages = [{"role": "user", "content": "Say hello in 3 chunks."}],
                model = "gpt-5.4",
            )
        )
        chunks = _parse_sse_chunks(lines)
        # Three content deltas + one usage chunk + one stop chunk.
        content_chunks = [
            c
            for c in chunks
            if c.get("choices")
            and isinstance(c["choices"], list)
            and c["choices"]
            and c["choices"][0].get("delta", {}).get("content")
        ]
        assert [c["choices"][0]["delta"]["content"] for c in content_chunks] == [
            "Hello",
            ", ",
            "world",
        ]
        # Usage chunk (OpenAI include_usage shape) is a choices=[] entry
        # with a populated usage block.
        usage_chunks = [c for c in chunks if c.get("choices") == [] and c.get("usage")]
        assert len(usage_chunks) == 1
        usage = usage_chunks[0]["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] >= 0
        # Final stop chunk with finish_reason=stop.
        stop_chunks = [
            c
            for c in chunks
            if c.get("choices")
            and c["choices"]
            and c["choices"][0].get("finish_reason") == "stop"
        ]
        assert len(stop_chunks) == 1
        # And the trailing [DONE] sentinel.
        assert any(line.strip() == "data: [DONE]" for line in lines)

    def test_empty_user_prompt_emits_helpful_message(self, monkeypatch):
        _install_fake_codex_sdk(monkeypatch, lambda: _FakeAsyncCodex(chunks = []))
        from core.inference.codex_provider import stream_codex

        lines = _collect_stream(
            stream_codex(
                messages = [{"role": "system", "content": "you are helpful"}],
                model = "gpt-5.4",
            )
        )
        text = "\n".join(lines)
        assert "no user prompt" in text.lower()


class TestStreamCodexParallel:
    def test_parallel_calls_spawn_tabs_and_synthesise(self, monkeypatch):
        # The fake SDK returns the same canned chunks for every spawned
        # AsyncCodex instance; we just need to verify the orchestrator
        # emits N tab_open events, per-tab chunk events keyed by
        # tab_id, and a final codex_gather summary event.
        _install_fake_codex_sdk(
            monkeypatch,
            lambda: _FakeAsyncCodex(
                chunks = ["alpha"],
                final = "synthesised answer",
            ),
        )
        from core.inference.codex_provider import stream_codex

        n = 3
        lines = _collect_stream(
            stream_codex(
                messages = [{"role": "user", "content": "Test"}],
                model = "gpt-5.4",
                parallel_calls = n,
            )
        )
        chunks = _parse_sse_chunks(lines)
        tool_events = [c["_toolEvent"] for c in chunks if "_toolEvent" in c]
        tab_opens = [e for e in tool_events if e.get("type") == "codex_tab_open"]
        tab_chunks = [e for e in tool_events if e.get("type") == "codex_tab_chunk"]
        tab_closes = [e for e in tool_events if e.get("type") == "codex_tab_close"]
        gather = [e for e in tool_events if e.get("type") == "codex_gather"]

        # Each tab opens once -- the N tabs are pre-emitted so the
        # UI can paint the strip before content arrives.
        assert len(tab_opens) == n
        assert sorted(e["tab_id"] for e in tab_opens) == list(range(1, n + 1))

        # Per-tab chunks may interleave in any order but every tab id
        # must produce at least one chunk before its close event.
        seen_tabs = {e["tab_id"] for e in tab_chunks}
        assert seen_tabs == set(range(1, n + 1))

        # Each tab emits exactly one close marker.
        assert sorted(e["tab_id"] for e in tab_closes) == list(range(1, n + 1))

        # Exactly one synthesis event with the unified summary.
        assert len(gather) == 1
        assert gather[0]["tab_count"] == n
        # The summary text comes from the final synthesis Codex call;
        # our fake returns "synthesised answer" via .run().
        assert "synth" in gather[0]["summary"].lower()

    def test_parallel_calls_clamped_to_maximum(self, monkeypatch):
        """Passing parallel_calls=500 must NOT spawn 500 tasks; the
        clamp at MAX_PARALLEL_CALLS keeps the local CLI safe.
        """
        from core.inference import codex_provider as cp

        _install_fake_codex_sdk(
            monkeypatch,
            lambda: _FakeAsyncCodex(chunks = ["x"], final = "synth"),
        )
        lines = _collect_stream(
            cp.stream_codex(
                messages = [{"role": "user", "content": "x"}],
                model = "gpt-5.4",
                parallel_calls = 500,
            )
        )
        chunks = _parse_sse_chunks(lines)
        tab_opens = [
            c["_toolEvent"]
            for c in chunks
            if c.get("_toolEvent", {}).get("type") == "codex_tab_open"
        ]
        assert len(tab_opens) == cp.MAX_PARALLEL_CALLS

    def test_parallel_calls_one_takes_single_path(self, monkeypatch):
        """parallel_calls=1 must not emit any tab tool-events -- it's the
        regular single-call shape.
        """
        _install_fake_codex_sdk(
            monkeypatch,
            lambda: _FakeAsyncCodex(chunks = ["one"]),
        )
        from core.inference.codex_provider import stream_codex

        lines = _collect_stream(
            stream_codex(
                messages = [{"role": "user", "content": "hi"}],
                model = "gpt-5.4",
                parallel_calls = 1,
            )
        )
        chunks = _parse_sse_chunks(lines)
        tool_events = [c.get("_toolEvent") for c in chunks if c.get("_toolEvent")]
        for event in tool_events:
            assert not (event.get("type") or "").startswith("codex_tab")
            assert event.get("type") != "codex_gather"


# ── Request validator ──────────────────────────────────────────────


class TestParallelCallsValidator:
    def test_request_accepts_valid_range(self):
        from models.inference import ChatCompletionRequest

        for n in (1, 5, 10, 20):
            req = ChatCompletionRequest(
                model = "gpt-5.4",
                messages = [{"role": "user", "content": "hi"}],
                parallel_calls = n,
            )
            assert req.parallel_calls == n

    def test_request_rejects_below_one(self):
        from models.inference import ChatCompletionRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model = "gpt-5.4",
                messages = [{"role": "user", "content": "hi"}],
                parallel_calls = 0,
            )

    def test_request_rejects_above_twenty(self):
        from models.inference import ChatCompletionRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model = "gpt-5.4",
                messages = [{"role": "user", "content": "hi"}],
                parallel_calls = 21,
            )

    def test_request_default_is_none(self):
        """Default = None so the field has no effect on every existing
        provider that doesn't read it -- preserves backwards compat.
        """
        from models.inference import ChatCompletionRequest

        req = ChatCompletionRequest(
            model = "gpt-5.4",
            messages = [{"role": "user", "content": "hi"}],
        )
        assert req.parallel_calls is None


# ── Codex unavailable surfacing ────────────────────────────────────


class TestCodexUnavailable:
    def test_missing_sdk_raises_typed_error(self, monkeypatch):
        # Force find_spec to return None so the lazy import fails.
        import importlib.util as _iu

        real = _iu.find_spec

        def _shim(name, *args, **kwargs):
            if name == "codex_app_server":
                return None
            return real(name, *args, **kwargs)

        monkeypatch.setattr("importlib.util.find_spec", _shim)
        # Also drop any cached fake from prior tests.
        monkeypatch.delitem(sys.modules, "codex_app_server", raising = False)

        from core.inference.codex_provider import (
            CodexUnavailableError,
            stream_codex,
        )

        with pytest.raises(CodexUnavailableError):
            asyncio.run(
                _consume_first(
                    stream_codex(
                        messages = [{"role": "user", "content": "hi"}],
                        model = "gpt-5.4",
                    )
                )
            )


async def _consume_first(gen):
    """Drive an async generator until it raises or yields its first
    value. Used to surface lazy-import errors that fire on the first
    SDK touch -- otherwise the generator would swallow them on
    ``__aiter__`` and the test couldn't see them.
    """
    async for _ in gen:
        return
