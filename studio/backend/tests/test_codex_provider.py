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

    def test_request_default_is_one(self):
        """Default = 1 so the field matches the single-call code path
        and the schema documentation. Non-codex providers ignore the
        field regardless of its value, so backwards compat is
        preserved.
        """
        from models.inference import ChatCompletionRequest

        req = ChatCompletionRequest(
            model = "gpt-5.4",
            messages = [{"role": "user", "content": "hi"}],
        )
        assert req.parallel_calls == 1


# ── Codex unavailable surfacing ────────────────────────────────────


class TestCodexUnavailable:
    def test_missing_sdk_raises_typed_error(self, monkeypatch):
        # Force find_spec to return None so the lazy import fails.
        # The provider probes both the canonical upstream name
        # ``openai_codex`` and the legacy alias ``codex_app_server``,
        # so we have to suppress both for the import to fail.
        import importlib.util as _iu

        real = _iu.find_spec
        _SDK_NAMES = {"openai_codex", "codex_app_server"}

        def _shim(name, *args, **kwargs):
            if name in _SDK_NAMES:
                return None
            return real(name, *args, **kwargs)

        monkeypatch.setattr("importlib.util.find_spec", _shim)
        # Also drop any cached fakes from prior tests.
        for _name in _SDK_NAMES:
            monkeypatch.delitem(sys.modules, _name, raising = False)

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


class TestCodexHardenedRegressions:
    """Tests covering the post-review hardening pass.

    Each test pins a specific regression: the wrong subcommand
    (``codex auth login`` → ``codex login``), the wrong SDK package
    name (``codex_app_server`` → ``openai_codex`` with legacy alias),
    the ``not logged in`` substring footgun, the ANSI-wrapped device
    URL, and the fan-out cancellation contract.
    """

    def test_sdk_probes_openai_codex_first(self, monkeypatch):
        """The canonical upstream name must be tried before the alias."""
        import importlib.util as _iu

        real = _iu.find_spec
        calls: list[str] = []

        def _shim(name, *args, **kwargs):
            if name in ("openai_codex", "codex_app_server"):
                calls.append(name)
                return None
            return real(name, *args, **kwargs)

        monkeypatch.setattr("importlib.util.find_spec", _shim)
        from core.inference.codex_availability import _sdk_importable

        assert _sdk_importable() is False
        assert (
            calls and calls[0] == "openai_codex"
        ), f"availability probe must check openai_codex first; saw {calls}"

    def test_login_status_uses_login_subcommand(self):
        """Upstream is `codex login status`, NOT `codex auth status`."""
        src = (
            "/mnt/disks/unslothai/ubuntu/workspace_11/unsloth_pr5724/"
            "studio/backend/core/inference/codex_availability.py"
        )
        text = open(src).read()
        assert (
            '"auth", "status"' not in text
        ), "_detect_logged_in must use `codex login status`, not `codex auth status`"
        assert '"login", "status"' in text

    def test_device_login_uses_login_subcommand(self):
        src = (
            "/mnt/disks/unslothai/ubuntu/workspace_11/unsloth_pr5724/"
            "studio/backend/core/inference/codex_provider.py"
        )
        text = open(src).read()
        assert (
            '"auth", "login", "--device-auth"' not in text
        ), "stream_codex_device_login must use `codex login --device-auth`"
        assert '"login", "--device-auth"' in text

    def test_not_logged_in_not_misparsed_as_logged_in(self):
        """The substring "logged in" inside "not logged in" must not
        flip the detection to True."""
        import asyncio

        from core.inference import codex_availability as av

        async def _fake_run_cli(args, **kw):
            return (0, "Not logged in. Run `codex login` to authenticate.", "")

        orig = av._run_cli
        av._run_cli = _fake_run_cli  # type: ignore[assignment]
        try:
            result = asyncio.run(av._detect_logged_in())
            assert result is False, "'Not logged in' was misparsed as logged_in=True"
        finally:
            av._run_cli = orig  # type: ignore[assignment]

    def test_logged_in_is_detected(self):
        import asyncio

        from core.inference import codex_availability as av

        async def _fake_run_cli(args, **kw):
            return (0, "Logged in using ChatGPT", "")

        orig = av._run_cli
        av._run_cli = _fake_run_cli  # type: ignore[assignment]
        try:
            result = asyncio.run(av._detect_logged_in())
            assert result is True
        finally:
            av._run_cli = orig  # type: ignore[assignment]

    def test_multi_turn_prompt_includes_prior_turns(self):
        """The Codex prompt MUST contain prior assistant turns."""
        from core.inference.codex_provider import _last_user_prompt

        msgs = [
            {"role": "user", "content": "what is the capital of france?"},
            {"role": "assistant", "content": "Paris."},
            {"role": "user", "content": "and germany?"},
        ]
        prompt = _last_user_prompt(msgs)
        assert "and germany?" in prompt
        assert (
            "Paris" in prompt
        ), f"PRIOR ASSISTANT TURN DROPPED — multi-turn broken. Prompt:\n{prompt}"
        assert "capital of france" in prompt.lower()

    def test_single_turn_prompt_unchanged(self):
        """Single-turn case must not get the User:/Assistant: framing."""
        from core.inference.codex_provider import _last_user_prompt

        prompt = _last_user_prompt([{"role": "user", "content": "hi"}])
        assert prompt == "hi"

    def test_default_models_no_o3(self):
        """The Codex registry must not advertise `o3` (not in upstream)."""
        from core.inference.providers import PROVIDER_REGISTRY

        codex = PROVIDER_REGISTRY["codex"]
        assert (
            "o3" not in codex["default_models"]
        ), "o3 is not a Codex model; remove from default_models"
        assert "gpt-5.5" in codex["default_models"]

    def test_inference_route_no_raw_exc_leak(self):
        """SSE error frame must NOT echo str(exc) verbatim (CodeQL)."""
        import re

        src = (
            "/mnt/disks/unslothai/ubuntu/workspace_11/unsloth_pr5724/"
            "studio/backend/routes/inference.py"
        )
        text = open(src).read()
        bad = re.findall(r'f["\']Codex error:\s*\{exc\}["\']', text)
        assert not bad, f"raw exception in SSE: {bad}"

    def test_codex_route_no_raw_exc_leak(self):
        """codex.py SSE stream wrapping must also not leak str(exc)."""
        import re

        src = (
            "/mnt/disks/unslothai/ubuntu/workspace_11/unsloth_pr5724/"
            "studio/backend/routes/codex.py"
        )
        text = open(src).read()
        for line in text.splitlines():
            ls = line.strip()
            if ls.startswith("yield ") and re.search(r"\{exc\}|\{e\}", ls):
                assert False, f"raw exception leaked: {ls}"

    def test_parallel_tab_error_sanitised(self, monkeypatch):
        """A worker that raises with a path-leaking message must NOT
        send that text to the client; the SSE codex_tab_error event
        must carry a generic message + exception_type.
        """
        fake = _FakeAsyncCodex(
            raise_on_start = RuntimeError(
                "secret /home/alice/.codex/config.json token=abc"
            )
        )
        _install_fake_codex_sdk(monkeypatch, lambda: fake)
        from core.inference.codex_provider import stream_codex

        chunks: list[str] = []

        async def _collect():
            async for c in stream_codex(
                messages = [{"role": "user", "content": "hi"}],
                model = "gpt-5.5",
                parallel_calls = 2,
            ):
                chunks.append(c)

        asyncio.run(_collect())
        body = "".join(chunks)
        assert (
            "secret /home/alice" not in body
        ), "raw exception text leaked into codex_tab_error SSE frame"
        assert "Codex tab failed" in body or "exception_type" in body

    def test_thread_turn_stream_path_taken(self, monkeypatch):
        """The canonical openai_codex API uses thread.turn(prompt).stream();
        the provider must prefer that over the legacy run_streaming hook.
        """
        events_seen = {"turn_called": False, "run_streaming_called": False}

        class _TurnEvent:
            def __init__(self, txt):
                self.payload = {"text": txt}

        class _TurnHandle:
            def __init__(self, prompt):
                self.prompt = prompt

            async def stream(self):
                yield _TurnEvent("hello ")
                yield _TurnEvent("from turn.stream")

        class _ThreadWithTurn:
            def turn(self, prompt):
                events_seen["turn_called"] = True
                return _TurnHandle(prompt)

            def run_streaming(self, prompt):
                events_seen["run_streaming_called"] = True
                raise AssertionError("should not be called when turn().stream() works")

            async def run(self, prompt):
                raise AssertionError("should not fall through to buffered run()")

        class _Async:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def thread_start(self, **kw):
                return _ThreadWithTurn()

        _install_fake_codex_sdk(monkeypatch, _Async)
        from core.inference.codex_provider import stream_codex

        chunks: list[str] = []

        async def _collect():
            async for c in stream_codex(
                messages = [{"role": "user", "content": "hi"}],
                model = "gpt-5.5",
                parallel_calls = 1,
            ):
                chunks.append(c)

        asyncio.run(_collect())
        assert events_seen["turn_called"], "thread.turn() never called"
        assert not events_seen["run_streaming_called"]
        body = "".join(chunks)
        # Each text chunk wraps in its own SSE delta, so check both pieces.
        assert '"content": "hello "' in body
        assert '"content": "from turn.stream"' in body


async def _consume_first(gen):
    """Drive an async generator until it raises or yields its first
    value. Used to surface lazy-import errors that fire on the first
    SDK touch -- otherwise the generator would swallow them on
    ``__aiter__`` and the test couldn't see them.
    """
    async for _ in gen:
        return
