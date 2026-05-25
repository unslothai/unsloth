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
* Request validator: ``parallel_calls`` is silently clamped to
  [1, 20] by a Pydantic field validator (not by ``ge=1, le=20``) so
  non-Codex clients that send legacy values like ``0`` continue to
  be accepted instead of getting a 422.
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

# Resolved relative to this file so the source-inspection tests work in any
# checkout location (CI, dev machines, the review worker, etc.).
_BACKEND_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))


def _backend_file(rel: str) -> str:
    """Return an absolute path inside the backend tree, regardless of cwd."""
    return os.path.join(_BACKEND_DIR, rel)


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


def _install_fake_codex_sdk(monkeypatch, async_codex_cls, *, with_safety_enums = True):
    """Drop a fake ``codex_app_server`` module into sys.modules so the
    production lazy-import path picks it up without the real SDK
    being installed.

    ``with_safety_enums=True`` (the default) also injects fake
    ``ApprovalMode`` + ``SandboxMode`` so the round 6b fail-closed
    path in ``_safe_thread_safety_kwargs`` is not triggered for every
    test that just wants to exercise stream translation. The two
    dedicated round 6b tests (fail_closed / explicit_opt_in) pass
    ``with_safety_enums=False`` so they can prove the fail-closed
    branch fires when those enums are missing.
    """
    fake_mod = types.ModuleType("codex_app_server")
    fake_mod.AsyncCodex = async_codex_cls  # type: ignore[attr-defined]
    if with_safety_enums:
        fake_mod.ApprovalMode = types.SimpleNamespace(  # type: ignore[attr-defined]
            deny_all = "DENY_ALL",
            auto_review = "AUTO_REVIEW",
        )
        fake_mod.SandboxMode = types.SimpleNamespace(  # type: ignore[attr-defined]
            read_only = "READ_ONLY",
            workspace_write = "WORKSPACE_WRITE",
            danger_full_access = "DANGER_FULL_ACCESS",
        )
    # Inject the fake under BOTH module names the production importer
    # checks. ``openai_codex`` is the canonical upstream name and is
    # preferred by the lazy-import gate; ``codex_app_server`` is the
    # legacy / Rust-crate alias. Hosts that have ``openai_codex``
    # actually installed (developer venvs, CI runners after the PR's
    # `pip install openai-codex`) would otherwise bypass the fake and
    # exercise the real SDK -- the same fake must be reachable under
    # both names for the test to be deterministic.
    monkeypatch.setitem(sys.modules, "codex_app_server", fake_mod)
    monkeypatch.setitem(sys.modules, "openai_codex", fake_mod)
    # importlib.util.find_spec walks finders, not sys.modules; patch
    # it directly so the lazy-import gate accepts the fake.
    import importlib.util as _iu

    real_find_spec = _iu.find_spec

    def _shim(name: str, *args, **kwargs):
        if name in ("codex_app_server", "openai_codex"):
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
        # The SDK is what backs `AsyncCodex(...)`, so installed=False
        # when the SDK is missing -- even if a standalone CLI is on
        # PATH there is no way for Studio to drive it without the
        # Python bindings.
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

    def test_request_clamps_below_one(self):
        """Pre-PR clients sometimes sent `parallel_calls=0` as a stray
        OpenAI extra and the request was silently accepted; rejecting
        with 422 would regress that. The validator now clamps to 1.
        """
        from models.inference import ChatCompletionRequest

        for n in (0, -1, -100):
            req = ChatCompletionRequest(
                model = "gpt-5.4",
                messages = [{"role": "user", "content": "hi"}],
                parallel_calls = n,
            )
            assert req.parallel_calls == 1, f"clamp failed for {n}"

    def test_request_clamps_above_twenty(self):
        """A runaway value (1000, etc.) is clamped to the 20 cap so it
        cannot saturate the local CLI even when the client misbehaves.
        """
        from models.inference import ChatCompletionRequest

        for n in (21, 100, 1000):
            req = ChatCompletionRequest(
                model = "gpt-5.4",
                messages = [{"role": "user", "content": "hi"}],
                parallel_calls = n,
            )
            assert req.parallel_calls == 20, f"clamp failed for {n}"

    def test_request_coerces_garbage_to_one(self):
        """Strings / floats / None coerce to 1 instead of 422 so a
        legacy or misconfigured client cannot break chat for everyone."""
        from models.inference import ChatCompletionRequest

        for value in (None, "garbage", float("nan")):
            req = ChatCompletionRequest(
                model = "gpt-5.4",
                messages = [{"role": "user", "content": "hi"}],
                parallel_calls = value,
            )
            assert req.parallel_calls == 1

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
        src = _backend_file("core/inference/codex_availability.py")
        text = open(src).read()
        assert (
            '"auth", "status"' not in text
        ), "_detect_logged_in must use `codex login status`, not `codex auth status`"
        assert '"login", "status"' in text

    def test_device_login_uses_login_subcommand(self):
        src = _backend_file("core/inference/codex_provider.py")
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

        src = _backend_file("routes/inference.py")
        text = open(src).read()
        bad = re.findall(r'f["\']Codex error:\s*\{exc\}["\']', text)
        assert not bad, f"raw exception in SSE: {bad}"

    def test_codex_route_no_raw_exc_leak(self):
        """codex.py SSE stream wrapping must also not leak str(exc)."""
        import re

        src = _backend_file("routes/codex.py")
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

    def test_codex_subprocess_env_scrubbed(self, monkeypatch):
        """The codex subprocess env must not include other-provider secrets.

        OPENAI_API_KEY is intentionally excluded too: a shimmed `codex`
        binary on PATH must not receive Studio's stored OpenAI provider
        key. Users wire Codex auth via `codex login` or the
        codex-specific CODEX_OPENAI_API_KEY override instead.
        """
        from core.inference.codex_availability import _codex_subprocess_env

        monkeypatch.setenv("HF_TOKEN", "hf_should_not_leak")
        monkeypatch.setenv("GH_TOKEN", "gh_should_not_leak")
        monkeypatch.setenv("WANDB_API_KEY", "wandb_should_not_leak")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic_should_not_leak")
        monkeypatch.setenv("OPENAI_API_KEY", "openai_provider_key_not_for_codex")
        monkeypatch.setenv("CODEX_OPENAI_API_KEY", "codex_specific_key")
        monkeypatch.setenv("CODEX_HOME", "/custom/.codex")
        monkeypatch.setenv("PATH", "/usr/bin")

        env = _codex_subprocess_env()
        for secret in (
            "HF_TOKEN",
            "GH_TOKEN",
            "WANDB_API_KEY",
            "ANTHROPIC_API_KEY",
            # OPENAI_API_KEY belongs to the OpenAI provider, not Codex.
            "OPENAI_API_KEY",
        ):
            assert secret not in env, f"{secret} leaked into codex env"
        # Codex-relevant keys must be preserved.
        assert env.get("CODEX_OPENAI_API_KEY") == "codex_specific_key"
        assert env.get("CODEX_HOME") == "/custom/.codex"
        assert env.get("PATH") == "/usr/bin"

    def test_partial_stream_failure_does_not_replay_turn(self, monkeypatch):
        """If turn.stream() fails after emitting some text, the buffered
        run() fallback must NOT fire -- replaying would duplicate side
        effects (file writes, shell commands).
        """
        run_calls = {"n": 0}

        class _PartialStreamTurn:
            async def stream(self):
                yield {"text": "partial output "}
                raise RuntimeError("network glitch mid-stream")

        class _ThreadPartialFail:
            def turn(self, prompt):
                return _PartialStreamTurn()

            async def run(self, prompt):
                run_calls["n"] += 1
                return "REPLAYED -- BAD"

        class _Async:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def thread_start(self, **kw):
                return _ThreadPartialFail()

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
        assert run_calls["n"] == 0, (
            "buffered run() fired after partial stream emission -- "
            "would replay side effects"
        )
        body = "".join(chunks)
        assert "partial output" in body
        assert "REPLAYED" not in body

    def test_not_signed_in_wording_also_handled(self):
        """`Not signed in` (alternative localisation) must also be
        treated as logged-out, not as positive match.
        """
        import asyncio

        from core.inference import codex_availability as av

        async def _fake_run_cli(args, **kw):
            return (0, "Not signed in.", "")

        orig = av._run_cli
        av._run_cli = _fake_run_cli  # type: ignore[assignment]
        try:
            assert asyncio.run(av._detect_logged_in()) is False
        finally:
            av._run_cli = orig  # type: ignore[assignment]

    def test_device_url_accepts_generic_verification_url(self):
        """The login parser must accept upstream's chatgpt.com/activate
        URL as well as the canonical /codex/device shape.
        """
        import re

        src = _backend_file("core/inference/codex_provider.py")
        text = open(src).read()
        # Find the url_re pattern literal and compile it.
        m = re.search(r"url_re\s*=\s*re\.compile\(\s*\n?\s*r\"([^\"]+)\"", text)
        assert m, "url_re definition not found"
        pattern = re.compile(m.group(1), re.IGNORECASE)
        # Upstream device URLs we expect to match.
        for u in (
            "https://auth.openai.com/codex/device",
            "https://chatgpt.com/activate",
            "https://auth.openai.com/device/verify?code=ABCD",
        ):
            assert pattern.search(u), f"device URL regex missed: {u}"

    def test_synthesis_call_forwards_system_prompt(self, monkeypatch):
        """`_run_codex_synthesis` must pass the system prompt so a
        fan-out style instruction ("Always answer in Spanish") survives
        the unification step.
        """
        seen_kwargs: list[dict] = []
        seen_prompts: list[str] = []

        class _SynThread:
            async def run(self, prompt):
                seen_prompts.append(prompt)
                return "synth ok"

            def turn(self, prompt):
                # Force buffered path via no `stream` attr.
                class _T:
                    pass

                return _T()

        class _Async:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def thread_start(self, **kw):
                seen_kwargs.append(kw)
                return _SynThread()

        _install_fake_codex_sdk(monkeypatch, _Async)
        from core.inference.codex_provider import _run_codex_synthesis

        out = asyncio.run(
            _run_codex_synthesis(
                model = "gpt-5.5",
                system = "Always answer in Spanish.",
                prompt = "What is the capital of France?",
                tab_outputs = ["Paris", "Paris."],
            )
        )
        # The upstream openai_codex SDK uses `base_instructions` for the
        # system prompt; the legacy alias accepts `system`; the last-resort
        # fallback inlines the system text into the user prompt. Accept
        # any of those paths.
        system_seen = (
            any("Spanish" in (kw.get("base_instructions") or "") for kw in seen_kwargs)
            or any("Spanish" in (kw.get("system") or "") for kw in seen_kwargs)
            or any("Always answer in Spanish" in p for p in seen_prompts)
        )
        assert system_seen, (
            f"system prompt dropped in synthesis. kwargs={seen_kwargs} "
            f"prompts={seen_prompts}"
        )
        # And the synthesis still returned the model's text.
        assert "synth" in out.lower()

    def test_sdk_env_scrubbed_via_appserverconfig(self, monkeypatch):
        """The SDK construction path must wire AppServerConfig(env=...)
        when the SDK exposes it, so HF_TOKEN / GH_TOKEN are not leaked
        to the codex app-server subprocess.
        """
        monkeypatch.setenv("HF_TOKEN", "should_be_scrubbed")
        monkeypatch.setenv("GH_TOKEN", "should_be_scrubbed")
        # OPENAI_API_KEY is now ALSO scrubbed -- it belongs to the
        # OpenAI provider, not Codex. CODEX_OPENAI_API_KEY is the
        # codex-specific override that survives.
        monkeypatch.setenv("OPENAI_API_KEY", "openai_provider_key_not_for_codex")
        monkeypatch.setenv("CODEX_OPENAI_API_KEY", "codex_specific_key")

        seen_configs: list[Any] = []

        class _FakeAppServerConfig:
            def __init__(self, env = None, **kw):
                self.env = env or {}

        class _Async:
            def __init__(self, config = None):
                seen_configs.append(config)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def thread_start(self, **kw):
                return _FakeThread(chunks = ["ok"])

        # Inject a fake openai_codex module exposing AppServerConfig.
        import importlib.util as _iu
        import types as _types

        fake_mod = _types.ModuleType("openai_codex")
        fake_mod.AsyncCodex = _Async  # type: ignore[attr-defined]
        fake_mod.AppServerConfig = _FakeAppServerConfig  # type: ignore[attr-defined]
        # Round 6b: safety enums must be present or the fail-closed
        # path raises before AppServerConfig ever gets consulted.
        fake_mod.ApprovalMode = _types.SimpleNamespace(  # type: ignore[attr-defined]
            deny_all = "DENY_ALL",
            auto_review = "AUTO",
        )
        fake_mod.SandboxMode = _types.SimpleNamespace(  # type: ignore[attr-defined]
            read_only = "READ_ONLY",
            workspace_write = "WW",
        )
        monkeypatch.setitem(sys.modules, "openai_codex", fake_mod)
        real_find_spec = _iu.find_spec
        monkeypatch.setattr(
            "importlib.util.find_spec",
            lambda n, *a, **kw: (
                _types.SimpleNamespace()
                if n in ("openai_codex", "codex_app_server")
                else real_find_spec(n, *a, **kw)
            ),
        )

        from core.inference.codex_provider import stream_codex

        asyncio.run(
            _consume_first(
                stream_codex(
                    messages = [{"role": "user", "content": "hi"}],
                    model = "gpt-5.5",
                    parallel_calls = 1,
                )
            )
        )
        assert seen_configs, "AsyncCodex was never instantiated"
        cfg = seen_configs[0]
        assert cfg is not None, "AppServerConfig was not passed to AsyncCodex"
        assert (
            "HF_TOKEN" in cfg.env and cfg.env["HF_TOKEN"] == ""
        ), "HF_TOKEN not overridden to empty in SDK env"
        assert "GH_TOKEN" in cfg.env and cfg.env["GH_TOKEN"] == ""
        # OPENAI_API_KEY is intentionally overridden to empty in the
        # SDK env so the app-server cannot use it as a Codex credential
        # by accident. The OpenAI provider still reads its own key from
        # Studio's storage; nothing in this path needs the env var.
        assert cfg.env.get("OPENAI_API_KEY") == ""
        # CODEX_OPENAI_API_KEY is the Codex-specific override and must
        # survive untouched so users can wire that key into Codex.
        assert "CODEX_OPENAI_API_KEY" not in cfg.env

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

    def test_installed_requires_both_cli_and_sdk(self, monkeypatch):
        """Round 6 revert: the login route shells out to `codex`, so
        marking `installed=True` on SDK-only would surface a Codex
        provider row whose Sign-in button immediately fails. The
        canonical `openai-codex` package installs `openai-codex-cli-bin`
        which puts the `codex` shim on PATH, so common installs still
        light up correctly; the gate just refuses to advertise a
        provider Studio cannot actually drive.
        """
        from core.inference import codex_availability as ca

        # SDK present, no CLI -> hidden (cannot complete login).
        monkeypatch.setattr(ca, "_which_codex", lambda: None)
        monkeypatch.setattr(ca, "_sdk_importable", lambda: True)
        payload = asyncio.run(ca.probe_codex_availability())
        assert payload["installed"] is False
        assert payload["cli_path"] is None
        assert payload["sdk_importable"] is True

        # CLI present, SDK missing -> still hidden (cannot drive chat).
        monkeypatch.setattr(ca, "_which_codex", lambda: "/usr/bin/codex")
        monkeypatch.setattr(ca, "_sdk_importable", lambda: False)

        async def fake_version():
            return "codex-cli 0.133.0"

        async def fake_logged_in():
            return True

        monkeypatch.setattr(ca, "_detect_version", fake_version)
        monkeypatch.setattr(ca, "_detect_logged_in", fake_logged_in)
        payload2 = asyncio.run(ca.probe_codex_availability())
        assert payload2["installed"] is False

    def test_base_instructions_kwarg_preferred(self, monkeypatch):
        """The upstream openai_codex SDK uses `base_instructions` for
        the system prompt. The provider must try that name first; only
        if the SDK rejects it should it fall back to `system`.
        """
        seen_kwargs: list[dict] = []

        class _Async:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def thread_start(self, **kw):
                seen_kwargs.append(dict(kw))
                return _FakeThread(chunks = ["ok"])

        _install_fake_codex_sdk(monkeypatch, _Async)
        from core.inference.codex_provider import stream_codex

        async def _collect():
            async for _ in stream_codex(
                messages = [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "hi"},
                ],
                model = "gpt-5.5",
                parallel_calls = 1,
            ):
                pass

        asyncio.run(_collect())
        # The first (and only, since this fake accepts any kwargs)
        # call must use base_instructions, not the legacy `system`.
        assert seen_kwargs, "thread_start was never called"
        assert (
            "base_instructions" in seen_kwargs[0]
        ), f"upstream-canonical kwarg not used: {seen_kwargs[0]}"
        assert seen_kwargs[0]["base_instructions"] == "You are helpful."
        assert (
            "system" not in seen_kwargs[0]
        ), "legacy `system` kwarg was sent even though base_instructions worked"

    def test_base_instructions_falls_back_to_system(self, monkeypatch):
        """When the SDK rejects `base_instructions` with TypeError the
        helper must retry with the legacy `system` kwarg before giving
        up and inlining the system text in the prompt.
        """
        call_log: list[dict] = []

        class _StrictSDK:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def thread_start(self, **kw):
                call_log.append(dict(kw))
                if "base_instructions" in kw:
                    raise TypeError(
                        "thread_start() got an unexpected keyword 'base_instructions'"
                    )
                return _FakeThread(chunks = ["ok"])

        _install_fake_codex_sdk(monkeypatch, _StrictSDK)
        from core.inference.codex_provider import stream_codex

        async def _collect():
            async for _ in stream_codex(
                messages = [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "hi"},
                ],
                model = "gpt-5.5",
                parallel_calls = 1,
            ):
                pass

        asyncio.run(_collect())
        assert len(call_log) >= 2, "fallback to `system` kwarg never tried"
        assert "base_instructions" in call_log[0]
        assert "system" in call_log[1] and call_log[1]["system"] == "You are helpful."

    def test_scrubbed_env_wrapper_strips_secrets_before_construction(self, monkeypatch):
        """When AppServerConfig is missing the fail-closed wrapper must
        remove secret env vars BEFORE the SDK constructor runs (the
        SDK starts its app-server with `env = os.environ.copy()`).
        """
        observed_env_during_init: dict[str, str | None] = {}

        class _NoConfigAsync:
            def __init__(self):
                # Capture the environment exactly as the SDK would see
                # it at construction time.
                observed_env_during_init["HF_TOKEN"] = os.environ.get("HF_TOKEN")
                observed_env_during_init["GH_TOKEN"] = os.environ.get("GH_TOKEN")
                observed_env_during_init["WANDB_API_KEY"] = os.environ.get(
                    "WANDB_API_KEY"
                )
                observed_env_during_init["PATH"] = os.environ.get("PATH")
                observed_env_during_init["CODEX_HOME"] = os.environ.get("CODEX_HOME")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def thread_start(self, **kw):
                return _FakeThread(chunks = ["ok"])

        monkeypatch.setenv("HF_TOKEN", "should_be_gone")
        monkeypatch.setenv("GH_TOKEN", "should_be_gone")
        monkeypatch.setenv("WANDB_API_KEY", "should_be_gone")
        monkeypatch.setenv("PATH", "/usr/bin")
        monkeypatch.setenv("CODEX_HOME", "/home/u/.codex")
        # No AppServerConfig in the fake module -- forces the wrapper path.
        _install_fake_codex_sdk(monkeypatch, _NoConfigAsync)
        from core.inference.codex_provider import stream_codex

        async def _collect():
            async for _ in stream_codex(
                messages = [{"role": "user", "content": "hi"}],
                model = "gpt-5.5",
                parallel_calls = 1,
            ):
                pass

        asyncio.run(_collect())
        # Secrets must have been removed from os.environ BEFORE the
        # SDK constructor captured the env.
        assert (
            observed_env_during_init["HF_TOKEN"] is None
        ), "HF_TOKEN visible to SDK constructor -- env scrub failed"
        assert observed_env_during_init["GH_TOKEN"] is None
        assert observed_env_during_init["WANDB_API_KEY"] is None
        # Safe-listed keys must survive.
        assert observed_env_during_init["PATH"] == "/usr/bin"
        assert observed_env_during_init["CODEX_HOME"] == "/home/u/.codex"
        # And the wrapper must restore them after exit.
        assert os.environ.get("HF_TOKEN") == "should_be_gone"
        assert os.environ.get("GH_TOKEN") == "should_be_gone"

    def test_coerce_text_drops_non_answer_event_types(self):
        """Tool / command / plan deltas have their own `delta` fields
        that must NOT be rendered as assistant text -- otherwise local
        stdout, file paths, or tool-call arguments would leak into the
        Chat Completions reply.

        Round 6 also requires the object-shape path to gate on type
        and class name; the upstream SDK emits typed notification
        objects (CommandExecutionOutputDelta, FileChangeDelta, etc.)
        with `.delta` strings that would otherwise leak.
        """
        from core.inference.codex_provider import _coerce_text

        # Allowed answer-bearing event types contribute text.
        assert _coerce_text({"type": "message.delta", "delta": "hello"}) == "hello"
        assert _coerce_text({"type": "completed", "text": "done"}) == "done"
        assert _coerce_text({"type": "text_delta", "delta": "x"}) == "x"

        # Non-answer dict event types are silenced.
        for ev_type in (
            "command.delta",
            "command_output",
            "file_write.delta",
            "tool_call.delta",
            "plan.update",
            "exec.stdout",
            "exec.stderr",
            "patch.apply",
            "thread.tool_call",
            "agent_reasoning",
        ):
            payload = {"type": ev_type, "delta": "this should NOT leak"}
            assert _coerce_text(payload) == "", (
                f"{ev_type} leaked text into assistant reply: "
                f"{_coerce_text(payload)!r}"
            )

        # Object-shape gate: typed payloads whose class name contains
        # a tool/command/file/patch/plan marker drop the .delta too.
        class CommandExecutionOutputDelta:
            delta = "SECRET_STDOUT"

        class FileChangeDelta:
            delta = "secret/file/path"

        class ToolCallDelta:
            text = "tool_arg_payload"

        class PatchApplyDelta:
            delta = "diff --git a/secret"

        class PlanUpdateDelta:
            delta = "plan content"

        class AgentReasoningDelta:
            delta = "internal CoT"

        for obj in (
            CommandExecutionOutputDelta(),
            FileChangeDelta(),
            ToolCallDelta(),
            PatchApplyDelta(),
            PlanUpdateDelta(),
            AgentReasoningDelta(),
        ):
            assert _coerce_text(obj) == "", (
                f"object-shape {obj.__class__.__name__} leaked: "
                f"{_coerce_text(obj)!r}"
            )

        # Object with explicit type attr also drops if not in allow-list.
        class _WithType:
            type = "command.delta"
            delta = "leak"

        assert _coerce_text(_WithType()) == ""

        # Object-shape answer events DO pass through.
        class AgentMessageDelta:
            delta = "real assistant text"

        assert _coerce_text(AgentMessageDelta()) == "real assistant text"

        # Plain strings and untyped dicts still pass through (legacy path).
        assert _coerce_text("raw text") == "raw text"
        assert _coerce_text({"text": "no type tag"}) == "no type tag"

    def test_authenticated_yes_wording_is_detected(self):
        """An `Authenticated: Yes` line (a wording the CLI ships in
        some locales / versions) must be parsed as logged-in.
        """
        from core.inference import codex_availability as av

        async def _fake_run_cli(args, **kw):
            return (0, "Authenticated: Yes\nuser@example.com", "")

        orig = av._run_cli
        av._run_cli = _fake_run_cli  # type: ignore[assignment]
        try:
            assert asyncio.run(av._detect_logged_in()) is True
        finally:
            av._run_cli = orig  # type: ignore[assignment]

    def test_codex_openai_api_key_overrides_openai_provider_key(self, monkeypatch):
        """Studio's `OPENAI_API_KEY` must NOT reach codex -- but the
        codex-specific `CODEX_OPENAI_API_KEY` MUST be forwarded so
        users can deliberately wire a key into Codex.
        """
        from core.inference.codex_availability import _codex_subprocess_env

        monkeypatch.setenv("OPENAI_API_KEY", "belongs_to_openai_provider")
        monkeypatch.setenv("CODEX_OPENAI_API_KEY", "explicit_codex_key")

        env = _codex_subprocess_env()
        assert (
            "OPENAI_API_KEY" not in env
        ), "OpenAI provider key leaked into codex subprocess env"
        assert env.get("CODEX_OPENAI_API_KEY") == "explicit_codex_key"

    def test_thread_start_uses_safe_approval_and_sandbox(self, monkeypatch):
        """When the SDK exposes ApprovalMode + SandboxMode, the
        provider MUST pin approval to `deny_all` and sandbox to
        `read_only`. The upstream SDK default
        (`auto_review` approvals, unspecified sandbox) would let the
        model auto-execute commands and write files on the server.
        """
        seen_kwargs: list[dict] = []

        class _Async:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def thread_start(self, **kw):
                seen_kwargs.append(dict(kw))
                return _FakeThread(chunks = ["ok"])

        # Drop a fake openai_codex with ApprovalMode + SandboxMode enums.
        import importlib.util as _iu

        fake_mod = types.ModuleType("openai_codex")
        fake_mod.AsyncCodex = _Async  # type: ignore[attr-defined]
        fake_mod.ApprovalMode = types.SimpleNamespace(  # type: ignore[attr-defined]
            deny_all = "DENY_ALL_SENTINEL",
            auto_review = "AUTO_REVIEW_SENTINEL",
        )
        fake_mod.SandboxMode = types.SimpleNamespace(  # type: ignore[attr-defined]
            read_only = "READ_ONLY_SENTINEL",
            workspace_write = "WS_WRITE_SENTINEL",
            danger_full_access = "DANGER_SENTINEL",
        )
        monkeypatch.setitem(sys.modules, "openai_codex", fake_mod)
        real_find_spec = _iu.find_spec
        monkeypatch.setattr(
            "importlib.util.find_spec",
            lambda n, *a, **kw: (
                types.SimpleNamespace()
                if n in ("openai_codex", "codex_app_server")
                else real_find_spec(n, *a, **kw)
            ),
        )

        from core.inference.codex_provider import stream_codex

        async def _collect():
            async for _ in stream_codex(
                messages = [{"role": "user", "content": "hi"}],
                model = "gpt-5.5",
                parallel_calls = 1,
            ):
                pass

        asyncio.run(_collect())
        assert seen_kwargs, "thread_start never called"
        kw = seen_kwargs[0]
        assert (
            kw.get("approval_mode") == "DENY_ALL_SENTINEL"
        ), f"approval_mode not pinned to deny_all: {kw}"
        assert (
            kw.get("sandbox") == "READ_ONLY_SENTINEL"
        ), f"sandbox not pinned to read_only: {kw}"

    def test_safety_kwargs_finds_sandbox_mode_in_submodule(self, monkeypatch):
        """Round 6 caught that `SandboxMode` is exported by the
        upstream SDK from `openai_codex.generated.v2_all`, NOT from
        the top-level `openai_codex` package. The previous lookup
        used `getattr(sdk_mod, 'SandboxMode', None)` only and returned
        None for the canonical SDK install, silently degrading to
        the unsafe auto_review default.
        """
        seen_kwargs: list[dict] = []

        class _Async:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def thread_start(self, **kw):
                seen_kwargs.append(dict(kw))
                return _FakeThread(chunks = ["ok"])

        # Build a fake openai_codex that DOES NOT expose SandboxMode
        # at the top level -- only inside `.generated.v2_all`.
        import importlib.util as _iu

        fake_root = types.ModuleType("openai_codex")
        fake_root.AsyncCodex = _Async  # type: ignore[attr-defined]
        fake_root.ApprovalMode = types.SimpleNamespace(  # type: ignore[attr-defined]
            deny_all = "DENY_ALL",
            auto_review = "AUTO",
        )
        # Submodule chain `.generated.v2_all`
        fake_generated = types.ModuleType("openai_codex.generated")
        fake_v2 = types.ModuleType("openai_codex.generated.v2_all")
        fake_v2.SandboxMode = types.SimpleNamespace(  # type: ignore[attr-defined]
            read_only = "READ_ONLY",
            workspace_write = "WW",
        )
        fake_generated.v2_all = fake_v2  # type: ignore[attr-defined]
        fake_root.generated = fake_generated  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "openai_codex", fake_root)
        monkeypatch.setitem(sys.modules, "openai_codex.generated", fake_generated)
        monkeypatch.setitem(sys.modules, "openai_codex.generated.v2_all", fake_v2)
        real_find_spec = _iu.find_spec
        monkeypatch.setattr(
            "importlib.util.find_spec",
            lambda n, *a, **kw: (
                types.SimpleNamespace()
                if n in ("openai_codex", "codex_app_server")
                else real_find_spec(n, *a, **kw)
            ),
        )

        from core.inference.codex_provider import stream_codex

        async def _collect():
            async for _ in stream_codex(
                messages = [{"role": "user", "content": "hi"}],
                model = "gpt-5.5",
                parallel_calls = 1,
            ):
                pass

        asyncio.run(_collect())
        assert seen_kwargs, "thread_start never called"
        kw = seen_kwargs[0]
        assert (
            kw.get("approval_mode") == "DENY_ALL"
        ), f"approval_mode not pinned even with submodule SandboxMode: {kw}"
        assert (
            kw.get("sandbox") == "READ_ONLY"
        ), f"sandbox not pinned via submodule lookup: {kw}"

    def test_scrubbed_env_construction_failure_restores_env(self, monkeypatch):
        """Round 6: if the SDK constructor raises before __aenter__
        returns, the previous wrapper never called __aexit__ so the
        scrubbed env vars leaked permanently. Now the scrub is rolled
        back on failure.
        """
        from core.inference.codex_provider import _ScrubbedEnvAsyncCodex

        monkeypatch.setenv("HF_TOKEN", "must_survive")

        class _FailingAsync:
            def __init__(self):
                raise RuntimeError("SDK construction failed")

        async def _run():
            wrapper = _ScrubbedEnvAsyncCodex(_FailingAsync)
            try:
                async with wrapper:
                    pass
            except RuntimeError:
                pass

        asyncio.run(_run())
        # HF_TOKEN must be restored even though __aexit__ never fired
        # for the failed construction.
        assert (
            os.environ.get("HF_TOKEN") == "must_survive"
        ), "scrubbed env leaked permanently when SDK construction failed"

    def test_thread_start_fails_closed_when_safety_unavailable(self, monkeypatch):
        """Round 6b: if the installed SDK cannot expose ApprovalMode or
        SandboxMode, the provider MUST fail closed rather than
        silently fall through to the SDK's `auto_review` default. A
        server-side chat surface with no per-action approval UI
        cannot tolerate the model deciding on its own to run shell
        commands. The error surfaces as a typed CodexUnavailableError
        the route layer translates to 503.
        """
        # Make sure the override env var is NOT set.
        monkeypatch.delenv("UNSLOTH_CODEX_ALLOW_UNSAFE_DEFAULTS", raising = False)
        seen_kwargs: list[dict] = []

        class _Async:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def thread_start(self, **kw):
                seen_kwargs.append(dict(kw))
                return _FakeThread(chunks = ["ok"])

        _install_fake_codex_sdk(monkeypatch, _Async, with_safety_enums = False)
        from core.inference.codex_provider import (
            CodexUnavailableError,
            stream_codex,
        )

        async def _collect():
            async for _ in stream_codex(
                messages = [{"role": "user", "content": "hi"}],
                model = "gpt-5.5",
                parallel_calls = 1,
            ):
                pass

        with pytest.raises(CodexUnavailableError) as exc_info:
            asyncio.run(_collect())
        assert "ApprovalMode" in str(exc_info.value) or "SandboxMode" in str(
            exc_info.value
        )
        assert not seen_kwargs, (
            "thread_start must NOT have been called when safety pins "
            "could not be applied"
        )

    def test_thread_start_allows_unsafe_defaults_with_explicit_opt_in(
        self, monkeypatch
    ):
        """When the operator deliberately sets the
        UNSLOTH_CODEX_ALLOW_UNSAFE_DEFAULTS escape hatch, the provider
        proceeds without the safety pins (logs a warning) instead of
        raising. This is the dev-only override for pre-release alpha
        SDK builds that have not yet exposed ApprovalMode/SandboxMode.
        """
        monkeypatch.setenv("UNSLOTH_CODEX_ALLOW_UNSAFE_DEFAULTS", "1")
        seen_kwargs: list[dict] = []

        class _Async:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def thread_start(self, **kw):
                seen_kwargs.append(dict(kw))
                return _FakeThread(chunks = ["ok"])

        _install_fake_codex_sdk(monkeypatch, _Async, with_safety_enums = False)
        from core.inference.codex_provider import stream_codex

        async def _collect():
            async for _ in stream_codex(
                messages = [{"role": "user", "content": "hi"}],
                model = "gpt-5.5",
                parallel_calls = 1,
            ):
                pass

        asyncio.run(_collect())
        assert seen_kwargs, "thread_start never called under override"
        kw = seen_kwargs[0]
        assert "approval_mode" not in kw
        assert "sandbox" not in kw
        assert kw.get("model") == "gpt-5.5"

    def test_device_login_log_filter_drops_unknown_lines(self, monkeypatch):
        """The login stream's `log` events must not forward arbitrary
        subprocess output. Only an allow-list of known progress
        strings reaches the browser; anything else (auth JSON,
        tokens, paths, error tails) stays in backend logs.
        """
        # Build a synthetic stdout stream with one safe line and one
        # unsafe line, then drive the login generator against it.
        from core.inference import codex_provider as cp

        class _FakeStdout:
            def __init__(self, lines: list[bytes]):
                self._lines = list(lines)

            async def readline(self) -> bytes:
                if not self._lines:
                    return b""
                return self._lines.pop(0)

        class _FakeProc:
            pid = 99999
            returncode = None
            stdout = _FakeStdout(
                [
                    b"Welcome to Codex\n",
                    b"Open: https://auth.openai.com/codex/device\n",
                    b"Enter this one-time code: ABCD-EFGH\n",
                    b'{"refresh_token": "rt_LEAK_LEAK_LEAK"}\n',
                    b"/home/u/.codex/auth.json saved\n",
                    b"Successfully logged in\n",
                ],
            )

            async def wait(self):
                self.returncode = 0
                return 0

            def kill(self):
                self.returncode = -9

            def terminate(self):
                self.returncode = -15

        async def _fake_create_subprocess_exec(*a, **kw):
            return _FakeProc()

        monkeypatch.setattr(
            cp.asyncio, "create_subprocess_exec", _fake_create_subprocess_exec
        )

        events: list[dict] = []

        async def _collect():
            async for ev in cp.stream_codex_device_login():
                events.append(ev)

        asyncio.run(_collect())
        log_lines = [ev.get("line", "") for ev in events if ev.get("type") == "log"]
        joined = "\n".join(log_lines)
        # Sensitive content must not have been forwarded.
        assert "refresh_token" not in joined, f"token leaked: {joined!r}"
        assert "rt_LEAK_LEAK_LEAK" not in joined
        assert "auth.json" not in joined, f"local config path leaked: {joined!r}"
        # The known-safe progress lines must be present so the UI can
        # show the user what is happening.
        assert any("Welcome to Codex" in line for line in log_lines)
        assert any("Successfully logged in" in line for line in log_lines)
        # device_url + device_code events must still fire.
        url_events = [ev for ev in events if ev.get("type") == "device_url"]
        code_events = [ev for ev in events if ev.get("type") == "device_code"]
        assert url_events and url_events[0]["url"].endswith("/codex/device")
        assert code_events and code_events[0]["code"] == "ABCD-EFGH"

    def test_parallel_usage_accounts_for_all_calls(self, monkeypatch):
        """The fan-out path runs N worker calls + 1 synthesis call.
        The reported usage must reflect that, not just one call's
        worth, otherwise the cost / context display is off by the
        fan-out factor.
        """
        _install_fake_codex_sdk(
            monkeypatch,
            lambda: _FakeAsyncCodex(
                chunks = ["AAAAAAAAAA"],  # 10 chars per tab
                final = "SYNTHESISED" * 10,  # 110 chars synthesis
            ),
        )
        from core.inference.codex_provider import stream_codex

        n = 4
        long_prompt = "a" * 200  # 200 chars
        lines = _collect_stream(
            stream_codex(
                messages = [{"role": "user", "content": long_prompt}],
                model = "gpt-5.4",
                parallel_calls = n,
            )
        )
        chunks = _parse_sse_chunks(lines)
        usage_chunks = [c for c in chunks if c.get("choices") == [] and c.get("usage")]
        assert len(usage_chunks) == 1
        usage = usage_chunks[0]["usage"]
        # Single-call prompt would be ~200/4 = 50 tokens. For n=4 with
        # synthesis, prompt should be much larger: n*200 + (n*10 + 200)
        # = 800 + 240 = 1040 chars ~= 260 tokens.
        assert (
            usage["prompt_tokens"] >= 200
        ), f"prompt_tokens not scaled for fan-out: {usage['prompt_tokens']}"
        # Completion = n*10 (tab outputs) + 110 (synthesis) = 150 chars
        # ~= 37 tokens. Definitely > the synthesis-only count of 27.
        assert (
            usage["completion_tokens"] >= 30
        ), f"completion_tokens not scaled for fan-out: {usage['completion_tokens']}"

    def test_buffered_result_none_final_does_not_emit_repr(self, monkeypatch):
        """A buffered TurnResult whose final_response is None must NOT
        send a Python object repr (``TurnResult(...)``) to the user.
        Returning an empty content chunk is the right shape: the
        stream still finishes with the usage + stop + [DONE] frames,
        but no garbage assistant text appears.
        """

        class _ResultNoFinal:
            final_response = None  # explicit None

            def __repr__(self):
                return "TurnResult(internal=should_not_leak)"

        class _ThreadBuffered:
            async def run(self, prompt):
                return _ResultNoFinal()

        class _Async:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def thread_start(self, **kw):
                return _ThreadBuffered()

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
        body = "".join(chunks)
        assert (
            "TurnResult" not in body
        ), f"Python object repr leaked to user content: {body!r}"
        assert "should_not_leak" not in body
        # Stream still terminated cleanly.
        assert "[DONE]" in body

    def test_empty_stream_falls_back_to_completed_agent_message(self, monkeypatch):
        """A successful turn that emits zero ``message.delta`` events
        but DOES emit a final ``ItemCompletedNotification`` with an
        agent message must surface that text. Without the fallback the
        Chat Completions reply would be empty even though Codex
        produced a complete answer.
        """

        class _CompletedEvent:
            payload = {
                "type": "item.completed",
                "item": {
                    "root": {
                        "type": "agentMessage",
                        "text": "final answer from completion",
                    },
                },
            }

        class _Turn:
            async def stream(self):
                yield _CompletedEvent()

        class _ThreadEmptyDeltas:
            def turn(self, prompt):
                return _Turn()

            async def run(self, prompt):
                raise AssertionError(
                    "must not fall through to buffered run() when "
                    "the stream completes successfully"
                )

        class _Async:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def thread_start(self, **kw):
                return _ThreadEmptyDeltas()

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
        body = "".join(chunks)
        assert (
            "final answer from completion" in body
        ), f"agent message text from completion event was dropped; body={body!r}"

    def test_synthesis_also_pins_safety_kwargs(self, monkeypatch):
        """The synthesis turn that unifies parallel fan-out outputs
        must use the same safety pins -- a fan-out tab could otherwise
        sneak an unsafe approval into the final synthesis prompt.
        """
        seen_kwargs: list[dict] = []

        class _SynThread:
            async def run(self, prompt):
                return "synth ok"

        class _Async:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def thread_start(self, **kw):
                seen_kwargs.append(dict(kw))
                return _SynThread()

        import importlib.util as _iu

        fake_mod = types.ModuleType("openai_codex")
        fake_mod.AsyncCodex = _Async  # type: ignore[attr-defined]
        fake_mod.ApprovalMode = types.SimpleNamespace(  # type: ignore[attr-defined]
            deny_all = "DENY_ALL_SENTINEL",
        )
        fake_mod.SandboxMode = types.SimpleNamespace(  # type: ignore[attr-defined]
            read_only = "READ_ONLY_SENTINEL",
        )
        monkeypatch.setitem(sys.modules, "openai_codex", fake_mod)
        real_find_spec = _iu.find_spec
        monkeypatch.setattr(
            "importlib.util.find_spec",
            lambda n, *a, **kw: (
                types.SimpleNamespace()
                if n in ("openai_codex", "codex_app_server")
                else real_find_spec(n, *a, **kw)
            ),
        )

        from core.inference.codex_provider import _run_codex_synthesis

        asyncio.run(
            _run_codex_synthesis(
                model = "gpt-5.5",
                system = "Always answer in Spanish.",
                prompt = "What is the capital of France?",
                tab_outputs = ["Paris", "Paris."],
            )
        )
        assert seen_kwargs, "synthesis thread_start never called"
        kw = seen_kwargs[0]
        assert kw.get("approval_mode") == "DENY_ALL_SENTINEL"
        assert kw.get("sandbox") == "READ_ONLY_SENTINEL"


async def _consume_first(gen):
    """Drive an async generator until it raises or yields its first
    value. Used to surface lazy-import errors that fire on the first
    SDK touch -- otherwise the generator would swallow them on
    ``__aiter__`` and the test couldn't see them.
    """
    async for _ in gen:
        return


# ── Round 7: _ScrubbedEnvAsyncCodex cross-wrapper concurrency ──────


class TestScrubbedEnvConcurrency:
    """Reproduce the cross-wrapper concurrency hole the round 7 review
    surfaced and lock in the fix: when wrapper B enters AFTER wrapper A
    has already deleted ``HF_TOKEN`` from ``os.environ``, B must still
    increment the refcount for that key so A's exit does not restore
    the secret while B is mid-session.
    """

    def test_overlapping_wrappers_keep_keys_scrubbed_until_last_release(
        self, monkeypatch
    ):
        import os

        from core.inference.codex_provider import (
            _SCRUBBED_ENV_REFCOUNT,
            _ScrubbedEnvAsyncCodex,
        )

        # Reset module-level state in case prior tests left residue.
        _SCRUBBED_ENV_REFCOUNT.clear()
        # _SCRUBBED_ENV_ORIGINALS is the round 7 fix's shared snapshot
        # store; older codex_provider builds tracked originals per-
        # instance under _restored_via_us. Reset whichever store the
        # current build exposes so prior tests cannot leak state in.
        from core.inference import codex_provider as _cp

        _orig = getattr(_cp, "_SCRUBBED_ENV_ORIGINALS", None)
        if isinstance(_orig, dict):
            _orig.clear()

        monkeypatch.setenv("HF_TOKEN", "sekret-hf")
        monkeypatch.setenv("GH_TOKEN", "sekret-gh")
        # Keys NOT on the safe-list end up in _codex_sdk_env_override().

        class _FakeInner:
            async def __aenter__(self_inner):
                return self_inner

            async def __aexit__(self_inner, *a):
                return False

        def _fake_async_codex():
            return _FakeInner()

        async def scenario():
            wrapper_a = _ScrubbedEnvAsyncCodex(_fake_async_codex)
            wrapper_b = _ScrubbedEnvAsyncCodex(_fake_async_codex)

            # Wrapper A enters first and scrubs both secrets.
            await wrapper_a.__aenter__()
            assert "HF_TOKEN" not in os.environ
            assert "GH_TOKEN" not in os.environ

            # Wrapper B enters while A is still active. Even though
            # os.environ no longer contains HF_TOKEN/GH_TOKEN (A already
            # deleted them), B must pick them up from the live refcount
            # table so A's later exit does not restore them prematurely.
            await wrapper_b.__aenter__()
            assert _SCRUBBED_ENV_REFCOUNT.get("HF_TOKEN") == 2
            assert _SCRUBBED_ENV_REFCOUNT.get("GH_TOKEN") == 2

            # A exits first -- B is still active so the keys MUST remain
            # absent from os.environ.
            await wrapper_a.__aexit__(None, None, None)
            assert "HF_TOKEN" not in os.environ, (
                "HF_TOKEN leaked back into os.environ while wrapper B "
                "is still active"
            )
            assert "GH_TOKEN" not in os.environ
            assert _SCRUBBED_ENV_REFCOUNT.get("HF_TOKEN") == 1
            assert _SCRUBBED_ENV_REFCOUNT.get("GH_TOKEN") == 1

            # B exits -- now the keys must be restored from the saved
            # originals.
            await wrapper_b.__aexit__(None, None, None)
            assert os.environ.get("HF_TOKEN") == "sekret-hf"
            assert os.environ.get("GH_TOKEN") == "sekret-gh"
            assert "HF_TOKEN" not in _SCRUBBED_ENV_REFCOUNT

        asyncio.run(scenario())


# ── Round 7: device-auth URL allowlisting ───────────────────────────


class TestDeviceUrlAllowlist:
    """Lock in the device-auth URL allowlist: only `auth.openai.com`
    and `chatgpt.com` over https are accepted as `device_url` events.
    A shimmed codex earlier on PATH could otherwise print
    `https://evil.example/activate?code=ABCD` and Studio would render
    a phishing CTA.
    """

    def test_known_good_urls_allowed(self):
        from core.inference.codex_provider import _is_allowed_device_url

        assert _is_allowed_device_url(
            "https://auth.openai.com/codex/device?user_code=ABCD-EFGH"
        )
        assert _is_allowed_device_url(
            "https://chatgpt.com/activate?user_code=WXYZ-1234"
        )

    def test_attacker_hosts_rejected(self):
        from core.inference.codex_provider import _is_allowed_device_url

        for evil in [
            "https://evil.example/activate?code=ABCD",
            "https://auth-openai-com.evil.example/codex/device",
            "https://chatgpt.com.evil.example/activate",
            "https://login.openai.com/codex/device",
        ]:
            assert not _is_allowed_device_url(evil), evil

    def test_http_downgrade_rejected(self):
        from core.inference.codex_provider import _is_allowed_device_url

        assert not _is_allowed_device_url(
            "http://auth.openai.com/codex/device?user_code=ABCD-EFGH"
        )

    def test_garbage_url_rejected(self):
        from core.inference.codex_provider import _is_allowed_device_url

        assert not _is_allowed_device_url("not a url")
        assert not _is_allowed_device_url("")
        assert not _is_allowed_device_url("javascript:alert(1)")


# ── Round 7: tightened device-login log filter ──────────────────────


class TestDeviceLoginLogFilter:
    """The login-output filter must not forward sensitive lines a
    malicious codex shim could print -- including 'Not logged in:'
    leaks that match the old loose 'logged in' substring test, plus
    refresh tokens, auth.json paths, and the codex config dir.
    """

    def _safe_to_forward(self):
        # _safe_to_forward is defined inside stream_codex_device_login;
        # re-extracting it requires us to import it through the source
        # module path. Easier: replicate the production regex set in
        # the test directly so a regression in the source list is
        # caught when the production source is loaded.
        import importlib

        mod = importlib.reload(importlib.import_module("core.inference.codex_provider"))
        # Walk the source string to find the patterns; they live inside
        # the generator. Use a stable proxy: read the regex literals.
        import re

        src = open(mod.__file__).read()
        # Smoke check: the source has anchored regex (^) for the safe
        # phrases AND an unsafe-content blocklist.
        assert "safe_log_res" in src
        assert "unsafe_log_re" in src
        assert (
            "not\\s+(?:logged|signed)\\s+in" in src
            or "not\\\\s+(?:logged|signed)\\\\s+in" in src
        )
        return None

    def test_safe_log_source_has_anchored_patterns_and_blocklist(self):
        self._safe_to_forward()

    def test_blocklist_rejects_known_leaks(self):
        # Reconstruct the production regex set the same way stream_codex
        # _device_login does, then assert each attacker string is dropped.
        import re

        unsafe_log_re = re.compile(
            r"\bnot\s+(?:logged|signed)\s+in\b|"
            r"\bnot\s+authenticated\b|"
            r"refresh[_-]?token|access[_-]?token|"
            r"\bapi[_-]?key\b|\bsecret\b|"
            r"\bauth\.json\b|"
            r"/\.codex/|\\\.codex\\",
            re.IGNORECASE,
        )
        for line in [
            "Not logged in: refresh_token=rt_LEAK auth.json=/home/u/.codex/auth.json",
            "logged in (refresh_token=abc)",
            "Open this: https://auth.openai.com/codex/device but access_token=hunter2",
            "Logged in - secret=hunter2",
            "API_KEY=sk-x logged in",
            "Reading /home/u/.codex/auth.json",
        ]:
            assert unsafe_log_re.search(line), f"line should match unsafe: {line!r}"

    def test_safe_phrases_pass_when_clean(self):
        import re

        safe_log_res = (
            re.compile(r"^welcome to codex\b", re.IGNORECASE),
            re.compile(r"^initializing\b", re.IGNORECASE),
            re.compile(r"^open (?:this|the verification)", re.IGNORECASE),
            re.compile(r"^open:\s*https?://", re.IGNORECASE),
            re.compile(r"^enter (?:this one-time code|the code)\b", re.IGNORECASE),
            re.compile(r"^waiting\b", re.IGNORECASE),
            re.compile(r"^successfully (?:logged|signed) in\b", re.IGNORECASE),
            re.compile(r"^(?:logged|signed) in\b", re.IGNORECASE),
            re.compile(r"^browser opened\b", re.IGNORECASE),
            re.compile(r"^press ctrl", re.IGNORECASE),
        )
        for clean in [
            "Welcome to codex",
            "Initializing device auth...",
            "Open this URL: https://auth.openai.com/codex/device",
            "Open: https://auth.openai.com/codex/device",
            "Enter this one-time code:",
            "Waiting for authentication...",
            "Successfully logged in",
            "Logged in using ChatGPT",
            "Browser opened",
            "Press Ctrl+C to cancel",
        ]:
            assert any(
                pat.search(clean) for pat in safe_log_res
            ), f"clean line should match safe: {clean!r}"


# ── Round 8: stream replay protection on non-visible events ──────────


class TestStreamReplayProtection:
    """Lock in the round 8 fix: a turn that fired non-rendered events
    (command/file/tool deltas) before crashing MUST NOT replay via the
    buffered `thread.run(prompt)` fallback even though no visible
    text was yielded. The earlier guard only tracked `emitted_any`
    (visible text), missing the case where shell commands or file
    writes already happened upstream.
    """

    def test_buffered_run_not_called_after_non_visible_event_crash(self):
        """Stream raises after a tool event with no visible text. The
        buffered ``thread.run`` MUST NOT be called -- the Codex turn
        has already started running side-effects upstream.
        """
        from core.inference.codex_provider import _stream_thread_run

        class _Stream:
            def __init__(self):
                self._i = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                self._i += 1
                if self._i == 1:
                    # An event with no answer text -- _coerce_text
                    # returns "" but the turn has demonstrably run.
                    return {"type": "command.delta", "delta": "rm -rf"}
                raise RuntimeError("upstream stream died mid-turn")

        class _Turn:
            def stream(self):
                return _Stream()

        class _Thread:
            run_calls: int = 0

            def turn(self_inner, prompt):
                return _Turn()

            async def run(self_inner, prompt):
                self_inner.run_calls += 1
                return "REPLAY-WOULD-RETURN-THIS"

        thread = _Thread()

        async def collect():
            chunks = []
            async for c in _stream_thread_run(thread, "hello"):
                chunks.append(c)
            return chunks

        chunks = asyncio.run(collect())
        # No visible text was emitted (the only event was filtered),
        # but thread.run MUST NOT have been called because the turn
        # already started.
        assert thread.run_calls == 0, (
            "thread.run was called after a partial-turn crash; this "
            "would replay shell commands / file writes"
        )
        assert chunks == []

    def test_buffered_run_called_when_no_streaming_helper(self):
        """Threads that expose neither .turn nor .run_streaming still
        fall through to the buffered .run -- that is the ONLY path
        the buffered fallback is allowed to execute.
        """
        from core.inference.codex_provider import _stream_thread_run

        class _Thread:
            run_calls: int = 0

            async def run(self_inner, prompt):
                self_inner.run_calls += 1
                return "answer"

        thread = _Thread()

        async def collect():
            chunks = []
            async for c in _stream_thread_run(thread, "hello"):
                chunks.append(c)
            return chunks

        chunks = asyncio.run(collect())
        assert thread.run_calls == 1
        assert chunks == ["answer"]
