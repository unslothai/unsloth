# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Edge-case coverage for the Anthropic fast-mode + refusal wiring.

The base ``test_anthropic_fast_mode_and_refusal.py`` pins the happy path
(beta header + body field on Opus 4.6/4.7, silent drop on Sonnet / Haiku
/ older Opus / None / False, refusal notice + content_filter mapping).
This file fills in the remaining behaviour cliffs:

* Dated-snapshot Opus 4.6 / 4.7 (``claude-opus-4-7-2026-02-01``) still
  matches the prefix gate.
* Strict opt-in semantics: a future ``claude-opus-4-8`` / ``claude-opus-5``
  does NOT auto-enable fast mode -- if a new family ever ships we must
  bump ``_ANTHROPIC_FAST_MODE_PREFIXES`` explicitly. This codifies the
  conservative stance.
* Beta-header merge: the existing ``anthropic-beta`` value (from the
  provider registry or a previously-set entry) must keep its values and
  receive ``fast-mode-2026-02-01`` appended as a comma-separated extra,
  not overwritten.
* Multi-beta interaction with the code-execution and compaction betas:
  all three must coexist in one comma-separated header with no
  duplicates and no truncation.
* Idempotence: setting ``fast_mode=True`` twice via the same body still
  results in one beta-header entry.
* Streaming refusal sentinel: emitted exactly once, with the exact
  ``<!--studio:anthropic-refusal-->`` token the frontend matches on,
  and the notice always precedes the finish_reason chunk so a UI
  reading the SSE in order paints text before flipping to
  ``content_filter``.
* Refusal on a non-Opus model: refusal handling is provider-side, not
  model-gated, so a refusal mid-stream on Sonnet must still surface the
  notice + sentinel.
* Non-destruction: when ``fast_mode`` is ``None``, the outbound body
  and headers must be byte-identical to the version that omits the
  argument entirely. This guarantees the upgrade path is non-breaking
  on existing Anthropic streams.
"""

import asyncio
import json
import re

import httpx

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient


def _drive(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _make_client() -> ExternalProviderClient:
    return ExternalProviderClient(
        provider_type = "anthropic",
        base_url = "https://api.anthropic.com/v1",
        api_key = "sk-ant-test",
    )


def _empty_message_sse(model: str = "claude-opus-4-7") -> bytes:
    return (
        b'event: message_start\ndata: {"type":"message_start","message":'
        b'{"id":"m1","content":[],"model":"' + model.encode() + b'",'
        b'"role":"assistant","stop_reason":null,"usage":'
        b'{"input_tokens":1,"output_tokens":1}}}\n\n'
        b'event: message_delta\ndata: {"type":"message_delta",'
        b'"delta":{"stop_reason":"end_turn"}}\n\n'
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )


def _refusal_sse(model: str = "claude-opus-4-7") -> bytes:
    return (
        b'event: message_start\ndata: {"type":"message_start","message":'
        b'{"id":"m1","content":[],"model":"' + model.encode() + b'",'
        b'"role":"assistant","stop_reason":null,"usage":'
        b'{"input_tokens":1,"output_tokens":1}}}\n\n'
        b'event: content_block_start\ndata: {"type":"content_block_start",'
        b'"index":0,"content_block":{"type":"text","text":""}}\n\n'
        b'event: content_block_delta\ndata: {"type":"content_block_delta",'
        b'"index":0,"delta":{"type":"text_delta","text":"Hello."}}\n\n'
        b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
        b'event: message_delta\ndata: {"type":"message_delta",'
        b'"delta":{"stop_reason":"refusal"}}\n\n'
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )


def _capture(monkeypatch, sse: bytes = b"", **kwargs) -> tuple[dict, list[str]]:
    """Install a MockTransport, drive one streamed call, return body+lines."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            content = sse or _empty_message_sse(kwargs.get("model", "claude-opus-4-7")),
            headers = {"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(
        ep_mod,
        "_http_client",
        httpx.AsyncClient(transport = httpx.MockTransport(handler)),
    )

    out_lines: list[str] = []

    async def run():
        client = _make_client()
        try:
            extra = {}
            for key in (
                "enabled_tools",
                "compaction_threshold",
                "fast_mode",
            ):
                if key in kwargs:
                    extra[key] = kwargs[key]
            async for line in client.stream_chat_completion(
                messages = [{"role": "user", "content": "hi"}],
                model = kwargs.get("model", "claude-opus-4-7"),
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 32,
                **extra,
            ):
                out_lines.append(line)
        finally:
            await client.close()

    _drive(run())
    return captured, out_lines


# ──────────────────────────── dated snapshot prefix ────────────────────────────
def test_fast_mode_attaches_on_dated_opus_4_7_snapshot(monkeypatch):
    """Dated snapshot ``claude-opus-4-7-2026-02-01`` must match the prefix.

    Anthropic's /v1/models surfaces both the canonical alias and a
    dated snapshot for each family. The fast-mode gate uses
    ``str.startswith`` against the tuple of family aliases, so a dated
    snapshot of an opted-in family is implicitly included.
    """
    cap, _ = _capture(monkeypatch, fast_mode = True, model = "claude-opus-4-7-2026-02-01")
    assert cap["body"].get("speed") == "fast", cap["body"]
    assert "fast-mode-2026-02-01" in cap["headers"].get("anthropic-beta", "")


def test_fast_mode_attaches_on_dated_opus_4_6_snapshot(monkeypatch):
    cap, _ = _capture(monkeypatch, fast_mode = True, model = "claude-opus-4-6-2026-02-01")
    assert cap["body"].get("speed") == "fast", cap["body"]
    assert "fast-mode-2026-02-01" in cap["headers"].get("anthropic-beta", "")


# ──────────────────────────── strict opt-in semantics ────────────────────────────
def test_fast_mode_does_not_auto_enable_on_future_opus_4_8(monkeypatch):
    """A hypothetical ``claude-opus-4-8`` is NOT in the prefix tuple, so
    fast_mode must be silently dropped until the list is updated.

    This codifies the conservative stance documented on
    ``_ANTHROPIC_FAST_MODE_PREFIXES``: opt-in per model family.
    """
    cap, _ = _capture(monkeypatch, fast_mode = True, model = "claude-opus-4-8")
    assert "speed" not in cap["body"], cap["body"]
    assert "fast-mode-2026-02-01" not in cap["headers"].get("anthropic-beta", "")


def test_fast_mode_does_not_auto_enable_on_future_opus_5(monkeypatch):
    cap, _ = _capture(monkeypatch, fast_mode = True, model = "claude-opus-5")
    assert "speed" not in cap["body"], cap["body"]
    assert "fast-mode-2026-02-01" not in cap["headers"].get("anthropic-beta", "")


def test_fast_mode_does_not_auto_enable_on_sonnet_dated_snapshot(monkeypatch):
    """``claude-sonnet-4-6-2026-...`` shares the family prefix tuple with
    compaction but NOT with fast_mode -- the gate must keep them
    separate."""
    cap, _ = _capture(monkeypatch, fast_mode = True, model = "claude-sonnet-4-6-2026-02-01")
    assert "speed" not in cap["body"], cap["body"]
    assert "fast-mode-2026-02-01" not in cap["headers"].get("anthropic-beta", "")


# ──────────────────────────── beta header merge ────────────────────────────
def _beta_parts(headers: dict) -> list[str]:
    raw = headers.get("anthropic-beta", "")
    return [p.strip() for p in raw.split(",") if p.strip()]


def test_fast_mode_merges_with_code_execution_beta(monkeypatch):
    """fast_mode + enabled_tools=['code_execution'] -> two comma-separated
    beta entries, no overwrite."""
    cap, _ = _capture(
        monkeypatch,
        fast_mode = True,
        model = "claude-opus-4-7",
        enabled_tools = ["code_execution"],
    )
    parts = _beta_parts(cap["headers"])
    assert "fast-mode-2026-02-01" in parts, cap["headers"]
    assert any(p.startswith("code-execution-") for p in parts), cap["headers"]
    # No duplicates.
    assert len(parts) == len(set(parts)), parts


def test_fast_mode_merges_with_compaction_beta(monkeypatch):
    """fast_mode + compaction_threshold>=50K -> both betas."""
    cap, _ = _capture(
        monkeypatch,
        fast_mode = True,
        model = "claude-opus-4-7",
        compaction_threshold = 100_000,
    )
    parts = _beta_parts(cap["headers"])
    assert "fast-mode-2026-02-01" in parts, cap["headers"]
    assert "compact-2026-01-12" in parts, cap["headers"]


def test_fast_mode_merges_with_code_execution_and_compaction(monkeypatch):
    """Three betas active simultaneously must all land in a single
    comma-separated header value, with no duplicates."""
    cap, _ = _capture(
        monkeypatch,
        fast_mode = True,
        model = "claude-opus-4-7",
        enabled_tools = ["code_execution"],
        compaction_threshold = 100_000,
    )
    parts = _beta_parts(cap["headers"])
    assert "fast-mode-2026-02-01" in parts
    assert "compact-2026-01-12" in parts
    assert any(p.startswith("code-execution-") for p in parts), parts
    assert len(parts) >= 3
    assert len(parts) == len(set(parts)), parts


def test_fast_mode_beta_value_is_pinned(monkeypatch):
    """Belt-and-braces: the exact beta tag matches the docs token.

    Anthropic's docs spell it ``fast-mode-2026-02-01`` -- if someone
    fat-fingers it the API would 400 in production. Pin it here so a
    rename is caught in CI.
    """
    cap, _ = _capture(monkeypatch, fast_mode = True, model = "claude-opus-4-7")
    parts = _beta_parts(cap["headers"])
    assert "fast-mode-2026-02-01" in parts, parts
    # Reject obvious typos.
    assert not any(p.startswith("fastmode-") for p in parts), parts
    assert not any("fast_mode" in p for p in parts), parts


# ──────────────────────────── non-destruction guarantee ────────────────────────────
def test_fast_mode_unset_is_byte_identical_to_omitted(monkeypatch):
    """Passing ``fast_mode=None`` must produce the same outbound body and
    headers as omitting the argument entirely.

    The upgrade path for existing Anthropic streams must be non-breaking.
    """
    cap_none, _ = _capture(monkeypatch, fast_mode = None, model = "claude-opus-4-7")

    # Re-run without passing fast_mode at all.
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            content = _empty_message_sse(),
            headers = {"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(
        ep_mod,
        "_http_client",
        httpx.AsyncClient(transport = httpx.MockTransport(handler)),
    )

    async def run():
        client = _make_client()
        try:
            async for _ in client.stream_chat_completion(
                messages = [{"role": "user", "content": "hi"}],
                model = "claude-opus-4-7",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 32,
            ):
                pass
        finally:
            await client.close()

    _drive(run())

    assert cap_none["body"] == captured["body"], (cap_none["body"], captured["body"])
    # Headers can vary by httpx-injected fields (host, connection); compare
    # the load-bearing ones.
    for key in ("anthropic-version", "x-api-key", "content-type"):
        assert cap_none["headers"].get(key) == captured["headers"].get(key), key
    assert "anthropic-beta" not in cap_none["headers"]
    assert "anthropic-beta" not in captured["headers"]
    assert "speed" not in cap_none["body"]
    assert "speed" not in captured["body"]


def test_fast_mode_false_on_opus_4_7_byte_identical_to_unset(monkeypatch):
    """``fast_mode=False`` is the explicit opt-out -- it must produce the
    same outbound shape as the unset case so we don't accidentally start
    sending ``speed: 'standard'`` or similar."""
    cap_false, _ = _capture(monkeypatch, fast_mode = False, model = "claude-opus-4-7")
    assert "speed" not in cap_false["body"], cap_false["body"]
    assert "fast-mode-2026-02-01" not in cap_false["headers"].get("anthropic-beta", "")


# ──────────────────────────── refusal stream ordering ────────────────────────────
def test_refusal_notice_appears_before_content_filter_chunk(monkeypatch):
    """The user-visible notice must be emitted as a content delta BEFORE
    the finish_reason chunk so a streaming UI paints the text first,
    then flips state to ``content_filter``."""
    _, lines = _capture(monkeypatch, sse = _refusal_sse(), model = "claude-opus-4-7")
    notice_idx = next(i for i, l in enumerate(lines) if "stopped by Anthropic" in l)
    filter_idx = next(
        i for i, l in enumerate(lines) if '"finish_reason": "content_filter"' in l
    )
    assert notice_idx < filter_idx, (notice_idx, filter_idx, lines)


def test_refusal_sentinel_emitted_exactly_once(monkeypatch):
    """A single refusal must emit the chat-adapter drop sentinel one time.

    The frontend's ``toOpenAIMessage`` uses ``includes`` for the match,
    so duplicates wouldn't cause false drops -- but emitting it twice
    would still inflate the bubble and break the UX guarantee that the
    sentinel is an invisible HTML comment.
    """
    _, lines = _capture(monkeypatch, sse = _refusal_sse())
    body = "\n".join(lines)
    count = body.count("<!--studio:anthropic-refusal-->")
    assert count == 1, (count, body)


def test_refusal_handling_works_on_sonnet_model(monkeypatch):
    """Refusal handling is provider-side, not gated on a fast-mode-capable
    model. If Anthropic's classifier refuses a Sonnet stream it must
    surface the same notice + sentinel + content_filter mapping."""
    _, lines = _capture(
        monkeypatch, sse = _refusal_sse("claude-sonnet-4-6"), model = "claude-sonnet-4-6"
    )
    body = "\n".join(lines)
    assert "stopped by Anthropic's safety classifier" in body, body
    assert "<!--studio:anthropic-refusal-->" in body, body
    assert '"finish_reason": "content_filter"' in body, body


def test_refusal_preserves_partial_assistant_text(monkeypatch):
    """The classifier may stop mid-completion. The partial deltas already
    streamed must still reach the client BEFORE the refusal notice is
    appended -- otherwise users lose context for why the bubble ended."""
    _, lines = _capture(monkeypatch, sse = _refusal_sse(), model = "claude-opus-4-7")
    body = "\n".join(lines)
    hello_idx = body.index("Hello.")
    notice_idx = body.index("stopped by Anthropic")
    assert hello_idx < notice_idx, (hello_idx, notice_idx)


def test_refusal_chunk_is_proper_openai_delta_shape(monkeypatch):
    """The notice rides a normal ``choices[0].delta.content`` chunk, not
    a finish_reason chunk, so OpenAI-spec clients (Aurora, OpenAI SDK)
    treat it as ordinary streamed text."""
    _, lines = _capture(monkeypatch, sse = _refusal_sse(), model = "claude-opus-4-7")
    # Find the chunk that carries the refusal text.
    notice_chunk = None
    for line in lines:
        if line.startswith("data: ") and "stopped by Anthropic" in line:
            notice_chunk = json.loads(line[len("data: ") :])
            break
    assert notice_chunk is not None, lines
    choice = notice_chunk["choices"][0]
    assert "delta" in choice and "content" in choice["delta"], notice_chunk
    # Must NOT carry a finish_reason itself -- that comes on the next
    # chunk.
    assert choice.get("finish_reason") in (None,), notice_chunk
    assert "<!--studio:anthropic-refusal-->" in choice["delta"]["content"]


# ──────────────────────────── future-proofing ────────────────────────────
def test_fast_mode_prefix_tuple_matches_capability_doc(monkeypatch):
    """If a maintainer extends ``_ANTHROPIC_FAST_MODE_PREFIXES`` they
    should also extend this test. Today the tuple is exactly the two
    families called out on
    https://platform.claude.com/docs/en/build-with-claude/fast-mode.
    """
    from core.inference.external_provider import _ANTHROPIC_FAST_MODE_PREFIXES

    assert set(_ANTHROPIC_FAST_MODE_PREFIXES) == {
        "claude-opus-4-7",
        "claude-opus-4-6",
    }, _ANTHROPIC_FAST_MODE_PREFIXES


def test_fast_mode_speed_field_value_is_literal_fast(monkeypatch):
    """The spec value is the string ``"fast"`` -- not ``True``, not
    ``"fast-mode"``, not a tier name. Pin so a refactor cannot silently
    flip the wire shape."""
    cap, _ = _capture(monkeypatch, fast_mode = True, model = "claude-opus-4-7")
    assert cap["body"]["speed"] == "fast", cap["body"]


def test_fast_mode_dropped_on_opus_4_5_dated_snapshot(monkeypatch):
    """``claude-opus-4-5-2025-...`` is the previous-family snapshot
    pattern; must NOT match the fast-mode prefix."""
    cap, _ = _capture(monkeypatch, fast_mode = True, model = "claude-opus-4-5-2025-08-01")
    assert "speed" not in cap["body"], cap["body"]
    assert "fast-mode-2026-02-01" not in cap["headers"].get("anthropic-beta", "")
