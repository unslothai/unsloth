# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Edge-case coverage for Anthropic fast-mode + refusal wiring.

Complements ``test_anthropic_fast_mode_and_refusal.py`` (happy path) with
dated snapshots, strict opt-in (future Opus families do not auto-enable),
multi-beta header merging, refusal stream ordering, and the
non-destruction guarantee for unset/None fast_mode.
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


def _capture(
    monkeypatch,
    sse: bytes = b"",
    **kwargs,
) -> tuple[dict, list[str]]:
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
    """Dated snapshot ``claude-opus-4-7-2026-02-01`` must match the prefix."""
    cap, _ = _capture(monkeypatch, fast_mode = True, model = "claude-opus-4-7-2026-02-01")
    assert cap["body"].get("speed") == "fast", cap["body"]
    assert "fast-mode-2026-02-01" in cap["headers"].get("anthropic-beta", "")


def test_fast_mode_attaches_on_dated_opus_4_6_snapshot(monkeypatch):
    cap, _ = _capture(monkeypatch, fast_mode = True, model = "claude-opus-4-6-2026-02-01")
    assert cap["body"].get("speed") == "fast", cap["body"]
    assert "fast-mode-2026-02-01" in cap["headers"].get("anthropic-beta", "")


# ──────────────────────────── strict opt-in semantics ────────────────────────────
def test_fast_mode_does_not_auto_enable_on_future_opus_4_8(monkeypatch):
    """Future ``claude-opus-4-8`` must not auto-enable; per-family opt-in."""
    cap, _ = _capture(monkeypatch, fast_mode = True, model = "claude-opus-4-8")
    assert "speed" not in cap["body"], cap["body"]
    assert "fast-mode-2026-02-01" not in cap["headers"].get("anthropic-beta", "")


def test_fast_mode_does_not_auto_enable_on_future_opus_5(monkeypatch):
    cap, _ = _capture(monkeypatch, fast_mode = True, model = "claude-opus-5")
    assert "speed" not in cap["body"], cap["body"]
    assert "fast-mode-2026-02-01" not in cap["headers"].get("anthropic-beta", "")


def test_fast_mode_does_not_auto_enable_on_sonnet_dated_snapshot(monkeypatch):
    """Sonnet snapshots share the compaction prefix but not fast_mode."""
    cap, _ = _capture(monkeypatch, fast_mode = True, model = "claude-sonnet-4-6-2026-02-01")
    assert "speed" not in cap["body"], cap["body"]
    assert "fast-mode-2026-02-01" not in cap["headers"].get("anthropic-beta", "")


# ──────────────────────────── beta header merge ────────────────────────────
def _beta_parts(headers: dict) -> list[str]:
    raw = headers.get("anthropic-beta", "")
    return [p.strip() for p in raw.split(",") if p.strip()]


def test_fast_mode_merges_with_code_execution_beta(monkeypatch):
    """fast_mode + code_execution -> two comma-separated betas, no overwrite."""
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
    """fast_mode + compaction_threshold >= 50K -> both betas present."""
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
    """Three betas coexist in one comma-separated header, no duplicates."""
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
    """Pin the exact beta tag ``fast-mode-2026-02-01`` from the docs."""
    cap, _ = _capture(monkeypatch, fast_mode = True, model = "claude-opus-4-7")
    parts = _beta_parts(cap["headers"])
    assert "fast-mode-2026-02-01" in parts, parts
    # Reject obvious typos.
    assert not any(p.startswith("fastmode-") for p in parts), parts
    assert not any("fast_mode" in p for p in parts), parts


# ──────────────────────────── non-destruction guarantee ────────────────────────────
def test_fast_mode_unset_is_byte_identical_to_omitted(monkeypatch):
    """``fast_mode=None`` must produce the same body/headers as omission."""
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
    # Headers vary by httpx-injected fields (host, connection); compare
    # the load-bearing ones.
    for key in ("anthropic-version", "x-api-key", "content-type"):
        assert cap_none["headers"].get(key) == captured["headers"].get(key), key
    assert "anthropic-beta" not in cap_none["headers"]
    assert "anthropic-beta" not in captured["headers"]
    assert "speed" not in cap_none["body"]
    assert "speed" not in captured["body"]


def test_fast_mode_false_on_opus_4_7_byte_identical_to_unset(monkeypatch):
    """``fast_mode=False`` produces the same outbound shape as unset."""
    cap_false, _ = _capture(monkeypatch, fast_mode = False, model = "claude-opus-4-7")
    assert "speed" not in cap_false["body"], cap_false["body"]
    assert "fast-mode-2026-02-01" not in cap_false["headers"].get("anthropic-beta", "")


# ──────────────────────────── refusal stream ordering ────────────────────────────
def test_refusal_notice_appears_before_content_filter_chunk(monkeypatch):
    """The notice content delta must precede the finish_reason chunk."""
    _, lines = _capture(monkeypatch, sse = _refusal_sse(), model = "claude-opus-4-7")
    notice_idx = next(i for i, l in enumerate(lines) if "stopped by Anthropic" in l)
    filter_idx = next(i for i, l in enumerate(lines) if '"finish_reason": "content_filter"' in l)
    assert notice_idx < filter_idx, (notice_idx, filter_idx, lines)


def test_refusal_tool_event_emitted_exactly_once(monkeypatch):
    """A single refusal emits the chat-adapter drop signal exactly once."""
    _, lines = _capture(monkeypatch, sse = _refusal_sse())
    body = "\n".join(lines)
    count = body.count('"_toolEvent": {"type": "anthropic_refusal"}')
    assert count == 1, (count, body)


def test_refusal_text_carries_no_html_sentinel(monkeypatch):
    """Visible refusal text must not embed a ``studio:anthropic-refusal``
    sentinel; the drop signal rides _toolEvent only."""
    _, lines = _capture(monkeypatch, sse = _refusal_sse())
    body = "\n".join(lines)
    assert "studio:anthropic-refusal" not in body, body


def test_refusal_handling_works_on_sonnet_model(monkeypatch):
    """Refusal handling is provider-side; Sonnet refusals must also surface."""
    _, lines = _capture(
        monkeypatch, sse = _refusal_sse("claude-sonnet-4-6"), model = "claude-sonnet-4-6"
    )
    body = "\n".join(lines)
    assert "stopped by Anthropic's safety classifier" in body, body
    assert '"_toolEvent": {"type": "anthropic_refusal"}' in body, body
    assert '"finish_reason": "content_filter"' in body, body


def test_refusal_preserves_partial_assistant_text(monkeypatch):
    """Partial deltas already streamed must precede the refusal notice."""
    _, lines = _capture(monkeypatch, sse = _refusal_sse(), model = "claude-opus-4-7")
    body = "\n".join(lines)
    hello_idx = body.index("Hello.")
    notice_idx = body.index("stopped by Anthropic")
    assert hello_idx < notice_idx, (hello_idx, notice_idx)


def test_refusal_chunk_is_proper_openai_delta_shape(monkeypatch):
    """The notice rides ``choices[0].delta.content`` (not a finish chunk);
    OpenAI-spec clients treat it as ordinary streamed text."""
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
    # Must NOT carry a finish_reason itself -- that comes on the next chunk.
    assert choice.get("finish_reason") in (None,), notice_chunk
    # Refusal text is plain-spoken; no embedded sentinel.
    assert "studio:anthropic-refusal" not in choice["delta"]["content"]


def test_refusal_tool_event_chunk_shape(monkeypatch):
    """Drop signal rides an Unsloth `_toolEvent` envelope (delta={},
    finish_reason=null); the frontend latches on
    `_toolEvent.type == "anthropic_refusal"`."""
    _, lines = _capture(monkeypatch, sse = _refusal_sse(), model = "claude-opus-4-7")
    refusal_chunk = None
    for line in lines:
        if line.startswith("data: ") and "anthropic_refusal" in line:
            refusal_chunk = json.loads(line[len("data: ") :])
            break
    assert refusal_chunk is not None, lines
    assert refusal_chunk["_toolEvent"] == {"type": "anthropic_refusal"}, refusal_chunk
    choice = refusal_chunk["choices"][0]
    assert choice["delta"] == {}, refusal_chunk
    assert choice["finish_reason"] is None, refusal_chunk


# ──────────────────────────── future-proofing ────────────────────────────
def test_fast_mode_prefix_tuple_matches_capability_doc(monkeypatch):
    """Tuple must exactly match the two families in the upstream docs:
    https://platform.claude.com/docs/en/build-with-claude/fast-mode."""
    from core.inference.external_provider import _ANTHROPIC_FAST_MODE_PREFIXES
    assert set(_ANTHROPIC_FAST_MODE_PREFIXES) == {
        "claude-opus-4-7",
        "claude-opus-4-6",
    }, _ANTHROPIC_FAST_MODE_PREFIXES


def test_fast_mode_speed_field_value_is_literal_fast(monkeypatch):
    """Pin the wire value to the literal string ``"fast"``."""
    cap, _ = _capture(monkeypatch, fast_mode = True, model = "claude-opus-4-7")
    assert cap["body"]["speed"] == "fast", cap["body"]


def test_fast_mode_dropped_on_opus_4_5_dated_snapshot(monkeypatch):
    """Previous-family snapshots like ``claude-opus-4-5-2025-...`` must not match."""
    cap, _ = _capture(monkeypatch, fast_mode = True, model = "claude-opus-4-5-2025-08-01")
    assert "speed" not in cap["body"], cap["body"]
    assert "fast-mode-2026-02-01" not in cap["headers"].get("anthropic-beta", "")


def test_fast_mode_rejects_prefix_collision_4_70(monkeypatch):
    """IDs like ``claude-opus-4-70`` / ``-4-7b`` must not match the prefix."""
    cap, _ = _capture(monkeypatch, fast_mode = True, model = "claude-opus-4-70")
    assert "speed" not in cap["body"], cap["body"]
    assert "fast-mode-2026-02-01" not in cap["headers"].get("anthropic-beta", "")


def test_fast_mode_rejects_prefix_collision_4_7b(monkeypatch):
    cap, _ = _capture(monkeypatch, fast_mode = True, model = "claude-opus-4-7b")
    assert "speed" not in cap["body"], cap["body"]
    assert "fast-mode-2026-02-01" not in cap["headers"].get("anthropic-beta", "")


def test_fast_mode_rejects_prefix_collision_4_6_extra(monkeypatch):
    cap, _ = _capture(monkeypatch, fast_mode = True, model = "claude-opus-4-60")
    assert "speed" not in cap["body"], cap["body"]
    assert "fast-mode-2026-02-01" not in cap["headers"].get("anthropic-beta", "")


# ──────────────────────────── usage.speed propagation ────────────────────────────
def _fast_speed_sse(model: str = "claude-opus-4-7", speed: str = "fast") -> bytes:
    return (
        b'event: message_start\ndata: {"type":"message_start","message":'
        b'{"id":"m1","content":[],"model":"' + model.encode() + b'",'
        b'"role":"assistant","stop_reason":null,"usage":'
        b'{"input_tokens":4,"output_tokens":1}}}\n\n'
        b'event: content_block_start\ndata: {"type":"content_block_start",'
        b'"index":0,"content_block":{"type":"text","text":""}}\n\n'
        b'event: content_block_delta\ndata: {"type":"content_block_delta",'
        b'"index":0,"delta":{"type":"text_delta","text":"hi"}}\n\n'
        b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
        b'event: message_delta\ndata: {"type":"message_delta",'
        b'"delta":{"stop_reason":"end_turn"},'
        b'"usage":{"output_tokens":5,"speed":"' + speed.encode() + b'"}}\n\n'
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )


def test_usage_speed_propagates_to_final_usage_chunk_fast(monkeypatch):
    """``usage.speed == "fast"`` from upstream must reach the Unsloth usage chunk."""
    _, lines = _capture(monkeypatch, sse = _fast_speed_sse(speed = "fast"))
    usage_lines = [l for l in lines if l.startswith("data: ") and '"usage"' in l]
    assert usage_lines, lines
    parsed = [json.loads(l[len("data: ") :]) for l in usage_lines]
    speeds = [p["usage"].get("speed") for p in parsed if "usage" in p]
    assert "fast" in speeds, parsed


def test_usage_speed_propagates_to_final_usage_chunk_standard(monkeypatch):
    _, lines = _capture(monkeypatch, sse = _fast_speed_sse(speed = "standard"))
    parsed = [
        json.loads(l[len("data: ") :]) for l in lines if l.startswith("data: ") and '"usage"' in l
    ]
    speeds = [p["usage"].get("speed") for p in parsed if "usage" in p]
    assert "standard" in speeds, parsed


def test_usage_speed_absent_when_anthropic_does_not_report(monkeypatch):
    """Unsloth must not invent ``usage.speed`` when upstream omits it."""
    _, lines = _capture(monkeypatch)
    parsed = [
        json.loads(l[len("data: ") :]) for l in lines if l.startswith("data: ") and '"usage"' in l
    ]
    for p in parsed:
        usage = p.get("usage") or {}
        assert "speed" not in usage, p
