# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Mercury diffusion streaming through the Studio tool loop.

The client takes every text of a diffusing turn as the whole reply so far, so the final full text
must leave the loop as one chunk: a healer split would leave only its last piece on screen.
"""

from __future__ import annotations

import json

from .test_external_tool_edge_cases import (  # noqa: F401  (executed is a fixture)
    _DONE,
    FakeTransport,
    _answer_turn,
    _payloads,
    _run,
    _sse,
    executed,
)


class _ExternalTransport(FakeTransport):
    # As OAICompatTransport: frames were sanitized upstream, so the server-built replace frames stay.
    sanitizes_provider_frames = True


def _diffusing_turn(snapshots, final):
    lines = [
        "data: " + json.dumps({"choices": [{"index": 0, "delta": {}}], "_diffusionReplace": text})
        for text in snapshots
    ]
    lines.append(_sse({"content": final}, finish = "stop"))
    lines.append(_DONE)
    return lines


def _texts_after_diffusion(lines):
    """Every text the client would swap in as a snapshot, in order."""
    out = []
    started = False
    for payload in _payloads(lines):
        if isinstance(payload.get("_diffusionReplace"), str):
            started = True
            continue
        choices = payload.get("choices")
        delta = choices[0].get("delta") if isinstance(choices, list) and choices else None
        content = delta.get("content") if isinstance(delta, dict) else None
        if started and isinstance(content, str) and content:
            out.append(content)
    return out


def test_diffusing_prose_with_tool_call_marker_is_one_snapshot(executed):
    final = "Wrap a call in <tool_call> tags, or write <function=name> for one. That is all."
    transport = _ExternalTransport([_diffusing_turn(["Wrxp a cxll", final], final)])
    lines = _run(transport)

    assert any("_diffusionReplace" in line for line in lines)
    assert _texts_after_diffusion(lines) == [final]
    assert executed == []


def test_diffusing_healed_tool_call_still_runs(executed):
    call = '<tool_call>{"name": "web_search", "arguments": {"query": "paris"}}</tool_call>'
    final = "Let me search." + call
    transport = _ExternalTransport(
        [_diffusing_turn(["Lxt me sxarch.", final], final), _answer_turn("Paris it is.")]
    )
    lines = _run(transport)

    assert [c["name"] for c in executed] == ["web_search"]
    assert _texts_after_diffusion(lines)[0] == "Let me search."
    assert "<tool_call>" not in "".join(_texts_after_diffusion(lines))


def test_non_diffusing_split_is_unchanged(executed):
    """Without diffusion the healer still streams prose ahead of a held marker."""
    text = "Wrap a call in <tool_call> tags. That is all."
    transport = FakeTransport([[_sse({"content": text}), _sse(finish = "stop"), _DONE]])
    lines = _run(transport)
    contents = [
        payload["choices"][0]["delta"]["content"]
        for payload in _payloads(lines)
        if isinstance(payload.get("choices"), list)
        and payload["choices"]
        and isinstance(payload["choices"][0].get("delta"), dict)
        and payload["choices"][0]["delta"].get("content")
    ]
    assert "".join(contents) == text
    assert len(contents) > 1
