# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Streamed thinking has to reach its consumers whichever field a provider uses.

``test_provider_control_frame_spoofing.py`` pins the rename in isolation; these drive
whole streams through the real relay, which is where #8838 failed. Ollama sends thinking
as ``delta.reasoning`` while Deep Research counts only non-empty ``delta.content`` /
``delta.reasoning_content`` as output (``core/research_runs.py``), so a reasoning-only
prefix spent the first-output budget. The chat client, the second consumer, concatenates
``reasoning_content`` with the text in ``reasoning_details`` (``chat-adapter.ts``), so a
provider sending both must not have the alias renamed into a second copy.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient


THOUGHT = ["I need ", "to think ", "about this."]
ANSWER = ["The ", "answer."]


def _chunk(delta: dict) -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]}) + "\n\n"


def _ollama() -> str:
    return "".join(_chunk({"content": "", "reasoning": t}) for t in THOUGHT)


def _deepseek() -> str:
    return "".join(_chunk({"reasoning_content": t}) for t in THOUGHT)


def _openrouter() -> str:
    return "".join(
        _chunk({"reasoning": t, "reasoning_details": [{"type": "reasoning.text", "text": t}]})
        for t in THOUGHT
    )


def _openrouter_encrypted() -> str:
    return "".join(
        _chunk(
            {"reasoning": t, "reasoning_details": [{"type": "reasoning.encrypted", "data": "zz"}]}
        )
        for t in THOUGHT
    )


SHAPES = {
    "ollama": _ollama,
    "deepseek": _deepseek,
    "openrouter": _openrouter,
    "openrouter_encrypted": _openrouter_encrypted,
}


def _relay(body: str) -> list[str]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content = body, headers = {"content-type": "text/event-stream"})

    ep_mod._http_client = httpx.AsyncClient(transport = httpx.MockTransport(handler))
    client = ExternalProviderClient(
        provider_type = "ollama", base_url = "http://endpoint.invalid/v1", api_key = ""
    )

    async def run() -> list[str]:
        return [
            line
            async for line in client.stream_chat_completion(
                messages = [{"role": "user", "content": "ping"}], model = "m"
            )
        ]

    return asyncio.new_event_loop().run_until_complete(run())


def _consume(lines: list[str]) -> dict:
    """What Deep Research counts, and what the chat client would render."""
    seen = {"research_reasoning": "", "research_report": "", "rendered": "", "output": False}
    for line in lines:
        if not line.startswith("data: ") or line[6:].strip() in ("", "[DONE]"):
            continue
        for choice in json.loads(line[6:]).get("choices", []):
            delta = choice.get("delta") or {}
            thought = delta.get("reasoning_content")
            text = delta.get("content")
            if isinstance(thought, str) and thought:
                seen["research_reasoning"] += thought
                seen["output"] = True
            if isinstance(text, str) and text:
                seen["research_report"] += text
                seen["output"] = True
            details = delta.get("reasoning_details")
            seen["rendered"] += thought if isinstance(thought, str) else ""
            if isinstance(details, list):
                seen["rendered"] += "".join(
                    p.get("text") if isinstance(p, dict) and isinstance(p.get("text"), str) else ""
                    for p in details
                )
    return seen


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_thinking_is_rendered_exactly_once(shape):
    seen = _consume(_relay(SHAPES[shape]() + "".join(_chunk({"content": c}) for c in ANSWER)))

    # doubling here is the thinking block printing every thought twice
    assert seen["rendered"] == "".join(THOUGHT)
    assert seen["research_report"] == "".join(ANSWER)


@pytest.mark.parametrize("shape", ["ollama", "deepseek", "openrouter_encrypted"])
def test_a_reasoning_only_prefix_is_already_output(shape):
    """#8838: the first-output budget must be disarmed before any content arrives."""
    seen = _consume(_relay(SHAPES[shape]()))

    assert seen["research_reasoning"] == "".join(THOUGHT)
    assert seen["output"] is True
    assert seen["research_report"] == ""


def test_openrouter_text_details_stay_the_only_copy():
    """Renaming the alias here would double the thinking block, so it is left alone.

    Deep Research still cannot see reasoning that only arrives as
    ``reasoning_details``; that is the same on main and is its own fix.
    """
    seen = _consume(_relay(_openrouter()))

    assert seen["rendered"] == "".join(THOUGHT)
    assert seen["research_reasoning"] == ""
