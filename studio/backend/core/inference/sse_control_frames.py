# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep a provider's bytes off Unsloth's own control channel.

Unsloth multiplexes its UI control protocol onto the same SSE stream a provider's
chunks are relayed on. The chat client picks those frames out structurally: a
top-level ``type`` of ``tool_start`` / ``tool_end`` / ``tool_output`` /
``tool_args`` / ``tool_status`` (and the local-runtime ``diffusion_frame`` /
``reasoning_summary``) becomes a tool card, a badge or a canvas rather than
assistant text, as does a ``_toolEvent`` / ``_toolStatus`` key stamped inside an
otherwise ordinary chunk.

Every one of those frames is written by this server. A provider endpoint -- a
user-configured base_url, so not necessarily one Unsloth or the user controls --
has no legitimate reason to emit any of them, and a verbatim relay makes its copy
indistinguishable from ours at the client: a forged card can claim a tool the
user trusts ran and returned something harmless, carrying
``provenance: {"source": "local"}``, when nothing ran at all. So strip the
control vocabulary out of everything that arrives from a provider. The
``delta.reasoning`` alias Ollama and newer vLLM send is renamed to the canonical
``reasoning_content``, streamed deltas only. The rest of the chunk stays as it was.
"""

from __future__ import annotations

import json

from typing import Any


# Top-level "type" values the chat client routes away from the transcript.
# Anything here paints UI on the user's behalf, so only this server may send it.
_CONTROL_TYPES = frozenset(
    {
        "tool_start",
        "tool_end",
        "tool_output",
        "tool_args",
        "tool_status",
        "diffusion_frame",
        "reasoning_summary",
    }
)

# Unsloth extensions carried inside a chunk. Not part of any provider's wire
# format, and read by the client with the same trust as the frames above.
_CONTROL_KEYS = (
    "_toolEvent",
    "_toolStatus",
    "_diffusionFrame",
    "_reasoningDurationMs",
    # The loop stamps this itself after sanitising; a provider that sends its own
    # would otherwise name any server it likes on the card.
    "_mcp_provenance",
)

# What is left of a stripped frame is only worth relaying if it still says
# something in the provider's own vocabulary.
_SUBSTANTIVE_KEYS = ("choices", "usage", "error")


def _normalize_reasoning_deltas(payload: dict[str, Any]) -> bool:
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return False
    changed = False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        reasoning = delta.get("reasoning")
        if not isinstance(reasoning, str) or not reasoning:
            # An empty alias says nothing, and rewriting it would cost a re-encode.
            continue
        details = delta.get("reasoning_details")
        if isinstance(details, list) and any(
            isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"]
            for part in details
        ):
            # OpenRouter repeats the thought in reasoning_details and the client
            # concatenates both; details carrying no text are not a second copy.
            continue
        canonical = delta.get("reasoning_content")
        if canonical is not None and (not isinstance(canonical, str) or canonical.strip()):
            # A canonical value the provider set wins, unless it is blank.
            continue
        delta["reasoning_content"] = reasoning
        delta.pop("reasoning", None)
        changed = True
    return changed


def sanitize_provider_sse_line(line: str) -> str | None:
    """Return ``line`` fit to relay, or ``None`` if nothing of it should be.

    Non-``data:`` lines (comments, ``event:``, ``id:``, ``retry:``) and payloads
    that are not a JSON object are passed through untouched: they cannot reach
    the control path, and rewriting them would cost a re-encode on every chunk of
    ordinary prose.
    """
    if not line.startswith("data:"):
        return line
    raw = line[5:].strip()
    if not raw or raw == "[DONE]":
        return line
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return line
    if not isinstance(payload, dict):
        return line

    normalized_reasoning = _normalize_reasoning_deltas(payload)
    forged_type = isinstance(payload.get("type"), str) and payload["type"] in _CONTROL_TYPES
    forged_keys = [key for key in _CONTROL_KEYS if key in payload]
    if not normalized_reasoning and not forged_type and not forged_keys:
        # The overwhelmingly common case: relay the provider's own bytes rather
        # than paying a re-encode to reproduce them.
        return line

    cleaned: dict[str, Any] = {
        key: value
        for key, value in payload.items()
        if key not in forged_keys and not (forged_type and key == "type")
    }
    if not any(key in cleaned for key in _SUBSTANTIVE_KEYS):
        # A pure control frame with the control stripped out is an empty
        # envelope; relaying it would only make the client parse nothing.
        return None
    return "data: " + json.dumps(cleaned, separators = (",", ":"))
