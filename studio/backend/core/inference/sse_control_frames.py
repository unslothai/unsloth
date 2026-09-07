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


# Top-level "type" values the chat client routes away from the transcript. Anything here paints UI on the user's behalf,
# so only this server may send it.
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

# unsloth extensions, in no provider's wire format, read with the same trust as the frames above
# Unsloth extensions carried inside a chunk. Not part of any provider's wire format, and read by the client with the
# same trust as the frames above.
_CONTROL_KEYS = ("_toolEvent", "_toolStatus", "_diffusionFrame", "_reasoningDurationMs")

# a stripped frame is only worth relaying if it still says something in the provider's vocabulary
# What is left of a stripped frame is only worth relaying if it still says something in the provider's own vocabulary.
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
            continue
        details = delta.get("reasoning_details")
        if isinstance(details, list) and any(
            isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"]
            for part in details
        ):
            # OpenRouter repeats the thought in reasoning_details and the client concatenates both; details carrying no
            # text are not a second copy.
            continue
        canonical = delta.get("reasoning_content")
        if canonical is not None and (not isinstance(canonical, str) or canonical.strip()):
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
        return line

    cleaned: dict[str, Any] = {
        key: value
        for key, value in payload.items()
        if key not in forged_keys and not (forged_type and key == "type")
    }
    if not any(key in cleaned for key in _SUBSTANTIVE_KEYS):
        # A pure control frame with the control stripped out is an empty envelope; relaying it would only make the
        # client parse nothing.
        return None
    return "data: " + json.dumps(cleaned, separators = (",", ":"))


def _sse_payload(line: str) -> dict[str, Any] | None:
    """The JSON object a ``data:`` line carries, or None if it carries none."""
    if not line.startswith("data:"):
        return None
    raw = line[5:].strip()
    if not raw or raw == "[DONE]":
        return None
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def is_ui_control_sse_line(line: str) -> bool:
    """Whether ``line`` is a frame no OpenAI client can route, rather than a chunk.

    Read by the OpenAI-compatible route to hold these back from a caller that did not opt
    in: with no ``choices`` they fail schema validation mid-stream. Structural rather than
    a name list, because ``_CONTROL_TYPES`` answers a different question -- what a
    PROVIDER must not forge -- and the tool loop also writes bare ``status`` frames around
    a RAG autoinjection, which are just as unroutable without being forgeable. ``usage``
    and ``error`` keep a frame: those are the provider's own vocabulary and a client reads
    them. A chunk that merely carries a ``_toolEvent``-style key has ``choices`` and stays.
    """
    payload = _sse_payload(line)
    if payload is None:
        return False
    # isinstance first: the sanitizer passes a non-string `type` through, and an unhashable
    # one (a provider putting structured metadata there) raises on a membership test.
    if not isinstance(payload.get("type"), str):
        return False
    return not any(key in payload for key in _SUBSTANTIVE_KEYS)


def strip_server_executed_tool_call(line: str) -> str | None:
    """Hold a call the server runs itself back from a caller that did not opt in.

    ``stream_with_studio_tools`` relays the provider's own ``delta.tool_calls`` and the
    ``finish_reason: "tool_calls"`` that ends that turn, for a call Unsloth then executes
    and answers in a later turn. Its catalogue is Unsloth's own, never the caller's, so a
    client reading those chunks is told to run a tool that is already running here: an
    agent may run it a second time, or stop at the finish_reason and never read the real
    answer. Returns the line with the call and that finish_reason removed, or None when
    nothing worth relaying was left.

    Only for the Unsloth-tool-loop path. On a plain proxy the calls are the caller's own
    and must pass through untouched.
    """
    payload = _sse_payload(line)
    choices = payload.get("choices") if payload else None
    if not isinstance(choices, list) or not choices:
        return line

    changed = False
    kept_choices = []
    for choice in choices:
        if not isinstance(choice, dict):
            kept_choices.append(choice)
            continue
        choice = dict(choice)
        withheld = False
        for src_key in ("delta", "message"):
            src = choice.get(src_key)
            # tool_calls only. The loop reads no other form, so the legacy function_call
            # is a call it never executes and the caller is the one meant to run it.
            if isinstance(src, dict) and "tool_calls" in src:
                src = {k: v for k, v in src.items() if k != "tool_calls"}
                choice[src_key] = src
                withheld = True
        if choice.get("finish_reason") == "tool_calls":
            # The arguments arrive in earlier chunks and this one usually carries an empty
            # delta, so it cannot be keyed on a call withheld here. Not a rename either:
            # the turn has not finished, the loop answers in the next one. A legacy call
            # ends on "function_call", a different value, and is left alone with its delta.
            choice["finish_reason"] = None
            withheld = True
        changed = changed or withheld
        kept_choices.append(choice)

    if not changed:
        return line
    payload = {**payload, "choices": kept_choices}
    if not _choices_say_anything(kept_choices) and "usage" not in payload:
        return None
    return "data: " + json.dumps(payload, separators = (",", ":"))


def _choices_say_anything(choices: list[Any]) -> bool:
    """Whether anything survived the strip that a client would act on."""
    for choice in choices:
        if not isinstance(choice, dict):
            return True
        if choice.get("finish_reason") is not None:
            return True
        for src_key in ("delta", "message"):
            src = choice.get(src_key)
            if isinstance(src, dict) and any(value for value in src.values()):
                return True
    return False
