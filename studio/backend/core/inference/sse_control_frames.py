# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Strips Unsloth control frames from provider SSE so providers cannot forge tool cards or provenance."""

from __future__ import annotations

import json

from typing import Any


# Client routes these away from the transcript and paints UI, so only this server may send them.
_CONTROL_TYPES = frozenset(
    {
        "tool_start",
        "tool_end",
        "tool_output",
        "tool_args",
        "tool_status",
        "tool_model_note",
        "tool_approved",
        "skill_load",
        "diffusion_frame",
        "reasoning_summary",
        "context_usage",
        "generation_progress",
    }
)

# Unsloth extensions carried inside a chunk: in no provider's wire format, read with the same trust as the frames above.
_CONTROL_KEYS = (
    "_toolEvent",
    "_toolStatus",
    "_diffusionFrame",
    "_reasoningDurationMs",
    "_contextUsage",
    "_generationProgress",
    "_mcp_provenance",
    "quote_cut",
)

# A stripped frame is only worth relaying if it still says something in the provider's own vocabulary.
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
            # OpenRouter repeats the thought in reasoning_details and the client concatenates both.
            continue
        canonical = delta.get("reasoning_content")
        if canonical is not None and (not isinstance(canonical, str) or canonical.strip()):
            continue
        delta["reasoning_content"] = reasoning
        delta.pop("reasoning", None)
        changed = True
    return changed


def sanitize_provider_sse_line(line: str) -> str | None:
    """Non-data lines and non-object payloads pass untouched, so ordinary prose is never re-encoded."""
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
    """Frames without choices, which OpenAI clients cannot route; usage and error frames are never held."""
    payload = _sse_payload(line)
    if payload is None:
        return False
    # isinstance first: the sanitizer passes a non-string `type` through, and an unhashable one raises on `in`.
    if not isinstance(payload.get("type"), str):
        return False
    return not any(key in payload for key in _SUBSTANTIVE_KEYS)


def strip_server_executed_tool_call(line: str, pending_call: bool = False) -> str | None:
    """Only for the Unsloth tool-loop path, which strips the server-run call so a client does not rerun it; on a plain proxy calls must pass through untouched."""
    return _strip_payload_tool_call(line, _sse_payload(line), pending_call)


def _strip_payload_tool_call(
    line: str, payload: dict[str, Any] | None, pending_call: bool
) -> str | None:
    """``strip_server_executed_tool_call`` for a line whose payload is already decoded; never mutates it."""
    choices = payload.get("choices") if payload else None
    if not isinstance(choices, list) or not choices:
        return line

    not_really_final = ("tool_calls", "stop", "function_call") if pending_call else ("tool_calls",)
    changed = False
    calls_withheld = False
    kept_choices = []
    for choice in choices:
        if not isinstance(choice, dict):
            kept_choices.append(choice)
            continue
        choice = dict(choice)
        withheld = False
        for src_key in ("delta", "message"):
            src = choice.get(src_key)
            # tool_calls only: the loop reads no other form, so a legacy function_call is the caller's to run.
            if isinstance(src, dict) and "tool_calls" in src:
                src = {k: v for k, v in src.items() if k != "tool_calls"}
                choice[src_key] = src
                withheld = calls_withheld = True
        if choice.get("finish_reason") in not_really_final:
            # Blanked, not renamed: the loop answers in the next turn. "function_call" counts only once a
            # call is pending (LocalAI closes modern tool_calls streams on the legacy reason).
            choice["finish_reason"] = None
            withheld = True
        changed = changed or withheld
        kept_choices.append(choice)

    if not changed:
        return line
    payload = {**payload, "choices": kept_choices}
    if calls_withheld:
        # Names the withheld calls by id; the caller never saw them.
        payload.pop("_mcp_provenance", None)
    if not _choices_say_anything(kept_choices) and "usage" not in payload:
        return None
    return "data: " + json.dumps(payload, separators = (",", ":"))


def _payload_offers_tool_call(payload: dict[str, Any] | None) -> bool:
    """Keyed on the key being present, not truthy, so it agrees with the strip."""
    choices = payload.get("choices") if payload else None
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        for src_key in ("delta", "message"):
            src = choice.get(src_key)
            if isinstance(src, dict) and "tool_calls" in src:
                return True
    return False


class ServerToolCallStripper:
    """``strip_server_executed_tool_call`` plus turn state; one instance per request.

    llama.cpp and vLLM close a structured call on "stop", so once a call is withheld the
    "stop" closing that turn is blanked like "tool_calls" (else the client ends the turn
    before the loop's reply). The flag is raised BEFORE the strip and the turn boundary is
    read on every line, since one line can both carry the call and close the turn.
    A blanked last finish_reason is owed: openai-node raises "missing finish_reason for
    choice 0", so ``owed_terminal_chunk`` mints one. State is per line, not per choice:
    the loop reads choice 0 only and n > 1 is rejected upstream.
    """

    def __init__(self) -> None:
        self._pending_call = False
        self._owes_finish = False
        self._last_envelope: dict[str, Any] | None = None

    def arm(self) -> None:
        """Withholds the turn-ending reason for a text-healed tool call; the loop must call this before releasing the chunk that closes that turn."""
        self._pending_call = True
        self._owes_finish = True

    def end_turn(self) -> None:
        """Clears the withheld-call flag so the next turn's finish reasons are not stripped; the
        debt stays."""
        self._pending_call = False

    def strip(self, line: str) -> str | None:
        payload = _sse_payload(line)
        pending = self._pending_call or _payload_offers_tool_call(payload)
        out = _strip_payload_tool_call(line, payload, pending)
        ends_turn = _payload_ends_turn(payload)
        out_ends_turn = ends_turn if out is line else (out is not None and _line_ends_turn(out))
        # Turn closed: whatever the loop does next opens a turn whose finish_reason is the caller's to read.
        self._pending_call = pending and not ends_turn
        if pending or (ends_turn and not out_ends_turn):
            # Armed on withhold: a provider may close on [DONE] alone.
            self._owes_finish = True
        if out_ends_turn:
            self._owes_finish = False
        self._remember_payload_envelope(payload)
        return out

    def _remember_payload_envelope(self, payload: dict[str, Any] | None) -> None:
        """Keep the last chunk's identity, so a minted finish matches the stream."""
        if not payload or "choices" not in payload:
            return
        envelope = {
            key: payload[key] for key in ("id", "object", "created", "model") if key in payload
        }
        if envelope:
            self._last_envelope = envelope

    def owed_terminal_chunk(self) -> str | None:
        """finish_reason is required by the OpenAI schema, so a stream ending without one breaks clients."""
        if not self._owes_finish:
            return None
        self._owes_finish = False
        payload = dict(self._last_envelope or {})
        payload["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        return "data: " + json.dumps(payload, separators = (",", ":"))


def _line_ends_turn(line: str) -> bool:
    """Whether this line carries any finish_reason, i.e. closes the provider's turn."""
    return _payload_ends_turn(_sse_payload(line))


def _payload_ends_turn(payload: dict[str, Any] | None) -> bool:
    choices = payload.get("choices") if payload else None
    if not isinstance(choices, list):
        return False
    return any(
        isinstance(choice, dict) and choice.get("finish_reason") is not None for choice in choices
    )


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
