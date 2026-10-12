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


# Client routes these away from the transcript and paints UI, so only this server may send them.
_CONTROL_TYPES = frozenset(
    {
        "tool_start",
        "tool_end",
        "tool_output",
        "tool_args",
        "tool_status",
        "skill_load",
        "diffusion_frame",
        "reasoning_summary",
        "context_usage",
    }
)

# Unsloth extensions inside a chunk: same trust as the frames above.
_CONTROL_KEYS = (
    "_toolEvent",
    "_toolStatus",
    "_diffusionFrame",
    "_reasoningDurationMs",
    "_contextUsage",
    "_mcp_provenance",
    "quote_cut",
)

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
    # isinstance first: an unhashable non-string `type` raises on `in`.
    if not isinstance(payload.get("type"), str):
        return False
    return not any(key in payload for key in _SUBSTANTIVE_KEYS)


def strip_server_executed_tool_call(line: str, pending_call: bool = False) -> str | None:
    """Only for the Unsloth tool-loop path: strips the server-run call so a client does not run it again."""
    payload = _sse_payload(line)
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
            # tool_calls only: a legacy function_call is the caller's to run.
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
        payload.pop("_mcp_provenance", None)
    if not _choices_say_anything(kept_choices) and "usage" not in payload:
        return None
    return "data: " + json.dumps(payload, separators = (",", ":"))


def _line_offers_tool_call(line: str) -> bool:
    """Keyed on the key being present, not truthy, so it agrees with the strip."""
    payload = _sse_payload(line)
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
    """``strip_server_executed_tool_call`` with the one bit of turn state it needs.

    A turn whose calls this server runs is not over when the provider says it is, and the
    provider does not always say "tool_calls": llama.cpp and vLLM finish a structured call
    on "stop", which the loop runs anyway. Stripping only the call then leaves a caller
    holding an empty chunk marked ``finish_reason: "stop"``, and a client that ends the
    turn there never reads the answer the loop is about to stream -- the same lost reply
    the control-frame gate exists to prevent, arrived at from the other side.

    So remember, per stream, that a call was withheld and no reply has followed it, and
    treat the "stop" that closes that turn the way "tool_calls" is already treated. The
    next turn opens with the flag clear, so its own finish_reason is relayed untouched.

    The flag is raised BEFORE the strip reads it, because a provider may co-emit the call
    and the finish_reason on one line (any non-streamed upstream relayed as a single line
    does), and the turn boundary is read on every line, because that same line both opens
    and closes the turn. Getting either wrong leaks the empty "stop" this exists to hold
    back, or swallows a later turn's legitimate one.

    Blanking the last finish_reason of a stream would leave the caller none at all --
    openai-node raises "missing finish_reason for choice 0" outright -- so the withheld
    ones are counted and ``owed_terminal_chunk`` mints a replacement when the loop ends
    without opening the turn it promised (a spent tool budget, a discarded call, a
    provider that failed mid-loop).

    One instance per request. The state is per line, not per choice index, which is exact
    here because the loop reads choice 0 only and this path rejects n > 1 upstream.
    """

    def __init__(self) -> None:
        self._pending_call = False
        self._owes_finish = False
        self._last_envelope: dict[str, Any] | None = None

    def arm(self) -> None:
        """Withholds the turn-ending reason for a text-healed tool call, which never shows as tool_calls."""
        self._pending_call = True
        self._owes_finish = True

    def end_turn(self) -> None:
        """Clears the withheld-call flag so the next turn's finish reasons are not stripped; the
        debt stays."""
        self._pending_call = False

    def strip(self, line: str) -> str | None:
        pending = self._pending_call or _line_offers_tool_call(line)
        out = strip_server_executed_tool_call(line, pending_call = pending)
        ends_turn = _line_ends_turn(line)
        self._pending_call = pending and not ends_turn
        if pending or (ends_turn and (out is None or not _line_ends_turn(out))):
            # Armed on withhold: a provider may close on [DONE] alone. The second clause covers a removed
            # "tool_calls" reason with no emitted call (llama.cpp / vLLM parser bugs).
            self._owes_finish = True
        if out is not None and _line_ends_turn(out):
            self._owes_finish = False
        self._remember_envelope(line)
        return out

    def _remember_envelope(self, line: str) -> None:
        """Keep the last chunk's identity, so a minted finish matches the stream."""
        payload = _sse_payload(line)
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
    payload = _sse_payload(line)
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
