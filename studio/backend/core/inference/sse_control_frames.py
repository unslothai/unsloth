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


# Top-level "type" values the client routes away from the transcript: they paint UI on the user's behalf, so only this
# server may send them.
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

# Unsloth extensions carried inside a chunk: in no provider's wire format, read with the same trust as the frames above.
_CONTROL_KEYS = ("_toolEvent", "_toolStatus", "_diffusionFrame", "_reasoningDurationMs")

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
        # A pure control frame with the control stripped out is an empty envelope.
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
    # isinstance first: the sanitizer passes a non-string `type` through, and an unhashable one raises on `in`.
    if not isinstance(payload.get("type"), str):
        return False
    return not any(key in payload for key in _SUBSTANTIVE_KEYS)


def strip_server_executed_tool_call(line: str, pending_call: bool = False) -> str | None:
    """Hold a call the server runs itself back from a caller that did not opt in.

    ``stream_with_studio_tools`` relays the provider's own ``delta.tool_calls`` and the
    ``finish_reason: "tool_calls"`` that ends that turn, for a call Unsloth then executes
    and answers in a later turn. Its catalogue is Unsloth's own, never the caller's, so a
    client reading those chunks is told to run a tool that is already running here: an
    agent may run it a second time, or stop at the finish_reason and never read the real
    answer. Returns the line with the call and that finish_reason removed, or None when
    nothing worth relaying was left.

    ``pending_call`` says a call was already withheld earlier in this turn, which makes a
    ``finish_reason: "stop"`` on this line just as misleading as "tool_calls": llama.cpp
    and vLLM routinely end a perfectly good tool call on "stop" and the loop deliberately
    runs those (see studio_tool_loop's ``truncated`` rule), so the turn has not finished
    either. Callers that track the turn use ``ServerToolCallStripper`` rather than passing
    this by hand. It also covers the legacy "function_call", which a gateway may still use
    to close a turn it streamed modern ``delta.tool_calls`` for. "length" and
    "content_filter" are left alone on purpose: those are the two the loop refuses to run,
    so that turn really is the last one.

    Only for the Unsloth-tool-loop path. On a plain proxy the calls are the caller's own
    and must pass through untouched.
    """
    payload = _sse_payload(line)
    choices = payload.get("choices") if payload else None
    if not isinstance(choices, list) or not choices:
        return line

    not_really_final = ("tool_calls", "stop", "function_call") if pending_call else ("tool_calls",)
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
            # tool_calls only: the loop reads no other form, so a legacy function_call is the caller's to run.
            if isinstance(src, dict) and "tool_calls" in src:
                src = {k: v for k, v in src.items() if k != "tool_calls"}
                choice[src_key] = src
                withheld = True
        if choice.get("finish_reason") in not_really_final:
            # Blanked, not renamed: the turn has not finished, the loop answers in the next
            # one. Cannot be keyed on a call withheld on this line, since the arguments
            # arrived in earlier chunks and this delta is usually empty.
            #
            # "function_call" only counts once a call is pending, for the gateway that
            # streams modern delta.tool_calls then closes on the legacy reason (LocalAI does
            # whenever the client sent no tools). A genuine legacy call never sets pending
            # (_line_offers_tool_call keys on "tool_calls" alone), so it keeps its reason.
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


def _line_offers_tool_call(line: str) -> bool:
    """Whether this line carries a ``tool_calls`` fragment at all.

    Keyed on the key being present, not on it being truthy, so it reads the same
    condition the strip itself does: a line whose only evidence is an empty list is
    still stripped, and the two must not disagree about whether a call was withheld.
    """
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
        """Withhold the next turn-ending reason for a call that never reached the wire.

        A text-form ``<tool_call>`` healed out of ordinary content is executed by the loop
        but is invisible here: the markup is removed from the content it arrived in, and no
        ``tool_calls`` key ever appears, so ``_line_offers_tool_call`` cannot see it. The
        loop knows, and says so by calling this before it releases the chunk that closes
        that turn. Everything after is the structured path's behaviour exactly, debt
        included.
        """
        self._pending_call = True
        self._owes_finish = True

    def end_turn(self) -> None:
        """The loop finished a provider turn, whatever the provider said to close it.

        A turn boundary is normally read off the finish_reason on the wire, but a provider
        may close on ``[DONE]`` alone and the loop consumes that sentinel before this ever
        sees it, leaving the withheld-call flag raised into the next turn. The next turn's
        reasons are its own: a legacy ``function_call`` there belongs to the caller, and
        stripping it as though it closed the previous call means the caller never dispatches
        it. The debt is deliberately left alone -- it is still owed until a real terminal
        reaches the caller.
        """
        self._pending_call = False

    def strip(self, line: str) -> str | None:
        pending = self._pending_call or _line_offers_tool_call(line)
        out = strip_server_executed_tool_call(line, pending_call = pending)
        ends_turn = _line_ends_turn(line)
        # Turn closed: whatever the loop does next opens a turn whose finish_reason is the caller's to read.
        self._pending_call = pending and not ends_turn
        if pending or (ends_turn and (out is None or not _line_ends_turn(out))):
            # Armed as soon as a call is withheld, not only where a finish_reason was
            # removed: a provider closing the turn on [DONE] alone never offers one to
            # remove. Settled below the moment a real terminal is relayed.
            #
            # The second clause is the reverse: a reason removed with no call ever seen. A
            # provider reporting "tool_calls" for a call its own parser failed to emit
            # (open llama.cpp and vLLM bugs) gives `pending` nothing to latch onto, so the
            # reason is blanked and no debt recorded, leaving no finish_reason at all.
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
        """A finish chunk to send before [DONE], or None when the caller already has one.

        finish_reason is a required key in the OpenAI chunk schema, so a stream that ends
        without one is not merely unhelpful: openai-node raises, and openai-python hands
        back a parsed completion whose finish_reason is None in a field its own type
        declares non-nullable. Mirrors the GGUF passthrough's _synthetic_finish_line.
        """
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
