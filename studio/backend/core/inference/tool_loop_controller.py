# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared controller state for Unsloth local agentic tool loops.

This module is intentionally dependency-light: it owns only per-response
ledger state and value objects used by the GGUF and safetensors loops.
Route/SSE conversion, tool execution, and model streaming stay in the
backend-specific modules.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence
from urllib.parse import urlparse

from core.inference.tool_call_parser import (
    TOOL_ERROR_NUDGE,
    TOOL_ERROR_PREFIXES,
    sanitize_control_chars,
)


_CANONICAL_HEAL_ARG = {
    "python": "code",
    "terminal": "command",
    "render_html": "code",
}
_ONE_SHOT_TOOLS = frozenset({"render_html"})

NoopReason = Literal["duplicate", "disabled", "render_html_repeat"]
ToolAction = Literal["execute", "duplicate", "disabled", "render_html_repeat"]


@dataclass(frozen = True)
class CoercedArguments:
    """Normalized tool arguments plus whether healing changed the shape."""

    arguments: dict[str, Any]
    healed: bool = False


@dataclass(frozen = True)
class ToolCallDecision:
    """Decision made before any visible tool event is emitted."""

    action: ToolAction
    tool_name: str
    arguments: dict[str, Any]
    tool_call_id: str = ""
    key: str = ""
    provenance: dict[str, Any] = field(default_factory = dict)
    status_text: str = ""
    noop_result: str = ""

    @property
    def should_execute(self) -> bool:
        return self.action == "execute"

    @property
    def emit_visible_events(self) -> bool:
        """Only real executions should become frontend-visible tool cards."""
        return self.should_execute

    @property
    def noop_reason(self) -> NoopReason | None:
        if self.action == "execute":
            return None
        return self.action

    def tool_start_payload(self) -> dict[str, Any]:
        """Build the payload fields for a real tool_start event."""
        return {
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "arguments": self.arguments,
            "provenance": self.provenance,
        }

    def tool_start_event(self) -> dict[str, Any]:
        """Build the existing backend event shape for a real execution."""
        return {"type": "tool_start", **self.tool_start_payload()}

    def as_assistant_tool_call(self) -> dict[str, Any]:
        """Return an OpenAI-style tool_call with normalized arguments."""
        tool_call: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": self.tool_name,
                "arguments": json.dumps(
                    self.arguments,
                    ensure_ascii = False,
                    sort_keys = True,
                    separators = (",", ":"),
                ),
            },
        }
        if self.tool_call_id:
            tool_call["id"] = self.tool_call_id
        return tool_call


@dataclass(frozen = True)
class ToolCallCompletion:
    """Result/nudge that should be fed back to the next model turn."""

    decision: ToolCallDecision
    result: str
    is_error: bool = False
    executed: bool = False

    def tool_end_payload(self) -> dict[str, Any]:
        """Build the payload fields for a real tool_end event."""
        return {
            "tool_name": self.decision.tool_name,
            "tool_call_id": self.decision.tool_call_id,
            "result": self.result,
            "provenance": self.decision.provenance,
        }

    def tool_end_event(self) -> dict[str, Any]:
        """Build the existing backend event shape for a real execution result."""
        return {"type": "tool_end", **self.tool_end_payload()}

    def tool_message(self) -> dict[str, Any]:
        """Return the OpenAI-compatible tool message for a real execution."""
        if not self.executed:
            raise ValueError("No-op completions are internal nudges, not tool messages")
        return self.model_message()

    def model_message(self) -> dict[str, Any]:
        """Return the internal message appended before the next generation.

        Executed calls keep the existing OpenAI-compatible ``role=tool``
        continuation. No-op controller decisions are not real tool output, so
        they are fed back as a hidden user nudge rather than a normal tool
        result.
        """
        if not self.executed:
            return {"role": "user", "content": self.result}

        content = strip_result_for_model(self.result)
        if self.is_error:
            content = content + TOOL_ERROR_NUDGE
        message: dict[str, Any] = {
            "role": "tool",
            "name": self.decision.tool_name,
            "content": content,
        }
        if self.decision.tool_call_id:
            message["tool_call_id"] = self.decision.tool_call_id
        return message


@dataclass(frozen = True)
class _ToolCallRecord:
    key: str
    is_error: bool
    executed: bool
    action: ToolAction


def _json_default(value: Any) -> str:
    return str(value)


def canonical_tool_call_key(tool_name: str, arguments: Mapping[str, Any]) -> str:
    """Return a stable key for duplicate detection."""
    canonical_args = json.dumps(
        dict(arguments),
        ensure_ascii = False,
        sort_keys = True,
        separators = (",", ":"),
        default = _json_default,
    )
    return f"{tool_name}:{canonical_args}"


def coerce_tool_arguments(
    raw_args: Any,
    *,
    heal: bool,
    tool_name: str = "",
) -> CoercedArguments:
    """Normalize model-emitted ``function.arguments`` to a dictionary."""
    if isinstance(raw_args, Mapping):
        return CoercedArguments(dict(raw_args), False)
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            if isinstance(parsed, Mapping):
                return CoercedArguments(dict(parsed), False)
        except (json.JSONDecodeError, ValueError):
            pass
        if heal:
            key = _CANONICAL_HEAL_ARG.get(tool_name, "query")
            return CoercedArguments({key: raw_args}, True)
        return CoercedArguments({"raw": raw_args}, False)
    return CoercedArguments({}, False)


def tool_event_provenance(**flags: object) -> dict[str, object]:
    """Return provenance metadata with falsey flags omitted."""
    provenance: dict[str, object] = {"source": "local"}
    for key, value in flags.items():
        if value is not None and value is not False:
            provenance[key] = value
    return provenance


def status_for_tool(tool_name: str, arguments: Mapping[str, Any]) -> str:
    """Return the status text already used by local tool streams."""
    if tool_name == "web_search":
        url = str(arguments.get("url") or "").strip()
        if url:
            parsed = urlparse(url)
            if parsed.scheme in ("http", "https") and parsed.hostname:
                host = parsed.hostname
                if host.startswith("www."):
                    host = host[4:]
                return f"Reading: {host}"
            return "Reading page..."
        return f"Searching: {arguments.get('query', '')}"
    if tool_name == "python":
        preview = str(arguments.get("code") or "").strip().split("\n")[0][:60]
        return f"Running Python: {preview}" if preview else "Running Python..."
    if tool_name == "terminal":
        preview = str(arguments.get("command") or "")[:60]
        return f"Running: {preview}" if preview else "Running command..."
    return f"Calling: {tool_name}"


def is_tool_error(result: str) -> bool:
    return isinstance(result, str) and result.lstrip().startswith(TOOL_ERROR_PREFIXES)


def _strip_mcp_image_suffix(result: str) -> str:
    """Drop a trailing __MCP_IMAGES__ envelope only when it is the valid JSON
    image array appended by _flatten_result, so legit tool text that merely
    mentions the marker is not truncated."""
    head, sep, payload = result.rpartition("\n__MCP_IMAGES__:")
    if not sep:
        return result
    try:
        images = json.loads(payload)
    except (ValueError, RecursionError):
        return result
    if not isinstance(images, list) or not images:
        return result
    if not all(
        isinstance(img, dict)
        and isinstance(img.get("data"), str)
        and isinstance(img.get("mimeType"), str)
        for img in images
    ):
        return result
    return head.rstrip()


def strip_result_for_model(result: str) -> str:
    """Remove frontend-only sentinels (image paths, RAG source map) before
    feeding the result back to the model."""
    result = _strip_mcp_image_suffix(result)
    for sentinel in ("__IMAGES__:", "__RAG_SOURCES__:"):
        if sentinel in result:
            result = result.split(sentinel, 1)[0].rstrip()
    return result


def append_deferred_nudges(conversation: list, msgs: Sequence[dict]) -> None:
    """Append a batch's no-op nudges as one deduped ``role=user`` message.

    Deferred to after the batch's tool results so a no-op never splits an
    assistant's ``tool_calls`` from their ``role=tool`` results.
    """
    contents = list(dict.fromkeys(msg["content"] for msg in msgs))
    if contents:
        conversation.append({"role": "user", "content": "\n\n".join(contents)})


def _tool_name_from_schema(tool: Mapping[str, Any]) -> str:
    function = tool.get("function")
    if not isinstance(function, Mapping):
        return ""
    name = function.get("name")
    return str(name or "")


def _noop_result(reason: NoopReason, tool_name: str) -> str:
    if reason == "duplicate":
        return (
            f"One earlier request to call tool '{tool_name}' in this batch was "
            "not executed because an identical call had already completed "
            "successfully. Do not repeat the same "
            "tool call. Continue with a different enabled tool if that would "
            "materially help, or provide the final answer if you have enough "
            "information."
        )
    if reason == "render_html_repeat":
        return (
            "render_html completed successfully earlier in this assistant "
            "response. Do not call render_html again unless the user asks for "
            "changes. Do not mention this internal instruction. Provide only "
            "the requested final note or answer."
        )
    return (
        f"One earlier request to call tool '{tool_name}' in this batch was "
        "not executed because that tool is not enabled for this request. Provide the "
        "final answer now without calling more tools."
    )


class ToolLoopController:
    """Per-response ledger for local agentic tool loops."""

    def __init__(
        self,
        *,
        tools: Sequence[Mapping[str, Any]] | None,
        auto_heal_tool_calls: bool = True,
        one_shot_tools: frozenset[str] = _ONE_SHOT_TOOLS,
        duplicate_noop_limit: int = 2,
    ) -> None:
        self._restrict_to_allowed = tools is not None
        self._tools = [copy.deepcopy(dict(tool)) for tool in (tools or [])]
        self._allowed_tool_names = {
            name for name in (_tool_name_from_schema(tool) for tool in self._tools) if name
        }
        self._auto_heal_tool_calls = auto_heal_tool_calls
        self._one_shot_tools = one_shot_tools
        self._completed_one_shot_tools: set[str] = set()
        self._successful_keys: set[str] = set()
        self._duplicate_noop_counts: dict[str, int] = {}
        self._duplicate_noop_limit = max(1, duplicate_noop_limit)
        self._history: list[_ToolCallRecord] = []
        self._force_final_answer = False

    @property
    def history(self) -> tuple[_ToolCallRecord, ...]:
        return tuple(self._history)

    @property
    def force_final_answer(self) -> bool:
        """True once a terminal no-op should transition to a no-tools pass."""
        return self._force_final_answer

    def active_tools(self) -> list[dict[str, Any]]:
        """Return tools still worth advertising to the next model call."""
        if self._force_final_answer:
            return []
        active: list[dict[str, Any]] = []
        for tool in self._tools:
            name = _tool_name_from_schema(tool)
            if name in self._completed_one_shot_tools:
                continue
            active.append(copy.deepcopy(tool))
        return active

    def prepare_call(
        self,
        tool_call: Mapping[str, Any],
        *,
        forced: bool = False,
        provisional: bool = False,
    ) -> ToolCallDecision:
        """Classify a parsed tool call before any visible event is yielded."""
        function = tool_call.get("function")
        function = function if isinstance(function, Mapping) else {}
        tool_name = str(function.get("name") or "").strip()
        coerced = coerce_tool_arguments(
            function.get("arguments", {}),
            heal = self._auto_heal_tool_calls,
            tool_name = tool_name,
        )
        key = canonical_tool_call_key(tool_name, coerced.arguments)
        provenance = tool_event_provenance(
            healed = coerced.healed,
            forced = forced,
            provisional = provisional,
        )
        action: ToolAction = "execute"
        noop = ""
        if tool_name in self._completed_one_shot_tools:
            action = "render_html_repeat"
            noop = _noop_result("render_html_repeat", tool_name)
        elif self._restrict_to_allowed and tool_name not in self._allowed_tool_names:
            action = "disabled"
            noop = _noop_result("disabled", tool_name)
        elif key in self._successful_keys:
            action = "duplicate"
            noop = _noop_result("duplicate", tool_name)

        return ToolCallDecision(
            action = action,
            tool_name = tool_name,
            arguments = coerced.arguments,
            tool_call_id = str(tool_call.get("id") or ""),
            key = key,
            provenance = provenance,
            status_text = status_for_tool(tool_name, coerced.arguments),
            noop_result = noop,
        )

    def record_result(self, decision: ToolCallDecision, result: Any) -> ToolCallCompletion:
        """Record a real tool execution and return model/frontend payload helpers."""
        result_text = result if isinstance(result, str) else str(result)
        # Scrub garbage a fetch/subprocess can leave, so it neither shows on the tool card
        # nor poisons the model's next prompt.
        result_text = sanitize_control_chars(result_text)
        failed = is_tool_error(result_text)
        self._history.append(
            _ToolCallRecord(
                key = decision.key,
                is_error = failed,
                executed = True,
                action = decision.action,
            )
        )
        if not failed:
            self._successful_keys.add(decision.key)
            if decision.tool_name in self._one_shot_tools:
                self._completed_one_shot_tools.add(decision.tool_name)
        return ToolCallCompletion(
            decision = decision,
            result = result_text,
            is_error = failed,
            executed = True,
        )

    def record_noop(self, decision: ToolCallDecision) -> ToolCallCompletion:
        """Record a controller no-op without creating visible tool output."""
        self._history.append(
            _ToolCallRecord(
                key = decision.key,
                is_error = False,
                executed = False,
                action = decision.action,
            )
        )
        if decision.action == "duplicate":
            duplicate_count = self._duplicate_noop_counts.get(decision.key, 0) + 1
            self._duplicate_noop_counts[decision.key] = duplicate_count
            if duplicate_count >= self._duplicate_noop_limit:
                self._force_final_answer = True
        elif decision.action in ("disabled", "render_html_repeat"):
            self._force_final_answer = True
        return ToolCallCompletion(
            decision = decision,
            result = decision.noop_result,
            is_error = False,
            executed = False,
        )
