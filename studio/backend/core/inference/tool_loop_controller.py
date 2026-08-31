# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared controller state for Unsloth local agentic tool loops.

This module is intentionally dependency-light: it owns only per-response
ledger state and value objects used by the GGUF and safetensors loops.
Route/SSE conversion, tool execution, and model streaming stay in the
backend-specific modules.
"""

from __future__ import annotations

import ast
import copy
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Collection, Literal, Mapping, Sequence
from urllib.parse import urlparse

from core.inference.tool_call_parser import TOOL_ERROR_NUDGE, TOOL_ERROR_PREFIXES


_CANONICAL_HEAL_ARG = {
    "python": "code",
    "terminal": "command",
    "render_html": "code",
    # Not derivable: web_search declares no REQUIRED argument, because a call carrying
    # only `url` fetches that page without searching. A bare string is still a query,
    # which is what the old catch-all default got right for this tool and only this one.
    "web_search": "query",
}

# Where a bare string lands when the tool it was sent to has no argument that could hold
# it. Read by `execute_tool`, which answers with what actually went wrong instead of
# letting the tool report a missing key it was never given.
UNPARSED_ARGUMENTS_KEY = "__unsloth_unparsed_arguments__"


# Anything that ends a JSON token. A tail free of all of them never finished arriving.
_JSON_STRUCTURAL = frozenset(',:{}[]" \t\n\r')


def _looks_like_broken_json(raw: str) -> bool:
    """Whether this text was MEANT to be a JSON object and stopped before finishing.

    Opening with a bracket is necessary but not sufficient, and treating it as sufficient
    was wrong in both directions: `{not json at all` is a bare string a model sent for a
    single-argument tool, and healing it into that argument is the right answer, while
    `{"code":"html = ...` is a call cut off mid-stream that must not become the program.

    What separates them is WHERE the decode fails. A call that was cut off runs out of
    input -- either inside a string that never closes, or at the very end of what arrived.
    Text that merely opens with a brace fails earlier, with input still to go.
    """
    text = raw.strip()
    if not text.startswith(("{", "[")):
        return False
    try:
        json.loads(text)
    except json.JSONDecodeError as error:
        if error.msg.startswith("Unterminated string") or error.pos >= len(text):
            return True
        # A value cut mid-token reports at the token's START, not at the end of input, so
        # the two tests above miss `{"flag":tru` (Expecting value) and `{"n":1e`
        # (Expecting ',' delimiter). What they share is that everything from the failure
        # to the end is one unfinished token: no delimiter, no quote, no space.
        #
        # Excluded: a bad PROPERTY NAME. After `{` a bare word is malformed rather than
        # cut off, which is what keeps `{oops` and `{not json at all` healable.
        if error.msg.startswith("Expecting property name"):
            return False
        # Excluded for the opposite reason: a COMPLETE document with something after it.
        # `{"a": 1} trailing` decodes fully and then finds junk, so nothing was lost.
        if error.msg.startswith("Extra data"):
            return False
        remainder = text[error.pos :]
        return bool(remainder) and not any(ch in _JSON_STRUCTURAL for ch in remainder)
    return False


def _single_string_argument(function: Mapping) -> "str | None":
    """That function's one required string argument, or None when it has anything else."""
    parameters = function.get("parameters")
    if not isinstance(parameters, Mapping):
        return None
    properties = parameters.get("properties")
    required = parameters.get("required")
    if not isinstance(properties, Mapping) or not isinstance(required, list):
        return None
    required_names = [item for item in required if isinstance(item, str)]
    if len(required_names) != 1:
        return None
    key = required_names[0]
    schema = properties.get(key)
    if not isinstance(schema, Mapping):
        return None
    kind = schema.get("type")
    if kind == "string" or (isinstance(kind, list) and "string" in kind):
        return key
    return None


def _healable_keys_from_schemas() -> "dict[str, str]":
    """Built-in tool -> its single required string argument, for the tools that have one.

    Derived from the schemas rather than hand-listed. The map above was hand-kept, so it
    silently went stale the moment a tool was added: `edit_file` landed with three
    required arguments and no entry, and every unparseable call to it was healed into a
    "query" key that only exists on the search tools. A schema-derived answer cannot rot
    the same way -- a new tool is either single-string and healable, or it is not and says
    so.
    """
    try:
        from core.inference.tools import ALL_TOOLS  # noqa: PLC0415 -- cycle at import time
    except Exception:  # noqa: BLE001 -- healing must never break a chat
        return {}
    return _healable_keys_from(ALL_TOOLS)


def _healable_keys_from(tools) -> "dict[str, str]":
    keys: dict[str, str] = {}
    for tool in tools or []:
        function = tool.get("function") if isinstance(tool, Mapping) else None
        if not isinstance(function, Mapping):
            continue
        name = str(function.get("name") or "")
        if not name:
            continue
        key = _single_string_argument(function)
        if key is not None:
            keys[name] = key
    return keys


_HEAL_ARG_CACHE: "dict[str, str] | None" = None


def _heal_arg_key(tool_name: str, tool_schemas = None) -> "str | None":
    """The argument a bare string should become, or None when there isn't one.

    ``tool_schemas`` is the REQUEST's tool array. MCP tools are discovered at runtime and
    so are absent from `ALL_TOOLS`; without them an MCP tool with one required string
    argument lost the auto-healing every other single-string tool has, and a bare string
    reached `execute_tool` as the unparsed sentinel instead.
    """
    global _HEAL_ARG_CACHE
    if tool_name in _CANONICAL_HEAL_ARG:
        return _CANONICAL_HEAL_ARG[tool_name]
    if _HEAL_ARG_CACHE is None:
        _HEAL_ARG_CACHE = _healable_keys_from_schemas()
    key = _HEAL_ARG_CACHE.get(tool_name)
    if key is not None or not tool_schemas:
        return key
    # Not cached: the request's tools belong to the request, and caching them by name
    # would let one chat's MCP server decide another chat's healing.
    return _healable_keys_from(tool_schemas).get(tool_name)


def _unreadable_arguments_summary(fragment: str) -> dict[str, str]:
    """What stands in for a call whose arguments never finished arriving.

    Small and parseable on purpose. The replayed `arguments` of an assistant tool call is
    read as JSON by the server rendering the template, so anything that does not parse
    fails the whole request rather than just that call.
    """
    return {
        "error": (f"arguments were cut off after {len(fragment)} characters and could not be read")
    }


_ONE_SHOT_TOOLS = frozenset({"render_html"})

NoopReason = Literal["duplicate", "disabled", "forced_mismatch", "render_html_repeat"]
ToolAction = Literal["execute", "duplicate", "disabled", "forced_mismatch", "render_html_repeat"]


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

    @property
    def unparsed_fragment(self) -> "str | None":
        """The raw text of a call whose JSON could not be read, if this is one.

        `UNPARSED_ARGUMENTS_KEY` is plumbing between the coercion and `execute_tool`, and
        it escaped into both places a caller looks: the tool card showed the user
        `{"__unsloth_unparsed_arguments__": ...}`, and the replayed assistant turn taught
        the model a key no tool declares. Both boundaries go through here instead.
        """
        if isinstance(self.arguments, Mapping) and UNPARSED_ARGUMENTS_KEY in self.arguments:
            return str(self.arguments.get(UNPARSED_ARGUMENTS_KEY) or "")
        return None

    def tool_start_payload(self) -> dict[str, Any]:
        """Build the payload fields for a real tool_start event."""
        fragment = self.unparsed_fragment
        # `raw` is the shape this module already uses for arguments it could not read into
        # a schema, so the card shows the model's own text under a name that means
        # something rather than an internal sentinel.
        arguments = {"raw": fragment} if fragment is not None else self.arguments
        return {
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "arguments": arguments,
            "provenance": self.provenance,
        }

    def tool_start_event(self) -> dict[str, Any]:
        """Build the existing backend event shape for a real execution."""
        return {"type": "tool_start", **self.tool_start_payload()}

    def as_assistant_tool_call(self) -> dict[str, Any]:
        """Return an OpenAI-style tool_call with normalized arguments."""
        fragment = self.unparsed_fragment
        tool_call: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": self.tool_name,
                # Whatever goes here MUST parse as JSON. llama-server parses it while
                # rendering the template and answers 500 otherwise, which is what replaying
                # the fragment verbatim caused: it is unparseable by definition, that being
                # why it is here. So the replay is a short valid object that says the call
                # was cut off, and the tool result carries the detail. The fragment itself
                # is not worth resending -- it is the content that overflowed the window.
                "arguments": json.dumps(_unreadable_arguments_summary(fragment))
                if fragment is not None
                else json.dumps(
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

        content = strip_result_for_model(self.result, self.decision.tool_name)
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


# "0"/"1" are left out: a native `0` arrives already typed, so they would mean two things.
_SCHEMA_TRUE_WORDS = frozenset({"true", "yes"})
_SCHEMA_FALSE_WORDS = frozenset({"false", "no"})
# Not ValueErrors, so a decode-shaped except would let deep model output escape as a 500.
_DECODE_ERRORS = (ValueError, RecursionError)
_LITERAL_ERRORS = (*_DECODE_ERRORS, SyntaxError, MemoryError)


# A JSON string, open or closed; group 1 is the closing quote, which `endswith` cannot stand
# in for because an open string can end on an escaped one. The `\?$` tail stops a started
# match from ever failing, which would send `finditer` back over every later quote.
_JSON_STRING_RE = re.compile(r'"(?:[^"\\]|\\.)*(?:(")|\\?$)', re.S)
_JSON_CLOSER = {"[": "]", "{": "}"}


def _balanced(segment: str, opened: "list[str]") -> str:
    kept: list[str] = []
    for ch in segment:
        if ch in _JSON_CLOSER:
            opened.append(_JSON_CLOSER[ch])
            kept.append(ch)
        elif ch in "]}":
            # The commonest slip: a closer of the wrong kind becomes the right one.
            if opened:
                kept.append(opened.pop())
        else:
            kept.append(ch)
    return "".join(kept)


def _repair_json_value(text: str) -> Any:
    """Parse near-valid JSON whose brackets do not balance, or None if it still will not."""
    parts: list[str] = []
    opened: list[str] = []
    cursor = 0
    open_string = ""
    for match in _JSON_STRING_RE.finditer(text):
        parts += [_balanced(text[cursor : match.start()], opened), match.group()]
        open_string = "" if match.group(1) else '"'
        cursor = match.end()
    parts += [_balanced(text[cursor:], opened), open_string, *reversed(opened)]
    try:
        return json.loads("".join(parts), strict = False)
    except _DECODE_ERRORS:
        return None


# Followed to find a declaration; `nullable` is OpenAPI 3.0's spelling of a type union.
_READ_KEYWORDS = frozenset(
    {
        "type",
        "nullable",
        "properties",
        "items",
        "prefixItems",
        "additionalItems",
        "additionalProperties",
    }
)
# Cannot move a declaration out of reach: annotations, and constraints that only reject.
_INERT_KEYWORDS = frozenset(
    {
        "title",
        "description",
        "default",
        "examples",
        "example",
        "deprecated",
        "readOnly",
        "writeOnly",
        "format",
        "$comment",
        "$schema",
        "$id",
        "$anchor",
        "$defs",
        "definitions",
        "enum",
        "const",
        "required",
        "dependentRequired",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minContains",
        "maxContains",
        "minProperties",
        "maxProperties",
    }
)
_UNION_KEYWORDS = frozenset({"anyOf", "oneOf"})
# Matched exactly: an unknown type name is as unreadable as an unknown keyword.
_JSON_SCHEMA_TYPES = frozenset(
    {
        "array",
        "boolean",
        "integer",
        "null",
        "number",
        "object",
        "string",
    }
)
_KNOWN_KEYWORDS = _READ_KEYWORDS | _INERT_KEYWORDS | _UNION_KEYWORDS


def _readable(spec: Any) -> bool:
    """An allowlist, because composition, reference and a later draft's keywords can all move
    a declaration out of this walk's reach."""
    return isinstance(spec, Mapping) and spec.keys() <= _KNOWN_KEYWORDS


def _read_schema(spec: Any) -> "tuple[Any, str | None, bool]":
    """The subschema to read against, its one declared type, and whether it admits null;
    ``(None, ...)`` leaves it alone. A union collapses to its single non-null branch, so
    every branch must name one: reading the integer branch of ``anyOf: [{integer}, {$ref}]``
    would turn ``"001"`` into 1."""
    if not _readable(spec):
        return None, None, False
    union = _UNION_KEYWORDS & spec.keys()
    if union and (len(union) > 1 or not _READ_KEYWORDS.isdisjoint(spec)):
        return None, None, False
    kind = spec.get("type")
    chosen = spec
    if isinstance(kind, str):
        named = [kind]
    elif isinstance(kind, list):
        named = list(kind)
    elif union:
        branches = spec.get("anyOf") or spec.get("oneOf")
        if not isinstance(branches, list) or not branches:
            return None, None, False
        named = []
        for branch in branches:
            name = branch.get("type") if _readable(branch) else None
            if not isinstance(name, str) or _UNION_KEYWORDS & branch.keys():
                return None, None, False
            named.append(name)
            if named[-1] != "null":
                chosen = branch
    elif kind is None:
        return spec, None, False
    else:
        return None, None, False
    if not named or not all(isinstance(name, str) and name in _JSON_SCHEMA_TYPES for name in named):
        return None, None, False
    if chosen.get("nullable") is True:
        named.append("null")
    rest = [k for k in named if k != "null"]
    if len(rest) > 1 or (not rest and "null" not in named):
        return None, None, False
    return chosen, rest[0] if rest else None, "null" in named


# A schema is model-facing data, so its nesting is not trusted to be shallow.
_MAX_SCHEMA_DEPTH = 8


def _coerce_declared_value(text: str, declared: str, repair: bool) -> Any:
    stripped = text.strip()
    if declared == "boolean":
        lowered = stripped.lower()
        if lowered in _SCHEMA_TRUE_WORDS:
            return True
        if lowered in _SCHEMA_FALSE_WORDS:
            return False
    elif declared in ("integer", "number"):
        try:
            # float() would round 9007199254740993, which an already-numeric argument keeps.
            return int(stripped)
        except ValueError:
            pass
        try:
            number = float(stripped)
        except (ValueError, OverflowError):
            return text
        # "nan"/"inf" parse, but json.dumps writes them bare and the client rejects that.
        # Exactness is unguarded: float64 IS JSON's number type, so a JSON call agrees.
        if math.isfinite(number) and (declared == "number" or number.is_integer()):
            return number
    elif declared == "array":
        return _coerce_container(text, list, repair)
    elif declared == "object":
        return _coerce_container(text, dict, repair)
    return text


def _usable_container(value: Any, want: type) -> bool:
    """A ``want`` that survives the JSON re-encoding of ``arguments``: both decoders admit
    what ``json.dumps`` will not write back, such as an integer key returning as a STRING."""
    if not isinstance(value, want):
        return False
    try:
        dumped = json.dumps(value, allow_nan = False, sort_keys = True, ensure_ascii = False)
        dumped.encode("utf-8")  # a lone surrogate survives dumps and dies encoding the reply
        return json.loads(dumped) == value
    except (TypeError, ValueError, RecursionError):
        return False


def _coerce_container(text: str, want: type, repair: bool) -> Any:
    """``text`` read as the DECLARED container, tolerating a Python literal and, when
    ``repair``, unbalanced brackets. Rewriting brackets is what auto-heal opts out of."""
    try:
        parsed = json.loads(text, strict = False)
    except _DECODE_ERRORS:
        parsed = None
    if _usable_container(parsed, want):
        return parsed
    try:
        literal = ast.literal_eval(text)
    except _LITERAL_ERRORS:
        literal = None
    if isinstance(literal, tuple):
        literal = list(literal)
    if _usable_container(literal, want):
        return literal
    repaired = _repair_json_value(text) if repair else None
    return repaired if _usable_container(repaired, want) else text


def _coerce_by_property(value: Any, spec: Any, depth: int, repair: bool) -> Any:
    if depth >= _MAX_SCHEMA_DEPTH:
        return value
    spec, declared, nullable = _read_schema(spec)
    if spec is None:
        return value
    if isinstance(value, str):
        if declared == "string":
            return value
        if nullable and value.strip().lower() == "null":
            return None
        if declared is None:
            return value
        value = _coerce_declared_value(value, declared, repair)
    # Descent needs the SAME declaration the conversion needs: without one a text value is
    # not decoded, so descending into an already-decoded one would make the syntaxes disagree.
    if declared == "object" and isinstance(value, Mapping):
        nested = spec.get("properties")
        nested = nested if isinstance(nested, Mapping) else {}
        extra = spec.get("additionalProperties")
        extra = extra if isinstance(extra, Mapping) else None
        if nested or extra:
            return {
                k: _coerce_by_property(v, nested.get(k, extra), depth + 1, repair)
                for k, v in value.items()
            }
    elif declared == "array" and isinstance(value, list):
        items = spec.get("items")
        prefix = items if isinstance(items, list) else spec.get("prefixItems")
        if isinstance(prefix, list):
            # A schema per position: draft-07 tuple `items`, 2020-12 `prefixItems`.
            rest = spec.get("additionalItems") if isinstance(items, list) else items
            return [
                _coerce_by_property(v, prefix[i] if i < len(prefix) else rest, depth + 1, repair)
                for i, v in enumerate(value)
            ]
        if isinstance(items, Mapping):
            return [_coerce_by_property(v, items, depth + 1, repair) for v in value]
    return value


def coerce_arguments_by_schema(
    arguments: Mapping[str, Any],
    properties: Any,
    *,
    repair: bool = False,
) -> dict:
    """Arguments with each string value that declares a non-string type read as that type.

    A tool-call parser is given tool NAMES, never schemas, so an XML-form parameter is stored
    as raw text: ``replace_all`` reaches the tool as ``"false"``, and ``bool("false")`` is
    True. A container's text IS its JSON; a scalar's carries no type, so it is read only
    where its spelling names the declared type. Anything else keeps its text.
    """
    if not isinstance(properties, Mapping) or not properties:
        return dict(arguments)
    return {k: _coerce_by_property(v, properties.get(k), 0, repair) for k, v in arguments.items()}


def _declared_properties(tool_name: str, tool_schemas) -> Any:
    for tool in tool_schemas or []:
        function = tool.get("function") if isinstance(tool, Mapping) else None
        if not isinstance(function, Mapping) or function.get("name") != tool_name:
            continue
        parameters = function.get("parameters")
        return parameters.get("properties") if isinstance(parameters, Mapping) else None
    return None


def coerce_tool_arguments(
    raw_args: Any,
    *,
    heal: bool,
    tool_name: str = "",
    tool_schemas = None,
) -> CoercedArguments:
    """Normalize model-emitted ``function.arguments`` to a dictionary.

    Typing against ``tool_schemas`` is not gated on ``heal``: healing invents structure the
    model never sent, while reading a value as its schema declares it is the tool's contract.
    """
    properties = _declared_properties(tool_name, tool_schemas) if tool_name else None
    if isinstance(raw_args, Mapping):
        return CoercedArguments(
            coerce_arguments_by_schema(raw_args, properties, repair = heal), False
        )
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            if isinstance(parsed, Mapping):
                return CoercedArguments(
                    coerce_arguments_by_schema(parsed, properties, repair = heal), False
                )
        except (json.JSONDecodeError, ValueError):
            pass
        if heal:
            # Healing exists for a model that sends its ONE argument as a bare string
            # instead of an object. Text that opens like JSON and fails to parse is not
            # that -- it is a broken object, usually one cut off mid-stream, and wrapping
            # it whole becomes the argument's value. Observed on `python`, which has a
            # single `code` argument and so was healable: a truncated call arrived as
            # `{"code":"html = ...`, the entire fragment was passed as the PROGRAM, and the
            # model read its own file back as `{"code":"html = ...` and spent the rest of
            # the turn convinced the sandbox had mangled it. Same defect `edit_file` had;
            # having a single string argument only hid it.
            key = (
                None
                if _looks_like_broken_json(raw_args)
                else _heal_arg_key(tool_name, tool_schemas)
            )
            if key is not None:
                return CoercedArguments({key: raw_args}, True)
            # No single argument this text could be. Inventing one used to default to
            # "query", which edit_file -- three required arguments, none of them a
            # query -- then reported as "'old_string' and 'new_string' must both be
            # strings": a type error blaming the model for a key it never sent, on a
            # call whose real problem was that its JSON never finished arriving.
            return CoercedArguments({UNPARSED_ARGUMENTS_KEY: raw_args}, False)
        return CoercedArguments({"raw": raw_args}, False)
    return CoercedArguments({}, False)


def tool_event_provenance(**flags: object) -> dict[str, object]:
    """Return provenance metadata with falsey flags omitted."""
    provenance: dict[str, object] = {"source": "local"}
    for key, value in flags.items():
        if value is not None and value is not False:
            provenance[key] = value
    return provenance


def mcp_display_parts(tool_name: str) -> "tuple[str, str] | None":
    """(server display name, bare tool name) for a resolvable mcp__ tool."""
    if not tool_name.startswith("mcp__"):
        return None
    parts = tool_name.split("__", 2)
    if len(parts) < 3 or not parts[1] or not parts[2]:
        return None
    try:
        from storage import mcp_servers_db
        server = mcp_servers_db.get_server(parts[1])
    except Exception:  # noqa: BLE001
        return None
    display = (server or {}).get("display_name")
    return (str(display), parts[2]) if display else None


def provisional_tool_provenance(tool_name: str) -> dict[str, object]:
    """Provisional-card provenance, MCP display name included so an early or
    orphaned card (cancel/error before the real tool_start) never shows the id."""
    mcp = mcp_display_parts(tool_name)
    return tool_event_provenance(
        provisional = True,
        mcp_server = mcp[0] if mcp else None,
    )


def status_for_tool(tool_name: str, arguments: Mapping[str, Any]) -> str:
    """Return the status text already used by local tool streams."""
    if tool_name == "web_search":
        url = str(arguments.get("url") or "").strip()
        if url:
            # Bare hosts are fetched as https, so normalize first or the badge
            # stays generic for exactly the URLs the fetch layer accepts.
            from core.inference.tools import _normalize_url_scheme

            try:
                parsed = urlparse(_normalize_url_scheme(url))
            except ValueError:
                # Runs in prepare_call, outside the fetch's exception handler:
                # raising here kills the turn instead of returning "Blocked:".
                return "Reading page..."
            if parsed.scheme in ("http", "https") and parsed.hostname:
                host = parsed.hostname
                if host.startswith("www."):
                    host = host[4:]
                return f"Reading: {host}"
            return "Reading page..."
        image_queries = arguments.get("image_queries")
        images = (
            ", ".join(str(q) for q in image_queries[:5])
            if isinstance(image_queries, list) and image_queries
            else ""
        )
        query = str(arguments.get("query") or "").strip()
        if images and not query:
            return f"Finding images: {images}"
        if images:
            return f"Searching: {query} (images: {images})"
        return f"Searching: {arguments.get('query', '')}"
    if tool_name == "python":
        preview = str(arguments.get("code") or "").strip().split("\n")[0][:60]
        return f"Running Python: {preview}" if preview else "Running Python..."
    if tool_name == "terminal":
        preview = str(arguments.get("command") or "")[:60]
        return f"Running: {preview}" if preview else "Running command..."
    if tool_name == "edit_file":
        # The name, not the patch: the tool card below already shows the edit.
        path = str(arguments.get("path") or "").strip()
        name = path.replace("\\", "/").rstrip("/").rpartition("/")[2]
        return f"Editing: {name}" if name else "Editing file..."
    mcp = mcp_display_parts(tool_name)
    if mcp:
        return f"Calling: {mcp[0]} · {mcp[1]}"
    return f"Calling: {tool_name}"


def awaiting_approval_status(tool_name: str) -> str:
    """Status text for a call parked on the approval prompt.

    It has not started, so reporting "Running ..." with a climbing timer reads
    as a hang.
    """
    if tool_name == "python":
        return "Waiting for approval: Python"
    if tool_name == "terminal":
        return "Waiting for approval: command"
    if tool_name == "edit_file":
        return "Waiting for approval: file edit"
    mcp = mcp_display_parts(tool_name)
    if mcp:
        return f"Waiting for approval: {mcp[0]} · {mcp[1]}"
    return f"Waiting for approval: {tool_name}"


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


def _strip_files_sentinel(result: str) -> str:
    """Drop a trailing ``__FILES__`` envelope, and only that.

    Validated rather than split on sight: a tool whose own output contains the
    literal text would otherwise lose everything after it.
    """
    marker = "\n__FILES__:"
    start = result.rfind(marker)
    if start == -1:
        return result
    payload_start = start + len(marker)
    end = result.find("\n__", payload_start)
    if end == -1:
        end = len(result)
    try:
        entries = json.loads(result[payload_start:end])
    except (ValueError, TypeError, RecursionError):
        return result
    # Every entry, not just the list: the executor emits {"name": str, "size":
    # int | None}, and anything else is a tool that happened to print the marker.
    if not isinstance(entries, list) or not all(_is_file_entry(e) for e in entries):
        return result
    return result[:start] + result[end:]


def _is_file_entry(entry: object) -> bool:
    return (
        isinstance(entry, dict)
        and isinstance(entry.get("name"), str)
        and bool(entry.get("name"))
        and (entry.get("size") is None or isinstance(entry.get("size"), int))
    )


# Only these emit the file envelope, and only their output is defused first. An
# MCP tool or a fetched page ending in a well-formed __FILES__ line is content,
# not an envelope, and stripping it would take that line away from the model.
_SANDBOX_TOOLS = frozenset({"python", "terminal"})


def strip_result_for_model(result: str, tool_name: "str | None" = None) -> str:
    """Remove frontend-only sentinels (image paths, RAG source map) before
    feeding the result back to the model."""
    if tool_name is None or tool_name == "web_search":
        from .search_images import strip_images_suffix
        result = strip_images_suffix(result)
    result = _strip_mcp_image_suffix(result)
    if tool_name is None or tool_name in _SANDBOX_TOOLS:
        result = _strip_files_sentinel(result)
    for sentinel in ("__IMAGES__:", "__RAG_SOURCES__:"):
        if sentinel in result:
            result = result.split(sentinel, 1)[0].rstrip()
    return result


def deferred_nudge_text(msgs: Sequence[dict]) -> str:
    """Join a batch's no-op nudges into one deduped body.

    Callers that keep the feedback inside the tool exchange need the text
    without the ``role=user`` wrapper, so both forms stay in sync here.
    """
    return "\n\n".join(dict.fromkeys(msg["content"] for msg in msgs))


def append_deferred_nudges(conversation: list, msgs: Sequence[dict]) -> None:
    """Append a batch's no-op nudges as one deduped ``role=user`` message.

    Deferred to after the batch's tool results so a no-op never splits an
    assistant's ``tool_calls`` from their ``role=tool`` results.
    """
    if msgs:
        conversation.append({"role": "user", "content": deferred_nudge_text(msgs)})


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
    if reason == "forced_mismatch":
        return (
            f"One earlier request to call tool '{tool_name}' in this batch was "
            "not executed because it does not match the required tool choice. "
            "Call the required tool instead."
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
        allowed_tool_names: Collection[str] | None = None,
    ) -> ToolCallDecision:
        """Classify a parsed tool call before any visible event is yielded."""
        function = tool_call.get("function")
        function = function if isinstance(function, Mapping) else {}
        tool_name = str(function.get("name") or "").strip()
        coerced = coerce_tool_arguments(
            function.get("arguments", {}),
            heal = self._auto_heal_tool_calls,
            tool_name = tool_name,
            tool_schemas = self._tools,
        )
        key = canonical_tool_call_key(tool_name, coerced.arguments)
        mcp = mcp_display_parts(tool_name)
        provenance = tool_event_provenance(
            healed = coerced.healed,
            forced = forced,
            provisional = provisional,
            mcp_server = mcp[0] if mcp else None,
        )
        action: ToolAction = "execute"
        noop = ""
        if tool_name in self._completed_one_shot_tools:
            action = "render_html_repeat"
            noop = _noop_result("render_html_repeat", tool_name)
        elif allowed_tool_names is not None and tool_name not in allowed_tool_names:
            action = "forced_mismatch"
            noop = _noop_result("forced_mismatch", tool_name)
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
