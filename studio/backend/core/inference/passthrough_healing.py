# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Turns text tool calls into structured ones for declared tools only, response side; never deletes
unmatched output; response_format requests untouched; auto_heal_tool_calls false and
UNSLOTH_DISABLE_TOOL_CALL_HEALING=1 opt out"""

import bisect
import json
import os
import re
from collections.abc import Mapping
from typing import Any, Callable, NamedTuple, Optional

from core.inference.chat_template_helpers import _template_strings
from core.inference.tool_call_parser import (
    _ATTR_FUNC_OPEN_RE,
    _MINIMAX_INVOKE_CLOSE,
    _MINIMAX_INVOKE_OPEN_RE,
    TOOL_XML_SIGNALS,
    _MinimaxValueIndex,
    _parse_deepseek_tool_calls,
    _parse_glm_tool_calls,
    _parse_kimi_tool_calls,
    _parse_minimax_tool_calls,
)
from core.inference.tool_loop_controller import (
    coerce_arguments_by_schema,
    coerce_tool_arguments,
)
from core.tool_healing import (
    _MISTRAL_ARRAY_RE,
    _MISTRAL_BRACKET_RE,
    _THINK_OPEN_RE,
    _THINK_TAG_RE,
    parse_tool_calls_from_text,
)

# Only formats this parser can promote, narrower than TOOL_XML_SIGNALS: a loop-only marker would buffer a streamed call as prose unpromoted.
_HEAL_SIGNALS = (
    "<tool_call>",
    "<|tool_call>",
    "<function=",
    "[TOOL_CALLS]",
    # TML Inkling native call marker (leaks as text when the server-side parser misses a narration-then-call turn).
    "<|content_invoke_tool_json|>",
)


def _has_heal_signal(text: str) -> bool:
    return any(s in text for s in _HEAL_SIGNALS)


# Read once at import (same convention as the other UNSLOTH_* switches).
_HEALING_DISABLED = os.environ.get("UNSLOTH_DISABLE_TOOL_CALL_HEALING", "0") == "1"
# Nudging is opt-in: per-request nudge_tool_calls=true or UNSLOTH_TOOL_CALL_NUDGE=1 by default.
_NUDGE_DEFAULT = os.environ.get("UNSLOTH_TOOL_CALL_NUDGE", "0") == "1"


def response_format_constrains_decoding(response_format: Any) -> bool:
    return response_format is not None and response_format != {"type": "text"}


def nudge_enabled(request_flag: Optional[bool], *, response_format: Any = None) -> bool:
    if response_format_constrains_decoding(response_format):
        return False
    return _NUDGE_DEFAULT if request_flag is None else bool(request_flag)


_MAX_SIGNAL_LEN = max(len(s) for s in _HEAL_SIGNALS)
# Hold limit: a suspected tool block longer than this is declared a false alarm and flushed as text, bounding memory on a rambling model.
_MAX_HOLD_CHARS = 64 * 1024
# Each closer a native block fails to parse at costs a reparse of the whole block.
_MAX_NATIVE_CLOSERS = 16


def heal_gate(
    auto_heal: Optional[bool],
    tools: Optional[list],
    tool_choice: Any = None,
    *,
    response_format: Any = None,
) -> Optional[set]:
    """Declared tool names, the promotion allowlist so healing never invents a tool; none under tool_choice none or a constraining response_format."""
    if _HEALING_DISABLED or auto_heal is False:
        return None
    if tool_choice == "none":
        return None
    if response_format_constrains_decoding(response_format):
        return None
    names = set()
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.add(function["name"])
    if isinstance(tool_choice, dict):
        function = tool_choice.get("function")
        forced = function.get("name") if isinstance(function, dict) else None
        if isinstance(forced, str):
            names &= {forced}
    return names or None


def _tool_schemas_by_name(tools: Optional[list]) -> dict[str, Any]:
    schemas: dict[str, Any] = {}
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if isinstance(name, str):
            schemas[name] = function.get("parameters")
    return schemas


def _string_arg_key_from_schema(schema: Any) -> Optional[str]:
    if not isinstance(schema, dict):
        return None
    properties = schema.get("properties")
    required = schema.get("required")
    if not isinstance(properties, dict) or not isinstance(required, list):
        return None
    required_names = [name for name in required if isinstance(name, str)]
    if len(required_names) != 1:
        return None
    key = required_names[0]

    if key not in properties:
        return None
    prop_schema = properties.get(key)
    if isinstance(prop_schema, dict):
        prop_type = prop_schema.get("type")
        if isinstance(prop_type, list):
            if "string" not in prop_type:
                return None
        elif prop_type is not None and prop_type != "string":
            return None
    return key


def _coerce_promoted_arguments(
    raw_args: Any, tool_name: str, tool_schemas: Optional[dict]
) -> Optional[dict]:
    """Promoted args are typed to the schema; XML arrives as raw strings and truncation is healed here."""
    parameters = (tool_schemas or {}).get(tool_name)
    properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
    parsed = raw_args
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except (json.JSONDecodeError, ValueError):
            parsed = raw_args
    if isinstance(parsed, Mapping):
        return coerce_arguments_by_schema(parsed, properties, repair = True)
    if isinstance(raw_args, str):
        if tool_schemas is not None:
            key = _string_arg_key_from_schema(tool_schemas.get(tool_name))
            return {key: raw_args} if key else None
    coerced = coerce_tool_arguments(raw_args, heal = True, tool_name = tool_name)
    return coerced.arguments


def _promote(
    calls: list,
    allowed_tools: set,
    id_offset: int = 0,
    tool_schemas: Optional[dict] = None,
) -> list:
    """Ambiguous schemas keep the call as text rather than inventing a key for bare string arguments."""
    promoted = []
    for call in calls:
        function = call.get("function") if isinstance(call, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if name not in allowed_tools:
            continue
        arguments = _coerce_promoted_arguments(function.get("arguments"), name, tool_schemas)
        if arguments is None:
            continue
        promoted.append(
            {
                "id": f"call_{id_offset + len(promoted)}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii = False),
                },
            }
        )
    return promoted


class NativeToolFormat(NamedTuple):
    """A model's own tool-call block format. llama-server parses these itself, but an in-process
    model's output reaches the client healer raw; as there, only the template's format is healed."""

    open: str
    close: str
    call: re.Pattern
    parse: Callable[..., list]
    # GLM shares <tool_call> with Hermes; its own blocks have this shape.
    shape: Optional[re.Pattern] = None


def _loaded_arguments(call: dict) -> Any:
    arguments = call["function"].get("arguments")
    try:
        return json.loads(arguments) if isinstance(arguments, str) else arguments
    except ValueError:
        return arguments


def _string_leaves(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _string_leaves(key)
            yield from _string_leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _string_leaves(item)


_SEED_OPEN = "<seed:tool_call>"
_FOREIGN_SIGNALS = frozenset({*_HEAL_SIGNALS, *TOOL_XML_SIGNALS, _SEED_OPEN})


def _parse_seed_block(
    block: str,
    *,
    id_offset: int,
    allow_incomplete: bool = True,
) -> list:
    body = block[len(_SEED_OPEN) :]
    calls = parse_tool_calls_from_text(body, id_offset = id_offset, allow_incomplete = allow_incomplete)
    # Strict parsing reads a JSON function body as no arguments, so a disagreement is a misread.
    lenient = parse_tool_calls_from_text(body, id_offset = id_offset, allow_incomplete = True)
    if [_loaded_arguments(c) for c in calls] != [_loaded_arguments(c) for c in lenient]:
        return []
    return calls


def _parse_minimax_block(
    block: str,
    *,
    id_offset: int,
    allow_incomplete: bool = True,
) -> list:
    calls = _parse_minimax_tool_calls(block, id_offset = id_offset, allow_incomplete = allow_incomplete)
    # The parser skips a body it cannot read, so a call with no arguments must have had none.
    inside = _MinimaxValueIndex(block).inside
    invokes = [m for m in _MINIMAX_INVOKE_OPEN_RE.finditer(block) if not inside(m.start())]
    for call, invoke in zip(calls, invokes) if len(calls) == len(invokes) else ():
        close = block.find(_MINIMAX_INVOKE_CLOSE, invoke.end())
        if (
            not _loaded_arguments(call)
            and block[invoke.end() : close if close >= 0 else None].strip()
        ):
            return []
    return calls


def _parse_glm_block(
    block: str,
    *,
    id_offset: int,
    allow_incomplete: bool = True,
) -> list:
    # Raw, so the client's schema decides a value's type: a string parameter keeps JSON-looking text.
    return _parse_glm_tool_calls(
        block, id_offset = id_offset, allow_incomplete = allow_incomplete, raw_values = True
    )


# Most specific marker first.
_NATIVE_TOOL_FORMATS = (
    (
        "<minimax:tool_call>",
        NativeToolFormat(
            "<minimax:tool_call>",
            "</minimax:tool_call>",
            re.compile(r"<invoke\s"),
            _parse_minimax_block,
        ),
    ),
    (
        _SEED_OPEN,
        NativeToolFormat(
            _SEED_OPEN, "</seed:tool_call>", re.compile("<function="), _parse_seed_block
        ),
    ),
    (
        "<|tool_calls_section_begin|>",
        NativeToolFormat(
            "<|tool_calls_section_begin|>",
            "<|tool_calls_section_end|>",
            re.compile(re.escape("<|tool_call_begin|>")),
            _parse_kimi_tool_calls,
        ),
    ),
    (
        "<｜tool▁calls▁begin｜>",
        NativeToolFormat(
            "<｜tool▁calls▁begin｜>",
            "<｜tool▁calls▁end｜>",
            re.compile("<｜tool▁call▁begin｜>"),
            _parse_deepseek_tool_calls,
        ),
    ),
    (
        "<arg_key>",
        NativeToolFormat(
            "<tool_call>",
            "</tool_call>",
            re.compile("<tool_call>"),
            _parse_glm_block,
            # Anything but arg pairs after the name is a body the parser would skip.
            re.compile(r"<tool_call>\s*[\w.\-]+\s*(?:<arg_key>|</tool_call>)"),
        ),
    ),
)


_FORCED_CALL_OPENERS = {
    "hermes": ('<tool_call>\n{"name":', lambda name: f' {json.dumps(name)}, "arguments":'),
    "qwen3_coder": ("<tool_call>\n<function=", lambda name: f"{name}>\n"),
    "gemma4": ("<|tool_call>call:", lambda name: f"{name}{{"),
}


def forced_call_prefix(names: Optional[set], *templates: Any) -> str:
    """Opening of the model's own call markup for a forced tool_choice (in-process engines have no
    grammar to force one); empty unless every template body with a call format agrees on it."""
    from core.inference.engine_adapters import tool_parser_for_template

    parsers = {
        "gemma4" if "<|tool_call>call:" in body else tool_parser_for_template(body, "vllm")
        for template in templates
        for body in _template_strings(template)
    } - {None}
    opener = _FORCED_CALL_OPENERS.get(parsers.pop()) if len(parsers) == 1 else None
    if not names or opener is None:
        return ""
    head, named = opener
    return head + named(next(iter(names))) if len(names) == 1 else head


def native_tool_format(*templates: Any) -> Optional[NativeToolFormat]:
    """The block format the model's chat template writes tool calls in, if it is one of these."""
    bodies = [body for template in templates for body in _template_strings(template)]
    for marker, fmt in _NATIVE_TOOL_FORMATS:
        if any(marker in body for body in bodies):
            return fmt
    return None


def _native_block_calls(
    fmt: NativeToolFormat, block: str, allow_incomplete: bool
) -> Optional[list]:
    """The block's calls when each of its call markers parsed to exactly one call."""
    if fmt.shape is not None and not fmt.shape.match(block):
        return None
    try:
        calls = fmt.parse(block, id_offset = 0, allow_incomplete = allow_incomplete)
        leaves = [leaf for call in calls for leaf in _string_leaves(_loaded_arguments(call))]
    except RecursionError:
        return None

    # Markers inside a parsed argument are that argument's data; the rest must be one envelope.
    def bare(count: Callable[[str], int]) -> int:
        return count(block) - sum(map(count, leaves))

    if (
        not calls
        or bare(lambda s: len(fmt.call.findall(s))) != len(calls)
        or bare(lambda s: s.count(fmt.close)) != block.endswith(fmt.close)
        # Other formats' markup is refused even in arguments: decoded text does not locate it.
        or _ATTR_FUNC_OPEN_RE.search(block)
        or any(m in block for m in _FOREIGN_SIGNALS if m != fmt.open and not fmt.call.fullmatch(m))
    ):
        return None
    return calls


def _native_block(
    text: str, begin: int, fmt: NativeToolFormat, allow_incomplete: bool
) -> Optional[tuple]:
    close = text.find(fmt.close, begin + len(fmt.open))
    if close < 0:
        calls = _native_block_calls(fmt, text[begin:], True) if allow_incomplete else None
        return None if calls is None else (begin, len(text), calls)
    for _attempt in range(_MAX_NATIVE_CLOSERS):
        end = close + len(fmt.close)
        calls = _native_block_calls(fmt, text[begin:end], False)
        if calls is not None:
            return begin, end, calls
        # An argument may hold the closer, so the block may end at a later one.
        close = text.find(fmt.close, end)
        if close < 0:
            return None
    return None


def _heal_native_text(
    text: str,
    fmt: NativeToolFormat,
    allowed_tools: set,
    tool_schemas: Optional[dict],
    *,
    allow_incomplete: bool,
    start: int = 0,
) -> Optional[tuple[list, dict]]:
    """Events for ``text[start:]`` with every block promoted, and each call's source; None keeps the
    text. All or nothing: other-format markup outside the blocks could quote a block that would run."""
    blocks: list = []
    # Markup quoted in a leading JSON answer is data, as on the generic path.
    lead = len(text) - len(text.lstrip())
    try:
        lead = (
            json.JSONDecoder().raw_decode(text, lead)[1] if text.startswith(("{", "["), lead) else 0
        )
    except (ValueError, RecursionError):
        lead = 0
    pos = lead
    while (begin := text.find(fmt.open, pos)) >= 0:
        think = _THINK_OPEN_RE.search(text, pos, begin)
        if think is not None:
            pos = _THINK_TAG_RE.match(text, think.start()).end()
            continue
        block = _native_block(text, begin, fmt, allow_incomplete)
        if block is None:
            return None
        blocks.append(block)
        pos = block[1]
    if not blocks:
        return None
    bounds = [lead, *(edge for begin, end, _calls in blocks for edge in (begin, end)), len(text)]
    outside = "".join(text[a:b] for a, b in zip(bounds[::2], bounds[1::2]))
    foreign = _FOREIGN_SIGNALS - {fmt.open}
    if any(m in outside for m in foreign) or _ATTR_FUNC_OPEN_RE.search(outside):
        return None
    calls = [call for _begin, _end, block_calls in blocks for call in block_calls]
    promoted = _promote(calls, allowed_tools, tool_schemas = tool_schemas)
    if len(promoted) != len(calls):
        return None
    ids = [call["id"] for call in calls]
    if len(set(ids)) == len(ids):
        # Kimi's functions.NAME:IDX ids replay through its template; generated ones restart per block.
        for call, call_id in zip(promoted, ids):
            call["id"] = call_id
    events: list = []
    sources: dict = {}
    order = iter(promoted)
    pos = start
    for begin, end, block_calls in blocks:
        if text[pos:begin]:
            events.append(("text", text[pos:begin]))
        for index in range(len(block_calls)):
            call = next(order)
            events.append(("tool_call", call))
            # One source per block, so un-promoting every call relays it once.
            sources[call["id"]] = "" if index else text[begin:end]
        pos = end
    if text[pos:]:
        events.append(("text", text[pos:]))
    return events, sources


def heal_openai_message_events(
    msg: dict,
    allowed_tools: set,
    tools: Optional[list] = None,
    allow_incomplete: bool = True,
    native: Optional[NativeToolFormat] = None,
) -> Optional[list]:
    if not isinstance(msg, dict) or msg.get("tool_calls"):
        return None
    content = msg.get("content")
    if not isinstance(content, str):
        return None
    if native is not None:
        tool_schemas = _tool_schemas_by_name(tools) if tools is not None else None
        healed = _heal_native_text(
            content, native, allowed_tools, tool_schemas, allow_incomplete = allow_incomplete
        )
        return healed[0] if healed else None
    if not _has_heal_signal(content):
        return None
    parsed, spans = parse_tool_calls_from_text(content, allow_incomplete = True, with_spans = True)
    # A strict parse alone would promote calls quoted inside a rejected unfinished call.
    closed = (
        None
        if allow_incomplete
        else set(parse_tool_calls_from_text(content, allow_incomplete = False, with_spans = True)[1])
    )
    tool_schemas = _tool_schemas_by_name(tools) if tools is not None else None
    events: list = []
    pos = 0
    call_count = 0
    for call, (start, end) in zip(parsed, spans):
        promoted = (
            _promote([call], allowed_tools, id_offset = call_count, tool_schemas = tool_schemas)
            if closed is None or (start, end) in closed
            else None
        )
        if promoted:
            if content[pos:start]:
                events.append(("text", content[pos:start]))
            events.append(("tool_call", promoted[0]))
            call_count += 1
        else:
            events.append(("text", content[pos:end]))
        pos = end
    if not call_count:
        return None
    if content[pos:]:
        events.append(("text", content[pos:]))
    return events


def heal_openai_message(
    msg: dict,
    allowed_tools: set,
    tools: Optional[list] = None,
    allow_incomplete: bool = True,
    native: Optional[NativeToolFormat] = None,
) -> bool:
    """Edits in place; only promoted calls' markup is removed, undeclared calls stay byte-intact."""
    events = heal_openai_message_events(msg, allowed_tools, tools, allow_incomplete, native)
    if not events:
        return False
    calls = [value for kind, value in events if kind == "tool_call"]
    content = "".join(value for kind, value in events if kind == "text").strip()
    msg["tool_calls"] = calls
    # OpenAI requires content = null on a pure tool-call turn.
    msg["content"] = content or None
    return True


def _earliest_signal(buffer: str) -> int:
    best = -1
    for signal in _HEAL_SIGNALS:
        index = buffer.find(signal)
        if index >= 0 and (best < 0 or index < best):
            best = index
    return best


def _closed_signal_span(buffer: str) -> Optional[tuple[int, int]]:
    spans = []
    for open_tag, close_tag in (
        ("<tool_call>", "</tool_call>"),
        ("<|tool_call>", "<tool_call|>"),
        ("<function=", "</function>"),
    ):
        start = buffer.find(open_tag)
        if start < 0:
            continue
        end = buffer.find(close_tag, start)
        if end >= 0:
            spans.append((start, end + len(close_tag)))
    return min(spans, key = lambda span: span[0]) if spans else None


def _inside_open_wrapper(buffer: str, start: int) -> bool:
    for open_tag, close_tag in (("<tool_call>", "</tool_call>"), ("<|tool_call>", "<tool_call|>")):
        opened = buffer.rfind(open_tag, 0, start)
        if opened >= 0 and buffer.find(close_tag, opened) < 0:
            return True
    return False


# A strict parse needs a JSON/Gemma close tag or </function>; else only a balanced Mistral/rehearsal bracket call parses.
_STRICT_CLOSE_TOKENS = ("</tool_call>", "<|end_message|>", "<tool_call|>", "</function>")
_CALLS_TAG = "[TOOL_CALLS]"
_ARGS_TAG = "[ARGS]"
_WS_RE = re.compile(r"\s*")
# A rehearsal name[ARGS] needs a name character right before the tag.
_NAME_CHAR_RE = re.compile(r"[\w-]")
_SPAN_SKIP_RE = re.compile(r'[^"{}\\]*')
_STRING_SKIP_RE = re.compile(r'[^"\\]*')
_BRACKET_SKIP_RE = re.compile(r'[^"\[\]{}]*')


def _resume_json_span(text: str, state: list) -> Optional[int]:
    """``_balanced_json_span`` resumed from ``state`` (i, depth, in_string, escape) over appended text."""
    i, depth, in_string, escape = state
    n = len(text)
    while i < n:
        if escape:
            escape = False
            i += 1
            continue
        i = (_STRING_SKIP_RE if in_string else _SPAN_SKIP_RE).match(text, i).end()
        if i >= n:
            break
        ch = text[i]
        if ch == "\\":
            escape = True
        elif in_string:
            in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    state[:] = (i, depth, in_string, escape)
    return None


def _resume_bracket_end(text: str, state: list) -> Optional[int]:
    """``_balanced_bracket_end`` resumed from ``state`` (i, depth, in_string); a string escape at EOF waits."""
    i, depth, in_string, _escape = state
    n = len(text)
    while i < n:
        if in_string:
            i = _STRING_SKIP_RE.match(text, i).end()
            if i >= n:
                break
            if text[i] == "\\":
                if i + 1 >= n:
                    break
                i += 2
                continue
            in_string = False
            i += 1
            continue
        i = _BRACKET_SKIP_RE.match(text, i).end()
        if i >= n:
            break
        ch = text[i]
        if ch == '"':
            in_string = True
        elif ch in "[{":
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    state[:] = (i, depth, in_string, False)
    return None


class _StrictParseGate:
    """Per-healer scan of a held, append-only buffer: False only when a strict parse of it is provably empty.

    Each chunk scans only the appended text; any buffer that does not extend the last one starts over."""

    __slots__ = ("_buf", "_closer", "_calls", "_args", "_scans", "_brace_close", "_any_close")

    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self._buf = ""
        self._closer = False
        self._calls: list = []
        self._args: list = []
        # (open index, is array) -> resumable scan state, or the close index once balanced.
        self._scans: dict = {}
        # Last "}" and last "]" or "}": an opener after them cannot balance yet.
        self._brace_close = -1
        self._any_close = -1

    def release(self) -> None:
        """Drop the scan once the healer lets go of its held text."""
        if self._buf:
            self._reset()

    def may_parse(self, buffer: str) -> bool:
        prev_len = len(self._buf)
        if not buffer.startswith(self._buf):
            self._reset()
            prev_len = 0
        self._buf = buffer
        if self._closer:
            return True
        for token in _STRICT_CLOSE_TOKENS:
            # Back off so a token split across chunks is still found.
            if buffer.find(token, max(0, prev_len - len(token) + 1)) >= 0:
                self._closer = True
                return True
        for tag, found in ((_CALLS_TAG, self._calls), (_ARGS_TAG, self._args)):
            pos = buffer.find(tag, max(0, prev_len - len(tag) + 1))
            while pos >= 0:
                if found is self._calls or (pos and _NAME_CHAR_RE.match(buffer, pos - 1)):
                    found.append(pos)
                pos = buffer.find(tag, pos + len(tag))
        if not self._calls and not self._args:
            return False
        brace_close = buffer.rfind("}", prev_len)
        if brace_close >= 0:
            self._brace_close = brace_close
        brace_close = self._brace_close
        any_close = max(buffer.rfind("]", prev_len), brace_close, self._any_close)
        self._any_close = any_close
        # Openers _iter_bracket_spans would scan; regexes rerun since a tail match can still change.
        opens = []
        calls = self._calls
        for pos in calls[: bisect.bisect_right(calls, any_close)]:
            m = _MISTRAL_BRACKET_RE.match(buffer, pos)
            if m is not None and m.end() < brace_close:
                opens.append((m.end(), False))
            m = _MISTRAL_ARRAY_RE.match(buffer, pos)
            if m is not None and m.end() < any_close:
                opens.append((m.end(), True))
        args = self._args
        for pos in args[: bisect.bisect_right(args, brace_close)]:
            brace = _WS_RE.match(buffer, pos + len(_ARGS_TAG)).end()
            if brace < brace_close and buffer.startswith("{", brace):
                opens.append((brace, False))
        scans = self._scans
        for key in opens:
            state = scans.get(key)
            if state is None:
                state = scans[key] = [key[0], 0, False, False]
            elif not isinstance(state, list):
                return True
            end = (_resume_bracket_end if key[1] else _resume_json_span)(buffer, state)
            if end is not None:
                scans[key] = end
                return True
        return False


def _partial_signal_suffix(
    buffer: str,
    signals: tuple = _HEAL_SIGNALS,
    longest: int = _MAX_SIGNAL_LEN,
) -> int:
    """Length of the longest buffer suffix that is a proper prefix of a signal."""
    for length in range(min(len(buffer), longest - 1), 0, -1):
        tail = buffer[-length:]
        if any(signal.startswith(tail) for signal in signals):
            return length
    return 0


class StreamToolCallHealer:
    """Buffer-and-repair state machine for streamed passthrough content.

    ``feed(text)`` / ``finalize()`` yield ``("text", str)`` events for content
    to relay and ``("tool_call", dict)`` events carrying an OpenAI-shaped call
    (string ``function.arguments``). Normal prose is forwarded immediately; only
    a trailing partial-signal window (< max signal length) or a suspected tool
    block is ever withheld, so streaming latency stays bounded. A false alarm
    (the buffer can no longer become a parseable declared call) flushes the held
    text verbatim.
    """

    def __init__(
        self,
        allowed_tools: set,
        tools: Optional[list] = None,
        native: Optional[NativeToolFormat] = None,
    ) -> None:
        self._allowed = set(allowed_tools)
        # With a native format, text from its first opener is held to the end and healed whole.
        self._native = native
        self._relayed: Optional[str] = ""

        self._tool_schemas = _tool_schemas_by_name(tools) if tools is not None else None
        self._buffer = ""
        self._holding = False
        self._id_offset = 0
        # Promotion is destructive; keep spans so a discarded call can return the model's words.
        self._promoted_spans: dict[str, str] = {}
        self._strict_gate = _StrictParseGate()
        # Structured delta.tool_calls upstream: grammar mode already works, so healing goes dormant.
        self.dormant = False

    @property
    def healed(self) -> bool:
        return self._id_offset > 0

    def promoted_source(self, call_id: str) -> str:
        """Only for un-promoting: relaying it next to the call would double the output."""
        return self._promoted_spans.get(call_id, "")

    def structured_tool_call_seen(self) -> list:
        """Go dormant; flush anything held so no text is swallowed."""
        self.dormant = True
        self._strict_gate.release()
        held, self._buffer, self._holding = self._buffer, "", False
        return [("text", held)] if held else []

    def feed(self, text: str) -> list:
        if self.dormant:
            return [("text", text)] if text else []
        if self._native is not None:
            return self._feed_native(text)
        self._buffer += text
        return self._drain()

    def _feed_native(self, text: str) -> list:
        if self._relayed is None:
            return [("text", text)] if text else []
        self._buffer += text
        events: list = []
        if not self._holding:
            start = self._buffer.find(self._native.open)
            cut = (
                start
                if start >= 0
                else len(self._buffer)
                - _partial_signal_suffix(self._buffer, (self._native.open,), len(self._native.open))
            )
            if cut:
                events.append(("text", self._buffer[:cut]))
                self._relayed += self._buffer[:cut]
                self._buffer = self._buffer[cut:]
            self._holding = start >= 0
        if self._holding and len(self._buffer) > _MAX_HOLD_CHARS:
            events.append(("text", self._buffer))
            self._buffer, self._holding, self._relayed = "", False, None
        return events

    def _drain(self) -> list:
        events: list = []
        while True:
            if not self._holding:
                self._strict_gate.release()
                start = _earliest_signal(self._buffer)
                if start >= 0:
                    if start:
                        events.append(("text", self._buffer[:start]))
                    self._buffer = self._buffer[start:]
                    self._holding = True
                else:
                    keep = _partial_signal_suffix(self._buffer)
                    emit = self._buffer[: len(self._buffer) - keep]
                    if emit:
                        events.append(("text", emit))
                    self._buffer = self._buffer[len(self._buffer) - keep :]
                    return events
            # A whole-buffer reparse per chunk is quadratic in the held size.
            may_parse = self._strict_gate.may_parse(self._buffer)
            parsed, spans = (
                parse_tool_calls_from_text(
                    self._buffer,
                    id_offset = self._id_offset,
                    allow_incomplete = False,
                    with_spans = True,
                )
                if may_parse
                else ([], [])
            )
            # A bracket call after a still-open tag block is that block's argument text.
            if (
                parsed
                and self._buffer.startswith("<")
                and not self._buffer.startswith("<", spans[0][0])
                and _closed_signal_span(self._buffer) is None
            ):
                parsed, spans = [], []
            # Qwen3.5's <function=...> closes inside a still-open <tool_call>; promoting it now would relay the wrapper as text.
            if (
                parsed
                and len(self._buffer) <= _MAX_HOLD_CHARS
                and _inside_open_wrapper(self._buffer, spans[0][0])
            ):
                return events
            if not parsed:
                # Its close tags are gate tokens, so a closed gate means no closed span.
                closed_span = _closed_signal_span(self._buffer) if may_parse else None
                if closed_span:
                    _start, end = closed_span
                    events.append(("text", self._buffer[:end]))
                    self._buffer = self._buffer[end:]
                    self._holding = False
                    continue
                if len(self._buffer) > _MAX_HOLD_CHARS:
                    events.append(("text", self._buffer))
                    self._buffer = ""
                    self._holding = False
                    continue
                return events
            pos = 0
            run_end = spans[0][1]
            for order, (call, (start, end)) in enumerate(zip(parsed, spans)):
                # Stop at the first gap or incomplete trailing block, so it re-holds and does not flush text early.
                if order and start != run_end:
                    break
                promoted = _promote(
                    [call],
                    self._allowed,
                    id_offset = self._id_offset,
                    tool_schemas = self._tool_schemas,
                )
                if promoted:
                    # Flush any leading text, then drop the promoted markup span.
                    if self._buffer[pos:start]:
                        events.append(("text", self._buffer[pos:start]))
                    events.append(("tool_call", promoted[0]))
                    self._promoted_spans[promoted[0]["id"]] = self._buffer[start:end]
                    self._id_offset += 1
                else:
                    # Undeclared/unusable name: markup is DATA, flush it (and prior text) verbatim.
                    events.append(("text", self._buffer[pos:end]))
                pos = end
                run_end = end
            # Everything past the drained run (later blocks) stays and is rescanned.
            self._buffer = self._buffer[run_end:]
            self._holding = False

    def finalize(self, promote: bool = True) -> list:
        """Keeps event order; only promoted call markup is dropped and all other residue flushes as text."""
        self._strict_gate.release()
        if not self._buffer:
            return []
        residue, self._buffer = self._buffer, ""
        holding, self._holding = self._holding, False
        if self.dormant or not holding:
            return [("text", residue)]
        if self._native is not None:
            # Held whole, so its closed blocks promote even when cut-off calls may not.
            healed = _heal_native_text(
                self._relayed + residue,
                self._native,
                self._allowed,
                self._tool_schemas,
                allow_incomplete = promote,
                start = len(self._relayed),
            )
            if healed is None:
                return [("text", residue)]
            events, sources = healed
            self._promoted_spans.update(sources)
            self._id_offset += len(sources)
            return events
        if not promote:
            return [("text", residue)]
        parsed, spans = parse_tool_calls_from_text(
            residue,
            id_offset = self._id_offset,
            allow_incomplete = True,
            with_spans = True,
        )
        events: list = []
        pos = 0
        any_promoted = False
        for call, (start, end) in zip(parsed, spans):
            promoted = _promote(
                [call],
                self._allowed,
                id_offset = self._id_offset,
                tool_schemas = self._tool_schemas,
            )
            if promoted:
                if residue[pos:start]:
                    events.append(("text", residue[pos:start]))
                events.append(("tool_call", promoted[0]))
                self._promoted_spans[promoted[0]["id"]] = residue[start:end]
                self._id_offset += 1
                any_promoted = True
            else:
                events.append(("text", residue[pos:end]))
            pos = end
        if not any_promoted:
            return [("text", residue)]
        tail = residue[pos:].strip()
        if tail:
            events.append(("text", tail))
        return events


def _first_choice_message(data: Any) -> Optional[dict]:
    """Upstream error bodies can carry a null message or no choices, so the shape is never assumed."""
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None
    return message if isinstance(message, dict) else None


def _last_assistant_text(data: Any) -> str:
    """First-choice assistant content of a non-streaming chat response, or ''."""
    message = _first_choice_message(data)
    content = message.get("content") if message else None
    return content if isinstance(content, str) else ""


def _heal_would_promote(
    text: str,
    allowed_tools: set,
    tools: Optional[list] = None,
) -> bool:
    """Whether ``heal_openai_message`` would promote at least one call."""
    parsed = parse_tool_calls_from_text(text, allow_incomplete = True)
    tool_schemas = _tool_schemas_by_name(tools) if tools is not None else None
    return bool(_promote(parsed, allowed_tools, tool_schemas = tool_schemas))


def response_has_promotable_calls(
    data: Any,
    allowed_tools: set,
    tools: Optional[list] = None,
) -> bool:
    """Used to judge a nudge retry: an undeclared hallucinated call is not an improvement."""
    message = _first_choice_message(data)
    if not message:
        return False
    tool_calls = message.get("tool_calls")
    if tool_calls:
        # ALL structured calls must be declared: one hallucinated name would reach the client.
        return all(
            isinstance(tc, dict)
            and isinstance(tc.get("function"), dict)
            and tc["function"].get("name") in allowed_tools
            for tc in tool_calls
        )
    text = message.get("content")
    if not isinstance(text, str):
        return False
    return _heal_would_promote(text, allowed_tools, tools)


def nudge_should_retry(
    data: Any,
    allowed_tools: Optional[set],
    tools: Optional[list] = None,
) -> bool:
    """Retry only when text has a tool signal that healing did not promote; clean prose never retries."""
    if not allowed_tools:
        return False
    message = _first_choice_message(data)
    if not message or message.get("tool_calls"):
        return False
    text = message.get("content")
    if not isinstance(text, str) or not _has_heal_signal(text):
        return False
    return not _heal_would_promote(text, allowed_tools, tools)


def nudge_messages(data: Any, allowed_tools: set) -> list:
    """Appended to the original body so the prompt prefix stays byte-identical for llama-server's cache."""
    tool_hint = " or ".join(f"`{name}`" for name in sorted(allowed_tools)) or "an available tool"
    return [
        {"role": "assistant", "content": _last_assistant_text(data)},
        {
            "role": "user",
            "content": (
                "You have access to the declared tools. If a tool is needed to "
                f"complete the action you described, call {tool_hint} now using the "
                "native tool-call format with valid JSON arguments, not prose. If no "
                "tool is needed, provide the final answer directly."
            ),
        },
    ]
