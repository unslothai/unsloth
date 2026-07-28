# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Dependency-light wrapper around tokenizer.apply_chat_template with a kwarg
fallback for templates that reject reasoning/tools args, plus the shared
native-chat-template fallback used by the transformers and MLX backends.
"""

import copy
import json
import logging
from dataclasses import dataclass
from typing import Optional

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
# Invisible separator: neutralized markup still looks like the original tag but no
# longer matches structural parsers or special tokens (#7066). U+2060 WORD JOINER,
# not U+200B ZERO WIDTH SPACE: U+200B is line-break class ZW, so a neutralized tag
# could wrap mid-tag; WORD JOINER (class WJ) forbids that break (#7334).
_THINK_NEUTRAL_ZW = "\u2060"
_GEMMA_CHANNEL_START = "<|channel>"
_GEMMA_THOUGHT_OPEN = "<|channel>thought"
_GEMMA_THOUGHT_CLOSE = "<channel|>"
_GEMMA_TEMPLATE_OPENERS = (
    _GEMMA_THOUGHT_OPEN + "\n",
    _GEMMA_THOUGHT_OPEN + "\\n",
    _GEMMA_THOUGHT_OPEN + _GEMMA_THOUGHT_CLOSE,
)

# Markers that must not reach a non-assistant turn (user / system / tool) as raw
# text, or templates / think extractors / stop sequences read it as markup.
_NON_ASSISTANT_CONTROL_MARKERS: tuple[tuple[str, str], ...] = (
    (_THINK_CLOSE, f"</{_THINK_NEUTRAL_ZW}think>"),
    (_THINK_OPEN, f"<{_THINK_NEUTRAL_ZW}think>"),
    ("<|im_start|>", f"<|{_THINK_NEUTRAL_ZW}im_start|>"),
    ("<|im_end|>", f"<|{_THINK_NEUTRAL_ZW}im_end|>"),
    # Gemma-4 GGUF thinking sentinels: raw, they inject a fake thought channel
    # (#7066). A no-op for other templates.
    (_GEMMA_CHANNEL_START, f"<|{_THINK_NEUTRAL_ZW}channel>"),
    (_GEMMA_THOUGHT_CLOSE, f"<{_THINK_NEUTRAL_ZW}channel|>"),
    # The same templates (assets/chat_templates/gemma-4*.jinja) delimit every turn,
    # tool block and tool result with these and quote schema strings with <|"|>:
    # raw, they end their own block or forge a model / tool_response one (#7066).
    ("<|turn>", f"<|{_THINK_NEUTRAL_ZW}turn>"),
    ("<turn|>", f"<{_THINK_NEUTRAL_ZW}turn|>"),
    ("<|tool_call>", f"<|{_THINK_NEUTRAL_ZW}tool_call>"),
    ("<tool_call|>", f"<{_THINK_NEUTRAL_ZW}tool_call|>"),
    ("<|tool_response>", f"<|{_THINK_NEUTRAL_ZW}tool_response>"),
    ("<tool_response|>", f"<{_THINK_NEUTRAL_ZW}tool_response|>"),
    ("<|tool>", f"<|{_THINK_NEUTRAL_ZW}tool>"),
    ("<tool|>", f"<{_THINK_NEUTRAL_ZW}tool|>"),
    # gemma-4.jinja turns thinking on with <|think|> in the first system turn, so a
    # raw one in non-assistant text switches reasoning mode.
    ("<|think|>", f"<|{_THINK_NEUTRAL_ZW}think|>"),
    ('<|"|>', f'<|{_THINK_NEUTRAL_ZW}"|>'),
    # Llama-3 turn delimiters (chat_eos.py / tool_call_parser.py treat them as turn
    # ends): raw, they close their own turn and inject a fake assistant one,
    # ``<|eot_id|><|start_header_id|>assistant`` (#7066).
    ("<|eot_id|>", f"<|{_THINK_NEUTRAL_ZW}eot_id|>"),
    ("<|start_header_id|>", f"<|{_THINK_NEUTRAL_ZW}start_header_id|>"),
    ("<|end_header_id|>", f"<|{_THINK_NEUTRAL_ZW}end_header_id|>"),
    # The remaining chat_eos turn-end tokens (Llama tool turns, Gemma, Phi,
    # OpenChat) plus Gemma's turn opener, same hole as <|eot_id|>.
    # test_neutralize_covers_every_turn_end_token pins this against chat_eos.
    ("<|eom_id|>", f"<|{_THINK_NEUTRAL_ZW}eom_id|>"),
    ("<end_of_turn>", f"<{_THINK_NEUTRAL_ZW}end_of_turn>"),
    ("<start_of_turn>", f"<{_THINK_NEUTRAL_ZW}start_of_turn>"),
    ("<|end_of_turn|>", f"<|{_THINK_NEUTRAL_ZW}end_of_turn|>"),
    ("<|end|>", f"<|{_THINK_NEUTRAL_ZW}end|>"),
    # Zephyr / Phi-3 open turns with a bare role sentinel instead of a header pair,
    # so these ARE the turn boundary there ("<|user|>\n" + content + eos_token):
    # raw, an EOS followed by "<|assistant|>" tokenizes as a forged model turn.
    ("<|user|>", f"<|{_THINK_NEUTRAL_ZW}user|>"),
    ("<|assistant|>", f"<|{_THINK_NEUTRAL_ZW}assistant|>"),
    ("<|system|>", f"<|{_THINK_NEUTRAL_ZW}system|>"),
)


def neutralize_think_markup(text: str) -> str:
    """Neutralize structural ``<think>`` / ``</think>`` inside free text.

    Used when wrapping ``reasoning_content`` into synthetic think tags or when
    a mid-thought literal close must stay inside the reasoning drawer (#7066).
    """
    if not text or (_THINK_OPEN not in text and _THINK_CLOSE not in text):
        return text
    return text.replace(_THINK_CLOSE, f"</{_THINK_NEUTRAL_ZW}think>").replace(
        _THINK_OPEN, f"<{_THINK_NEUTRAL_ZW}think>"
    )


def think_markup_holdback(text: str) -> int:
    """Trailing chars that may be a prefix of a think marker (split-chunk safe)."""
    markers = (_THINK_CLOSE, _THINK_OPEN)
    max_marker = max(len(marker) for marker in markers)
    for size in range(min(len(text), max_marker - 1), 0, -1):
        suffix = text[-size:]
        if any(marker.startswith(suffix) for marker in markers):
            return size
    return 0


def neutralize_think_markup_streaming(buffer: str, *, finalize: bool = False) -> tuple[str, str]:
    """Neutralize complete think markers in *buffer*, retaining a trailing holdback.

    Returns ``(emit, remaining_buffer)`` for streaming ``reasoning_content`` chunks
    that may split a literal ``</think>`` across SSE boundaries (#7066).
    """
    if not buffer:
        return "", ""
    if finalize:
        return neutralize_think_markup(buffer), ""
    keep = think_markup_holdback(buffer)
    if keep == len(buffer):
        return "", buffer
    emit = buffer[:-keep] if keep else buffer
    remaining = buffer[-keep:] if keep else ""
    return neutralize_think_markup(emit), remaining


def neutralize_non_assistant_control_markup(text: str) -> str:
    """Neutralize think + ChatML control markers in user/system/tool text (#7066)."""
    return _neutralize_markers(text, _NON_ASSISTANT_CONTROL_MARKERS)


def _neutralize_markers(text: str, markers) -> str:
    if not text:
        return text
    out = text
    for src, dst in markers:
        if src in out:
            out = out.replace(src, dst)
    return out


# Neutralized in assistant content too: replayed history is client-controlled, and
# a raw boundary there truncates that turn or injects a new one. The assistant's
# own think / channel / tool markup is structural and stays (#7066).
_TURN_BOUNDARY_NAMES = frozenset(
    {
        "<|im_start|>",
        "<|im_end|>",
        "<|eot_id|>",
        "<|eom_id|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<start_of_turn>",
        "<end_of_turn>",
        "<|end_of_turn|>",
        "<|end|>",
        "<|turn>",
        "<turn|>",
        # Zephyr / Phi-3 open a turn with these alone, so they are that template's
        # turn boundary and must not survive assistant replay.
        "<|user|>",
        "<|assistant|>",
        "<|system|>",
    }
)
_TURN_BOUNDARY_MARKERS: tuple[tuple[str, str], ...] = tuple(
    pair for pair in _NON_ASSISTANT_CONTROL_MARKERS if pair[0] in _TURN_BOUNDARY_NAMES
)


def neutralize_turn_boundary_markup(text: str) -> str:
    """Neutralize only the turn-boundary sentinels, for assistant text (#7066)."""
    return _neutralize_markers(text, _TURN_BOUNDARY_MARKERS)


# Entries REFERENCE declared property names, not prose. Keys are preserved, so
# rewriting these would name a property the schema no longer declares (OpenAI
# strict mode rejects it; Gemini needs every ``propertyOrdering`` entry valid) (#7066).
_SCHEMA_NAME_LIST_KEYS = frozenset({"required", "propertyOrdering"})
# Same, one level deeper: {"dependentRequired": {"a": ["b"]}}. The object-valued
# (sub-schema) form of ``dependencies`` is prose-bearing, so it still gets walked.
_SCHEMA_NAME_MAP_KEYS = frozenset({"dependentRequired", "dependencies"})
# Pointers and their anchors: "#/$defs/<name>" must keep matching the $defs key it
# names, which this pass leaves alone (#7066).
_SCHEMA_REF_KEYS = frozenset({"$ref", "$dynamicRef", "$id", "$anchor", "$dynamicAnchor", "$schema"})
# Values the model must reproduce byte for byte: llama.cpp compiles const/enum into
# literal GBNF rules and pattern into a regex rule (common/json-schema-to-grammar.cpp)
# and constrains sampling with them, so a rewrite makes the decoder emit the REWRITTEN
# value and nothing maps it back. It also buys nothing: a </think> here only reaches
# the prompt, and the think parser reads model OUTPUT (#7334).
_SCHEMA_VALUE_KEYS = frozenset({"const", "default", "enum", "examples", "pattern"})
# Maps a CALLER-CHOSEN name to a sub-schema, so a property genuinely called "enum"
# or "pattern" must not be read as the keyword and skip neutralization (#7334).
_SCHEMA_SUBSCHEMA_MAP_KEYS = frozenset(
    {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
)


def _is_schema_name_list(item) -> bool:
    return isinstance(item, list) and all(isinstance(entry, str) for entry in item)


def _is_schema_name_reference(key, item) -> bool:
    """True when ``item`` under ``key`` lists property names, not prompt text."""
    if not isinstance(key, str):
        return False
    if key in _SCHEMA_REF_KEYS:
        return isinstance(item, str)
    return key in _SCHEMA_NAME_LIST_KEYS and _is_schema_name_list(item)


def _is_schema_constrained_value(key) -> bool:
    """True when ``key`` holds a value the model must emit exactly, not prose."""
    return isinstance(key, str) and key in _SCHEMA_VALUE_KEYS


def _is_schema_dependency_map(key, item) -> bool:
    """True for ``dependencies`` / ``dependentRequired``: name -> names or schema."""
    return isinstance(key, str) and key in _SCHEMA_NAME_MAP_KEYS and isinstance(item, dict)


def _neutralize_schema_dependency_map(value):
    """Walk a dependency map, preserving its name-list entries individually.

    Draft-7 ``dependencies`` may mix name arrays with sub-schemas, so the arrays
    are kept as references while the sub-schemas still go through the walk.
    """
    changed = False
    out = {}
    for key, item in value.items():
        if _is_schema_name_list(item):
            out[key] = item
            continue
        new_item = neutralize_control_markup_deep(item, schema = True)
        if new_item is not item and new_item != item:
            changed = True
        out[key] = new_item
    return out if changed else value


def neutralize_control_markup_deep(value, *, schema: bool = False, named_keys: bool = False):
    """Recursively neutralize control markers in every string *value* of a
    nested dict/list structure (tool schemas / tool-call argument JSON).

    Dict keys are left untouched; only leaf strings are rewritten. Keys are
    identifiers, not prompt prose: renaming a schema property would hand the
    model an argument name the client never declared, and nothing maps it back
    on the generated tool call. With ``schema = True`` the name lists mirroring
    those keys (``required`` and friends) are preserved for the same reason, and
    so are the constrained values (``enum`` and friends) the schema compiles
    into the decoder's grammar; tool-call arguments carry neither, so their data
    is always rewritten. ``named_keys`` marks a mapping whose own keys are
    caller-chosen names (``properties`` and friends), so they are not read as
    schema keywords. Returns the same object when nothing changed so callers
    keep byte-identical payloads on the common path (#7066).
    """
    if isinstance(value, str):
        return neutralize_non_assistant_control_markup(value)
    if isinstance(value, dict):
        changed = False
        out = {}
        keywords = schema and not named_keys
        for key, item in value.items():
            if keywords and (
                _is_schema_name_reference(key, item) or _is_schema_constrained_value(key)
            ):
                out[key] = item
                continue
            if keywords and _is_schema_dependency_map(key, item):
                new_item = _neutralize_schema_dependency_map(item)
            else:
                new_item = neutralize_control_markup_deep(
                    item,
                    schema = schema,
                    named_keys = keywords and key in _SCHEMA_SUBSCHEMA_MAP_KEYS,
                )
            if new_item is not item and new_item != item:
                changed = True
            out[key] = new_item
        return out if changed else value
    if isinstance(value, list):
        changed = False
        out = []
        for item in value:
            new_item = neutralize_control_markup_deep(item, schema = schema)
            if new_item is not item and new_item != item:
                changed = True
            out.append(new_item)
        return out if changed else value
    return value


def neutralize_tools_control_markup(tools):
    """Neutralize think / ChatML control markers in client tool schemas (#7066).

    Tool function descriptions and parameter prose are rendered into the chat
    template as prompt text, so a schema containing ``</think>`` or
    ``<|im_start|>`` would otherwise bypass message-level neutralization.

    Two categories are preserved verbatim instead. ``required`` /
    ``propertyOrdering`` name the declared properties, whose keys this pass
    leaves alone, so rewriting one would point the schema at a property it no
    longer declares. ``enum`` / ``const`` / ``default`` / ``examples`` /
    ``pattern`` carry values, and a schema is not only prompt text: llama-server
    compiles it into the GBNF grammar that constrains tool-call sampling, so
    rewriting one makes the decoder emit the rewritten value and nothing maps it
    back before the call reaches the client. Prose keeps its rewrite because a
    ``</think>`` in the PROMPT is harmless anyway - the think parser reads model
    OUTPUT - while a turn sentinel there is not (#7334).
    """
    if not tools:
        return tools
    return neutralize_control_markup_deep(tools, schema = True)


def _neutralize_tool_arguments_json(args: str) -> str:
    """Neutralize a JSON-string argument payload, keeping its object keys.

    Argument names mirror the schema property keys this pass preserves, so a
    plain string rewrite would rename one and hand the template an argument the
    client never declared, and disagree with the parsed-dict path. Payloads
    without a marker keep their exact bytes; only a payload that has one is
    parsed and re-serialized (#7066).
    """
    neutral = neutralize_non_assistant_control_markup(args)
    if neutral == args:
        return args
    try:
        parsed = json.loads(args)
    except (TypeError, ValueError):
        return neutral  # not JSON: nothing to key-preserve, rewrite the text
    if not isinstance(parsed, (dict, list)):
        return neutral
    cleaned = neutralize_control_markup_deep(parsed)
    if cleaned is parsed:
        return args
    return json.dumps(cleaned, ensure_ascii = False)


def neutralize_tool_call_arguments(tool_calls):
    """Neutralize control markers inside assistant tool calls.

    Assistant prose keeps its real ``<think>`` structure, but a replayed
    ``tool_calls[].function.arguments`` string is user/model-derived data that
    must not smuggle a literal ``</think>`` or ``<|im_start|>`` into the next
    chat template (#7066). The call ``id`` gets the same treatment: several
    native templates render it, and the rewrite is deterministic, so it still
    matches the ``tool_call_id`` of its result message, which
    :func:`neutralize_control_markup_in_messages` rewrites the same way.
    Returns the same list when nothing changed.
    """
    if not isinstance(tool_calls, list) or not tool_calls:
        return tool_calls
    changed = False
    out = []
    for call in tool_calls:
        if isinstance(call, dict):
            call_id = call.get("id")
            if isinstance(call_id, str) and call_id:
                new_id = neutralize_non_assistant_control_markup(call_id)
                if new_id != call_id:
                    call = {**call, "id": new_id}
                    changed = True
            fn = call.get("function")
            # Gemma-4 concatenates the name into the <|tool_call> block, so
            # "lookup<tool_call|>" would close it. The deep schema sanitizer
            # rewrites the same name on the tool definition side.
            if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                new_name = neutralize_non_assistant_control_markup(fn["name"])
                if new_name != fn["name"]:
                    fn = {**fn, "name": new_name}
                    call = {**call, "function": fn}
                    changed = True
            if isinstance(fn, dict) and fn.get("arguments") is not None:
                args = fn["arguments"]
                if isinstance(args, str):
                    new_args = _neutralize_tool_arguments_json(args)
                else:
                    # On the retry path _normalize_tool_call_arguments() has
                    # already parsed the JSON string, so a marker inside a parsed
                    # value would render raw unless walked too (#7066).
                    new_args = neutralize_control_markup_deep(args)
                if new_args is not args and new_args != args:
                    call = {**call, "function": {**fn, "arguments": new_args}}
                    changed = True
        out.append(call)
    return out if changed else tool_calls


def _split_marker_boundary(text: str, ahead: str, markers) -> bool:
    """True when ``text`` and what follows only form a marker once joined.

    Templates concatenate adjacent text parts with no separator and trim each
    (``gemma-4.jinja:333-340``), so a marker cut across two parts survives the
    per-part pass and is rebuilt in the rendered prompt (#7066).
    """
    tail, head = text.rstrip(), ahead.lstrip()
    if not tail or not head:
        return False
    longest = max(len(src) for src, _ in markers) - 1
    if longest <= 0:
        return False
    tail, head = tail[-longest:], head[:longest]
    joined = tail + head
    for src, _ in markers:
        at = joined.find(src)
        while at != -1:
            # Only counts when it straddles the join; a marker inside either side
            # alone was already neutralized by that part.
            if at < len(tail) < at + len(src):
                return True
            at = joined.find(src, at + 1)
    return False


def _rendered_lookahead(texts: list, index: int, limit: int) -> str:
    """The first ``limit`` chars the template renders after ``texts[index]``.

    Adjacent text parts are concatenated with no separator and each is trimmed
    (``gemma-4.jinja:333-340``), so a marker can be split across THREE or more
    of them (``</`` + ``thi`` + ``nk>``). Reading only the next part missed
    those and rendered a raw sentinel, which is the injection this pass exists
    to stop; the OpenAI schema allows any number of text parts per message
    (#7334).
    """
    if limit <= 0:
        return ""
    out: list[str] = []
    total = 0
    for text in texts[index + 1 :]:
        if not isinstance(text, str):
            continue
        chunk = text.strip()
        if not chunk:
            continue
        out.append(chunk)
        total += len(chunk)
        if total >= limit:
            break
    return "".join(out)


def neutralize_message_content_for_role(role: Optional[str], content):
    """Apply control-markup neutralization to message content.

    Assistant turns keep their structural think / channel / tool markup, but
    even there the turn-boundary sentinels are neutralized: replayed history is
    client-controlled and a raw one truncates that turn or injects a new one.
    String content and OpenAI text parts are rewritten; other part types pass
    through. Returns ``content`` unchanged when nothing needed rewriting.
    """
    rewrite = (
        neutralize_turn_boundary_markup
        if (role or "").strip().lower() == "assistant"
        else neutralize_non_assistant_control_markup
    )
    if isinstance(content, str):
        return rewrite(content)
    if isinstance(content, list):
        markers = (
            _TURN_BOUNDARY_MARKERS
            if (role or "").strip().lower() == "assistant"
            else _NON_ASSISTANT_CONTROL_MARKERS
        )
        # Each part as the template renders it, so a marker cut across parts is
        # spotted before the parts are rewritten.
        texts = [
            part if isinstance(part, str) else part.get("text") if isinstance(part, dict) else None
            for part in content
        ]
        # The marker may straddle the seam, so that many following chars suffice.
        lookahead = max((len(src) for src, _ in markers), default = 0)
        changed = False
        out = []
        for index, part in enumerate(content):
            # A neutral char at the seam breaks a marker only completed by what
            # follows, leaving both parts' own text intact.
            seam = ""
            if isinstance(texts[index], str):
                ahead = _rendered_lookahead(texts, index, lookahead)
                if _split_marker_boundary(texts[index], ahead, markers):
                    seam = _THINK_NEUTRAL_ZW
            if isinstance(part, str):
                new_part = rewrite(part) + seam
                changed = changed or new_part != part
                out.append(new_part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                new_text = rewrite(part["text"]) + seam
                if new_text != part["text"]:
                    out.append({**part, "text": new_text})
                    changed = True
                else:
                    out.append(part)
            else:
                out.append(part)
        return out if changed else content
    return content


# Replayed thoughts: free text the template wraps in its own thinking delimiters,
# never structural markup itself (#7066).
_ASSISTANT_REASONING_FIELDS = ("reasoning_content", "reasoning")


def neutralize_control_markup_in_messages(messages: list) -> list:
    """Return a copy of ``messages`` with non-assistant control markup neutralized.

    No-op (returns the same list object) when nothing changes, so callers can
    keep byte-identical prompts on the common path.
    """
    if not messages:
        return messages
    changed = False
    out: list = []
    for msg in messages:
        if not isinstance(msg, dict):
            out.append(msg)
            continue
        content = msg.get("content")
        new_content = neutralize_message_content_for_role(msg.get("role"), content)
        content_changed = new_content is not content and new_content != content
        # A replayed thought is free text the template wraps in its own delimiters
        # (gemma-4: between <|channel>thought and <channel|>), so a literal marker
        # inside it closes that channel early; `content` keeps real tags (#7066).
        # ``tool_call_id`` and ``name`` (the tool_response fallback Gemma-4 splices
        # in when no call id matches) get the same rewrite as the ``id`` /
        # ``function.name`` of the call they answer, so the pairs still match.
        scalar_updates = {}
        for field in (*_ASSISTANT_REASONING_FIELDS, "tool_call_id", "name"):
            value = msg.get(field)
            if isinstance(value, str) and value:
                new_value = neutralize_non_assistant_control_markup(value)
                if new_value != value:
                    scalar_updates[field] = new_value
        # Tool-call arguments are data, not prose, so they are neutralized even
        # though assistant content is preserved (#7066).
        tool_calls = msg.get("tool_calls")
        new_tool_calls = neutralize_tool_call_arguments(tool_calls)
        tool_calls_changed = new_tool_calls is not tool_calls and new_tool_calls != tool_calls
        if content_changed or tool_calls_changed or scalar_updates:
            new_msg = {**msg, **scalar_updates}
            if content_changed:
                new_msg["content"] = new_content
            if tool_calls_changed:
                new_msg["tool_calls"] = new_tool_calls
            out.append(new_msg)
            changed = True
        else:
            out.append(msg)
    return out if changed else messages


def _tokenizer_objects(tokenizer) -> tuple:
    """Return a processor/tokenizer and its distinct nested tokenizer."""
    if tokenizer is None:
        return ()
    nested = getattr(tokenizer, "tokenizer", None)
    return (tokenizer,) if nested is None or nested is tokenizer else (tokenizer, nested)


def _selected_template_strings_from_value(
    template,
    tools = None,
    *,
    prefer_tool_use: bool = True,
) -> tuple[str, ...]:
    """Return the named chat template matching HF's default selection rules."""
    tools = tools or None
    if isinstance(template, str):
        return (template,)
    if not isinstance(template, dict):
        return ()
    if prefer_tool_use and tools and isinstance(template.get("tool_use"), str):
        return (template["tool_use"],)
    if isinstance(template.get("default"), str):
        return (template["default"],)
    values = tuple(value for value in template.values() if isinstance(value, str))
    return values if len(values) == 1 else ()


def _selected_chat_template_strings(tokenizer, tools = None) -> tuple[str, ...]:
    """Return the active chat template selected for this request."""
    tools = tools or None
    getter = getattr(tokenizer, "get_chat_template", None)
    if callable(getter):
        for kwargs in ({"chat_template": None, "tools": tools}, {"tools": tools}, {}):
            try:
                selected = getter(**kwargs)
            except Exception:
                continue
            if isinstance(selected, str):
                return (selected,)
    # ProcessorMixin.apply_chat_template does not switch to "tool_use" implicitly;
    # it uses "default" unless chat_template= names another template.
    is_processor = getattr(tokenizer, "tokenizer", None) is not None and callable(
        getattr(tokenizer, "apply_chat_template", None)
    )
    return _selected_template_strings_from_value(
        getattr(tokenizer, "chat_template", None),
        tools,
        prefer_tool_use = not is_processor,
    )


def _detect_reasoning_channel_markers_from_templates(
    templates: tuple[str, ...],
) -> Optional[tuple[str, str]]:
    """Return Gemma native reasoning markers only when a template emits them."""
    if any(opener in template for template in templates for opener in _GEMMA_TEMPLATE_OPENERS):
        return _GEMMA_THOUGHT_OPEN, _GEMMA_THOUGHT_CLOSE
    return None


def detect_reasoning_channel_markers(tokenizer, tools = None) -> Optional[tuple[str, str]]:
    """Return native Gemma thought-channel markers supported by a tokenizer.

    Detection uses the active chat template rather than model names or vocabulary
    membership. Some models expose Gemma control tokens without using the native
    thought-channel response protocol, and those must keep normal
    ``skip_special_tokens`` streaming.
    """
    for obj in _tokenizer_objects(tokenizer):
        templates = _selected_chat_template_strings(obj, tools)
        if templates:
            return _detect_reasoning_channel_markers_from_templates(templates)
    return None


def detect_reasoning_channel_markers_from_template(
    template, tools = None
) -> Optional[tuple[str, str]]:
    """Return native Gemma thought-channel markers from a raw template value."""
    return _detect_reasoning_channel_markers_from_templates(
        _selected_template_strings_from_value(template, tools)
    )


def detect_reasoning_channel_markers_from_model_info(
    tokenizer,
    model_info: Optional[dict] = None,
    tools = None,
) -> Optional[tuple[str, str]]:
    """Return reasoning markers from the active or cached native template."""
    markers = detect_reasoning_channel_markers(tokenizer, tools = tools)
    if markers is not None or not isinstance(model_info, dict):
        return markers

    native_templates = (
        model_info.get("native_chat_template"),
        (model_info.get("chat_template_info") or {}).get("template"),
    )
    for template in native_templates:
        markers = detect_reasoning_channel_markers_from_template(template, tools)
        if markers is not None:
            return markers
    return None


@dataclass(frozen = True)
class ChatTemplateRenderResult:
    """Prompt plus response-protocol metadata selected by the renderer."""

    prompt: str
    reasoning_channel_markers: Optional[tuple[str, str]] = None


def _split_partial_marker(text: str, marker: str) -> tuple[str, str]:
    """Hold the longest suffix that may become ``marker`` in the next chunk."""
    for length in range(min(len(text), len(marker) - 1), 0, -1):
        if text.endswith(marker[:length]):
            return text[:-length], text[-length:]
    return text, ""


class ReasoningChannelNormalizer:
    """Incrementally convert one native reasoning channel to ``<think>``.

    The parser follows mlx-vlm's streaming boundary behavior but emits Unsloth's
    established canonical text contract. Only the configured opening and
    closing markers are consumed; tool-call and other control markers remain
    available to downstream parsers.
    """

    def __init__(self, opening_marker: str, closing_marker: str):
        self._opening_marker = opening_marker
        self._closing_marker = closing_marker
        self._buffer = ""
        self._in_reasoning = False
        self._reasoning_done = False
        self._skip_opening_newline = False

    def feed(self, text: str) -> str:
        """Consume a raw text delta and return the stable canonical delta."""
        self._buffer += text or ""
        output: list[str] = []
        while self._buffer:
            if self._reasoning_done:
                output.append(self._buffer)
                self._buffer = ""
                break

            if self._in_reasoning and self._skip_opening_newline:
                if self._buffer.startswith("\n"):
                    self._buffer = self._buffer[1:]
                self._skip_opening_newline = False
                if not self._buffer:
                    break

            marker = self._closing_marker if self._in_reasoning else self._opening_marker
            index = self._buffer.find(marker)
            if index < 0:
                stable, self._buffer = _split_partial_marker(self._buffer, marker)
                output.append(stable)
                break

            output.append(self._buffer[:index])
            self._buffer = self._buffer[index + len(marker) :]
            if self._in_reasoning:
                output.append(_THINK_CLOSE)
                self._in_reasoning = False
                self._reasoning_done = True
            else:
                output.append(_THINK_OPEN)
                self._in_reasoning = True
                self._skip_opening_newline = True
        return "".join(output)

    def finish(self) -> str:
        """Flush a naturally completed stream and close an open think block."""
        output = self.drain()
        if self._in_reasoning:
            output += _THINK_CLOSE
            self._in_reasoning = False
            self._reasoning_done = True
        return output

    def drain(self) -> str:
        """Flush buffered literal text without synthesizing a closing tag."""
        output = self._buffer
        self._buffer = ""
        return output


def normalize_reasoning_snapshots(
    stream,
    tokenizer = None,
    cancel_event = None,
    markers: Optional[tuple[str, str]] = None,
    tools = None,
):
    """Normalize a prefix-monotonic cumulative text stream when supported."""
    markers = markers or detect_reasoning_channel_markers(tokenizer, tools = tools)
    if markers is None:
        yield from stream
        return

    normalizer = ReasoningChannelNormalizer(*markers)
    raw_output = ""
    normalized_output = ""
    for snapshot in stream:
        if not snapshot.startswith(raw_output):
            raise RuntimeError("Reasoning normalization requires cumulative text snapshots")
        delta = normalizer.feed(snapshot[len(raw_output) :])
        raw_output = snapshot
        if delta:
            normalized_output += delta
            yield normalized_output

    cancelled = cancel_event is not None and cancel_event.is_set()
    tail = normalizer.drain() if cancelled else normalizer.finish()
    if tail:
        normalized_output += tail
        yield normalized_output


def detect_think_prefill(prompt: Optional[str], special_tokens = None) -> str:
    """Return the trailing open ``<think>`` prefill of a rendered prompt.

    Reasoning templates (Qwen3.6, DeepSeek-R1-style) end the generation
    prompt with ``<think>\\n`` so the model starts reasoning immediately.
    Because that opening tag is part of the *prompt*, skip_prompt streaming
    never emits it, and the frontend's ``<think>``/``</think>`` parser shows
    the reasoning as plain text instead of a thinking block. (The GGUF path
    is unaffected: llama-server's reasoning parser returns
    ``reasoning_content``, which gets re-wrapped in think tags.)

    Returns the exact prompt tail to re-emit at the start of the generated
    stream (e.g. ``"<think>\\n"``), or ``""`` when the prompt does not end
    with an open think block, including the ``enable_thinking=False`` case
    where templates prefill an already-closed ``<think>\\n\\n</think>``.

    ``special_tokens`` is the tokenizer's special-token list. If ``</think>``
    is one, the streamer's skip_special_tokens strips the model's closing tag,
    so re-emitting the open would leave an unclosed block that swallows the
    answer. In that case return ``""`` and fall back to plain text.
    """
    if not prompt:
        return ""
    open_idx = prompt.rfind(_THINK_OPEN)
    if open_idx == -1:
        return ""
    tail = prompt[open_idx:]
    if _THINK_CLOSE in tail or tail.strip() != _THINK_OPEN:
        return ""
    if special_tokens and _THINK_CLOSE in set(special_tokens):
        return ""
    return tail


logger = logging.getLogger(__name__)


def _normalize_tool_call_arguments(messages: list) -> list:
    """Coerce each assistant ``tool_calls[].function.arguments`` from a JSON
    string to a dict.

    The OpenAI wire format carries ``arguments`` as a JSON string, but some chat
    templates (e.g. the stricter Qwen tool templates shipped with mlx-community
    checkpoints) iterate ``arguments.items()`` and raise
    ``TypeError: Can only get item pairs from a mapping.`` on the string form
    when a prior tool call is re-rendered on the next turn. A dict works on both
    strict and lenient templates, so parse the string; leave non-JSON or non-dict
    values untouched. Returns the original list unchanged when nothing needed
    coercing (no copy)."""
    mutated = False
    out: list = []
    for msg in messages:
        tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else None
        if not tool_calls:
            out.append(msg)
            continue
        new_calls = []
        msg_changed = False
        for call in tool_calls:
            fn = call.get("function") if isinstance(call, dict) else None
            args = fn.get("arguments") if isinstance(fn, dict) else None
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                except (ValueError, TypeError):
                    parsed = None
                if isinstance(parsed, dict):
                    call = {**call, "function": {**fn, "arguments": parsed}}
                    msg_changed = True
            new_calls.append(call)
        if msg_changed:
            out.append({**msg, "tool_calls": new_calls})
            mutated = True
        else:
            out.append(msg)
    return out if mutated else messages


def _take_tool_result(pending: list, call_id) -> Optional[dict]:
    if call_id:
        for i, result in enumerate(pending):
            if result.get("tool_call_id") == call_id:
                return pending.pop(i)
    for i, result in enumerate(pending):
        if not result.get("tool_call_id"):
            return pending.pop(i)
    return None


def _split_parallel_tool_calls(messages: list) -> list:
    """Llama 3.x templates render one call per message, so split parallel calls
    into consecutive single-call messages, each followed by its own result."""
    if not any(isinstance(m, dict) and len(m.get("tool_calls") or ()) > 1 for m in messages):
        return messages

    out: list = []
    i = 0
    total = len(messages)
    while i < total:
        msg = messages[i]
        calls = msg.get("tool_calls") if isinstance(msg, dict) else None
        if not calls or len(calls) <= 1:
            out.append(msg)
            i += 1
            continue

        # Tool results right after this message answer its calls.
        j = i + 1
        pending: list = []
        while (
            j < total
            and isinstance(messages[j], dict)
            and messages[j].get("role") in ("tool", "ipython")
        ):
            pending.append(messages[j])
            j += 1

        for idx, call in enumerate(calls):
            piece = {**msg, "tool_calls": [call]}
            if idx:
                piece["content"] = ""
            out.append(piece)
            result = _take_tool_result(pending, call.get("id") if isinstance(call, dict) else None)
            if result is not None:
                out.append(result)
        out.extend(pending)
        i = j
    return out


def apply_chat_template_for_generation(
    tokenizer,
    messages: list,
    *,
    tools: Optional[list] = None,
    enable_thinking: Optional[bool] = None,
    reasoning_effort: Optional[str] = None,
    preserve_thinking: Optional[bool] = None,
) -> str:
    """Render the chat prompt. Try richest kwargs first; drop one
    group at a time on TypeError. Jinja / missing-variable errors
    propagate."""
    reasoning_kwargs: dict = {}
    if enable_thinking is not None:
        reasoning_kwargs["enable_thinking"] = enable_thinking
    if reasoning_effort is not None:
        reasoning_kwargs["reasoning_effort"] = reasoning_effort
    if preserve_thinking is not None:
        reasoning_kwargs["preserve_thinking"] = preserve_thinking

    attempts: list[dict] = []
    if tools and reasoning_kwargs:
        attempts.append({"tools": tools, **reasoning_kwargs})
    if tools:
        attempts.append({"tools": tools})
    if reasoning_kwargs:
        attempts.append(dict(reasoning_kwargs))
    attempts.append({})

    def _render(msgs: list) -> str:
        last_exc: Optional[Exception] = None
        for kwargs in attempts:
            try:
                return tokenizer.apply_chat_template(
                    msgs,
                    tokenize = False,
                    add_generation_prompt = True,
                    **kwargs,
                )
            except TypeError as e:
                last_exc = e
                continue
            except Exception as e:
                last_exc = e
                break
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("apply_chat_template_for_generation: no attempt produced a result")

    try:
        return _render(neutralize_control_markup_in_messages(messages))
    except Exception:
        # Retry with repairs applied cumulatively. Originals render first, so
        # working templates stay byte-identical. Repairs run on the RAW messages
        # because ``_normalize_tool_call_arguments`` parses ``arguments`` as JSON
        # and the neutralizer injects word joiners; neutralization is applied last,
        # right before each render, as on the first attempt above.
        candidates: list = []
        normalized = _normalize_tool_call_arguments(messages)
        if normalized is not messages:
            candidates.append(normalized)
        split = _split_parallel_tool_calls(normalized)
        if split is not normalized:
            candidates.append(split)
        for candidate in candidates:
            try:
                return _render(neutralize_control_markup_in_messages(candidate))
            except Exception:
                continue
        raise


def render_native_template(
    *,
    model_info: dict,
    active_model_name: Optional[str],
    messages: list,
    tools: list,
    enable_thinking: Optional[bool] = None,
    reasoning_effort: Optional[str] = None,
    preserve_thinking: Optional[bool] = None,
    apply_fn = None,
    hf_token: Optional[str] = None,
    return_metadata: bool = False,
):
    """Render ``messages`` + ``tools`` with the model's NATIVE chat template.

    Some Unsloth override templates (e.g. ``mistral``, ``gemma-4``) do not emit
    the ``tools`` schema, so a tool-calling turn silently stops advertising tools.
    The native template ships in the model repo and carries the family's
    tool-calling syntax. It is loaded straight from the repo (bypassing any
    override on the live tokenizer) and cached on ``model_info``. Returns the
    rendered prompt only if the native template actually emits the tools (render
    differs with vs without tools); otherwise ``None``. With ``return_metadata``,
    returns ``ChatTemplateRenderResult`` so callers can stream with the response
    protocol selected by this request's template.

    ``hf_token`` is the token the model was loaded with -- passed to the repo load
    so a gated/private model's native template can still be fetched (otherwise the
    fallback fails silently and keeps the override prompt that dropped tools).

    ``trust_remote_code`` is sourced from ``model_info`` (the value the model was
    actually loaded with) rather than a call-site argument, so the native-template
    reload uses exactly the consent already granted at load. A custom-code tokenizer
    repo raises in ``AutoTokenizer.from_pretrained`` unless ``trust_remote_code`` is
    passed, so without this the fallback fails silently and keeps the tool-dropping
    prompt for a model the user already consented to run remote code for. For a LoRA
    adapter the reload targets the base model, whose remote code was gated and loaded
    under the same stored flag, so re-passing it executes no unconsented code.
    """
    # ``apply_fn`` lets a backend inject its own render; defaults to the module helper.
    if apply_fn is None:
        apply_fn = apply_chat_template_for_generation
    native_tpl = model_info.get("native_chat_template")
    if native_tpl is None:
        # A LoRA adapter's native template lives on the base model, not the adapter id.
        template_source = model_info.get("base_model") or active_model_name
        # Re-use the load-time trust_remote_code so a custom-code tokenizer repo can
        # instantiate its class (the stored flag already covers template_source).
        trust_remote_code = bool(model_info.get("trust_remote_code", False))
        try:
            from transformers import AutoTokenizer
            nt = AutoTokenizer.from_pretrained(
                template_source,
                token = hf_token if hf_token and hf_token.strip() else None,
                trust_remote_code = trust_remote_code,
            )
            native_tpl = nt.chat_template or False
        except Exception as exc:
            logger.warning(
                "Could not load native chat template for '%s': %s",
                template_source,
                exc,
            )
            # A failed fetch is not "no template": leave the sentinel unset so the next
            # call retries (caching False would pin the tool-dropping override).
            return None
        model_info["native_chat_template"] = native_tpl
    if not native_tpl:
        return None

    tokenizer = model_info.get("tokenizer") or model_info.get("processor")
    if tokenizer is None:
        return None
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    # Render on a shallow copy: mutating the shared tokenizer.chat_template (outside the
    # generation lock) races concurrent requests.
    try:
        render_tokenizer = copy.copy(tokenizer)
        render_tokenizer.chat_template = native_tpl
    except Exception as exc:
        logger.warning(
            "Could not clone tokenizer for native-template render of '%s': %s",
            active_model_name,
            exc,
        )
        return None
    try:
        with_tools = apply_fn(
            render_tokenizer,
            messages,
            tools = tools,
            enable_thinking = enable_thinking,
            reasoning_effort = reasoning_effort,
            preserve_thinking = preserve_thinking,
        )
        no_tools = apply_fn(
            render_tokenizer,
            messages,
            tools = None,
            enable_thinking = enable_thinking,
            reasoning_effort = reasoning_effort,
            preserve_thinking = preserve_thinking,
        )
    except Exception as exc:
        logger.warning(
            "Native-template tool render failed for '%s': %s",
            active_model_name,
            exc,
        )
        return None
    if with_tools == no_tools:
        return None
    if return_metadata:
        return ChatTemplateRenderResult(
            with_tools,
            _detect_reasoning_channel_markers_from_templates(
                _selected_template_strings_from_value(native_tpl, tools)
            ),
        )
    return with_tools


def render_with_native_template_fallback(
    *,
    formatted_prompt: str,
    tokenizer,
    model_info: dict,
    active_model_name: Optional[str],
    messages: list,
    tools: Optional[list],
    enable_thinking: Optional[bool] = None,
    reasoning_effort: Optional[str] = None,
    preserve_thinking: Optional[bool] = None,
    apply_fn = None,
    hf_token: Optional[str] = None,
    return_metadata: bool = False,
):
    """Return ``formatted_prompt``, swapping in a native-template render when an
    override template dropped the ``tools`` schema.

    If ``tools`` were requested but the live render is identical with and without
    them (detected by comparison, robust against tool names in the system prompt),
    re-render with the model's native template. Shared by the transformers and MLX
    backends so both advertise tools consistently. ``hf_token`` is forwarded so a
    gated/private model's native template can still be fetched. With
    ``return_metadata``, returns the selected prompt plus reasoning-channel markers
    for the exact template used by this request."""
    live_markers = detect_reasoning_channel_markers(tokenizer, tools = tools)

    def _result(prompt: str, markers = live_markers):
        if return_metadata:
            return ChatTemplateRenderResult(prompt, markers)
        return prompt

    if not tools:
        # Gemma 4 can emit its native reasoning protocol even when a generation-time
        # Unsloth override rendered a marker-free prompt. Preserve the live-verified
        # no-tools thinking behavior without letting cached native metadata describe
        # unrelated tool prompts that kept the active override.
        markers = live_markers
        if markers is None:
            markers = detect_reasoning_channel_markers_from_model_info(
                tokenizer, model_info, tools = None
            )
        return _result(formatted_prompt, markers)
    if apply_fn is None:
        apply_fn = apply_chat_template_for_generation
    # Probe whether the live template dropped the schema. A tools-requiring template
    # can raise here; on any error keep the valid tools prompt rather than lose it.
    try:
        probe_no_tools = apply_fn(
            tokenizer,
            messages,
            tools = None,
            enable_thinking = enable_thinking,
            reasoning_effort = reasoning_effort,
            preserve_thinking = preserve_thinking,
        )
    except Exception as exc:
        logger.warning(
            "No-tools probe failed for '%s'; keeping the existing tools prompt: %s",
            active_model_name,
            exc,
        )
        return _result(formatted_prompt)
    if formatted_prompt != probe_no_tools:
        return _result(formatted_prompt)  # template already emits the tools schema
    native_prompt = render_native_template(
        model_info = model_info,
        active_model_name = active_model_name,
        messages = messages,
        tools = tools,
        enable_thinking = enable_thinking,
        reasoning_effort = reasoning_effort,
        preserve_thinking = preserve_thinking,
        apply_fn = apply_fn,
        hf_token = hf_token,
        return_metadata = return_metadata,
    )
    if native_prompt:
        logger.info(
            "Override template for '%s' dropped tool schemas; using the model's "
            "native template for this tool-calling turn.",
            active_model_name,
        )
        return native_prompt
    return _result(formatted_prompt)
