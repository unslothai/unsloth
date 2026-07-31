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
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Sentinel for "this argument string is not JSON we can walk", distinct from a payload
# that legitimately decodes to None.
_UNPARSED = object()

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_GEMMA_CHANNEL_START = "<|channel>"
_GEMMA_THOUGHT_OPEN = "<|channel>thought"
_GEMMA_THOUGHT_CLOSE = "<channel|>"
_GEMMA_TEMPLATE_OPENERS = (
    _GEMMA_THOUGHT_OPEN + "\n",
    _GEMMA_THOUGHT_OPEN + "\\n",
    _GEMMA_THOUGHT_OPEN + _GEMMA_THOUGHT_CLOSE,
)

# Chat-template control markup must not reach the prompt as raw text from a user /
# system / tool turn: a literal "</think>" ends the reasoning block early, and
# "<|start|>assistant<|channel|>final<|message|>" in a tool result forges an
# assistant turn (#7066). One lookahead over the four shapes the templates emit
# (<|name|>/<|name>, <name>/</name>, <name|>, [NAME]/[/NAME]), so a single sub()
# breaks any marker by inserting one space after the opener. The name list is closed
# on purpose: bare words match only in the pipe shape and brackets only in the exact
# uppercase spelling, so "<div>", "List<String>", "[1]" and "[inst]" stay as typed.
# [THINK] is absent because no template emits it; the output parsers consume it.
# DeepSeek is the one family that delimits with the fullwidth bar U+FF5C rather than
# "|", so it needs its own branch, written as \uXXXX escapes to keep this file pure
# ASCII. That branch is the one exception to the closed list: it matches ANY Latin /
# U+2581 name between fullwidth bars, because DeepSeek spells a dozen of them and keeps
# adding more. The shape is distinctive enough to carry it -- "<" immediately followed by
# a fullwidth bar is not something prose produces -- but a CJK author writing a
# fullwidth-bar placeholder around a Latin key does get rewritten. Restricting the
# charset keeps it off real CJK content, which is the common case.
_CONTROL_MARKUP = re.compile(
    r"<(?="
    # "/?" after the bar because Phi-4 Mini closes with "<|/tool|>" and "<|/tool_call|>"
    # (ollama_template_mappers.py:1023, 1029) rather than a separate closing name, so an
    # MCP description carrying one closed the catalog and rose to system level (#7066).
    r"\|/?(?:(?:start|end)_(?:header_id|of_role)|tool(?:_call|_response)?"
    # Kimi K2 / Moonshot wrap history in a section and each call in a begin/end
    # pair (tool_call_parser.py:20, 55-56, 86-89); none of them is the short
    # "tool_call" spelling, so a paste could fabricate a historical call (#7066).
    r"|tool_calls?(?:_section)?_(?:begin|end)|tool_call_argument_begin"
    r"|end(?:_of_(?:turn|text))?"
    # Document boundaries: begin_of_text is Llama-3.1 / Llama-4's BOS and endoftext is
    # the GPT-2-lineage EOS that Qwen2.5, Qwen3, Phi, gpt-oss and GLM-4.5 all still
    # carry. Reserved vocabulary, so this is the same argument as the media
    # placeholders below: the trie splits a pasted copy back out to the real token id,
    # so client text lands in the prompt as a document break the template never
    # opened, mid-conversation (#7066).
    r"|begin_of_text|endoftext"
    # header_start / header_end / <|eot|> are Llama-4's spelling of Llama-3's
    # start_header_id / end_header_id / eot_id, and im_sep is Phi-4's role separator.
    r"|header_(?:start|end)|im_(?:start|end|sep)"
    r"|assistant|constrain|channel|message|eo[tm](?:_id)?|final"
    # TML Inkling's native call envelope, "<|message_model|>NAME
    # <|content_invoke_tool_json|>{...}<|end_message|>". Longer than the "message" and
    # "end" names above, so all three were passing through even though the repo parses
    # them as a tool call (tool_call_parser.py:58, tool_healing.py:129-132, 701-707).
    r"|message_model|content_invoke_tool_json|end_message"
    # Command-R / Aya spell every delimiter in caps: <|START_OF_TURN_TOKEN|> etc.
    r"|(?:START|END)_OF_TURN_TOKEN|(?:USER|SYSTEM|CHATBOT)_TOKEN"
    # Gemma-4's media placeholders (its own image_token / audio_token / video_token,
    # also chat_templates.py:917-921) and Llama-3.1's built-in-tool sentinel
    # (chat_templates.py:496). These are reserved vocabulary, so a pasted copy is not
    # cosmetic: a processor counts "<|image|>" against the media it was handed, and one
    # extra is a hard ValueError out of MllamaProcessor / Gemma4Processor on the very
    # vision and audio renders neutralized above (#7066). The optional close also
    # covers Gemma-4's <|image> / <|audio> openers.
    r"|image|audio|video|python_tag"
    r"|return|system|start|think|turn|user|call|\")\|?>"
    # The parser also recognises the space and backslash-escaped spellings of the same
    # openers (tool_call_parser.py:47-53, 62-66), so the class has to admit both. The
    # name still has to start with a letter, which keeps "<\uff5c \uff5c>" and other
    # fullwidth-punctuation pairs out while the bars keep this off ordinary text.
    r"|\uff5c[A-Za-z][A-Za-z\u2581_ \\]{0,39}\uff5c>"
    # "tools" is the Qwen / Hermes tool catalog block. The template interpolates the
    # system message INTO the same system turn that holds "<tools>...</tools>", so a
    # "</tools>" in client text closes the real catalog and everything after it reads
    # as a tool declaration the server never registered (#7066).
    # Gemma 3 / 3n spell their media placeholders as bare tags rather than the Gemma-4
    # pipe shape (chat_templates.py:677, 845-847), for the same reason those are here: a
    # pasted copy is a placeholder for media the processor was never handed.
    # GLM 4.5-4.7 and Qwen3.5 nest their call protocol inside the outer tool tag:
    # "<tool_call>NAME<arg_key>k</arg_key><arg_value>v</arg_value>" and
    # "<function=name><parameter=k>v</parameter></function>". tool_call_parser.py
    # treats all of them as structural, so a replayed value or a tool result can
    # close the current value and inject another key or call (#7066).
    # Gemma spells its own BOS / EOS as bare tags rather than the pipe shape, and both
    # are in its tokenizer's added-token trie (google/gemma-3-4b-it), so this is the
    # bare-tag half of the begin_of_text / endoftext pair above.
    # "</s>" is the Llama-2 / Mistral / Zephyr EOS (their own tokenizer_config.json), and
    # "<s>" the matching BOS, so the same trie argument applies. This is the one addition
    # that collides with a real HTML tag, the strikethrough "<s>", and the collision is
    # accepted for the reason "<think>" and "<tools>" are: a live document boundary in a
    # prompt is worth more than a space in a rare tag, and the rewrite stays readable.
    r"|/?(?:(?:start|end)_of_turn|tool_(?:call|response)|tools|think|eos|bos|s"
    r"|start_of_image|image_soft_token|audio_soft_token"
    r"|arg_key|arg_value|function|parameter|param)>"
    # The opening halves carry an "=value", so they need their own anchor. The parser
    # accepts both spellings of the opener (tool_call_parser.py:_TOOL_CLOSED_PATS), so
    # the attribute form needs its own alternative rather than riding on "function=".
    r"|(?:function|parameter)=|(?:function|param(?:eter)?)\s+name=\""
    r"|(?:tool(?:_call|_response)?|channel|turn)\|>"
    r")"
    # "[ARGS]", Mistral v11's "[CALL_ID]" and Devstral's "[TOOL_CONTENT]" are absent on
    # purpose. All three are metadata WITHIN a block, never its opener, and the openers
    # ("[TOOL_CALLS]", "[TOOL_RESULTS]") are broken, so none of them can start or close
    # anything alone; on the way back in they are read by tool_healing.py:36-56, out of
    # model output rather than out of a prompt. "[ARGS]" also collides with real text:
    # it is the standard CLI-synopsis metavariable ("usage: tool [OPTIONS] [ARGS]"), it
    # appears as a live string in this repo, and inside a schema "enum" / "pattern" the
    # rewrite would turn it into a grammar literal the model is then forced to emit.
    r"|\[(?=/?(?:INST|SYSTEM_PROMPT|AVAILABLE_TOOLS|TOOL_RESULTS)\]|TOOL_CALLS\])"
    # Llama-2 opens its system block with "<<SYS>>" INSIDE the first [INST], so the
    # doubled angle is the opener, not a single "<". Both meta-llama/Llama-2-*-chat-hf's
    # own template and the "llama" entry this repo's MODEL_TO_TEMPLATE_MAPPER installs
    # at generate time emit it, and a later user turn carries no system block at all,
    # so a pasted pair invents one (#7066). Anchored on the second "<" of the pair, so
    # "<SYS>>", a C++ "cout << SYS" and a heredoc "<<-'SYS'" all stay as typed.
    r"|(?<=<)<(?=/?SYS>>)"
)

# Turn-boundary subset, for replayed ASSISTANT content: that text is
# client-controlled too, so a raw boundary in it truncates or forges a turn (#7066).
# Everything else stays byte-identical, because the assistant's own think / channel /
# tool markup is structure the template re-renders. Boundaries also cover the bare
# Zephyr / Phi-3 role sentinels, Granite's <|start_of_role|> ... <|end_of_text|>, the
# Llama-4 and Command-R turn tokens, DeepSeek's fullwidth role markers and the
# Mistral / Llama-2 [INST] / [SYSTEM_PROMPT] bracket pairs. DeepSeek's fullwidth TOOL
# markers stay out for the same reason <|tool_call> does: they are the assistant's own.
_TURN_BOUNDARY_MARKUP = re.compile(
    r"<(?="
    r"\|/?(?:(?:start|end)_(?:header_id|of_role)|im_(?:start|end|sep)"
    r"|end(?:_of_(?:turn|text))?|eo[tm](?:_id)?|header_(?:start|end)"
    # A document boundary is never the assistant's own structure, so unlike think /
    # channel / tool markup these belong in the replay subset too.
    r"|begin_of_text|endoftext"
    r"|(?:START|END)_OF_TURN_TOKEN|(?:USER|SYSTEM|CHATBOT)_TOKEN"
    # A media placeholder is reserved vocabulary, not reasoning or tool structure, so a
    # replayed one is an extra placeholder the processor was handed no media for and the
    # Gemma / mllama count check fails. Never legitimate in a replay, so it belongs here
    # even though think / channel / tool markup does not.
    r"|image|audio|video|python_tag"
    # A tool RESULT is the tool role's structure, not the assistant's, so a replay
    # carrying one fabricates an observation the model reads as trusted context. The
    # tool CALL spellings stay out, because those the assistant really does emit.
    r"|tool_response"
    r"|assistant|return|system|start|turn|user|call)\|?>"
    r"|\uff5c(?:User|Assistant|(?:begin|end)\u2581of\u2581sentence)\uff5c>"
    # "/?" for the same reason the control pattern has it: Gemma's delimiters are bare
    # tags, so a replayed "</start_of_turn>" is as much a boundary as "<start_of_turn>".
    r"|/?(?:(?:start|end)_of_turn|eos|bos|s|tool_response"
    r"|start_of_image|image_soft_token|audio_soft_token)>"
    r"|(?:turn|tool_response)\|>"
    r")"
    # Same split in the bracket family: Mistral renders assistant .Content verbatim
    # (ollama_template_mappers.py:125-127) and spells a tool observation
    # "[TOOL_RESULTS]...[/TOOL_RESULTS]" (:133) and the catalog "[AVAILABLE_TOOLS]"
    # (:123), so a replay can forge either. "[TOOL_CALLS]" is left out: that one the
    # assistant does emit (:129).
    r"|\[(?=/?(?:INST|SYSTEM_PROMPT|AVAILABLE_TOOLS|TOOL_RESULTS)\])"
    # Llama-2's system section is a boundary for the same reason [SYSTEM_PROMPT] is:
    # the template only ever emits it in the first user turn, never in an assistant one.
    r"|(?<=<)<(?=/?SYS>>)"
)


# TTS is not a chat template: the codec prompt is built by concatenation, so the text
# sits between codec delimiters instead of template ones (inference.py:1918, 1948-1954,
# llama_cpp.py:_TTS_PROMPTS). A closer pasted into it ends the text segment early or
# opens the audio / global-token segment, which yields truncated or garbled audio (#7066).
# Per codec, and deliberately NOT the chat sweep: the text here is meant to be SPOKEN, so
# "please say <s>hello</s>" or "read [INST] literally" has to reach the tokenizer as
# typed. Only what actually delimits the active codec's prompt, plus its real stop
# tokens, is structure.
_TTS_MARKUP_BY_CODEC = {
    # <custom_token_3>{text}<|eot_id|><custom_token_4>, stop <custom_token_2>. Exactly
    # those three, not any number: the transformers path spells the same ones as bare ids
    # (inference.py:1886-1888), so "say <custom_token_999>" is ordinary text here.
    "snac": re.compile(r"<(?=custom_token_[234]>|\|eot_id\|>)"),
    # <|task_tts|><|start_content|>{text}<|end_content|><|start_global_token|>,
    # stop <|im_end|> and </s>.
    "bicodec": re.compile(
        r"<(?=\|(?:task_tts|(?:start|end)_(?:content|global_token|semantic_token)"
        r"|im_end)\|>|/s>)"
    ),
    # <|im_start|>\n<|text_start|>{text}<|text_end|>\n<|audio_start|>
    # <|global_features_start|>\n, stop <|im_end|> and <|audio_end|>.
    "dac": re.compile(
        r"<(?=\|(?:im_(?:start|end)|text_(?:start|end)|audio_(?:start|end)"
        r"|global_features_(?:start|end))\|>)"
    ),
    # CSM has no sentinel of its own: _generate_csm interpolates into "[speaker_id]text"
    # (inference.py:1911-1918) and the processor tokenizes that directly. The only
    # structure is the leading speaker id, and only in the leading position can a paste
    # shadow the real one, so nothing else in the text is touched.
    "csm": re.compile(r"\A\[(?=\d+\])"),
}
# An unrecognised codec gets the union: still far narrower than the chat sweep, but it
# does not assume a prompt shape this module has not seen.
_TTS_MARKUP_DEFAULT = re.compile(
    "|".join(f"(?:{pattern.pattern})" for pattern in _TTS_MARKUP_BY_CODEC.values())
)


def _spaced_out(pattern, text: str) -> str:
    """Insert one space after every marker opener *pattern* found."""
    if not text or ("<" not in text and "[" not in text):
        return text
    return pattern.sub(r"\g<0> ", text)


def neutralize_control_markup(text: str) -> str:
    """Break control markup in free text by spacing out the opener (#7066).

    "</think>" becomes "< /think>", "[/INST]" becomes "[ /INST]": readable, but no
    longer a delimiter to the template, the think extractor or the stop-sequence
    matcher. A plain space, because every tokenizer vocabulary has one; U+2060 can
    fall back to byte junk.
    """
    return _spaced_out(_CONTROL_MARKUP, text)


def neutralize_turn_boundary_markup(text: str) -> str:
    """Break only the turn-boundary sentinels, for replayed assistant text (#7066)."""
    return _spaced_out(_TURN_BOUNDARY_MARKUP, text)


def neutralize_tts_prompt_text(text: str, audio_type = None) -> str:
    """Break the active codec's own delimiters in the text of a TTS prompt (#7066).

    Scoped to *audio_type* on purpose: this text is going to be spoken, so anything that
    is not structure in THIS codec's prompt has to survive byte-exact.
    """
    return _spaced_out(_TTS_MARKUP_BY_CODEC.get(audio_type, _TTS_MARKUP_DEFAULT), text)


def _neutralize_leaves(
    value,
    rewrite,
    warn_on_key_collision: bool = False,
):
    """Apply *rewrite* to every string leaf, keys included, of a nested structure.

    Iterative rather than recursive: how deep this goes is the client's choice, and a
    schema that ``json.loads`` accepts must not be able to exhaust the interpreter stack
    and turn the request into a 500. Containers are rebuilt in reverse breadth-first
    order, so a child is always finished before the parent that holds it, and a repeated
    or self-referencing node is visited once.

    With *warn_on_key_collision*, a dict whose keys collide after the rewrite is logged.
    Rewriting keys is not injective: "a<think>" and "a< think>" both land on
    "a< think>", so such a dict keeps only the last value. Reaching a collision takes two
    keys differing only in markup the rewrite touches, which no real schema has, and the
    alternative -- keeping one key raw so both survive -- would put the markup back in
    the prompt. So the merge stands and is logged.
    """
    if isinstance(value, str):
        return rewrite(value)
    if not isinstance(value, (dict, list)):
        return value

    order: list = []
    queue: list = [value]
    seen = {id(value)}
    while queue:
        node = queue.pop()
        order.append(node)
        for child in node.values() if isinstance(node, dict) else node:
            if isinstance(child, (dict, list)) and id(child) not in seen:
                seen.add(id(child))
                queue.append(child)

    def _leaf(item):
        return rewrite(item) if isinstance(item, str) else item

    done: dict = {}
    for node in reversed(order):
        if isinstance(node, dict):
            rebuilt: dict = {}
            for key, item in node.items():
                new_key = rewrite(key) if isinstance(key, str) else key
                if warn_on_key_collision and new_key in rebuilt:
                    logger.warning(
                        "Two argument keys neutralize onto %r; keeping the later value.",
                        new_key,
                    )
                rebuilt[new_key] = done[id(item)] if id(item) in done else _leaf(item)
            done[id(node)] = rebuilt
        else:
            done[id(node)] = [done[id(item)] if id(item) in done else _leaf(item) for item in node]
    return done[id(value)]


# A media payload stays opaque: it is a URL or a base64 blob the processor resolves, not
# prompt text, and rewriting one breaks the fetch rather than the prompt. Gated on the
# part's own type, because "data" and "url" are ordinary content keys on anything else --
# a "{'type': 'json', 'data': ...}" part is prompt text that Llama-3.1 serializes with
# tojson, and exempting it unconditionally put the markup straight back in the prompt.
# "input_image" is in here because the MLX image counter recognises it
# (mlx_inference.py:130) and the registered VLM renderer passes those messages through
# this sweep, so its payload is a URL to fetch, not prompt text.
_MEDIA_PART_TYPES = frozenset(
    {"image", "image_url", "input_image", "input_audio", "audio", "audio_url", "video", "video_url"}
)
_OPAQUE_PART_KEYS = frozenset(
    {"image_url", "input_audio", "image", "audio", "video", "url", "data", "b64_json"}
)


def _neutralize_content_parts(
    content: list,
    rewrite,
    media_opaque: bool = True,
):
    """Neutralize an OpenAI-style content parts list (#7066).

    Two things the naive per-part rewrite missed. A part that is a mapping without a
    string "text" was passed through whole, yet /generate/stream accepts one and Llama-3.1
    serializes the entire iterable with tojson, so any leaf of it reaches the prompt. And
    a marker split across two adjacent text parts survived both sweeps, because Gemma-4
    concatenates them with no separator (gemma-4.jinja:304) and reassembles the opener.
    Inserting whitespace between them is not a fix, since the sibling paths trim each part
    (gemma-4.jinja:339), so a run that only becomes a marker once joined is swept as one
    string and collapses into a single part. A run that is already clean keeps its parts.
    """
    parts: list = []
    for part in content:
        if isinstance(part, str):
            parts.append(rewrite(part))
        elif isinstance(part, dict):
            # isinstance first: GenerateRequest.messages is an untyped List[dict], so
            # "type" can be a list or a dict and an unhashable value would raise
            # TypeError out of the set lookup and 500 the request before rendering.
            part_type = part.get("type")
            if isinstance(part_type, str) and part_type in _MEDIA_PART_TYPES and media_opaque:
                opaque = {k: v for k, v in part.items() if k in _OPAQUE_PART_KEYS}
                swept = _neutralize_leaves(
                    {k: v for k, v in part.items() if k not in _OPAQUE_PART_KEYS}, rewrite
                )
                parts.append({**swept, **opaque} if opaque else swept)
            else:
                # Every field, not just "text": a part carrying both is still serialized
                # whole by the tojson templates, so sweeping only "text" left the rest
                # live (#7066).
                parts.append(_neutralize_leaves(part, rewrite))
        else:
            parts.append(part)

    def _text_of(part):
        if isinstance(part, str):
            return part
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            return part["text"]
        return None

    # Every text part in the list can end up adjacent, not just the contiguous ones.
    # Gemma-4 concatenates them all into one string before emitting any media placeholder
    # (gemma-4.jinja:301-306), and its message loop simply skips a type it does not know
    # (:334-344), so a part in between is no separator at all. The check therefore spans
    # the whole list, in both the raw and the per-part-trimmed spelling the renderer
    # produces.
    texts = [_text_of(part) for part in parts]
    carriers = [index for index, text in enumerate(texts) if text is not None]
    if len(carriers) > 1:
        raw = "".join(texts[index] for index in carriers)
        trimmed = "".join(texts[index].strip() for index in carriers)
        if rewrite(raw) != raw or rewrite(trimmed) != trimmed:
            # Only a list a paste split mid-marker collapses; the joined text lands on the
            # first carrier so it stays broken whichever way the renderer joins them.
            swept = rewrite(trimmed)
            first = parts[carriers[0]]
            merged = swept if isinstance(first, str) else {**first, "text": swept}
            dropped = set(carriers[1:])
            return [
                merged if index == carriers[0] else part
                for index, part in enumerate(parts)
                if index not in dropped
            ]
    return parts


def _differs(new, old) -> bool:
    """True when the rewrite changed *old* into *new*.

    The client controls how deep these structures nest and ``==`` recurses in C, so a
    comparison that overflows must not turn the request into a 500. An overflow counts
    as changed, which keeps the neutralized copy: the safe direction (#7066).
    """
    try:
        return new != old
    except RecursionError:
        return True


def _neutralize_argument_leaves(value):
    """Break control markup in every string leaf (keys included) of *value*."""
    return _neutralize_leaves(value, neutralize_control_markup, warn_on_key_collision = True)


def _neutralized_arguments(arguments):
    """Neutralize a replayed call's ``arguments``, or None when already clean.

    OpenAI ships ``arguments`` as JSON *text*, and every consumer decodes it back to an
    object AFTER this runs: ``_normalize_tool_call_arguments`` below re-renders through
    ``json.loads`` when a template rejects a string, and llama.cpp does the same in
    ``workaround::func_args_not_string`` for any template whose capability probe reports
    object arguments. So rewriting the raw text lets "\\u003ctool_call|\\u003e" through
    untouched and the decoded marker forges a turn (#7066). Parse first, rewrite the
    decoded leaves, then re-serialize, and leave a clean payload byte-identical so the
    prefix cache still hits.
    """
    if isinstance(arguments, str):
        decoded = safe = _UNPARSED
        try:
            decoded = json.loads(arguments)
            safe = _neutralize_argument_leaves(decoded)
        # RecursionError as well as a parse error: json.loads blows the stack at roughly
        # 1000 levels of nesting, and so does the walk, so an otherwise valid
        # '[' * 1000 + '0' + ']' * 1000 would turn a request the server used to forward
        # into a 500. Fall through to the text rewrite, which cannot recurse -- nothing
        # downstream can decode that payload either, so no marker hides behind it.
        except (ValueError, TypeError, RecursionError):
            decoded = safe = _UNPARSED
        if decoded is not _UNPARSED:
            if safe != decoded:
                # ensure_ascii keeps a decoded lone surrogate ("\ud800") as an escape:
                # emitting it raw makes the outer request unencodable and raises
                # UnicodeEncodeError on a payload that used to forward fine (#7066).
                return json.dumps(safe, ensure_ascii = True)
            # Parsed clean: the text itself cannot hold a marker the decode would show.
            return None
    new_arguments = _neutralize_argument_leaves(arguments)
    return new_arguments if new_arguments != arguments else None


def _neutralize_replayed_tool_call(tool_calls: list) -> list:
    """Neutralize a replayed tool call's name and arguments, keeping "id" exact.

    Gemma-4 renders "<|tool_call>call:NAME{key:<|"|>value<|"|>}<tool_call|>", so an
    argument or a name echoing pasted text can close the call block and open a
    "<|tool_response>" or "<|turn>model" of its own (#7066). The rewrite is the
    identity on every dispatchable name (Studio composes ^[a-zA-Z0-9_-]{1,64}$), and
    a tool result's "name" takes the same rewrite, so the two still agree when
    Gemma-4 pairs them by name.

    Both replay shapes are covered: the OpenAI nested one and the flat
    {"id", "name", "arguments"} one that every template's "if tool_call.function"
    guard exists to render.
    """
    out: list = []
    for call in tool_calls:
        if not isinstance(call, dict):
            out.append(call)
            continue
        function = call.get("function")
        # Same fallback the catalog below takes, for the same reason: Harmony / gpt-oss,
        # Qwen 2.5 / 3, Granite-4 and Llama-4 all guard with "{%- if tool_call.function
        # %}{%- set tool_call = tool_call.function %}" and otherwise read "name" /
        # "arguments" off the call itself, so a flat replay renders the identical
        # concatenation and needs the identical rewrite (#7066).
        target = function if isinstance(function, dict) else call
        updates: dict = {}
        name = target.get("name")
        if isinstance(name, str) and name:
            new_name = neutralize_control_markup(name)
            if new_name != name:
                updates["name"] = new_name
        # Harmony concatenates "content_type" straight before "<|message|>"
        # (chat_templates.py:1332-1334), so a replayed "json<|message|><|end|><|start|>"
        # closes the commentary call and opens an assistant channel of its own (#7066).
        content_type = target.get("content_type")
        if isinstance(content_type, str) and content_type:
            new_content_type = neutralize_control_markup(content_type)
            if new_content_type != content_type:
                updates["content_type"] = new_content_type
        arguments = target.get("arguments")
        if arguments is not None:
            new_arguments = _neutralized_arguments(arguments)
            if new_arguments is not None:
                updates["arguments"] = new_arguments
        if not updates:
            out.append(call)
        elif target is function:
            out.append({**call, "function": {**function, **updates}})
        else:
            # "id" is the dispatch handle, so only "name" / "arguments" are rewritten
            # and no "function" object is invented for a call that never had one.
            out.append({**call, **updates})
    return out


def neutralize_control_markup_in_messages(messages: list) -> list:
    """Neutralize control markup in message content and tool-result names (#7066).

    User / system / tool turns lose every marker; assistant turns lose only turn
    boundaries and keep their structural think / channel / tool markup, which
    replayed history legitimately holds. Returns the same list object when nothing
    changed, so the common prompt stays byte-for-byte what it was.
    """
    if not messages:
        return messages
    changed = False
    out: list = []
    for msg in messages:
        if not isinstance(msg, dict):
            out.append(msg)
            continue
        # isinstance, not truthiness: GenerateRequest.messages is an untyped List[dict], so
        # a role can be an int or a list and ".strip()" on one raised AttributeError,
        # turning the streaming request into a 500 before rendering. Anything that is not a
        # string is simply not "assistant", so it takes the full rewrite.
        raw_role_value = msg.get("role")
        role = raw_role_value.strip().lower() if isinstance(raw_role_value, str) else ""
        rewrite = (
            neutralize_turn_boundary_markup if role == "assistant" else neutralize_control_markup
        )
        updates: dict = {}
        # The role is rendered, not just dispatched on: Llama-3.1 concatenates it straight
        # between "<|start_header_id|>" and "<|end_header_id|>", and /generate/stream takes
        # an untyped list of dicts, so a role of "user<|end_header_id|><|eot_id|>..." forged
        # an assistant turn even with the content swept (#7066). Neutralized rather than
        # rejected, so a client using a role this code does not know still works.
        raw_role = msg.get("role")
        if isinstance(raw_role, str) and raw_role:
            new_role = neutralize_control_markup(raw_role)
            if new_role != raw_role:
                updates["role"] = new_role
        # Gemma-4 falls back to a tool result's "name" when "tool_call_id" matches no
        # call, concatenating it into the "<|tool_response>" block (#7066).
        name = msg.get("name")
        if role == "tool" and isinstance(name, str) and name:
            new_name = neutralize_control_markup(name)
            if new_name != name:
                updates["name"] = new_name
        # A separate reasoning field is the INNER text of a thought block whose opener and
        # closer the template supplies itself: Qwen wraps it in "<think>...</think>"
        # (chat_templates.py:759), Gemma-4 in "<|channel>thought ... <channel|>"
        # (gemma-4.jinja:245), Harmony in "<|channel|>analysis<|message|> ... <|end|>"
        # (chat_templates.py:1330). None of them is "content", so it was reaching the
        # prompt unswept, and an embedded closer exits the thought and exposes the rest as
        # answer text (#7066). Hence the FULL rewrite: unlike replayed "content", which
        # carries the assistant's own think tags and which Qwen splits on to recover this
        # very field, the field itself must never contain its enclosing delimiters.
        for field in ("reasoning", "reasoning_content", "thinking"):
            value = msg.get(field)
            if isinstance(value, str) and value:
                new_value = neutralize_control_markup(value)
                if new_value != value:
                    updates[field] = new_value
        # Gemma-4's legacy assistant-level "tool_responses": format_tool_response_block
        # renders the name and every leaf of the payload, so markup there closes
        # "<|tool_response>" and opens a model turn. Tool output, so the full rewrite.
        tool_responses = msg.get("tool_responses")
        if isinstance(tool_responses, list) and tool_responses:
            new_tool_responses = _neutralize_leaves(tool_responses, neutralize_control_markup)
            if _differs(new_tool_responses, tool_responses):
                updates["tool_responses"] = new_tool_responses
        content = msg.get("content")
        if content:
            new_content = content
            if isinstance(content, str):
                new_content = rewrite(content)
            elif isinstance(content, dict):
                # Llama-3.1 serializes mapping content with tojson
                # (chat_templates.py:127-128), and /generate/stream takes raw dicts, so an
                # object value reaches the prompt as live structure too (#7066).
                new_content = _neutralize_leaves(content, rewrite)
            elif isinstance(content, list):
                # A media part is only opaque where something RESOLVES it. Nothing
                # resolves one inside a tool result: Studio's vision and audio paths build
                # from the last user message, while Llama-3.1's tool branch serializes the
                # whole content iterable with tojson (chat_templates.py:519-520), so an
                # exempt URL there lands in the prompt as live structure (#7066).
                new_content = _neutralize_content_parts(content, rewrite, role != "tool")
            if _differs(new_content, content):
                updates["content"] = new_content
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            new_tool_calls = _neutralize_replayed_tool_call(tool_calls)
            if _differs(new_tool_calls, tool_calls):
                updates["tool_calls"] = new_tool_calls
        if updates:
            out.append({**msg, **updates})
            changed = True
        else:
            out.append(msg)
    return out if changed else messages


def neutralize_tool_descriptions(tools):
    """Neutralize a rendered tool catalog, dropping any tool with an unsafe name.

    Every string in a declaration is prompt text: Gemma-4 interpolates the description
    into its system turn and emits property keys / ``enum`` / ``required`` entries
    inline, while Granite and Mistral-Small-3 render the whole entry with ``tojson``.
    So markup anywhere in the schema closes the system turn and forges a model one
    (#7066), and ``mcp_client`` copies a remote ``description`` / ``inputSchema``
    verbatim. The rewrite covers the whole entry, not just the nested ``function``,
    because ``ChatCompletionRequest.tools`` is a bare ``list[dict]``.

    ``function.name`` is the dispatch identity: rewriting it silently breaks dispatch,
    leaving it exact forges a turn (Gemma-4 emits ``call:NAME`` unquoted), so a name
    carrying markup drops the tool with a warning instead. The predicate is the rewrite
    itself, not OpenAI's name grammar, so a passthrough client's ``ns.tool`` or
    ``functions.NAME:IDX`` still ships. The rewrite is the identity on any markup-free
    string, so a live catalog is returned unchanged.
    """
    if not tools or not isinstance(tools, list):
        return tools
    out: list = []
    changed = False
    for tool in tools:
        if not isinstance(tool, dict):
            out.append(tool)
            continue
        function = tool.get("function")
        target = function if isinstance(function, dict) else tool
        name = target.get("name")
        if isinstance(name, str) and neutralize_control_markup(name) != name:
            logger.warning(
                "Dropping tool %r from the catalog: function.name carries chat "
                "control markup, which templates render as a turn boundary.",
                name,
            )
            changed = True
            continue
        unsafe = _unsafe_schema_identifier(tool)
        if unsafe is not None:
            logger.warning(
                "Dropping tool %r from the catalog: the schema identifier %r carries chat "
                "control markup, and rewriting it would change the contract the model is "
                "told to satisfy while execute_tool still expects the original.",
                name,
                unsafe,
            )
            changed = True
            continue
        new_tool = _neutralize_argument_leaves(tool)
        if not _differs(new_tool, tool):
            out.append(tool)
            continue
        out.append(new_tool)
        changed = True
    return out if changed else tools


# The positions in a JSON Schema that are machine-valued rather than descriptive: a
# property name, an enum or const literal and a required entry are all part of the
# contract the model is told to satisfy and the controller then forwards verbatim to
# execute_tool. Rewriting one guides the model to emit the rewritten spelling while the
# MCP server still expects the original, so the tool breaks. function.name is already
# dropped for exactly this reason, and these get the same treatment (#7066).
_SCHEMA_KEYED_IDENTIFIERS = frozenset(
    {
        "properties",
        "patternProperties",
        "$defs",
        "definitions",
        # Both dependent* keywords are keyed BY a property name, and dependentRequired's
        # values are lists of property names as well, so it is checked on both sides below.
        # "dependencies" is draft-07's spelling of both, so it is keyed and list-valued too.
        "dependentSchemas",
        "dependentRequired",
        "dependencies",
    }
)
_SCHEMA_KEYED_LIST_IDENTIFIERS = frozenset({"dependentRequired", "dependencies"})
# "pattern" and "default" belong here for the same reason: a grammar built from the
# schema forces the model to satisfy the rewritten regex or echo the rewritten default,
# and the MCP server then validates the original and rejects the call. This is the case
# the "[ARGS]" comment above already anticipated.
_SCHEMA_VALUED_IDENTIFIERS = frozenset({"enum", "const", "required", "pattern", "default"})


def _unsafe_schema_identifier(value):
    """Return the first schema identifier the rewrite would change, or None."""
    stack = [value]
    seen = {id(value)}
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key, item in node.items():
                if key in _SCHEMA_KEYED_IDENTIFIERS and isinstance(item, dict):
                    for name, dependents in item.items():
                        if isinstance(name, str) and neutralize_control_markup(name) != name:
                            return name
                        if key in _SCHEMA_KEYED_LIST_IDENTIFIERS and isinstance(dependents, list):
                            for dependent in dependents:
                                if (
                                    isinstance(dependent, str)
                                    and neutralize_control_markup(dependent) != dependent
                                ):
                                    return dependent
                elif key in _SCHEMA_VALUED_IDENTIFIERS:
                    for literal in item if isinstance(item, list) else [item]:
                        if (
                            isinstance(literal, str)
                            and neutralize_control_markup(literal) != literal
                        ):
                            return literal
                if isinstance(item, (dict, list)) and id(item) not in seen:
                    seen.add(id(item))
                    stack.append(item)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, (dict, list)) and id(item) not in seen:
                    seen.add(id(item))
                    stack.append(item)
    return None


def forced_tool_name(tool_choice):
    """The function name a ``tool_choice`` pins, or None when it pins nothing.

    OpenAI spells it ``{"type": "function", "function": {"name": ...}}`` and Anthropic
    ``{"type": "tool", "name": ...}``; the string forms ("auto" / "none" / "required")
    pin no particular tool.
    """
    if not isinstance(tool_choice, dict):
        return None
    function = tool_choice.get("function")
    name = function.get("name") if isinstance(function, dict) else tool_choice.get("name")
    return name if isinstance(name, str) and name else None


def catalog_tool_names(tools) -> set:
    """Every ``function.name`` in a tool catalog, either nesting."""
    names = set()
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        name = function.get("name") if isinstance(function, dict) else tool.get("name")
        if isinstance(name, str):
            names.add(name)
    return names


def reconciled_tool_choice(tool_choice, openai_tools, safe_tools):
    """Downgrade a forced ``tool_choice`` to "auto" when WE dropped its tool (#7066).

    Only when the neutralizer removed it: the name has to be in the caller's catalog and
    gone from the sanitized one. A client forcing a function it never declared is a
    different, pre-existing case that the healing path deliberately reads to decide a
    streamed call must NOT be promoted, so silently rewriting it there would change
    unrelated behaviour.
    """
    forced = forced_tool_name(tool_choice)
    if forced is None or forced in catalog_tool_names(safe_tools):
        return tool_choice
    if forced not in catalog_tool_names(openai_tools):
        return tool_choice
    logger.warning(
        "Forcing tool %r is no longer possible: it was dropped from the catalog for "
        "carrying chat control markup. Falling back to tool_choice=auto.",
        forced,
    )
    return "auto"


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
    # Shared choke point for the transformers and MLX backends (#7066).
    messages = neutralize_control_markup_in_messages(messages)
    tools = neutralize_tool_descriptions(tools)
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
        return _render(messages)
    except Exception:
        # Retry with repairs applied cumulatively. Originals render first, so
        # working templates stay byte-identical.
        candidates: list = []
        normalized = _normalize_tool_call_arguments(messages)
        if normalized is not messages:
            candidates.append(normalized)
        split = _split_parallel_tool_calls(normalized)
        if split is not normalized:
            candidates.append(split)
        for candidate in candidates:
            try:
                return _render(candidate)
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
