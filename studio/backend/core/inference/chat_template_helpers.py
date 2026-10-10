# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Dependency-light wrapper around tokenizer.apply_chat_template with a kwarg fallback for templates
that reject reasoning/tools args, plus the shared native-chat-template fallback used by the
transformers and MLX backends."""

import copy
import functools
import inspect
import json
import logging
import re
import string
import weakref
from dataclasses import dataclass
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

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

# Both header prefixes occur: the first header arrives without the prompt's "<|start|>assistant".
_ATEM_REASONING_RECIPIENT = "self"
_ATEM_REPLY_RECIPIENT = "user"
_ATEM_CLOSE = "<|eom|>"
_ATEM_END_OF_TURN = "<|eot|>"
_ATEM_BLOCK_ENDS = (_ATEM_CLOSE, _ATEM_END_OF_TURN)
_ATEM_BLOCK_END_MAX_LEN = max(len(marker) for marker in _ATEM_BLOCK_ENDS)
_ATEM_TEMPLATE_OPENER = "<|start|>assistant to=self<|message|>"
_ATEM_HEADER_PREFIXES = ("<|start|>assistant to=", "to=")
# Generous, not a name cap: keeps the holdback below at least as long as any real header.
_ATEM_RECIPIENT_MAX_LEN = 256
_ATEM_HEADER_RE = re.compile(
    r"(?:<\|start\|>assistant )?to=(?P<recipient>[^<\s]{1,%d})<\|message\|>"
    % _ATEM_RECIPIENT_MAX_LEN
)
_ATEM_PARTIAL_TAIL_RE = re.compile(
    r"[^<\s]*(?:<(?:\|(?:m(?:e(?:s(?:s(?:a(?:g(?:e(?:\|)?)?)?)?)?)?)?)?)?)?"
)
_ATEM_HEADER_MAX_LEN = len("<|start|>assistant to=") + _ATEM_RECIPIENT_MAX_LEN + len("<|message|>")
# Call syntax from the response_template grammar the checkpoint ships.
_ATEM_INVOKE_OPEN_RE = re.compile(r'<atem:invoke\b[^>]*?\bname="(?P<name>[^"]+)">')
_ATEM_INVOKE_CLOSE = "</atem:invoke>"
_ATEM_PARAMETER_RE = re.compile(
    r'<atem:parameter\b[^>]*?\bname="(?P<key>[^"]+)"[^>]*?>(?P<value>.*?)</atem:parameter>',
    re.DOTALL,
)
_ATEM_CALLS_ENVELOPE = ("<atem:function_calls>", "</atem:function_calls>")
_ATEM_TAG_START_RE = re.compile(r"</?atem:")
_ATEM_NAME_CHARS = frozenset(string.ascii_letters + string.digits + "_-:")
_ATEM_TAGS = (
    "<atem:function_calls>",
    "</atem:function_calls>",
    "<atem:invoke",
    "</atem:invoke>",
    "<atem:parameter",
    "</atem:parameter>",
)

# Control markup from user/system/tool turns must not reach the prompt raw: it can forge turns (#7066).
# The name list is closed on purpose so "<div>", "List<String>" and "[1]" stay as typed.
# \uXXXX escapes keep this file ASCII.
_CONTROL_MARKUP = re.compile(
    r"<(?="
    # Phi-4 Mini closes with "<|/tool|>" / "<|/tool_call|>".
    r"\|/?(?:(?:start|end)_(?:header_id|of_role)|tool(?:_call|_response)?"
    # Kimi K2 / Moonshot section and call pairs (tool_call_parser.py).
    r"|tool_calls?(?:_section)?_(?:begin|end)|tool_call_argument_begin"
    r"|end(?:_of_(?:turn|text))?"
    # Document boundaries: Llama BOS and the GPT-2-lineage EOS.
    r"|begin_of_text|endoftext"
    # Llama-4 headers, Phi-4 im_sep, Kimi K2 im_system / im_middle.
    r"|header_(?:start|end)|im_(?:start|end|sep|system|middle|user|assistant)"
    # DeepSeek-V4-Flash role boundaries (case-sensitive).
    r"|User|Assistant|System"
    r"|assistant|constrain|channel|message|eo[tm](?:_id)?|final"
    # TML Inkling's call envelope.
    r"|message_model|content_invoke_tool_json|end_message"
    # Command-R / Aya.
    r"|(?:START|END)_OF_TURN_TOKEN|(?:USER|SYSTEM|CHATBOT)_TOKEN"
    # Reserved media placeholders: an extra one is a hard ValueError in the processor.
    r"|image|audio|video|python_tag"
    # Qwen 2.5 Coder FIM tokens.
    r"|fim_prefix|fim_suffix|fim_middle"
    # Qwen2-VL / Qwen2.5-VL pad tokens.
    r"|vision_start|vision_end|vision_pad|image_pad|video_pad"
    r"|return|system|start|think|turn|user|call|\")\|?>"
    # The parser also accepts space and backslash-escaped spellings; name must start with a letter.
    r"|\uff5c[A-Za-z][A-Za-z\u2581_ \\]{0,39}\uff5c>"
    # Bare-tag families. "<s>" / "</s>" (Llama-2 BOS/EOS) collide with HTML; accepted on purpose.
    r"|/?(?:(?:start|end)_of_turn|tool_(?:call|response)|tools|think|eos|bos|s|sop"
    r"|start_of_image|image_soft_token|audio_soft_token"
    r"|arg_key|arg_value|function|parameter|param)>"
    r"|(?:function|parameter)=|(?:function|param(?:eter)?)\s+name=\""
    r"|(?:tool(?:_call|_response)?|channel|turn)\|>"
    r")"
    # "[ARGS]", "[CALL_ID]" and "[TOOL_CONTENT]" are absent on purpose: they cannot open a block alone.
    r"|\[(?=/?(?:INST|SYSTEM_PROMPT|AVAILABLE_TOOLS|TOOL_RESULTS|TOOL_CALLS"
    r"|PREFIX|MIDDLE|SUFFIX|gMASK)\])"
    # Anchored on the second "<" so "<SYS>>" and "cout << SYS" stay as typed.
    r"|(?<=<)<(?=/?SYS>>)"
)

# Turn-boundary subset for replayed ASSISTANT content; its own think/channel/tool markup is kept.
_TURN_BOUNDARY_MARKUP = re.compile(
    r"<(?="
    r"\|/?(?:(?:start|end)_(?:header_id|of_role)"
    r"|im_(?:start|end|sep|system|middle|user|assistant)"
    r"|User|Assistant|System"
    r"|end(?:_of_(?:turn|text))?|eo[tm](?:_id)?|header_(?:start|end)"
    r"|begin_of_text|endoftext"
    r"|(?:START|END)_OF_TURN_TOKEN|(?:USER|SYSTEM|CHATBOT)_TOKEN"
    r"|image|audio|video|python_tag"
    r"|fim_prefix|fim_suffix|fim_middle"
    r"|vision_start|vision_end|vision_pad|image_pad|video_pad"
    # Tool RESULT / CATALOG markup forges trusted context; tool CALL spellings are the assistant's own.
    r"|tool_response|tool"
    r"|assistant|return|system|start|turn|user|call)\|?>"
    r"|\uff5c(?:User|Assistant|(?:begin|end)\u2581of\u2581sentence)\uff5c>"
    r"|/?(?:(?:start|end)_of_turn|eos|bos|s|sop|tool_response|tools"
    r"|start_of_image|image_soft_token|audio_soft_token)>"
    r"|(?:turn|tool_response)\|>"
    r")"
    # "[TOOL_CALLS]" is out: the assistant emits that one.
    r"|\[(?=/?(?:INST|SYSTEM_PROMPT|AVAILABLE_TOOLS|TOOL_RESULTS"
    r"|PREFIX|MIDDLE|SUFFIX|gMASK)\])"
    r"|(?<=<)<(?=/?SYS>>)"
)


# Per codec, NOT the chat sweep: this text is spoken, so "<s>hello</s>" must stay as typed.
_MOSS_TTS_MARKUP = re.compile(
    r"<(?=/?user_inst>|\|(?:im_(?:start|end)|audio(?:_start|_end|_pad)?"
    r"|vision_pad|video_pad)\|>)"
)
_TTS_MARKUP_BY_CODEC = {
    # Only these three: "say <custom_token_999>" is ordinary text here.
    "snac": re.compile(r"<(?=custom_token_[234]>|\|eot_id\|>)"),
    "bicodec": re.compile(
        r"<(?=\|(?:task_tts|(?:start|end)_(?:content|global_token|semantic_token)"
        r"|im_end)\|>|/s>)"
    ),
    "dac": re.compile(
        r"<(?=\|(?:im_(?:start|end)|text_(?:start|end)|audio_(?:start|end)"
        r"|global_features_(?:start|end))\|>)"
    ),
    # The processor tokenizes "[speaker_id]text" flat, so a "[1]" anywhere starts a speaker turn.
    "csm": re.compile(r"\[(?=\d+\])|<(?=\|(?:AUDIO|audio_eos|begin_of_text|end_of_text)\|>)"),
    "higgs_tts2": re.compile(
        r"<(?=\|(?:begin_of_text|end_of_text|start_header_id|end_header_id"
        r"|scene_desc_start|scene_desc_end|eot_id|audio_out_bos|AUDIO_OUT"
        r"|audio_eos|reserved_special_token_6)\|>)"
    ),
    "higgs_tts3": re.compile(r"<(?=\|(?:tts|ref_audio|ref_text|text|audio)\|>)"),
    "minimax_music3": re.compile(
        r"<(?=\|(?:im_(?:start|end)|caption_(?:start|end)|lyrics_(?:start|end)"
        r"|audio_(?:start|end|cfg))\|>)"
    ),
    "moss_tts_local": _MOSS_TTS_MARKUP,
    "moss_tts_nano": _MOSS_TTS_MARKUP,
}
_TTS_MARKUP_DEFAULT = re.compile(
    "|".join(f"(?:{pattern.pattern})" for pattern in _TTS_MARKUP_BY_CODEC.values())
)


_DELIMITER_SHAPED = re.compile(r"\A(?:<[^\s<>]{1,60}>|\[[^\s\[\]]{1,40}\])\Z")
# The bracket half excludes quotes and digits so Jinja's "messages[0]" indexing stays out.
_TEMPLATE_DELIMITERS = re.compile(
    '<[A-Za-z_][A-Za-z0-9_.\\-]{0,38}\\s+[A-Za-z_][A-Za-z0-9_.\\-]{0,38}="[^"<>]{0,60}">'
    # Before the single-angle arm, which would match Llama-2's inner "<SYS>".
    "|<</?[A-Za-z_][A-Za-z0-9_.\\-]{0,38}>>"
    "|<[^\\s<>'\"]{1,60}>"
    "|\\[/?[A-Za-z_][A-Za-z0-9_.\\-]{0,38}\\]"
)
# Comments never reach the prompt; gptoss mentions "<|final|>" only in one.
_JINJA_COMMENT = re.compile(r"\{#.*?#\}", re.S)
_BLOCK_METADATA = frozenset({"[ARGS]", "[CALL_ID]", "[TOOL_CONTENT]"})
# Phi-3 builds role sentinels by concatenation ("+" or "~"), so no literal appears.
_DYNAMIC_PIPE_ROLE = re.compile(r"""['"]<\|['"]\s*[+~]|[+~]\s*['"]\|>['"]""")
_CONCATENATED_OPENER = re.compile(r"""['"]<(/?[A-Za-z_][A-Za-z0-9_.\-]{0,30})=['"]\s*[+~]""")


_ROLE_NAMES = (
    "system",
    "user",
    "assistant",
    "tool",
    "ipython",
    "function",
    "developer",
    "human",
)
# The name is filled at render time: "<function=pay>" must break on a "<function=example>" template.
_DYNAMIC_OPENER = re.compile(r"\A<(/?[A-Za-z_][A-Za-z0-9_.\-]{0,30})=[^\s<>]*>\Z")
_DYNAMIC_ATTR_OPENER = re.compile(
    r"\A<([A-Za-z_][A-Za-z0-9_.\-]{0,38})\s+([A-Za-z_][A-Za-z0-9_.\-]{0,38})=\"[^\"<>]{0,60}\">\Z"
)


# tool_call_parser accepts the space and backslash-escaped DeepSeek spellings too.
_FULLWIDTH_MARKER = re.compile("\\A<\uff5c([A-Za-z][A-Za-z\u2581_ \\\\]{0,39})\uff5c>\\Z")
_ALIAS_SEPARATORS = "(?:\u2581|\\\\?_| )"


@functools.lru_cache(maxsize = 1)
def _deepseek_opener_pattern():
    """The tool-call-parser's own DeepSeek opener alternation, or None if unavailable. Single source
    of truth: tool_call_parser keeps the five spellings llama.cpp accepts, and a profile that
    breaks only the one spelling a vocabulary happens to hold leaves the other four live (#7066)."""
    try:
        from core.inference.tool_call_parser import (
            _DEEPSEEK_OPEN_RE_SRC,
            TOOL_XML_SIGNALS,
        )
    except Exception:  # pragma: no cover - parser unavailable
        return None
    signals = [
        re.escape(signal)
        for signal in TOOL_XML_SIGNALS
        if isinstance(signal, str) and signal.startswith("<\uff5c") and signal.endswith("\uff5c>")
    ]
    return "|".join([_DEEPSEEK_OPEN_RE_SRC, *signals]) if signals else _DEEPSEEK_OPEN_RE_SRC


def _marker_pattern_source(marker: str) -> str:
    """The regex for one harvested marker: exact, unless its name is dynamic."""
    fullwidth = _FULLWIDTH_MARKER.match(marker)
    if fullwidth:
        # Reuse the parser's own alternation so the two cannot drift.
        deepseek = _deepseek_opener_pattern()
        if deepseek is not None and re.fullmatch(deepseek, marker):
            return deepseek
        name = fullwidth.group(1)
        parts = re.split("[\u2581_ ]", name)
        if len(parts) > 1:
            return "<\uff5c" + _ALIAS_SEPARATORS.join(re.escape(p) for p in parts) + "\uff5c>"
    dynamic = _DYNAMIC_OPENER.match(marker)
    if dynamic:
        return "<" + re.escape(dynamic.group(1)) + "=[^\\s<>]*>"
    attr = _DYNAMIC_ATTR_OPENER.match(marker)
    if attr:
        return "<" + re.escape(attr.group(1)) + "\\s+" + re.escape(attr.group(2)) + '="[^"<>]*">'
    return re.escape(marker)


def _template_strings(chat_template) -> list:
    """Every template body a tokenizer exposes, whatever shape it uses. A tokenizer may carry one
    string, a dict of named templates, or a list of ``{"name", "template"}`` entries (Hermes-3
    ships the list form). Profiling only the string case would silently drop every literal a
    named template emits."""
    out: list = []
    if isinstance(chat_template, str):
        out.append(chat_template)
    elif isinstance(chat_template, dict):
        for value in chat_template.values():
            out.extend(_template_strings(value))
    elif isinstance(chat_template, (list, tuple)):
        for entry in chat_template:
            if isinstance(entry, dict):
                out.extend(_template_strings(entry.get("template")))
            else:
                out.extend(_template_strings(entry))
    return out


def delimiter_shaped_tokens(tokens) -> list:
    """The delimiter-shaped entries of a vocabulary, for a caller that cannot keep it all."""
    return [t for t in tokens or () if isinstance(t, str) and _DELIMITER_SHAPED.match(t)]


class ModelMarkup:
    """The markers one model actually treats as structure, and the patterns for them. Built from the
    model's own chat template and token list rather than from the curated patterns below, because
    a vocabulary is authoritative where a hand-written list cannot be: it covers a sentinel this
    module never enumerated, and it leaves alone one that belongs to some other family. A Llama-3
    checkpoint has no "</think>" in either place, so a user pasting a script that contains one
    keeps their text byte-for-byte (#7066)."""

    __slots__ = (
        "control",
        "boundary",
        "markers",
        "rewrite_control",
        "rewrite_boundary",
        "selected_with_tools",
    )

    def __init__(
        self,
        markers: set,
        selected_with_tools: bool = False,
    ):
        self.markers = markers
        self.selected_with_tools = selected_with_tools
        self.control = _alternation(markers)
        # A marker this module does not recognise is treated as a boundary.
        boundary = {
            marker
            for marker in markers
            if _TURN_BOUNDARY_MARKUP.search(marker) or not _CONTROL_MARKUP.search(marker)
        }
        self.boundary = _alternation(boundary)
        # Bound once per profile so a cache keyed on the callable can hit.
        self.rewrite_control = functools.partial(neutralize_control_markup, markup = self)
        self.rewrite_boundary = functools.partial(neutralize_turn_boundary_markup, markup = self)


def _alternation(markers: set):
    """A pattern matching any of *markers*, longest first so no prefix shadows a longer one."""
    if not markers:
        return None
    return re.compile(
        "|".join(_marker_pattern_source(m) for m in sorted(markers, key = len, reverse = True))
    )


# Llama-3.1 emits "{{ bos_token }}", so the literal never appears in the template text.
_SPECIAL_TOKEN_VARIABLES = (
    "bos_token",
    "eos_token",
    "pad_token",
    "unk_token",
    "sep_token",
    "cls_token",
    "mask_token",
)


def model_markup(
    chat_template,
    tokens = None,
    tools = None,
    prefer_tool_use: bool = True,
    specials = None,
) -> Optional[ModelMarkup]:
    """Profile one model's structural markers, or None when nothing is known about it. None means
    "sweep everything the curated patterns know", which is the safe direction for a model whose
    template and vocabulary could not be read."""
    markers: set = set()
    for token in tokens or ():
        if not isinstance(token, str) or not token or token[0] not in "<[":
            continue
        if not _DELIMITER_SHAPED.match(token):
            continue
        if token in _BLOCK_METADATA:
            continue
        # Intersect with the curated pattern: Gemma reserves "<table>" and harvesting it mangled HTML.
        if _CONTROL_MARKUP.search(token):
            markers.add(token)
    known = {token for token in tokens or () if isinstance(token, str)}
    # Only the template this request renders with, or a no-tools turn rewrites "<tools>".
    bodies = _selected_template_strings_from_value(
        chat_template, tools, prefer_tool_use = prefer_tool_use
    )
    for body in bodies or _template_strings(chat_template):
        # Blanked rather than removed, so every offset the index check relies on survives.
        body = _JINJA_COMMENT.sub(lambda m: " " * len(m.group(0)), body)
        expressions = _jinja_expression_spans(body)
        for match in _TEMPLATE_DELIMITERS.finditer(body):
            marker = match.group(0)
            if marker.startswith("[") and _within(expressions, match.start()):
                continue
            if marker in _BLOCK_METADATA:
                continue
            # Qwen's "<function-name>" placeholders are prose, not structure.
            if marker in known or _CONTROL_MARKUP.search(marker):
                markers.add(marker)
        for name, value in (specials or {}).items():
            if not isinstance(value, str) or not value:
                continue
            if not any(name in code for code in _jinja_code(body)):
                continue
            # Shape, not the curated pattern: this is for unknown-family boundaries.
            if value in known or _DELIMITER_SHAPED.match(value):
                markers.add(value)
        if _DYNAMIC_PIPE_ROLE.search(body):
            markers.update(f"<|{role}|>" for role in _ROLE_NAMES)
        # The example spelling lets the dynamic rule match any render-time name.
        for built in _CONCATENATED_OPENER.findall(body):
            markers.add(f"<{built}=example>")
    return ModelMarkup(markers, bool(tools)) if markers else None


def _spaced_out(pattern, text: str) -> str:
    """Insert one space after every marker opener *pattern* found."""
    if not text or ("<" not in text and "[" not in text):
        return text
    return pattern.sub(r"\g<0> ", text)


def _spaced_out_markers(pattern, text: str) -> str:
    """Insert one space after the opener of every whole marker *pattern* matches."""
    if not text or ("<" not in text and "[" not in text):
        return text
    return pattern.sub(lambda m: m.group(0)[0] + " " + m.group(0)[1:], text)


def neutralize_control_markup(text: str, markup: "ModelMarkup" = None) -> str:
    """Break control markup in free text by spacing out the opener (#7066). "</think>" -> "<
    /think>", "[/INST]" -> "[ /INST]": readable, but no longer a delimiter to the template, the
    think extractor or the stop-sequence matcher. A plain space, because every tokenizer
    vocabulary has one; U+2060 can fall back to byte junk. With a *markup* profile only that
    model's own markers are broken, so text naming some other family's sentinel is left exactly
    as the caller wrote it."""
    if markup is not None:
        return _spaced_out_markers(markup.control, text) if markup.control else text
    return _spaced_out(_CONTROL_MARKUP, text)


def neutralize_turn_boundary_markup(text: str, markup: "ModelMarkup" = None) -> str:
    """Break only the turn-boundary sentinels, for replayed assistant text (#7066)."""
    if markup is not None:
        return _spaced_out_markers(markup.boundary, text) if markup.boundary else text
    return _spaced_out(_TURN_BOUNDARY_MARKUP, text)


def neutralize_tts_prompt_text(text: str, audio_type = None) -> str:
    """Break the active codec's own delimiters in a TTS prompt (#7066). Scoped to *audio_type*: this
    text is spoken, so anything that is not structure in THIS codec's prompt has to survive
    byte-exact."""
    return _spaced_out(_TTS_MARKUP_BY_CODEC.get(audio_type, _TTS_MARKUP_DEFAULT), text)


def build_dac_tts_prompt(text: str) -> str:
    return f"<|im_start|>\n<|text_start|>{text}<|text_end|>\n<|audio_start|>\n"


def _neutralize_leaves(
    value,
    rewrite,
    warn_on_key_collision: bool = False,
):
    """Apply *rewrite* to every string leaf, keys included, of a nested structure.

    Iterative, not recursive: the client picks the depth, and a schema ``json.loads`` accepts must
    not exhaust the interpreter stack and turn the request into a 500. Containers are rebuilt in
    reverse breadth-first order, so a child is finished before its parent and a repeated or
    self-referencing node is visited once.

    Rewriting keys is not injective ("a<think>" and "a< think>" both land on "a< think>"), so a
    colliding dict keeps only the last value; *warn_on_key_collision* logs it. The merge stands
    because the alternative, keeping one key raw so both survive, would put the markup back in the
    prompt.
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


# Media payloads stay opaque, gated on the part's own type: "data" is a normal key elsewhere.
_TOOL_RESULT_ROLES = frozenset({"tool", "ipython"})
# Gemma-4 maps "assistant" onto "model", so both name the replayed turn.
_ASSISTANT_ROLES = frozenset({"assistant", "model"})
_SCHEMA_ROLES = frozenset({"system", "user", "assistant", "tool", "ipython", "developer", "model"})
_MEDIA_PART_TYPES = frozenset(
    {"image", "image_url", "input_image", "input_audio", "audio", "audio_url", "video", "video_url"}
)
_OPAQUE_PART_KEYS = frozenset(
    {
        "image_url",
        "audio_url",
        "video_url",
        "input_audio",
        "image",
        "audio",
        "video",
        "url",
        "data",
        "b64_json",
    }
)


def _redistribute_swept(
    texts: list,
    rewrite,
    contiguous = None,
):
    """Sweep the joined *texts* and hand each carrier back its own share, or None. Every carrier
    keeps its own text in its own position: nothing is moved past a neighbour, so a caption still
    sits on the side of the item it describes. *contiguous*[i] says whether carrier i+1 directly
    follows carrier i in the parts list; where it does not -- an image or a JSON part sits
    between them -- the opener is NOT migrated, because moving a character across a media item
    reorders the text around it and a renderer that keeps the media would bind the caption to the
    wrong side (#7066)."""
    swept = rewrite("".join(texts))
    pieces: list = []
    inserted: list = []
    position = 0
    for text in texts:
        chars: list = []
        flags: list = []
        consumed = 0
        while consumed < len(text) and position < len(swept):
            same = swept[position] == text[consumed]
            chars.append(swept[position])
            flags.append(not same)
            if same:
                consumed += 1
            position += 1
        pieces.append(chars)
        inserted.append(flags)
    if position < len(swept):
        pieces[-1].extend(swept[position:])
        inserted[-1].extend([True] * (len(swept) - position))
    # A break at a carrier's start is trimmed by some renderers, so the opener walks into the next carrier.
    for index in range(len(pieces) - 1):
        if contiguous is not None and not contiguous[index]:
            continue
        while pieces[index] and inserted[index + 1] and inserted[index + 1][0]:
            pieces[index + 1].insert(0, pieces[index].pop())
            inserted[index + 1].insert(0, inserted[index].pop())
    out = ["".join(chars) for chars in pieces]
    trimmed = "".join(piece.strip() for piece in out)
    return out if rewrite(trimmed) == trimmed else None


def _neutralize_content_parts(
    content: list,
    rewrite,
    media_opaque: bool = True,
):
    """Neutralize an OpenAI-style content parts list (#7066).

    Two gaps a per-part rewrite misses. A mapping part without a string "text" was passed through
    whole, yet /generate/stream accepts one and Llama-3.1 serializes the entire iterable with
    tojson. And a marker split across two adjacent text parts survived both sweeps, because Gemma-4
    concatenates them with no separator and reassembles the opener. Whitespace between them is no
    fix, since the sibling paths trim each part, so a run that only becomes a marker once joined is
    swept as one string and collapses into one part. A clean run keeps its parts.
    """
    parts: list = []
    for part in content:
        if isinstance(part, str):
            parts.append(rewrite(part))
        elif isinstance(part, dict):
            # isinstance first: "type" can be unhashable in untyped request dicts.
            part_type = part.get("type")
            if isinstance(part_type, str) and part_type in _MEDIA_PART_TYPES and media_opaque:
                opaque = {k: v for k, v in part.items() if k in _OPAQUE_PART_KEYS}
                swept = _neutralize_leaves(
                    {k: v for k, v in part.items() if k not in _OPAQUE_PART_KEYS}, rewrite
                )
                parts.append({**swept, **opaque} if opaque else swept)
            else:
                # Every field: tojson templates serialize the part whole.
                parts.append(_neutralize_leaves(part, rewrite))
        else:
            parts.append(part)

    def _text_of(part):
        if isinstance(part, str):
            return part
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            return part["text"]
        return None

    # No part reliably separates text: gemma-4.jinja silently drops unknown media types.
    texts = [_text_of(part) for part in parts]

    def _joinable_runs():
        run = [index for index, text in enumerate(texts) if text is not None]
        if len(run) > 1:
            yield run

    merged: dict = {}
    for carriers in list(_joinable_runs()):
        raw = "".join(texts[index] for index in carriers)
        trimmed = "".join(texts[index].strip() for index in carriers)
        if rewrite(raw) == raw and rewrite(trimmed) == trimmed:
            continue
        # Keep each carrier's own text: Llama-3.1 serializes the list in order.
        run = [texts[index] for index in carriers]
        redistributed = _redistribute_swept(
            run,
            rewrite,
            [carriers[i + 1] == carriers[i] + 1 for i in range(len(carriers) - 1)],
        )
        if redistributed is None:
            redistributed = _redistribute_swept(run, rewrite)
        if redistributed is None:
            # Last resort; unreachable today for every known marker.
            redistributed = [rewrite(trimmed)] + [""] * (len(carriers) - 1)
        for index, text in zip(carriers, redistributed):
            part = parts[index]
            merged[index] = text if isinstance(part, str) else {**part, "text": text}
    if not merged:
        return parts
    return [merged.get(index, part) for index, part in enumerate(parts)]


def _differs(new, old) -> bool:
    """True when the rewrite changed *old* into *new*. The client picks the nesting depth and ``==``
    recurses in C, so an overflowing comparison must not 500 the request. It counts as changed,
    keeping the neutralized copy: the safe direction (#7066)."""
    try:
        return new != old
    except RecursionError:
        return True


def _neutralize_argument_leaves(value, markup = None):
    """Break control markup in every string leaf (keys included) of *value*."""
    rewrite = neutralize_control_markup if markup is None else markup.rewrite_control
    return _neutralize_leaves(value, rewrite, warn_on_key_collision = True)


def _neutralized_arguments(arguments, markup = None):
    """parse with ``json.loads`` before rewriting so decoded escapes cannot forge a turn; keep clean input byte-identical."""
    if isinstance(arguments, str):
        decoded = safe = _UNPARSED
        try:
            decoded = json.loads(arguments)
            safe = _neutralize_argument_leaves(decoded, markup)
        # RecursionError: json.loads and the walk blow the stack near 1000 levels.
        except (ValueError, TypeError, RecursionError):
            decoded = safe = _UNPARSED
        if decoded is not _UNPARSED:
            # _differs, not "!=": deep comparison recurses in C and can blow the stack.
            if _differs(safe, decoded):
                # ensure_ascii keeps a decoded lone surrogate escaped, else the request is unencodable.
                return json.dumps(safe, ensure_ascii = True)
            # Duplicate keys: json.loads keeps the last value while the template renders the raw string.
            rewrite = neutralize_control_markup if markup is None else markup.rewrite_control
            if rewrite(arguments) != arguments:
                return json.dumps(safe, ensure_ascii = True)
            return None
    new_arguments = _neutralize_argument_leaves(arguments, markup)
    return new_arguments if _differs(new_arguments, arguments) else None


def _replayed_ids(msg: dict):
    """Every tool-call id a message carries, on the call side and the result side."""
    result_id = msg.get("tool_call_id")
    if isinstance(result_id, str) and result_id:
        yield result_id
    tool_calls = msg.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if isinstance(call, dict):
                call_id = call.get("id")
                if isinstance(call_id, str) and call_id:
                    yield call_id


def _injective_id_map(messages: list, markup = None) -> dict:
    """Map each replayed tool-call id to a swept id that is still unique. The sweep is not
    injective: "call<|end|>" and "call< |end|>" both break to "call< |end|>". Gemma resolves a
    result by comparing ids and lets the last match win, so two calls sharing one id would
    attribute both observations to the same call. A collision is therefore given a numeric
    suffix, which is checked against the ids that stay as they are so it cannot land on one of
    those either."""
    originals: list = []
    seen: set = set()
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        for value in _replayed_ids(msg):
            if value not in seen:
                seen.add(value)
                originals.append(value)
    swept = {original: neutralize_control_markup(original, markup) for original in originals}
    # Reserved first: a rewritten id must never be handed an untouched id's value.
    taken = {original for original in originals if swept[original] == original}
    mapping: dict = {}
    for original in originals:
        candidate = swept[original]
        if candidate == original:
            continue
        if candidate in taken:
            base, suffix = candidate, 2
            while candidate in taken:
                candidate = f"{base}-{suffix}"
                suffix += 1
            logger.warning(
                "Two replayed tool-call ids break to the same value; disambiguating one "
                "as %r so each call keeps its own result.",
                candidate,
            )
        taken.add(candidate)
        mapping[original] = candidate
    return mapping


def _neutralize_replayed_tool_call(
    tool_calls: list,
    id_map: dict = None,
    markup = None,
) -> list:
    """Neutralize a replayed tool call's name, arguments and id, in every shape it carries.

    Gemma-4 renders "<|tool_call>call:NAME{key:<|\"|>value<|\"|>}<tool_call|>", so a name or
    argument echoing pasted text can close the call block and open a "<|tool_response>" or
    "<|turn>model" of its own (#7066). The rewrite is the identity on every dispatchable name
    (Unsloth composes ^[a-zA-Z0-9_-]{1,64}$), and a tool result's "name" takes the same rewrite, so
    the two still agree when Gemma-4 pairs them by name.

    Both replay shapes are swept, the OpenAI nested one and the flat {"id", "name", "arguments"}
    one, rather than only whichever a particular guard would pick. Harmony / gpt-oss, Qwen 2.5 / 3,
    Granite-4 and Llama-4 select with "{%- if tool_call.function %}", a truthiness test, so an empty
    nested object sends them to the flat fields; and a flat-shaped template reads "name" off the
    call whatever the nested object holds.
    """

    def _field_updates(source: dict) -> dict:
        updates: dict = {}
        name = source.get("name")
        if isinstance(name, str) and name:
            new_name = neutralize_control_markup(name, markup)
            if new_name != name:
                updates["name"] = new_name
        # Harmony concatenates "content_type" straight before "<|message|>".
        content_type = source.get("content_type")
        if isinstance(content_type, str) and content_type:
            new_content_type = neutralize_control_markup(content_type, markup)
            if new_content_type != content_type:
                updates["content_type"] = new_content_type
        arguments = source.get("arguments")
        if arguments is not None:
            new_arguments = _neutralized_arguments(arguments, markup)
            if new_arguments is not None:
                updates["arguments"] = new_arguments
        return updates

    out: list = []
    for call in tool_calls:
        if not isinstance(call, dict):
            out.append(call)
            continue
        function = call.get("function")
        flat_updates = _field_updates(call)
        nested_updates = _field_updates(function) if isinstance(function, dict) else {}
        # Kimi interpolates the id inside the call envelope; same rewrite as tool_call_id so pairs match.
        id_updates: dict = {}
        call_id = call.get("id")
        if isinstance(call_id, str) and call_id:
            new_call_id = (id_map or {}).get(call_id, call_id)
            if new_call_id != call_id:
                id_updates["id"] = new_call_id
        if not flat_updates and not nested_updates and not id_updates:
            out.append(call)
            continue
        merged = {**call, **flat_updates, **id_updates}
        if nested_updates:
            merged["function"] = {**function, **nested_updates}
        out.append(merged)
    return out


def sweep_cache() -> dict:
    """A cache for a caller that sweeps the same history more than once.

    The agentic tool loop re-sweeps the whole conversation on every iteration, because a tool result
    lands in it as the loop goes and a forged turn in one would render for real. Every earlier turn
    is then swept again with identical text, which is pure repeated work: the rewrite is a function
    of the string alone.

    The cache is handed in by the caller rather than kept here on purpose. It lives as long as the
    request that owns it and is dropped with it, so message text is never retained past the call in
    module state, and it needs no size bound because it can only ever hold text the caller is
    already holding.
    """
    return {}


def _memoized(rewrite, cache: dict):
    """Wrap *rewrite* so repeated text is rewritten once per cache."""
    store = cache.get(rewrite)
    if store is None:
        store = cache[rewrite] = {}

    def cached(text: str):
        # Membership, not "or": a rewrite legitimately returns "" for "".
        if text in store:
            return store[text]
        result = store[text] = rewrite(text)
        return result

    return cached


def neutralize_control_markup_in_messages(
    messages: list,
    cache: dict = None,
    markup = None,
) -> list:
    """Neutralize control markup in message content and names (#7066). User / system /
    tool turns lose every marker; assistant turns lose only turn boundaries and keep the think /
    channel / tool markup replayed history legitimately holds. Returns the same list object when
    nothing changed, so the prompt stays byte-for-byte what it was. Pass a ``sweep_cache()`` when
    sweeping the same growing history repeatedly; results are identical either way, since it only
    memoizes a pure rewrite."""
    if not messages:
        return messages
    changed = False
    out: list = []
    id_map = _injective_id_map(messages, markup)
    for msg in messages:
        if not isinstance(msg, dict):
            out.append(msg)
            continue
        # isinstance, not truthiness: a role can be an int in untyped request dicts.
        raw_role_value = msg.get("role")
        role = raw_role_value.strip().lower() if isinstance(raw_role_value, str) else ""
        assistant = role in _ASSISTANT_ROLES
        if markup is None:
            rewrite = neutralize_turn_boundary_markup if assistant else neutralize_control_markup
        else:
            rewrite = markup.rewrite_boundary if assistant else markup.rewrite_control
        if cache is not None:
            # Keyed by the bound rewrite, so two models cannot share an entry.
            rewrite = _memoized(rewrite, cache)
        updates: dict = {}
        dropped_keys: set = set()
        # The role is rendered too (Llama-3.1 header), so it is neutralized as well.
        raw_role = msg.get("role")
        if isinstance(raw_role, str) and raw_role:
            new_role = neutralize_control_markup(raw_role, markup)
            # Templates compare roles case-sensitively, so a known role is canonicalized.
            if role in _SCHEMA_ROLES and new_role != role:
                new_role = role
            # Phi-3 renders an unknown role as "<|" + role + "|>", so "end" spells its terminator. Falls back
            # to "user" rather than padding, since a template that trims the role would undo a space.
            elif role not in _SCHEMA_ROLES:
                wrapped = f"<|{new_role}|>"
                if neutralize_control_markup(wrapped, markup) != wrapped:
                    logger.warning(
                        "Rewriting role %r to 'user': a template that wraps a role in its "
                        "own delimiters would render it as a turn boundary.",
                        new_role,
                    )
                    new_role = "user"
            if new_role != raw_role:
                updates["role"] = new_role
        # Gemma-4 renders a tool result's "name" when tool_call_id matches no call.
        result_id = msg.get("tool_call_id")
        if isinstance(result_id, str) and result_id:
            new_result_id = id_map.get(result_id, result_id)
            if new_result_id != result_id:
                updates["tool_call_id"] = new_result_id
        name = msg.get("name")
        if isinstance(name, str) and name:
            new_name = neutralize_control_markup(name, markup)
            if new_name != name:
                updates["name"] = new_name
        recipient = msg.get("recipient")
        if isinstance(recipient, str) and recipient:
            new_recipient = neutralize_control_markup(recipient, markup)
            if new_recipient != recipient:
                updates["recipient"] = new_recipient
        # The template wraps reasoning itself, so it must never carry its own delimiters: full rewrite.
        for field in ("reasoning", "reasoning_content", "thinking"):
            value = msg.get(field)
            if isinstance(value, str) and value:
                new_value = neutralize_control_markup(value, markup)
                if new_value != value:
                    updates[field] = new_value
        tool_responses = msg.get("tool_responses")
        # Gemma-4 renders tool_responses regardless of role: assistant-only like tool_calls.
        if tool_responses is not None and role not in _ASSISTANT_ROLES:
            logger.warning(
                "Dropping tool_responses from a %r message: templates wrap it as a tool "
                "observation regardless of the role.",
                role or "<missing>",
            )
            dropped_keys.add("tool_responses")
        elif isinstance(tool_responses, list) and tool_responses:
            new_tool_responses = _neutralize_leaves(
                tool_responses,
                neutralize_control_markup if markup is None else markup.rewrite_control,
            )
            if _differs(new_tool_responses, tool_responses):
                updates["tool_responses"] = new_tool_responses
        content = msg.get("content")
        if content:
            new_content = content
            if isinstance(content, str):
                new_content = rewrite(content)
            elif isinstance(content, dict):
                new_content = _neutralize_leaves(content, rewrite)
            elif isinstance(content, list):
                # Nothing resolves media inside a tool result; Llama-3.1 serializes it with tojson.
                is_tool_result = role in _TOOL_RESULT_ROLES
                new_content = _neutralize_content_parts(content, rewrite, not is_tool_result)
            if _differs(new_content, content):
                updates["content"] = new_content
        tool_calls = msg.get("tool_calls")
        # Llama-3.1 branches on tool_calls before the role, so other roles drop the field.
        if tool_calls is not None and role not in _ASSISTANT_ROLES:
            logger.warning(
                "Dropping tool_calls from a %r message: templates render it as an "
                "assistant tool-call turn regardless of the role.",
                role or "<missing>",
            )
            dropped_keys.add("tool_calls")
        elif isinstance(tool_calls, list) and tool_calls:
            new_tool_calls = _neutralize_replayed_tool_call(tool_calls, id_map, markup)
            if _differs(new_tool_calls, tool_calls):
                updates["tool_calls"] = new_tool_calls
        if updates or dropped_keys:
            merged = {**msg, **updates}
            for key in dropped_keys:
                merged.pop(key, None)
            out.append(merged)
            changed = True
        else:
            out.append(msg)
    return out if changed else messages


def neutralize_tool_descriptions(
    tools,
    cache: dict = None,
    markup = None,
):
    """Neutralize a rendered tool catalog, dropping any tool with an unsafe name.

    Every string in a declaration is prompt text: Gemma-4 interpolates the description into its
    system turn and emits property keys / ``enum`` / ``required`` entries inline, while Granite and
    Mistral-Small-3 render the whole entry with ``tojson``, and ``mcp_client`` copies a remote
    ``description`` / ``inputSchema`` verbatim. So markup anywhere in the schema closes the system
    turn and forges a model one (#7066). The rewrite covers the whole entry, not just the nested
    ``function``, because ``ChatCompletionRequest.tools`` is a bare ``list[dict]``.

    ``function.name`` is the dispatch identity: rewriting it silently breaks dispatch, leaving it
    exact forges a turn (Gemma-4 emits ``call:NAME`` unquoted), so a name carrying markup drops the
    tool with a warning instead. The predicate is the rewrite itself, not OpenAI's name grammar, so
    a passthrough client's ``ns.tool`` or ``functions.NAME:IDX`` still ships; it is the identity on
    markup-free strings, so a live catalog is returned unchanged.
    """
    # Keyed on the serialized catalog so in-place mutation cannot make the cache stale.
    key = None
    if cache is not None:
        try:
            key = ("catalog", json.dumps(tools, sort_keys = True, default = str))
        except (TypeError, ValueError):
            key = None
        if key is not None and key in cache:
            return cache[key]
    if not tools or not isinstance(tools, list):
        return tools
    out: list = []
    changed = False
    for tool in tools:
        if not isinstance(tool, dict):
            out.append(tool)
            continue
        function = tool.get("function")
        target = function if isinstance(function, dict) and function else tool
        # Both spellings: an entry may carry an empty "function" alongside a flat "name".
        name = target.get("name")
        unsafe_name = next(
            (
                candidate
                for candidate in (name, tool.get("name"))
                if isinstance(candidate, str)
                and neutralize_control_markup(candidate, markup) != candidate
            ),
            None,
        )
        if unsafe_name is not None:
            logger.warning(
                "Dropping tool %r from the catalog: function.name carries chat "
                "control markup, which templates render as a turn boundary.",
                unsafe_name,
            )
            changed = True
            continue
        unsafe = _unsafe_schema_identifier(_schema_roots(tool) + _schema_roots(target), markup)
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
        new_tool = _neutralize_argument_leaves(tool, markup)
        if not _differs(new_tool, tool):
            out.append(tool)
            continue
        out.append(new_tool)
        changed = True
    result = out if changed else tools
    if key is not None:
        cache[key] = result
    return result


# Machine-valued schema positions are forwarded verbatim to execute_tool, so never rewrite them.
_SCHEMA_KEYED_IDENTIFIERS = frozenset(
    {
        "properties",
        "patternProperties",
        "$defs",
        "definitions",
        "dependentSchemas",
        "dependentRequired",
        "dependencies",
        "$vocabulary",
    }
)
_SCHEMA_KEYED_LIST_IDENTIFIERS = frozenset({"dependentRequired", "dependencies"})
# "pattern" and "default" too: a grammar forces the model to match the rewritten value.
_SCHEMA_VALUED_IDENTIFIERS = frozenset(
    {
        "enum",
        "const",
        "required",
        "pattern",
        "default",
        "format",
        # "contentSchema" stays out: the recursive scan already reads its subschema.
        "contentEncoding",
        "contentMediaType",
        "discriminator",
        "xml",
        # References are resolved, not read; "$ref" can also name an external URI.
        "$ref",
        "$id",
        "$anchor",
        "$schema",
        "$dynamicRef",
        "$dynamicAnchor",
        # Draft-2019-09 and draft-04 spellings of the same resolution targets.
        "$recursiveRef",
        "$recursiveAnchor",
        "id",
    }
)


def _first_unsafe_leaf(value, markup = None):
    """The first string leaf, dict key included, that the rewrite would change."""
    stack = [value]
    seen = {id(value)}
    while stack:
        node = stack.pop()
        if isinstance(node, str):
            if neutralize_control_markup(node, markup) != node:
                return node
        elif isinstance(node, dict):
            for key, item in node.items():
                if isinstance(key, str) and neutralize_control_markup(key, markup) != key:
                    return key
                if id(item) not in seen:
                    seen.add(id(item))
                    stack.append(item)
        elif isinstance(node, list):
            for item in node:
                if id(item) not in seen:
                    seen.add(id(item))
                    stack.append(item)
    return None


# Anchor the scan on real schema roots: vendor extension fields are ordinary prose.
_SCHEMA_ROOT_KEYS = (
    "parameters",
    "input_schema",
    "inputSchema",
    "outputSchema",
    "output_schema",
    "returns",
    "response",
)


def _schema_roots(target):
    """The schema values of a tool declaration, or an empty list."""
    if not isinstance(target, dict):
        return []
    return [target[key] for key in _SCHEMA_ROOT_KEYS if isinstance(target.get(key), (dict, list))]


# "examples" holds instance samples, never subschemas, so do not descend into them.
_SCHEMA_INSTANCE_KEYS = frozenset({"examples", "example"})


def _unsafe_schema_identifier(value, markup = None):
    """Return the first schema identifier the rewrite would change, or None."""
    stack = [value]
    seen = {id(value)}
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key, item in node.items():
                if key in _SCHEMA_KEYED_IDENTIFIERS and isinstance(item, dict):
                    for name, dependents in item.items():
                        if (
                            isinstance(name, str)
                            and neutralize_control_markup(name, markup) != name
                        ):
                            return name
                        if key in _SCHEMA_KEYED_LIST_IDENTIFIERS and isinstance(dependents, list):
                            for dependent in dependents:
                                if (
                                    isinstance(dependent, str)
                                    and neutralize_control_markup(dependent, markup) != dependent
                                ):
                                    return dependent
                    # Only the VALUES are subschemas: a property named "format" is not the keyword.
                    for value in item.values():
                        if isinstance(value, (dict, list)) and id(value) not in seen:
                            seen.add(id(value))
                            stack.append(value)
                    continue
                elif key in _SCHEMA_VALUED_IDENTIFIERS:
                    unsafe = _first_unsafe_leaf(item, markup)
                    if unsafe is not None:
                        return unsafe
                if key in _SCHEMA_INSTANCE_KEYS:
                    continue
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
    """The function name a ``tool_choice`` pins, or None when it pins nothing. OpenAI: ``{"type":
    "function", "function": {"name": ...}}``; Anthropic: ``{"type": "tool", "name": ...}``. The
    string forms pin no particular tool."""
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
        nested = function.get("name") if isinstance(function, dict) else None
        for name in (nested, tool.get("name")):
            if isinstance(name, str):
                names.add(name)
    return names


def _tokenizer_strings(inner) -> Optional[list]:
    """Every string a tokenizer exposes: added tokens AND the base vocabulary. Neither source alone
    is enough: a tokenizer can carry unrelated added tokens while its chat sentinels stay in the
    base SentencePiece vocabulary, so short-circuiting on a populated added_tokens_decoder left a
    model's own turn boundaries out of the profile and a pasted copy reached the prompt
    byte-exact (#7066)."""
    out: list = []
    added = getattr(inner, "added_tokens_decoder", None)
    if isinstance(added, dict):
        out.extend(getattr(v, "content", v) for v in added.values())
    vocab = getattr(inner, "get_vocab", None)
    if callable(vocab):
        try:
            out.extend(vocab())
        except Exception:
            pass
    return out or None


def _vocabulary_of(tokenizer) -> Optional[list]:
    """The delimiter-shaped side of a tokenizer's vocabulary, or None."""
    inner = getattr(tokenizer, "tokenizer", tokenizer)
    return _tokenizer_strings(inner)


def mapped_chat_template(model_info: dict, active_model_name):
    """The template the generate-time mapper will install, resolved once and cached.

    ``_generate_chat_response_inner`` applies ``get_chat_template`` only when it renders, so a
    profile or an authorization catalog built before that saw the LOAD-time template, and a tool
    whose schema carries a delimiter the mapped template introduces was dropped from the prompt but
    still authorized for healing or execution (#7066).

    Resolved on a COPY of the tokenizer: ``get_chat_template`` assigns ``tokenizer.chat_template``,
    and this runs before the generation lock, so handing it the shared object would let this setup
    mutate a tokenizer another request is rendering with. Only the resulting template string is
    kept.
    """
    if not isinstance(model_info, dict):
        return None
    if "mapped_chat_template" in model_info:
        return model_info["mapped_chat_template"]
    mapped = None
    try:
        from utils.datasets import MODEL_TO_TEMPLATE_MAPPER
        from unsloth.chat_templates import get_chat_template

        name = (active_model_name or "").lower()
        if name in MODEL_TO_TEMPLATE_MAPPER:
            source = model_info.get("tokenizer")
            # Shallow copy: get_chat_template writes chat_template onto whatever it is given, and a
            # concurrent generation may be rendering with the shared object.
            try:
                probe = copy.copy(source)
            except Exception:
                probe = None
            if probe is None:
                return None  # cannot resolve safely; retry next turn
            remapped = get_chat_template(probe, chat_template = MODEL_TO_TEMPLATE_MAPPER[name])
            mapped = getattr(remapped, "chat_template", None)
    except Exception as exc:
        logger.debug("Could not resolve the mapped chat template early: %s", exc)
        return None  # unresolved, so retry next turn rather than pinning None
    model_info["mapped_chat_template"] = mapped
    return mapped


def _is_processor(obj) -> bool:
    """True for a container processor: it holds a tokenizer AND renders chats itself.
    ``ProcessorMixin.apply_chat_template`` does not switch to "tool_use" implicitly, so a
    processor renders "default" unless a template is named. Three call sites need that
    distinction and each had grown its own copy of the test."""
    return getattr(obj, "tokenizer", None) is not None and callable(
        getattr(obj, "apply_chat_template", None)
    )


def chat_render_target(processor, tokenizer = None):
    """The object whose chat template a render will actually use. ``_generate_vlm`` falls back to
    the nested tokenizer when the processor cannot render a chat itself, so anything profiling
    the prompt ahead of the render has to make the same choice. Reproducing the rule at the call
    site let the two drift: a processor without a usable ``chat_template`` was profiled as a
    processor, selecting "default", while the render used the nested tokenizer's tool_use
    template (#7066)."""
    if processor is None:
        return tokenizer
    if (
        getattr(processor, "apply_chat_template", None) is None
        or getattr(processor, "chat_template", None) is None
    ):
        nested = getattr(processor, "tokenizer", None)
        return processor if nested is None else nested
    return processor


_TOOLS_VARIABLE = re.compile(r"\btools\b")
# Only what Jinja evaluates counts: "{{ 'no tools available' }}" is not a tools read.
_JINJA_CODE = re.compile(r"\{\{(.*?)\}\}|\{%(.*?)%\}", re.S)
# Per-quote and escape aware so a stray apostrophe cannot swallow the expression.
_JINJA_STRING = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"", re.S)


# DeepSeek-R1 renders tool_calls and tool outputs without ever reading the tools variable.
_TOOL_TURN = re.compile(
    r"\.tool_calls\b"
    r"|\[\s*['\"]tool_calls['\"]\s*\]"
    r"|['\"]tool_calls['\"]\s+in\b"
    r"|\btool_calls\s+in\b"
    r"|role['\"\]\s]*==\s*['\"]tool['\"]"
    r"|['\"]tool['\"]\s*==[\s\['\"]*role"
)


def _jinja_code(body: str):
    """Yield only what Jinja evaluates: the inside of every {{ }} and {% %}."""
    for match in _JINJA_CODE.finditer(_JINJA_COMMENT.sub("", body)):
        yield match.group(1) or match.group(2) or ""


def _jinja_expression_spans(body: str) -> tuple:
    """The character ranges Jinja evaluates, with string literals taken back out. A "[" inside one
    of these is real indexing. Anywhere else, raw template text or inside a quoted literal the
    template prints, it is output the prompt will show."""
    spans: list = []
    for match in _JINJA_CODE.finditer(body):
        group = 1 if match.group(1) is not None else 2
        code, start = match.group(group), match.start(group)
        cursor = start
        for literal in _JINJA_STRING.finditer(code):
            spans.append((cursor, start + literal.start()))
            cursor = start + literal.end()
        spans.append((cursor, start + len(code)))
    return tuple(spans)


def _within(spans, index: int) -> bool:
    """True when *index* falls inside one of *spans*."""
    return any(start <= index < end for start, end in spans)


_ENDRAW = re.compile(r"\{%-?\s*endraw\s*-?%\}")


def _evaluated_spans(template: str) -> tuple:
    """The ranges Jinja evaluates, walked quote-aware, with string literals taken out.

    ``_jinja_expression_spans`` ends a block at the first ``}}``, so a template printing literal
    Jinja -- ``{{ "{{ example.0 }}" }}`` -- reads as code. That scanner is shared with
    ``model_markup`` (#7066) and stays as it is; this repair edits the template llama-server
    launches with, so it walks the blocks itself.

    A comment and a ``{% raw %}`` body are both skipped. Raw is tracked through this same walk
    rather than matched in the source, so tag text that only appears inside a comment or a literal
    cannot make a real expression between two of them look verbatim. Inside a raw body nothing is
    interpreted at all, the way Jinja reads it: the walk runs to the terminator, so a comment marker
    there stays text.
    """
    spans: list = []
    index, end = 0, len(template)
    verbatim = False
    while index < end - 1:
        if verbatim:
            terminator = _ENDRAW.search(template, index)
            if not terminator:
                break
            index = terminator.end()
            verbatim = False
            continue
        if template[index] != "{" or template[index + 1] not in "{%#":
            index += 1
            continue
        if template[index + 1] == "#":
            closed = template.find("#}", index + 2)
            index = end if closed < 0 else closed + 2
            continue
        closer = "}}" if template[index + 1] == "{" else "%}"
        block: list = []
        cursor = index + 2
        run = cursor
        quote = ""
        closed = False
        while cursor < end:
            char = template[cursor]
            if quote:
                if char == "\\":
                    cursor += 2
                    continue
                if char == quote:
                    quote = ""
                    run = cursor + 1
            elif char in "'\"":
                block.append((run, cursor))
                quote = char
            elif template.startswith(closer, cursor):
                block.append((run, cursor))
                closed = True
                break
            cursor += 1
        if not closed:
            break
        tag = template[index + 2 : cursor].strip().strip("-").strip()
        if closer == "%}" and tag == "raw":
            verbatim = True
        else:
            spans.extend(block)
        index = cursor + 2
    return tuple(spans)


_NUMERIC_MEMBER = re.compile(r"([A-Za-z_]\w*|\)|\])((?:\s*\.\s*\d+)+)")


def repair_numeric_member_access(template) -> Optional[str]:
    """Rewrite ``x.0`` as ``x[0]`` in evaluated code, or None when nothing needs it.

    llama.cpp's Jinja rejects a numeric member property. The throw lands inside its capability
    probe, which swallows it, so ``supports_object_arguments`` stays false and a replayed tool
    call's arguments are never decoded back into an object -- the template's own
    ``arguments.items()`` then dies on the JSON string (GLM-5.3).

    Only the ranges Jinja evaluates are rewritten, never prompt text, a quoted literal, a raw block,
    or Jinja syntax a template prints as an example. A chain rewrites whole ("x.0.1" -> "x[0][1]"):
    leaving the tail behind still throws. Jinja lets whitespace sit around each dot, and llama.cpp
    throws on that spelling too.
    """
    if not isinstance(template, str) or not template:
        return None
    spans = _evaluated_spans(template)
    out: list = []
    cursor = 0
    for match in _NUMERIC_MEMBER.finditer(template):
        if not _within(spans, match.start()):
            continue
        out.append(template[cursor : match.start()])
        indices = "".join(f"[{n}]" for n in re.findall(r"\d+", match.group(2)))
        out.append(f"{match.group(1)}{indices}")
        cursor = match.end()
    if not out:
        return None
    out.append(template[cursor:])
    return "".join(out)


def _reads_tools_variable(body: str) -> bool:
    """True when *body* evaluates the ``tools`` variable, rather than printing the word."""
    return any(_TOOLS_VARIABLE.search(_JINJA_STRING.sub("", code)) for code in _jinja_code(body))


def _round_trips_tool_calls(body: str) -> bool:
    """True when *body* renders assistant tool calls or tool results."""
    return any(_TOOL_TURN.search(code) for code in _jinja_code(body))


def _template_reads_tools(
    value,
    tools,
    prefer_tool_use: bool = True,
    require_tools_variable: bool = False,
) -> bool:
    """True unless the template selected out of *value* takes no part in tool calling.

    Reading the ``tools`` variable is the direct case. Replaying tool calls counts too: such a
    template round-trips a tool turn it never advertised, so the schema came from the caller's own
    system prompt and the catalog is authorized after all.

    ``require_tools_variable`` drops that second clause, for callers asking whether THIS render puts
    the schema in the prompt: a template that only round-trips renders byte-identically with and
    without a catalog, so the healer would otherwise promote calls for tools the model never saw
    (#7066).
    """
    bodies = _selected_template_strings_from_value(value, tools, prefer_tool_use = prefer_tool_use)
    if not bodies:
        # Unreadable, not proven silent: emptying the catalog would disable healing broadly.
        return True
    if require_tools_variable:
        return any(_reads_tools_variable(body) for body in bodies)
    return any(_reads_tools_variable(body) or _round_trips_tool_calls(body) for body in bodies)


def _accepts_tools_kwarg(target) -> bool:
    """False only when ``apply_chat_template`` provably rejects ``tools=``.
    apply_chat_template_for_generation catches that TypeError and succeeds with a no-tools
    attempt, so the prompt carries no schema however often the template body names the variable.
    Signature rather than a probe render: a render needs messages this function does not have,
    and would raise for unrelated reasons on a strict template."""
    apply = getattr(target, "apply_chat_template", None)
    if apply is None:
        return True
    try:
        parameters = inspect.signature(apply).parameters
    except (TypeError, ValueError):
        return True
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return True
    return "tools" in parameters


def _renders_tool_schema(
    target,
    template,
    tools,
    template_is_processor: bool = False,
) -> bool:
    """True unless the template *target* will select provably cannot advertise tools. A processor is
    held to the stricter test: its render has no native-template fallback behind it (a text
    model's template cannot place the image), so what the processor's own body does with
    ``tools`` is the whole answer. ``template_is_processor`` says the *template* is a processor
    body even though *target* is not: under the orchestrator the route has only the body mirrored
    through worker IPC, which would otherwise be judged by the permissive tokenizer rule
    (#10092)."""
    if tools and not _accepts_tools_kwarg(target):
        return False
    value = template or getattr(target, "chat_template", None)
    if not value:
        value = getattr(getattr(target, "tokenizer", None), "chat_template", None)
    is_processor = template_is_processor or _is_processor(target)
    return _template_reads_tools(
        value,
        tools,
        prefer_tool_use = not is_processor,
        require_tools_variable = is_processor,
    )


def renderable_tool_catalog_for_targets(
    tools,
    targets,
    model_info,
    cache = None,
    active_model_name = None,
    template = None,
    template_is_processor: bool = False,
):
    """The catalog safe under every object a backend could render this turn with.

    The two backends disagree about which object renders a text turn on a vision model: MLX keeps
    the processor when it has a usable chat template, while the transformers path unwraps to the
    nested tokenizer unconditionally. Their profiles can differ, so authorizing against one of them
    lets the other's render drop a tool that stays in the healer's catalog (#7066).

    Rather than guessing which backend will serve the request, this takes the same conservative
    intersection ``renderable_tool_catalog`` already takes across the active and native templates: a
    tool has to survive every candidate to stay authorized. Chained rather than intersected by name,
    so the surviving descriptions carry every candidate's sweep too. Sanitizing an already-sanitized
    catalog is stable, since a broken marker no longer matches, so a clean catalog stays
    byte-identical.
    """
    live = [target for target in targets if target is not None]
    if not live:
        # The parent-side mirror carries no tokenizer; a None target takes the curated sweep (#7066).
        return renderable_tool_catalog(
            tools, None, model_info, cache, active_model_name, template, template_is_processor
        )
    catalog = tools
    for target in live:
        catalog = renderable_tool_catalog(
            catalog, target, model_info, cache, active_model_name, template, template_is_processor
        )
        if not catalog:
            return catalog
    return catalog


def renderable_tool_catalog(
    tools,
    tokenizer,
    model_info,
    cache = None,
    active_model_name = None,
    template = None,
    template_is_processor: bool = False,
):
    """The catalog that survives EVERY template this request could render with.

    A tool-calling turn may render with the active template or, when that template drops the schema,
    with the model's native one, and the two profiles can disagree about which tools carry markup. A
    healer or controller is an authorization boundary, so it has to be built from the catalog that
    is safe either way: promoting a call for a tool the prompt never advertised is the failure this
    guards (#7066).

    The cost of the conservative direction is narrow and one-sided. A tool dropped only by the
    template that did NOT get selected stays advertised and directly callable; it just is not
    auto-healed out of text-form output that round.
    """
    # The mapper installs its template on the TOKENIZER; a processor render never sees it.
    if template is None and not _is_processor(tokenizer):
        template = mapped_chat_template(model_info or {}, active_model_name)
    safe = neutralize_tool_descriptions(
        tools, cache, markup_for_tokenizer(tokenizer, tools, template)
    )
    if not safe:
        return safe

    def _unadvertised():
        logger.info(
            "No chat template this request could select renders tool schemas; text-form "
            "tool calls will be relayed as prose rather than healed."
        )
        return []

    active_renders_tools = _renders_tool_schema(
        tokenizer, template, tools, template_is_processor = template_is_processor
    )
    # A processor renders "default" with no native fallback, so a template ignoring tools advertises none.
    if (template_is_processor or _is_processor(tokenizer)) and not active_renders_tools:
        return _unadvertised()
    # Resolved, not read: the cache is still empty on the first request needing the fallback.
    native_tpl = resolve_native_chat_template(
        model_info or {}, active_model_name, (model_info or {}).get("hf_token")
    )
    # With no native template to fall back to, the no-tools prompt stands and the catalog must say so.
    if not native_tpl:
        return safe if active_renders_tools else _unadvertised()
    if not active_renders_tools and not _template_reads_tools(native_tpl, tools):
        return _unadvertised()
    native = model_markup(
        native_tpl,
        _vocabulary_of(tokenizer),
        tools,
        specials = _special_token_strings(getattr(tokenizer, "tokenizer", tokenizer)),
    )
    if native is None:
        return safe
    kept = catalog_tool_names(neutralize_tool_descriptions(tools, cache, native))
    narrowed = [t for t in safe if catalog_tool_names([t]) <= kept]
    return safe if len(narrowed) == len(safe) else narrowed


def reconciled_tool_choice(tool_choice, openai_tools, safe_tools):
    """Downgrade a forced ``tool_choice`` to "auto" when WE dropped its tool (#7066). Only when the
    neutralizer removed it: the name has to be in the caller's catalog and gone from the
    sanitized one. A client forcing a function it never declared is a different, pre-existing
    case the healing path deliberately reads to decide a streamed call must NOT be promoted, so
    rewriting it there would change unrelated behaviour."""
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


def forced_tool_catalog(tool_choice, tools):
    forced = forced_tool_name(tool_choice)
    if forced is None:
        return []
    return [
        tool
        for tool in tools or []
        if isinstance(tool, dict)
        and isinstance(tool.get("function"), dict)
        and tool["function"].get("name") == forced
    ]


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
    # Hermes-3 ships the list form, which selects the same as the dict form.
    if isinstance(template, (list, tuple)):
        named = {
            entry["name"]: entry["template"]
            for entry in template
            if isinstance(entry, dict)
            and isinstance(entry.get("name"), str)
            and isinstance(entry.get("template"), str)
        }
        if named:
            return _selected_template_strings_from_value(
                named, tools, prefer_tool_use = prefer_tool_use
            )
        return ()
    if not isinstance(template, dict):
        return ()
    if prefer_tool_use and tools and isinstance(template.get("tool_use"), str):
        return (template["tool_use"],)
    if isinstance(template.get("default"), str):
        return (template["default"],)
    values = tuple(value for value in template.values() if isinstance(value, str))
    return values if len(values) == 1 else ()


def _selected_chat_template_strings(tokenizer, tools = None) -> tuple[str, ...]:
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
    # ProcessorMixin.apply_chat_template uses "default" unless chat_template= names another.
    return _selected_template_strings_from_value(
        getattr(tokenizer, "chat_template", None),
        tools,
        prefer_tool_use = not _is_processor(tokenizer),
    )


def _detect_reasoning_channel_markers_from_templates(
    templates: tuple[str, ...],
) -> Optional[tuple[str, ...]]:
    """Return native reasoning markers only when a template emits them."""
    if any(opener in template for template in templates for opener in _GEMMA_TEMPLATE_OPENERS):
        return _GEMMA_THOUGHT_OPEN, _GEMMA_THOUGHT_CLOSE
    if any(_ATEM_TEMPLATE_OPENER in template for template in templates):
        return _ATEM_REASONING_RECIPIENT, _ATEM_REPLY_RECIPIENT
    return None


def detect_reasoning_channel_markers(tokenizer, tools = None) -> Optional[tuple[str, ...]]:
    """Return the native reasoning-channel markers a tokenizer's template emits. Detection uses the
    active chat template rather than model names or vocabulary membership: some models expose
    Gemma control tokens without using the native thought-channel response protocol, and those
    must keep normal ``skip_special_tokens`` streaming."""
    for obj in _tokenizer_objects(tokenizer):
        templates = _selected_chat_template_strings(obj, tools)
        if templates:
            return _detect_reasoning_channel_markers_from_templates(templates)
    return None


def detect_reasoning_channel_markers_from_template(
    template, tools = None
) -> Optional[tuple[str, ...]]:
    """Return native reasoning-channel markers from a raw template value."""
    return _detect_reasoning_channel_markers_from_templates(
        _selected_template_strings_from_value(template, tools)
    )


def detect_reasoning_channel_markers_from_model_info(
    tokenizer,
    model_info: Optional[dict] = None,
    tools = None,
) -> Optional[tuple[str, ...]]:
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
    reasoning_channel_markers: Optional[tuple[str, ...]] = None
    # The native fallback can drop a tool the active profile kept.
    advertised_tools: Optional[list] = None


def _atem_header_can_extend(tail: str) -> bool:
    """Whether ``tail`` is a prefix of some recipient header."""
    for prefix in _ATEM_HEADER_PREFIXES:
        if prefix.startswith(tail):
            return True
        if tail.startswith(prefix):
            rest = tail[len(prefix) :]
            if _ATEM_PARTIAL_TAIL_RE.fullmatch(rest):
                return True
    return False


def _split_partial_atem_header(text: str) -> tuple[str, str]:
    """Hold the longest suffix that may still become a recipient header."""
    for start in range(max(0, len(text) - _ATEM_HEADER_MAX_LEN), len(text)):
        if _atem_header_can_extend(text[start:]):
            return text[:start], text[start:]
    return text, ""


def _split_partial_marker(text: str, marker: str) -> tuple[str, str]:
    """Hold the longest suffix that may become ``marker`` in the next chunk."""
    for length in range(min(len(text), len(marker) - 1), 0, -1):
        if text.endswith(marker[:length]):
            return text[:-length], text[-length:]
    return text, ""


def _find_atem_block_end(text: str, start: int = 0) -> tuple[int, int]:
    """Earliest block terminator at or after ``start``. Callers that keep feeding the same buffer
    pass what they have already searched, so a held block costs one scan."""
    index, length = -1, 0
    for marker in _ATEM_BLOCK_ENDS:
        found = text.find(marker, start)
        if found >= 0 and (index < 0 or found < index):
            index, length = found, len(marker)
    return index, length


def _atem_partial_framing_len(text: str) -> int:
    """Length of the trailing fragment that can only be framing the model never finished: one
    opening with "<", or a bare first header that has reached its "<|message|>". "...auto=true"
    is still prose, so it stays."""
    for size in range(min(len(text), _ATEM_HEADER_MAX_LEN), 0, -1):
        tail = text[-size:]
        if not (tail.startswith("<") or (tail.startswith("to=") and "<|" in tail)):
            continue
        if any(marker.startswith(tail) for marker in _ATEM_BLOCK_ENDS):
            return size
        if _atem_header_can_extend(tail):
            return size
    return 0


def _split_partial_atem_boundary(text: str) -> tuple[str, str]:
    """Between blocks either a header or a block terminator may follow, and both are consumed there,
    so a partial of either has to be held for the next chunk."""
    stable, held = _split_partial_atem_header(text)
    candidate, tail = _split_partial_atem_block_end(text)
    return (candidate, tail) if len(tail) > len(held) else (stable, held)


def _split_partial_atem_block_end(text: str) -> tuple[str, str]:
    stable, held = text, ""
    for marker in _ATEM_BLOCK_ENDS:
        candidate, tail = _split_partial_marker(text, marker)
        if len(tail) > len(held):
            stable, held = candidate, tail
    return stable, held


def _atem_parameter_value(raw: str):
    """JSON where it parses, otherwise the text as written (the grammar's ``allow_non_json``).
    Anything that will not re-serialize stays text: json.loads accepts NaN and the infinities,
    which RFC 8259 s6 does not, and 1e400 becomes inf without ever being one."""
    try:
        value = json.loads(raw)
        json.dumps(value, allow_nan = False)
    except (ValueError, RecursionError):
        return raw
    return value


def _atem_unfinished_tag(text: str) -> int:
    """Offset of the first ATEM tag the model had not finished writing, or -1. Only the last tag
    start can be unfinished, and it must match a tag name to a name boundary, so "<atem:invoker
    ..." stays the prose it is."""
    starts = [match.start() for match in _ATEM_TAG_START_RE.finditer(text)]
    if not starts:
        return -1
    start = starts[-1]
    fragment = text[start:]
    if ">" in fragment:
        return -1
    for tag in _ATEM_TAGS:
        if tag.startswith(fragment):
            return start
        if fragment.startswith(tag) and fragment[len(tag) : len(tag) + 1] not in _ATEM_NAME_CHARS:
            return start
    return -1


def _atem_surrounding_text(segment: str, *, complete: bool = True) -> str:
    """Text around calls, minus the envelope; whitespace alone is framing. ``complete`` is False for
    the tail of a block the stream cut short, where a trailing fragment of markup the model had
    not finished writing is framing too."""
    for tag in _ATEM_CALLS_ENVELOPE:
        segment = segment.replace(tag, "")
    if not complete:
        unfinished = _atem_unfinished_tag(segment)
        if unfinished >= 0:
            segment = segment[:unfinished]
    return segment if segment.strip() else ""


def _atem_block_pieces(block: str, *, complete: bool) -> Optional[list[tuple[bool, str]]]:
    """Split a tool-addressed block into ``(is_call, text)`` pieces, or None when it holds no call
    syntax: nothing downstream recognizes the native form, so a block passed through is shown as
    prose instead of executed. Text around the calls keeps its place among them. ``complete`` is
    False for a block the stream cut short."""
    pieces: list[tuple[bool, str]] = []
    cursor = 0
    saw_call = False
    for opening in _ATEM_INVOKE_OPEN_RE.finditer(block):
        if opening.start() < cursor:
            continue
        saw_call = True
        pieces.append((False, _atem_surrounding_text(block[cursor : opening.start()])))
        end = block.find(_ATEM_INVOKE_CLOSE, opening.end())
        if end < 0:
            return pieces
        body = block[opening.end() : end]
        parameters = list(_ATEM_PARAMETER_RE.finditer(body))
        if len(parameters) != body.count("<atem:parameter") or "<atem:invoke" in body:
            # Unclosed or stolen closer would run a tool the model did not specify.
            pieces.append((False, block[opening.start() : end + len(_ATEM_INVOKE_CLOSE)]))
            cursor = end + len(_ATEM_INVOKE_CLOSE)
            continue
        arguments = {
            parameter.group("key"): _atem_parameter_value(parameter.group("value"))
            for parameter in parameters
        }
        pieces.append(
            (
                True,
                "<tool_call>"
                + json.dumps({"name": opening.group("name"), "arguments": arguments})
                + "</tool_call>",
            )
        )
        cursor = end + len(_ATEM_INVOKE_CLOSE)
    if not saw_call:
        return None
    pieces.append((False, _atem_surrounding_text(block[cursor:], complete = complete)))
    return pieces


class ReasoningChannelNormalizer:
    """Incrementally convert one native reasoning channel to ``<think>``. The parser follows
    mlx-vlm's streaming boundary behavior but emits Unsloth's canonical text contract. Only the
    configured opening and closing markers are consumed; tool-call and other control markers
    remain available to downstream parsers."""

    def __init__(
        self,
        opening_marker: str,
        closing_marker: str,
        *,
        in_reasoning: bool = False,
    ):
        self._opening_marker = opening_marker
        self._closing_marker = closing_marker
        self._buffer = ""
        self._in_reasoning = in_reasoning
        self._reasoning_done = False
        self._skip_opening_newline = False
        # A prompt-supplied opener never reaches the stream, so <think> is owed to the first text delta.
        self._pending_open = in_reasoning

    def feed(self, text: str) -> str:
        """Consume a raw text delta and return the stable canonical delta."""
        self._buffer += text or ""
        output: list[str] = []
        if self._pending_open and self._buffer:
            output.append(_THINK_OPEN)
            self._pending_open = False
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
        """Flush a stream that ended and close an open think block. Ended, not merely finished
        generating: a stop sequence ends a turn as a stop token does, and the block it cut inside
        is still owed its close."""
        output = self.drain()
        if self._in_reasoning:
            if not self._pending_open:
                output += _THINK_CLOSE
            self._pending_open = False
            self._in_reasoning = False
            self._reasoning_done = True
        return output

    def drain(self) -> str:
        """Flush buffered literal text without synthesizing a closing tag."""
        output = self._buffer
        self._buffer = ""
        return output


class RecipientChannelNormalizer:
    """Convert a recipient-addressed reasoning protocol to ``<think>``.

    Muse Glimmer addresses every assistant block to a recipient: "self" carries reasoning, "user"
    the reply, any other name a tool call. Blocks repeat, so this tracks the recipient of each
    rather than one opener/closer pair. Generation resumes after the prompt's trailing
    ``<|start|>assistant``, so the first header arrives without that prefix. A tool-addressed block
    is held until it closes and then rewritten as a canonical ``<tool_call>``: no downstream parser
    recognizes the native form.
    """

    def __init__(
        self,
        reasoning_recipient: str = "self",
        reply_recipient: str = "user",
    ):
        self._reasoning_recipient = reasoning_recipient
        self._reply_recipient = reply_recipient
        self._buffer = ""
        self._in_reasoning = False
        self._in_reply = False
        self._tool_header = None
        self._passthrough = False
        self._skip_opening_newline = False
        self._between_blocks = True
        self._tool_scanned = 0

    def feed(self, text: str) -> str:
        """Consume the next slice of model output and return the text it settles."""
        self._buffer += text or ""
        output: list[str] = []
        while self._buffer:
            if self._passthrough:
                output.append(self._buffer)
                self._buffer = ""
                break

            if self._tool_header is not None:
                index, length = _find_atem_block_end(self._buffer, self._tool_scanned)
                if index < 0:
                    self._tool_scanned = max(0, len(self._buffer) - _ATEM_BLOCK_END_MAX_LEN + 1)
                    break
                block = self._buffer[:index]
                self._buffer = self._buffer[index + length :]
                self._tool_header = None
                self._tool_scanned = 0
                pieces = _atem_block_pieces(block, complete = True)
                self._between_blocks = True
                if pieces is not None:
                    output.append("".join(text for _, text in pieces))
                    continue
                output.append(_atem_surrounding_text(block))
                continue

            if self._in_reply:
                index, length = _find_atem_block_end(self._buffer)
                if index < 0:
                    stable, self._buffer = _split_partial_atem_block_end(self._buffer)
                    output.append(stable)
                    break
                output.append(self._buffer[:index])
                self._buffer = self._buffer[index + length :]
                self._in_reply = False
                self._between_blocks = True
                continue

            if self._in_reasoning:
                if self._skip_opening_newline:
                    if self._buffer == "\r":
                        break  # cannot tell "\r\n" from a lone "\r" yet
                    for newline in ("\r\n", "\n"):
                        if self._buffer.startswith(newline):
                            self._buffer = self._buffer[len(newline) :]
                            break
                    self._skip_opening_newline = False
                    if not self._buffer:
                        break
                index, length = _find_atem_block_end(self._buffer)
                if index < 0:
                    stable, self._buffer = _split_partial_atem_block_end(self._buffer)
                    output.append(stable)
                    break
                output.append(self._buffer[:index])
                self._buffer = self._buffer[index + length :]
                output.append(_THINK_CLOSE)
                self._in_reasoning = False
                self._between_blocks = True
                continue

            # Whitespace between blocks is framing; emitted, it splits one reasoning pass in the UI.
            if self._between_blocks:
                stripped = self._buffer.lstrip()
                if stripped != self._buffer:
                    self._buffer = stripped
                    if not self._buffer:
                        break

            match = _ATEM_HEADER_RE.search(self._buffer)
            block_end, end_len = _find_atem_block_end(self._buffer)
            if block_end >= 0 and (match is None or block_end < match.start()):
                output.append(self._buffer[:block_end])
                self._buffer = self._buffer[block_end + end_len :]
                self._between_blocks = True
                continue
            if match is None:
                stable, self._buffer = _split_partial_atem_boundary(self._buffer)
                output.append(stable)
                self._between_blocks = self._between_blocks and not stable
                break

            recipient = match.group("recipient")
            preface = self._buffer[: match.start()]
            self._between_blocks = self._between_blocks and not preface
            if recipient == self._reasoning_recipient:
                output.append(self._buffer[: match.start()])
                self._buffer = self._buffer[match.end() :]
                output.append(_THINK_OPEN)
                self._in_reasoning = True
                self._skip_opening_newline = True
            elif recipient == self._reply_recipient:
                output.append(self._buffer[: match.start()])
                self._buffer = self._buffer[match.end() :]
                self._in_reply = True
            else:
                output.append(self._buffer[: match.start()])
                self._tool_header = self._buffer[match.start() : match.end()]
                self._buffer = self._buffer[match.end() :]
        return "".join(output)

    def finish(self) -> str:
        """Flush a naturally completed stream, closing an open reasoning block and keeping any call
        that closed inside a block the model never terminated."""
        output = self._flush(keep_calls = True)
        if self._in_reasoning:
            output += _THINK_CLOSE
            self._in_reasoning = False
        return output

    def drain(self) -> str:
        """Flush buffered text without completing anything the model left open. Text held back
        survives; a call held back does not, so cancelling a turn can never be the thing that
        starts a tool running."""
        return self._flush(keep_calls = False)

    def _flush(self, *, keep_calls: bool) -> str:
        output = self._buffer
        self._buffer = ""
        if self._tool_header is not None:
            pieces = _atem_block_pieces(output, complete = False)
            if pieces is None:
                output = _atem_surrounding_text(output, complete = False)
            else:
                output = "".join(t for is_call, t in pieces if keep_calls or not is_call)
            self._tool_header = None
        elif output:
            held = _atem_partial_framing_len(output)
            output = output[: len(output) - held] if held else output
        if not self._in_reasoning:
            # Text before a later delta is already emitted, so it must not be re-parsed as a header.
            self._passthrough = True
        return output


def make_reasoning_normalizer(markers: tuple[str, ...], *, in_reasoning: bool = False):
    if markers and markers[0] == _ATEM_REASONING_RECIPIENT:
        # The generation prompt ends at "<|start|>assistant", so the model always writes its own header.
        return RecipientChannelNormalizer(*markers)
    return ReasoningChannelNormalizer(*markers, in_reasoning = in_reasoning)


def prompt_opens_reasoning_channel(
    prompt: Optional[str],
    markers: Optional[tuple[str, str]],
    continued: bool = False,
) -> bool:
    """Whether a rendered prompt *ends* by opening the native reasoning channel.

    Gemma-style templates end a post-tool generation prompt with the opener, so generation starts
    inside reasoning and emits only the closing marker.

    Only the tail decides -- the rule ``strip_open_reasoning_prefill`` already uses for ``<think>``.
    Position alone cannot tell a template's own prefill from replayed history, which keeps this
    markup through ``neutralize_control_markup_in_messages``, so an opener with content after it
    reads as closed; otherwise history could hide a plain answer in a think block. The cost is that
    a spliced continuation resuming inside a channel also reads as closed, leaving its reasoning
    visible as it was before.

    ``continued`` means this render resumed a trailing assistant turn, so the prompt ends on client
    text whose tail proves nothing. It is the render's own state, not the request flag, which
    outlives the continuation: the next tool-loop pass keeps the flag but renders an ordinary
    post-tool prompt that must still be read.
    """
    if continued or not prompt or not markers:
        return False
    opening_marker = markers[0]
    opened_at = prompt.rfind(opening_marker)
    if opened_at < 0:
        return False
    return not prompt[opened_at + len(opening_marker) :].strip()


def normalize_reasoning_snapshots(
    stream,
    tokenizer = None,
    cancel_event = None,
    markers: Optional[tuple[str, ...]] = None,
    tools = None,
    prompt: Optional[str] = None,
    continued: bool = False,
    ended = None,
):
    """Normalize a prefix-monotonic cumulative text stream when supported. ``ended`` is read after
    the stream: a turn a stop sequence ended still owes its open block a close, even if a cancel
    landed on the same step."""
    markers = markers or detect_reasoning_channel_markers(tokenizer, tools = tools)
    if markers is None:
        yield from stream
        return

    normalizer = make_reasoning_normalizer(
        markers,
        in_reasoning = prompt_opens_reasoning_channel(prompt, markers, continued),
    )
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

    cancelled = (
        not (ended is not None and ended()) and cancel_event is not None and cancel_event.is_set()
    )
    tail = normalizer.drain() if cancelled else normalizer.finish()
    if tail:
        normalized_output += tail
        yield normalized_output


def detect_think_prefill(
    prompt: Optional[str],
    special_tokens = None,
    *,
    preserves_think_close: bool = False,
    resumes_thought: bool = False,
) -> str:
    """Return the trailing open ``<think>`` prefill of a rendered prompt.

    Reasoning templates (Qwen3.6, DeepSeek-R1-style) end the generation prompt with ``<think>\\n``
    so the model starts reasoning immediately. Because that opening tag is part of the *prompt*,
    skip_prompt streaming never emits it, and the frontend's ``<think>``/``</think>`` parser shows
    the reasoning as plain text instead of a thinking block. (The GGUF path is unaffected:
    llama-server's reasoning parser returns ``reasoning_content``, which gets re-wrapped in think
    tags.)

    Returns the exact prompt tail to re-emit at the start of the generated stream, or ``""`` when
    the prompt does not end with an open think block, including the ``enable_thinking=False`` case
    where templates prefill an already-closed ``<think>\\n\\n</think>``.

    ``special_tokens`` is the tokenizer's special-token list. If ``</think>`` is one, the streamer's
    skip_special_tokens strips the model's closing tag, so re-emitting the open would leave an
    unclosed block that swallows the answer; in that case return ``""`` and fall back to plain text.

    ``preserves_think_close`` says the stream keeps that closer anyway, as
    ``NativeToolTokenDecoder`` does so the parser can see a call rehearsed inside the block, and as
    a path streaming the detokenizer's own text does. The special-token list then says nothing, and
    skipping the opener is the same bug mirrored: a stray ``</think>``.

    ``resumes_thought``: the prompt ends inside a resumed thought (the client's text), so only the
    bare opener is returned.
    """
    if not prompt:
        return ""
    open_idx = prompt.rfind(_THINK_OPEN)
    if open_idx == -1:
        return ""
    tail = prompt[open_idx:]
    if _THINK_CLOSE in tail or (tail.strip() != _THINK_OPEN and not resumes_thought):
        return ""
    if not preserves_think_close and special_tokens and _THINK_CLOSE in set(special_tokens):
        return ""
    return _THINK_OPEN if resumes_thought else tail


def _normalize_tool_call_arguments(messages: list) -> list:
    """Coerce each assistant ``tool_calls[].function.arguments`` from a JSON string to a dict. The
    OpenAI wire format carries ``arguments`` as a JSON string, but some chat templates (e.g. the
    stricter Qwen tool templates shipped with mlx-community checkpoints) iterate
    ``arguments.items()`` and raise ``TypeError: Can only get item pairs from a mapping.`` on the
    string form when a prior tool call is re-rendered on the next turn. A dict works on both
    strict and lenient templates, so parse the string; leave non-JSON or non-dict values
    untouched. Returns the original list unchanged when nothing needed coercing (no copy)."""
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
                except (ValueError, TypeError, RecursionError):
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
    """Llama 3.x templates render one call per message, so split parallel calls into consecutive
    single-call messages, each followed by its own result."""
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


def _repair_orphan_tool_results(messages: list) -> list:
    """Placeholder call before each tool result lacking one (gpt-oss refuses orphans). Fallback only:
    the model reads it. A text turn takes the calls, since a second assistant turn breaks alternation."""
    mutated = False
    out: list = []
    linked = False
    repaired_at = None

    for message in messages:
        role = message.get("role") if isinstance(message, dict) else None
        if role != "tool":
            linked = role == "assistant" and bool(message.get("tool_calls"))
            repaired_at = None
            out.append(message)
            continue
        if linked and repaired_at is None:
            out.append(message)
            continue

        call_id = message.get("tool_call_id") or f"replayed_tool_{len(out)}"
        call = {
            "id": call_id,
            "type": "function",
            "function": {"name": message.get("name") or "tool", "arguments": {}},
        }
        if repaired_at is not None:
            calls = out[repaired_at]["tool_calls"]
            if all(c.get("id") != call_id for c in calls):
                out[repaired_at] = {**out[repaired_at], "tool_calls": [*calls, call]}
        elif out and isinstance(out[-1], dict) and out[-1].get("role") == "assistant":
            out[-1] = {**out[-1], "tool_calls": [call]}
            repaired_at = len(out) - 1
        else:
            out.append({"role": "assistant", "content": "", "tool_calls": [call]})
            repaired_at = len(out) - 1
        if not message.get("tool_call_id"):
            message = {**message, "tool_call_id": call_id}
        out.append(message)
        linked = True
        mutated = True

    return out if mutated else messages


_MARKUP_BY_TOKENIZER: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _special_token_strings(tokenizer) -> dict:
    """The concrete spelling of each special-token variable a template may emit."""
    specials: dict = {}
    for name in _SPECIAL_TOKEN_VARIABLES:
        try:
            value = getattr(tokenizer, name, None)
        except Exception:
            continue
        if value is not None and not isinstance(value, str):
            value = getattr(value, "content", None)
        if isinstance(value, str) and value:
            specials[name] = value
    return specials


def markup_for_tokenizer(
    tokenizer,
    tools = None,
    template = None,
) -> Optional[ModelMarkup]:
    """Profile the loaded tokenizer's own structural markers, cached per tokenizer. Returns None
    when the template and vocabulary cannot be read, which falls back to the curated patterns: an
    unreadable model stays fully swept rather than unprotected."""
    if tokenizer is None:
        return None
    try:
        cached = _MARKUP_BY_TOKENIZER.get(tokenizer)
    except TypeError:
        cached = None
    # Vision models: chat_template on the processor, vocabulary on the inner tokenizer.
    inner = getattr(tokenizer, "tokenizer", tokenizer)
    if not template:
        template = getattr(tokenizer, "chat_template", None)
    if not template:
        template = getattr(inner, "chat_template", None)
    # Keyed on template too: get_chat_template installs a mapped template on the SAME object later.
    is_processor = _is_processor(tokenizer)
    selector = bool(tools) and not is_processor
    if not isinstance(template, str):
        try:
            template_key = json.dumps(template, sort_keys = True, default = str)
        except (TypeError, ValueError):
            template_key = repr(template)
    else:
        template_key = template
    if isinstance(cached, dict):
        hit = cached.get((template_key, selector), _UNPARSED)
        if hit is not _UNPARSED:
            return hit
    else:
        cached = None
    tokens = None
    tokens = _tokenizer_strings(inner)
    profile = model_markup(
        template,
        tokens,
        tools,
        prefer_tool_use = not is_processor,
        specials = _special_token_strings(inner),
    )
    try:
        entry = cached if isinstance(cached, dict) else {}
        # Two selectors per template; trim if a tokenizer somehow cycles templates.
        if len(entry) >= 4:
            entry.clear()
        entry[(template_key, selector)] = profile
        _MARKUP_BY_TOKENIZER[tokenizer] = entry
    except TypeError:
        pass
    return profile


def trailing_assistant_text(messages: list) -> Optional[str]:
    """Plain text of a trailing assistant turn, else None. Only plain text resumes: tool calls and
    image parts have no resume point."""
    if not messages:
        return None
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "assistant":
        return None
    if last.get("tool_calls"):
        return None
    content = last.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "text":
                return None
            texts.append(str(part.get("text") or ""))
        return "".join(texts)
    return None


def trailing_assistant_resume_kind(messages: list) -> Optional[str]:
    """Return the trailing assistant field to resume, or None; prefer content over reasoning."""
    text = trailing_assistant_text(messages)
    if text:
        return "content"
    if text is None:
        return None
    reasoning = messages[-1].get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        return "reasoning_content"
    return None


def last_user_text(messages: list) -> str:
    """Text of the newest user turn, with any ``<img>`` markup stripped. Scans back rather than
    reading ``messages[-1]``: a continuation ends on the assistant partial. Stops at the newest
    user turn even when it is empty (an image-only message), so an older question is never
    resurrected."""
    from core.inference.message_content import content_to_text

    for message in reversed(messages or []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        return re.sub(r"<img[^>]*>", "", content_to_text(message.get("content"))).strip()
    return ""


_STRUCTURED_IMAGE_TYPES = ("image", "image_url", "input_image")
_STRUCTURED_VIDEO_TYPES = ("video", "video_url", "input_video")


def _count_structured_parts(content, types) -> int:
    if isinstance(content, list):
        return sum(_count_structured_parts(item, types) for item in content)
    if not isinstance(content, dict):
        return 0
    if str(content.get("type", "")).lower() in types:
        return 1
    return _count_structured_parts(content.get("content"), types)


def count_structured_images(content) -> int:
    return _count_structured_parts(content, _STRUCTURED_IMAGE_TYPES)


def count_structured_videos(content) -> int:
    return _count_structured_parts(content, _STRUCTURED_VIDEO_TYPES)


def structured_media_reprs(content) -> set:
    media_types = _STRUCTURED_IMAGE_TYPES + _STRUCTURED_VIDEO_TYPES
    if isinstance(content, list):
        values = (
            {str(content), json.dumps(content, ensure_ascii = False)}
            if _count_structured_parts(content, media_types)
            else set()
        )
        for item in content:
            values.update(structured_media_reprs(item))
        return values
    if not isinstance(content, dict):
        return set()
    if str(content.get("type", "")).lower() in media_types:
        return {str(content), json.dumps(content, ensure_ascii = False)}
    return structured_media_reprs(content.get("content"))


def prompt_serializes_structured_media(prompt, messages) -> bool:
    """Detect templates that embed the exact structured media object repr."""
    from core.inference.message_content import content_to_text

    media_reprs = set()
    for message in messages:
        if isinstance(message, dict):
            media_reprs.update(structured_media_reprs(message.get("content")))
    text_content = [
        content_to_text(message.get("content")) for message in messages if isinstance(message, dict)
    ]
    return any(
        prompt.count(media_repr) > sum(content.count(media_repr) for content in text_content)
        for media_repr in media_reprs
    )


def vlm_prompt_issue(prompt, messages) -> Optional[str]:
    """Name the way a VLM render came back unusable, else None. Shared by both backends so a defect
    one of them refuses stays refused by the other."""
    if not isinstance(prompt, str) or not prompt.strip():
        return "an empty prompt"
    if prompt_serializes_structured_media(prompt, messages):
        return "serialized structured image content"
    return None


def messages_have_tool_history(messages) -> bool:
    """True when the conversation replays a tool call or a tool result."""
    return any(
        isinstance(message, dict)
        and (
            message.get("role") == "tool"
            or message.get("tool_calls")
            or message.get("tool_call_id")
        )
        for message in messages
    )


def alternating_turns(messages: list) -> list:
    """User/assistant text turns, alternating and ending on the newest user turn as sent; of two
    same-role neighbours the later one is kept."""
    from core.inference.message_content import content_to_text, named_turn

    messages = list(messages or [])
    newest = max(
        (i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "user"),
        default = -1,
    )
    turns = []
    for index, message in enumerate(messages[: newest + 1]):
        turn = message
        if index != newest:
            if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
                continue
            text = content_to_text(message.get("content")).strip()
            # Empty user turns are kept: an earlier recording or picture replays with no text.
            if not text and message["role"] == "assistant":
                continue
            turn = named_turn({"role": message["role"], "content": text}, message)
        if turns and turns[-1]["role"] == turn["role"]:
            turns[-1] = turn
        elif turns or turn["role"] == "user":
            turns.append(turn)
    return turns


def messages_with_attached_image(
    messages: list,
    system_prompt: str = "",
    fallback_user_text: str = "",
    structured_content: bool = False,
    image: int = 1,
    video: bool = False,
    audio: Any = None,
    extra_audio: Sequence[Any] = (),
) -> list:
    """The conversation to render for a turn that carries attached media.

    Prepends *system_prompt* as a leading system turn, then injects *image* ``{"type": "image"}``
    parts, or a ``{"type": "video"}`` part, plus any *audio* waveform as an ``{"type": "audio"}``
    part (one more per *extra_audio* clip, in order), into the LAST user turn and leaves every
    other turn -- assistant ``tool_calls`` and ``role="tool"`` results included -- exactly as the
    caller sent it.
    Rebuilding from the newest user TEXT instead dropped the folded system instruction and the
    tool history an OpenAI tool loop replays (#10092). Nothing the caller owns is mutated: callers
    still read those dicts after generation, and a retry re-renders the same list.

    *structured_content* wraps content in part lists, as a processor template expects; MLX may
    render through the nested text tokenizer, whose template expects a string. *fallback_user_text*
    stands in for a user turn with no text of its own, and opens one when there is no user turn at
    all. Left empty, the conversation is unchanged, so a backend that would rather refuse an image
    nobody asked about keeps refusing it.
    """
    conversation = list(messages or [])
    if structured_content:
        # EVERY message: a processor template raises on a replayed turn left as a string.
        def _as_parts(message):
            if not isinstance(message, dict):
                return message
            body = message.get("content")
            if isinstance(body, list):
                return message
            if isinstance(body, str) and body:
                return {**message, "content": [{"type": "text", "text": body}]}
            return {**message, "content": []}

        conversation = [_as_parts(m) for m in conversation]
    if system_prompt:
        conversation.insert(
            0,
            {
                "role": "system",
                "content": (
                    [{"type": "text", "text": system_prompt}]
                    if structured_content
                    else system_prompt
                ),
            },
        )
    parts = [
        {"type": part_type}
        for part_type, wanted, counter in (
            ("image", image, count_structured_images),
            ("video", video, count_structured_videos),
        )
        if not any(
            isinstance(m, dict) and isinstance(m.get("content"), list) and counter(m["content"])
            for m in conversation
        )
        for _ in range(int(wanted))
    ]
    if audio is not None:
        parts.append({"type": "audio", "audio": audio})
        parts.extend({"type": "audio", "audio": clip} for clip in extra_audio)
    if not parts and not fallback_user_text:
        return conversation
    for index in range(len(conversation) - 1, -1, -1):
        message = conversation[index]
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            text = content if content.strip() else fallback_user_text or content
            content = [{"type": "text", "text": text}]
        elif not isinstance(content, list):
            break
        elif fallback_user_text and not last_user_text([message]):
            content = [*content, {"type": "text", "text": fallback_user_text}]
        conversation[index] = {**message, "content": parts + list(content)}
        return conversation
    if parts and fallback_user_text:
        conversation.append(
            {"role": "user", "content": parts + [{"type": "text", "text": fallback_user_text}]}
        )
    return conversation


def render_advertising_tools(render, tools):
    """Render with *tools*, and say whether the prompt actually carries them.

    Returns ``(prompt, advertised)``; *render* takes a catalog and returns a prompt. Comparing the
    two renders rather than reading the body: a body that merely names the ``tools`` variable can
    still drop the schema, and a renderer taking ``tools=`` through ``**kwargs`` can swallow it
    without ever raising. The no-tools probe runs FIRST so the prompt this turn uses is the last
    thing the renderer produced, for anything that caches or observes the render. A probe that
    raises answers "advertised": the render that failed is the throwaway one.
    """
    if not tools:
        return render(None), False
    try:
        without_tools = render(None)
    except Exception as exc:
        logger.debug("No-tools probe failed; keeping the tools prompt: %s", exc)
        return render(tools), True
    prompt = render(tools)
    return prompt, prompt != without_tools


def append_assistant_turn(
    conversation: list,
    assistant_msg: dict,
    *,
    continue_final_message: bool = False,
) -> None:
    """Append a generated assistant turn to *conversation*.

    A continuation leaves the resumed partial trailing, and the partial plus what the model just
    added are one turn, so they are merged: appending would instead give two consecutive assistant
    messages and break role alternation. Self-limiting, since after a tool result the conversation
    no longer ends with a plain assistant turn.

    Merge over the resumed turn rather than replacing it, or every key the partial carried but the
    continuation does not repeat is lost. ``extra_content`` is such a key, and Gemini reads the text
    part's thought signature back from it alone, so a resumed turn replayed without it is rejected.
    """
    prev_text = trailing_assistant_text(conversation) if continue_final_message else None
    if prev_text is not None and isinstance(assistant_msg.get("content"), str):
        merged_msg = {**conversation[-1], **assistant_msg}
        merged_msg["content"] = f"{prev_text}{assistant_msg['content']}"
        added_reasoning = assistant_msg.get("reasoning_content")
        if (
            isinstance(added_reasoning, str)
            and trailing_assistant_resume_kind(conversation) == "reasoning_content"
        ):
            merged_msg["reasoning_content"] = (
                f"{conversation[-1]['reasoning_content']}{added_reasoning}"
            )
        conversation[-1] = merged_msg
        return
    conversation.append(assistant_msg)


def strip_open_reasoning_prefill(prefix: str) -> str:
    """Drop a generation prompt's unclosed ``<think>`` opener. A splice appends the visible partial
    to this prefix, so on a template that prefills an open block (DeepSeek-R1, QwQ,
    Qwen3-Thinking) the answer would resume as reasoning. Only an opener the prompt itself ends
    on counts: a bare "<think>" typed into an earlier turn is rendered text, and cutting there
    would eat the transcript."""
    open_at = prefix.rfind(_THINK_OPEN)
    if open_at == -1 or open_at < prefix.rfind(_THINK_CLOSE):
        return prefix
    if prefix[open_at + len(_THINK_OPEN) :].strip():
        return prefix
    return prefix[:open_at]


class ThoughtUnresumableError(ValueError):
    """A trailing thought this model cannot reopen; the message is client-safe."""

    public = True
    openai_param = "continue_final_message"

    def __init__(self):
        super().__init__("This model cannot resume a response that stopped mid-thought. Use Retry.")


def template_resumes_thought(tokenizer, tools = None) -> bool:
    """Whether a thought can be reopened as ``<think>`` text (not native reasoning channels)."""
    if detect_reasoning_channel_markers(tokenizer, tools = tools) is not None:
        return False
    return any(
        _THINK_OPEN in template for template in _selected_chat_template_strings(tokenizer, tools)
    )


def splice_resumed_thought(prefix: str, thought: str) -> str:
    """Reopen *thought* on a generation prompt, cutting the template's own reasoning prefill (open
    or empty closed block) as llama-server does."""
    open_at = prefix.rfind(_THINK_OPEN)
    if open_at != -1 and prefix[open_at + len(_THINK_OPEN) :].strip() in ("", _THINK_CLOSE):
        prefix = prefix[:open_at]
    return f"{prefix}{_THINK_OPEN}{thought}"


def render_prompt_with_boundary(
    processor,
    messages: list,
    continue_final_message: bool = False,
    tools: Optional[list] = None,
) -> str:
    """Render *messages* through a renderer's own chat template. With *continue_final_message* the
    prompt ends inside the trailing assistant turn, so the model resumes it. Processors predating
    the kwarg get a manual splice, taking the partial from *messages* (which the caller already
    swept) rather than a separate copy: a raw partial could close the turn or open another role
    instead of resuming (#7066)."""
    from core.inference.mcp_images import prepare_image_turn_boundaries

    for template in _selected_chat_template_strings(processor, tools):
        messages = prepare_image_turn_boundaries(messages, template)
    extra = {"tools": tools} if tools else {}
    partial = trailing_assistant_text(messages) if continue_final_message else None
    if not partial:
        return processor.apply_chat_template(
            messages, add_generation_prompt = True, tokenize = False, **extra
        )
    try:
        return processor.apply_chat_template(
            messages,
            add_generation_prompt = False,
            continue_final_message = True,
            tokenize = False,
            **extra,
        )
    except TypeError:
        prefix = processor.apply_chat_template(
            messages[:-1], add_generation_prompt = True, tokenize = False, **extra
        )
        return f"{strip_open_reasoning_prefill(prefix)}{partial}"


def neutralize_for_render(tokenizer, messages: list, tools: Optional[list]):
    """Sweep the catalog and the messages for control markup, in the one correct order. Returns
    ``(messages, tools, markup)``. One place because the sweep is order dependent: a call site
    that re-derived it swept the messages against a profile the render would not select (#7066)."""
    markup = markup_for_tokenizer(tokenizer, tools)
    tools = neutralize_tool_descriptions(tools, None, markup)
    # Sanitizing can empty the catalog and flip the selector, so re-profile first.
    if bool(tools) != bool(markup and getattr(markup, "selected_with_tools", False)):
        markup = markup_for_tokenizer(tokenizer, tools)
    return neutralize_control_markup_in_messages(messages, None, markup), tools, markup


def apply_chat_template_for_generation(
    tokenizer,
    messages: list,
    *,
    tools: Optional[list] = None,
    enable_thinking: Optional[bool] = None,
    reasoning_effort: Optional[str] = None,
    preserve_thinking: Optional[bool] = None,
    continue_final_message: bool = False,
) -> str:
    """Render the chat prompt. Try richest kwargs first; drop one group at a time on TypeError.
    Jinja / missing-variable errors propagate. With *continue_final_message* the prompt ends
    inside the trailing assistant turn, so the model resumes the partial instead of restarting
    it."""
    from core.inference.mcp_images import prepare_image_turn_boundaries

    for template in _selected_chat_template_strings(tokenizer, tools):
        messages = prepare_image_turn_boundaries(messages, template)
    messages, tools, _markup = neutralize_for_render(tokenizer, messages, tools)
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

    # A tools=-less retry selects "default", so its messages need that template's profile.
    _fallback_markup = _UNPARSED

    def _swept_for(kwargs: dict, msgs: list) -> list:
        nonlocal _fallback_markup
        if not tools or "tools" in kwargs:
            return msgs
        if _markup is None:
            return msgs  # already swept with the curated superset
        if _fallback_markup is _UNPARSED:
            _fallback_markup = markup_for_tokenizer(tokenizer, None)
        return neutralize_control_markup_in_messages(msgs, None, _fallback_markup)

    _continue_text = trailing_assistant_text(messages) if continue_final_message else None
    _continuing = bool(_continue_text)
    _resumes_thought = (
        continue_final_message and trailing_assistant_resume_kind(messages) == "reasoning_content"
    )
    if _resumes_thought and not template_resumes_thought(tokenizer, tools):
        raise ThoughtUnresumableError()
    _boundary_kwargs = (
        {"add_generation_prompt": False, "continue_final_message": True}
        if _continuing
        else {"add_generation_prompt": True}
    )

    def _render(
        msgs: list,
        boundary: Optional[dict] = None,
        typeerror_fallback: Optional[list] = None,
    ) -> str:
        boundary = _boundary_kwargs if boundary is None else boundary
        last_exc: Optional[Exception] = None
        for kwargs in attempts:
            try:
                return tokenizer.apply_chat_template(
                    _swept_for(kwargs, msgs),
                    tokenize = False,
                    **boundary,
                    **kwargs,
                )
            except TypeError as e:
                last_exc = e
                if typeerror_fallback is not None:
                    try:
                        return tokenizer.apply_chat_template(
                            _swept_for(kwargs, typeerror_fallback),
                            tokenize = False,
                            **boundary,
                            **kwargs,
                        )
                    except Exception:
                        pass
                continue
            except Exception as e:
                last_exc = e
                break
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("apply_chat_template_for_generation: no attempt produced a result")

    def _render_continuation_manually(msgs: list, typeerror_fallback: Optional[list] = None) -> str:
        """For tokenizers predating ``continue_final_message`` (TypeError above). Prefix and partial
        come from the SAME swept copy: an attempt that drops the tools kwarg re-sweeps for the
        default template, whose markup would otherwise survive raw."""
        for kwargs in attempts:
            swept = _swept_for(kwargs, msgs)
            try:
                prefix = tokenizer.apply_chat_template(
                    swept[:-1], tokenize = False, add_generation_prompt = True, **kwargs
                )
            except TypeError:
                if typeerror_fallback is None:
                    continue
                swept = _swept_for(kwargs, typeerror_fallback)
                try:
                    prefix = tokenizer.apply_chat_template(
                        swept[:-1], tokenize = False, add_generation_prompt = True, **kwargs
                    )
                except Exception:
                    continue
            partial = trailing_assistant_text(swept) or _continue_text
            return f"{strip_open_reasoning_prefill(prefix)}{partial}"
        raise TypeError("no attempt rendered the continuation prefix")

    def _render_thought_continuation(msgs: list, typeerror_fallback: Optional[list] = None) -> str:
        # Templates render a final thought closed, so there is no boundary to cut at.
        for kwargs in attempts:
            swept = _swept_for(kwargs, msgs)
            try:
                prefix = tokenizer.apply_chat_template(
                    swept[:-1], tokenize = False, add_generation_prompt = True, **kwargs
                )
            except TypeError:
                if typeerror_fallback is None:
                    continue
                swept = _swept_for(kwargs, typeerror_fallback)
                try:
                    prefix = tokenizer.apply_chat_template(
                        swept[:-1], tokenize = False, add_generation_prompt = True, **kwargs
                    )
                except Exception:
                    continue
            return splice_resumed_thought(prefix, swept[-1]["reasoning_content"])
        raise TypeError("no attempt rendered the thought continuation prefix")

    def _render_with_fallback(msgs: list, typeerror_fallback: Optional[list] = None) -> str:
        if _resumes_thought:
            return _render_thought_continuation(msgs, typeerror_fallback)
        try:
            return _render(msgs, typeerror_fallback = typeerror_fallback)
        except TypeError:
            if not _continuing:
                raise
            return _render_continuation_manually(msgs, typeerror_fallback)

    # mappings first because Qwen3.5 renders string arguments as an empty call instead of raising
    normalized = _normalize_tool_call_arguments(messages)
    try:
        return _render_with_fallback(normalized, messages if normalized is not messages else None)
    except Exception:
        candidates: list = []
        if normalized is not messages:
            candidates.append(messages)
        split = _split_parallel_tool_calls(normalized)
        if split is not normalized:
            candidates.append(split)
        for candidate in candidates:
            try:
                return _render_with_fallback(candidate)
            except Exception:
                continue
        # Last and lazy: a history an earlier candidate renders never pays for the scan.
        repaired = _repair_orphan_tool_results(split)
        if repaired is not split:
            try:
                # Split again: gpt-oss renders only tool_calls[0] and names every later result after it.
                return _render_with_fallback(_split_parallel_tool_calls(repaired))
            except Exception:
                pass
        raise


def resolve_native_chat_template(
    model_info: dict,
    active_model_name,
    hf_token = None,
):
    """The model's native chat template, fetched once and cached on *model_info*. Returns False when
    the repo has none and None when the fetch failed, so a failure is retried rather than pinned.
    Shared by the render path and by the authorization catalog, which must know the native
    template on the FIRST request too: it is fetched during rendering, so a catalog built before
    that saw no native profile at all and could authorize a tool the native render then left out
    of the prompt (#7066)."""
    native_tpl = model_info.get("native_chat_template")
    if native_tpl is not None:
        return native_tpl
    template_source = model_info.get("base_model") or active_model_name
    if not template_source:
        return None
    # Reuse the load-time trust_remote_code; the stored flag already covers template_source.
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
        logger.warning("Could not load native chat template for '%s': %s", template_source, exc)
        # Do not cache False on a failed fetch: it would pin the tool-dropping override.
        return None
    model_info["native_chat_template"] = native_tpl
    return native_tpl


def render_native_template(
    *,
    model_info: dict,
    active_model_name: Optional[str],
    messages: list,
    tools: list,
    enable_thinking: Optional[bool] = None,
    reasoning_effort: Optional[str] = None,
    preserve_thinking: Optional[bool] = None,
    continue_final_message: bool = False,
    apply_fn = None,
    hf_token: Optional[str] = None,
    return_metadata: bool = False,
):
    """Render ``messages`` + ``tools`` with the model's NATIVE chat template.

    Some Unsloth override templates (e.g. ``mistral``, ``gemma-4``) do not emit the ``tools``
    schema, so a tool-calling turn silently stops advertising tools. The native template ships in
    the model repo and carries the family's tool-calling syntax; it is loaded straight from the repo
    (bypassing any override on the live tokenizer) and cached on ``model_info``. Returns the
    rendered prompt only if the native template actually emits the tools (render differs with vs
    without tools); otherwise ``None``. With ``return_metadata``, returns
    ``ChatTemplateRenderResult`` so callers can stream with the response protocol selected by this
    request's template.

    ``hf_token`` is the token the model was loaded with, passed to the repo load so a gated/private
    model's native template can still be fetched (otherwise the fallback fails silently and keeps
    the override prompt that dropped tools).

    ``trust_remote_code`` is sourced from ``model_info`` (the value the model was actually loaded
    with) rather than a call-site argument, so the reload uses exactly the consent already granted
    at load: a custom-code tokenizer repo raises in ``AutoTokenizer.from_pretrained`` without it,
    and the fallback would fail silently. For a LoRA adapter the reload targets the base model,
    whose remote code was gated and loaded under the same stored flag, so re-passing it executes no
    unconsented code.
    """
    if apply_fn is None:
        apply_fn = apply_chat_template_for_generation
    native_tpl = resolve_native_chat_template(model_info, active_model_name, hf_token)
    if not native_tpl:
        return None

    tokenizer = model_info.get("tokenizer") or model_info.get("processor")
    if tokenizer is None:
        return None
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    # Shallow copy: mutating the shared tokenizer.chat_template races concurrent requests.
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
            continue_final_message = continue_final_message,
        )
        no_tools = apply_fn(
            render_tokenizer,
            messages,
            tools = None,
            enable_thinking = enable_thinking,
            reasoning_effort = reasoning_effort,
            preserve_thinking = preserve_thinking,
            continue_final_message = continue_final_message,
        )
    except Exception as exc:
        logger.warning(
            "Native-template tool render failed for '%s': %s",
            active_model_name,
            exc,
        )
        return None
    if tools and with_tools == no_tools:
        return None
    if return_metadata:
        return ChatTemplateRenderResult(
            with_tools,
            _detect_reasoning_channel_markers_from_templates(
                _selected_template_strings_from_value(native_tpl, tools)
            ),
            # gate healing and execution on NATIVE: "default" can advertise tools dropped by "tool_use" (#7066).
            neutralize_tool_descriptions(
                tools, None, markup_for_tokenizer(render_tokenizer, tools)
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
    continue_final_message: bool = False,
    apply_fn = None,
    hf_token: Optional[str] = None,
    return_metadata: bool = False,
):
    """Return ``formatted_prompt``, swapping in a native-template render when an override template
    dropped the ``tools`` schema. If ``tools`` were requested but the live render is identical
    with and without them (detected by comparison, robust against tool names in the system
    prompt), re-render with the model's native template. Shared by the transformers and MLX
    backends so both advertise tools consistently. ``hf_token`` is forwarded so a gated/private
    model's native template can still be fetched. With ``return_metadata``, returns the selected
    prompt plus reasoning-channel markers for the exact template used by this request."""
    live_markers = detect_reasoning_channel_markers(tokenizer, tools = tools)

    def _result(
        prompt: str,
        markers = live_markers,
        advertised = None,
    ):
        if return_metadata:
            if advertised is None and tools:
                advertised = neutralize_tool_descriptions(
                    tools, None, markup_for_tokenizer(tokenizer, tools)
                )
            return ChatTemplateRenderResult(prompt, markers, advertised)
        return prompt

    if not tools:
        markers = live_markers
        if markers is None:
            markers = detect_reasoning_channel_markers_from_model_info(
                tokenizer, model_info, tools = None
            )
        return _result(formatted_prompt, markers)
    if apply_fn is None:
        apply_fn = apply_chat_template_for_generation
    try:
        probe_no_tools = apply_fn(
            tokenizer,
            messages,
            tools = None,
            enable_thinking = enable_thinking,
            reasoning_effort = reasoning_effort,
            preserve_thinking = preserve_thinking,
            continue_final_message = continue_final_message,
        )
    except Exception as exc:
        logger.warning(
            "No-tools probe failed for '%s'; keeping the existing tools prompt: %s",
            active_model_name,
            exc,
        )
        return _result(formatted_prompt)
    if formatted_prompt != probe_no_tools:
        return _result(formatted_prompt)
    native_prompt = render_native_template(
        model_info = model_info,
        active_model_name = active_model_name,
        messages = messages,
        tools = tools,
        enable_thinking = enable_thinking,
        reasoning_effort = reasoning_effort,
        preserve_thinking = preserve_thinking,
        continue_final_message = continue_final_message,
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
