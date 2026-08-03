# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Dependency-light wrapper around tokenizer.apply_chat_template with a kwarg
fallback for templates that reject reasoning/tools args, plus the shared
native-chat-template fallback used by the transformers and MLX backends.
"""

import copy
import functools
import json
import logging
import re
import weakref
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# "Not JSON we can walk", distinct from a payload that legitimately decodes to None.
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

# Control markup must not reach the prompt as raw text from a user / system / tool turn:
# "</think>" ends the reasoning block early, and a tool result holding
# "<|start|>assistant<|channel|>final<|message|>" forges an assistant turn (#7066). One
# lookahead over the four shapes templates emit (<|name|>/<|name>, <name>/</name>, <name|>,
# [NAME]/[/NAME]), so one sub() breaks any marker with a space after the opener. The name
# list is closed on purpose: bare words match only in the pipe shape, brackets only in the
# exact uppercase spelling, so "<div>", "List<String>", "[1]" and "[inst]" stay as typed.
# [THINK] is absent: no template emits it, the output parsers consume it. DeepSeek alone
# delimits with the fullwidth bar U+FF5C, so it gets its own branch (\uXXXX escapes keep
# this file ASCII) and is the one exception to the closed list, matching ANY Latin / U+2581
# name between fullwidth bars because DeepSeek keeps adding spellings. "<" then a fullwidth
# bar is not prose, so the shape carries it; a CJK author's fullwidth-bar placeholder around
# a Latin key is rewritten, but the charset restriction keeps it off real CJK content.
_CONTROL_MARKUP = re.compile(
    r"<(?="
    # "/?" after the bar: Phi-4 Mini closes with "<|/tool|>" / "<|/tool_call|>"
    # (ollama_template_mappers.py:1023, 1029) rather than a separate closing name, so an
    # MCP description carrying one closed the catalog and rose to system level (#7066).
    r"\|/?(?:(?:start|end)_(?:header_id|of_role)|tool(?:_call|_response)?"
    # Kimi K2 / Moonshot wrap history in a section and each call in a begin/end pair
    # (tool_call_parser.py:20, 55-56, 86-89); none is the short "tool_call" spelling,
    # so a paste could fabricate a historical call (#7066).
    r"|tool_calls?(?:_section)?_(?:begin|end)|tool_call_argument_begin"
    r"|end(?:_of_(?:turn|text))?"
    # Document boundaries: begin_of_text is Llama-3.1 / Llama-4's BOS, endoftext the
    # GPT-2-lineage EOS Qwen2.5, Qwen3, Phi, gpt-oss and GLM-4.5 all still carry. Reserved
    # vocabulary, same argument as the media placeholders below: the trie splits a pasted
    # copy back out to the real token id, so client text lands mid-conversation as a
    # document break the template never opened (#7066).
    r"|begin_of_text|endoftext"
    # header_start / header_end / <|eot|> are Llama-4's spelling of Llama-3's
    # start_header_id / end_header_id / eot_id, and im_sep is Phi-4's role separator.
    # im_system / im_middle are Kimi K2's, alongside the ChatML three.
    r"|header_(?:start|end)|im_(?:start|end|sep|system|middle|user|assistant)"
    # DeepSeek-V4-Flash spells its role boundaries with ASCII bars and a capital, unlike R1's
    # fullwidth ones; this pattern is case-sensitive, so the lowercase names below miss them.
    r"|User|Assistant|System"
    r"|assistant|constrain|channel|message|eo[tm](?:_id)?|final"
    # TML Inkling's call envelope, "<|message_model|>NAME<|content_invoke_tool_json|>{...}
    # <|end_message|>". Longer than the "message" / "end" names above, so all three passed
    # through even though the repo parses them as a tool call (tool_call_parser.py:58,
    # tool_healing.py:129-132, 701-707).
    r"|message_model|content_invoke_tool_json|end_message"
    # Command-R / Aya spell every delimiter in caps: <|START_OF_TURN_TOKEN|> etc.
    r"|(?:START|END)_OF_TURN_TOKEN|(?:USER|SYSTEM|CHATBOT)_TOKEN"
    # Gemma-4's media placeholders (its image_token / audio_token / video_token, also
    # chat_templates.py:917-921) and Llama-3.1's built-in-tool sentinel
    # (chat_templates.py:496). Reserved vocabulary, so a pasted copy is not cosmetic: a
    # processor counts "<|image|>" against the media it was handed, and one extra is a hard
    # ValueError out of MllamaProcessor / Gemma4Processor on the very vision and audio
    # renders neutralized above (#7066). The optional close covers <|image> / <|audio> too.
    r"|image|audio|video|python_tag"
    # Qwen 2.5 Coder builds its fill-in-the-middle prompt from these three special
    # tokens (ollama_template_mappers.py:881) while interpolating chat .Content at
    # :908-909, the pipe-token equivalent of Codestral's [PREFIX]/[MIDDLE]/[SUFFIX].
    r"|fim_prefix|fim_suffix|fim_middle"
    # Qwen2-VL / Qwen2.5-VL reserve these for the processor, which expands a pad token
    # per image or video patch (mapper.py:679-697). A pasted one is counted as media
    # with no image behind it, so embeddings bind at the wrong prompt position.
    r"|vision_start|vision_end|vision_pad|image_pad|video_pad"
    r"|return|system|start|think|turn|user|call|\")\|?>"
    # The parser also recognises the space and backslash-escaped spellings of these openers
    # (tool_call_parser.py:47-53, 62-66), so the class admits both. The name must still start
    # with a letter, keeping "<\uff5c \uff5c>" and other fullwidth-punctuation pairs out.
    r"|\uff5c[A-Za-z][A-Za-z\u2581_ \\]{0,39}\uff5c>"
    # "tools" is the Qwen / Hermes catalog block. The template interpolates the system
    # message INTO the same turn that holds "<tools>...</tools>", so a "</tools>" in client
    # text closes the real catalog and the rest reads as undeclared tools (#7066).
    # Gemma 3 / 3n spell their media placeholders as bare tags rather than the Gemma-4 pipe
    # shape (chat_templates.py:677, 845-847), for the same reason those are here.
    # GLM 4.5-4.7 and Qwen3.5 nest their call protocol inside the outer tool tag:
    # "<tool_call>NAME<arg_key>k</arg_key><arg_value>v</arg_value>" and
    # "<function=name><parameter=k>v</parameter></function>". tool_call_parser.py treats all
    # of them as structural, so a replayed value or tool result can close the current value
    # and inject another key or call (#7066).
    # Gemma spells its BOS / EOS as bare tags, both in its added-token trie
    # (google/gemma-3-4b-it): the bare-tag half of the begin_of_text / endoftext pair above.
    # "</s>" / "<s>" are the Llama-2 / Mistral / Zephyr EOS / BOS (their own
    # tokenizer_config.json), same trie argument. This is the one addition that collides with
    # a real HTML tag, the strikethrough "<s>", accepted for the reason "<think>" and
    # "<tools>" are: a live document boundary beats a space in a rare tag.
    r"|/?(?:(?:start|end)_of_turn|tool_(?:call|response)|tools|think|eos|bos|s|sop"
    r"|start_of_image|image_soft_token|audio_soft_token"
    r"|arg_key|arg_value|function|parameter|param)>"
    # The opening halves carry an "=value", so they need their own anchor. The parser accepts
    # both opener spellings (tool_call_parser.py:_TOOL_CLOSED_PATS), so the attribute form
    # needs its own alternative rather than riding on "function=".
    r"|(?:function|parameter)=|(?:function|param(?:eter)?)\s+name=\""
    r"|(?:tool(?:_call|_response)?|channel|turn)\|>"
    r")"
    # "[ARGS]", Mistral v11's "[CALL_ID]" and Devstral's "[TOOL_CONTENT]" are absent on
    # purpose: all three are metadata WITHIN a block, never its opener, and the openers
    # ("[TOOL_CALLS]", "[TOOL_RESULTS]") are broken, so none can start or close anything
    # alone; inbound they are read by tool_healing.py:36-56 out of model output, not a
    # prompt. "[ARGS]" also collides with real text: it is the standard CLI-synopsis
    # metavariable ("usage: tool [OPTIONS] [ARGS]"), a live string in this repo, and inside a
    # schema "enum" / "pattern" the rewrite makes it a grammar literal the model must emit.
    # Codestral's Modelfile declares [PREFIX]/[MIDDLE]/[SUFFIX] as stop tokens and builds
    # its fill-in-the-middle prompt out of them, while the chat branch of the same template
    # interpolates .Content between [INST] and [/INST]
    # (ollama_template_mappers.py:266-286), so pasted text spelling one asks for FIM
    # semantics instead of staying ordinary content (#7066).
    r"|\[(?=/?(?:INST|SYSTEM_PROMPT|AVAILABLE_TOOLS|TOOL_RESULTS|TOOL_CALLS"
    r"|PREFIX|MIDDLE|SUFFIX|gMASK)\])"
    # Llama-2 opens its system block with "<<SYS>>" INSIDE the first [INST], so the doubled
    # angle is the opener, not a single "<". Both meta-llama/Llama-2-*-chat-hf's template and
    # the "llama" entry MODEL_TO_TEMPLATE_MAPPER installs at generate time emit it, and a
    # later user turn carries no system block, so a pasted pair invents one (#7066). Anchored
    # on the second "<", so "<SYS>>", "cout << SYS" and a heredoc "<<-'SYS'" stay as typed.
    r"|(?<=<)<(?=/?SYS>>)"
)

# Turn-boundary subset for replayed ASSISTANT content: that text is client-controlled too,
# so a raw boundary in it truncates or forges a turn (#7066). Everything else stays
# byte-identical, because the assistant's own think / channel / tool markup is structure the
# template re-renders. Boundaries also cover the bare Zephyr / Phi-3 role sentinels,
# Granite's <|start_of_role|> ... <|end_of_text|>, the Llama-4 and Command-R turn tokens,
# DeepSeek's fullwidth role markers and the Mistral / Llama-2 [INST] / [SYSTEM_PROMPT] pairs.
# DeepSeek's fullwidth TOOL markers stay out for the same reason <|tool_call> does: they are
# the assistant's own.
_TURN_BOUNDARY_MARKUP = re.compile(
    r"<(?="
    r"\|/?(?:(?:start|end)_(?:header_id|of_role)"
    # Kimi spells a turn "<|im_user|>user<|im_middle|>...<|im_end|>", so the role
    # sentinels are boundaries exactly as im_system and im_middle already are.
    r"|im_(?:start|end|sep|system|middle|user|assistant)"
    r"|User|Assistant|System"
    r"|end(?:_of_(?:turn|text))?|eo[tm](?:_id)?|header_(?:start|end)"
    # A document boundary is never the assistant's own structure, so unlike think / channel /
    # tool markup these belong in the replay subset too.
    r"|begin_of_text|endoftext"
    r"|(?:START|END)_OF_TURN_TOKEN|(?:USER|SYSTEM|CHATBOT)_TOKEN"
    # A media placeholder is reserved vocabulary, not reasoning or tool structure: a replayed
    # one is media the processor was handed none of, failing the Gemma / mllama count check.
    # Never legitimate in a replay, unlike think / channel / tool markup.
    r"|image|audio|video|python_tag"
    # Qwen 2.5 Coder builds its fill-in-the-middle prompt from these three special
    # tokens (ollama_template_mappers.py:881) while interpolating chat .Content at
    # :908-909, the pipe-token equivalent of Codestral's [PREFIX]/[MIDDLE]/[SUFFIX].
    r"|fim_prefix|fim_suffix|fim_middle"
    # Qwen2-VL / Qwen2.5-VL reserve these for the processor, which expands a pad token
    # per image or video patch (mapper.py:679-697). A pasted one is counted as media
    # with no image behind it, so embeddings bind at the wrong prompt position.
    r"|vision_start|vision_end|vision_pad|image_pad|video_pad"
    # A tool RESULT is the tool role's structure and a tool CATALOG is the system's, so a
    # replay carrying either fabricates trusted context. The tool CALL spellings stay out:
    # those the assistant does emit. "tool" alone is Phi-4 Mini's catalog wrapper around
    # .Tools (ollama_template_mappers.py:1022-1029), not its call syntax.
    r"|tool_response|tool"
    r"|assistant|return|system|start|turn|user|call)\|?>"
    r"|\uff5c(?:User|Assistant|(?:begin|end)\u2581of\u2581sentence)\uff5c>"
    # "/?" as in the control pattern: Gemma's delimiters are bare tags, so a replayed
    # "</start_of_turn>" is as much a boundary as "<start_of_turn>".
    # "tools" is the Qwen catalog block around the system turn (chat_templates.py:556-568).
    r"|/?(?:(?:start|end)_of_turn|eos|bos|s|sop|tool_response|tools"
    r"|start_of_image|image_soft_token|audio_soft_token)>"
    r"|(?:turn|tool_response)\|>"
    r")"
    # Same split in the bracket family: Mistral renders assistant .Content verbatim
    # (ollama_template_mappers.py:125-127) and spells a tool observation
    # "[TOOL_RESULTS]...[/TOOL_RESULTS]" (:133) and the catalog "[AVAILABLE_TOOLS]" (:123),
    # so a replay can forge either. "[TOOL_CALLS]" is out: that one the assistant emits (:129).
    # The FIM tokens are stop tokens, so a real generation halts instead of emitting one:
    # a replay carrying one did not come from the model (ollama_template_mappers.py:284-286).
    r"|\[(?=/?(?:INST|SYSTEM_PROMPT|AVAILABLE_TOOLS|TOOL_RESULTS"
    r"|PREFIX|MIDDLE|SUFFIX|gMASK)\])"
    # Llama-2's system section is a boundary for the same reason [SYSTEM_PROMPT] is: the
    # template only emits it in the first user turn, never an assistant one.
    r"|(?<=<)<(?=/?SYS>>)"
)


# TTS is not a chat template: the codec prompt is concatenated, so the text sits between
# codec delimiters instead of template ones (inference.py:1918, 1948-1954,
# llama_cpp.py:_TTS_PROMPTS). A pasted closer ends the text segment early or opens the audio
# / global-token segment, yielding truncated or garbled audio (#7066). Per codec, and
# deliberately NOT the chat sweep: this text is meant to be SPOKEN, so "please say
# <s>hello</s>" or "read [INST] literally" must reach the tokenizer as typed. Only the
# active codec's own delimiters and stop tokens count as structure.
_TTS_MARKUP_BY_CODEC = {
    # <custom_token_3>{text}<|eot_id|><custom_token_4>, stop <custom_token_2>. Those three
    # only, not any number: the transformers path spells the same ones as bare ids
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
    # _generate_csm interpolates into "[speaker_id]text" (inference.py:1943-1948) and the
    # processor tokenizes that flat string directly, so a "[1]" ANYWHERE in the text reads
    # as a second speaker turn, not just a leading one. "<|AUDIO|>" and "<|audio_eos|>" are
    # the codec's own tokenizer tokens, the pair CSM is detected by (model_config.py:
    # 992-995): a pasted opener is counted as audio with none behind it and the EOS ends the
    # spoken text early. The processor is called with add_special_tokens = True, so the
    # document boundaries it adds are forgeable from the text as well (#7066).
    "csm": re.compile(r"\[(?=\d+\])|<(?=\|(?:AUDIO|audio_eos|begin_of_text|end_of_text)\|>)"),
}
# An unrecognised codec gets the union: still far narrower than the chat sweep, but it does
# not assume a prompt shape this module has not seen.
_TTS_MARKUP_DEFAULT = re.compile(
    "|".join(f"(?:{pattern.pattern})" for pattern in _TTS_MARKUP_BY_CODEC.values())
)


# A delimiter-shaped token: "<...>" or "[...]" with no whitespace inside, which is the
# only shape any template in this module uses as structure. Ordinary added tokens such as a
# plain word or a "\u2581" piece are not delimiters and are left alone.
_DELIMITER_SHAPED = re.compile(r"\A(?:<[^\s<>]{1,60}>|\[[^\s\[\]]{1,40}\])\Z")
# The same shapes, for harvesting literals a template writes out that are not vocab entries.
# The bracket half deliberately excludes quotes and bare digits: a Jinja template indexes
# with "message['content']" and "messages[0]", which are implementation syntax that never
# reaches the prompt, and harvesting them would rewrite ordinary code containing
# "['content']" -- the exact cross-family mangling this profiling removes (#7066).
#
# The third arm is the attribute form "<function name=\"NAME\">" that MiniCPM-5 and
# MiniMax-M2 use and that ``tool_call_parser`` parses as a live call opener. It carries a
# space and quotes, so the first arm can never see it, and a profile built without it kept
# only the closing "</function>" -- leaving client text free to open a tool-call envelope
# on exactly the models that honour it (#7066).
_TEMPLATE_DELIMITERS = re.compile(
    '<[A-Za-z_][A-Za-z0-9_.\\-]{0,38}\\s+[A-Za-z_][A-Za-z0-9_.\\-]{0,38}="[^"<>]{0,60}">'
    # Llama-2 opens its system block with the DOUBLED angle "<<SYS>>", so this arm has to
    # come before the single-angle one: that would otherwise match the inner "<SYS>", and a
    # profile holding a marker the curated pattern does not recognize is dropped by the
    # structure gate, leaving "<<SYS>>" unbroken (#7066).
    "|<</?[A-Za-z_][A-Za-z0-9_.\\-]{0,38}>>"
    "|<[^\\s<>'\"]{1,60}>"
    "|\\[/?[A-Za-z_][A-Za-z0-9_.\\-]{0,38}\\]"
)
# A Jinja variable index, "loop_messages[i]" or "x[j]", binds directly to the name in front
# of it. A delimiter a template writes out never does: "[INST]" follows a space, a quote, a
# newline or "}". Gating on the preceding character keeps the shipped Gemma-4 loop indexes
# out of the profile without also dropping the Mistral delimiters (#7066).
_JINJA_INDEX = re.compile(r"[\w\]\)]\Z")
# "{# ... #}" never reaches the prompt. The shipped gptoss template mentions "<|final|>"
# only in a comment (chat_templates.py:1311) while the live protocol emits
# "<|channel|>final<|message|>", so harvesting comment text rewrote ordinary user and tool
# text containing "<|final|>" (#7066).
_JINJA_COMMENT = re.compile(r"\{#.*?#\}", re.S)
# Bracket names the curated pattern deliberately leaves alone: metadata WITHIN a block,
# never its opener, so with the openers broken none can start or close anything by itself.
# "[ARGS]" also collides with real text, being the standard CLI-synopsis metavariable
# ("usage: tool [OPTIONS] [ARGS]"). Harvesting them from a Mistral-style template would
# reintroduce exactly the false rewrite the curated list documents avoiding (#7066).
_BLOCK_METADATA = frozenset({"[ARGS]", "[CALL_ID]", "[TOOL_CONTENT]"})
# A template that builds its role sentinel by concatenation, "'<|' + message['role'] + '|>'"
# (Phi-3, chat_templates.py:383), never writes "<|system|>" out as a literal, so harvesting
# literals alone leaves it out of a profile that is otherwise non-empty -- and a non-empty
# profile is what disables the curated fallback. Client text could then carry a trusted role
# marker into a user turn (#7066).
_DYNAMIC_PIPE_ROLE = re.compile(r"""['"]<\|['"]\s*\+|\+\s*['"]\|>['"]""")
# The roles a chat template can interpolate. Closed on purpose: this adds exactly what the
# construction can emit, rather than re-enabling a match on any "<|word|>".
# A template can build the equals-form opener from fragments too, "{{ '<function=' +
# call.name + '>' }}", so no complete literal ever appears. Harvesting closers alone left a
# NON-EMPTY profile -- which disables the curated fallback -- while "<function=pay>" stayed
# byte-exact, and tool_call_parser treats that as a live call envelope (#7066).
_CONCATENATED_OPENER = re.compile(r"""['"]<(/?[A-Za-z_][A-Za-z0-9_.\-]{0,30})=['"]\s*\+""")


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
# A marker whose name is filled in at render time, so a template only ever shows one
# example. "<function=pay>" must break on a model whose template spells
# "<function=example>", which an alternation over literals alone cannot do.
_DYNAMIC_OPENER = re.compile(r"\A<(/?[A-Za-z_][A-Za-z0-9_.\-]{0,30})=[^\s<>]*>\Z")
# The attribute spelling of the same thing: the value is the render-time name.
_DYNAMIC_ATTR_OPENER = re.compile(
    r"\A<([A-Za-z_][A-Za-z0-9_.\-]{0,38})\s+([A-Za-z_][A-Za-z0-9_.\-]{0,38})=\"[^\"<>]{0,60}\">\Z"
)


# DeepSeek spells one marker several ways: "<\uff5ctool\u2581calls\u2581begin\uff5c>" in the
# vocabulary, but tool_call_parser.py:47-53 also accepts the space and backslash-escaped
# spellings. The curated fullwidth arm matched any name between the bars, so all three broke;
# an exact literal from the profile breaks only the one the vocabulary happens to hold, and a
# pasted alias opens a tool-call envelope (#7066).
_FULLWIDTH_MARKER = re.compile("\\A<\uff5c([A-Za-z][A-Za-z\u2581_ \\\\]{0,39})\uff5c>\\Z")
_ALIAS_SEPARATORS = "(?:\u2581|\\\\?_| )"


@functools.lru_cache(maxsize = 1)
def _deepseek_opener_pattern():
    """The tool-call-parser's own DeepSeek opener alternation, or None if unavailable.

    Single source of truth: tool_call_parser keeps the five spellings llama.cpp accepts,
    and a profile that breaks only the one spelling a vocabulary happens to hold leaves the
    other four live (#7066)."""
    try:
        from core.inference.tool_call_parser import (
            _DEEPSEEK_OPEN_RE_SRC,
            TOOL_XML_SIGNALS,
        )
    except Exception:  # pragma: no cover - parser unavailable
        return None
    # The outer-block alternation plus every fullwidth signal the parser flips on, which is
    # where the per-call "<\uff5ctool\u2581call\u2581begin\uff5c>" lives.
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
        # A DeepSeek opener has aliases the parser accepts that are not separator
        # respellings at all: it treats the short "<\uff5ctool\u2581calls\uff5c>" and the
        # singular "<\uff5ctool\u2581call\u2581begin\uff5c>" as the same live opener. Reuse the
        # parser's own alternation rather than restating it, so the two cannot drift and a
        # profiled DeepSeek prompt cannot be handed an envelope the parser will honour but
        # the profile never learned to break (#7066).
        deepseek = _deepseek_opener_pattern()
        if deepseek is not None and re.fullmatch(deepseek, marker):
            return deepseek
        # Otherwise every separator position accepts the three spellings the parser accepts.
        name = fullwidth.group(1)
        parts = re.split("[\u2581_ ]", name)
        if len(parts) > 1:
            return "<\uff5c" + _ALIAS_SEPARATORS.join(re.escape(p) for p in parts) + "\uff5c>"
    dynamic = _DYNAMIC_OPENER.match(marker)
    if dynamic:
        return "<" + re.escape(dynamic.group(1)) + "=[^\\s<>]*>"
    attr = _DYNAMIC_ATTR_OPENER.match(marker)
    if attr:
        # Whitespace stays loose: a template may render one space where a client sends
        # several, and both open the same envelope.
        return "<" + re.escape(attr.group(1)) + "\\s+" + re.escape(attr.group(2)) + '="[^"<>]*">'
    return re.escape(marker)


def _template_strings(chat_template) -> list:
    """Every template body a tokenizer exposes, whatever shape it uses.

    A tokenizer may carry one string, a dict of named templates, or a list of
    ``{"name", "template"}`` entries (Hermes-3 ships the list form). Profiling only the
    string case would silently drop every literal a named template emits."""
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
    """The markers one model actually treats as structure, and the patterns for them.

    Built from the model's own chat template and token list rather than from the curated
    patterns below, because a vocabulary is authoritative where a hand-written list cannot
    be: it covers a sentinel this module never enumerated, and it leaves alone one that
    belongs to some other family. A Llama-3 checkpoint has no "</think>" in either place,
    so a user pasting a script that contains one keeps their text byte-for-byte (#7066).
    """

    __slots__ = (
        "control",
        "boundary",
        "markers",
        "rewrite_control",
        "rewrite_boundary",
        # Which named template this profile was selected for, so a caller whose catalog
        # emptied during sanitizing can tell its profile is now for the wrong one (#7066).
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
        # Which of the model's own markers open a turn. The curated patterns hold the one
        # thing a vocabulary cannot say: whether the ASSISTANT legitimately emits a marker.
        # A marker this module does not recognise at all is treated as a boundary, since a
        # replayed assistant turn is client text and a forged turn costs more than a spaced
        # one in history it should not have contained.
        boundary = {
            marker
            for marker in markers
            if _TURN_BOUNDARY_MARKUP.search(marker) or not _CONTROL_MARKUP.search(marker)
        }
        self.boundary = _alternation(boundary)
        # Bound once per profile, not per call: a fresh partial each time would be a fresh
        # identity, so a sweep cache keyed on the callable would never hit and would grow
        # one entry per message instead.
        self.rewrite_control = functools.partial(neutralize_control_markup, markup = self)
        self.rewrite_boundary = functools.partial(neutralize_turn_boundary_markup, markup = self)


def _alternation(markers: set):
    """A pattern matching any of *markers*, longest first so no prefix shadows a longer one."""
    if not markers:
        return None
    return re.compile(
        "|".join(_marker_pattern_source(m) for m in sorted(markers, key = len, reverse = True))
    )


# The special-token variables a chat template may emit instead of writing the literal.
# Llama-3.1 opens with "{{ bos_token }}", so the concrete spelling is never in the template
# text and a literal-only scan cannot see it.
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
    """Profile one model's structural markers, or None when nothing is known about it.

    None means "sweep everything the curated patterns know", which is the safe direction
    for a model whose template and vocabulary could not be read.
    """
    markers: set = set()
    for token in tokens or ():
        # A base vocabulary is six figures and almost none of it can be a delimiter, so the
        # cheap first-character test runs before the regex: this is walked once per model.
        if not isinstance(token, str) or not token or token[0] not in "<[":
            continue
        if not _DELIMITER_SHAPED.match(token):
            continue
        if token in _BLOCK_METADATA:
            continue
        # A dedicated vocabulary entry proves only that the string has a token, not that
        # anything treats it as structure. Gemma reserves "<table>", "<caption>", "<tr>"
        # and "<td>", so harvesting the whole delimiter-shaped vocabulary turned an HTML
        # prompt into "< table>< caption>" -- the exact cross-family mangling this
        # profiling exists to remove. The curated pattern is the repo's record of what the
        # renderer and the parsers actually treat as structure, so the vocabulary side is
        # its intersection: "</think>" is kept on granite, whose template never emits it
        # and whose vocabulary marks it special=False, while "<table>" is not (#7066).
        if _CONTROL_MARKUP.search(token):
            markers.add(token)
    known = {token for token in tokens or () if isinstance(token, str)}
    # Only the template this request will render with. A named-template dict carries both
    # "default" and "tool_use", and unioning them made a no-tools turn rewrite "<tools>",
    # which cannot appear in the prompt it is about to send (#7066).
    bodies = _selected_template_strings_from_value(
        chat_template, tools, prefer_tool_use = prefer_tool_use
    )
    for body in bodies or _template_strings(chat_template):
        # Blanked rather than removed, so every offset the index check relies on survives.
        body = _JINJA_COMMENT.sub(lambda m: " " * len(m.group(0)), body)
        for match in _TEMPLATE_DELIMITERS.finditer(body):
            marker = match.group(0)
            if marker.startswith("[") and _JINJA_INDEX.search(body, 0, match.start()):
                continue  # "loop_messages[i]": an index, not something the prompt shows.
            if marker in _BLOCK_METADATA:
                continue
            # A literal a template writes out is structure when the tokenizer has a token
            # for it, or when the curated pattern already knows it. Neither holds for the
            # instructional placeholders Qwen prints inside its tool-use prose,
            # "<function-name>" and "<args-json-object>" (chat_templates.py:561, 716):
            # those are words the model reads, and rewriting them mangles ordinary code and
            # tool descriptions that mention them (#7066).
            if marker in known or _CONTROL_MARKUP.search(marker):
                markers.add(marker)
        # A template that emits "{{ bos_token }}" inserts and tokenizes that value as a
        # document boundary, but its concrete spelling is never in the template text. The
        # vocabulary pass covers it only when the token was harvested, which a partial or
        # stubbed vocabulary does not guarantee, so resolve it from the tokenizer (#7066).
        for name, value in (specials or {}).items():
            if not isinstance(value, str) or not value:
                continue
            if not any(name in code for code in _jinja_code(body)):
                continue
            # Shape, not the curated pattern. _CONTROL_MARKUP is this repo's record of the
            # families it has seen, so gating on it would keep exactly the tokens the
            # vocabulary pass already covers and miss the unknown-family boundary this is
            # for. The tokenizer declaring the value AND the template evaluating it is the
            # stronger proof; the shape test only keeps it from turning a plain word into
            # a marker that would sweep ordinary prose.
            if value in known or _DELIMITER_SHAPED.match(value):
                markers.add(value)
        if _DYNAMIC_PIPE_ROLE.search(body):
            markers.update(f"<|{role}|>" for role in _ROLE_NAMES)
        # "<function=" built by concatenation: record the example spelling the dynamic rule
        # already knows how to generalize, so any render-time name matches.
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
    """Break control markup in free text by spacing out the opener (#7066).

    "</think>" -> "< /think>", "[/INST]" -> "[ /INST]": readable, but no longer a
    delimiter to the template, the think extractor or the stop-sequence matcher. A plain
    space, because every tokenizer vocabulary has one; U+2060 can fall back to byte junk.

    With a *markup* profile only that model's own markers are broken, so text naming some
    other family's sentinel is left exactly as the caller wrote it."""
    if markup is not None:
        return _spaced_out_markers(markup.control, text) if markup.control else text
    return _spaced_out(_CONTROL_MARKUP, text)


def neutralize_turn_boundary_markup(text: str, markup: "ModelMarkup" = None) -> str:
    """Break only the turn-boundary sentinels, for replayed assistant text (#7066)."""
    if markup is not None:
        return _spaced_out_markers(markup.boundary, text) if markup.boundary else text
    return _spaced_out(_TURN_BOUNDARY_MARKUP, text)


def neutralize_tts_prompt_text(text: str, audio_type = None) -> str:
    """Break the active codec's own delimiters in a TTS prompt (#7066).

    Scoped to *audio_type*: this text is spoken, so anything that is not structure in THIS
    codec's prompt has to survive byte-exact."""
    return _spaced_out(_TTS_MARKUP_BY_CODEC.get(audio_type, _TTS_MARKUP_DEFAULT), text)


def _neutralize_leaves(
    value,
    rewrite,
    warn_on_key_collision: bool = False,
):
    """Apply *rewrite* to every string leaf, keys included, of a nested structure.

    Iterative, not recursive: the client picks the depth, and a schema ``json.loads``
    accepts must not exhaust the interpreter stack and turn the request into a 500.
    Containers are rebuilt in reverse breadth-first order, so a child is finished before
    its parent and a repeated or self-referencing node is visited once.

    Rewriting keys is not injective ("a<think>" and "a< think>" both land on "a< think>"),
    so a colliding dict keeps only the last value; *warn_on_key_collision* logs it. The
    merge stands because the alternative, keeping one key raw so both survive, would put
    the markup back in the prompt."""
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


# A media payload stays opaque: it is a URL or base64 blob the processor resolves, not
# prompt text, so rewriting one breaks the fetch rather than the prompt. Gated on the part's
# own type, because "data" and "url" are ordinary content keys on anything else -- a
# "{'type': 'json', 'data': ...}" part is prompt text Llama-3.1 serializes with tojson, and
# exempting it unconditionally put the markup straight back in the prompt. "input_image" is
# here because the MLX image counter recognises it (mlx_inference.py:130) and the registered
# VLM renderer passes those messages through this sweep.
# Llama-3.1 renders both roles below through one tool-result branch (chat_templates.py:517),
# and neither resolves media: the content is serialized whole with tojson.
_TOOL_RESULT_ROLES = frozenset({"tool", "ipython"})
# The roles a template actually compares against. A role differing from one of these only by
# case or padding is canonicalized, so the sweep and the template agree on the turn.
# Gemma-4 maps "assistant" onto "model" and leaves an incoming "model" alone
# (gemma-4.jinja:234), so both name the same replayed turn: they take the boundary subset
# rather than the full sweep, and keep the assistant-only structured fields.
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
    """Sweep the joined *texts* and hand each carrier back its own share, or None.

    Every carrier keeps its own text in its own position: nothing is moved past a
    neighbour, so a caption still sits on the side of the item it describes.

    *contiguous*[i] says whether carrier i+1 directly follows carrier i in the parts list.
    Where it does not -- an image or a JSON part sits between them -- the opener is NOT
    migrated, because moving a character across a media item reorders the text around it
    and a renderer that keeps the media would bind the caption to the wrong side (#7066).
    """
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
    # A break landing at the very start of a carrier is stripped by a renderer that trims
    # each part, which would let the marker re-form. The opener walks forward into the
    # carrier holding the rest of the marker instead, so the break sits inside one carrier.
    # Only the marker's own characters move, and only across the split they already
    # straddle, so no text passes a neighbour (#7066).
    for index in range(len(pieces) - 1):
        if contiguous is not None and not contiguous[index]:
            continue  # a media or JSON part sits here; nothing crosses it.
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

    Two gaps a per-part rewrite misses. A mapping part without a string "text" was passed
    through whole, yet /generate/stream accepts one and Llama-3.1 serializes the entire
    iterable with tojson. And a marker split across two adjacent text parts survived both
    sweeps, because Gemma-4 concatenates them with no separator (gemma-4.jinja:304) and
    reassembles the opener. Whitespace between them is no fix, since the sibling paths trim
    each part (gemma-4.jinja:339), so a run that only becomes a marker once joined is swept
    as one string and collapses into one part. A clean run keeps its parts."""
    parts: list = []
    for part in content:
        if isinstance(part, str):
            parts.append(rewrite(part))
        elif isinstance(part, dict):
            # isinstance first: GenerateRequest.messages is an untyped List[dict], so "type"
            # can be unhashable and the set lookup would 500 the request before rendering.
            part_type = part.get("type")
            if isinstance(part_type, str) and part_type in _MEDIA_PART_TYPES and media_opaque:
                opaque = {k: v for k, v in part.items() if k in _OPAQUE_PART_KEYS}
                swept = _neutralize_leaves(
                    {k: v for k, v in part.items() if k not in _OPAQUE_PART_KEYS}, rewrite
                )
                parts.append({**swept, **opaque} if opaque else swept)
            else:
                # Every field, not just "text": the tojson templates serialize the part
                # whole, so sweeping only "text" left the rest live (#7066).
                parts.append(_neutralize_leaves(part, rewrite))
        else:
            parts.append(part)

    def _text_of(part):
        if isinstance(part, str):
            return part
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            return part["text"]
        return None

    # No part reliably separates the text around it. A renderer emits a placeholder only for
    # the media types it knows and silently skips the rest: gemma-4.jinja:334-347 renders
    # image / image_url / audio / input_audio / video and drops video_url, audio_url and
    # input_image, so a part that looks like a separator can render as nothing at all and
    # leave the fragments adjacent. A tool body aggregates them anyway (gemma-4.jinja:
    # 301-306). Every text carrier is therefore one run, which costs nothing now that a
    # split marker's opener migrates instead of the run collapsing: no text moves past an
    # item it sits beside, so a fragment that a renderer really does separate is unharmed
    # beyond one inserted space (#7066).
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
        # Break the marker inside the carrier holding its opener, so each keeps its own
        # text: Llama-3.1 serializes the list in order with "message.content | tojson"
        # (chat_templates.py:517-523), so moving text between carriers would put a caption
        # on the wrong side of the item it describes (#7066).
        run = [texts[index] for index in carriers]
        # First choice: migrate an opener only between carriers that really are adjacent,
        # so nothing crosses an image or a JSON part sitting between them.
        redistributed = _redistribute_swept(
            run,
            rewrite,
            [carriers[i + 1] == carriers[i] + 1 for i in range(len(carriers) - 1)],
        )
        if redistributed is None:
            # The split straddles a non-text part and a trimming renderer would re-form the
            # marker. Moving the opener across that part is a smaller harm than the collapse
            # below: one marker character changes side, rather than every carrier's text
            # being pulled into the first one (#7066).
            redistributed = _redistribute_swept(run, rewrite)
        if redistributed is None:
            # A last resort only: migrating the opener leaves the break inside a carrier for
            # every split of every marker this module knows, so nothing reaches this today.
            # It stays because collapsing is safe when redistribution somehow is not, and a
            # marker that survives a trimming renderer would be worse than reordered text.
            redistributed = [rewrite(trimmed)] + [""] * (len(carriers) - 1)
        for index, text in zip(carriers, redistributed):
            part = parts[index]
            merged[index] = text if isinstance(part, str) else {**part, "text": text}
    if not merged:
        return parts
    return [merged.get(index, part) for index, part in enumerate(parts)]


def _differs(new, old) -> bool:
    """True when the rewrite changed *old* into *new*.

    The client picks the nesting depth and ``==`` recurses in C, so an overflowing
    comparison must not 500 the request. It counts as changed, keeping the neutralized
    copy: the safe direction (#7066)."""
    try:
        return new != old
    except RecursionError:
        return True


def _neutralize_argument_leaves(value, markup = None):
    """Break control markup in every string leaf (keys included) of *value*."""
    rewrite = neutralize_control_markup if markup is None else markup.rewrite_control
    return _neutralize_leaves(value, rewrite, warn_on_key_collision = True)


def _neutralized_arguments(arguments, markup = None):
    """Neutralize a replayed call's ``arguments``, or None when already clean.

    OpenAI ships ``arguments`` as JSON *text*, and every consumer decodes it back to an
    object AFTER this runs: ``_normalize_tool_call_arguments`` below re-renders through
    ``json.loads`` when a template rejects a string, and llama.cpp does the same in
    ``workaround::func_args_not_string``. So rewriting the raw text lets
    "\\u003ctool_call|\\u003e" through and the decoded marker forges a turn (#7066). Parse
    first, rewrite the decoded leaves, re-serialize; a clean payload stays byte-identical
    so the prefix cache still hits."""
    if isinstance(arguments, str):
        decoded = safe = _UNPARSED
        try:
            decoded = json.loads(arguments)
            safe = _neutralize_argument_leaves(decoded, markup)
        # RecursionError as well as a parse error: json.loads and the walk both blow the
        # stack near 1000 levels, so a valid '[' * 1000 + '0' + ']' * 1000 would 500 a
        # request the server used to forward. Fall through to the text rewrite, which cannot
        # recurse; nothing downstream can decode that payload either, so no marker hides.
        except (ValueError, TypeError, RecursionError):
            decoded = safe = _UNPARSED
        if decoded is not _UNPARSED:
            # _differs, not "!=": comparing two distinct deep structures recurses in C, so
            # a payload that decoded fine could still blow the stack on the comparison and
            # 500 a request that used to forward (#7066).
            if _differs(safe, decoded):
                # ensure_ascii keeps a decoded lone surrogate ("\ud800") as an escape: raw,
                # it makes the outer request unencodable and raises UnicodeEncodeError on a
                # payload that used to forward fine (#7066).
                return json.dumps(safe, ensure_ascii = True)
            # Parsed clean, but the DECODE can hide a marker the template still renders:
            # a duplicate key means json.loads keeps only the last value, so
            # '{"x":"</tool_call><|im_end|>...","x":"safe"}' decodes to {"x": "safe"} while
            # Qwen3 interpolates the raw string verbatim. When the text carries markup the
            # decoded value does not, hand back the canonical re-serialization, which drops
            # the shadowed duplicate exactly as the parser would (#7066).
            rewrite = neutralize_control_markup if markup is None else markup.rewrite_control
            if rewrite(arguments) != arguments:
                return json.dumps(safe, ensure_ascii = True)
            return None
    new_arguments = _neutralize_argument_leaves(arguments, markup)
    # Same guard for arguments that arrived already decoded, which never passed through
    # json.loads and so were never depth-limited by it.
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
    """Map each replayed tool-call id to a swept id that is still unique.

    The sweep is not injective: "call<|end|>" and "call< |end|>" both break to
    "call< |end|>". Gemma resolves a result by comparing ids and lets the last match win
    (gemma-4.jinja:289-294), so two calls sharing one id would attribute both observations
    to the same call. A collision is therefore given a numeric suffix, which is checked
    against the ids that stay as they are so it cannot land on one of those either."""
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
    # Reserved first: an id the sweep leaves alone keeps its own spelling, so a rewritten
    # one must never be handed that value.
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

    Gemma-4 renders "<|tool_call>call:NAME{key:<|"|>value<|"|>}<tool_call|>", so a name or
    argument echoing pasted text can close the call block and open a "<|tool_response>" or
    "<|turn>model" of its own (#7066). The rewrite is the identity on every dispatchable
    name (Studio composes ^[a-zA-Z0-9_-]{1,64}$), and a tool result's "name" takes the same
    rewrite, so the two still agree when Gemma-4 pairs them by name.

    Both replay shapes are swept, the OpenAI nested one and the flat {"id", "name",
    "arguments"} one, rather than only whichever a particular guard would pick. Harmony /
    gpt-oss, Qwen 2.5 / 3, Granite-4 and Llama-4 select with "{%- if tool_call.function %}"
    (chat_templates.py:771-780), which is a truthiness test, so an empty nested object sends
    them to the flat fields; and a flat-shaped template reads "name" off the call whatever
    the nested object holds. Sweeping both removes the need to guess which one renders."""

    def _field_updates(source: dict) -> dict:
        updates: dict = {}
        name = source.get("name")
        if isinstance(name, str) and name:
            new_name = neutralize_control_markup(name, markup)
            if new_name != name:
                updates["name"] = new_name
        # Harmony concatenates "content_type" straight before "<|message|>"
        # (chat_templates.py:1332-1334), so a replayed "json<|message|><|end|><|start|>"
        # closes the commentary call and opens an assistant channel (#7066).
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
        # Kimi interpolates the id straight between "<|tool_call_begin|>" and
        # "<|tool_call_argument_begin|>", so an id carrying the closer ends the envelope the
        # template opened and injects structure after it, with the name and arguments clean
        # (#7066). Rewritten with the same function as a tool result's "tool_call_id", so a
        # replayed pair still matches on both sides.
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
            # No "function" object is invented for a call that never had one.
            merged["function"] = {**function, **nested_updates}
        out.append(merged)
    return out


def sweep_cache() -> dict:
    """A cache for a caller that sweeps the same history more than once.

    The agentic tool loop re-sweeps the whole conversation on every iteration, because a
    tool result lands in it as the loop goes and a forged turn in one would render for
    real. Every earlier turn is then swept again with the identical text, which is pure
    repeated work: the rewrite is a function of the string alone.

    The cache is handed in by the caller rather than kept here on purpose. It lives as long
    as the request that owns it and is dropped with it, so message text is never retained
    past the call in module state, and it needs no size bound because it can only ever hold
    text the caller is already holding.
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
    """Neutralize control markup in message content and tool-result names (#7066).

    User / system / tool turns lose every marker; assistant turns lose only turn boundaries
    and keep the think / channel / tool markup replayed history legitimately holds. Returns
    the same list object when nothing changed, so the prompt stays byte-for-byte what it
    was.

    Pass a ``sweep_cache()`` when sweeping the same growing history repeatedly; results are
    identical either way, since it only memoizes a pure rewrite."""
    if not messages:
        return messages
    changed = False
    out: list = []
    id_map = _injective_id_map(messages, markup)
    for msg in messages:
        if not isinstance(msg, dict):
            out.append(msg)
            continue
        # isinstance, not truthiness: GenerateRequest.messages is an untyped List[dict], so a
        # role can be an int and ".strip()" on one 500'd the stream before rendering.
        # A non-string is simply not "assistant", so it takes the full rewrite.
        raw_role_value = msg.get("role")
        role = raw_role_value.strip().lower() if isinstance(raw_role_value, str) else ""
        assistant = role in _ASSISTANT_ROLES
        if markup is None:
            rewrite = neutralize_turn_boundary_markup if assistant else neutralize_control_markup
        else:
            rewrite = markup.rewrite_boundary if assistant else markup.rewrite_control
        if cache is not None:
            # Keyed by the bound rewrite, so two models in one process cannot share an entry.
            rewrite = _memoized(rewrite, cache)
        updates: dict = {}
        dropped_keys: set = set()
        # The role is rendered, not just dispatched on: Llama-3.1 concatenates it between
        # "<|start_header_id|>" and "<|end_header_id|>", and /generate/stream takes untyped
        # dicts, so a role of "user<|end_header_id|><|eot_id|>..." forged an assistant turn
        # even with the content swept (#7066). Neutralized rather than rejected, so a client
        # using a role this code does not know still works.
        raw_role = msg.get("role")
        if isinstance(raw_role, str) and raw_role:
            new_role = neutralize_control_markup(raw_role, markup)
            # "Assistant" and " assistant " mean assistant here but not to a template, which
            # compares case-sensitively. That gap let a padded spelling take the lenient
            # assistant treatment while still rendering as one, so a known role is
            # canonicalized and the two agree again (#7066).
            if role in _SCHEMA_ROLES and new_role != role:
                new_role = role
            # Phi-3 wraps an unrecognised role as "<|" + role + "|>"
            # (chat_templates.py:382-383), so a role of "end" spells that template's own
            # turn terminator while carrying no markup of its own and passing the sweep
            # untouched. A canonical role is MEANT to render that way, so only an unknown
            # one is checked, and it falls back to the least trusted role rather than being
            # padded: a template that trims the role would undo a space (#7066).
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
        # Gemma-4 falls back to a tool result's "name" when "tool_call_id" matches no
        # call, concatenating it into the "<|tool_response>" block (#7066).
        # Same rewrite as the call's "id" above, so a replayed pair still matches.
        result_id = msg.get("tool_call_id")
        if isinstance(result_id, str) and result_id:
            new_result_id = id_map.get(result_id, result_id)
            if new_result_id != result_id:
                updates["tool_call_id"] = new_result_id
        name = msg.get("name")
        if role == "tool" and isinstance(name, str) and name:
            new_name = neutralize_control_markup(name, markup)
            if new_name != name:
                updates["name"] = new_name
        # A separate reasoning field is the INNER text of a thought block the template wraps
        # itself: Qwen in "<think>...</think>" (chat_templates.py:759), Gemma-4 in
        # "<|channel>thought ... <channel|>" (gemma-4.jinja:245), Harmony in
        # "<|channel|>analysis<|message|> ... <|end|>" (chat_templates.py:1330). None is
        # "content", so it reached the prompt unswept, and an embedded closer exits the
        # thought and exposes the rest as answer text (#7066). Hence the FULL rewrite: unlike
        # replayed "content", which legitimately carries the assistant's own think tags, this
        # field must never contain its enclosing delimiters.
        for field in ("reasoning", "reasoning_content", "thinking"):
            value = msg.get(field)
            if isinstance(value, str) and value:
                new_value = neutralize_control_markup(value, markup)
                if new_value != value:
                    updates[field] = new_value
        # Gemma-4's legacy assistant-level "tool_responses": format_tool_response_block
        # renders the name and every leaf, so markup there closes "<|tool_response>" and
        # opens a model turn. Tool output, so the full rewrite.
        tool_responses = msg.get("tool_responses")
        # Gemma-4 reads tool_responses independently of the role (gemma-4.jinja:232-279) and
        # supplies the "<|tool_response>" wrapper itself, so a user or system message
        # carrying one fabricates a trusted observation with no marker for the sweep to
        # catch. Assistant-only, exactly like tool_calls (#7066).
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
                # Llama-3.1 serializes mapping content with tojson (chat_templates.py:127-128)
                # and /generate/stream takes raw dicts, so object values reach the prompt as
                # live structure too (#7066).
                new_content = _neutralize_leaves(content, rewrite)
            elif isinstance(content, list):
                # A media part is only opaque where something RESOLVES it, and nothing does
                # inside a tool result: Studio's vision and audio paths build from the last
                # user message, while Llama-3.1's tool branch serializes the whole content
                # iterable with tojson, so an exempt URL there lands in the prompt as live
                # structure. That branch keys on "tool" OR "ipython" (chat_templates.py:517),
                # so both roles count (#7066).
                is_tool_result = role in _TOOL_RESULT_ROLES
                new_content = _neutralize_content_parts(content, rewrite, not is_tool_result)
            if _differs(new_content, content):
                updates["content"] = new_content
        tool_calls = msg.get("tool_calls")
        # Llama-3.1 branches on "'tool_calls' in message" BEFORE the role
        # (chat_templates.py:487-489) and emits an assistant tool-call turn, so the field on a
        # user or tool message fabricates assistant history however clean its text. It is
        # assistant-only in the OpenAI schema too, so any other role drops it (#7066).
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

    Every string in a declaration is prompt text: Gemma-4 interpolates the description into
    its system turn and emits property keys / ``enum`` / ``required`` entries inline, while
    Granite and Mistral-Small-3 render the whole entry with ``tojson``, and ``mcp_client``
    copies a remote ``description`` / ``inputSchema`` verbatim. So markup anywhere in the
    schema closes the system turn and forges a model one (#7066). The rewrite covers the
    whole entry, not just the nested ``function``, because ``ChatCompletionRequest.tools``
    is a bare ``list[dict]``.

    ``function.name`` is the dispatch identity: rewriting it silently breaks dispatch,
    leaving it exact forges a turn (Gemma-4 emits ``call:NAME`` unquoted), so a name
    carrying markup drops the tool with a warning instead. The predicate is the rewrite
    itself, not OpenAI's name grammar, so a passthrough client's ``ns.tool`` or
    ``functions.NAME:IDX`` still ships; it is the identity on markup-free strings, so a live
    catalog is returned unchanged."""
    # The agentic loop re-sanitizes the catalog on every iteration, because a one-shot
    # tool can retire between turns, so an unchanged catalog was swept again from scratch.
    # Keyed on the serialized catalog rather than the list itself: a value snapshot cannot
    # go stale if a caller mutates its own list in place, and building one is still five
    # times cheaper than the sweep it skips (#7066).
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
        # Both spellings, not just the selected one: an entry may carry an empty
        # "function" mapping alongside a flat "name", and picking the nested level on
        # "isinstance" alone skipped the identity that actually dispatches. The flat name
        # was then rewritten in place, so the model saw a name execute_tool no longer
        # answers to (#7066).
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
        # Both levels: OpenAI nests the schema under "function", MCP carries
        # "input_schema" on the entry itself.
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


# The machine-valued rather than descriptive positions in a JSON Schema: a property name, an
# enum or const literal, a required entry are all part of the contract the model must satisfy
# and the controller forwards verbatim to execute_tool. Rewriting one makes the model emit
# the rewritten spelling while the MCP server still expects the original, so the tool breaks.
# function.name is already dropped for this reason; these get the same treatment (#7066).
_SCHEMA_KEYED_IDENTIFIERS = frozenset(
    {
        "properties",
        "patternProperties",
        "$defs",
        "definitions",
        # Both dependent* keywords are keyed BY a property name, and dependentRequired's
        # values are property-name lists too, so it is checked on both sides below.
        # "dependencies" is draft-07's spelling of both, so keyed and list-valued as well.
        "dependentSchemas",
        "dependentRequired",
        "dependencies",
        # Keyed by vocabulary URI, so its keys are identifiers like a property name.
        "$vocabulary",
    }
)
_SCHEMA_KEYED_LIST_IDENTIFIERS = frozenset({"dependentRequired", "dependencies"})
# "pattern" and "default" belong here for the same reason: a grammar built from the schema
# forces the model to satisfy the rewritten regex or echo the rewritten default, then the MCP
# server validates the original and rejects the call. The case the "[ARGS]" comment above
# anticipated.
_SCHEMA_VALUED_IDENTIFIERS = frozenset(
    {
        "enum",
        "const",
        "required",
        "pattern",
        "default",
        # Under format assertion (or a custom validator) this is a constraint the MCP
        # server checks, so a rewrite leaves the model targeting a different contract
        # from the one the server enforces, exactly as for "pattern".
        "format",
        # Same contract argument for the content vocabulary: both are machine-valued
        # strings a validator decodes against. "contentSchema" stays out on purpose, since
        # it holds a subschema whose keyword positions the recursive scan already reads.
        "contentEncoding",
        "contentMediaType",
        # An OpenAPI discriminator holds only "propertyName" and a "mapping" whose keys and
        # targets are identifiers, with no prose field to protect, so every leaf under it is
        # machine-valued: the server resolves the original while the model sees the rewrite.
        "discriminator",
        # Same shape: an OpenAPI xml object is "name" / "namespace" / "prefix" plus two
        # booleans, all serialization identifiers and no prose, so a rewrite would advertise
        # element names the server does not produce.
        "xml",
        # A reference is resolved, not read: rewriting "$id", "$anchor" or a "$ref" pointing
        # at them leaves the model and llama-server's grammar on a different schema than the
        # MCP server registered. "$ref" can also name an external URI, which no "$defs" drop
        # would cover.
        "$ref",
        "$id",
        "$anchor",
        # The dialect a validator resolves the whole schema against.
        "$schema",
        "$dynamicRef",
        "$dynamicAnchor",
        # Draft-2019-09 spells the same recursion "$recursiveRef" / "$recursiveAnchor", and
        # draft-04 spells "$id" as a bare "id", so a schema in an older dialect has the same
        # base URI and resolution targets under different names.
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


# Where a real JSON Schema starts in a declaration. The semantic scan anchors here rather
# than on the whole entry, because a declaration also carries vendor extension fields, and a
# "default" or "properties" key inside one of those is ordinary prose to neutralize, not a
# reason to drop the tool.
_SCHEMA_ROOT_KEYS = (
    "parameters",
    "input_schema",
    "inputSchema",
    "outputSchema",
    "output_schema",
    "returns",
    # Gemma-4 emits a response declaration from "function.response"
    # (gemma-4.jinja:115-124), and a JSON-serializing template exposes the rest of it.
    "response",
)


def _schema_roots(target):
    """The schema values of a tool declaration, or an empty list."""
    if not isinstance(target, dict):
        return []
    return [target[key] for key in _SCHEMA_ROOT_KEYS if isinstance(target.get(key), (dict, list))]


# "examples" carries instance samples, never subschemas, so a sample that happens to hold a
# key like "required" is ordinary annotation text. Descending into it would read that key as
# the JSON Schema keyword and drop a usable tool, so the scan stops here and the rewrite
# neutralizes the sample as descriptive metadata instead.
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
                    # The keys of this map are names, so only its VALUES are subschemas.
                    # Descending into the map itself would read a property literally named
                    # "format", "default" or "id" as the keyword of the same name and drop a
                    # perfectly ordinary tool.
                    for value in item.values():
                        if isinstance(value, (dict, list)) and id(value) not in seen:
                            seen.add(id(value))
                            stack.append(value)
                    continue
                elif key in _SCHEMA_VALUED_IDENTIFIERS:
                    # Every leaf, not just a top-level string: an enum entry or const can be
                    # any value, so "enum": [["<s>"]] and "const": {"tag": "</think>"} are
                    # literals the model must reproduce exactly.
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
    """The function name a ``tool_choice`` pins, or None when it pins nothing.

    OpenAI: ``{"type": "function", "function": {"name": ...}}``; Anthropic:
    ``{"type": "tool", "name": ...}``. The string forms pin no particular tool."""
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
        # Both spellings: an entry may carry an empty "function" mapping beside the flat
        # name that actually dispatches, and reading only the nested level made a dropped
        # tool look as though it had never been in the caller's catalog -- so a forced
        # tool_choice for it was forwarded unchanged (#7066).
        nested = function.get("name") if isinstance(function, dict) else None
        for name in (nested, tool.get("name")):
            if isinstance(name, str):
                names.add(name)
    return names


def _tokenizer_strings(inner) -> Optional[list]:
    """Every string a tokenizer exposes: added tokens AND the base vocabulary.

    Neither source alone is enough. A tokenizer can carry unrelated added tokens while its
    chat sentinels stay in the base SentencePiece vocabulary, so short-circuiting on a
    populated added_tokens_decoder left a model's own turn boundaries out of the profile
    and a pasted copy reached the prompt byte-exact (#7066).
    """
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

    ``_generate_chat_response_inner`` applies ``get_chat_template`` only when it renders, so
    a profile or an authorization catalog built before that saw the LOAD-time template. A
    tool whose schema carries a delimiter the mapped template introduces was then dropped
    from the prompt but still authorized for healing or execution (#7066).

    Resolved on a COPY of the tokenizer: ``get_chat_template`` assigns
    ``tokenizer.chat_template`` (unsloth/chat_templates.py), and this runs before the
    generation lock, so handing it the shared object would let this setup mutate a tokenizer
    another request is rendering with. ``render_native_template`` clones for the same
    reason. Only the resulting template string is kept."""
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
            # Shallow copy: get_chat_template writes chat_template onto whatever it is
            # given, and a concurrent generation may be rendering with the shared object.
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
    """The object whose chat template a render will actually use.

    ``_generate_vlm`` falls back to the nested tokenizer when the processor cannot render
    a chat itself (mlx_inference.py), so anything profiling the prompt ahead of the render
    has to make the same choice. Reproducing the rule at the call site let the two drift:
    a processor without a usable ``chat_template`` was profiled as a processor, selecting
    "default", while the render used the nested tokenizer's tool_use template (#7066)."""
    if processor is None:
        return tokenizer
    if (
        getattr(processor, "apply_chat_template", None) is None
        or getattr(processor, "chat_template", None) is None
    ):
        nested = getattr(processor, "tokenizer", None)
        return processor if nested is None else nested
    return processor


# The Jinja variable HF passes the schema in. A template that never reads it cannot put a
# tool in the prompt, whatever the caller asked for.
_TOOLS_VARIABLE = re.compile(r"\btools\b")
# Only what Jinja evaluates counts. A raw word search over the whole body also matched the
# word in prose the template merely prints, so "{{ 'no tools available' }}" read as a
# template that renders schemas and the catalog stayed authorized (#7066).
_JINJA_CODE = re.compile(r"\{\{(.*?)\}\}|\{%(.*?)%\}", re.S)
# String literals are data, not a variable read: the same prose moved inside an expression
# would otherwise pass. Non-greedy and per-quote, so an apostrophe in a double-quoted
# string cannot swallow the rest of the expression.
_JINJA_STRING = re.compile(r"'[^']*'|\"[^\"]*\"", re.S)


# A template that replays assistant tool_calls and tool results takes part in tool calling
# by design, with the schema supplied by the caller's system prompt rather than by the
# template: DeepSeek-R1 renders message['tool_calls'] and <|tool outputs|> and never reads
# the tools variable at all. Unlike the tools read, this one keeps string literals, because
# the name appears as a mapping key.
_TOOL_TURN = re.compile(
    # message.tool_calls / message["tool_calls"] / "tool_calls" in message, and the role
    # comparison a template uses to render a tool result. Matching the bare word instead
    # let "{{ 'tool_calls are unsupported' }}" count as taking part in tool calling.
    r"\.tool_calls\b"
    r"|\[\s*['\"]tool_calls['\"]\s*\]"
    r"|['\"]tool_calls['\"]\s+in\b"
    r"|\btool_calls\s+in\b"
    # "message['role'] == 'tool'" as well as "message.role == 'tool'": the closing quote
    # and bracket sit between the name and the comparison.
    r"|role['\"\]\s]*==\s*['\"]tool['\"]"
    r"|['\"]tool['\"]\s*==[\s\['\"]*role"
)


def _jinja_code(body: str):
    """Yield only what Jinja evaluates: the inside of every {{ }} and {% %}."""
    for match in _JINJA_CODE.finditer(_JINJA_COMMENT.sub("", body)):
        yield match.group(1) or match.group(2) or ""


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
) -> bool:
    """True unless the template selected out of *value* takes no part in tool calling.

    Reading the ``tools`` variable is the direct case. Replaying tool calls counts too:
    such a template round-trips a tool turn it never advertised, so the schema came from
    the caller's own system prompt and the catalog is authorized after all."""
    bodies = _selected_template_strings_from_value(value, tools, prefer_tool_use = prefer_tool_use)
    if not bodies:
        # Unreadable, not proven silent. Emptying the catalog here would disable healing
        # for every model whose template shape this module cannot parse, which is a
        # feature regression rather than the narrow authorization fix (#7066).
        return True
    return any(_reads_tools_variable(body) or _round_trips_tool_calls(body) for body in bodies)


def _renders_tool_schema(target, template, tools) -> bool:
    """True unless the template *target* will select provably cannot advertise tools."""
    value = template or getattr(target, "chat_template", None)
    if not value:
        value = getattr(getattr(target, "tokenizer", None), "chat_template", None)
    return _template_reads_tools(value, tools, prefer_tool_use = not _is_processor(target))


def renderable_tool_catalog_for_targets(
    tools,
    targets,
    model_info,
    cache = None,
    active_model_name = None,
    template = None,
):
    """The catalog safe under every object a backend could render this turn with.

    The two backends disagree about which object renders a text turn on a vision model.
    MLX keeps the processor when it has a usable chat template (mlx_inference.py), while
    the transformers path unwraps to the nested tokenizer unconditionally
    (inference.py). Their profiles can differ, so authorizing against one of them lets the
    other's render drop a tool that stays in the healer's catalog (#7066).

    Rather than guessing which backend will serve the request, this takes the same
    conservative intersection ``renderable_tool_catalog`` already takes across the active
    and native templates: a tool has to survive every candidate to stay authorized.

    Chained rather than intersected by name, so the surviving descriptions carry every
    candidate's sweep too. Sanitizing an already-sanitized catalog is stable, since a
    broken marker no longer matches, so a clean catalog stays byte-identical."""
    catalog = tools
    for target in targets:
        if target is None:
            continue
        catalog = renderable_tool_catalog(
            catalog, target, model_info, cache, active_model_name, template
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
):
    """The catalog that survives EVERY template this request could render with.

    A tool-calling turn may render with the active template or, when that template drops
    the schema, with the model's native one, and the two profiles can disagree about which
    tools carry markup. A healer or controller is an authorization boundary, so it has to
    be built from the catalog that is safe either way: promoting a call for a tool the
    prompt never advertised is the failure this guards (#7066).

    The cost of the conservative direction is narrow and one-sided. A tool dropped only by
    the template that did NOT get selected stays advertised and directly callable; it just
    is not auto-healed out of text-form output that round.
    """
    # The mapper installs its template on the TOKENIZER at generate time. A processor
    # render reads the processor's own chat_template and never sees it (_generate_vlm),
    # so resolving it for a processor target profiled a prompt that render cannot
    # produce, and a tool carrying a processor-only delimiter survived the catalog while
    # the actual render dropped it (#7066).
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

    active_renders_tools = _renders_tool_schema(tokenizer, template, tools)
    # A processor stays on "default" and the VLM path renders straight through
    # apply_chat_template_for_generation, with no native-template fallback behind it
    # (mlx_inference.py). When that default body never reads ``tools`` the schema cannot
    # reach the prompt at all, so every tool in the catalog is unadvertised and healing a
    # text-form call would promote one the model was never shown (#7066).
    if _is_processor(tokenizer) and not active_renders_tools:
        return _unadvertised()
    # Resolved rather than read: render_native_template fetches it during the render, so on
    # the FIRST request needing the fallback the cache is still empty and this would hand
    # back the active-profile catalog unchanged (#7066).
    native_tpl = resolve_native_chat_template(
        model_info or {}, active_model_name, (model_info or {}).get("hf_token")
    )
    # The tokenizer path is normally rescued by that fallback: an active template which
    # drops the schema is re-rendered with the native one. When there is no native template
    # to reach -- unresolvable, private, or a failed fetch -- nothing is left that could
    # advertise, and render_with_native_template_fallback keeps the no-tools prompt. The
    # catalog has to say so, or the healer promotes a call the prompt never showed (#7066).
    if not native_tpl:
        return safe if active_renders_tools else _unadvertised()
    if not active_renders_tools and not _template_reads_tools(native_tpl, tools):
        return _unadvertised()
    native = model_markup(native_tpl, _vocabulary_of(tokenizer), tools)
    if native is None:
        return safe
    kept = catalog_tool_names(neutralize_tool_descriptions(tools, cache, native))
    narrowed = [t for t in safe if catalog_tool_names([t]) <= kept]
    return safe if len(narrowed) == len(safe) else narrowed


def reconciled_tool_choice(tool_choice, openai_tools, safe_tools):
    """Downgrade a forced ``tool_choice`` to "auto" when WE dropped its tool (#7066).

    Only when the neutralizer removed it: the name has to be in the caller's catalog and
    gone from the sanitized one. A client forcing a function it never declared is a
    different, pre-existing case the healing path deliberately reads to decide a streamed
    call must NOT be promoted, so rewriting it there would change unrelated behaviour."""
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
    # The list form Hermes-3 ships, [{"name": ..., "template": ...}], is the same selection
    # as the dict form; returning nothing for it made every caller fall back to the union,
    # so a no-tools turn inherited the tool_use markers (#7066).
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
    return _selected_template_strings_from_value(
        getattr(tokenizer, "chat_template", None),
        tools,
        prefer_tool_use = not _is_processor(tokenizer),
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
    # The tool catalog the SELECTED template actually rendered. The native-template
    # fallback sanitizes with the native model's profile, which can drop a tool the active
    # profile kept, so a healer or controller built from the active catalog could promote a
    # call for a tool the prompt never advertised (#7066).
    advertised_tools: Optional[list] = None


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


_MARKUP_BY_TOKENIZER: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _special_token_strings(tokenizer) -> dict:
    """The concrete spelling of each special-token variable a template may emit."""
    specials: dict = {}
    for name in _SPECIAL_TOKEN_VARIABLES:
        try:
            value = getattr(tokenizer, name, None)
        except Exception:
            # A tokenizer property can raise on a partially loaded model; a missing
            # special is simply one fewer marker, never a failed request.
            continue
        # AddedToken rather than str on some tokenizers, and its str() is the content.
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
    """Profile the loaded tokenizer's own structural markers, cached per tokenizer.

    Returns None when the template and vocabulary cannot be read, which falls back to the
    curated patterns: an unreadable model stays fully swept rather than unprotected."""
    if tokenizer is None:
        return None
    try:
        cached = _MARKUP_BY_TOKENIZER.get(tokenizer)
    except TypeError:  # not weak-referenceable
        cached = None
    # A vision model stores the container processor here: the chat_template lives on the
    # processor while the vocabulary lives on the inner tokenizer, so each is read from
    # whichever actually has it (mirrors chat_eos's template_source unwrap).
    inner = getattr(tokenizer, "tokenizer", tokenizer)
    # An explicit *template* is the one this request will actually render with: the
    # generate-time mapper installs its template later, so profiling the load-time one left
    # the authorization catalog a step behind the prompt it was gating (#7066).
    if not template:
        template = getattr(tokenizer, "chat_template", None)
    if not template:
        template = getattr(inner, "chat_template", None)
    # Keyed on the template as well as the tokenizer: get_chat_template() installs a mapped
    # template on the SAME object at generate time, so a profile built at load time would
    # otherwise be reused and the mapped template's own delimiters would go unswept (#7066).
    # Keyed on whether tools are present too: a named-template dict renders "tool_use" for a
    # tool-calling turn and "default" otherwise, and the two emit different literals. Both
    # live in the same entry, so a conversation alternating tool and no-tool turns keeps
    # hitting instead of rebuilding on every message.
    is_processor = _is_processor(tokenizer)
    # A processor always renders "default", so its profile does not vary with tools and the
    # cache must not key two identical entries under different selectors.
    selector = bool(tools) and not is_processor
    # A named-template dict is unhashable, so the key is a stable serialization of it;
    # without this the cache write raised TypeError and every call rebuilt the profile.
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
    # ProcessorMixin.apply_chat_template does NOT switch to "tool_use" implicitly; it
    # renders "default" unless chat_template= names another. _selected_chat_template_strings
    # already documents this, so profiling with the tokenizer rule left a processor's own
    # default-template boundary unswept while the render emitted it (#7066).
    profile = model_markup(
        template,
        tokens,
        tools,
        prefer_tool_use = not is_processor,
        specials = _special_token_strings(inner),
    )
    try:
        entry = cached if isinstance(cached, dict) else {}
        # Two selectors and one template per tokenizer; a template swap adds a key rather
        # than growing without bound, so trim if a tokenizer somehow cycles templates.
        if len(entry) >= 4:
            entry.clear()
        entry[(template_key, selector)] = profile
        _MARKUP_BY_TOKENIZER[tokenizer] = entry
    except TypeError:
        pass
    return profile


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
    # Shared choke point for the transformers and MLX backends (#7066). Gated on the
    # loaded model's own markers, so text naming another family's sentinel is left alone.
    _markup = markup_for_tokenizer(tokenizer, tools)
    tools = neutralize_tool_descriptions(tools, None, _markup)
    # Sanitizing can empty the catalog, and an empty catalog renders with "default" rather
    # than "tool_use". Re-profile before sweeping the messages, or they are swept against a
    # template this request will not use and a default-only delimiter reaches the prompt
    # raw. Order matters: the catalog is sanitized first so the selector is settled (#7066).
    if bool(tools) != bool(_markup and getattr(_markup, "selected_with_tools", False)):
        _markup = markup_for_tokenizer(tokenizer, tools)
    messages = neutralize_control_markup_in_messages(messages, None, _markup)
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

    # An attempt that drops the tools kwarg selects "default" rather than "tool_use", and
    # the messages above were swept for the profile of the template this request meant to
    # use. A custom or older tokenizer that rejects tools= therefore rendered messages
    # against a template they were not swept for, leaving a default-only boundary
    # byte-exact in client text. Built at most once, and only if such an attempt is
    # reached, so the ordinary path pays nothing (#7066).
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

    def _render(msgs: list) -> str:
        last_exc: Optional[Exception] = None
        for kwargs in attempts:
            try:
                return tokenizer.apply_chat_template(
                    _swept_for(kwargs, msgs),
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


def resolve_native_chat_template(
    model_info: dict,
    active_model_name,
    hf_token = None,
):
    """The model's native chat template, fetched once and cached on *model_info*.

    Returns False when the repo has none and None when the fetch failed, so a failure is
    retried rather than pinned. Shared by the render path and by the authorization catalog,
    which must know the native template on the FIRST request too: it is fetched during
    rendering, so a catalog built before that saw no native profile at all and could
    authorize a tool the native render then left out of the prompt (#7066)."""
    native_tpl = model_info.get("native_chat_template")
    if native_tpl is not None:
        return native_tpl
    # A LoRA adapter's native template lives on the base model, not the adapter id.
    template_source = model_info.get("base_model") or active_model_name
    if not template_source:
        return None
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
        logger.warning("Could not load native chat template for '%s': %s", template_source, exc)
        # A failed fetch is not "no template": leave the sentinel unset so the next call
        # retries (caching False would pin the tool-dropping override).
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
    native_tpl = resolve_native_chat_template(model_info, active_model_name, hf_token)
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
            # The NATIVE profile decided this render's catalog: it can drop a tool the
            # active profile kept, so callers must gate healing and tool execution on this
            # list rather than on the one they sanitized themselves (#7066).
            # With *tools*, matching the render above: a named native template selects
            # "tool_use" for a tool-calling turn, and profiling it without them read
            # "default" instead, so a tool the render dropped was reported as advertised.
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

    def _result(
        prompt: str,
        markers = live_markers,
        advertised = None,
    ):
        if return_metadata:
            if advertised is None and tools:
                # With *tools*, so the reported catalog is the one the render sanitized:
                # apply_chat_template_for_generation profiles with them, and omitting them
                # here described a "default" template the request never rendered (#7066).
                advertised = neutralize_tool_descriptions(
                    tools, None, markup_for_tokenizer(tokenizer, tools)
                )
            return ChatTemplateRenderResult(prompt, markers, advertised)
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
