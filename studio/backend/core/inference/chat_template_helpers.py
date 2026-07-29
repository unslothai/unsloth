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

# Chat-template control markup that must not reach the prompt as raw text from a
# user / system / tool turn: a literal "</think>" ends the reasoning block early,
# and "<|start|>assistant<|channel|>final<|message|>" in a tool result forges an
# assistant turn (#7066). One lookahead over the three shapes the templates emit,
# so a single sub() breaks every marker by inserting one space after the "<":
#   <|name|> / <|name>   ChatML, Llama-3, Harmony/gpt-oss, Zephyr/Phi-3, Gemma-4,
#                        Granite (<|start_of_role|>role<|end_of_role|> ... <|end_of_text|>)
#   <name>   / </name>   Qwen tool XML, Gemma turn delimiters, think tags
#   <name|>              Gemma-4 closing delimiters
# The name list is closed on purpose: bare words match only in the pipe shape, so
# "<div>", "<End>" and "List<String>" are untouched. The bare words that do match
# are template delimiters in their own right, so they break even inside a code
# fence, the same trade the structural parsers make.
_CONTROL_MARKUP = re.compile(
    r"<(?="
    r"\|(?:(?:start|end)_(?:header_id|of_role)|tool(?:_call|_response)?"
    r"|end(?:_of_(?:turn|text))?"
    r"|im_(?:start|end)|assistant|constrain|channel|message|eo[tm]_id"
    r"|return|system|start|think|turn|user|call|\")\|?>"
    r"|/?(?:(?:start|end)_of_turn|tool_(?:call|response)|think)>"
    r"|(?:tool(?:_call|_response)?|channel|turn)\|>"
    r")"
)

# Turn-boundary subset, for replayed ASSISTANT content: that text is
# client-controlled too, so a raw boundary in it truncates or forges a turn.
# Everything else stays byte-identical, because the assistant's own think /
# channel / tool markup is structural and rewriting it would corrupt the
# transcript the template re-renders. Harmony opens every message with <|start|>
# and stops on <|call|> / <|return|>, Zephyr / Phi-3 open a turn with a bare
# <|user|> / <|assistant|> / <|system|>, and Granite opens one with
# <|start_of_role|>role<|end_of_role|> and ends it on its eos <|end_of_text|>, so
# those are boundaries too (#7066).
_TURN_BOUNDARY_MARKUP = re.compile(
    r"<(?="
    r"\|(?:(?:start|end)_(?:header_id|of_role)|im_(?:start|end)"
    r"|end(?:_of_(?:turn|text))?|eo[tm]_id"
    r"|assistant|return|system|start|turn|user|call)\|?>"
    r"|(?:start|end)_of_turn>"
    r"|turn\|>"
    r")"
)


def neutralize_control_markup(text: str) -> str:
    """Break chat-template control markup in free text by spacing out the "<".

    "</think>" becomes "< /think>": still readable, but no longer a delimiter to
    the template, the think extractor or the stop-sequence matcher (#7066). A
    plain space, not an invisible joiner: a space is in every tokenizer
    vocabulary, while U+2060 can fall back to byte junk.
    """
    if not text or "<" not in text:
        return text
    return _CONTROL_MARKUP.sub("< ", text)


def neutralize_turn_boundary_markup(text: str) -> str:
    """Break only the turn-boundary sentinels, for replayed assistant text (#7066)."""
    if not text or "<" not in text:
        return text
    return _TURN_BOUNDARY_MARKUP.sub("< ", text)


def _neutralize_argument_leaves(value):
    """Break control markup in every string leaf (keys included) of *value*."""
    if isinstance(value, str):
        return neutralize_control_markup(value)
    if isinstance(value, dict):
        return {
            neutralize_control_markup(key) if isinstance(key, str) else key: (
                _neutralize_argument_leaves(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_neutralize_argument_leaves(item) for item in value]
    return value


def _neutralize_replayed_tool_call(tool_calls: list) -> list:
    """Neutralize a replayed tool call's name and arguments, keeping "id" exact.

    Gemma-4 renders "<|tool_call>call:NAME{key:<|"|>value<|"|>}<tool_call|>", so
    an argument that echoes pasted text can close the call block and open a
    "<|tool_response>" or a "<|turn>model" of its own (#7066). Arguments are
    data, not transcript structure, so they get the same full rewrite a tool
    result's content gets.

    The name is concatenated straight after "call:" and is client-supplied on
    replay, so it gets the same rewrite -- which is the identity on every name
    that can dispatch. Studio composes names as ^[a-zA-Z0-9_-]{1,64}$ and its
    parsers only ever yield [\\w.\\-]+, so a dispatchable name holds no "<" for the
    pattern to match; a name this rewrites matches no registered tool either way.
    A tool result's "name" takes the same rewrite, so the two still agree when
    Gemma-4 pairs them by name. "id" is opaque and stays byte-exact.
    """
    out: list = []
    for call in tool_calls:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict):
            out.append(call)
            continue
        updates: dict = {}
        name = function.get("name")
        if isinstance(name, str) and name:
            new_name = neutralize_control_markup(name)
            if new_name != name:
                updates["name"] = new_name
        arguments = function.get("arguments")
        if arguments is not None:
            new_arguments = _neutralize_argument_leaves(arguments)
            if new_arguments != arguments:
                updates["arguments"] = new_arguments
        out.append({**call, "function": {**function, **updates}} if updates else call)
    return out


def neutralize_control_markup_in_messages(messages: list) -> list:
    """Neutralize control markup in message content and tool-result names (#7066).

    User / system / tool turns lose every marker; assistant turns lose only the
    turn boundaries and keep their structural think / channel / tool markup,
    which replayed history legitimately holds. Returns the same list object when
    nothing changed, so the common prompt stays byte-for-byte what it was.
    """
    if not messages:
        return messages
    changed = False
    out: list = []
    for msg in messages:
        if not isinstance(msg, dict):
            out.append(msg)
            continue
        role = (msg.get("role") or "").strip().lower()
        rewrite = (
            neutralize_turn_boundary_markup if role == "assistant" else neutralize_control_markup
        )
        updates: dict = {}
        # A tool result's "name" is prompt text too: Gemma-4 falls back to it when
        # "tool_call_id" matches no preceding call and concatenates it into the
        # "<|tool_response>" block, so a marker there forges a turn (#7066).
        name = msg.get("name")
        if role == "tool" and isinstance(name, str) and name:
            new_name = neutralize_control_markup(name)
            if new_name != name:
                updates["name"] = new_name
        content = msg.get("content")
        if content:
            new_content = content
            if isinstance(content, str):
                new_content = rewrite(content)
            elif isinstance(content, list):
                # OpenAI-style parts: rewrite each text, pass images / audio through.
                new_content = [
                    {**part, "text": rewrite(part["text"])}
                    if isinstance(part, dict) and isinstance(part.get("text"), str)
                    else rewrite(part)
                    if isinstance(part, str)
                    else part
                    for part in content
                ]
            if new_content != content:
                updates["content"] = new_content
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            new_tool_calls = _neutralize_replayed_tool_call(tool_calls)
            if new_tool_calls != tool_calls:
                updates["tool_calls"] = new_tool_calls
        if updates:
            out.append({**msg, **updates})
            changed = True
        else:
            out.append(msg)
    return out if changed else messages


def neutralize_tool_descriptions(tools):
    """Neutralize control markup in a rendered tool catalog, keeping names exact.

    A tool declaration is prompt text, and every string in it is rendered: Gemma-4
    interpolates the description into its system turn and ``format_parameters``
    emits property keys unquoted plus ``enum`` / ``required`` entries inline,
    while Granite renders the whole spec with ``tojson``. So a
    "<turn|><|turn>model" anywhere in the schema closes the system turn and forges
    a model one (#7066). ``mcp_client`` copies a server's ``description`` and its
    ``inputSchema`` verbatim, which is how remote text gets there.

    ``function.name`` is the one exception: the client matches the name the model
    echoes back against the one it registered. Everything else takes the rewrite,
    which is the identity on a schema that holds no control markup -- a real
    ``enum`` value or property key has no "<" for the pattern to match, so a live
    catalog is returned unchanged and two distinct keys cannot collide onto one.
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
        updates: dict = {}
        for key, value in target.items():
            if key == "name":
                continue
            new_value = _neutralize_argument_leaves(value)
            if new_value != value:
                updates[key] = new_value
        if not updates:
            out.append(tool)
            continue
        new_target = {**target, **updates}
        out.append({**tool, "function": new_target} if target is function else new_target)
        changed = True
    return out if changed else tools


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
