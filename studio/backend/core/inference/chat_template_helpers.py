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
from typing import Optional

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_GEMMA_CHANNEL_START = "<|channel>"
_GEMMA_THOUGHT_OPEN = "<|channel>thought\n"
_GEMMA_THOUGHT_CLOSE = "<channel|>"
_GEMMA_TEMPLATE_OPENERS = (_GEMMA_THOUGHT_OPEN, _GEMMA_THOUGHT_OPEN.replace("\n", "\\n"))


def _tokenizer_objects(tokenizer) -> tuple:
    """Return a processor/tokenizer and its distinct nested tokenizer."""
    if tokenizer is None:
        return ()
    nested = getattr(tokenizer, "tokenizer", None)
    return (tokenizer,) if nested is None or nested is tokenizer else (tokenizer, nested)


def _has_token(tokenizer, token: str) -> bool:
    """Check a vocabulary token without assuming ``get_vocab`` is cheap."""
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        try:
            token_id = convert(token)
            if token_id is not None and token_id != getattr(tokenizer, "unk_token_id", None):
                convert_back = getattr(tokenizer, "convert_ids_to_tokens", None)
                if callable(convert_back) and convert_back(token_id) == token:
                    return True
        except Exception:
            pass
    try:
        return token in tokenizer.get_vocab()
    except Exception:
        return False


def _has_token_pair(tokenizers: tuple, first: str, second: str) -> bool:
    """Resolve a marker pair through one processor or nested tokenizer."""
    return any(
        _has_token(tokenizer, first) and _has_token(tokenizer, second) for tokenizer in tokenizers
    )


def _chat_template_strings(tokenizer) -> tuple[str, ...]:
    """Return string templates from scalar or named-template metadata."""
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, str):
        return (template,)
    if isinstance(template, dict):
        return tuple(value for value in template.values() if isinstance(value, str))
    return ()


def detect_reasoning_channel_markers(tokenizer) -> Optional[tuple[str, str]]:
    """Return native Gemma thought-channel markers supported by a tokenizer.

    Detection uses tokenizer metadata rather than model names, so local paths
    and aliases behave like Hub ids. When no template is exposed, named
    ``soc_token`` / ``eoc_token`` metadata must resolve back to real tokens.
    """
    objects = _tokenizer_objects(tokenizer)
    if any(
        any(opener in template for opener in _GEMMA_TEMPLATE_OPENERS)
        for obj in objects
        for template in _chat_template_strings(obj)
    ):
        return _GEMMA_THOUGHT_OPEN, _GEMMA_THOUGHT_CLOSE
    for obj in objects:
        if (
            getattr(obj, "soc_token", None) == _GEMMA_CHANNEL_START
            and getattr(obj, "eoc_token", None) == _GEMMA_THOUGHT_CLOSE
            and _has_token_pair(objects, _GEMMA_CHANNEL_START, _GEMMA_THOUGHT_CLOSE)
        ):
            return _GEMMA_THOUGHT_OPEN, _GEMMA_THOUGHT_CLOSE
    return None


def _split_partial_marker(text: str, marker: str) -> tuple[str, str]:
    """Hold the longest suffix that may become ``marker`` in the next chunk."""
    for length in range(min(len(text), len(marker) - 1), 0, -1):
        if text.endswith(marker[:length]):
            return text[:-length], text[-length:]
    return text, ""


class ReasoningChannelNormalizer:
    """Incrementally convert one native reasoning channel to ``<think>``.

    The parser follows mlx-vlm's streaming boundary behavior but emits Studio's
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

    def feed(self, text: str) -> str:
        """Consume a raw text delta and return the stable canonical delta."""
        self._buffer += text or ""
        output: list[str] = []
        while self._buffer:
            if self._reasoning_done:
                output.append(self._buffer)
                self._buffer = ""
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
    tokenizer,
    cancel_event = None,
):
    """Normalize a prefix-monotonic cumulative text stream when supported."""
    markers = detect_reasoning_channel_markers(tokenizer)
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
        return _render(messages)
    except Exception:
        # Strict tool templates reject the JSON-string ``arguments`` form via
        # TypeError or a broad Jinja raise_exception, so retry with dicts coerced.
        # Original messages render first, so working templates stay byte-identical.
        normalized = _normalize_tool_call_arguments(messages)
        if normalized is messages:
            raise
        return _render(normalized)


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
) -> Optional[str]:
    """Render ``messages`` + ``tools`` with the model's NATIVE chat template.

    Some Unsloth override templates (e.g. ``mistral``, ``gemma-4``) do not emit
    the ``tools`` schema, so a tool-calling turn silently stops advertising tools.
    The native template ships in the model repo and carries the family's
    tool-calling syntax. It is loaded straight from the repo (bypassing any
    override on the live tokenizer) and cached on ``model_info``. Returns the
    rendered prompt only if the native template actually emits the tools (render
    differs with vs without tools); otherwise ``None``.

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
    return with_tools if with_tools != no_tools else None


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
) -> str:
    """Return ``formatted_prompt``, swapping in a native-template render when an
    override template dropped the ``tools`` schema.

    If ``tools`` were requested but the live render is identical with and without
    them (detected by comparison, robust against tool names in the system prompt),
    re-render with the model's native template. Shared by the transformers and MLX
    backends so both advertise tools consistently. ``hf_token`` is forwarded so a
    gated/private model's native template can still be fetched."""
    if not tools:
        return formatted_prompt
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
        return formatted_prompt
    if formatted_prompt != probe_no_tools:
        return formatted_prompt  # template already emits the tools schema
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
    )
    if native_prompt:
        logger.info(
            "Override template for '%s' dropped tool schemas; using the model's "
            "native template for this tool-calling turn.",
            active_model_name,
        )
        return native_prompt
    return formatted_prompt
