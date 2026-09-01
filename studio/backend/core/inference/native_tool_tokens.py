# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Preserve native tool-control tokens without leaking unrelated special tokens."""

from __future__ import annotations

import re


# These are provenance-bearing control tokens emitted by supported native tool
# templates. They must survive decoding so the parser can distinguish a native
# call from ambiguous markerless prose. Ordinary special tokens (EOS, role
# boundaries, etc.) remain suppressed exactly as with skip_special_tokens=True.
NATIVE_TOOL_CONTROL_TOKENS = frozenset(
    {
        "<tool_call>",
        "</tool_call>",
        "<function=",
        '<function name="',
        "</function>",
        "<parameter=",
        '<parameter name="',
        "</parameter>",
        "<param=",
        '<param name="',
        "</param>",
        # GLM argument markup.
        "<arg_key>",
        "</arg_key>",
        "<arg_value>",
        "</arg_value>",
        "<|python_tag|>",
        # Gemma native wrapper and quoted-string delimiter.
        "<|tool_call>",
        "<tool_call|>",
        '<|"|>',
        "[TOOL_CALLS]",
        "[/TOOL_CALLS]",
        "[CALL_ID]",
        "[ARGS]",
        "<｜tool▁calls▁begin｜>",
        "<｜tool_calls_begin｜>",
        "<｜tool▁calls｜>",
        "<｜tool calls begin｜>",
        "<｜tool\\_calls\\_begin｜>",
        "<｜tool▁calls▁end｜>",
        "<｜tool▁call▁begin｜>",
        "<｜tool▁sep｜>",
        "<｜tool▁call▁end｜>",
        "<|tool_calls_section_begin|>",
        "<|tool_calls_section_end|>",
        "<|tool_call_begin|>",
        "<|tool_call_argument_begin|>",
        "<|tool_call_end|>",
        # TML Inkling's role opener is deliberately NOT here: nothing consumes a standalone
        # one, so it would begin every ordinary reply with raw markup, and the invocation
        # marker below is what makes a call recognizable. ``parse_tool_calls_from_text``
        # folds the bare name echo into the span either way, so the call still strips clean.
        "<|content_invoke_tool_json|>",
        "<|end_message|>",
        # Kept whenever tool controls are: the parser skips a call rehearsed inside one, so
        # dropping them makes ``[THINK][TOOL_CALLS]terminal[ARGS]{..}[/THINK]`` a real call.
        "<think>",
        "</think>",
        "[THINK]",
        "[/THINK]",
    }
)


# Which openers make a native CLOSER load-bearing. A closer is only worth keeping when its
# own envelope was opened: "the text mentions some tool signal" is far too broad, since an
# ordinary answer that merely says ``[ARGS]`` would then keep an orphan ``<|end_message|>``.
_NATIVE_CONTROL_OPENERS = {
    "</tool_call>": ("<tool_call>",),
    "<tool_call|>": ("<|tool_call>",),
    "</function>": ("<function=", '<function name="'),
    "</parameter>": ("<parameter=", '<parameter name="'),
    "</param>": ("<param=", '<param name="'),
    "</arg_key>": ("<arg_key>",),
    "</arg_value>": ("<arg_value>",),
    "[/TOOL_CALLS]": ("[TOOL_CALLS]",),
    "<｜tool▁calls▁end｜>": (
        "<｜tool▁calls▁begin｜>",
        "<｜tool_calls_begin｜>",
        "<｜tool▁calls｜>",
        "<｜tool calls begin｜>",
        "<｜tool\\_calls\\_begin｜>",
    ),
    "<｜tool▁call▁end｜>": ("<｜tool▁call▁begin｜>",),
    "<|tool_calls_section_end|>": ("<|tool_calls_section_begin|>",),
    "<|tool_call_end|>": ("<|tool_call_begin|>", "<|tool_call_argument_begin|>"),
    # Only the call marker: ``_TC_JSON_START_RE`` recognizes a TML call at
    # ``<|content_invoke_tool_json|>{``, so the role opener alone leaves the closer inert and
    # ``<|message_model|>hello<|end_message|>`` would keep a token nothing can consume.
    "<|end_message|>": ("<|content_invoke_tool_json|>",),
    "</think>": ("<think>",),
    "[/THINK]": ("[THINK]",),
}


# Openers the parser only honors with a body behind them, so a bare mention opens nothing.
# ``_TC_JSON_START_RE`` reads a TML call at ``<|content_invoke_tool_json|>\s*{``, which makes
# ``The marker <|content_invoke_tool_json|> starts a call.`` inert. Deliberately NOT applied
# to ``<tool_call>``: that wrapper legitimately holds ``<function=..>`` markup instead of an
# object, and demanding a brace there would drop a closer a real call needs.
_OPENER_REQUIRES_BODY = {"<|content_invoke_tool_json|>": re.compile(r"\s*\{")}


def closes_an_open_envelope(text: str, token: str) -> bool:
    """True when ``token`` is a native CLOSER whose own opener appears in ``text``.

    Used where a runtime stop token has to be kept for the parser rather than trimmed for
    display: only a closer that something actually opened is load-bearing.
    """
    openers = _NATIVE_CONTROL_OPENERS.get(token)
    if not openers:
        return False
    body = text[: text.rfind(token)] if token in text else text
    for opener in openers:
        body_re = _OPENER_REQUIRES_BODY.get(opener)
        if body_re is None:
            if opener in body:
                return True
            continue
        pos = body.find(opener)
        while pos != -1:
            if body_re.match(body, pos + len(opener)):
                return True
            pos = body.find(opener, pos + 1)
    return False


_ATEM_REASONING_MARKERS = ("self", "user")
_ATEM_REASONING_CONTROL_TOKENS = frozenset({"<|start|>", "<|message|>", "<|eom|>", "<|eot|>"})


def reasoning_control_tokens(markers) -> frozenset[str]:
    """Special-token strings required by the selected reasoning protocol."""
    if not markers:
        return frozenset()
    if tuple(markers) == _ATEM_REASONING_MARKERS:
        return _ATEM_REASONING_CONTROL_TOKENS
    return frozenset(str(marker) for marker in markers if marker)


def _decode_without_special_spacing(tokenizer, token_ids, *, skip_special_tokens: bool) -> str:
    """Decode without slow-tokenizer spaces between preserved control segments."""
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens = skip_special_tokens,
            spaces_between_special_tokens = False,
        )
    except TypeError as exc:
        # Lightweight tokenizers may not expose the option, and do not add that spacing
        # themselves; re-raise anything else rather than masking it.
        if "spaces_between_special_tokens" not in str(exc):
            raise
        return tokenizer.decode(token_ids, skip_special_tokens = skip_special_tokens)


def _token_is_preserved(token, preserved_tokens) -> bool:
    if not isinstance(token, str) or not token:
        return False
    if token in NATIVE_TOOL_CONTROL_TOKENS or token in preserved_tokens:
        return True
    # A protocol marker may combine one special token with ordinary text, e.g.
    # ``<|channel>thought``. Preserve the special-token component.
    return any(token in marker or marker in token for marker in preserved_tokens)


def _special_token_sets(tokenizer, preserved_tokens = ()) -> tuple[frozenset[int], frozenset[int]]:
    """Return ``(all_special_ids, native_tool_ids)`` for ``tokenizer``.

    If a tokenizer does not expose special-token ids, the caller retains the
    previous fail-closed ``skip_special_tokens=True`` behavior.
    """
    try:
        special_ids = frozenset(int(token_id) for token_id in tokenizer.all_special_ids)
    except (AttributeError, TypeError, ValueError):
        return frozenset(), frozenset()

    preserved_tokens = frozenset(str(token) for token in preserved_tokens if token)
    kept_ids: set[int] = set()
    for token_id in special_ids:
        try:
            token = tokenizer.convert_ids_to_tokens(token_id)
        except Exception:  # noqa: BLE001 -- third-party tokenizer adapters vary
            token = None
        if _token_is_preserved(token, preserved_tokens):
            kept_ids.add(token_id)
            continue
        try:
            decoded = _decode_without_special_spacing(
                tokenizer, [token_id], skip_special_tokens = False
            )
        except Exception:  # noqa: BLE001 -- absence means this id stays suppressed
            decoded = None
        if _token_is_preserved(decoded, preserved_tokens):
            kept_ids.add(token_id)
    return special_ids, frozenset(kept_ids)


def _decode_with_token_sets(tokenizer, token_ids, special_ids, tool_ids) -> str:
    if not special_ids:
        return _decode_without_special_spacing(tokenizer, token_ids, skip_special_tokens = True)
    filtered = [
        int(token_id)
        for token_id in token_ids
        if int(token_id) not in special_ids or int(token_id) in tool_ids
    ]
    return _decode_without_special_spacing(tokenizer, filtered, skip_special_tokens = False)


def decode_with_native_tool_tokens(
    tokenizer,
    token_ids,
    preserved_tokens = (),
) -> str:
    """Decode ids while retaining only recognized native tool special tokens."""
    return _decode_with_token_sets(
        tokenizer, token_ids, *_special_token_sets(tokenizer, preserved_tokens)
    )


def decoder_preserves_token(
    tokenizer,
    token: str,
    preserved_tokens = (),
) -> bool:
    """Whether a ``NativeToolTokenDecoder`` over ``tokenizer`` would really keep ``token``."""
    if tokenizer is None:
        return False
    try:
        return NativeToolTokenDecoder(tokenizer, preserved_tokens = preserved_tokens).preserves(token)
    except Exception:  # noqa: BLE001 -- an unusable tokenizer keeps the fail-closed answer
        return False


class NativeToolTokenDecoder:
    """Tokenizer proxy used by ``TextIteratorStreamer`` on tool-enabled turns."""

    def __init__(
        self,
        tokenizer,
        *,
        preserved_tokens = (),
    ):
        self._tokenizer = tokenizer
        self._special_ids, self._tool_ids = _special_token_sets(tokenizer, preserved_tokens)

    def preserves(self, token: str) -> bool:
        """Whether this decoder actually keeps ``token``.

        Not the same as "it is in the allowlist": a tokenizer that exposes no usable
        ``all_special_ids`` sends every decode back through ``skip_special_tokens=True``,
        which drops the token regardless. Callers that re-emit a matching opener have to ask
        the decoder, not the policy.
        """
        if not self._special_ids:
            return False
        for token_id in self._tool_ids:
            # Same two steps `_special_token_sets` used to retain the id: the converted
            # token, then the decoded one, since an adapter may only answer the second.
            for lookup in (
                lambda: self._tokenizer.convert_ids_to_tokens(token_id),
                lambda: _decode_without_special_spacing(
                    self._tokenizer, [token_id], skip_special_tokens = False
                ),
            ):
                try:
                    if lookup() == token:
                        return True
                except Exception:  # noqa: BLE001 -- third-party tokenizer adapters vary
                    continue
        return False

    def decode(self, token_ids, **_decode_kwargs) -> str:
        return _decode_with_token_sets(
            self._tokenizer, token_ids, self._special_ids, self._tool_ids
        )

    def decode_stream_token(self, token_id, fallback_text: str) -> str:
        """Decode one streamed id, using MLX text only for ordinary ids."""
        token_id = int(token_id)
        if token_id not in self._special_ids:
            return fallback_text
        if token_id not in self._tool_ids:
            return ""
        return _decode_without_special_spacing(
            self._tokenizer, [token_id], skip_special_tokens = False
        )

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)
