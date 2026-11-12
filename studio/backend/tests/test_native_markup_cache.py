# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The native-template profile in renderable_tool_catalog is cached per tokenizer, not rebuilt per request."""

import core.inference.chat_template_helpers as helpers
from core.inference.chat_template_helpers import (
    catalog_tool_names,
    model_markup,
    renderable_tool_catalog,
)

_NATIVE = (
    "{% for t in tools %}{{ t }}{% endfor %}"
    "{% for m in messages %}<zeta_turn>{{ m }}<|im_end|>{% endfor %}"
)


class _Tok:
    chat_template = (
        "{% for t in tools %}{{ t }}{% endfor %}{% for m in messages %}{{ m }}{% endfor %}"
    )
    eos_token = "<|im_end|>"

    def __init__(self, vocab):
        self.vocab = list(vocab)
        self.added_tokens_decoder: dict = {}

    def get_vocab(self):
        return {token: i for i, token in enumerate(self.vocab)}

    @property
    def vocab_size(self):
        return len(self.vocab)


def _tool(name, prop = "x"):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "d",
            "parameters": {"type": "object", "properties": {prop: {"type": "string"}}},
        },
    }


def _native_calls(monkeypatch):
    calls = []
    real = helpers.model_markup

    def counting(chat_template, *args, **kwargs):
        if chat_template is not _Tok.chat_template:
            calls.append(chat_template)
        return real(chat_template, *args, **kwargs)

    monkeypatch.setattr(helpers, "model_markup", counting)
    return calls


def _fields(markup):
    if markup is None:
        return None
    return (
        sorted(markup.markers),
        markup.control and markup.control.pattern,
        markup.boundary and markup.boundary.pattern,
        markup.selected_with_tools,
    )


def test_native_markup_is_built_once_for_repeated_requests(monkeypatch):
    calls = _native_calls(monkeypatch)
    tok = _Tok(["a", "<zeta_turn>", "<|im_end|>"])
    info = {"native_chat_template": _NATIVE}
    tools = [_tool("ok"), _tool("bad", prop = "<zeta_turn>")]
    first = renderable_tool_catalog(tools, tok, info)
    second = renderable_tool_catalog([_tool("other"), _tool("ok")], tok, info)
    assert catalog_tool_names(first) == {"ok"}
    assert catalog_tool_names(second) == {"other", "ok"}
    assert len(calls) == 1
    fresh = model_markup(
        _NATIVE,
        helpers._vocabulary_of(tok),
        tools,
        specials = helpers._special_token_strings(tok),
    )
    assert _fields(helpers._native_markup_for_tokenizer(tok, _NATIVE, tools)) == _fields(fresh)


def test_a_different_template_or_vocabulary_is_not_served_from_the_cache(monkeypatch):
    calls = _native_calls(monkeypatch)
    tok = _Tok(["a", "<|im_end|>"])
    tools = [_tool("ok"), _tool("bad", prop = "<zeta_turn>")]
    # "<zeta_turn>" is structure only once the vocabulary holds it.
    before = renderable_tool_catalog(tools, tok, {"native_chat_template": _NATIVE})
    assert catalog_tool_names(before) == {"ok", "bad"}
    tok.vocab.append("<zeta_turn>")
    after = renderable_tool_catalog(tools, tok, {"native_chat_template": _NATIVE})
    assert catalog_tool_names(after) == {"ok"}
    other = _NATIVE.replace("<zeta_turn>", "<|im_start|>")
    renderable_tool_catalog(tools, tok, {"native_chat_template": other})
    assert calls == [_NATIVE, _NATIVE, other]
    named = {"default": _NATIVE, "tool_use": _NATIVE}
    renamed = {"default": _NATIVE, "tool_use": other}
    renderable_tool_catalog(tools, tok, {"native_chat_template": named})
    renderable_tool_catalog(tools, tok, {"native_chat_template": renamed})
    assert calls[3:] == [named, renamed]
    renderable_tool_catalog(tools, _Tok(["a", "<|im_end|>"]), {"native_chat_template": _NATIVE})
    assert len(calls) == 6
