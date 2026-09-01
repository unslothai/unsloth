# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Regression test for get_chat_template on a processor (vision models).

FastModel.from_pretrained hands back a processor for a multimodal checkpoint, and a
processor keeps `padding_side` on the tokenizer it wraps rather than on itself. Every
other tokenizer attribute get_chat_template touches is read through
`getattr(..., default)`, but `padding_side` was read directly, so the call raised
`AttributeError: 'Gemma3Processor' object has no attribute 'padding_side'`.

chat_templates.py is loaded under a stub package here rather than importing unsloth,
which needs a GPU, the same reason test_bad_mappings_redirect.py execs its target.
"""

import importlib.util
import os
import sys
import types

import json

import pytest

_UNSLOTH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "unsloth"))

_STUBBED = ("unsloth", "unsloth.chat_templates", "unsloth.ollama_template_mappers")


@pytest.fixture(scope = "module")
def get_chat_template():
    saved = {name: sys.modules.get(name) for name in _STUBBED}
    try:
        package = types.ModuleType("unsloth")
        package.__path__ = [_UNSLOTH]
        sys.modules["unsloth"] = package

        spec = importlib.util.spec_from_file_location(
            "unsloth.chat_templates", os.path.join(_UNSLOTH, "chat_templates.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["unsloth.chat_templates"] = module
        spec.loader.exec_module(module)
        yield module.get_chat_template
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class _Tokenizer:
    """The bare surface get_chat_template reads off a non-fast tokenizer."""

    is_fast = False
    padding_side = "right"
    eos_token = "<eos>"
    bos_token = "<bos>"
    unk_token = "<unk>"
    pad_token = "<pad>"
    added_tokens_decoder = {}


class _Backend:
    def __init__(self, tokens):
        self.tokens = tokens

    def to_str(self):
        return json.dumps({"model": {"vocab": {token: i for i, token in enumerate(self.tokens)}}})

    def from_str(self, string_vocab):
        return _Backend(list(json.loads(string_vocab)["model"]["vocab"]))


class _FastTokenizer:
    is_fast = True
    added_tokens_decoder = {}

    def __init__(
        self,
        tokenizer_object = None,
        tokens = None,
        eos_token = "<eos>",
        pad_token = "<pad>",
        bos_token = "<bos>",
        unk_token = "<unk>",
    ):
        self._tokenizer = tokenizer_object or _Backend(tokens)
        self.padding_side = "right"
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.unk_token = unk_token


class GemmaProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


class Gemma4Processor(GemmaProcessor):
    pass


class _Processor:
    """Shaped like transformers' Gemma3Processor: no padding_side of its own."""

    def __init__(self):
        self.tokenizer = _Tokenizer()


@pytest.fixture
def light_patch_tokenizer(monkeypatch):
    # Keep the call off unsloth_zoo (and off unsloth.models, which needs a GPU).
    stub = types.ModuleType("unsloth_zoo.tokenizer_utils")
    stub.patch_tokenizer = lambda model, tokenizer: (model, tokenizer)

    sentencepiece_stub = types.ModuleType("unsloth.tokenizer_utils")
    sentencepiece_stub.fix_sentencepiece_tokenizer = lambda tokenizer, rebuilt, mapping: rebuilt
    monkeypatch.setitem(sys.modules, "unsloth_zoo.tokenizer_utils", stub)

    monkeypatch.setitem(sys.modules, "unsloth.tokenizer_utils", sentencepiece_stub)


def _apply(get_chat_template, tokenizer):
    return get_chat_template(
        tokenizer,
        chat_template = ("{{ messages }}", "<eos>"),
        patch_saving = False,
        use_zoo_tokenizer_patch = True,
    )


def test_processor_gets_its_chat_template(get_chat_template, light_patch_tokenizer):
    processor = _Processor()
    patched = _apply(get_chat_template, processor)

    assert patched is processor
    assert patched.chat_template == "{{ messages }}"
    # Taken from the tokenizer the processor wraps, which is where it lives.
    assert patched.padding_side == "right"
    assert patched.tokenizer.padding_side == "right"


def test_plain_tokenizer_is_unchanged(get_chat_template, light_patch_tokenizer):
    tokenizer = _Tokenizer()
    patched = _apply(get_chat_template, tokenizer)

    assert patched is tokenizer
    assert patched.chat_template == "{{ messages }}"
    assert patched.padding_side == "right"


def _assert_processor_maps_chatml_tokens(get_chat_template, processor, old_tokens):
    patched = get_chat_template(
        processor,
        patch_saving = False,
        use_zoo_tokenizer_patch = True,
    )

    string_vocab = patched.tokenizer._tokenizer.to_str()
    assert patched is processor
    assert patched.tokenizer.is_fast is True
    assert patched.chat_template == patched.tokenizer.chat_template
    assert patched.padding_side == patched.tokenizer.padding_side == "right"
    assert patched.eos_token == patched.tokenizer.eos_token == "<|im_end|>"
    assert '"<|im_start|>"' in string_vocab
    assert '"<|im_end|>"' in string_vocab
    for token in old_tokens:
        assert f'"{token}"' not in string_vocab


def test_processor_uses_wrapped_fast_tokenizer_for_vocab_mapping(
    get_chat_template, light_patch_tokenizer
):
    tokenizer = _FastTokenizer(
        tokens = ["<unk>", "<eos>", "<pad>", "<bos>", "<start_of_turn>"],
    )
    _assert_processor_maps_chatml_tokens(
        get_chat_template,
        GemmaProcessor(tokenizer),
        ("<start_of_turn>", "<eos>"),
    )


def test_gemma4_processor_maps_native_turn_tokens_for_chatml(
    get_chat_template, light_patch_tokenizer
):
    tokenizer = _FastTokenizer(
        tokens = ["<unk>", "<eos>", "<pad>", "<bos>", "<|turn>", "<turn|>"],
        eos_token = "<turn|>",
    )
    _assert_processor_maps_chatml_tokens(
        get_chat_template,
        Gemma4Processor(tokenizer),
        ("<|turn>", "<turn|>"),
    )
