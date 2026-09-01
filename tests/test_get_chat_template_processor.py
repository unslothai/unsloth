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


class _Processor:
    """Shaped like transformers' Gemma3Processor: no padding_side of its own."""

    def __init__(self):
        self.tokenizer = _Tokenizer()


@pytest.fixture
def light_patch_tokenizer(monkeypatch):
    # Keep the call off unsloth_zoo (and off unsloth.models, which needs a GPU).
    stub = types.ModuleType("unsloth_zoo.tokenizer_utils")
    stub.patch_tokenizer = lambda model, tokenizer: (model, tokenizer)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.tokenizer_utils", stub)


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
