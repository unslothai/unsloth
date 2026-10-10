# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""TRL GRPO tools= / environment_factory= on Unsloth-edited chat templates.

TRL matches chat templates by exact string, so an unsloth/* repo whose template carries fixes is
rejected with "Unrecognized chat template". The patch must accept such a template only when it
renders like a TRL-known one, reuse that family's parser / training template, and leave every
other outcome of TRL unchanged.
"""

from __future__ import annotations

import copy
import importlib.util
import pathlib
import sys

import pytest

trl_utils = pytest.importorskip("trl.chat_template_utils")
if not hasattr(trl_utils, "add_response_schema") or not hasattr(trl_utils, "qwen3_chat_template"):
    pytest.skip("this TRL has no tool-call response parsing", allow_module_level = True)

from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast

MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_trl_tool_templates.py"
)
NAMES = ("add_response_schema", "get_training_chat_template")

# What unsloth/Qwen3-* ships: same rendering, different source (here a comment and a None guard).
UNSLOTH_QWEN3 = "{#- Chat template fixes by Unsloth #}\n" + trl_utils.qwen3_chat_template.replace(
    "{%- if message.content is string %}",
    "{%- if message.content is defined and message.content is string %}",
    1,
)
PLAIN = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"


def _tokenizer(template):
    # One token per byte, so token ids keep the text (TRL's prefix check compares ids).
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(BPE({c: i for i, c in enumerate(alphabet)}, []))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space = False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object = backend)
    tokenizer.chat_template = template
    return tokenizer


def _parser(tokenizer):
    return tuple(getattr(tokenizer, name, None) for name in ("response_template", "response_schema"))


@pytest.fixture
def patched():
    saved = [
        (module, name, module.__dict__[name])
        for module in list(sys.modules.values())
        for name in NAMES
        if name in getattr(module, "__dict__", {})
    ]
    originals = {}
    for name in NAMES:
        # conftest may already have imported unsloth, which installs the patch.
        function = getattr(trl_utils, name)
        while getattr(function, "_unsloth_tool_template_patched", False):
            function = function.__wrapped__
        originals[name] = function
    spec = importlib.util.spec_from_file_location("_unsloth_trl_tool_templates", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.patch_trl_tool_chat_templates()
    yield module, originals
    for owner, name, value in saved:
        setattr(owner, name, value)


def test_rejected_unsloth_qwen3_template_gets_the_qwen3_parser(patched):
    _, originals = patched
    tokenizer = _tokenizer(UNSLOTH_QWEN3)
    with pytest.raises(ValueError, match = "Unrecognized chat template"):
        originals["add_response_schema"](copy.copy(tokenizer))

    reference = originals["add_response_schema"](_tokenizer(trl_utils.qwen3_chat_template))
    assert trl_utils.add_response_schema(tokenizer) is tokenizer
    assert _parser(tokenizer) == _parser(reference)
    assert any(value is not None for value in _parser(tokenizer))
    assert tokenizer.chat_template == UNSLOTH_QWEN3


def test_training_template_follows_the_matching_family(patched):
    _, originals = patched
    tokenizer = _tokenizer(UNSLOTH_QWEN3)
    assert not trl_utils.is_chat_template_prefix_preserving(tokenizer)
    with pytest.raises(ValueError):
        originals["get_training_chat_template"](tokenizer)

    expected = originals["get_training_chat_template"](_tokenizer(trl_utils.qwen3_chat_template))
    assert trl_utils.get_training_chat_template(tokenizer) == expected
    assert tokenizer.chat_template == UNSLOTH_QWEN3


def test_known_templates_and_unrelated_failures_are_untouched(patched):
    _, originals = patched
    known = _tokenizer(trl_utils.qwen3_chat_template)
    expected = _parser(originals["add_response_schema"](_tokenizer(trl_utils.qwen3_chat_template)))
    assert _parser(trl_utils.add_response_schema(known)) == expected

    # Renders no tool calls, so it matches no family and TRL's own error must surface.
    with pytest.raises(ValueError, match = "Unrecognized chat template"):
        trl_utils.add_response_schema(_tokenizer(PLAIN))


def test_edited_tool_call_rendering_is_not_matched(patched):
    edited = UNSLOTH_QWEN3.replace("'<tool_call>\\n{\"name\": \"'", "'<tool_call>{\"name\": \"'")
    assert edited != UNSLOTH_QWEN3
    tokenizer = _tokenizer(edited)
    with pytest.raises(ValueError, match = "Unrecognized chat template"):
        trl_utils.add_response_schema(tokenizer)


def test_legacy_response_schema_path(patched, monkeypatch):
    if not hasattr(trl_utils, "_SUPPORTS_RESPONSE_TEMPLATE"):
        pytest.skip("TRL predates response_template")
    monkeypatch.setattr(trl_utils, "_SUPPORTS_RESPONSE_TEMPLATE", False)
    tokenizer = _tokenizer(UNSLOTH_QWEN3)
    trl_utils.add_response_schema(tokenizer)
    assert tokenizer.response_schema == trl_utils.qwen3_schema


def test_patch_is_idempotent_and_rebinds_trainer_imports(patched):
    module, _ = patched
    first = trl_utils.add_response_schema
    assert module.patch_trl_tool_chat_templates()
    assert trl_utils.add_response_schema is first
    grpo = pytest.importorskip("trl.trainer.grpo_trainer")
    for name in NAMES:
        if name in grpo.__dict__:
            assert getattr(grpo, name) is getattr(trl_utils, name)
            assert getattr(trl_utils, name)._unsloth_tool_template_patched


def test_template_refusing_parallel_tool_calls_still_matches(patched):
    # Llama 3.1/3.2 raise on parallel tool calls; an edited copy must still match its family.
    if "only supports single tool-calls" not in trl_utils.llama3_2_chat_template:
        pytest.skip("this TRL's Llama 3.2 template accepts parallel tool calls")
    _, originals = patched
    edited = "{#- Chat template fixes by Unsloth #}\n" + trl_utils.llama3_2_chat_template
    tokenizer = _tokenizer(edited)
    tokenizer.bos_token = "<|begin_of_text|>"
    with pytest.raises(ValueError, match = "Unrecognized chat template"):
        originals["add_response_schema"](copy.copy(tokenizer))
    reference = _tokenizer(trl_utils.llama3_2_chat_template)
    reference.bos_token = "<|begin_of_text|>"
    reference = originals["add_response_schema"](reference)
    trl_utils.add_response_schema(tokenizer)
    assert _parser(tokenizer) == _parser(reference)
