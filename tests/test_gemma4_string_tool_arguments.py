# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Gemma 4 renders JSON-string tool arguments in its trained call syntax (#18847)."""

import ast
import copy
import importlib
import importlib.util
import os
from pathlib import Path

import pytest

pytest.importorskip("transformers")
from tokenizers import Tokenizer, models  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
_MODULES = (
    "transformers.utils.chat_template_utils",
    "transformers.tokenization_utils_base",
    "transformers.processing_utils",
)
TRAINED_CALL = 'call:get_weather{city:<|"|>Paris<|"|>,days:3}'


def _load_import_fixes():
    path = os.environ.get("UNSLOTH_IMPORT_FIXES", str(REPO / "unsloth" / "import_fixes.py"))
    spec = importlib.util.spec_from_file_location("unsloth_import_fixes_gemma4_args", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gemma4_templates():
    path = REPO / "unsloth" / "chat_templates.py"
    namespace = {}
    for node in ast.parse(path.read_text(encoding = "utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) in ("gemma4_template", "gemma4_thinking_template")
            for t in node.targets
        ):
            exec(compile(ast.Module([node], []), str(path), "exec"), namespace)
    return {k: namespace[k] for k in ("gemma4_template", "gemma4_thinking_template")}


@pytest.fixture
def fixed():
    saved = {name: importlib.import_module(name).render_jinja_template for name in _MODULES}
    # conftest imports unsloth, which may already have patched these: start from upstream's.
    for name, function in saved.items():
        while getattr(function, "_unsloth_patched_gemma4_tool_arguments", False):
            function = function.__wrapped__
        importlib.import_module(name).render_jinja_template = function
    fixes = _load_import_fixes()
    fixes.fix_gemma4_string_tool_arguments()
    yield fixes
    for name, original in saved.items():
        importlib.import_module(name).render_jinja_template = original


def _tokenizer(chat_template):
    vocab = {"<unk>": 0, "<bos>": 1, "<eos>": 2}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = Tokenizer(models.WordLevel(vocab, unk_token = "<unk>")),
        bos_token = "<bos>",
        eos_token = "<eos>",
        unk_token = "<unk>",
    )
    tokenizer.chat_template = chat_template
    return tokenizer


def _messages(arguments):
    return [
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"type": "function", "function": {"name": "get_weather", "arguments": arguments}}
            ],
        },
        {"role": "tool", "content": "sunny"},
        {"role": "assistant", "content": "It is sunny."},
    ]


def _render(template, arguments):
    return _tokenizer(template).apply_chat_template(_messages(arguments), tokenize = False)


@pytest.mark.parametrize("name", ["gemma4_template", "gemma4_thinking_template"])
def test_json_string_arguments_render_like_a_dict(fixed, name):
    template = _gemma4_templates()[name]
    string_render = _render(template, '{"days": 3, "city": "Paris"}')
    assert string_render == _render(template, {"city": "Paris", "days": 3})
    assert TRAINED_CALL in string_render


def test_batched_render_and_input_untouched(fixed):
    template = _gemma4_templates()["gemma4_template"]
    conversation = _messages('{"city": "Paris", "days": 3}')
    before = copy.deepcopy(conversation)
    rendered = _tokenizer(template).apply_chat_template(
        [conversation, conversation], tokenize = False
    )
    assert all(TRAINED_CALL in r for r in rendered)
    assert conversation == before


@pytest.mark.parametrize(
    "arguments", ["not-json", "[1, 2]", '{"city": ', "[" * 100000 + "]" * 100000]
)
def test_non_object_strings_render_as_before(fixed, arguments):
    template = _gemma4_templates()["gemma4_template"]
    patched = _render(template, arguments)
    for name in _MODULES:
        module = importlib.import_module(name)
        module.render_jinja_template = module.render_jinja_template.__wrapped__
    assert patched == _render(template, arguments)


def test_other_templates_are_not_touched(fixed):
    template = (
        "{% for m in messages %}{% for c in m.tool_calls or [] %}"
        "{{ c.function.arguments is string }}|{{ c.function.arguments }}{% endfor %}{% endfor %}"
    )
    assert _render(template, '{"city": "Paris"}') == 'True|{"city": "Paris"}'


def test_every_import_site_patched_once(fixed):
    fixed.fix_gemma4_string_tool_arguments()
    wrappers = {importlib.import_module(name).render_jinja_template for name in _MODULES}
    assert len(wrappers) == 1
    (wrapper,) = wrappers
    assert getattr(wrapper, "_unsloth_patched_gemma4_tool_arguments", False)
    assert not getattr(wrapper.__wrapped__, "_unsloth_patched_gemma4_tool_arguments", False)
