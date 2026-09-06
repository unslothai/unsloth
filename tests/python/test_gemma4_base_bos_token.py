# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gemma 4 base tokenizers must prepend <bos> at load time.

unsloth/gemma-4-* base mirrors ship without add_bos_token: true while
google/gemma-4-* includes it. Without the runtime fix, generation repeats
degenerate text. See unslothai/unsloth#7903.

Detection keys off the loaded tokenizer / config, not the Hub repo name, so
local folders and extra quant suffixes still get the fix.
"""

import types
from unittest.mock import patch

import pytest

import unsloth.tokenizer_utils as tu


class _Tok:
    def __init__(
        self,
        add_bos_token = False,
        bos_token_id = 2,
        processor_class = None,
        think_token = None,
        boa_token = None,
        chat_template = None,
        eos_token = "<eos>",
        init_kwargs = None,
    ):
        self.add_bos_token = add_bos_token
        self.bos_token_id = bos_token_id
        self.processor_class = processor_class
        self.think_token = think_token
        self.boa_token = boa_token
        self.chat_template = chat_template
        self.eos_token = eos_token
        if init_kwargs is not None:
            self.init_kwargs = init_kwargs


class _Proc:
    def __init__(
        self,
        tokenizer,
        processor_class = "Gemma4Processor",
        chat_template = None,
    ):
        self.tokenizer = tokenizer
        self.processor_class = processor_class
        self.chat_template = chat_template


class _Gemma4Processor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


def _gemma4_base(**kwargs):
    kwargs.setdefault("processor_class", "Gemma4Processor")
    return _Tok(**kwargs)


def test_gemma4_from_processor_class():
    assert tu._is_gemma4_tokenizer(_gemma4_base()) is True


def test_gemma4_from_think_and_boa_tokens():
    tok = _Tok(processor_class = None, think_token = "<think>", boa_token = "<boa>")
    assert tu._is_gemma4_tokenizer(tok) is True


def test_gemma4_from_init_kwargs():
    tok = _Tok(init_kwargs = {"processor_class": "Gemma4Processor"})
    assert tu._is_gemma4_tokenizer(tok) is True


def test_gemma4_from_processor_wrapper():
    proc = _Proc(_Tok())
    assert tu._is_gemma4_tokenizer(proc) is True


def test_gemma4_from_class_name():
    assert tu._is_gemma4_tokenizer(_Gemma4Processor(_Tok())) is True


def test_gemma3_processor_is_not_gemma4():
    tok = _Tok(processor_class = "Gemma3Processor")
    assert tu._is_gemma4_tokenizer(tok) is False
    assert tu._needs_gemma4_base_bos(tok) is False


def test_plain_tokenizer_is_not_gemma4():
    assert tu._is_gemma4_tokenizer(_Tok()) is False


def test_gemma4_config_model_type():
    config = types.SimpleNamespace(model_type = "gemma4", text_config = None)
    assert tu._is_gemma4_config(config) is True
    assert tu._needs_gemma4_base_bos(_Tok(), config = config) is True


def test_gemma4_config_nested_text_config():
    config = types.SimpleNamespace(
        model_type = "gemma4",
        text_config = types.SimpleNamespace(model_type = "gemma4_text"),
    )
    assert tu._is_gemma4_config(config) is True


def test_name_alone_does_not_trigger_fix():
    tok = _Tok()
    # Repo / folder names are ignored: a generic tokenizer must not flip BOS.
    fixed = tu._fix_gemma4_base_bos_token(tok)
    assert fixed.add_bos_token is False


def test_fix_sets_flag_for_quant_and_local_shapes():
    tok = _gemma4_base(add_bos_token = False)
    fixed = tu._fix_gemma4_base_bos_token(tok)
    assert fixed.add_bos_token is True


def test_fix_sets_flag_on_wrapped_processor():
    inner = _Tok(add_bos_token = False)
    proc = _Proc(inner)
    tu._fix_gemma4_base_bos_token(proc)
    assert inner.add_bos_token is True


def test_fix_skips_chat_template_that_emits_bos():
    tok = _gemma4_base(
        add_bos_token = False,
        chat_template = "{{- bos_token -}}{{ messages }}",
    )
    fixed = tu._fix_gemma4_base_bos_token(tok)
    assert fixed.add_bos_token is False


def test_fix_skips_turn_eos_instruct():
    tok = _gemma4_base(add_bos_token = False, eos_token = "<turn|>")
    fixed = tu._fix_gemma4_base_bos_token(tok)
    assert fixed.add_bos_token is False


def test_fix_honors_fix_tokenizer_false():
    tok = _gemma4_base(add_bos_token = False)
    fixed = tu._apply_post_load_tokenizer_fixes(tok, fix_tokenizer = False)
    assert fixed.add_bos_token is False


def test_load_correct_tokenizer_enables_bos_for_gemma4_base():
    def from_pretrained(model_name, **kwargs):
        return _gemma4_base(add_bos_token = False)

    with patch.object(tu, "AutoTokenizer", types.SimpleNamespace(from_pretrained = from_pretrained)):
        result = tu._load_correct_tokenizer("/models/gemma-4-31B-bnb-4bit", fix_tokenizer = True)

    assert result.add_bos_token is True


def test_load_correct_tokenizer_skips_instruct():
    def from_pretrained(model_name, **kwargs):
        return _gemma4_base(
            add_bos_token = False,
            chat_template = "{{- bos_token -}}",
            eos_token = "<turn|>",
        )

    with patch.object(tu, "AutoTokenizer", types.SimpleNamespace(from_pretrained = from_pretrained)):
        result = tu._load_correct_tokenizer("unsloth/gemma-4-E2B-it", fix_tokenizer = True)

    assert result.add_bos_token is False


def test_load_correct_tokenizer_uses_model_config_when_tokenizer_is_generic():
    # Stripped local tokenizers have no processor_class; FastLanguageModel still
    # has model.config.model_type == gemma4.
    def from_pretrained(model_name, **kwargs):
        return _Tok(add_bos_token = False)

    config = types.SimpleNamespace(model_type = "gemma4", text_config = None)
    with patch.object(tu, "AutoTokenizer", types.SimpleNamespace(from_pretrained = from_pretrained)):
        result = tu._load_correct_tokenizer(
            "/models/local-gemma4-bnb-4bit",
            fix_tokenizer = True,
            config = config,
        )

    assert result.add_bos_token is True


def test_fastmodel_processor_path_heals_from_config():
    # FastModel loads Gemma4Processor, then heals after the processor is final.
    inner = _Tok(add_bos_token = False)
    processor = types.SimpleNamespace(
        tokenizer = inner,
        image_processor = object(),
        chat_template = None,
    )
    config = types.SimpleNamespace(model_type = "gemma4", text_config = None)

    fixed = tu._apply_post_load_tokenizer_fixes(processor, fix_tokenizer = True, config = config)

    assert fixed is processor
    assert inner.add_bos_token is True


def test_fastmodel_processor_path_skips_instruct_template():
    inner = _Tok(add_bos_token = False, chat_template = "{{- bos_token -}}")
    processor = types.SimpleNamespace(
        tokenizer = inner,
        image_processor = object(),
        chat_template = "{{- bos_token -}}{{ messages }}",
    )
    config = types.SimpleNamespace(model_type = "gemma4", text_config = None)

    tu._apply_post_load_tokenizer_fixes(processor, fix_tokenizer = True, config = config)
    assert inner.add_bos_token is False


@pytest.mark.e2e
def test_gemma4_e2b_hub_tokenizer_prepends_bos():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    tok = tu.load_correct_tokenizer("unsloth/gemma-4-E2B", fix_tokenizer = True)
    assert tok.add_bos_token is True
    ids = tok("This book is largely concerned with Hobbits,")["input_ids"]
    assert ids[0] == tok.bos_token_id

    # Control: raw Hub tokenizer still omits BOS without the fix.
    raw = AutoTokenizer.from_pretrained("unsloth/gemma-4-E2B", trust_remote_code = True)
    raw_ids = raw("This book is largely concerned with Hobbits,")["input_ids"]
    assert raw_ids[0] != raw.bos_token_id
