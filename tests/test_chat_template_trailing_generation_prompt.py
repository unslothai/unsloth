# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""ERNIE-4.5-style templates always append the generation prompt after the loop; both repair paths must fix them."""

import pytest

tokenizers = pytest.importorskip("tokenizers")
from transformers import PreTrainedTokenizerFast

from unsloth.tokenizer_utils import _fix_chat_template, _apply_post_load_tokenizer_fixes

# Tail of baidu/ERNIE-4.5-21B-A3B-Thinking's template, message loop simplified.
ERNIE_LIKE = (
    "{{- '<|im_start|>system\\n<global_setting>\\nthink_mode=True\\n</global_setting><|im_end|>\\n\\n' }}"
    "{%- for message in messages %}"
    "{%- if message.role == 'user' %}{{- '<|im_start|>user\\n' + message.content + '<|im_end|>\\n\\n' }}"
    "{%- elif message.role == 'assistant' %}{{- '<|im_start|>assistant\\n<response>\\n' + message.content + '\\n</response>\\n<|im_end|>\\n\\n' }}"
    "{%- endif %}"
    '{%- endfor %}\n {{- "<|im_start|>assistant\\n<think>\\n"}}'
)
HEADER = "<|im_start|>assistant\n<think>\n"
CONVO = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]


def _tokenizer(template):
    from tokenizers import Tokenizer, models, pre_tokenizers

    tk = Tokenizer(models.WordLevel({"[UNK]": 0, "a": 1}, unk_token = "[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object = tk, unk_token = "[UNK]", eos_token = "[UNK]")
    tok.chat_template = template
    tok.name_or_path = "baidu/ERNIE-4.5-21B-A3B-Thinking"
    return tok


def test_trailing_expression_after_whitespace_is_wrapped():
    fixed = _fix_chat_template(ERNIE_LIKE)
    assert fixed != ERNIE_LIKE
    tok = _tokenizer(fixed)
    no = tok.apply_chat_template(CONVO, tokenize = False, add_generation_prompt = False)
    yes = tok.apply_chat_template(CONVO, tokenize = False, add_generation_prompt = True)
    assert no.endswith("</response>\n<|im_end|>\n\n") and not no.endswith(HEADER)
    assert yes == no + HEADER


def test_original_render_with_prompt_is_unchanged():
    orig = _tokenizer(ERNIE_LIKE).apply_chat_template(
        CONVO, tokenize = False, add_generation_prompt = True
    )
    fixed = _tokenizer(_fix_chat_template(ERNIE_LIKE)).apply_chat_template(
        CONVO, tokenize = False, add_generation_prompt = True
    )
    assert orig == fixed


def test_fastmodel_post_load_fixes_repair_the_template():
    tok = _tokenizer(ERNIE_LIKE)
    assert tok.apply_chat_template(CONVO, tokenize = False, add_generation_prompt = False).endswith(
        HEADER
    )
    tok = _apply_post_load_tokenizer_fixes(tok, fix_tokenizer = True, config = None)
    assert not tok.apply_chat_template(CONVO, tokenize = False, add_generation_prompt = False).endswith(
        HEADER
    )


def test_fix_tokenizer_false_leaves_template_alone():
    tok = _apply_post_load_tokenizer_fixes(_tokenizer(ERNIE_LIKE), fix_tokenizer = False, config = None)
    assert tok.chat_template == ERNIE_LIKE


def test_template_that_honours_the_flag_is_untouched():
    good = ERNIE_LIKE.replace(
        '{%- endfor %}\n {{- "<|im_start|>assistant\\n<think>\\n"}}',
        "{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}",
    )
    tok = _apply_post_load_tokenizer_fixes(_tokenizer(good), fix_tokenizer = True, config = None)
    assert tok.chat_template == good
