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
    ):
        self.add_bos_token = add_bos_token
        self.bos_token_id = bos_token_id


@pytest.mark.parametrize(
    "model_name,expected",
    [
        ("unsloth/gemma-4-E2B", True),
        ("unsloth/gemma-4-E4B", True),
        ("unsloth/gemma-4-31B", True),
        ("unsloth/gemma-4-26B-A4B", True),
        ("unsloth/gemma-4-E2B-unsloth-bnb-4bit", True),
        ("unsloth/gemma-4-E2B-it", False),
        ("unsloth/gemma-4-E2B-it-unsloth-bnb-4bit", False),
        ("google/gemma-4-E2B-it", False),
        ("unsloth/gemma-3-4b", False),
        ("unsloth/Qwen2.5-7B-Instruct", False),
    ],
)
def test_is_gemma4_base_model_name(model_name, expected):
    assert tu._is_gemma4_base_model_name(model_name) is expected


def test_fix_gemma4_base_bos_token_sets_flag():
    tok = _Tok(add_bos_token = False)
    fixed = tu._fix_gemma4_base_bos_token(tok, "unsloth/gemma-4-E2B")
    assert fixed.add_bos_token is True


def test_fix_gemma4_base_bos_token_skips_it_variants():
    tok = _Tok(add_bos_token = False)
    fixed = tu._fix_gemma4_base_bos_token(tok, "unsloth/gemma-4-E2B-it")
    assert fixed.add_bos_token is False


def test_load_correct_tokenizer_enables_bos_for_gemma4_base():
    name = "unsloth/gemma-4-E2B"

    def from_pretrained(model_name, **kwargs):
        return _Tok(add_bos_token = False)

    with patch.object(tu, "AutoTokenizer", types.SimpleNamespace(from_pretrained = from_pretrained)):
        result = tu._load_correct_tokenizer(name, fix_tokenizer = True)

    assert result.add_bos_token is True


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
