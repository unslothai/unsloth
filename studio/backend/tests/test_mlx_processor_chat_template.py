# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from test_mlx_training_worker_config import _load_worker_module  # noqa: E402

CHATML = "{% for m in messages %}<|im_start|>{{ m['role'] }}{% endfor %}"


class TokenizerWrapper:
    def __init__(self, inner):
        self._tokenizer = inner


@pytest.fixture(scope = "module")
def carry():
    return _load_worker_module()._carry_mlx_chat_template_to_processor


def _base_processor():
    return SimpleNamespace(chat_template = None, tokenizer = SimpleNamespace(chat_template = None))


def test_base_checkpoint_processor_gets_the_training_template(carry):
    processor = _base_processor()
    trained = SimpleNamespace(chat_template = CHATML)
    assert carry(SimpleNamespace(_processor = processor), trained)
    assert processor.chat_template == CHATML
    assert processor.tokenizer is trained


def test_mlx_lm_wrapper_is_unwrapped(carry):
    processor = _base_processor()
    trained = SimpleNamespace(chat_template = CHATML)
    assert carry(SimpleNamespace(_processor = processor), TokenizerWrapper(trained))
    assert processor.tokenizer is trained


def test_hf_fast_tokenizer_backend_is_not_unwrapped(carry):
    # PreTrainedTokenizerFast keeps its Rust backend on _tokenizer too.
    processor = _base_processor()
    trained = SimpleNamespace(chat_template = CHATML, _tokenizer = object())
    assert carry(SimpleNamespace(_processor = processor), trained)
    assert processor.tokenizer is trained


@pytest.mark.parametrize("owner", ["processor", "tokenizer"])
def test_checkpoints_with_a_template_are_left_alone(carry, owner):
    processor = _base_processor()
    own_tokenizer = processor.tokenizer
    if owner == "processor":
        processor.chat_template = "native"
    else:
        own_tokenizer.chat_template = "native"
    assert not carry(SimpleNamespace(_processor = processor), SimpleNamespace(chat_template = CHATML))
    assert processor.tokenizer is own_tokenizer
    assert processor.chat_template == ("native" if owner == "processor" else None)


def test_no_processor_or_no_template_is_a_no_op(carry):
    trained = SimpleNamespace(chat_template = CHATML)
    assert not carry(SimpleNamespace(), trained)
    processor = _base_processor()
    assert not carry(SimpleNamespace(_processor = processor), SimpleNamespace(chat_template = None))
    assert processor.chat_template is None


def test_template_set_in_place_on_the_processor_tokenizer_still_reaches_the_processor(carry):
    processor = _base_processor()
    processor.tokenizer.chat_template = CHATML
    assert carry(SimpleNamespace(_processor = processor), processor.tokenizer)
    assert processor.chat_template == CHATML
