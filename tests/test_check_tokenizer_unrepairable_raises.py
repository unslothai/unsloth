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

"""check_tokenizer must raise for a tokenizer whose added ids exceed the vocab and can't be repaired."""

import types
from unittest.mock import patch

import pytest

import unsloth.tokenizer_utils as tu

VOCAB = 8


class _BrokenTokenizer:
    # "<sep>" sits past the embedding and is a special token with no generic *_token slot, so it can't be removed.
    eos_token = "</s>"

    def __init__(self):
        self._added_tokens_encoder = {"</s>": 2, "<sep>": VOCAB + 2}
        self._added_tokens_decoder = {v: k for k, v in self._added_tokens_encoder.items()}
        self.special_tokens_map = {"eos_token": "</s>", "additional_special_tokens": ["<sep>"]}

    @property
    def added_tokens_decoder(self):
        return dict(self._added_tokens_decoder)


def _model():
    weight = types.SimpleNamespace(shape = (VOCAB, 4))
    return types.SimpleNamespace(
        model = types.SimpleNamespace(embed_tokens = types.SimpleNamespace(weight = weight))
    )


def test_unrepairable_tokenizer_raises():
    auto = types.SimpleNamespace(from_pretrained = lambda *a, **k: _BrokenTokenizer())
    with patch.object(tu, "AutoTokenizer", auto):
        with pytest.raises(RuntimeError, match = "out of bounds memory accesses"):
            tu.check_tokenizer(_model(), _BrokenTokenizer(), model_name = "stub")


def test_slow_reload_failure_still_returns_the_tokenizer():
    def from_pretrained(*a, **k):
        raise OSError("no slow tokenizer")

    original = _BrokenTokenizer()
    with patch.object(tu, "AutoTokenizer", types.SimpleNamespace(from_pretrained = from_pretrained)):
        assert tu.check_tokenizer(_model(), original, model_name = "stub") is original
