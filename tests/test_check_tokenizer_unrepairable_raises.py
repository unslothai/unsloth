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

"""check_tokenizer repairs a <sep> past the embedding (Starling) and raises if it can't."""

import os
import types
from unittest.mock import patch

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

import unsloth.tokenizer_utils as tu

VOCAB = {"<unk>": 0, "<s>": 1, "</s>": 2, "hello": 3, "world": 4}


def _model(rows):
    weight = types.SimpleNamespace(shape = (rows, 4))
    return types.SimpleNamespace(
        model = types.SimpleNamespace(embed_tokens = types.SimpleNamespace(weight = weight))
    )


def _tiny_tokenizer(path, **special):
    backend = Tokenizer(models.WordLevel(VOCAB, unk_token = "<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = backend, unk_token = "<unk>", bos_token = "<s>", eos_token = "</s>"
    )
    if special:
        tokenizer.add_special_tokens(special)
    tokenizer.save_pretrained(path)
    return str(path)


def _max_added_id(tokenizer):
    return max(tokenizer.added_tokens_decoder)


def test_sep_token_past_embedding_is_repaired(tmp_path):
    path = _tiny_tokenizer(tmp_path, sep_token = "<sep>")
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer = tu.check_tokenizer(_model(len(VOCAB)), tokenizer, model_name = path)
    assert _max_added_id(tokenizer) < len(VOCAB)
    assert tokenizer.sep_token is None
    assert max(tokenizer("hello <sep> world").input_ids) < len(VOCAB)


def test_unrepairable_tokenizer_raises(tmp_path):
    # An extra special token has no named *_token slot, so it can't be removed.
    path = _tiny_tokenizer(tmp_path, additional_special_tokens = ["<x>"])
    with pytest.raises(RuntimeError, match = "out of bounds memory accesses"):
        tu.check_tokenizer(_model(len(VOCAB)), AutoTokenizer.from_pretrained(path), model_name = path)


def test_in_range_tokenizer_is_returned_unchanged(tmp_path):
    path = _tiny_tokenizer(tmp_path, sep_token = "<sep>")
    original = AutoTokenizer.from_pretrained(path)
    tokenizer = tu.check_tokenizer(_model(len(VOCAB) + 1), original, model_name = path)
    assert tokenizer is original
    assert tokenizer.sep_token_id == len(VOCAB)


def _no_slow_reload():
    def from_pretrained(*a, **k):
        raise OSError("no slow tokenizer")

    return patch.object(tu, "AutoTokenizer", types.SimpleNamespace(from_pretrained = from_pretrained))


def test_failed_slow_reload_still_repairs(tmp_path):
    # transformers 4.x can't load a fast-only repo with use_fast = False.
    path = _tiny_tokenizer(tmp_path, sep_token = "<sep>")
    original = AutoTokenizer.from_pretrained(path)
    with _no_slow_reload():
        tokenizer = tu.check_tokenizer(_model(len(VOCAB)), original, model_name = path)
    assert _max_added_id(tokenizer) < len(VOCAB)
    assert tokenizer.sep_token is None


def test_failed_slow_reload_still_raises_when_unrepairable(tmp_path):
    path = _tiny_tokenizer(tmp_path, additional_special_tokens = ["<x>"])
    original = AutoTokenizer.from_pretrained(path)
    with _no_slow_reload(), pytest.raises(RuntimeError, match = "out of bounds memory accesses"):
        tu.check_tokenizer(_model(len(VOCAB)), original, model_name = path)


@pytest.fixture(scope = "module")
def starling():
    # Tokenizer files only (~2 MB); the model this check was written for, <sep> at id 32002.
    try:
        from huggingface_hub import hf_hub_download

        files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
        files.append("tokenizer.model")
        paths = [hf_hub_download("berkeley-nest/Starling-LM-7B-alpha", f) for f in files]
    except Exception as e:
        pytest.skip(f"Starling tokenizer not reachable: {e}")
    return os.path.dirname(paths[0])


def test_starling_sep_is_repaired(starling):
    tokenizer = AutoTokenizer.from_pretrained(starling)
    tokenizer = tu.check_tokenizer(_model(32002), tokenizer, model_name = starling)
    assert _max_added_id(tokenizer) < 32002
    assert tokenizer.sep_token is None
    assert max(tokenizer("Hello <sep> world").input_ids) < 32002
    assert tokenizer.eos_token == "<|end_of_turn|>"


def test_starling_with_room_is_unchanged(starling):
    original = AutoTokenizer.from_pretrained(starling)
    assert tu.check_tokenizer(_model(32003), original, model_name = starling) is original
    assert original.sep_token_id == 32002
