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

"""A slow tokenizer transformers cannot convert (remote-code SPTokenizer in openGPT-X/Teuken-7B) must be kept, not crash the load."""

from transformers import PreTrainedTokenizer
from transformers.convert_slow_tokenizer import convert_slow_tokenizer

import unsloth.tokenizer_utils as tu


class SPTokenizer(PreTrainedTokenizer):
    # Like Teuken's remote-code tokenizer: a custom slow class with no vocab_file attribute.
    def __init__(self, **kwargs):
        self._vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "hello": 3, "world": 4}
        super().__init__(unk_token = "<unk>", bos_token = "<s>", eos_token = "</s>", **kwargs)

    @property
    def vocab_size(self):
        return len(self._vocab)

    def get_vocab(self):
        return dict(self._vocab)

    def _tokenize(self, text, **kwargs):
        return text.split()

    def _convert_token_to_id(self, token):
        return self._vocab.get(token, 0)

    def _convert_id_to_token(self, index):
        return {v: k for k, v in self._vocab.items()}.get(index, "<unk>")


def test_unconvertible_slow_tokenizer_is_returned_unchanged():
    tok = SPTokenizer()
    try:
        convert_slow_tokenizer(tok)
    except Exception:
        pass
    else:
        raise AssertionError("premise: transformers must fail to convert this tokenizer")

    assert tu.convert_to_fast_tokenizer(tok) is tok


if __name__ == "__main__":
    test_unconvertible_slow_tokenizer_is_returned_unchanged()
    print("ok")
