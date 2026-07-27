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

"""Regression test: the IGNORED_TOKENIZER_NAMES guard must match case-insensitively."""

import types
from unittest.mock import patch

import unsloth.tokenizer_utils as tu


class _Tok:
    def __init__(self, kind):
        self.kind = kind


def _fake_auto_tokenizer():
    # _load_correct_tokenizer loads a slow tokenizer (from_slow = True) and a fast one;
    # return distinguishable stand-ins so we can see which one is returned.
    def from_pretrained(name, **kwargs):
        return _Tok("slow" if kwargs.get("from_slow") else "fast")

    return types.SimpleNamespace(from_pretrained = from_pretrained)


def test_ignored_tokenizer_name_matched_case_insensitively():
    # IGNORED_TOKENIZER_NAMES is stored fully lowercased, but users pass canonical
    # mixed-case ids. The names in this list (e.g. the Qwen2.5-Coder tokenizers) must be
    # returned untouched; before the fix the mixed-case name never matched, so it fell
    # through into the slow/fast reconciliation + conversion path.
    name = "unsloth/Qwen2.5-Coder-7B-Instruct"  # name.lower() is in IGNORED_TOKENIZER_NAMES
    assert name.lower() in tu.IGNORED_TOKENIZER_NAMES  # guard the test's own premise

    with (
        patch.object(tu, "AutoTokenizer", _fake_auto_tokenizer()),
        patch.object(tu, "assert_same_tokenization", lambda a, b: False),
        patch.object(tu, "convert_to_fast_tokenizer", lambda slow, **k: _Tok("converted")),
    ):
        result = tu._load_correct_tokenizer(name, fix_tokenizer = True)

    assert result.kind == "fast", (
        "an ignored tokenizer (matched case-insensitively) must be returned unchanged, "
        f"but it fell through to conversion (got {result.kind!r})"
    )


if __name__ == "__main__":
    test_ignored_tokenizer_name_matched_case_insensitively()
    print("ok")
