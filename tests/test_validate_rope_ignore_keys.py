# SPDX-License-Identifier: AGPL-3.0-only
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
"""Remote configuration code written against transformers 5.0 calls
``validate_rope(ignore_keys = ...)``; 5.1 dropped the parameter, so such a config
died in AutoConfig.from_pretrained before a weight was read."""

import inspect

import pytest

transformers = pytest.importorskip("transformers")
from unsloth.import_fixes import fix_transformers_validate_rope_ignore_keys


def _mixin():
    try:
        from transformers.modeling_rope_utils import RotaryEmbeddingConfigMixin
    except Exception:
        pytest.skip("no RotaryEmbeddingConfigMixin on this transformers")
    return RotaryEmbeddingConfigMixin


def test_validate_rope_accepts_ignore_keys_after_the_fix():
    from transformers import LlamaConfig

    fix_transformers_validate_rope_ignore_keys()
    mixin = _mixin()
    assert "ignore_keys" in inspect.signature(mixin.validate_rope).parameters or getattr(
        mixin.__dict__.get("validate_rope"), "_unsloth_ignore_keys", False
    )
    config = LlamaConfig(
        hidden_size = 8,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        intermediate_size = 8,
        vocab_size = 16,
    )
    # what the 5.0-era remote code does
    config.validate_rope(ignore_keys = {"rope_type"})
    config.validate_rope()


def test_the_fix_is_idempotent_and_keeps_the_original_reachable():
    fix_transformers_validate_rope_ignore_keys()
    mixin = _mixin()
    first = mixin.__dict__["validate_rope"]
    fix_transformers_validate_rope_ignore_keys()
    assert mixin.__dict__["validate_rope"] is first
    original = getattr(first, "__wrapped__", None)
    if original is not None:
        assert "ignore_keys" not in inspect.signature(original).parameters
