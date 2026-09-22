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
from unsloth.import_fixes import (
    fix_transformers_validate_rope_ignore_keys,
    fix_transformers_is_torch_fx_available,
)


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
    # A model-specific key, as DeepSeek-style remote configs pass. "rope_type" is required, so
    # 5.0 to 5.3 themselves raise KeyError when it is ignored; it is not a valid probe.
    config.validate_rope(ignore_keys = {"mscale", "mscale_all_dim"})
    config.validate_rope({"mscale", "mscale_all_dim"})  # the 5.0 positional form
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


def test_the_mlx_branch_installs_the_fix_too():
    import pathlib

    source = (pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "__init__.py").read_text(
        encoding = "utf-8"
    )
    mlx_branch = source[source.find("if _IS_MLX:") :]
    assert "fix_transformers_validate_rope_ignore_keys" in mlx_branch


def test_is_torch_fx_available_is_importable_from_transformers_utils_after_the_fix():
    """Ling-2.6-flash's hub modeling file does `from transformers.utils import
    is_torch_fx_available`; transformers 5 removed the symbol."""
    fix_transformers_is_torch_fx_available()
    from transformers.utils import is_torch_fx_available
    import transformers.utils.import_utils as import_utils

    # The 4.x definition: whatever torch availability says (False on a torch-less MLX host).
    assert is_torch_fx_available() == import_utils.is_torch_available()
    assert import_utils.is_torch_fx_available() == import_utils.is_torch_available()
    fix_transformers_is_torch_fx_available()  # idempotent
    assert import_utils.is_torch_fx_available is import_utils.is_torch_fx_available


def test_the_mlx_branch_installs_the_fx_shim_too():
    import pathlib
    source = (pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "__init__.py").read_text(
        encoding = "utf-8"
    )
    assert "fix_transformers_is_torch_fx_available" in source[source.find("if _IS_MLX:") :]


def test_a_config_with_its_own_validator_accepts_ignore_keys_too():
    """Phi3Config (and a remote subclass of it) resolves validate_rope to its own override,
    not the mixin's; classes defined after the fix are covered by the subclass hook."""
    fix_transformers_validate_rope_ignore_keys()
    from transformers import Phi3Config

    if not hasattr(Phi3Config, "validate_rope"):
        pytest.skip("this transformers has no validate_rope (4.x)")
    config = Phi3Config(
        hidden_size = 32,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        intermediate_size = 32,
        vocab_size = 16,
    )
    config.validate_rope(ignore_keys = {"mscale", "mscale_all_dim"})

    class LaterConfig(Phi3Config):  # defined after the fix, like a remote configuration
        model_type = "later_phi3_for_test"

        def validate_rope(self):
            return "own"

    assert LaterConfig.validate_rope(config, ignore_keys = {"x"}) == "own"


def test_ignore_keys_keep_their_meaning_on_a_validator_without_the_parameter():
    """5.4 moved ``ignore_keys`` onto ``ignore_keys_at_rope_validation``; dropping the keys
    instead brings back the "Unrecognized keys" warning 5.0 to 5.3 suppressed for them."""
    import logging

    from transformers import LlamaConfig

    fix_transformers_validate_rope_ignore_keys()
    mixin = _mixin()
    if not getattr(mixin.__dict__.get("validate_rope"), "_unsloth_ignore_keys", False):
        pytest.skip("this transformers still takes ignore_keys itself")
    config = LlamaConfig(
        hidden_size = 8,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        intermediate_size = 8,
        vocab_size = 16,
        rope_scaling = {"rope_type": "linear", "factor": 2.0, "model_specific_key": 1},
    )
    records = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Collect()
    rope_logger = logging.getLogger("transformers.modeling_rope_utils")
    rope_logger.addHandler(handler)
    try:
        before = set(config.ignore_keys_at_rope_validation or ())
        for call in (
            lambda: config.validate_rope(ignore_keys = {"model_specific_key"}),
            lambda: config.validate_rope({"model_specific_key"}),
        ):
            records.clear()
            call()
            assert not [m for m in records if "Unrecognized keys" in m], records
            # scoped to the call, as the 5.0 parameter was
            assert set(config.ignore_keys_at_rope_validation or ()) == before
        records.clear()
        config.validate_rope()
        assert [m for m in records if "Unrecognized keys" in m]  # the control: the key is unknown
    finally:
        rope_logger.removeHandler(handler)
