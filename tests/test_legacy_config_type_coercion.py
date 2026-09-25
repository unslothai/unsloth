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
"""Llama 4 configs saved on 4.51 (attn_temperature_tuning: 4) must load on transformers 5.4+."""

import json
import logging
import os
import pathlib

import pytest

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
transformers = pytest.importorskip("transformers")

import unsloth  # noqa: E402,F401  (installs the fix, as a user's import would)
from packaging.version import Version  # noqa: E402

TRANSFORMERS_5 = Version(transformers.__version__) >= Version("5.0.0")


def _strict_configs():
    from transformers.configuration_utils import PretrainedConfig
    return isinstance(getattr(PretrainedConfig, "__validators__", None), dict)


needs_strict = pytest.mark.skipif(
    not (TRANSFORMERS_5 and _strict_configs()),
    reason = "configs are not @strict dataclasses on this transformers (4.x, 5.0 to 5.3)",
)


def _llama4_config_dict(attn_temperature_tuning = 4):
    return {
        "architectures": ["Llama4ForConditionalGeneration"],
        "model_type": "llama4",
        "boi_token_index": 200080,
        "eoi_token_index": 200081,
        "image_token_index": 200092,
        "tie_word_embeddings": False,
        "transformers_version": "4.51.0",
        "text_config": {
            "model_type": "llama4_text",
            "attn_temperature_tuning": attn_temperature_tuning,
            "attention_chunk_size": 64,
            "floor_scale": 8192,
            "attn_scale": 0.1,
            "hidden_size": 64,
            "head_dim": 16,
            "intermediate_size": 64,
            "intermediate_size_mlp": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_hidden_layers": 2,
            "num_local_experts": 2,
            "num_experts_per_tok": 1,
            "interleave_moe_layer_step": 1,
            "vocab_size": 256,
            "max_position_embeddings": 256,
            "rope_theta": 500000.0,
            "rope_scaling": {
                "rope_type": "llama3",
                "factor": 8.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 64,
            },
            "use_qk_norm": True,
            "no_rope_layers": [],
        },
        "vision_config": {
            "model_type": "llama4_vision_model",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "image_size": 28,
            "patch_size": 14,
            "vision_output_dim": 32,
            "projector_input_dim": 32,
            "projector_output_dim": 32,
        },
    }


def _write(tmp_path, config):
    (tmp_path / "config.json").write_text(json.dumps(config), encoding = "utf-8")
    return str(tmp_path)


def test_llama4_4x_config_loads_through_autoconfig(tmp_path):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(_write(tmp_path, _llama4_config_dict(4)))
    value = config.text_config.attn_temperature_tuning
    if TRANSFORMERS_5 and _strict_configs():
        assert value is True
    else:
        # 4.x and 5.0 to 5.3 load the int unchanged; the fix must not touch them.
        assert value == 4 and type(value) is int


def test_llama4_zero_means_false(tmp_path):
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(_write(tmp_path, _llama4_config_dict(0)))
    assert not config.text_config.attn_temperature_tuning


def test_llama4_bool_config_is_untouched(tmp_path):
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(_write(tmp_path, _llama4_config_dict(True)))
    assert config.text_config.attn_temperature_tuning is True


@needs_strict
def test_direct_construction_and_save_round_trip(tmp_path):
    from transformers import Llama4Config, Llama4TextConfig

    assert Llama4TextConfig(attn_temperature_tuning = 4).attn_temperature_tuning is True
    config = Llama4Config(text_config = _llama4_config_dict(4)["text_config"])
    assert config.text_config.attn_temperature_tuning is True
    config.save_pretrained(str(tmp_path))
    saved = json.loads((tmp_path / "config.json").read_text(encoding = "utf-8"))
    assert saved["text_config"].get("attn_temperature_tuning", True) is True


@needs_strict
def test_lossless_conversions():
    from transformers import LlamaConfig

    config = LlamaConfig(
        hidden_size = 64.0,  # integral float for int
        initializer_range = 1,  # int for float
        mlp_bias = 1,  # 0/1 for bool
        tie_word_embeddings = 0,
        num_attention_heads = 4,
        num_key_value_heads = 2,
    )
    assert config.hidden_size == 64 and type(config.hidden_size) is int
    assert config.initializer_range == 1.0 and type(config.initializer_range) is float
    assert config.mlp_bias is True
    assert config.tie_word_embeddings is False


@needs_strict
def test_values_transformers_accepts_are_not_touched():
    from transformers import LlamaConfig

    # attention_dropout is annotated int | float | None: an int is valid and must stay an int.
    config = LlamaConfig(attention_dropout = 0)
    assert type(config.attention_dropout) is int


@needs_strict
@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_size": "64"},  # string for a number
        {"hidden_size": 64.5},  # fractional float for int
        {"mlp_bias": 2},  # non 0/1 int for a bool that was never a truthiness int
        {"mlp_bias": "true"},  # string for bool
        {"problem_type": "not_a_problem_type"},  # string outside a Literal
    ],
)
def test_real_errors_still_raise(kwargs):
    from transformers import LlamaConfig
    with pytest.raises(Exception) as info:
        LlamaConfig(**kwargs)
    assert "Validation" in type(info.value).__name__ or isinstance(
        info.value, (TypeError, ValueError)
    )


@needs_strict
def test_other_int_for_llama4_is_a_real_error_only_outside_the_allowlist():
    from transformers import Llama4TextConfig
    with pytest.raises(Exception):
        Llama4TextConfig(use_qk_norm = 4)
    with pytest.raises(Exception):
        Llama4TextConfig(attn_temperature_tuning = "4")


@needs_strict
def test_classes_defined_after_import_are_covered():
    from huggingface_hub.dataclasses import strict
    from transformers.configuration_utils import PretrainedConfig

    @strict
    class LateConfig(PretrainedConfig):
        model_type = "unsloth_late_probe"
        flag: bool = False
        sizes: tuple[int, ...] = (1,)
        scale: float = 1.0

    config = LateConfig(flag = 1, sizes = [2, 3], scale = 2)
    assert config.flag is True
    assert config.sizes == (2, 3)
    assert type(config.scale) is float
    with pytest.raises(Exception):
        LateConfig(flag = 3)


@needs_strict
def test_coercion_is_logged_once(caplog):
    from transformers import Llama4TextConfig

    import unsloth.import_fixes as import_fixes

    import_fixes._legacy_config_coercions_logged.discard(
        ("Llama4TextConfig", "attn_temperature_tuning", "int")
    )
    with caplog.at_level(logging.WARNING, logger = "unsloth.import_fixes"):
        Llama4TextConfig(attn_temperature_tuning = 4)
        Llama4TextConfig(attn_temperature_tuning = 4)
    hits = [r for r in caplog.records if "attn_temperature_tuning" in r.getMessage()]
    assert len(hits) == 1


@needs_strict
def test_fix_is_idempotent_and_original_reachable():
    from transformers import LlamaConfig
    from transformers.configuration_utils import PretrainedConfig

    from unsloth.import_fixes import (
        _LEGACY_CONFIG_INIT_FLAG,
        fix_transformers5_legacy_config_types,
    )

    init = LlamaConfig.__dict__["__init__"]
    hook = PretrainedConfig.__dict__["__init_subclass__"]
    assert getattr(init, _LEGACY_CONFIG_INIT_FLAG, False)
    fix_transformers5_legacy_config_types()
    assert LlamaConfig.__dict__["__init__"] is init
    assert PretrainedConfig.__dict__["__init_subclass__"] is hook
    assert getattr(init, "__wrapped__", None) is not None


@pytest.mark.skipif(TRANSFORMERS_5, reason = "checks the 4.x no-op")
def test_noop_on_transformers_4():
    from transformers import LlamaConfig

    from unsloth.import_fixes import (
        _LEGACY_CONFIG_INIT_FLAG,
        fix_transformers5_legacy_config_types,
    )

    fix_transformers5_legacy_config_types()
    assert not getattr(LlamaConfig.__dict__.get("__init__"), _LEGACY_CONFIG_INIT_FLAG, False)


def test_the_mlx_branch_installs_the_fix_too():
    source = (pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "__init__.py").read_text(
        encoding = "utf-8"
    )
    mlx_branch = source[source.find("if _IS_MLX:") :]
    assert "fix_transformers5_legacy_config_types" in mlx_branch


@needs_strict
def test_fields_strict_does_not_validate_are_not_touched():
    """A config that already loads must load byte-identically."""
    from transformers import LlamaConfig
    from transformers.configuration_utils import PretrainedConfig

    class NotStrictConfig(PretrainedConfig):
        model_type = "unsloth_not_strict_probe"
        flag: bool = False
        scale: float = 1.0
        sizes: tuple[int, ...] = (1,)

    config = NotStrictConfig(flag = 1, scale = 2, sizes = [2, 3])
    assert type(config.flag) is int and type(config.scale) is int and type(config.sizes) is list

    class RetypedConfig(LlamaConfig):
        model_type = "unsloth_retyped_probe"
        hidden_size: float = 64.0

    assert type(RetypedConfig(hidden_size = 64).hidden_size) is int
