# SPDX-License-Identifier: AGPL-3.0-only
# Unregistered towers got flash_attention_2: "Apertus1p5VisionTokenizerModel does not support Flash Attention 2 yet".
import sys
import types

import pytest
import unsloth  # noqa: F401
import transformers
from transformers import LlamaConfig, PretrainedConfig, PreTrainedModel

from unsloth.models import _utils


def _flash_available(monkeypatch):
    monkeypatch.setattr(_utils, "HAS_FLASH_ATTENTION", True)
    monkeypatch.setattr(_utils, "_get_flash_attention_disable_reason", lambda config: None)


def _composite(module_name):
    mod = types.ModuleType(module_name)

    class TowerConfig(PretrainedConfig):
        model_type = "unsloth_test_unregistered_tower"

    class CompositeConfig(PretrainedConfig):
        model_type = "unsloth_test_composite"
        sub_configs = {"text_config": LlamaConfig, "tower_config": TowerConfig}

        def __init__(
            self,
            text_config = None,
            tower_config = None,
            **kwargs,
        ):
            self.text_config = LlamaConfig(**(text_config or {}))
            self.tower_config = TowerConfig(**(tower_config or {}))
            super().__init__(**kwargs)

    class TowerPreTrainedModel(PreTrainedModel):
        config_class = TowerConfig
        _supports_sdpa = False

    class TowerModel(TowerPreTrainedModel):
        pass

    class CompositeForConditionalGeneration(PreTrainedModel):
        config_class = CompositeConfig
        _supports_flash_attn = True
        _supports_sdpa = True

    for klass in (
        TowerConfig,
        CompositeConfig,
        TowerPreTrainedModel,
        TowerModel,
        CompositeForConditionalGeneration,
    ):
        klass.__module__ = module_name
        setattr(mod, klass.__name__, klass)
    return mod, CompositeForConditionalGeneration, CompositeConfig()


@pytest.fixture
def composite(monkeypatch):
    name = "unsloth_test_fake_modeling_composite"
    mod, model_class, config = _composite(name)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod, model_class, config


def test_unregistered_tower_found_through_the_modeling_module(composite):
    mod, model_class, config = composite
    assert (
        _utils._sibling_model_class_for_config(model_class, config.tower_config) is mod.TowerModel
    )
    assert _utils._flash_unsupported_sub_configs(config, model_class) == {"tower_config": "eager"}


def test_without_model_class_behaviour_is_unchanged(composite):
    _, _, config = composite
    assert _utils._flash_unsupported_sub_configs(config) == {}


def test_resolve_scopes_flash_away_from_the_tower(monkeypatch, composite):
    _flash_available(monkeypatch)
    _, model_class, config = composite
    if not _utils._transformers_supports_attn_impl_mapping():
        pytest.skip("needs the Transformers attn_implementation mapping form")
    impl = _utils.resolve_attention_implementation(model_class, config, supports_sdpa = True)
    assert impl == {"": "flash_attention_2", "tower_config": "eager"}
    explicit = _utils.resolve_attention_implementation(
        model_class,
        type(config)(),
        requested_attn_implementation = "flash_attention_2",
        supports_sdpa = True,
    )
    assert explicit == {"": "flash_attention_2", "tower_config": "eager"}


def test_sibling_lookup_tolerates_unknown_module():
    class Orphan:
        __module__ = "unsloth_test_module_that_does_not_exist"

    assert _utils._sibling_model_class_for_config(Orphan, LlamaConfig()) is None


@pytest.mark.skipif(
    not hasattr(transformers, "Apertus1p5Config"), reason = "needs transformers with apertus1p5"
)
def test_apertus1p5_vision_tokenizer_kept_off_flash(monkeypatch):
    _flash_available(monkeypatch)
    from transformers.models.apertus1p5.modeling_apertus1p5 import (
        Apertus1p5ForConditionalGeneration,
    )

    config = transformers.Apertus1p5Config()
    unsupported = _utils._flash_unsupported_sub_configs(config, Apertus1p5ForConditionalGeneration)
    # README-pinned port: vision_tokenizer_config; upstream PR: vision_config.
    vision_fields = [k for k in unsupported if k.startswith("vision")]
    assert vision_fields and all(unsupported[k] == "eager" for k in vision_fields)
    assert "text_config" not in unsupported
