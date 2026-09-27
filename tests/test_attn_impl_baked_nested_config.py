# SPDX-License-Identifier: AGPL-3.0-only
import pytest
import unsloth  # noqa: F401
from transformers import PretrainedConfig

from unsloth.models import _utils


class _Llm(PretrainedConfig):
    model_type = "baked_llm"


class _Vision(PretrainedConfig):
    model_type = "baked_vision"


class _OmniLike(PretrainedConfig):
    model_type = "baked_omni"

    def __init__(
        self,
        attn_implementation = "flash_attention_2",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm_config = _Llm()
        self.vision_config = _Vision()
        self._attn_implementation = attn_implementation
        self.vision_config.use_flash_attn = "flash_attention" in attn_implementation
        self.llm_config._attn_implementation = attn_implementation


class _FlashOnlyModel:
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = False


@pytest.mark.parametrize("requested", [None, "sdpa", "eager"])
def test_non_flash_resolution_reaches_the_baked_decoder(monkeypatch, requested):
    monkeypatch.setattr(_utils, "HAS_FLASH_ATTENTION", False)
    config = _OmniLike()
    impl = _utils.resolve_attention_implementation(
        _FlashOnlyModel, config, requested_attn_implementation = requested, supports_sdpa = False
    )
    assert not str(impl).startswith("flash")
    assert config.llm_config._attn_implementation == impl
    assert config.vision_config.use_flash_attn is False


def test_flash_resolution_keeps_the_decoder_on_flash(monkeypatch):
    monkeypatch.setattr(_utils, "HAS_FLASH_ATTENTION", True)
    monkeypatch.setattr(_utils, "_get_flash_attention_disable_reason", lambda config: None)
    config = _OmniLike()
    impl = _utils.resolve_attention_implementation(_FlashOnlyModel, config, supports_sdpa = False)
    assert impl == "flash_attention_2"
    assert config.llm_config._attn_implementation == "flash_attention_2"
    assert config.vision_config.use_flash_attn is True


def test_nested_config_with_its_own_choice_is_left_alone():
    config = _OmniLike(attn_implementation = "sdpa")
    config.llm_config._attn_implementation = "eager"
    _utils._set_attn_impl(config, "flex_attention")
    assert config.llm_config._attn_implementation == "eager"
