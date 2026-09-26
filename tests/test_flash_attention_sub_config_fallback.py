# SPDX-License-Identifier: AGPL-3.0-only
# A plain "flash_attention_2" reaches every sub-config; LFM2-VL's SigLIP2 tower raises at init on it.
import pytest
import unsloth  # noqa: F401
import transformers

from unsloth.models import _utils

pytestmark = pytest.mark.skipif(
    not hasattr(transformers, "Lfm2VlConfig"), reason = "needs transformers with LFM2-VL"
)


def _flash_available(monkeypatch):
    monkeypatch.setattr(_utils, "HAS_FLASH_ATTENTION", True)
    monkeypatch.setattr(_utils, "_get_flash_attention_disable_reason", lambda config: None)


def _lfm2_vl():
    from transformers.models.lfm2_vl.modeling_lfm2_vl import Lfm2VlForConditionalGeneration
    return Lfm2VlForConditionalGeneration, transformers.Lfm2VlConfig()


def test_lfm2_vl_keeps_flash_off_the_siglip2_tower(monkeypatch):
    _flash_available(monkeypatch)
    model_class, config = _lfm2_vl()
    assert model_class._supports_flash_attn
    impl = _utils.resolve_attention_implementation(model_class, config, supports_sdpa = True)
    assert impl == {"": "flash_attention_2", "vision_config": "sdpa"}
    assert config.vision_config._attn_implementation == "sdpa"
    assert config.text_config._attn_implementation == "flash_attention_2"


def test_explicit_flash_request_is_scoped_too(monkeypatch):
    _flash_available(monkeypatch)
    model_class, config = _lfm2_vl()
    impl = _utils.resolve_attention_implementation(
        model_class,
        config,
        requested_attn_implementation = "flash_attention_2",
        supports_sdpa = True,
    )
    assert impl == {"": "flash_attention_2", "vision_config": "sdpa"}


def test_scoped_mapping_constructs_the_model(monkeypatch):
    _flash_available(monkeypatch)
    import torch
    from transformers import modeling_utils

    if not hasattr(modeling_utils, "lazy_import_flash_attention") or not hasattr(
        modeling_utils.PreTrainedModel, "_flash_attn_import_error"
    ):
        pytest.skip(reason = "flash_attn package checks not stubbable on this Transformers")
    # Stub only the package / kernel import so the per-class support check runs without flash_attn.
    monkeypatch.setattr(
        modeling_utils.PreTrainedModel, "_flash_attn_import_error", lambda self, **kwargs: None
    )
    monkeypatch.setattr(modeling_utils, "lazy_import_flash_attention", lambda *a, **k: None)
    model_class, config = _lfm2_vl()
    config.text_config.num_hidden_layers = 2
    config.text_config.layer_types = ["full_attention", "conv"]
    config.vision_config.num_hidden_layers = 1
    impl = _utils.resolve_attention_implementation(model_class, config, supports_sdpa = True)
    with torch.device("meta"):
        model_class._from_config(config, attn_implementation = impl, dtype = torch.bfloat16)
    with torch.device("meta"), pytest.raises(ValueError, match = "Flash Attention 2"):
        model_class._from_config(
            transformers.Lfm2VlConfig(),
            attn_implementation = "flash_attention_2",
            dtype = torch.bfloat16,
        )


def test_all_flash_capable_sub_models_keep_the_plain_string(monkeypatch):
    _flash_available(monkeypatch)
    from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration

    config = transformers.Gemma3Config()
    assert _utils._flash_unsupported_sub_configs(config) == {}
    assert _utils._scoped_flash_attention(config, True) == "flash_attention_2"
    assert Gemma3ForConditionalGeneration is not None


def test_text_only_config_is_unchanged(monkeypatch):
    _flash_available(monkeypatch)
    from transformers.models.lfm2.modeling_lfm2 import Lfm2ForCausalLM

    config = transformers.Lfm2Config()
    impl = _utils.resolve_attention_implementation(Lfm2ForCausalLM, config, supports_sdpa = True)
    assert impl == "flash_attention_2"


@pytest.mark.parametrize(
    "unsupported, expected",
    [({"vision_config": "sdpa"}, "sdpa"), ({"vision_config": "eager"}, "eager")],
)
def test_without_the_mapping_form_the_global_fallback_suits_every_tower(
    monkeypatch, unsupported, expected
):
    monkeypatch.setattr(_utils, "_transformers_supports_attn_impl_mapping", lambda: False)
    monkeypatch.setattr(_utils, "_flash_unsupported_sub_configs", lambda config: unsupported)
    assert _utils._scoped_flash_attention(object(), True) == expected
    assert _utils._scoped_flash_attention(object(), False) == "eager"
