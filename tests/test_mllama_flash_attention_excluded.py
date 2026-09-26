# SPDX-License-Identifier: AGPL-3.0-only
# Mllama declares flash support, but MllamaVisionAttention and MllamaTextCrossAttention have no
# is_causal, which Transformers' flash path reads, so a flash-attn install broke Llama 3.2 Vision.
import pytest
import torch
import unsloth  # noqa: F401
import transformers

from unsloth.models import _utils

pytestmark = pytest.mark.skipif(
    not hasattr(transformers, "MllamaConfig"), reason = "needs transformers with Mllama"
)


def _tiny_mllama():
    from transformers.models.mllama.modeling_mllama import MllamaForConditionalGeneration

    config = transformers.MllamaConfig()
    v = config.vision_config
    v.hidden_size, v.attention_heads, v.intermediate_size = 64, 4, 128
    v.num_hidden_layers, v.num_global_layers, v.vision_output_dim = 2, 1, 128
    v.image_size, v.patch_size, v.max_num_tiles = 56, 14, 1
    v.intermediate_layers_indices, v.supported_aspect_ratios = [0], [[1, 1]]
    t = config.text_config
    t.hidden_size, t.num_attention_heads, t.num_key_value_heads, t.intermediate_size = 64, 4, 2, 128
    t.num_hidden_layers, t.cross_attention_layers, t.vocab_size, t.pad_token_id = 3, [1], 256, 0
    config.image_token_index = 255
    return MllamaForConditionalGeneration, config


def test_mllama_resolves_off_flash_even_when_flash_is_available(monkeypatch):
    monkeypatch.setattr(_utils, "HAS_FLASH_ATTENTION", True)
    model_class, config = _tiny_mllama()
    assert model_class._supports_flash_attn
    impl = _utils.resolve_attention_implementation(model_class, config, supports_sdpa = True)
    assert impl in ("sdpa", "eager")
    assert _utils._get_flash_attention_disable_reason(config) is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_mllama_image_forward_runs_on_the_resolved_implementation(monkeypatch):
    monkeypatch.setattr(_utils, "HAS_FLASH_ATTENTION", True)
    model_class, config = _tiny_mllama()
    impl = _utils.resolve_attention_implementation(model_class, config, supports_sdpa = True)
    model = (
        model_class._from_config(config, attn_implementation = impl, dtype = torch.bfloat16)
        .cuda()
        .eval()
    )
    ids = torch.tensor([[255, 5, 6, 7, 8, 9]], device = "cuda")
    pixels = torch.randn(1, 1, 1, 3, 56, 56, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        out = model(
            input_ids = ids,
            pixel_values = pixels,
            aspect_ratio_ids = torch.tensor([[1]], device = "cuda"),
            aspect_ratio_mask = torch.ones(1, 1, 1, device = "cuda", dtype = torch.long),
            cross_attention_mask = torch.ones(1, 6, 1, 1, device = "cuda", dtype = torch.long),
        )
    assert torch.isfinite(out.logits.float()).all()
