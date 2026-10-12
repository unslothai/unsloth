# SPDX-License-Identifier: AGPL-3.0-only
"""Phi-4-reasoning-vision generate() on transformers 5: "'DynamicCache' object is not subscriptable"."""

import pytest
import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.cache_utils import DynamicCache

from unsloth.models.remote_code_shims import apply_remote_code_shims


class _Config(PretrainedConfig):
    model_type = "phi4-siglip-stub"


def _remote_class():
    class Phi4ForCausalLMVStub(PreTrainedModel):
        config_class = _Config

        def __init__(self, config):
            super().__init__(config)
            self.embed_tokens = torch.nn.Embedding(8, 4)

        def get_input_embeddings(self):
            return self.embed_tokens

        def forward(
            self,
            input_ids = None,
            labels = None,
        ):
            return None

        def prepare_inputs_labels_for_multimodal(
            self, input_ids, position_ids, attention_mask, past_key_values, labels, images
        ):
            if past_key_values is not None and images is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                            dtype = attention_mask.dtype,
                            device = attention_mask.device,
                        ),
                    ),
                    dim = 1,
                )
                position_ids = torch.sum(attention_mask, dim = 1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

    Phi4ForCausalLMVStub.__module__ = "transformers_modules.microsoft.phi4v.modeling_phi4_visionr"
    return Phi4ForCausalLMVStub


def _cache(past_len = 7):
    cache = DynamicCache()
    for layer in range(2):
        cache.update(torch.zeros(1, 2, past_len, 4), torch.zeros(1, 2, past_len, 4), layer)
    return cache


def _decode_step(model, cache, positional):
    args = (torch.ones(1, 1, dtype = torch.long), None, torch.ones(1, 3, dtype = torch.long))
    if positional:
        return model.prepare_inputs_labels_for_multimodal(*args, cache, None, torch.zeros(1))
    return model.prepare_inputs_labels_for_multimodal(
        *args,
        past_key_values = cache,
        labels = None,
        images = torch.zeros(1),
    )


@pytest.mark.skipif(
    hasattr(DynamicCache, "__getitem__"), reason = "transformers 4.x Cache is subscriptable"
)
def test_unrepaired_remote_code_fails_on_transformers_5():
    model = _remote_class()(_Config())
    with pytest.raises(TypeError, match = "not subscriptable"):
        _decode_step(model, _cache(), positional = True)


@pytest.mark.parametrize("positional", [True, False])
def test_repaired_decode_step_extends_mask_to_cache_length(positional):
    cls = _remote_class()
    model = cls(_Config())
    repaired = apply_remote_code_shims(model)
    assert f"{cls.__name__}.prepare_inputs_labels_for_multimodal" in repaired
    assert model._supports_static_cache is False
    cache = _cache(past_len = 7)
    _, position_ids, attention_mask, returned, _, _ = _decode_step(model, cache, positional)
    assert returned is cache
    assert attention_mask.shape == (1, 8)
    assert position_ids.tolist() == [[7]]
    assert f"{cls.__name__}.prepare_inputs_labels_for_multimodal" not in apply_remote_code_shims(
        model
    )


def test_non_remote_class_is_untouched():
    cls = _remote_class()
    cls.__module__ = "my_package.models"
    model = cls(_Config())
    assert not any(
        "prepare_inputs_labels_for_multimodal" in r for r in apply_remote_code_shims(model)
    )


def test_cache_api_remote_prep_keeps_real_cache():
    cls = _remote_class()

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images
    ):
        past = past_key_values.get_seq_length() if past_key_values is not None else 0
        return input_ids, position_ids, attention_mask, past_key_values, past, labels

    cls.prepare_inputs_labels_for_multimodal = prepare_inputs_labels_for_multimodal
    model = cls(_Config())
    assert not any(
        "prepare_inputs_labels_for_multimodal" in r for r in apply_remote_code_shims(model)
    )
    cache = _cache(past_len = 5)
    output = model.prepare_inputs_labels_for_multimodal(None, None, None, cache, None, None)
    assert output[3] is cache and output[4] == 5
