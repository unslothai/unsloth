# SPDX-License-Identifier: AGPL-3.0-only
"""The root dispatch hook runs where the input embedding lives, not on the first device in
the map, so attention_mask follows input_ids."""

from types import SimpleNamespace

import pytest
import torch


def _helper():
    from unsloth.models.vision import _align_root_hook_with_input_embeddings
    return _align_root_hook_with_input_embeddings


class _RemoteModel(SimpleNamespace):
    pass


# The Teacher's model class comes from the checkpoint's own modeling file.
_RemoteModel.__module__ = "transformers_modules.teacher.modeling_nemotron_h"


def _model(
    device_map,
    root_device,
    embedding_device,
    cls = _RemoteModel,
    config = None,
):
    embedding = SimpleNamespace(weight = SimpleNamespace(device = embedding_device))
    return cls(
        hf_device_map = device_map,
        _hf_hook = SimpleNamespace(execution_device = root_device),
        get_input_embeddings = lambda: embedding,
        config = config or SimpleNamespace(),
    )


def test_root_moves_to_the_embedding_device():
    align = _helper()
    model = _model(
        {"model.layers.1": 2, "model.embeddings": 1, "lm_head": 2, "model.layers.0": 0},
        0,
        torch.device("cuda", 1),
    )
    assert align(model) == torch.device("cuda", 1)
    assert model._hf_hook.execution_device == torch.device("cuda", 1)


def test_root_already_on_the_embedding_device_is_untouched():
    align = _helper()
    for root in (0, torch.device("cuda", 0), "cuda:0"):
        model = _model({"model.embeddings": 0, "lm_head": 1}, root, torch.device("cuda", 0))
        assert align(model) is None
        assert model._hf_hook.execution_device == root


def test_single_device_and_unhooked_models_are_untouched():
    align = _helper()
    model = _model({"": 0}, 0, torch.device("cuda", 1))
    assert align(model) is None
    model = _model({"model.embeddings": 0, "lm_head": 1}, 0, torch.device("cuda", 1))
    model._hf_hook = None
    assert align(model) is None
    model = _model({"model.embeddings": 0, "lm_head": 1}, 0, torch.device("cuda", 1))
    del model._hf_hook
    assert align(model) is None


def test_offloaded_or_meta_embedding_is_left_to_its_hook():
    align = _helper()
    for device in (torch.device("cpu"), torch.device("meta")):
        model = _model({"model.embeddings": "cpu", "lm_head": 0}, 0, device)
        assert align(model) is None
        assert model._hf_hook.execution_device == 0


def test_model_without_embedding_accessor_is_untouched():
    align = _helper()
    model = _model({"model.embeddings": 1, "lm_head": 0}, 0, torch.device("cuda", 1))

    def broken():
        raise NotImplementedError

    model.get_input_embeddings = broken
    assert align(model) is None


def test_native_and_multimodal_models_are_untouched():
    """Native code moves the mask itself; the root hook moves every input, so a vision or
    audio tower would get its tensors on the text card."""
    align = _helper()
    split = {"model.layers.1": 2, "model.embeddings": 1, "lm_head": 2, "model.layers.0": 0}
    native = _model(split, 0, torch.device("cuda", 1), cls = SimpleNamespace)
    assert align(native) is None
    assert native._hf_hook.execution_device == 0
    for name in ("vision_config", "audio_config"):
        config = SimpleNamespace(**{name: SimpleNamespace()})
        multimodal = _model(split, 0, torch.device("cuda", 1), config = config)
        assert align(multimodal) is None
        assert multimodal._hf_hook.execution_device == 0


def test_omni_config_with_nested_towers_is_untouched():
    """Omni checkpoints keep vision / audio under thinker_config; they are still multimodal."""
    align = _helper()
    split = {"model.layers.1": 2, "model.embeddings": 1, "lm_head": 2, "model.layers.0": 0}
    config = SimpleNamespace(thinker_config = SimpleNamespace(vision_config = SimpleNamespace()))
    omni = _model(split, 0, torch.device("cuda", 1), config = config)
    assert align(omni) is None
    assert omni._hf_hook.execution_device == 0


@pytest.mark.parametrize(
    "tower", ["vision_encoder_config", "audio_encoder_config", "encoder_config"]
)
def test_encoder_spelled_tower_configs_are_untouched(tower):
    """The same spellings _uses_flash_attention_for_generation already treats as non-language."""
    align = _helper()
    split = {"model.layers.1": 2, "model.embeddings": 1, "lm_head": 2, "model.layers.0": 0}
    multimodal = _model(
        split, 0, torch.device("cuda", 1), config = SimpleNamespace(**{tower: SimpleNamespace()})
    )
    assert align(multimodal) is None
    assert multimodal._hf_hook.execution_device == 0
