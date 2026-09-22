"""The root dispatch hook runs where the input embedding lives.

accelerate picks the root execution device as the first member of the SET of devices in the
map. The planner put the Nemotron Teacher's embedding on cuda:1 (head on cuda:2, a few
blocks on cuda:0), so every batch went to cuda:0 first, the embedding hook carried input_ids
on, and attention_mask stayed behind. The hub `_update_causal_mask` then raised "Expected all
tensors to be on the same device, but found at least two devices, cuda:1 and cuda:0".
"""
from types import SimpleNamespace

import pytest
import torch


def _helper():
    from unsloth.models.vision import _align_root_hook_with_input_embeddings
    return _align_root_hook_with_input_embeddings


def _model(device_map, root_device, embedding_device):
    embedding = SimpleNamespace(weight = SimpleNamespace(device = embedding_device))
    return SimpleNamespace(
        hf_device_map = device_map,
        _hf_hook = SimpleNamespace(execution_device = root_device),
        get_input_embeddings = lambda: embedding,
    )


def test_root_moves_to_the_embedding_device():
    align = _helper()
    model = _model({"model.layers.1": 2, "model.embeddings": 1, "lm_head": 2, "model.layers.0": 0}, 0, torch.device("cuda", 1))
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
