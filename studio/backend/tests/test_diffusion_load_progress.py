# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import itertools

import pytest

from core.training.diffusion_load_progress import report_model_load_progress


def test_real_sharded_model_counts_loaded_tensors_and_preserves_weights(tmp_path, monkeypatch):
    import torch
    from diffusers import FluxTransformer2DModel
    from diffusers.models import model_loading_utils

    ticks = itertools.count()
    monkeypatch.setattr("core.training.diffusion_load_progress.time.monotonic", lambda: next(ticks))
    model = FluxTransformer2DModel(
        patch_size = 1,
        in_channels = 4,
        num_layers = 1,
        num_single_layers = 1,
        attention_head_dim = 8,
        num_attention_heads = 2,
        joint_attention_dim = 16,
        pooled_projection_dim = 8,
        axes_dims_rope = (2, 2, 4),
    )
    model.save_pretrained(tmp_path, max_shard_size = "5KB")
    expected = model.state_dict()
    events = []
    original = model_loading_utils.load_model_dict_into_meta
    with report_model_load_progress(events.append):
        loaded = FluxTransformer2DModel.from_pretrained(tmp_path, low_cpu_mem_usage = True)
    assert model_loading_utils.load_model_dict_into_meta is original
    weights = [e for e in events if e["unit"] == "tensors"]
    assert weights[0]["done"] == 0
    assert weights[-1]["done"] == weights[-1]["total"] == len(expected)
    assert {e["total"] for e in weights} == {len(expected)}
    assert [e["done"] for e in weights] == sorted(e["done"] for e in weights)
    assert any(0 < e["done"] < e["total"] for e in weights)
    for key, value in expected.items():
        assert torch.equal(loaded.state_dict()[key], value), key


@pytest.mark.parametrize("fail", [False, True])
def test_quantized_tensor_is_counted_only_after_success_and_hooks_restore(fail):
    import torch
    from diffusers.models import model_loading_utils as loading

    model = torch.nn.Linear(3, 2, device = "meta")
    values = torch.nn.Linear(3, 2).state_dict()
    original_set, original_meta = (
        loading.set_module_tensor_to_device,
        loading.load_model_dict_into_meta,
    )
    events = []

    class Quantizer:
        pre_quantized = False

        def check_if_quantized_param(self, model, value, name, state, **kwargs):
            return name == "weight"

        def create_quantized_param(self, model, value, name, device, *args, **kwargs):
            assert events[-1]["done"] == 0  # No credit for merely starting conversion.
            if fail:
                raise RuntimeError("conversion failed")
            original_set(model, name, device, value = value)

    try:
        with report_model_load_progress(events.append):
            loading.load_model_dict_into_meta(
                model,
                values,
                dtype = torch.float32,
                device_map = {"": "cpu"},
                hf_quantizer = Quantizer(),
            )
    except RuntimeError as exc:
        assert fail and str(exc) == "conversion failed"
    else:
        assert not fail
    assert loading.set_module_tensor_to_device is original_set
    assert loading.load_model_dict_into_meta is original_meta
    assert events[-1]["done"] == (0 if fail else 2)
    assert events[-1]["total"] == 2


def test_component_counter_works_when_library_bars_are_disabled():
    from diffusers.utils import logging

    original = logging.tqdm
    enabled = logging.is_progress_bar_enabled()
    logging.disable_progress_bar()
    events = []
    try:
        with report_model_load_progress(events.append):
            bar = logging.tqdm(
                ["encoder", "vae"], desc = "Loading pipeline components...", disable = True
            )
            iterator = iter(bar)
            assert next(iterator) == "encoder"
            assert events[-1]["done"] == 0
            assert next(iterator) == "vae"
            iterator.close()  # The unfinished VAE must not be credited.
            assert bar.n == 1
            assert not any(e["done"] == 2 for e in events)
    finally:
        if enabled:
            logging.enable_progress_bar()
    assert logging.tqdm is original


def test_no_callback_leaves_library_untouched():
    from diffusers.models import model_loading_utils
    original = model_loading_utils.load_model_dict_into_meta
    with report_model_load_progress(None):
        assert model_loading_utils.load_model_dict_into_meta is original


def test_unsupported_loader_api_falls_back_without_blocking_training(monkeypatch):
    from diffusers.models import model_loading_utils as loading

    original = loading.load_model_dict_into_meta
    monkeypatch.delattr(loading, "set_module_tensor_to_device")
    events = []
    with report_model_load_progress(events.append):
        assert loading.load_model_dict_into_meta is original
    assert events == []
