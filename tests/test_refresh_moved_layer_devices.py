# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A model loaded with device_map={"": 1} is published on cuda:1 at load; accelerate's prepare then moves a
single-device model to cuda:0, and fast decode moved activations back to cuda:1 ("cuda:1 and cuda:0" at
`X *= self.weight`). Decode now republishes when a layer's weights left the published device."""

from __future__ import annotations

import pathlib
import re

import pytest
import torch

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
DECODE_LOOPS = (
    "unsloth/models/llama.py",
    "unsloth/models/cohere.py",
    "unsloth/models/gemma.py",
    "unsloth/models/gemma2.py",
    "unsloth/models/granite.py",
)


class _Layer(torch.nn.Module):
    def __init__(self, published):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2), requires_grad = False)
        self._per_layer_device = published
        self._per_layer_device_index = (
            published.index if published.index is not None else published.type
        )


def _verify_and_set_device():
    try:
        from unsloth_zoo.patching_utils import verify_and_set_device
    except Exception as exception:
        pytest.skip(f"unsloth_zoo not importable: {exception}")
    return verify_and_set_device


def test_a_layer_moved_to_another_index_is_republished():
    _verify_and_set_device()
    from unsloth.models._utils import per_layer_device, refresh_moved_layer_devices

    # CPU stands in for an accelerator: published cpu:1 (the load-time card), weights now on cpu (index None).
    layers = torch.nn.ModuleList([_Layer(torch.device("cpu", 1)) for _ in range(3)])
    assert per_layer_device(layers[1])[0] == torch.device("cpu", 1)  # stale, and memoised
    refresh_moved_layer_devices(layers)
    for layer in layers:
        assert layer._per_layer_device == torch.device("cpu")
        assert per_layer_device(layer)[0] == torch.device("cpu")


def test_an_up_to_date_layer_is_left_alone():
    from unsloth.models._utils import refresh_moved_layer_devices

    layers = torch.nn.ModuleList([_Layer(torch.device("cpu")) for _ in range(2)])
    published = [layer._per_layer_device for layer in layers]
    refresh_moved_layer_devices(layers)
    assert [layer._per_layer_device for layer in layers] == published
    assert all(a is b for a, b in zip((layer._per_layer_device for layer in layers), published))


def test_block_swap_and_other_device_types_are_left_alone():
    from unsloth.models._utils import refresh_moved_layer_devices

    # Block swap / offload park weights on CPU while the published device is the accelerator they run on.
    layers = torch.nn.ModuleList([_Layer(torch.device("meta")) for _ in range(2)])
    refresh_moved_layer_devices(layers)
    assert all(layer._per_layer_device == torch.device("meta") for layer in layers)
    layers = torch.nn.ModuleList([_Layer(torch.device("cpu", 1)) for _ in range(2)])
    refresh_moved_layer_devices(layers, block_swap = object())
    assert all(layer._per_layer_device == torch.device("cpu", 1) for layer in layers)


@pytest.mark.parametrize("path", DECODE_LOOPS)
def test_every_decode_loop_refreshes_before_reading_layer_devices(path):
    source = (REPOSITORY_ROOT / path).read_text()
    refresh = source.find("refresh_moved_layer_devices(self.model.layers, block_swap)")
    read = source.find("per_layer_device(decoder_layer)")
    assert 0 <= refresh < read, path
    assert re.search(
        r"^from \._utils import .*refresh_moved_layer_devices", source, re.MULTILINE
    ), path
