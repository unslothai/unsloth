# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Gemma 4 + padding-free on multi-device maps (#11952)."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from unsloth.trainer import (  # noqa: E402
    _block_gemma4_multidevice_padding_free,
    _model_spans_multiple_devices,
)
from unsloth.models._utils import _unsloth_align_batch_tensor_devices  # noqa: E402


class _FakeGemma4Model(torch.nn.Module):
    def __init__(self, device_map):
        super().__init__()
        self.hf_device_map = device_map
        self.config = SimpleNamespace(model_type = "gemma4")


def test_model_spans_multiple_devices():
    single = _FakeGemma4Model({"layer.0": "cuda:0", "layer.1": "cuda:0"})
    multi = _FakeGemma4Model({"layer.0": "cuda:0", "layer.1": "cuda:1"})
    assert not _model_spans_multiple_devices(single)
    assert _model_spans_multiple_devices(multi)


def test_block_gemma4_multidevice_padding_free():
    multi = _FakeGemma4Model({"a": "cuda:0", "b": "cuda:1"})
    single = _FakeGemma4Model({"a": "cuda:0", "b": "cuda:0"})
    types = ("gemma4",)
    assert _block_gemma4_multidevice_padding_free(multi, types)
    assert not _block_gemma4_multidevice_padding_free(single, types)
    assert not _block_gemma4_multidevice_padding_free(multi, ("llama",))


def test_align_batch_tensor_devices():
    inputs = {
        "input_ids": torch.tensor([[1, 2]], device = "cpu"),
        "packed_seq_lengths": torch.tensor([2], dtype = torch.int32),
        "position_ids": torch.tensor([[0, 1]]),
    }
    _unsloth_align_batch_tensor_devices(inputs)
    assert inputs["packed_seq_lengths"].device == inputs["input_ids"].device
    assert inputs["position_ids"].device == inputs["input_ids"].device
