# SPDX-License-Identifier: Apache-2.0
"""#3607: hf_device_map normalization for accelerate >= 0.34.1 compatibility.

Accelerate's prepare_model() calls torch.device(device.index) on every hf_device_map
entry. A bare torch.device('cuda') (index=None) raises TypeError:
    device() received an invalid combination of arguments - got (NoneType)

_normalize_hf_device_map in-place converts bare CUDA devices to torch.device('cuda', 0).
"""
import torch

from unsloth.models.vision import _normalize_hf_device_map


def test_none_is_noop():
    """Passing None must not crash."""
    _normalize_hf_device_map(None)


def test_empty_is_noop():
    _normalize_hf_device_map({})


def test_bare_cuda_gets_explicit_index():
    d = {"": torch.device("cuda")}
    _normalize_hf_device_map(d)
    assert d[""] == torch.device("cuda", 0)
    assert d[""].index == 0


def test_already_explicit_index_unchanged():
    d = {"": torch.device("cuda", 0)}
    original = d[""]
    _normalize_hf_device_map(d)
    assert d[""] is original  # no replacement


def test_cpu_string_unchanged():
    d = {"": "cpu"}
    _normalize_hf_device_map(d)
    assert d[""] == "cpu"


def test_multi_device_keeps_indices():
    d = {
        "": torch.device("cuda", 0),
        "layer.0": torch.device("cuda", 1),
    }
    _normalize_hf_device_map(d)
    assert d[""] == torch.device("cuda", 0)
    assert d["layer.0"] == torch.device("cuda", 1)


def test_bare_cuda_in_multi_device_map():
    """Even with multiple devices, bare cuda must get an index."""
    d = {
        "": torch.device("cuda"),
        "layer.0": torch.device("cuda", 1),
    }
    _normalize_hf_device_map(d)
    assert d[""].index == 0


def test_meta_device_passes_through():
    d = {"": torch.device("meta")}
    _normalize_hf_device_map(d)
    assert d[""].type == "meta"


def test_bare_cuda_string_is_normalized():
    d = {"": "cuda"}
    _normalize_hf_device_map(d)
    assert d[""] == torch.device("cuda", 0)


def test_cuda_colon_0_string_is_unchanged():
    """A string 'cuda:0' is NOT the same as bare 'cuda' — it already has an index."""
    d = {"": "cuda:0"}
    _normalize_hf_device_map(d)
    assert d[""] == "cuda:0"