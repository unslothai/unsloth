# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Restoring dropped block-fp8 `weight_scale_inv` tensors on load (#6200).

Some block-scale fp8 checkpoints leave a Linear (e.g. `mlp.gate_proj`) unconverted, so its raw
quantized values land in a plain bf16 weight and its `weight_scale_inv` is dropped, producing a
garbage un-scaled weight. `_restore_dropped_fp8_scales` dequantizes such orphaned weights in place
using the scale from the checkpoint. Runs offline on CPU with synthetic checkpoints.
"""

import json
import os
import tempfile
from types import SimpleNamespace

import torch
from torch import nn
from safetensors.torch import save_file

# Import unsloth first to set UNSLOTH_IS_PRESENT env var.
import unsloth
from unsloth.models.loader_utils import _restore_dropped_fp8_scales


_SHARD = "model-00001-of-00001.safetensors"


def _write_checkpoint(path, tensors, include_index = True):
    save_file(tensors, os.path.join(path, _SHARD))
    if include_index:
        weight_map = {name: _SHARD for name in tensors}
        with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": weight_map}, f)


def _fp8_config(block = (2, 2)):
    return SimpleNamespace(
        quantization_config = {
            "quant_method": "fp8",
            "weight_block_size": list(block),
        }
    )


def _expand(scale, block, shape):
    bs0, bs1 = block
    expanded = scale.repeat_interleave(bs0, dim = 0).repeat_interleave(bs1, dim = 1)
    return expanded[: shape[0], : shape[1]]


def test_restore_dequantizes_orphaned_scale():
    """A plain bf16 weight whose scale was dropped is dequantized in place."""
    torch.manual_seed(0)
    raw = torch.randn(4, 4, dtype = torch.bfloat16)
    scale = torch.rand(2, 2, dtype = torch.float32) + 0.1

    model = nn.Module()
    model.config = _fp8_config((2, 2))
    model.layer = nn.Linear(4, 4, bias = False).to(torch.bfloat16)
    with torch.no_grad():
        model.layer.weight.copy_(raw)

    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(d, {
            "layer.weight": raw.to(torch.float32),
            "layer.weight_scale_inv": scale,
        })
        restored, skipped = _restore_dropped_fp8_scales(model, d, local_files_only = True)

    assert restored == 1 and skipped == 0
    expected = (raw.to(torch.float32) * _expand(scale, (2, 2), (4, 4))).to(torch.bfloat16)
    assert torch.equal(model.layer.weight.data, expected)


def test_skips_already_fp8_weight():
    """A correctly converted fp8 weight is skipped, never double-scaled."""
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        return
    weight = torch.randn(4, 4).to(fp8_dtype)
    before = weight.clone()

    model = nn.Module()
    model.config = _fp8_config((2, 2))
    model.layer = nn.Linear(4, 4, bias = False)
    model.layer.weight = nn.Parameter(weight, requires_grad = False)

    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(d, {"layer.weight_scale_inv": torch.rand(2, 2)})
        restored, skipped = _restore_dropped_fp8_scales(model, d, local_files_only = True)

    assert restored == 0 and skipped == 1
    assert torch.equal(model.layer.weight.data.float(), before.float())


def test_non_block_divisible_shape():
    """Block scale is expanded then sliced to a non-divisible weight shape."""
    raw = torch.randn(3, 4, dtype = torch.bfloat16)
    scale = torch.rand(2, 2, dtype = torch.float32) + 0.1

    model = nn.Module()
    model.config = _fp8_config((2, 2))
    model.layer = nn.Linear(4, 3, bias = False).to(torch.bfloat16)  # weight shape [3, 4]
    with torch.no_grad():
        model.layer.weight.copy_(raw)

    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(d, {"layer.weight_scale_inv": scale})
        restored, skipped = _restore_dropped_fp8_scales(model, d, local_files_only = True)

    assert restored == 1
    expected = (raw.to(torch.float32) * _expand(scale, (2, 2), (3, 4))).to(torch.bfloat16)
    assert torch.equal(model.layer.weight.data, expected)


def test_noop_without_scale_keys():
    model = nn.Module()
    model.config = _fp8_config((2, 2))
    model.layer = nn.Linear(4, 4, bias = False)
    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(d, {"layer.weight": torch.randn(4, 4)})
        assert _restore_dropped_fp8_scales(model, d, local_files_only = True) == (0, 0)


def test_noop_without_index():
    model = nn.Module()
    model.config = _fp8_config((2, 2))
    model.layer = nn.Linear(4, 4, bias = False)
    with tempfile.TemporaryDirectory() as d:
        assert _restore_dropped_fp8_scales(model, d, local_files_only = True) == (0, 0)


def test_noop_when_not_block_fp8():
    """A non-fp8 (or non-block) quantization config is ignored."""
    scale = torch.rand(2, 2)
    model = nn.Module()
    model.config = SimpleNamespace(
        quantization_config = {"quant_method": "compressed-tensors"}
    )
    model.layer = nn.Linear(4, 4, bias = False)
    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(d, {"layer.weight_scale_inv": scale})
        assert _restore_dropped_fp8_scales(model, d, local_files_only = True) == (0, 0)


def test_nested_submodule_restored():
    """Nested module names (language_model.layers.N...) resolve via get_submodule."""
    raw = torch.randn(2, 2, dtype = torch.bfloat16)
    scale = torch.rand(1, 1, dtype = torch.float32) + 0.1

    model = nn.Module()
    model.config = _fp8_config((2, 2))
    model.language_model = nn.Module()
    model.language_model.gate_proj = nn.Linear(2, 2, bias = False).to(torch.bfloat16)
    with torch.no_grad():
        model.language_model.gate_proj.weight.copy_(raw)

    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(d, {"language_model.gate_proj.weight_scale_inv": scale})
        restored, skipped = _restore_dropped_fp8_scales(model, d, local_files_only = True)

    assert restored == 1
    expected = (raw.to(torch.float32) * _expand(scale, (2, 2), (2, 2))).to(torch.bfloat16)
    assert torch.equal(model.language_model.gate_proj.weight.data, expected)
