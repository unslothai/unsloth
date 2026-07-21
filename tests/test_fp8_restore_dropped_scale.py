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
from unsloth.models.loader_utils import _restore_dropped_fp8_scales, _FP8_DTYPES


_SHARD = "model-00001-of-00001.safetensors"
_FP8 = _FP8_DTYPES[0] if _FP8_DTYPES else None


def _write_checkpoint(
    path,
    tensors,
    filename = _SHARD,
    include_index = True,
):
    save_file(tensors, os.path.join(path, filename))
    if include_index:
        weight_map = {name: filename for name in tensors}
        with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": weight_map}, f)


def _fp8_config(block = (2, 2)):
    return SimpleNamespace(
        quantization_config = {
            "quant_method": "fp8",
            "weight_block_size": list(block),
        }
    )


def _fp8_anchor():
    """A module carrying a real fp8 weight, so the model looks like a genuine fp8 load."""
    m = nn.Linear(2, 2, bias = False)
    m.weight = nn.Parameter(torch.randn(2, 2).to(_FP8), requires_grad = False)
    return m


def _bf16_linear(out_f, in_f, raw):
    m = nn.Linear(in_f, out_f, bias = False).to(torch.bfloat16)
    with torch.no_grad():
        m.weight.copy_(raw)
    return m


def _expand(scale, block, shape):
    bs0, bs1 = block
    expanded = scale.repeat_interleave(bs0, dim = 0).repeat_interleave(bs1, dim = 1)
    return expanded[: shape[0], : shape[1]]


def test_restore_dequantizes_orphaned_scale():
    """A plain bf16 weight whose scale was dropped is dequantized in place."""
    if _FP8 is None:
        return
    torch.manual_seed(0)
    raw = torch.randn(4, 4, dtype = torch.bfloat16)
    scale = torch.rand(2, 2, dtype = torch.float32) + 0.1

    model = nn.Module()
    model.config = _fp8_config((2, 2))
    model.anchor = _fp8_anchor()
    model.layer = _bf16_linear(4, 4, raw)

    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(
            d,
            {
                "layer.weight": raw.to(torch.float32),
                "layer.weight_scale_inv": scale,
            },
        )
        restored, skipped = _restore_dropped_fp8_scales(model, d, local_files_only = True)

    assert restored == 1
    expected = (raw.to(torch.float32) * _expand(scale, (2, 2), (4, 4))).to(
        torch.bfloat16
    )
    assert torch.equal(model.layer.weight.data, expected)


def test_skips_already_fp8_weight():
    """A correctly converted fp8 weight is skipped, never double-scaled."""
    if _FP8 is None:
        return
    weight = torch.randn(4, 4).to(_FP8)
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


def test_skips_offloaded_meta_weight():
    """A disk-offloaded layer (weight on the meta device) is skipped without error or restore."""
    if _FP8 is None:
        return
    raw = torch.randn(4, 4, dtype = torch.bfloat16)
    scale = torch.rand(2, 2, dtype = torch.float32) + 0.1

    model = nn.Module()
    model.config = _fp8_config((2, 2))
    model.anchor = _fp8_anchor()
    model.layer = nn.Linear(4, 4, bias = False)
    # Simulate an offloaded weight living on the meta device.
    model.layer.weight = nn.Parameter(
        torch.empty(4, 4, dtype = torch.bfloat16, device = "meta"), requires_grad = False
    )

    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(
            d,
            {
                "layer.weight": raw.to(torch.float32),
                "layer.weight_scale_inv": scale,
            },
        )
        restored, skipped = _restore_dropped_fp8_scales(model, d, local_files_only = True)

    assert restored == 0
    assert model.layer.weight.device.type == "meta"


def test_noop_when_fully_dequantized():
    """If the model has no fp8 weights at all (e.g. load_in_16bit dequantize), do not rescale."""
    raw = torch.randn(4, 4, dtype = torch.bfloat16)
    scale = torch.rand(2, 2, dtype = torch.float32) + 0.1

    model = nn.Module()
    model.config = _fp8_config((2, 2))
    model.layer = _bf16_linear(4, 4, raw)  # no fp8 anchor -> looks dequantized

    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(d, {"layer.weight_scale_inv": scale})
        restored, skipped = _restore_dropped_fp8_scales(model, d, local_files_only = True)

    assert (restored, skipped) == (0, 0)
    assert torch.equal(model.layer.weight.data, raw)  # untouched


def test_non_block_divisible_shape():
    """Block scale is expanded then sliced to a non-divisible weight shape."""
    if _FP8 is None:
        return
    raw = torch.randn(3, 4, dtype = torch.bfloat16)
    scale = torch.rand(2, 2, dtype = torch.float32) + 0.1

    model = nn.Module()
    model.config = _fp8_config((2, 2))
    model.anchor = _fp8_anchor()
    model.layer = _bf16_linear(3, 4, raw)  # weight shape [3, 4]

    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(d, {"layer.weight_scale_inv": scale})
        restored, skipped = _restore_dropped_fp8_scales(model, d, local_files_only = True)

    assert restored == 1
    expected = (raw.to(torch.float32) * _expand(scale, (2, 2), (3, 4))).to(
        torch.bfloat16
    )
    assert torch.equal(model.layer.weight.data, expected)


def test_transposed_scale_layout():
    """A scale stored in the transposed block grid is transposed before use."""
    if _FP8 is None:
        return
    raw = torch.randn(4, 2, dtype = torch.bfloat16)  # weight [4, 2] -> grid (2, 1)
    scale_correct = torch.rand(2, 1, dtype = torch.float32) + 0.1
    scale_stored = scale_correct.t().contiguous()  # stored transposed as (1, 2)

    model = nn.Module()
    model.config = _fp8_config((2, 2))
    model.anchor = _fp8_anchor()
    model.layer = _bf16_linear(4, 2, raw)

    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(d, {"layer.weight_scale_inv": scale_stored})
        restored, _ = _restore_dropped_fp8_scales(model, d, local_files_only = True)

    assert restored == 1
    expected = (raw.to(torch.float32) * _expand(scale_correct, (2, 2), (4, 2))).to(
        torch.bfloat16
    )
    assert torch.equal(model.layer.weight.data, expected)


def test_single_file_checkpoint_without_index():
    """Unsharded model.safetensors (no index) is still scanned for dropped scales."""
    if _FP8 is None:
        return
    raw = torch.randn(4, 4, dtype = torch.bfloat16)
    scale = torch.rand(2, 2, dtype = torch.float32) + 0.1

    model = nn.Module()
    model.config = _fp8_config((2, 2))
    model.anchor = _fp8_anchor()
    model.layer = _bf16_linear(4, 4, raw)

    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(
            d,
            {"layer.weight_scale_inv": scale},
            filename = "model.safetensors",
            include_index = False,
        )
        restored, _ = _restore_dropped_fp8_scales(model, d, local_files_only = True)

    assert restored == 1
    expected = (raw.to(torch.float32) * _expand(scale, (2, 2), (4, 4))).to(
        torch.bfloat16
    )
    assert torch.equal(model.layer.weight.data, expected)


def test_scalar_block_size_config():
    """A scalar weight_block_size (not a list) is handled without error."""
    if _FP8 is None:
        return
    raw = torch.randn(4, 4, dtype = torch.bfloat16)
    scale = torch.rand(2, 2, dtype = torch.float32) + 0.1

    model = nn.Module()
    model.config = SimpleNamespace(
        quantization_config = {"quant_method": "fp8", "weight_block_size": 2}
    )
    model.anchor = _fp8_anchor()
    model.layer = _bf16_linear(4, 4, raw)

    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(d, {"layer.weight_scale_inv": scale})
        restored, _ = _restore_dropped_fp8_scales(model, d, local_files_only = True)

    assert restored == 1


def test_text_only_prefix_mapping():
    """Checkpoint keys with a language_model prefix match the stripped text-only module names."""
    if _FP8 is None:
        return
    raw = torch.randn(2, 2, dtype = torch.bfloat16)
    scale = torch.rand(1, 1, dtype = torch.float32) + 0.1

    model = nn.Module()
    model.config = _fp8_config((2, 2))
    model.anchor = _fp8_anchor()
    model.model = nn.Module()
    model.model.gate_proj = _bf16_linear(
        2, 2, raw
    )  # module lacks the language_model prefix

    with tempfile.TemporaryDirectory() as d:
        # checkpoint key carries the language_model wrapper the text-only load stripped
        _write_checkpoint(d, {"model.language_model.gate_proj.weight_scale_inv": scale})
        restored, _ = _restore_dropped_fp8_scales(model, d, local_files_only = True)

    assert restored == 1
    expected = (raw.to(torch.float32) * _expand(scale, (2, 2), (2, 2))).to(
        torch.bfloat16
    )
    assert torch.equal(model.model.gate_proj.weight.data, expected)


def test_skips_variant_load():
    """A variant load (variant="fp8") is skipped to avoid applying default-checkpoint scales."""
    if _FP8 is None:
        return
    raw = torch.randn(4, 4, dtype = torch.bfloat16)
    scale = torch.rand(2, 2, dtype = torch.float32) + 0.1
    model = nn.Module()
    model.config = _fp8_config((2, 2))
    model.anchor = _fp8_anchor()
    model.layer = _bf16_linear(4, 4, raw)
    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(d, {"layer.weight_scale_inv": scale})
        result = _restore_dropped_fp8_scales(
            model, d, local_files_only = True, variant = "fp8"
        )
    assert result == (0, 0)
    assert torch.equal(model.layer.weight.data, raw)  # untouched


def test_vlm_language_model_model_alias():
    """A checkpoint key language_model.model.* matches a model.language_model.* module."""
    if _FP8 is None:
        return
    raw = torch.randn(2, 2, dtype = torch.bfloat16)
    scale = torch.rand(1, 1, dtype = torch.float32) + 0.1
    model = nn.Module()
    model.config = _fp8_config((2, 2))
    model.anchor = _fp8_anchor()
    model.model = nn.Module()
    model.model.language_model = nn.Module()
    model.model.language_model.gate_proj = _bf16_linear(
        2, 2, raw
    )  # -> model.language_model.gate_proj
    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(d, {"language_model.model.gate_proj.weight_scale_inv": scale})
        restored, _ = _restore_dropped_fp8_scales(model, d, local_files_only = True)
    assert restored == 1
    expected = (raw.to(torch.float32) * _expand(scale, (2, 2), (2, 2))).to(
        torch.bfloat16
    )
    assert torch.equal(model.model.language_model.gate_proj.weight.data, expected)


def test_noop_without_scale_keys():
    if _FP8 is None:
        return
    model = nn.Module()
    model.config = _fp8_config((2, 2))
    model.anchor = _fp8_anchor()
    model.layer = _bf16_linear(4, 4, torch.randn(4, 4, dtype = torch.bfloat16))
    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(d, {"layer.weight": torch.randn(4, 4)})
        assert _restore_dropped_fp8_scales(model, d, local_files_only = True) == (0, 0)


def test_noop_without_index_or_single_file():
    if _FP8 is None:
        return
    model = nn.Module()
    model.config = _fp8_config((2, 2))
    model.anchor = _fp8_anchor()
    model.layer = _bf16_linear(4, 4, torch.randn(4, 4, dtype = torch.bfloat16))
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
