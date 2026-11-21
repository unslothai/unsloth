# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""Full finetuning a pack-quantized compressed-tensors checkpoint: decompress_model keeps the int64
`weight_shape` parameter, which full finetuning's requires_grad_(True) rejected."""

import ast
import os
import types

import pytest
import torch


def _load_functions(*names):
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "unsloth",
        "models",
        "loader_utils.py",
    )
    tree = ast.parse(open(path, encoding = "utf-8").read())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body = nodes, type_ignores = []), path, "exec"), namespace)
    namespace.setdefault("_tag_compressed_tensors_modules", lambda model: None)
    namespace.setdefault("_route_compressed_tensors_nvfp4_to_unsloth", lambda model: False)
    namespace.setdefault("_route_compressed_tensors_fp8_to_unsloth", lambda model: True)
    return namespace


_NS = _load_functions(
    "_prepare_compressed_tensors_model",
    "_prepare_compressed_tensors_full_finetuning",
    "_compressed_tensors_format",
    "_decompress_compressed_tensors_model",
    "_remove_same_device_compressed_tensors_offload",
)
_prepare_compressed_tensors_model = _NS["_prepare_compressed_tensors_model"]


class _Status:
    value = "compressed"


class _Compressor:
    """Mimics compressed-tensors 0.19 decompress_model on a pack-quantized Linear."""

    def __init__(self, fails = False):
        self.fails = fails

    def decompress_model(self, model):
        if self.fails:
            raise RuntimeError("CUDA out of memory")
        lin = model.lin
        del lin.weight_packed
        lin.weight = torch.nn.Parameter(torch.ones(4, 8, dtype = torch.bfloat16), requires_grad = False)
        lin.quantization_status = None


def _model(fails = False):
    model = torch.nn.Module()
    model.lin = torch.nn.Linear(8, 4, bias = False)
    del model.lin.weight
    model.lin.weight_packed = torch.nn.Parameter(
        torch.zeros(4, 1, dtype = torch.int32), requires_grad = False
    )
    model.lin.weight_scale = torch.nn.Parameter(torch.ones(4, 1), requires_grad = False)
    model.lin.weight_shape = torch.nn.Parameter(torch.tensor([4, 8]), requires_grad = False)
    model.lin.quantization_status = _Status()
    model.config = types.SimpleNamespace(
        quantization_config = {"quant_method": "compressed-tensors", "format": "pack-quantized"},
        _name_or_path = "tiny-compressed",
    )
    model.hf_quantizer = types.SimpleNamespace(compressor = _Compressor(fails))
    return model


def test_full_finetuning_can_mark_every_parameter_trainable():
    model = _model()
    _prepare_compressed_tensors_model(model, full_finetuning = True)
    # What unsloth_zoo.prepare_model_for_training does for full_finetuning = True.
    for _, param in model.named_parameters():
        param.requires_grad_(True)
    assert model.lin.weight.requires_grad and model.lin.weight.dtype == torch.bfloat16
    # Still saved (compressed-tensors needs it to recompress), just no longer a parameter.
    assert "lin.weight_shape" in model.state_dict()
    assert torch.equal(model.state_dict()["lin.weight_shape"], torch.tensor([4, 8]))
    assert "weight_shape" not in dict(model.lin.named_parameters())


def test_full_finetuning_refuses_weights_left_packed():
    model = _model(fails = True)
    with pytest.raises(
        RuntimeError, match = r"compressed-tensors checkpoint \(pack-quantized\)"
    ) as e:
        _prepare_compressed_tensors_model(model, full_finetuning = True)
    assert "lin.weight_packed" in str(e.value) and "full_finetuning = False" in str(e.value)


def test_lora_path_keeps_the_compressed_tensors_layout():
    model = _model()
    _prepare_compressed_tensors_model(model, full_finetuning = False)
    assert "weight_shape" in dict(model.lin.named_parameters())
    assert "weight_packed" in dict(model.lin.named_parameters())


def test_non_compressed_tensors_models_are_left_alone():
    model = torch.nn.Module()
    model.idx = torch.nn.Parameter(torch.tensor([1, 2]), requires_grad = False)
    model.config = types.SimpleNamespace(quantization_config = {"quant_method": "bitsandbytes"})
    assert _NS["_prepare_compressed_tensors_full_finetuning"](model) == 0
    assert "idx" in dict(model.named_parameters())
