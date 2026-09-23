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


"""Unsloth decompresses a compressed-tensors checkpoint once at load, outside inference mode."""

import types

import pytest
import torch

from unsloth.models.loader_utils import _decompress_compressed_tensors_model


class _Status:
    value = "compressed"


class _Compressor:
    def __init__(self, fail_first = False):
        self.calls = []
        self.fail_first = fail_first

    def decompress_model(self, model):
        self.calls.append(torch.is_inference_mode_enabled())
        if self.fail_first and len(self.calls) == 1:
            raise RuntimeError("CUDA out of memory")
        lin = model.lin
        lin.weight = torch.nn.Parameter(torch.ones_like(lin.weight, dtype = torch.float32), requires_grad = False)
        lin.quantization_status = None
        if hasattr(model, "ct_decompress_hook"):
            model.ct_decompress_hook.remove()
            delattr(model, "ct_decompress_hook")


def _model(compressor):
    model = torch.nn.Module()
    model.lin = torch.nn.Linear(4, 4, bias = False)
    model.lin.quantization_status = _Status()
    model.config = types.SimpleNamespace(quantization_config = {"quant_method": "compressed-tensors"})
    model.hf_quantizer = types.SimpleNamespace(compressor = compressor)
    model.forward = lambda x: model.lin(x)
    model.ct_decompress_hook = model.register_forward_pre_hook(lambda m, a: compressor.decompress_model(m))
    return model


def test_decompresses_at_load_and_drops_the_lazy_hook():
    compressor = _Compressor()
    model = _model(compressor)
    assert _decompress_compressed_tensors_model(model)
    assert compressor.calls == [False]
    assert not hasattr(model, "ct_decompress_hook") and not model._forward_pre_hooks


def test_a_failed_load_decompression_falls_back_outside_inference_mode():
    compressor = _Compressor(fail_first = True)
    model = _model(compressor)
    assert not _decompress_compressed_tensors_model(model)
    with torch.inference_mode():
        model(torch.ones(1, 4))
    # The lazy hook ran inside `generate`'s inference_mode, yet the weights it made are normal tensors.
    assert compressor.calls == [False, False]
    assert not model.lin.weight.is_inference()
    x = torch.ones(1, 4, requires_grad = True)
    torch.nn.functional.linear(x, model.lin.weight).sum().backward()


def test_non_compressed_tensors_models_are_left_alone():
    compressor = _Compressor()
    model = _model(compressor)
    model.config.quantization_config = {"quant_method": "fp8"}
    assert not _decompress_compressed_tensors_model(model)
    assert compressor.calls == []
    assert hasattr(model, "ct_decompress_hook")
