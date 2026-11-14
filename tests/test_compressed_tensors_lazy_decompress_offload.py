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

import types
from dataclasses import dataclass

import pytest
import torch
from torch import nn

offload_cache = pytest.importorskip("compressed_tensors.offload.cache")
offload_module_lib = pytest.importorskip("compressed_tensors.offload.module")
model_compressor = pytest.importorskip(
    "compressed_tensors.compressors.model_compressors.model_compressor"
)
OffloadCache = offload_cache.OffloadCache
ModelCompressor = model_compressor.ModelCompressor

from unsloth.models import mxfp4_compressed_linear as mcl  # noqa: E402


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.Module()
        self.norm.weight = nn.Parameter(torch.ones(4))
        self.proj = nn.Linear(4, 4, bias = False)

    def forward(self, x):
        return self.proj(self.norm.weight * x)


class _LazyCompressor:
    add_decompress_hook = ModelCompressor.add_decompress_hook
    remove_decompression_hook = ModelCompressor.remove_decompression_hook

    def decompress_model(self, model):
        for module in model.modules():
            if module._parameters:
                offload_module_lib.offload_module(module, torch.device("cpu"), torch.device("cpu"))
        self.remove_decompression_hook(model)


def _caches(model):
    return sum(isinstance(m._parameters, OffloadCache) for m in model.modules())


def _lazy_model():
    model = _Tiny()
    _LazyCompressor().add_decompress_hook(model)
    return model


def test_stock_lazy_decompress_leaves_an_offload_cache_compile_cannot_trace():
    model = _lazy_model()
    model(torch.randn(2, 4))
    assert _caches(model) > 0
    fn = torch.compile(lambda m, x: m.weight * x, backend = "eager", fullgraph = True)
    with pytest.raises(Exception, match = "[Dd]ata pointer"):
        fn(model.norm, torch.randn(4))


def test_unwrap_hook_runs_after_the_decompress_hook_and_removes_itself():
    model = _lazy_model()
    reference = _Tiny()
    reference.load_state_dict(model.state_dict())
    x = torch.randn(2, 4)
    assert mcl._unwrap_offload_after_lazy_decompress(model) is True
    assert mcl._unwrap_offload_after_lazy_decompress(model) is False
    out = model(x)
    assert _caches(model) == 0
    assert not hasattr(model, "ct_decompress_hook")
    assert not hasattr(model, "_unsloth_ct_unwrap_hook")
    assert torch.equal(out, reference(x))
    fn = torch.compile(lambda m, x: m.weight * x, backend = "eager", fullgraph = True)
    assert torch.equal(fn(model.norm, x), x)
    assert torch.equal(model(x), out)


def test_no_hook_without_a_lazy_decompress():
    model = _Tiny()
    assert mcl._unwrap_offload_after_lazy_decompress(model) is False
    assert not model._forward_pre_hooks


def test_real_offload_is_left_alone():
    model = _Tiny()
    for module in model.modules():
        if module._parameters:
            offload_module_lib.offload_module(module, torch.device("meta"), torch.device("cpu"))
    from unsloth.models.loader_utils import _remove_same_device_compressed_tensors_offload

    assert _remove_same_device_compressed_tensors_offload(model) == 0
    assert _caches(model) == 2


def _split_device_metadata(model):
    index = 0
    for module in model.modules():
        if module._parameters or module._buffers:
            offload_module_lib.offload_module(module, torch.device("cpu"), torch.device("cpu"))
            for cache in (module._parameters, module._buffers):
                cache.onload_device = torch.device(f"cuda:{index % 2}")
                cache.offload_device = cache.onload_device
            index += 1
    return model


def test_split_device_model_keeps_the_input_transfer():
    model = _split_device_metadata(_Tiny())
    from unsloth.models.loader_utils import _remove_same_device_compressed_tensors_offload

    assert _remove_same_device_compressed_tensors_offload(model) == 2
    assert _caches(model) == 0
    assert not hasattr(model.proj, "_original_forward_func")
    assert not model.proj._forward_pre_hooks
    forward = model.proj.forward.__func__
    assert _remove_same_device_compressed_tensors_offload(model) == 0
    assert model.proj.forward.__func__ is forward


def test_split_cleanup_preserves_encoder_outputs():
    from transformers import BartConfig, BartForConditionalGeneration
    from transformers.modeling_outputs import BaseModelOutput
    from unsloth.models.loader_utils import _remove_same_device_compressed_tensors_offload

    config = BartConfig(
        vocab_size = 32,
        d_model = 8,
        encoder_layers = 1,
        decoder_layers = 1,
        encoder_attention_heads = 2,
        decoder_attention_heads = 2,
        encoder_ffn_dim = 16,
        decoder_ffn_dim = 16,
        max_position_embeddings = 8,
        dropout = 0.0,
        attention_dropout = 0.0,
        activation_dropout = 0.0,
    )
    model = BartForConditionalGeneration(config).eval()
    hidden = torch.randn(2, 3, 8, requires_grad = True)
    encoded = BaseModelOutput(last_hidden_state = hidden)
    ids = torch.tensor([[2, 3], [2, 4]])
    expected = model(encoder_outputs = encoded, decoder_input_ids = ids, use_cache = False).logits
    _split_device_metadata(model)
    assert _remove_same_device_compressed_tensors_offload(model) > 1
    actual = model(encoder_outputs = encoded, decoder_input_ids = ids, use_cache = False).logits
    torch.testing.assert_close(actual, expected, rtol = 0, atol = 0)
    actual.square().sum().backward()
    assert hidden.grad is not None


@pytest.mark.parametrize("dynamic", [False, True])
def test_split_model_output_fullgraph(dynamic):
    from transformers.modeling_outputs import BaseModelOutput
    from unsloth.models.loader_utils import _remove_same_device_compressed_tensors_offload

    class Reader(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.eye(4))

        def forward(self, encoded):
            return encoded.last_hidden_state @ self.weight

    model = _split_device_metadata(nn.Sequential(Reader(), nn.Linear(4, 4)))
    assert _remove_same_device_compressed_tensors_offload(model) == 2
    value = torch.randn(2, 4, requires_grad = True)
    encoded = BaseModelOutput(last_hidden_state = value)
    expected = model(encoded)
    expected.sum().backward()
    gradient = value.grad.clone()
    model.zero_grad()
    value.grad = None
    actual = torch.compile(model, backend = "eager", fullgraph = True, dynamic = dynamic)(encoded)
    torch.testing.assert_close(actual, expected, rtol = 0, atol = 0)
    actual.sum().backward()
    torch.testing.assert_close(value.grad, gradient, rtol = 0, atol = 0)


def test_split_forward_wrapper_deepcopy_uses_the_copy():
    import copy

    from unsloth.models.loader_utils import _remove_same_device_compressed_tensors_offload

    model = _split_device_metadata(nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4)))
    assert _remove_same_device_compressed_tensors_offload(model) == 2
    cloned = copy.deepcopy(model)
    x = torch.randn(2, 4, requires_grad = True)
    expected = model(x).detach()
    with torch.no_grad():
        cloned[0].weight.zero_()
        cloned[0].bias.fill_(2)
        cloned[1].weight.fill_(1)
        cloned[1].bias.zero_()
    actual = cloned(x)
    torch.testing.assert_close(actual, torch.full_like(actual, 8))
    torch.testing.assert_close(model(x), expected)
    actual.sum().backward()
    assert cloned[1].weight.grad is not None
    assert all(parameter.grad is None for parameter in model.parameters())
    cloned[0].to("meta")
    out = cloned[0](torch.randn(2, 4, requires_grad = True))
    assert out.device.type == "meta"
    assert out.requires_grad
    assert model[0].weight.device.type == "cpu"


@pytest.mark.parametrize("dynamic", [False, True])
def test_split_cleanup_preserves_compilation_and_dtype_conversion(dynamic):
    from unsloth.models.loader_utils import _remove_same_device_compressed_tensors_offload

    model = nn.Sequential(nn.BatchNorm1d(4), nn.Linear(4, 4)).eval()
    reference = nn.Sequential(nn.BatchNorm1d(4), nn.Linear(4, 4)).eval()
    reference.load_state_dict(model.state_dict())
    _split_device_metadata(model)
    assert _remove_same_device_compressed_tensors_offload(model) == 2
    model.double()
    reference.double()
    assert model[0].running_mean.dtype == torch.float64
    x = torch.randn(3, 4, dtype = torch.float64, requires_grad = True)
    expected_x = x.detach().clone().requires_grad_()
    compiled = torch.compile(model, backend = "eager", fullgraph = True, dynamic = dynamic)
    actual = compiled(x)
    expected = reference(expected_x)
    torch.testing.assert_close(actual, expected, rtol = 0, atol = 0)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(x.grad, expected_x.grad, rtol = 0, atol = 0)
    for param, ref_param in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(param.grad, ref_param.grad, rtol = 0, atol = 0)


def test_split_wrapper_follows_weight_parametrization():
    from unsloth.models.loader_utils import _remove_same_device_compressed_tensors_offload

    model = nn.Sequential(nn.Linear(4, 4, bias = False), nn.Linear(4, 4))
    x = torch.randn(2, 4)
    expected = model(x).detach()
    _split_device_metadata(model)
    assert _remove_same_device_compressed_tensors_offload(model) == 2
    torch.nn.utils.parametrize.register_parametrization(model[0], "weight", nn.Identity())
    torch.testing.assert_close(model(x), expected, rtol = 0, atol = 0)
    model(x).sum().backward()
    assert model[0].parametrizations.weight.original.grad is not None


def test_split_cleanup_moves_nested_inputs_to_the_live_device():
    from unsloth.models.loader_utils import _remove_same_device_compressed_tensors_offload

    @dataclass
    class Payload:
        tensor: torch.Tensor
        label: str

    class Nested(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4), requires_grad = False)

        def forward(self, values, *, payload):
            assert payload.label == "unchanged"
            assert values[1]["other"] is None
            return (values[0] + payload.tensor) * self.weight

    model = _split_device_metadata(nn.ModuleList([nn.Linear(4, 4), Nested()]))
    assert _remove_same_device_compressed_tensors_offload(model) == 2
    model[1].to("meta")
    x = torch.ones(4, requires_grad = True)
    out = model[1]([x, {"other": None}], payload = Payload(x, "unchanged"))
    assert out.device.type == "meta"
    assert out.requires_grad


def test_split_cleanup_aligns_a_parameterless_residual_container():
    from unsloth.models.loader_utils import _remove_same_device_compressed_tensors_offload

    class Residual(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4)

        def forward(self, x):
            return x + self.proj(x)

    model = _split_device_metadata(nn.Sequential(nn.Linear(4, 4), Residual()))
    residual = model[1]
    offload_module_lib.offload_module(residual, torch.device("cpu"), torch.device("cpu"))
    for cache in (residual._parameters, residual._buffers):
        cache.onload_device = cache.offload_device = torch.device("cuda:1")
    assert _remove_same_device_compressed_tensors_offload(model) == 3
    residual.to("meta")
    assert residual(torch.randn(2, 4)).device.type == "meta"
    residual.to_empty(device = "cpu")
    residual.proj.reset_parameters()
    x = torch.randn(2, 4, requires_grad = True)
    residual(x).sum().backward()
    assert x.grad is not None


def test_installed_quantizer_hook_registers_the_unwrap(monkeypatch):
    quantizer_lib = pytest.importorskip("transformers.quantizers.quantizer_compressed_tensors")
    cls = quantizer_lib.CompressedTensorsHfQuantizer
    monkeypatch.setattr(cls, "_unsloth_keep_packed", True, raising = False)
    monkeypatch.delattr(cls, "_unsloth_keep_packed")
    monkeypatch.setattr(cls, "_process_model_before_weight_loading", lambda self, model, **kw: None)
    monkeypatch.setattr(cls, "_process_model_after_weight_loading", lambda self, model, **kw: None)
    assert mcl.install_compressed_tensors_keep_packed()
    model = _lazy_model()
    cls._process_model_after_weight_loading(types.SimpleNamespace(), model)
    model(torch.randn(2, 4))
    assert _caches(model) == 0
