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

"""A compressed-tensors checkpoint loaded with plain transformers in a process where Unsloth is
imported (say, reloading what Unsloth saved) is decompressed in compressed-tensors' first-forward
hook, which wraps each module's parameters in an OffloadCache. Unsloth's compiled forwards cannot
trace that cache, so Unsloth unwraps the same-device ones right after the hook runs."""

import types

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
    """compressed-tensors' own add_decompress_hook / remove_decompression_hook, with a
    decompress_model that does what 0.19's does to the module tree (from_accelerate dispatch)."""

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
    assert mcl._unwrap_offload_after_lazy_decompress(model) is False  # idempotent
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
