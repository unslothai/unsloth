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

"""Finishing a 16bit load of a static per-tensor fp8 checkpoint.

transformers' `dequantize = True` folds `weight_scale_inv` into every `nn.Linear` weight it has
a converter for, but MoE expert stacks stored as `experts.gate_up_proj` with a sibling
`experts.gate_up_proj_scale_inv` (Mistral-Small-4-119B) match no converter: the raw fp8 values
land in the plain module and the scale is dropped. `_dequantize_leftover_fp8_params` reads the
checkpoint scale back and replaces the parameter with its 16bit dequantization. Offline, CPU,
synthetic checkpoints.
"""

import json
import os
import tempfile

import pytest
import torch
from torch import nn
from safetensors.torch import save_file

# Import unsloth first to set UNSLOTH_IS_PRESENT env var.
import unsloth
from unsloth.models.loader_utils import (
    _dequantize_leftover_fp8_params,
    _fp8_scale_grid_dequant,
    _FP8_DTYPES,
)

_SHARD = "model-00001-of-00001.safetensors"
_FP8 = _FP8_DTYPES[0] if _FP8_DTYPES else None

pytestmark = pytest.mark.skipif(_FP8 is None, reason = "torch has no float8 dtype")


def _write_checkpoint(path, tensors):
    save_file(tensors, os.path.join(path, _SHARD))
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {name: _SHARD for name in tensors}}, f)


class _Experts(nn.Module):
    def __init__(self, E, M, N, dtype):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(E, 2 * M, N, dtype = dtype), requires_grad = False)
        self.down_proj = nn.Parameter(torch.empty(E, N, M, dtype = dtype), requires_grad = False)


class _Model(nn.Module):
    """Mirrors the live tree of a Mistral-Small-4 16bit load: dense linears already dequantized by transformers, expert stacks left as raw fp8 in a plain module."""

    def __init__(self, E = 4, M = 8, N = 16, dtype = torch.bfloat16):
        super().__init__()
        self.q_proj = nn.Linear(N, N, bias = False, dtype = dtype)
        self.experts = _Experts(E, M, N, dtype)
        self.config = None


def _quantize(t):
    scale = t.abs().amax().float().clamp(min = 1e-8) / 448.0
    return (t.float() / scale).to(_FP8), scale


def _build(E = 4, M = 8, N = 16, hidden_state = True):
    torch.manual_seed(0)
    model = _Model(E, M, N)
    gate_up = torch.randn(E, 2 * M, N)
    down = torch.randn(E, N, M)
    q_gate_up, s_gate_up = _quantize(gate_up)
    q_down, s_down = _quantize(down)
    model.experts.gate_up_proj = nn.Parameter(q_gate_up, requires_grad = False)
    model.experts.down_proj = nn.Parameter(q_down, requires_grad = False)
    q_w, s_w = _quantize(model.q_proj.weight.detach())
    tensors = {
        "experts.gate_up_proj": q_gate_up,
        "experts.gate_up_proj_scale_inv": s_gate_up.reshape(1),
        "experts.gate_up_proj_activation_scale": torch.tensor([0.5]),
        "experts.down_proj": q_down,
        "experts.down_proj_scale_inv": s_down.reshape(1),
        "q_proj.weight": q_w,
        "q_proj.weight_scale_inv": s_w.reshape(1),
    }
    expected = {
        "experts.gate_up_proj": (q_gate_up.float() * s_gate_up).to(torch.bfloat16),
        "experts.down_proj": (q_down.float() * s_down).to(torch.bfloat16),
    }
    return model, tensors, expected


def test_leftover_expert_stacks_are_dequantized_from_checkpoint_scale():
    model, tensors, expected = _build()
    q_proj_before = model.q_proj.weight.detach().clone()
    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(d, tensors)
        done, skipped = _dequantize_leftover_fp8_params(model, d, torch.bfloat16)
    assert done == 2
    assert model.experts.gate_up_proj.dtype == torch.bfloat16
    assert model.experts.down_proj.dtype == torch.bfloat16
    assert torch.equal(model.experts.gate_up_proj.detach(), expected["experts.gate_up_proj"])
    assert torch.equal(model.experts.down_proj.detach(), expected["experts.down_proj"])
    # The already-dequantized dense weight is not touched a second time.
    assert torch.equal(model.q_proj.weight.detach(), q_proj_before)
    assert not any(p.dtype in _FP8_DTYPES for p in model.parameters())


def test_module_with_live_scale_is_left_in_fp8():
    """A converted fp8 module keeps its own scale and its fp8 forward: never rewrite it."""
    model, tensors, _ = _build()
    model.experts.gate_up_proj_scale_inv = nn.Parameter(tensors["experts.gate_up_proj_scale_inv"].clone(), requires_grad = False)
    model.experts.down_proj_scale_inv = nn.Parameter(tensors["experts.down_proj_scale_inv"].clone(), requires_grad = False)
    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(d, tensors)
        done, skipped = _dequantize_leftover_fp8_params(model, d, torch.bfloat16)
    assert done == 0
    assert skipped == 2
    assert model.experts.gate_up_proj.dtype == _FP8


def test_no_fp8_params_is_a_noop_without_reading_the_checkpoint():
    model = _Model()
    done, skipped = _dequantize_leftover_fp8_params(model, "/nonexistent/path", torch.bfloat16)
    assert (done, skipped) == (0, 0)


def test_vlm_key_remap_resolves_language_model_prefix():
    """Checkpoint keys `language_model.model.layers...` map onto a live `model.language_model.layers...` tree."""
    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].mlp = nn.Module()
            self.layers[0].mlp.experts = _Experts(2, 4, 8, torch.bfloat16)

    class _VLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.language_model = _Inner()

    torch.manual_seed(1)
    model = _VLM()
    experts = model.model.language_model.layers[0].mlp.experts
    gate_up = torch.randn(2, 8, 8)
    q, s = _quantize(gate_up)
    experts.gate_up_proj = nn.Parameter(q, requires_grad = False)
    tensors = {
        "language_model.model.layers.0.mlp.experts.gate_up_proj": q,
        "language_model.model.layers.0.mlp.experts.gate_up_proj_scale_inv": s.reshape(1),
    }
    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(d, tensors)
        done, _ = _dequantize_leftover_fp8_params(model, d, torch.bfloat16)
    assert done == 1
    assert torch.equal(experts.gate_up_proj.detach(), (q.float() * s).to(torch.bfloat16))


@pytest.mark.parametrize(
    "shape, scale_shape",
    [
        ((6, 8), ()),
        ((6, 8), (1,)),
        ((6, 8), (1, 1)),
        ((6, 8), (3, 2)),
        ((6, 8), (6,)),
        ((3, 6, 8), (3,)),
        ((3, 6, 8), (3, 1, 1)),
        ((3, 6, 8), (3, 3, 4)),
    ],
)
def test_scale_grid_dequant_matches_expanded_reference(shape, scale_shape):
    torch.manual_seed(2)
    q = torch.randn(*shape).to(_FP8)
    s = torch.rand(scale_shape) + 0.5
    out = _fp8_scale_grid_dequant(q, s, torch.float32)
    assert out is not None
    s_full = s.reshape(-1) if s.numel() == 1 else s
    if s.numel() == 1:
        ref = q.float() * s_full
    elif len(shape) == 2:
        s2 = s_full.view(-1, 1) if s_full.ndim == 1 else s_full
        p, qq = s2.shape
        ref = s2.repeat_interleave(shape[0] // p, 0).repeat_interleave(shape[1] // qq, 1) * q.float()
    else:
        s3 = s_full.view(shape[0], 1, 1) if s_full.ndim == 1 else s_full
        p, qq = s3.shape[1], s3.shape[2]
        ref = s3.repeat_interleave(shape[1] // p, 1).repeat_interleave(shape[2] // qq, 2) * q.float()
    assert torch.equal(out, ref)


def test_scale_grid_that_does_not_tile_is_refused():
    q = torch.randn(6, 8).to(_FP8)
    assert _fp8_scale_grid_dequant(q, torch.rand(4, 2), torch.float32) is None
    assert _fp8_scale_grid_dequant(torch.randn(3, 6, 8).to(_FP8), torch.rand(2, 1, 1), torch.float32) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_out_of_memory_on_the_device_is_finished_through_the_cpu(monkeypatch):
    """A card full to its 16bit plan has no room for the dequant transient: the stack is parked on the CPU, dequantized there and moved back."""
    from unsloth.models import loader_utils

    model, tensors, expected = _build()
    model = model.to("cuda")
    real = loader_utils._fp8_scale_grid_dequant
    calls = {"cuda": 0, "cpu": 0}

    def flaky(quantized, scale, out_dtype):
        calls[quantized.device.type] += 1
        if quantized.device.type == "cuda":
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
        return real(quantized, scale, out_dtype)

    monkeypatch.setattr(loader_utils, "_fp8_scale_grid_dequant", flaky)
    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(d, tensors)
        done, skipped = loader_utils._dequantize_leftover_fp8_params(model, d, torch.bfloat16)
    assert done == 2
    assert calls == {"cuda": 2, "cpu": 2}
    assert model.experts.gate_up_proj.device.type == "cuda"
    assert model.experts.gate_up_proj.dtype == torch.bfloat16
    assert torch.equal(model.experts.gate_up_proj.detach().cpu(), expected["experts.gate_up_proj"])
    assert torch.equal(model.experts.down_proj.detach().cpu(), expected["experts.down_proj"])
