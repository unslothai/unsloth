# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
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

"""`_dequantize_leftover_fp8_params` on synthetic fp8 checkpoints, offline on CPU."""

import json
import os
import tempfile

import pytest
from real_accelerator import (
    has_real_cuda,
)  # tests/_shared, on sys.path via tests/conftest.py
import torch
from torch import nn
from safetensors.torch import save_file

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
    def __init__(
        self,
        E = 4,
        M = 8,
        N = 16,
        dtype = torch.bfloat16,
    ):
        super().__init__()
        self.q_proj = nn.Linear(N, N, bias = False, dtype = dtype)
        self.experts = _Experts(E, M, N, dtype)
        self.config = None


def _quantize(t):
    scale = t.abs().amax().float().clamp(min = 1e-8) / 448.0
    return (t.float() / scale).to(_FP8), scale


def _build(
    E = 4,
    M = 8,
    N = 16,
    hidden_state = True,
):
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
    assert torch.equal(model.q_proj.weight.detach(), q_proj_before)
    assert not any(p.dtype in _FP8_DTYPES for p in model.parameters())


def test_module_with_live_scale_is_left_in_fp8():
    """A converted fp8 module keeps its own scale and its fp8 forward: never rewrite it."""
    model, tensors, _ = _build()
    model.experts.gate_up_proj_scale_inv = nn.Parameter(
        tensors["experts.gate_up_proj_scale_inv"].clone(), requires_grad = False
    )
    model.experts.down_proj_scale_inv = nn.Parameter(
        tensors["experts.down_proj_scale_inv"].clone(), requires_grad = False
    )
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
        ref = (
            s2.repeat_interleave(shape[0] // p, 0).repeat_interleave(shape[1] // qq, 1) * q.float()
        )
    else:
        s3 = s_full.view(shape[0], 1, 1) if s_full.ndim == 1 else s_full
        p, qq = s3.shape[1], s3.shape[2]
        ref = (
            s3.repeat_interleave(shape[1] // p, 1).repeat_interleave(shape[2] // qq, 2) * q.float()
        )
    assert torch.equal(out, ref)


def test_scale_grid_that_does_not_tile_is_refused():
    q = torch.randn(6, 8).to(_FP8)
    assert _fp8_scale_grid_dequant(q, torch.rand(4, 2), torch.float32) is None
    assert (
        _fp8_scale_grid_dequant(torch.randn(3, 6, 8).to(_FP8), torch.rand(2, 1, 1), torch.float32)
        is None
    )


# has_real_cuda(): another test spoofs torch.cuda.is_available() process-wide.
@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
def test_out_of_memory_on_the_device_is_finished_through_the_cpu(monkeypatch):
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


def test_standard_weight_and_weight_scale_inv_pair_is_dequantized():
    model, tensors, expected = _build()
    q_w = tensors["q_proj.weight"]
    s_w = tensors["q_proj.weight_scale_inv"]
    model.q_proj.weight = nn.Parameter(q_w, requires_grad = False)
    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(d, tensors)
        done, skipped = _dequantize_leftover_fp8_params(model, d, torch.bfloat16)
    assert done == 3
    assert model.q_proj.weight.dtype == torch.bfloat16
    assert torch.equal(model.q_proj.weight.detach(), (q_w.float() * s_w).to(torch.bfloat16))
    assert not any(p.dtype in _FP8_DTYPES for p in model.parameters())


def test_no_reference_to_the_fp8_parameter_survives_into_the_cpu_pass(monkeypatch):
    import gc
    import weakref
    from unsloth.models import loader_utils

    model, tensors, expected = _build()
    original = weakref.ref(model.experts.gate_up_proj)
    real = loader_utils._fp8_scale_grid_dequant
    state = {"raised": False, "alive_in_pass_2": None}

    def flaky(quantized, scale, out_dtype):
        if not state["raised"]:
            state["raised"] = True
            raise torch.OutOfMemoryError("out of memory (simulated)")
        if state["alive_in_pass_2"] is None:
            gc.collect()
            state["alive_in_pass_2"] = original() is not None
        return real(quantized, scale, out_dtype)

    monkeypatch.setattr(loader_utils, "_fp8_scale_grid_dequant", flaky)
    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(d, tensors)
        done, skipped = loader_utils._dequantize_leftover_fp8_params(model, d, torch.bfloat16)
    assert done == 2
    assert state["alive_in_pass_2"] is False
    assert torch.equal(model.experts.gate_up_proj.detach(), expected["experts.gate_up_proj"])


def test_disk_offloaded_leftover_is_refused_with_an_instruction():
    model, tensors, expected = _build()
    model.experts.gate_up_proj = nn.Parameter(
        torch.empty_like(model.experts.gate_up_proj, device = "meta"), requires_grad = False
    )
    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(d, tensors)
        with pytest.raises(RuntimeError, match = "offloaded to disk"):
            _dequantize_leftover_fp8_params(model, d, torch.bfloat16)


def test_activation_scale_survives_on_a_module_that_kept_its_fp8_weight():
    model, tensors, expected = _build()
    # q_proj keeps its own scale, so it stays fp8 and must keep its activation scale.
    model.q_proj.weight = nn.Parameter(tensors["q_proj.weight"], requires_grad = False)
    model.q_proj.weight_scale_inv = nn.Parameter(
        tensors["q_proj.weight_scale_inv"], requires_grad = False
    )
    model.q_proj.register_buffer("input_activation_scale", torch.ones(1))
    model.experts.register_buffer("input_activation_scale", torch.ones(1))
    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(d, tensors)
        done, skipped = _dequantize_leftover_fp8_params(model, d, torch.bfloat16)
    assert done == 2 and skipped >= 1
    assert model.q_proj.weight.dtype in _FP8_DTYPES
    assert hasattr(model.q_proj, "input_activation_scale")
    assert not hasattr(model.experts, "input_activation_scale")


def test_per_tensor_scale_on_a_3d_stack_is_chunked(monkeypatch):
    from unsloth.models import loader_utils

    E, M, N = 8, 32, 32
    # Budget of exactly two experts, so a chunked pass is visibly different from one that is not.
    monkeypatch.setattr(loader_utils, "_FP8_LEFTOVER_MAX_CHUNK", 2 * M * N)
    torch.manual_seed(4)
    q = torch.randn(E, M, N).to(_FP8)
    seen = []
    real_to = torch.Tensor.to

    def spy(self, *args, **kwargs):
        if args and args[0] is torch.float32 and self.dtype == _FP8:
            seen.append(self.shape[0])
        return real_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", spy, raising = True)
    out = loader_utils._fp8_scale_grid_dequant(q, torch.tensor([0.25]), torch.bfloat16)
    monkeypatch.undo()
    assert torch.equal(out, (q.float() * 0.25).to(torch.bfloat16))
    assert seen, "no fp32 cast observed"
    assert max(seen) == 2, (seen, E)


def test_activation_scale_cleanup_is_per_attribute():
    model, tensors, expected = _build()
    model.experts.down_proj_scale_inv = nn.Parameter(
        tensors["experts.down_proj_scale_inv"], requires_grad = False
    )
    model.experts.register_buffer("gate_up_proj_activation_scale", torch.ones(1))
    model.experts.register_buffer("down_proj_activation_scale", torch.ones(1))
    model.experts.register_buffer("input_activation_scale", torch.ones(1))
    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(d, tensors)
        done, skipped = _dequantize_leftover_fp8_params(model, d, torch.bfloat16)
    assert model.experts.gate_up_proj.dtype == torch.bfloat16
    assert model.experts.down_proj.dtype in _FP8_DTYPES
    assert not hasattr(model.experts, "gate_up_proj_activation_scale")
    assert hasattr(model.experts, "down_proj_activation_scale")
    assert hasattr(model.experts, "input_activation_scale")


def test_a_trainable_fp8_parameter_stays_trainable_after_dequantization():
    model, tensors, expected = _build()
    model.experts.gate_up_proj.requires_grad_(True)
    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(d, tensors)
        _dequantize_leftover_fp8_params(model, d, torch.bfloat16)
    assert model.experts.gate_up_proj.dtype == torch.bfloat16
    assert model.experts.gate_up_proj.requires_grad is True
    assert model.experts.down_proj.requires_grad is False


def test_a_transposed_block_grid_is_turned_around_by_the_configured_block_size():
    from unsloth.models.loader_utils import _fp8_scale_grid_dequant, _orient_block_scale

    raw = (torch.arange(8, dtype = torch.float32).reshape(4, 2) + 1).to(_FP8_DTYPES[0])
    scale = torch.tensor([[2.0], [4.0]])  # canonical (2, 1): rows blocks x col blocks
    expected = _fp8_scale_grid_dequant(raw, scale, torch.float32, block_size = (2, 2))
    stored_transposed = scale.t().contiguous()  # (1, 2)
    assert torch.equal(_orient_block_scale(stored_transposed, 4, 2, (2, 2)), scale)
    out = _fp8_scale_grid_dequant(raw, stored_transposed, torch.float32, block_size = (2, 2))
    assert torch.equal(out, expected)
    assert not torch.equal(_fp8_scale_grid_dequant(raw, stored_transposed, torch.float32), expected)


def test_generic_out_of_memory_runtime_errors_defer_to_the_cpu(monkeypatch):
    """Some backends raise a plain RuntimeError with "out of memory" in the text."""
    from unsloth.models import loader_utils

    model, tensors, expected = _build()
    calls = {"n": 0}
    original = loader_utils._fp8_scale_grid_dequant

    def flaky(
        quantized,
        scale,
        dtype,
        block_size = None,
    ):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("HIP out of memory. Tried to allocate 2 GiB")
        return original(quantized, scale, dtype, block_size = block_size)

    monkeypatch.setattr(loader_utils, "_fp8_scale_grid_dequant", flaky)
    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(d, tensors)
        done, skipped = _dequantize_leftover_fp8_params(model, d, torch.bfloat16)
    assert done == 2
    assert model.experts.gate_up_proj.dtype == torch.bfloat16


def test_a_variant_index_uses_transformers_naming(tmp_path):
    import json
    from unsloth.models.loader_utils import _load_fp8_weight_map

    (tmp_path / "model.safetensors.index.fp8.json").write_text(
        json.dumps({"weight_map": {"a.weight": "model-fp8-00001.safetensors"}})
    )
    assert _load_fp8_weight_map(str(tmp_path), True, None, variant = "fp8") == {
        "a.weight": "model-fp8-00001.safetensors"
    }
