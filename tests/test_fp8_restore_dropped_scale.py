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
    raw = torch.randn(4, 4).to(_FP8).to(torch.bfloat16)
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
    expected = (raw.to(torch.float32) * _expand(scale, (2, 2), (4, 4))).to(torch.bfloat16)
    assert torch.equal(model.layer.weight.data, expected)


def test_skips_already_fp8_weight():
    """A correctly converted fp8 module (fp8 weight plus its own scale, as FP8Linear) is skipped, never double-scaled."""
    if _FP8 is None:
        return
    weight = torch.randn(4, 4).to(_FP8)
    before = weight.clone()

    model = nn.Module()
    model.config = _fp8_config((2, 2))
    model.layer = nn.Linear(4, 4, bias = False)
    model.layer.weight = nn.Parameter(weight, requires_grad = False)
    model.layer.weight_scale_inv = nn.Parameter(torch.ones(2, 2), requires_grad = False)

    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(d, {"layer.weight_scale_inv": torch.rand(2, 2)})
        restored, skipped = _restore_dropped_fp8_scales(model, d, local_files_only = True)

    assert restored == 0 and skipped == 1
    assert torch.equal(model.layer.weight.data.float(), before.float())


def test_skips_offloaded_meta_weight():
    """A disk-offloaded layer (weight on the meta device) is skipped without error or restore."""
    if _FP8 is None:
        return
    raw = torch.randn(4, 4).to(_FP8).to(torch.bfloat16)
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
    assert torch.equal(model.layer.weight.data, raw)


def test_non_block_divisible_shape():
    """Block scale is expanded then sliced to a non-divisible weight shape."""
    if _FP8 is None:
        return
    raw = torch.randn(3, 4).to(_FP8).to(torch.bfloat16)
    scale = torch.rand(2, 2, dtype = torch.float32) + 0.1

    model = nn.Module()
    model.config = _fp8_config((2, 2))
    model.anchor = _fp8_anchor()
    model.layer = _bf16_linear(3, 4, raw)  # weight shape [3, 4]

    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(d, {"layer.weight_scale_inv": scale})
        restored, skipped = _restore_dropped_fp8_scales(model, d, local_files_only = True)

    assert restored == 1
    expected = (raw.to(torch.float32) * _expand(scale, (2, 2), (3, 4))).to(torch.bfloat16)
    assert torch.equal(model.layer.weight.data, expected)


def test_transposed_scale_layout():
    """A scale stored in the transposed block grid is transposed before use."""
    if _FP8 is None:
        return
    raw = torch.randn(4, 2).to(_FP8).to(torch.bfloat16)
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
    expected = (raw.to(torch.float32) * _expand(scale_correct, (2, 2), (4, 2))).to(torch.bfloat16)
    assert torch.equal(model.layer.weight.data, expected)


def test_single_file_checkpoint_without_index():
    """Unsharded model.safetensors (no index) is still scanned for dropped scales."""
    if _FP8 is None:
        return
    raw = torch.randn(4, 4).to(_FP8).to(torch.bfloat16)
    scale = torch.rand(2, 2, dtype = torch.float32) + 0.1

    model = nn.Module()
    model.config = _fp8_config((2, 2))
    model.anchor = _fp8_anchor()
    model.layer = _bf16_linear(4, 4, raw)

    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(
            d, {"layer.weight_scale_inv": scale}, filename = "model.safetensors", include_index = False
        )
        restored, _ = _restore_dropped_fp8_scales(model, d, local_files_only = True)

    assert restored == 1
    expected = (raw.to(torch.float32) * _expand(scale, (2, 2), (4, 4))).to(torch.bfloat16)
    assert torch.equal(model.layer.weight.data, expected)


def test_scalar_block_size_config():
    """A scalar weight_block_size (not a list) is handled without error."""
    if _FP8 is None:
        return
    raw = torch.randn(4, 4).to(_FP8).to(torch.bfloat16)
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
    raw = torch.randn(2, 2).to(_FP8).to(torch.bfloat16)
    scale = torch.rand(1, 1, dtype = torch.float32) + 0.1

    model = nn.Module()
    model.config = _fp8_config((2, 2))
    model.anchor = _fp8_anchor()
    model.model = nn.Module()
    model.model.gate_proj = _bf16_linear(2, 2, raw)  # module lacks the language_model prefix

    with tempfile.TemporaryDirectory() as d:
        # checkpoint key carries the language_model wrapper the text-only load stripped
        _write_checkpoint(d, {"model.language_model.gate_proj.weight_scale_inv": scale})
        restored, _ = _restore_dropped_fp8_scales(model, d, local_files_only = True)

    assert restored == 1
    expected = (raw.to(torch.float32) * _expand(scale, (2, 2), (2, 2))).to(torch.bfloat16)
    assert torch.equal(model.model.gate_proj.weight.data, expected)


def test_skips_variant_load():
    """A variant load (variant="fp8") is skipped to avoid applying default-checkpoint scales."""
    if _FP8 is None:
        return
    raw = torch.randn(4, 4).to(_FP8).to(torch.bfloat16)
    scale = torch.rand(2, 2, dtype = torch.float32) + 0.1
    model = nn.Module()
    model.config = _fp8_config((2, 2))
    model.anchor = _fp8_anchor()
    model.layer = _bf16_linear(4, 4, raw)
    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(d, {"layer.weight_scale_inv": scale})
        result = _restore_dropped_fp8_scales(model, d, local_files_only = True, variant = "fp8")
    assert result == (0, 0)
    assert torch.equal(model.layer.weight.data, raw)


def test_vlm_language_model_model_alias():
    """A checkpoint key language_model.model.* matches a model.language_model.* module."""
    if _FP8 is None:
        return
    raw = torch.randn(2, 2).to(_FP8).to(torch.bfloat16)
    scale = torch.rand(1, 1, dtype = torch.float32) + 0.1
    model = nn.Module()
    model.config = _fp8_config((2, 2))
    model.anchor = _fp8_anchor()
    model.model = nn.Module()
    model.model.language_model = nn.Module()
    model.model.language_model.gate_proj = _bf16_linear(2, 2, raw)
    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(d, {"language_model.model.gate_proj.weight_scale_inv": scale})
        restored, _ = _restore_dropped_fp8_scales(model, d, local_files_only = True)
    assert restored == 1
    expected = (raw.to(torch.float32) * _expand(scale, (2, 2), (2, 2))).to(torch.bfloat16)
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
    model.config = SimpleNamespace(quantization_config = {"quant_method": "compressed-tensors"})
    model.layer = nn.Linear(4, 4, bias = False)
    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(d, {"layer.weight_scale_inv": scale})
        assert _restore_dropped_fp8_scales(model, d, local_files_only = True) == (0, 0)


def _fp8_linear(out_f, in_f, raw_fp8):
    """A plain Linear holding raw fp8 values and no scale (unconverted text_only key)."""
    m = nn.Linear(in_f, out_f, bias = False)
    m.weight = nn.Parameter(raw_fp8, requires_grad = False)
    return m


def test_text_only_orphaned_fp8_weight_is_dequantized():
    """A text_only fp8 orphan is dequantized into the load dtype, matching the unrenamed load."""
    if _FP8 is None:
        return
    torch.manual_seed(0)
    raw_fp8 = (torch.randn(4, 4) * 100).to(_FP8)
    scale = torch.rand(2, 2, dtype = torch.float32) + 0.1

    # Text-only layout, weight left in fp8.
    text_only = nn.Module()
    text_only.config = _fp8_config((2, 2))
    text_only.model = nn.Module()
    text_only.model.gate_proj = _fp8_linear(4, 4, raw_fp8.clone())
    # Unrenamed layout, weight already cast to bf16 by transformers (the pre-existing restore path).
    full = nn.Module()
    full.config = _fp8_config((2, 2))
    full.anchor = _fp8_anchor()
    full.model = nn.Module()
    full.model.language_model = nn.Module()
    full.model.language_model.gate_proj = _bf16_linear(4, 4, raw_fp8.to(torch.bfloat16))

    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(d, {"model.language_model.gate_proj.weight_scale_inv": scale})
        restored, skipped = _restore_dropped_fp8_scales(
            text_only, d, local_files_only = True, dtype = torch.bfloat16
        )
        assert _restore_dropped_fp8_scales(full, d, local_files_only = True) == (1, 0)

    assert (restored, skipped) == (1, 0)
    got = text_only.model.gate_proj.weight
    assert isinstance(got, nn.Parameter) and got.dtype == torch.bfloat16
    expected = (raw_fp8.to(torch.float32) * _expand(scale, (2, 2), (4, 4))).to(torch.bfloat16)
    assert torch.equal(got.data, expected)
    assert torch.equal(got.data, full.model.language_model.gate_proj.weight.data)
    # Used to raise `BFloat16 != Float8_e4m3fn`.
    x = torch.randn(3, 4, dtype = torch.bfloat16)
    assert text_only.model.gate_proj(x).dtype == torch.bfloat16


def test_orphaned_fp8_weight_uses_requested_dtype():
    """The dequant target follows the load dtype (fp16 on T4 / V100), then config.dtype, then bfloat16."""
    if _FP8 is None:
        return
    raw_fp8 = torch.randn(4, 4).to(_FP8)
    scale = torch.rand(2, 2, dtype = torch.float32) + 0.1
    for kwargs, config_dtype, want in (
        ({"dtype": torch.float16}, None, torch.float16),
        ({}, "float16", torch.float16),
        ({}, torch.float32, torch.float32),
        ({}, None, torch.bfloat16),
    ):
        model = nn.Module()
        model.config = _fp8_config((2, 2))
        model.config.dtype = config_dtype
        model.layer = _fp8_linear(4, 4, raw_fp8.clone())
        with tempfile.TemporaryDirectory() as d:
            _write_checkpoint(d, {"layer.weight_scale_inv": scale})
            restored, _ = _restore_dropped_fp8_scales(model, d, local_files_only = True, **kwargs)
        assert restored == 1
        assert model.layer.weight.dtype == want, (kwargs, config_dtype, model.layer.weight.dtype)
        expected = (raw_fp8.to(torch.float32) * _expand(scale, (2, 2), (4, 4))).to(want)
        assert torch.equal(model.layer.weight.data, expected)


def test_fp8_module_with_scale_attr_untouched_next_to_orphan():
    """A real fp8 module keeps its weight and scale; only the scale-less orphan is dequantized."""
    if _FP8 is None:
        return
    raw_fp8 = torch.randn(4, 4).to(_FP8)
    scale = torch.rand(2, 2, dtype = torch.float32) + 0.1
    model = nn.Module()
    model.config = _fp8_config((2, 2))
    model.gate_proj = _fp8_linear(4, 4, raw_fp8.clone())
    try:
        from transformers.integrations.finegrained_fp8 import FP8Linear
        up = FP8Linear(4, 4, block_size = (2, 2))
    except Exception:
        up = nn.Linear(4, 4, bias = False)
        up.weight_scale_inv = nn.Parameter(torch.ones(2, 2), requires_grad = False)
    up.weight = nn.Parameter(raw_fp8.clone(), requires_grad = False)
    up.weight_scale_inv.data.copy_(scale)
    model.up_proj = up
    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(
            d,
            {
                "gate_proj.weight_scale_inv": scale.clone(),
                "up_proj.weight_scale_inv": scale.clone(),
            },
        )
        restored, skipped = _restore_dropped_fp8_scales(
            model, d, local_files_only = True, dtype = torch.bfloat16
        )
    assert (restored, skipped) == (1, 1)
    assert model.gate_proj.weight.dtype == torch.bfloat16
    assert model.up_proj.weight.dtype == _FP8
    assert torch.equal(model.up_proj.weight.data.float(), raw_fp8.float())
    assert torch.equal(model.up_proj.weight_scale_inv.data, scale)


def test_load_call_sites_pass_dtype():
    """Every loader call passes the load dtype, so an orphan is dequantized into what the load asked for."""
    import ast
    import inspect
    import unsloth.models.llama as llama_mod
    import unsloth.models.vision as vision_mod

    for mod, want in ((llama_mod, 2), (vision_mod, 1)):
        calls = [
            node
            for node in ast.walk(ast.parse(inspect.getsource(mod)))
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "_restore_dropped_fp8_scales"
        ]
        assert len(calls) == want, (mod.__name__, len(calls))
        for call in calls:
            assert "dtype" in {kw.arg for kw in call.keywords}, mod.__name__


def _offload(
    model,
    module_name,
    disk_dir = None,
):
    """Offload one submodule through accelerate like a sequential device map."""
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module
    from accelerate.utils import OffloadedWeightsLoader, PrefixedDataset, offload_state_dict

    module = dict(model.named_modules())[module_name]
    state = {f"{module_name}.{k}": v.detach().clone() for k, v in module.state_dict().items()}
    if disk_dir is None:
        store = OffloadedWeightsLoader(state_dict = state)
    else:
        offload_state_dict(disk_dir, state)
        store = OffloadedWeightsLoader(save_folder = disk_dir)
    hook = AlignDevicesHook(
        execution_device = "cpu",
        offload = True,
        weights_map = PrefixedDataset(store, f"{module_name}."),
    )
    add_hook_to_module(module, hook)
    assert module.weight.device.type == "meta"
    return store


def _offloaded_case(raw, placeholder_dtype):
    raw_fp8 = raw.to(_FP8)
    scale = torch.rand(2, 2, dtype = torch.float32) + 0.1
    model = nn.Module()
    model.config = _fp8_config((2, 2))
    model.anchor = _fp8_anchor()
    model.model = nn.Module()
    if placeholder_dtype == _FP8:
        model.model.gate_proj = _fp8_linear(4, 4, raw_fp8.clone())
    else:
        model.model.gate_proj = _bf16_linear(4, 4, raw_fp8.to(torch.bfloat16))
    return model, raw_fp8, scale


def test_offloaded_orphans_are_restored_through_weights_map():
    """An offloaded orphan (cpu or disk, fp8 or bf16) is dequantized in the weights map."""
    if _FP8 is None:
        return
    import pytest

    pytest.importorskip("accelerate")
    torch.manual_seed(0)
    raw = torch.randn(4, 4) * 100
    for placeholder_dtype in (_FP8, torch.bfloat16):
        for disk in (False, True):
            model, raw_fp8, scale = _offloaded_case(raw, placeholder_dtype)
            with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as off:
                _offload(model, "model.gate_proj", off if disk else None)
                _write_checkpoint(d, {"model.language_model.gate_proj.weight_scale_inv": scale})
                restored, skipped = _restore_dropped_fp8_scales(
                    model, d, local_files_only = True, dtype = torch.bfloat16
                )
                assert (restored, skipped) == (1, 0), (placeholder_dtype, disk)
                gate = model.model.gate_proj
                assert gate.weight.device.type == "meta" and gate.weight.dtype == torch.bfloat16
                expected = (raw_fp8.to(torch.float32) * _expand(scale, (2, 2), (4, 4))).to(
                    torch.bfloat16
                )
                x = torch.randn(3, 4, dtype = torch.bfloat16)
                out = gate(x)
                assert torch.equal(out, torch.nn.functional.linear(x, expected)), (
                    placeholder_dtype,
                    disk,
                )


def test_offloaded_fp8_module_with_scale_is_skipped_not_counted_offloaded():
    """A converted fp8 module that is offloaded still carries its scale placeholder: skipped, store untouched."""
    if _FP8 is None:
        return
    import pytest

    pytest.importorskip("accelerate")
    raw_fp8 = torch.randn(4, 4).to(_FP8)
    model = nn.Module()
    model.config = _fp8_config((2, 2))
    model.up_proj = _fp8_linear(4, 4, raw_fp8.clone())
    model.up_proj.weight_scale_inv = nn.Parameter(torch.ones(2, 2), requires_grad = False)
    store = _offload(model, "up_proj")
    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(d, {"up_proj.weight_scale_inv": torch.rand(2, 2)})
        assert _restore_dropped_fp8_scales(model, d, local_files_only = True) == (0, 1)
    assert store["up_proj.weight"].dtype == _FP8
    assert torch.equal(store["up_proj.weight"].float(), raw_fp8.float())


def _raw_and_folded(scale):
    """Raw fp8 values, and the same values with the block scale folded in, in bf16."""
    torch.manual_seed(0)
    raw_fp8 = (torch.randn(4, 4) * 100).to(_FP8)
    folded = (raw_fp8.to(torch.float32) * _expand(scale, (2, 2), (4, 4))).to(torch.bfloat16)
    return raw_fp8, folded


def test_orphan_is_scaled_exactly_once():
    """Orphans are scaled once; a second pass leaves them alone."""
    if _FP8 is None:
        return
    scale = torch.tensor([[0.0123, 0.0456], [0.0789, 0.0321]])
    raw_fp8, expected = _raw_and_folded(scale)
    for holds in ("fp8", "bf16"):
        model = nn.Module()
        model.config = _fp8_config((2, 2))
        model.anchor = _fp8_anchor()
        model.model = nn.Module()
        if holds == "fp8":
            model.model.gate_proj = _fp8_linear(4, 4, raw_fp8.clone())
        else:
            model.model.gate_proj = _bf16_linear(4, 4, raw_fp8.to(torch.bfloat16))
        with tempfile.TemporaryDirectory() as d:
            _write_checkpoint(d, {"model.language_model.gate_proj.weight_scale_inv": scale})
            first = _restore_dropped_fp8_scales(
                model, d, local_files_only = True, dtype = torch.bfloat16
            )
            second = _restore_dropped_fp8_scales(
                model, d, local_files_only = True, dtype = torch.bfloat16
            )
        assert first == (1, 0) and second == (0, 1), (holds, first, second)
        assert torch.equal(model.model.gate_proj.weight.data, expected), holds


def test_already_dequantized_weight_is_not_scaled_again():
    """A bf16 weight with its scale already folded in is skipped, never double-scaled."""
    if _FP8 is None:
        return
    scale = torch.tensor([[0.0123, 0.0456], [0.0789, 0.0321]])
    raw_fp8, folded = _raw_and_folded(scale)
    model = nn.Module()
    model.config = _fp8_config((2, 2))
    model.anchor = _fp8_anchor()
    model.model = nn.Module()
    model.model.gate_proj = _bf16_linear(4, 4, folded.clone())
    model.model.up_proj = _bf16_linear(4, 4, raw_fp8.to(torch.bfloat16))
    with tempfile.TemporaryDirectory() as d:
        _write_checkpoint(
            d,
            {
                "model.language_model.gate_proj.weight_scale_inv": scale,
                "model.language_model.up_proj.weight_scale_inv": scale.clone(),
            },
        )
        assert _restore_dropped_fp8_scales(model, d, local_files_only = True) == (1, 1)
    assert torch.equal(model.model.gate_proj.weight.data, folded)
    assert torch.equal(model.model.up_proj.weight.data, folded)


def test_offloaded_already_dequantized_weight_is_not_scaled_again():
    """Offload path: folded stored value left as is, raw one scaled once, repeat pass a no-op."""
    if _FP8 is None:
        return
    import pytest

    pytest.importorskip("accelerate")
    scale = torch.tensor([[0.0123, 0.0456], [0.0789, 0.0321]])
    raw_fp8, folded = _raw_and_folded(scale)
    for stored, disk in (("folded", False), ("folded", True), ("raw", False), ("raw", True)):
        model = nn.Module()
        model.config = _fp8_config((2, 2))
        model.anchor = _fp8_anchor()
        model.model = nn.Module()
        model.model.gate_proj = _bf16_linear(
            4, 4, folded.clone() if stored == "folded" else raw_fp8.to(torch.bfloat16)
        )
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as off:
            store = _offload(model, "model.gate_proj", off if disk else None)
            _write_checkpoint(d, {"model.language_model.gate_proj.weight_scale_inv": scale})
            first = _restore_dropped_fp8_scales(
                model, d, local_files_only = True, dtype = torch.bfloat16
            )
            second = _restore_dropped_fp8_scales(
                model, d, local_files_only = True, dtype = torch.bfloat16
            )
            assert first == ((0, 1) if stored == "folded" else (1, 0)), (stored, disk, first)
            assert second == (0, 1), (stored, disk, second)
            assert torch.equal(store["model.gate_proj.weight"].to(torch.bfloat16), folded), (
                stored,
                disk,
            )
            x = torch.randn(3, 4, dtype = torch.bfloat16)
            assert torch.equal(model.model.gate_proj(x), torch.nn.functional.linear(x, folded)), (
                stored,
                disk,
            )
