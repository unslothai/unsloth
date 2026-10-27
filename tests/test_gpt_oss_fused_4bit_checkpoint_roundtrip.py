# Unsloth - 2x faster, 60% less VRAM LLM training and finetuning
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

"""gpt-oss fused bnb 4-bit checkpoints: the layout probe reads the file transformers loads, and a
full save of fused 4-bit experts keeps their quantization state so the probe accepts it."""

import json
import os
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

import unsloth  # noqa: F401  (patches before transformers)
from unsloth.models import loader_utils

BNB_QC = {"quant_method": "bitsandbytes", "load_in_4bit": True}
P = "model.layers.0.mlp"
FUSED = [f"{P}.experts.gate_up_proj", f"{P}.experts.down_proj", f"{P}.router.weight"]
SPLIT = [f"{P}.experts.gate_up_projs.0.weight", f"{P}.experts.down_projs.0.weight"]


def _tensors(names):
    return {n: torch.zeros(2, 2, dtype = torch.bfloat16) for n in names}


def _stale_index(d, index_name, names):
    shard = "old-00001-of-00001" + (".bin" if index_name.startswith("pytorch") else ".safetensors")
    (d / index_name).write_text(
        json.dumps({"metadata": {}, "weight_map": {n: shard for n in names}})
    )


@pytest.mark.parametrize(
    "single, stale_index",
    [
        ("model.safetensors", "model.safetensors.index.json"),
        ("model.safetensors", "pytorch_model.bin.index.json"),
        ("pytorch_model.bin", "pytorch_model.bin.index.json"),
    ],
)
def test_lone_file_beats_a_stale_index_like_transformers(tmp_path, single, stale_index):
    # transformers resolves model.safetensors, then its index, then pytorch_model.bin, then its index.
    d = tmp_path / "ckpt"
    d.mkdir()
    if single.endswith(".safetensors"):
        save_file(_tensors(FUSED), str(d / single))
    else:
        torch.save(_tensors(FUSED), str(d / single))
    _stale_index(d, stale_index, SPLIT)
    cfg = SimpleNamespace(quantization_config = BNB_QC)
    assert loader_utils.gpt_oss_checkpoint_has_fused_experts(cfg, str(d)) is True


def test_safetensors_index_still_beats_a_lone_bin(tmp_path):
    d = tmp_path / "ckpt"
    d.mkdir()
    torch.save(_tensors(SPLIT), str(d / "pytorch_model.bin"))
    shard = "model-00001-of-00001.safetensors"
    save_file(_tensors(FUSED), str(d / shard))
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {n: shard for n in FUSED}})
    )
    cfg = SimpleNamespace(quantization_config = BNB_QC)
    assert loader_utils.gpt_oss_checkpoint_has_fused_experts(cfg, str(d)) is True


def _quantize_cpu(tensor):
    bnb = pytest.importorskip("bitsandbytes")
    param = bnb.nn.Params4bit(tensor.detach().clone(), requires_grad = False, quant_type = "nf4").to(
        "cpu"
    )
    assert param.quant_state is not None
    return param


def _tiny_gpt_oss():
    from transformers import GptOssConfig, GptOssForCausalLM

    config = GptOssConfig(
        vocab_size = 128,
        hidden_size = 64,
        intermediate_size = 64,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 32,
        num_local_experts = 2,
        num_experts_per_tok = 1,
        layer_types = ["full_attention"],
    )
    torch.manual_seed(0)
    return GptOssForCausalLM(config).to(torch.bfloat16)


def test_full_save_keeps_fused_params4bit_quant_state(tmp_path):
    bnb = pytest.importorskip("bitsandbytes")
    from unsloth.save import patch_saving_functions

    model = _tiny_gpt_oss()
    experts = model.model.layers[0].mlp.experts
    for name in ("gate_up_proj", "down_proj"):
        setattr(experts, name, _quantize_cpu(getattr(experts, name).data.normal_(0.0, 0.05)))
    model.config.quantization_config = dict(BNB_QC, bnb_4bit_quant_type = "nf4")
    patch_saving_functions(model)
    out = str(tmp_path / "saved")
    model.save_pretrained(out)

    saved = load_file(os.path.join(out, "model.safetensors"))
    for name in ("gate_up_proj", "down_proj"):
        key = f"{P}.experts.{name}"
        assert saved[key].dtype == torch.uint8
        assert f"{key}.absmax" in saved and f"{key}.quant_state.bitsandbytes__nf4" in saved
        stats = {k[len(key) + 1 :]: v for k, v in saved.items() if k.startswith(key + ".")}
        restored = bnb.nn.Params4bit.from_prequantized(
            data = saved[key], quantized_stats = stats, requires_grad = False, device = "cpu"
        )
        want = getattr(experts, name)
        assert restored.quant_state.shape == want.quant_state.shape
        torch.testing.assert_close(
            bnb.functional.dequantize_4bit(restored.data, restored.quant_state),
            bnb.functional.dequantize_4bit(want.data, want.quant_state),
        )
    # The hooks live only for the save.
    assert not any(k.endswith(".absmax") or ".quant_state." in k for k in model.state_dict())
    cfg = SimpleNamespace(quantization_config = BNB_QC)
    assert loader_utils.gpt_oss_checkpoint_has_fused_experts(cfg, out) is True


def test_full_save_without_bare_params4bit_is_unchanged(tmp_path):
    from unsloth.save import patch_saving_functions

    model = _tiny_gpt_oss()
    plain = str(tmp_path / "plain")
    model.save_pretrained(plain)
    patch_saving_functions(model)
    patched = str(tmp_path / "patched")
    model.save_pretrained(patched)
    for f in ("model.safetensors", "config.json"):
        with open(os.path.join(plain, f), "rb") as a, open(os.path.join(patched, f), "rb") as b:
            assert a.read() == b.read(), f


@pytest.mark.parametrize("positional", [False, True])
def test_full_save_with_a_supplied_state_dict_keeps_fused_quant_state(tmp_path, positional):
    """`save_pretrained(path, state_dict = model.state_dict())` (and the Trainer's saves) pass a dict
    built before the save installs its hooks, and transformers writes it as given."""
    bnb = pytest.importorskip("bitsandbytes")
    from unsloth.save import patch_saving_functions

    model = _tiny_gpt_oss()
    experts = model.model.layers[0].mlp.experts
    for name in ("gate_up_proj", "down_proj"):
        setattr(experts, name, _quantize_cpu(getattr(experts, name).data.normal_(0.0, 0.05)))
    model.config.quantization_config = dict(BNB_QC, bnb_4bit_quant_type = "nf4")
    patch_saving_functions(model)
    supplied = model.state_dict()
    keys = set(supplied)
    out = str(tmp_path / "saved")
    if positional:
        model.save_pretrained(out, True, supplied)  # save_directory, is_main_process, state_dict
    else:
        model.save_pretrained(out, state_dict = supplied)

    saved = load_file(os.path.join(out, "model.safetensors"))
    for name in ("gate_up_proj", "down_proj"):
        key = f"{P}.experts.{name}"
        assert f"{key}.absmax" in saved and f"{key}.quant_state.bitsandbytes__nf4" in saved
        stats = {k[len(key) + 1 :]: v for k, v in saved.items() if k.startswith(key + ".")}
        restored = bnb.nn.Params4bit.from_prequantized(
            data = saved[key], quantized_stats = stats, requires_grad = False, device = "cpu"
        )
        want = getattr(experts, name)
        torch.testing.assert_close(
            bnb.functional.dequantize_4bit(restored.data, restored.quant_state),
            bnb.functional.dequantize_4bit(want.data, want.quant_state),
        )
    cfg = SimpleNamespace(quantization_config = BNB_QC)
    assert loader_utils.gpt_oss_checkpoint_has_fused_experts(cfg, out) is True
    # The augmented dict is a copy (transformers may consume the dict it is given).
    assert not any(k.endswith(".absmax") or ".quant_state." in k for k in supplied)


def _quantized_tiny_gpt_oss():
    from unsloth.save import patch_saving_functions

    model = _tiny_gpt_oss()
    experts = model.model.layers[0].mlp.experts
    for name in ("gate_up_proj", "down_proj"):
        setattr(experts, name, _quantize_cpu(getattr(experts, name).data.normal_(0.0, 0.05)))
    model.config.quantization_config = dict(BNB_QC, bnb_4bit_quant_type = "nf4")
    patch_saving_functions(model)
    return model, experts


def test_supplied_foreign_packed_bytes_without_quant_state_are_refused(tmp_path):
    """Packed expert bytes that are not the live parameter's (an EMA snapshot, another checkpoint)
    must not be saved with the live parameter's absmax."""
    pytest.importorskip("bitsandbytes")
    model, experts = _quantized_tiny_gpt_oss()
    supplied = {k: v.clone() for k, v in model.state_dict().items()}
    key = f"{P}.experts.gate_up_proj"
    supplied[key] = torch.randint_like(supplied[key], 0, 255)
    assert not torch.equal(supplied[key], experts.gate_up_proj.data)
    with pytest.raises(ValueError, match = "not this model's parameter"):
        model.save_pretrained(str(tmp_path / "saved"), state_dict = supplied)


def test_supplied_quant_state_and_cloned_live_bytes_are_kept(tmp_path):
    """A supplied dict carrying its own quant state keeps it; cloned (equal) live bytes get the live one."""
    bnb = pytest.importorskip("bitsandbytes")
    model, experts = _quantized_tiny_gpt_oss()
    shape = experts.gate_up_proj.quant_state.shape
    other = _quantize_cpu(torch.randn(shape, dtype = torch.bfloat16).mul_(0.05))
    supplied = {k: v.clone() for k, v in model.state_dict().items()}  # copies, not the live storage
    key = f"{P}.experts.gate_up_proj"
    supplied[key] = other.data.clone()
    for entry, value in other.quant_state.as_dict(packed = True).items():
        supplied[f"{key}.{entry}"] = value
    out = str(tmp_path / "saved")
    model.save_pretrained(out, state_dict = supplied)

    saved = load_file(os.path.join(out, "model.safetensors"))
    for name, want in (("gate_up_proj", other), ("down_proj", experts.down_proj)):
        k = f"{P}.experts.{name}"
        stats = {s[len(k) + 1 :]: v for s, v in saved.items() if s.startswith(k + ".")}
        restored = bnb.nn.Params4bit.from_prequantized(
            data = saved[k], quantized_stats = stats, requires_grad = False, device = "cpu"
        )
        torch.testing.assert_close(
            bnb.functional.dequantize_4bit(restored.data, restored.quant_state),
            bnb.functional.dequantize_4bit(want.data, want.quant_state),
        )


def test_live_bytes_check_compares_in_bounded_chunks(monkeypatch):
    """Copied (not same-storage) packed bytes are compared slice by slice: no full-size copy of
    the live tensor is ever moved to the supplied tensor's device, and a difference exits early."""
    from unsloth import save as unsloth_save

    monkeypatch.setattr(unsloth_save, "_PARAMS4BIT_COMPARE_CHUNK_BYTES", 1024)
    live = torch.randint(0, 255, (10_000, 1), dtype = torch.uint8)
    param = SimpleNamespace(data = live, shape = live.shape, dtype = live.dtype, device = live.device)
    sizes = []
    real_equal = torch.equal

    def spy_equal(a, b):
        sizes.append(max(a.numel(), b.numel()))
        return real_equal(a, b)

    monkeypatch.setattr(unsloth_save.torch, "equal", spy_equal)
    assert unsloth_save._is_live_params4bit_data(param, live)  # same storage: no comparison
    assert sizes == []
    assert unsloth_save._is_live_params4bit_data(param, live.clone())
    assert sizes and max(sizes) <= 1024 and len(sizes) == 10
    sizes.clear()
    other = live.clone()
    other[5] ^= 1
    assert not unsloth_save._is_live_params4bit_data(param, other)
    assert len(sizes) == 1  # first chunk differs: stops
    sizes.clear()
    noncontig = torch.randint(0, 255, (100, 100), dtype = torch.uint8).t()
    assert not noncontig.is_contiguous()
    param_nc = SimpleNamespace(
        data = noncontig, shape = noncontig.shape, dtype = noncontig.dtype, device = noncontig.device
    )
    assert unsloth_save._is_live_params4bit_data(param_nc, noncontig.contiguous())
    assert sizes and max(sizes) <= 1024


def test_live_bytes_check_errors_are_not_reported_as_foreign(monkeypatch):
    """An error while comparing (an OOM moving a slice) propagates instead of becoming the
    misleading foreign-weights ValueError."""
    from unsloth import save as unsloth_save

    live = torch.zeros(64, 1, dtype = torch.uint8)
    param = SimpleNamespace(data = live, shape = live.shape, dtype = live.dtype, device = live.device)

    def boom(a, b):
        raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(unsloth_save.torch, "equal", boom)
    with pytest.raises(torch.OutOfMemoryError):
        unsloth_save._is_live_params4bit_data(param, live.clone())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_live_bytes_check_is_bitwise_for_float_quant_storage(monkeypatch, dtype):
    from unsloth import save as unsloth_save

    monkeypatch.setattr(unsloth_save, "_PARAMS4BIT_COMPARE_CHUNK_BYTES", 64)
    raw = torch.randint(0, 256, (257 * 4 * 4,), dtype = torch.uint8)
    live = raw.view(dtype).reshape(-1, 4)
    live[3, 1] = float("nan")
    live[7, 2] = 0.0
    assert torch.isnan(live).any()

    def param_of(t):
        return SimpleNamespace(data = t, shape = t.shape, dtype = t.dtype, device = t.device)

    param = param_of(live)
    assert unsloth_save._is_live_params4bit_data(param, live.clone())
    flipped = live.clone()
    flipped[7, 2] = -0.0  # different bytes, equal under ==
    assert not unsloth_save._is_live_params4bit_data(param, flipped)
    param_nc = param_of(live.t())
    assert unsloth_save._is_live_params4bit_data(param_nc, live.t().contiguous())
    assert not unsloth_save._is_live_params4bit_data(param_nc, flipped.t().contiguous())
    scalar = torch.tensor(float("nan"), dtype = dtype)
    assert unsloth_save._is_live_params4bit_data(param_of(scalar), scalar.clone())


@pytest.mark.parametrize("quant_storage", [torch.float16, torch.bfloat16, torch.float32])
def test_supplied_cloned_float_quant_storage_bytes_are_kept(tmp_path, quant_storage):
    bnb = pytest.importorskip("bitsandbytes")
    from unsloth.save import patch_saving_functions

    model = _tiny_gpt_oss()
    experts = model.model.layers[0].mlp.experts
    for name in ("gate_up_proj", "down_proj"):
        weight = getattr(experts, name).data.normal_(0.0, 0.05)
        param = bnb.nn.Params4bit(
            weight.detach().clone(),
            requires_grad = False,
            quant_type = "nf4",
            quant_storage = quant_storage,
        ).to("cpu")
        setattr(experts, name, param)
    assert any(torch.isnan(getattr(experts, n).data).any() for n in ("gate_up_proj", "down_proj"))
    model.config.quantization_config = dict(BNB_QC, bnb_4bit_quant_type = "nf4")
    patch_saving_functions(model)
    supplied = {k: v.clone() for k, v in model.state_dict().items()}
    out = str(tmp_path / "saved")
    model.save_pretrained(out, state_dict = supplied)

    saved = load_file(os.path.join(out, "model.safetensors"))
    for name in ("gate_up_proj", "down_proj"):
        key = f"{P}.experts.{name}"
        assert f"{key}.absmax" in saved and f"{key}.quant_state.bitsandbytes__nf4" in saved
        want = getattr(experts, name)
        assert torch.equal(saved[key].view(torch.uint8), want.data.view(torch.uint8))
