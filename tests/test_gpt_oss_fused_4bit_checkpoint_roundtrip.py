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
    (d / index_name).write_text(json.dumps({"metadata": {}, "weight_map": {n: shard for n in names}}))


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
    param = bnb.nn.Params4bit(
        tensor.detach().clone(), requires_grad = False, quant_type = "nf4"
    ).to("cpu")
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
