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

"""A bitsandbytes gpt-oss checkpoint's expert layout comes from its tensor names, not from the
presence of a quantization_config: a fused model quantized by Unsloth and saved keeps fused names."""

import json
import os
import shutil
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import unsloth  # noqa: F401  (patches before transformers)
from unsloth.models import loader_utils

BNB_QC = {"quant_method": "bitsandbytes", "load_in_4bit": True}
MXFP4_QC = {"quant_method": "mxfp4"}
P = "model.layers.0.mlp"

SPLIT = [
    f"{P}.experts.gate_up_projs.0.weight",
    f"{P}.experts.down_projs.0.weight",
    f"{P}.router.linear.weight",
]
# Fused experts packed to 4-bit with their quant state (what zoo's fused loader reads back).
FUSED_QS = [
    f"{P}.experts.gate_up_proj",
    f"{P}.experts.gate_up_proj.absmax",
    f"{P}.experts.gate_up_proj.quant_map",
    f"{P}.experts.down_proj",
    f"{P}.experts.down_proj.absmax",
    f"{P}.router.weight",
]
# Dense fused experts beside 4-bit attention (the unsloth/gpt-oss-20b-bnb-4bit layout).
FUSED_DENSE = [f"{P}.experts.gate_up_proj", f"{P}.experts.down_proj", f"{P}.router.weight"]
# Packed bytes with no quant state: what save_pretrained writes for Unsloth's fused 4-bit experts.
FUSED_PACKED = FUSED_DENSE


def _tensors(names, packed):
    return {
        n: torch.zeros(4, 1, dtype = torch.uint8)
        if packed and n.endswith(("experts.gate_up_proj", "experts.down_proj"))
        else torch.zeros(2, 2, dtype = torch.bfloat16)
        for n in names
    }


def _single(
    tmp_path,
    names,
    packed = False,
):
    d = tmp_path / "ckpt"
    d.mkdir()
    save_file(_tensors(names, packed), str(d / "model.safetensors"))
    return str(d)


def _sharded(
    tmp_path,
    names,
    packed = False,
):
    d = tmp_path / "ckpt"
    d.mkdir()
    shard = "model-00001-of-00001.safetensors"
    save_file(_tensors(names, packed), str(d / shard))
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {n: shard for n in names}})
    )
    return str(d)


def _fused(config, path):
    return loader_utils.gpt_oss_checkpoint_has_fused_experts(config, path)


@pytest.mark.parametrize("write", [_single, _sharded])
def test_bnb_split_prequantized_is_not_fused(tmp_path, write):
    cfg = SimpleNamespace(quantization_config = BNB_QC)
    assert _fused(cfg, write(tmp_path, SPLIT)) is False


@pytest.mark.parametrize("write", [_single, _sharded])
def test_bnb_fused_with_quant_state_is_fused(tmp_path, write):
    cfg = SimpleNamespace(quantization_config = BNB_QC)
    assert _fused(cfg, write(tmp_path, FUSED_QS)) is True


@pytest.mark.parametrize("write", [_single, _sharded])
def test_bnb_dense_fused_experts_are_fused(tmp_path, write):
    cfg = SimpleNamespace(quantization_config = BNB_QC)
    assert _fused(cfg, write(tmp_path, FUSED_DENSE)) is True


@pytest.mark.parametrize("write", [_single, _sharded])
def test_bnb_packed_fused_without_quant_state_refuses(tmp_path, write):
    cfg = SimpleNamespace(quantization_config = BNB_QC)
    with pytest.raises(RuntimeError, match = "quantization state"):
        _fused(cfg, write(tmp_path, FUSED_PACKED, packed = True))


def test_mxfp4_and_unreadable_keep_the_config_rule(tmp_path):
    assert (
        _fused(SimpleNamespace(quantization_config = MXFP4_QC), _single(tmp_path, FUSED_DENSE))
        is False
    )
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _fused(SimpleNamespace(quantization_config = BNB_QC), str(empty)) is False
    assert _fused(SimpleNamespace(quantization_config = BNB_QC), None) is False
    assert (
        loader_utils._gpt_oss_bnb_checkpoint_expert_layout({f"{P}.self_attn.q_proj.weight": "U8"})
        is None
    )
    assert _fused(SimpleNamespace(), str(empty)) is True
    assert _fused(None, str(empty)) is None


def test_sync_keeps_4bit_token_off_for_saved_fused_bnb(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_MODEL_NAME", "gpt_oss,_load_in_4bit_")
    monkeypatch.setenv("UNSLOTH_GPT_OSS_BNB4BIT_PATCHED", "0")
    loader_utils.sync_unsloth_model_name_bnb_flags(
        True,
        False,
        config = SimpleNamespace(quantization_config = BNB_QC),
        model_name = _single(tmp_path, FUSED_QS),
    )
    assert os.environ["UNSLOTH_MODEL_NAME"] == "gpt_oss,"


def test_hub_names_without_downloading_weights(monkeypatch):
    import huggingface_hub

    calls = []

    def no_weights(repo_id, filename, *args, **kwargs):
        calls.append((filename, kwargs.get("local_files_only")))
        if filename.endswith(".safetensors") and not kwargs.get("local_files_only"):
            raise AssertionError("downloaded weights")
        raise FileNotFoundError(filename)  # no index, weights not cached: header range read

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_weights)
    seen = []
    tensors = {n: SimpleNamespace(dtype = "BF16") for n in FUSED_DENSE}
    monkeypatch.setattr(
        huggingface_hub,
        "get_safetensors_metadata",
        lambda repo_id, **kw: seen.append(repo_id)
        or SimpleNamespace(
            weight_map = {n: "m" for n in FUSED_DENSE},
            files_metadata = {"m": SimpleNamespace(tensors = tensors)},
        ),
    )
    dtypes = loader_utils._checkpoint_tensor_dtypes("someone/gpt-oss-bnb")
    assert seen == ["someone/gpt-oss-bnb"] and dtypes == {n: "BF16" for n in FUSED_DENSE}
    assert calls == []
    # Offline: only cached files, never the weights.
    assert (
        loader_utils._checkpoint_tensor_dtypes("someone/gpt-oss-bnb", local_files_only = True) is None
    )
    assert ("model.safetensors", True) in calls and seen == ["someone/gpt-oss-bnb"]


def _expert_weight(experts, name):
    import bitsandbytes.functional as bnbF

    w = getattr(experts, name)
    if getattr(w, "quant_state", None) is not None:
        return bnbF.dequantize_4bit(w.data, w.quant_state).float().cpu()
    return w.detach().float().cpu()


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA GPU for bitsandbytes")
def test_reload_of_saved_fused_4bit_checkpoints(tmp_path):
    pytest.importorskip("bitsandbytes")
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    src = "trl-internal-testing/tiny-GptOssForCausalLM"
    ckpt = str(tmp_path / "bf16")
    ref = AutoModelForCausalLM.from_pretrained(src, dtype = torch.bfloat16)
    torch.manual_seed(0)
    with torch.no_grad():
        for name, param in ref.named_parameters():
            if "experts" in name or "router" in name:
                param.normal_(0.0, 0.05)
    ref.save_pretrained(ckpt)
    AutoTokenizer.from_pretrained(src).save_pretrained(ckpt)
    del ref
    original = load_file(os.path.join(ckpt, "model.safetensors"))

    from unsloth import FastLanguageModel

    load = dict(max_seq_length = 64, load_in_4bit = True, dtype = torch.bfloat16)
    model, tokenizer = FastLanguageModel.from_pretrained(ckpt, **load)
    saved = str(tmp_path / "saved_4bit")
    model.save_pretrained(saved)
    tokenizer.save_pretrained(saved)
    del model
    torch.cuda.empty_cache()

    tensors_saved = load_file(os.path.join(saved, "model.safetensors"))

    # bnb config with dense fused experts (the unsloth/gpt-oss-20b-bnb-4bit layout).
    dense = str(tmp_path / "dense_experts_4bit")
    os.makedirs(dense)
    for f in os.listdir(saved):
        if f != "model.safetensors":
            shutil.copy(os.path.join(saved, f), dense)
    tensors = dict(tensors_saved)
    for name in list(tensors):
        if name.endswith(("experts.gate_up_proj", "experts.down_proj")):
            tensors[name] = original[name]
    save_file(tensors, os.path.join(dense, "model.safetensors"), metadata = {"format": "pt"})

    model, _ = FastLanguageModel.from_pretrained(dense, **load)
    for layer_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        prefix = f"model.layers.{layer_idx}.mlp"
        assert not hasattr(mlp.experts, "gate_up_projs"), type(mlp.experts)
        router_w = mlp.router.linear.weight if hasattr(mlp.router, "linear") else mlp.router.weight
        torch.testing.assert_close(
            router_w.detach().float().cpu(), original[f"{prefix}.router.weight"].float()
        )
        for name in ("gate_up_proj", "down_proj"):
            got = _expert_weight(mlp.experts, name)
            want = original[f"{prefix}.experts.{name}"].float()
            if got.shape != want.shape:
                got = got.transpose(-1, -2)
            assert float((got - want).abs().max()) < 0.25 * float(want.abs().max()), (
                layer_idx,
                name,
            )
    del model
    torch.cuda.empty_cache()

    # Unsloth's own 4-bit save: never split classes with random experts.
    packed = tensors_saved["model.layers.0.mlp.experts.gate_up_proj"].dtype == torch.uint8
    if packed and "model.layers.0.mlp.experts.gate_up_proj.absmax" not in tensors_saved:
        with pytest.raises(RuntimeError, match = "quantization state"):
            FastLanguageModel.from_pretrained(saved, **load)
    else:
        model, _ = FastLanguageModel.from_pretrained(saved, **load)
        assert not hasattr(model.model.layers[0].mlp.experts, "gate_up_projs")
        del model
