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

from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py

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


@pytest.mark.skipif(not has_real_cuda(), reason = "needs a CUDA GPU for bitsandbytes")
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


# pytorch_model.bin checkpoints, variants, and the float16 refusal for fused bnb checkpoints.


def _variant(filename, variant):
    # transformers' _add_variant: model.safetensors -> model.fp16.safetensors.
    if not variant:
        return filename
    parts = filename.split(".")
    return ".".join(parts[:-1] + [variant, parts[-1]])


def _bin_sharded(
    tmp_path,
    names,
    variant = None,
):
    d = tmp_path / "ckpt"
    d.mkdir()
    shard = "pytorch_model-00001-of-00001.bin"
    torch.save(_tensors(names, False), str(d / shard))
    index = _variant("pytorch_model.bin.index.json", variant)
    (d / index).write_text(json.dumps({"metadata": {}, "weight_map": {n: shard for n in names}}))
    return str(d)


@pytest.mark.parametrize("names, expected", [(FUSED_QS, True), (SPLIT, False)])
def test_bin_index_names_are_read(tmp_path, names, expected):
    cfg = SimpleNamespace(quantization_config = BNB_QC)
    assert _fused(cfg, _bin_sharded(tmp_path, names)) is expected


@pytest.mark.parametrize("names, expected", [(FUSED_QS, True), (SPLIT, False)])
def test_lone_bin_names_are_read_without_an_index(tmp_path, names, expected):
    d = tmp_path / "ckpt"
    d.mkdir()
    torch.save(_tensors(names, False), str(d / "pytorch_model.bin"))
    assert _fused(SimpleNamespace(quantization_config = BNB_QC), str(d)) is expected


def test_lone_unreadable_bin_fails_closed(tmp_path):
    d = tmp_path / "ckpt"
    d.mkdir()
    # Legacy (non zip) serialization cannot be memory mapped, so the names are unreadable.
    torch.save(
        _tensors(FUSED_QS, False),
        str(d / "pytorch_model.bin"),
        _use_new_zipfile_serialization = False,
    )
    cfg = SimpleNamespace(quantization_config = BNB_QC)
    with pytest.raises(loader_utils.GptOssUnknownExpertLayoutError, match = "safe_serialization"):
        _fused(cfg, str(d))
    # Unsloth's split pre-quantized repos are named *-unsloth-bnb-4bit.
    named = tmp_path / "gpt-oss-20b-unsloth-bnb-4bit"
    shutil.copytree(d, named)
    assert _fused(cfg, str(named)) is False


@pytest.mark.parametrize("write", [_single, _sharded, _bin_sharded])
def test_variant_files_are_read(tmp_path, write):
    path = write(tmp_path, FUSED_QS)
    for f in os.listdir(path):
        if f.startswith(("model.safetensors", "pytorch_model.bin.index")):
            os.rename(os.path.join(path, f), os.path.join(path, _variant(f, "fp16")))
    cfg = SimpleNamespace(quantization_config = BNB_QC)
    assert loader_utils.gpt_oss_checkpoint_has_fused_experts(cfg, path, variant = "fp16") is True


def test_sync_passes_variant(tmp_path, monkeypatch):
    path = _single(tmp_path, FUSED_QS)
    os.rename(os.path.join(path, "model.safetensors"), os.path.join(path, "model.fp16.safetensors"))
    monkeypatch.setenv("UNSLOTH_MODEL_NAME", "gpt_oss,_load_in_4bit_")
    monkeypatch.setenv("UNSLOTH_GPT_OSS_BNB4BIT_PATCHED", "0")
    loader_utils.sync_unsloth_model_name_bnb_flags(
        True,
        False,
        config = SimpleNamespace(quantization_config = BNB_QC),
        model_name = path,
        variant = "fp16",
    )
    assert os.environ["UNSLOTH_MODEL_NAME"] == "gpt_oss,"


def _hub(
    monkeypatch,
    files,
    exists = (),
):
    """Fake Hub repo: `files` maps index json names to weight maps; nothing else is downloadable."""
    import huggingface_hub

    calls = []

    def download(repo_id, filename, *args, **kwargs):
        calls.append(filename)
        if filename in files:
            p = os.path.join(os.environ["TMPDIR"], "w17_hub_" + filename)
            with open(p, "w", encoding = "utf-8") as f:
                json.dump({"weight_map": files[filename]}, f)
            return p
        if not filename.endswith(".json") and not kwargs.get("local_files_only"):
            raise AssertionError(f"downloaded weights {filename}")
        raise FileNotFoundError(filename)

    def no_metadata(*a, **k):
        raise FileNotFoundError("no safetensors")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(huggingface_hub, "get_safetensors_metadata", no_metadata)
    monkeypatch.setattr(huggingface_hub, "file_exists", lambda repo, f, **k: f in exists)
    monkeypatch.setattr(loader_utils, "_hub_safetensors_header_dtypes", no_metadata, raising = False)
    return calls


def test_hub_bin_index_and_variant_index(monkeypatch, tmp_path):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    cfg = SimpleNamespace(quantization_config = BNB_QC)
    _hub(monkeypatch, {"pytorch_model.bin.index.json": {n: "x.bin" for n in FUSED_QS}})
    assert _fused(cfg, "someone/gpt-oss-bin") is True
    _hub(monkeypatch, {"model.safetensors.index.fp16.json": {n: "x" for n in FUSED_QS}})
    assert (
        loader_utils.gpt_oss_checkpoint_has_fused_experts(cfg, "someone/gpt-oss-v", variant = "fp16")
        is True
    )


def test_hub_lone_bin_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    cfg = SimpleNamespace(quantization_config = BNB_QC)
    _hub(monkeypatch, {}, exists = ("pytorch_model.bin",))
    with pytest.raises(loader_utils.GptOssUnknownExpertLayoutError):
        _fused(cfg, "someone/gpt-oss-bin")
    assert _fused(cfg, "unsloth/gpt-oss-20b-unsloth-bnb-4bit") is False
    _hub(monkeypatch, {})  # nothing readable and no bin: unknown keeps the config rule
    assert _fused(cfg, "someone/gpt-oss-none") is False


def _fused_bnb_tiny(tmp_path):
    from transformers import AutoConfig

    d = tmp_path / "fused_bnb"
    d.mkdir()
    config = AutoConfig.from_pretrained("trl-internal-testing/tiny-GptOssForCausalLM")
    config.quantization_config = dict(BNB_QC, bnb_4bit_compute_dtype = "float16")
    config.save_pretrained(str(d))
    save_file(_tensors(FUSED_QS, False), str(d / "model.safetensors"))
    return str(d)


@pytest.mark.parametrize("load_in_4bit", [True, False])
def test_fused_bnb_checkpoint_refused_in_float16_without_load_in_4bit(tmp_path, load_in_4bit):
    from unsloth import FastLanguageModel
    with pytest.raises(NotImplementedError, match = "overflow"):
        FastLanguageModel.from_pretrained(
            _fused_bnb_tiny(tmp_path),
            max_seq_length = 64,
            load_in_4bit = load_in_4bit,
            dtype = torch.float16,
        )


def test_bin_index_packed_without_quant_state_refuses(tmp_path):
    # The fused dtype is read from the indexed .bin shard, so packed bytes without .absmax are caught.
    d = tmp_path / "ckpt"
    d.mkdir()
    shard = "pytorch_model-00001-of-00001.bin"
    torch.save(_tensors(FUSED_PACKED, True), str(d / shard))
    (d / "pytorch_model.bin.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {n: shard for n in FUSED_PACKED}})
    )
    cfg = SimpleNamespace(quantization_config = BNB_QC)
    with pytest.raises(RuntimeError, match = "quantization state"):
        _fused(cfg, str(d))
    # Dense 16-bit experts in the same layout stay fused.
    torch.save(_tensors(FUSED_DENSE, False), str(d / shard))
    assert _fused(cfg, str(d)) is True


def test_hub_bin_index_with_unknown_fused_dtype_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    cfg = SimpleNamespace(quantization_config = BNB_QC)
    # Shard not cached and not a safetensors header: dense or packed is unknown.
    _hub(monkeypatch, {"pytorch_model.bin.index.json": {n: "x.bin" for n in FUSED_PACKED}})
    with pytest.raises(loader_utils.GptOssUnknownExpertLayoutError, match = "dtype"):
        _fused(cfg, "someone/gpt-oss-bin")


def _both_formats(tmp_path):
    # Older split safetensors beside a newer fused .bin.
    d = tmp_path / "ckpt"
    d.mkdir()
    save_file(_tensors(SPLIT, False), str(d / "model.safetensors"))
    torch.save(_tensors(FUSED_QS, False), str(d / "pytorch_model.bin"))
    return str(d)


def test_use_safetensors_selects_the_probed_file(tmp_path):
    path = _both_formats(tmp_path)
    cfg = SimpleNamespace(quantization_config = BNB_QC)
    assert loader_utils.gpt_oss_checkpoint_has_fused_experts(cfg, path) is False
    assert (
        loader_utils.gpt_oss_checkpoint_has_fused_experts(cfg, path, use_safetensors = True) is False
    )
    assert (
        loader_utils.gpt_oss_checkpoint_has_fused_experts(cfg, path, use_safetensors = False) is True
    )


def test_sync_passes_use_safetensors(tmp_path, monkeypatch):
    path = _both_formats(tmp_path)
    monkeypatch.setenv("UNSLOTH_MODEL_NAME", "gpt_oss,_load_in_4bit_")
    monkeypatch.setenv("UNSLOTH_GPT_OSS_BNB4BIT_PATCHED", "0")
    loader_utils.sync_unsloth_model_name_bnb_flags(
        True,
        False,
        config = SimpleNamespace(quantization_config = BNB_QC),
        model_name = path,
        use_safetensors = False,
    )
    assert os.environ["UNSLOTH_MODEL_NAME"] == "gpt_oss,"


def test_loader_forwards_use_safetensors_to_the_probe(tmp_path, monkeypatch):
    from unsloth import FastLanguageModel
    from unsloth.models import loader

    seen = []

    def probe(
        config,
        model_name = None,
        **kwargs,
    ):
        seen.append(kwargs.get("use_safetensors"))
        return True

    monkeypatch.setattr(loader, "gpt_oss_checkpoint_has_fused_experts", probe)
    with pytest.raises(NotImplementedError, match = "overflow"):
        FastLanguageModel.from_pretrained(
            _fused_bnb_tiny(tmp_path),
            max_seq_length = 64,
            load_in_4bit = True,
            dtype = torch.float16,
            use_safetensors = False,
        )
    assert seen and all(v is False for v in seen)
