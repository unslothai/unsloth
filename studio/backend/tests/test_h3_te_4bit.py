# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""H3 4-bit GGUF and int8 conditioner loads and the host-RAM pick, on a tiny CPU model."""

from __future__ import annotations

import json
import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import video_minimax_h3_te as te  # noqa: E402


_LAYERS = te.H3_TE_READ_LAYER + 2


def _tiny_config(hidden: int = 32):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3VLForConditionalGeneration"):
        pytest.skip("transformers without Qwen3-VL")
    from transformers import Qwen3VLConfig

    return Qwen3VLConfig(
        text_config = {
            "hidden_size": hidden,
            "intermediate_size": 2 * hidden,
            "num_hidden_layers": _LAYERS,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": hidden // 2,
            "vocab_size": 96,
            "max_position_embeddings": 128,
            "rope_scaling": {
                "rope_type": "default",
                "mrope_section": [2, 3, 3],
                "mrope_interleaved": True,
            },
        },
        vision_config = {
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": hidden,
            "patch_size": 4,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0],
        },
    )


def _layout_key_name(key: str) -> str:
    if key.startswith("model.visual."):
        return key[len("model.") :]
    if key.startswith("model.language_model."):
        return "model." + key[len("model.language_model.") :]
    return key


def _blob(meta: dict):
    return torch.tensor(list(json.dumps(meta).encode()), dtype = torch.uint8)


def _write_file(tmp_path):
    import transformers
    from safetensors.torch import save_file

    from core.inference.diffusion_convrot import quantize_convrot_int8_weight

    torch.manual_seed(2)
    # ConvRot rotates groups of 256 input columns, so the int8 layout needs 256-wide projections.
    config = _tiny_config(256)
    (tmp_path / "text_encoder").mkdir()
    config.save_pretrained(tmp_path / "text_encoder")
    ref = transformers.Qwen3VLForConditionalGeneration(config).to(torch.bfloat16).eval()
    out = {}
    for key, tensor in ref.state_dict().items():
        if key.startswith("lm_head") or key.endswith("language_model.norm.weight"):
            continue
        if ".layers." in key and "language_model" in key:
            if int(key.split("layers.")[1].split(".")[0]) >= te.H3_TE_READ_LAYER:
                continue
        name = _layout_key_name(key)
        stem = name[: -len(".weight")]
        if "language_model.layers." in key and key.endswith("_proj.weight"):
            codes, scale = quantize_convrot_int8_weight(tensor.float(), te.H3_TE_CONVROT_GROUP)
            out[name] = codes
            out[stem + ".weight_scale"] = scale.reshape(-1, 1).float()
            out[stem + ".comfy_quant"] = _blob(
                {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": te.H3_TE_CONVROT_GROUP,
                }
            )
        else:
            out[name] = tensor.contiguous()
    path = tmp_path / "int8.safetensors"
    save_file(out, str(path))
    return path, ref


def _load(monkeypatch, tmp_path, scheme, path):
    import sys
    monkeypatch.setitem(
        sys.modules,
        "utils.hf_xet_fallback",
        types.SimpleNamespace(hf_hub_download_with_xet_fallback = lambda *a, **k: str(path)),
    )
    return te.load_h3_quantized_text_encoder(
        "MiniMaxAI/MiniMax-H3",
        scheme,
        dtype = torch.bfloat16,
        local_base = str(tmp_path),
        local_files_only = True,
    )


def _hidden(model, ids):
    with torch.inference_mode():
        return (
            model.model(input_ids = ids, use_cache = False, output_hidden_states = True)
            .hidden_states[te.H3_TE_READ_LAYER]
            .float()
        )


def test_the_int8_file_still_loads_through_the_same_loader(monkeypatch, tmp_path):
    path, ref = _write_file(tmp_path)
    enc = _load(monkeypatch, tmp_path, "int8", path)
    assert enc is not None
    cls = te._int8_convrot_linear_class()
    assert sum(isinstance(m, cls) for m in enc.modules()) == 7 * te.H3_TE_READ_LAYER
    ids = torch.randint(0, 96, (1, 9))
    got, want = _hidden(enc, ids), _hidden(ref, ids)
    assert float((got - want).norm() / want.norm()) < 0.05


def test_a_file_under_the_wrong_scheme_is_refused(monkeypatch, tmp_path):
    path8, _ = _write_file(tmp_path)
    assert _load(monkeypatch, tmp_path, "q4_k_m", path8) is None
    other = tmp_path / "g"
    other.mkdir()
    pathg, _, _ = _write_gguf(other)
    assert _load(monkeypatch, other, "int8", pathg) is None


def _gguf_reader(monkeypatch, present: bool):
    import importlib.util
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: (object() if present else None) if name == "gguf" else real(name, *a),
    )


def test_an_unset_request_keeps_int8_on_every_host_while_the_switch_is_off(monkeypatch):
    monkeypatch.delenv(te.H3_TE_QUANT_AUTO_ENV, raising = False)
    _gguf_reader(monkeypatch, True)
    assert te.H3_TE_LOW_RAM_AUTO is False
    for gb in (10.0, 53.0, 200.0):
        assert te.h3_te_auto_scheme(host_capacity_gb = gb) == "int8"


def test_the_env_opts_in_to_q4_k_m(monkeypatch):
    monkeypatch.setenv(te.H3_TE_QUANT_AUTO_ENV, "Q4-K-M")
    assert te.h3_te_auto_scheme(host_capacity_gb = 500.0) == "q4_k_m"
    monkeypatch.setenv(te.H3_TE_QUANT_AUTO_ENV, "int8")
    assert te.h3_te_auto_scheme(host_capacity_gb = 10.0) == "int8"
    monkeypatch.setenv(te.H3_TE_QUANT_AUTO_ENV, "bogus")
    assert te.h3_te_auto_scheme(host_capacity_gb = 10.0) == "int8"


def test_the_switch_when_on_follows_host_ram(monkeypatch):
    monkeypatch.delenv(te.H3_TE_QUANT_AUTO_ENV, raising = False)
    monkeypatch.setattr(te, "H3_TE_LOW_RAM_AUTO", True)
    _gguf_reader(monkeypatch, True)
    assert te.h3_te_auto_scheme(host_capacity_gb = 200.0) == "int8"
    assert te.h3_te_auto_scheme(host_capacity_gb = te.H3_TE_INT8_HOST_SET_GB + 0.1) == "int8"
    assert te.h3_te_auto_scheme(host_capacity_gb = 53.0) == te.H3_TE_LOW_RAM_SCHEME
    assert te.h3_te_auto_scheme(host_capacity_gb = 66.0, denoiser_gb = 30.0) == te.H3_TE_LOW_RAM_SCHEME
    assert te.H3_TE_LOW_RAM_SCHEME in te.H3_TE_AUTO_ONLY_SCHEMES
    # no gguf module: int8 stays
    _gguf_reader(monkeypatch, False)
    assert te.h3_te_auto_scheme(host_capacity_gb = 53.0) == "int8"


def test_the_pick_reaches_the_video_resolver(monkeypatch):
    from core.inference.video import VideoBackend

    fam = types.SimpleNamespace(
        name = "minimax-h3", modular_workflow = "fl2va", base_repo = "MiniMaxAI/MiniMax-H3"
    )
    target = types.SimpleNamespace(
        device = "cuda", dtype = torch.bfloat16, supports_default_torch_compile = True
    )
    monkeypatch.delenv(te.H3_TE_QUANT_AUTO_ENV, raising = False)
    assert VideoBackend._h3_te_quant_scheme(fam, None, "MiniMaxAI/MiniMax-H3", target) == "int8"
    monkeypatch.setenv(te.H3_TE_QUANT_AUTO_ENV, "q4_k_m")
    assert VideoBackend._h3_te_quant_scheme(fam, None, "MiniMaxAI/MiniMax-H3", target) == "q4_k_m"
    assert VideoBackend._h3_te_quant_scheme(fam, "int8", "MiniMaxAI/MiniMax-H3", target) == "int8"
    assert VideoBackend._h3_te_quant_scheme(fam, "nvfp4", "MiniMaxAI/MiniMax-H3", target) is None


def test_the_4bit_file_is_not_a_request_value_and_is_sized():
    from core.inference.diffusion_precision import normalize_te_quant

    with pytest.raises(ValueError):
        normalize_te_quant(te.H3_TE_Q4_K_M)
    assert set(te.H3_TE_QUANT_FILES) == {"int8", "q4_k_m"}
    assert te.h3_te_resident_gb("q4_k_m", bf16_gb = 66.7) < te.h3_te_resident_gb("int8", bf16_gb = 66.7)


def _write_gguf(tmp_path):
    """Tiny GGUF in the hosted naming; Q8_0 since gguf-py cannot write Q4_K, which takes the same path."""
    gguf = pytest.importorskip("gguf")
    import numpy as np
    import transformers

    torch.manual_seed(3)
    config = _tiny_config(32)
    (tmp_path / "text_encoder").mkdir()
    config.save_pretrained(tmp_path / "text_encoder")
    ref = transformers.Qwen3VLForConditionalGeneration(config).to(torch.bfloat16).eval()
    writer = gguf.GGUFWriter(str(tmp_path / "te.gguf"), "qwen3vl")
    quantized = 0
    for key, tensor in ref.state_dict().items():
        if key.startswith("lm_head") or key.endswith("language_model.norm.weight"):
            continue
        if ".layers." in key and "language_model" in key:
            if int(key.split("layers.")[1].split(".")[0]) >= te.H3_TE_READ_LAYER:
                continue
        name = _layout_key_name(key)
        if "language_model.layers." in key and key.endswith("_proj.weight"):
            q = gguf.quants.quantize(tensor.float().numpy(), gguf.GGMLQuantizationType.Q8_0)
            writer.add_tensor(name, q, raw_dtype = gguf.GGMLQuantizationType.Q8_0)
            dense = gguf.quants.dequantize(q, gguf.GGMLQuantizationType.Q8_0)
            with torch.no_grad():
                ref.get_parameter(key).copy_(
                    torch.from_numpy(np.ascontiguousarray(dense)).to(torch.bfloat16)
                )
            quantized += 1
        elif key.endswith("embed_tokens.weight"):
            raw = tensor.contiguous().view(torch.int16).numpy().view(np.uint16)
            writer.add_tensor(
                name, raw, raw_shape = tuple(tensor.shape), raw_dtype = gguf.GGMLQuantizationType.BF16
            )
        else:
            arr = tensor.float()
            if (
                arr.dim() > 4
            ):  # GGUF holds 4 dimensions: the hosted file merges the patch conv's first two
                arr = arr.reshape(-1, *arr.shape[2:])
            writer.add_tensor(name, arr.numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return tmp_path / "te.gguf", ref, quantized


def test_the_gguf_conditioner_loads_as_gguf_linears(monkeypatch, tmp_path):
    pytest.importorskip(
        "diffusers.quantizers.gguf.utils"
    )  # the GGUF projections are diffusers' GGUFLinear
    path, ref, quantized = _write_gguf(tmp_path)
    enc = _load(monkeypatch, tmp_path, "q4_k_m", path)
    assert enc is not None
    from diffusers.quantizers.gguf.utils import GGUFLinear

    from core.inference.diffusion_weight_only_quant import is_dense_linear

    swapped = [m for m in enc.modules() if isinstance(m, GGUFLinear)]
    assert len(swapped) == quantized == 7 * te.H3_TE_READ_LAYER
    assert all(m.weight.dtype == torch.uint8 for m in swapped)
    assert not [n for n, m in enc.model.language_model.layers.named_modules() if is_dense_linear(m)]
    assert enc.get_input_embeddings().weight.dtype == torch.bfloat16
    ids = torch.randint(0, 96, (2, 6))
    got, want = _hidden(enc, ids), _hidden(ref, ids)
    assert float((got - want).norm() / want.norm()) < 0.02


def test_the_gguf_conditioner_comes_from_the_gguf_mirror():
    assert te.h3_te_quant_source("q4_k_m") == "unsloth/MiniMax-H3-GGUF"
    assert te.h3_te_quant_filename("q4_k_m") == "qwen3vl_32b_minimax_h3-Q4_K_M.gguf"
    from core.inference.video_minimax_h3 import H3_GGUF_REPO, H3_QWEN_Q4

    assert (te.h3_te_quant_source("q4_k_m"), te.h3_te_quant_filename("q4_k_m")) == (
        H3_GGUF_REPO,
        H3_QWEN_Q4,
    )
    assert "q4_k_m" in te.H3_TE_AUTO_ONLY_SCHEMES
