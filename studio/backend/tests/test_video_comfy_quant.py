# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ComfyUI-format quantized VIDEO denoisers: Wan / HunyuanVideo single files on Studio's int8 / fp8 runtimes,
LTX-2.3's bundled file, and MiniMax-H3's converter-less layout (rename + row-only splits + curve adaLN)."""

from __future__ import annotations

import json
import types

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")
from torch import nn  # noqa: E402

import core.inference.diffusion_comfy_quant as cq  # noqa: E402
from core.inference import video as vid  # noqa: E402
from core.inference import video_minimax_h3_comfy as h3c  # noqa: E402

DIM = 512


def _conf(**conf) -> torch.Tensor:
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype = torch.uint8)


def _int8(weight: torch.Tensor) -> tuple:
    w = weight.float()
    scale = w.abs().amax(dim = 1, keepdim = True).clamp_min(1e-8) / 127.0
    return torch.round(w / scale).clamp(-127, 127).to(torch.int8), scale


def _save(
    path,
    tensors,
    metadata = None,
) -> str:
    safetensors_torch.save_file(
        {k: v.contiguous().clone() for k, v in tensors.items()}, str(path), metadata = metadata
    )
    return str(path)


def _cos(a, b) -> float:
    return torch.nn.functional.cosine_similarity(
        a.flatten().float(), b.flatten().float(), dim = 0
    ).item()


def test_h3_key_map_renames_and_splits_rows_only():
    m = h3c.h3_comfy_key_map
    assert m("rope.inv_freq", (16,)) == []
    assert m("adaln_t_table", (1025, 8)) == [("time_embedder.table", None)]
    assert m("blocks.3.attn.out_proj.weight", (5376, 7168)) == [
        ("transformer_blocks.3.attn.to_out.0.weight", None)
    ]
    assert m("token_refiner.blocks.1.attn.q_norm.weight", (128,)) == [
        ("token_refiner.refiner_blocks.1.attn.norm_q.weight", None)
    ]
    assert m("final_layer.adaln_proj.linear.bias", (10752,)) == [("norm_out.linear.bias", None)]
    assert m("condition_proj.weight", (5376, 5120)) == [("context_embedder.weight", None)]
    assert m("blocks.0.mlp.fc2.weight", (5376, 14336)) == [
        ("transformer_blocks.0.ff.net.2.weight", None)
    ]
    assert m("blocks.0.attn.qkv_proj.weight", (21504, 5376)) == [
        ("transformer_blocks.0.attn.to_q.weight", [(0, 7168)]),
        ("transformer_blocks.0.attn.to_k.weight", [(7168, 7168)]),
        ("transformer_blocks.0.attn.to_v.weight", [(14336, 7168)]),
    ]
    # [gate; value] -> SwiGLU's [value; gate]
    assert m("blocks.0.mlp.fc1.weight", (28672, 5376)) == [
        ("transformer_blocks.0.ff.net.0.proj.weight", [(14336, 14336), (0, 14336)])
    ]
    with pytest.raises(ValueError, match = "three equal"):
        m("blocks.0.attn.qkv_proj.weight", (10, 4))


def test_h3_keeps_the_pruned_model_float32_tensors():
    assert h3c.h3_comfy_keep_dtype("blocks.7.adaln_proj.linear.weight") is torch.float32
    assert h3c.h3_comfy_keep_dtype("adaln_t_table") is torch.float32
    assert h3c.h3_comfy_keep_dtype("final_layer.video_out.bias") is torch.float32
    assert h3c.h3_comfy_keep_dtype("blocks.7.attn.qkv_proj.weight") is None
    assert h3c.h3_comfy_keep_dtype("condition_proj.weight") is None


@pytest.mark.parametrize(
    "name, ok, task",
    [
        ("minimax_h3_fl2va_pruned_int8_convrot.safetensors", True, "fl2va"),
        ("minimax_h3_ref2va_pruned_fp8_scaled.safetensors", True, "ref2va"),
        ("MiniMax-H3-Ref2VA-INT8-ConvRot-comfy.safetensors", True, "ref2va"),
        ("MiniMax-H3-INT8-ConvRot-comfy.safetensors", True, "fl2va"),
        ("qwen3vl_32b_minimax_h3_int8_convrot.safetensors", False, None),
        ("minimax_h3_video_vae_int8_convrot.safetensors", False, None),
        ("minimax_h3_fl2va_pruned-Q4_K.gguf", False, None),
        ("wan2.2_ti2v_5B_fp16.safetensors", False, None),
        ("minimax_h3_fl2va_pruned_bf16.safetensors", True, "fl2va"),
        ("minimax_h3_fl2va_pruned_nvfp4.safetensors", False, None),
        ("minimax_h3_fl2va_pruned_w6a8.safetensors", False, None),
    ],
)
def test_h3_single_file_names(name, ok, task):
    assert h3c.is_h3_comfy_name(name) is ok
    if task:
        assert h3c.h3_comfy_task(name) == task


def test_h3_curve_metadata_comes_from_the_table_shape(tmp_path):
    path = _save(tmp_path / "h3.safetensors", {"adaln_t_table": torch.zeros(1025, 8)})
    meta = h3c.h3_comfy_curve_metadata(path)
    assert meta["curve_grid"] == 1025 and meta["curve_dim"] == 8 and meta["adaln_form"] == "curve"
    dense = _save(tmp_path / "dense.safetensors", {"blocks.0.norm1.weight": torch.zeros(4)})
    assert h3c.h3_comfy_curve_metadata(dense) is None


class _Attn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q, self.to_k, self.to_v = (nn.Linear(DIM, DIM, bias = False) for _ in range(3))


class _FF(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(DIM, 2 * DIM, bias = False)


class _H3Like(nn.Module):
    """Diffusers-side names; the file below stores ComfyUI's fused qkv and [gate; value] fc1."""

    _keep_in_fp32_modules = None
    _keys_to_ignore_on_load_unexpected = None

    def __init__(self, dim: int = DIM) -> None:
        super().__init__()
        self.attn = _Attn()
        self.ff = _FF()
        self.head = nn.Linear(dim, 16)

    @classmethod
    def load_config(cls, repo, **_kwargs):
        return {"dim": DIM}

    @staticmethod
    def _get_signature_keys(_cls):
        return {"dim"}, set()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def _h3like_map(key, shape):
    rows = shape[0] if shape else 0
    if key == "attn.qkv.weight":
        t = rows // 3
        return [(f"attn.to_{p}.weight", [(i * t, t)]) for i, p in enumerate("qkv")]
    if key == "fc1.weight":
        h = rows // 2
        return [("ff.proj.weight", [(h, h), (0, h)])]
    return [(key, None)]


@pytest.fixture
def h3like_file(tmp_path):
    torch.manual_seed(3)
    dense = _H3Like()
    qkv = torch.cat(
        [dense.attn.to_q.weight, dense.attn.to_k.weight, dense.attn.to_v.weight]
    ).detach()
    value, gate = dense.ff.proj.weight.detach().chunk(2)
    fc1 = torch.cat([gate, value])  # ComfyUI's order
    tensors = {
        "head.weight": dense.head.weight.detach().half(),
        "head.bias": dense.head.bias.detach(),
    }
    for name, w in (("attn.qkv", qkv), ("fc1", fc1)):
        q, s = _int8(w)
        tensors[f"{name}.weight"], tensors[f"{name}.weight_scale"] = q, s
        tensors[f"{name}.comfy_quant"] = _conf(format = "int8_tensorwise")
    return _save(tmp_path / "h3like.safetensors", tensors), dense, tensors


def _load_h3like(path, **kwargs):
    prepared = []
    model = cq.load_comfy_quant_transformer(
        _H3Like,
        path,
        cq.refuse_comfy_quant(path),
        {"torch_dtype": torch.bfloat16, "config": "base/repo"},
        family = "minimax-h3",
        key_map = _h3like_map,
        prepare_model = prepared.append,
        keep_dtype = lambda key: torch.float32 if key.startswith("head.") else None,
        **kwargs,
    )
    return model, prepared


def test_key_map_dequant_path_rebuilds_the_dense_layers(h3like_file):
    path, dense, _ = h3like_file
    model, prepared = _load_h3like(path, int8_backend = None)
    assert len(prepared) == 1  # the hook ran on the freshly built model
    assert model._unsloth_comfy_quant["dequantized"] == 2
    for part in ("to_q", "to_k", "to_v"):
        assert _cos(getattr(model.attn, part).weight, getattr(dense.attn, part).weight) > 0.9999
    assert _cos(model.ff.proj.weight, dense.ff.proj.weight) > 0.9999  # halves swapped back
    # keep_dtype: the file's float16 head widens to float32 instead of narrowing to the compute dtype
    assert model.head.weight.dtype is torch.float32 and model.head.bias.dtype is torch.float32


def test_key_map_native_backend_keeps_codes_row_exact(h3like_file):
    from core.inference.diffusion_native_quant import is_native_linear

    path, _dense, tensors = h3like_file
    model, _ = _load_h3like(path, int8_backend = "native")
    assert model._unsloth_comfy_quant["int8"] == 4
    q, s = tensors["attn.qkv.weight"], tensors["attn.qkv.weight_scale"]
    for i, part in enumerate(("to_q", "to_k", "to_v")):
        linear = getattr(model.attn, part)
        assert is_native_linear(linear)
        assert torch.equal(linear.weight_q, q[i * DIM : (i + 1) * DIM])
        assert torch.equal(linear.weight_scale.view(torch.float32), s[i * DIM : (i + 1) * DIM, 0])
    fq, fs = tensors["fc1.weight"], tensors["fc1.weight_scale"]
    proj = model.ff.proj
    assert torch.equal(proj.weight_q, torch.cat([fq[DIM:], fq[:DIM]]))
    assert torch.equal(proj.weight_scale.view(torch.float32), torch.cat([fs[DIM:], fs[:DIM]])[:, 0])


def test_key_map_torchao_backend_keeps_codes(h3like_file):
    pytest.importorskip("torchao")
    path, _dense, tensors = h3like_file
    model, _ = _load_h3like(path, int8_backend = "torchao")
    if not model._unsloth_comfy_quant["int8"]:
        pytest.skip("this torchao has no Int8Tensor")
    q = tensors["attn.qkv.weight"]
    assert torch.equal(model.attn.to_v.weight.qdata, q[2 * DIM :])


def test_a_key_map_row_selection_outside_the_tensor_is_refused(h3like_file):
    path, _dense, _ = h3like_file
    with pytest.raises(ValueError, match = "outside"):
        cq.load_comfy_quant_transformer(
            _H3Like,
            path,
            cq.refuse_comfy_quant(path),
            {"torch_dtype": torch.bfloat16, "config": "base/repo"},
            int8_backend = None,
            key_map = lambda key, shape: [(key, [(0, int(shape[0]) + 1)])]
            if shape
            else [(key, None)],
        )


def test_keep_key_reads_only_the_denoiser_and_refuses_quantized_companions(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cq,
        "_mapping",
        lambda cls: (
            lambda checkpoint = None, **_k: checkpoint,
            types.SimpleNamespace(
                _should_convert_state_dict_to_diffusers = lambda *_a: False,
                _get_mapping_function_kwargs = lambda fn, **kw: {},
            ),
        ),
    )
    torch.manual_seed(4)
    dense = _H3Like()
    tensors = {f"model.diffusion_model.{k}": v.detach() for k, v in dense.state_dict().items()}
    q, s = _int8(dense.attn.to_q.weight)
    tensors["model.diffusion_model.attn.to_q.weight"] = q
    tensors["model.diffusion_model.attn.to_q.weight_scale"] = s
    tensors["model.diffusion_model.attn.to_q.comfy_quant"] = _conf(format = "int8_tensorwise")
    tensors["vae.decoder.weight"] = torch.ones(3)
    path = _save(tmp_path / "bundle.safetensors", tensors)
    seen = []

    def pre(state):
        seen.extend(state)
        return {k.removeprefix("model.diffusion_model."): v for k, v in state.items()}

    model = cq.load_comfy_quant_transformer(
        _H3Like,
        path,
        cq.refuse_comfy_quant(path),
        {"torch_dtype": torch.float32, "config": "base/repo"},
        int8_backend = None,
        keep_key = lambda k: k.startswith("model.diffusion_model."),
        pre_convert = pre,
    )
    assert not any(k.startswith("vae.") for k in seen)
    assert _cos(model.attn.to_q.weight, dense.attn.to_q.weight) > 0.9999
    # a quantized layer the keep_key filters out would be silently dropped: refused instead
    tensors["vae.decoder.weight"] = q
    tensors["vae.decoder.weight_scale"] = s
    tensors["vae.decoder.comfy_quant"] = _conf(format = "int8_tensorwise")
    path = _save(tmp_path / "bundle_q.safetensors", tensors)
    with pytest.raises(ValueError, match = "outside the denoiser"):
        cq.load_comfy_quant_transformer(
            _H3Like,
            path,
            cq.refuse_comfy_quant(path),
            {"torch_dtype": torch.float32, "config": "base/repo"},
            int8_backend = None,
            keep_key = lambda k: k.startswith("model.diffusion_model."),
            pre_convert = pre,
        )


def test_ltx23_pre_convert_strips_the_prefix_and_renames_23_keys():
    from core.inference.video_ltx2 import (
        _ltx23_pre_convert,
        ltx23_is_dit_key,
        ltx23_is_dit_or_connector_key,
    )

    out = _ltx23_pre_convert(
        {
            "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight": 1,
            "model.diffusion_model.prompt_adaln_single.linear.weight": 2,
            "audio_prompt_adaln_single.linear.bias": 3,
        }
    )
    assert out == {
        "transformer_blocks.0.attn1.to_q.weight": 1,
        "prompt_adaln.linear.weight": 2,
        "audio_prompt_adaln.linear.bias": 3,
    }
    assert ltx23_is_dit_key("model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight")
    assert not ltx23_is_dit_key("vae.decoder.conv_in.weight")
    assert not ltx23_is_dit_key("model.diffusion_model.video_embeddings_connector.x.weight")
    assert ltx23_is_dit_or_connector_key(
        "model.diffusion_model.video_embeddings_connector.x.weight"
    )
    assert not ltx23_is_dit_or_connector_key("vocoder.conv.weight")


def test_resident_mib_prices_kept_layers_as_stored_and_the_rest_at_bf16(tmp_path):
    rows, cols = 2048, 1024
    q = torch.zeros(rows, cols, dtype = torch.int8)
    tensors = {
        "a.weight": q,
        "a.weight_scale": torch.ones(rows, 1),
        "a.comfy_quant": _conf(format = "int8_tensorwise"),
        "audio.weight": q,
        "audio.weight_scale": torch.ones(rows, 1),
        "audio.comfy_quant": _conf(format = "int8_tensorwise"),
        "b.weight": torch.zeros(rows, cols, dtype = torch.float8_e4m3fn),
        "b.weight_scale": torch.ones(()),
        "b.comfy_quant": _conf(format = "float8_e4m3fn"),
        "n.weight": torch.zeros(rows, cols, dtype = torch.float32),
        "vae.w": torch.zeros(rows, cols, dtype = torch.bfloat16),
    }
    path = _save(tmp_path / "plan.safetensors", tensors)
    scan = cq.refuse_comfy_quant(path)
    n = rows * cols
    small = 8 * rows + 4 + sum(v.numel() for k, v in tensors.items() if k.endswith(".comfy_quant"))

    def mib(total):
        return -(-total // (1024 * 1024))

    # int8 kept (minus the excluded name), fp8 dequantized, fp32 narrowed, VAE out of the read
    got = cq.comfy_resident_mib(
        path,
        scan,
        keep_int8 = True,
        keep_fp8 = False,
        exclude_tokens = ("audio",),
        keep_key = lambda k: not k.startswith("vae."),
    )
    assert got == mib(n + 2 * n + 2 * n + 2 * n + small)
    everything = cq.comfy_resident_mib(path, scan, keep_int8 = True, keep_fp8 = True)
    assert everything == mib(3 * n + 2 * n + 2 * n + small)
    # nothing kept: every quantized layer at bf16, twice its stored size
    assert cq.comfy_resident_mib(path, scan, keep_int8 = False, keep_fp8 = False) == mib(
        3 * 2 * n + 2 * n + 2 * n + small
    )


def _plan(resident: bool):
    return types.SimpleNamespace(resident = resident)


def test_video_backends_follow_studio_rules(monkeypatch):
    monkeypatch.setattr(vid, "plan_keeps_transformer_resident", lambda p: p.resident)
    calls = []

    def int8(
        target,
        family,
        base,
        *,
        offload = False,
    ):
        calls.append(("int8", offload))
        return "native" if offload else "torchao"

    def fp8(
        target,
        family,
        base,
        *,
        offload = False,
    ):
        calls.append(("fp8", offload))
        return None if offload else "torchao"

    monkeypatch.setattr(vid, "comfy_int8_backend", int8)
    monkeypatch.setattr(vid, "comfy_fp8_backend", fp8)
    fam = types.SimpleNamespace(name = "wan2.2-ti2v-5b")
    assert vid._video_comfy_backends(fam, "b", None, _plan(True), _plan(True)) == {
        "int8_backend": "torchao",
        "fp8_backend": "torchao",
    }
    # either plan offloading the DiT means the hooks move it: int8 takes the torchao-free twin, fp8 dequantizes
    assert vid._video_comfy_backends(fam, "b", None, _plan(True), _plan(False)) == {
        "int8_backend": "native",
        "fp8_backend": None,
    }
    assert vid._video_comfy_backends(fam, "b", None, _plan(True), keep = False) == {
        "int8_backend": None,
        "fp8_backend": None,
    }


def test_video_plan_size_prices_only_what_runs_quantized(monkeypatch, tmp_path):
    rows, cols = 1024, 1024
    tensors = {
        "x.weight": torch.zeros(rows, cols, dtype = torch.float8_e4m3fn),
        "x.weight_scale": torch.ones(()),
        "x.comfy_quant": _conf(format = "float8_e4m3fn"),
    }
    path = _save(tmp_path / "f.safetensors", tensors)
    scan = cq.refuse_comfy_quant(path)
    fam = types.SimpleNamespace(name = "wan2.2-ti2v-5b")
    monkeypatch.setattr(vid, "comfy_int8_backend", lambda *a, **k: None)
    monkeypatch.setattr(vid, "comfy_fp8_backend", lambda *a, **k: "torchao")
    n = rows * cols
    assert (
        vid._video_comfy_resident_mib(fam, "b", None, path, scan) == 2
    )  # 1 MiB of codes + scale + config
    monkeypatch.setattr(vid, "comfy_fp8_backend", lambda *a, **k: None)
    assert vid._video_comfy_resident_mib(fam, "b", None, path, scan) == 3
    assert vid._video_comfy_resident_mib(fam, "b", None, path, scan, keep = False) == 3
    assert n == 2**20


def test_h3_single_file_validation_accepts_a_comfy_denoiser_and_refuses_the_rest():
    from core.inference.video import VideoBackend

    backend = VideoBackend()
    for name in (
        "qwen3vl_32b_minimax_h3_bf16.safetensors",
        "minimax_h3_video_vae_fp16.safetensors",
    ):
        with pytest.raises(ValueError, match = "single .safetensors checkpoint"):
            backend.validate_load_request(
                "MiniMaxAI/MiniMax-H3", gguf_filename = name, model_kind = "single_file"
            )
    try:
        backend.validate_load_request(
            "MiniMaxAI/MiniMax-H3",
            gguf_filename = "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            model_kind = "single_file",
        )
    except (
        ValueError,
        ImportError,
    ) as exc:  # later probes may refuse or need diffusers; never the modular refusal
        assert "single .safetensors checkpoint" not in str(exc)


def test_hv15_key_map_renames_splits_and_swaps_rows_only():
    from core.inference.video_hv15_comfy import hv15_comfy_key_map as m

    assert m("double_blocks.3.img_attn_qkv.weight", (6144, 2048)) == [
        ("transformer_blocks.3.attn.to_q.weight", [(0, 2048)]),
        ("transformer_blocks.3.attn.to_k.weight", [(2048, 2048)]),
        ("transformer_blocks.3.attn.to_v.weight", [(4096, 2048)]),
    ]
    assert m("double_blocks.3.txt_attn_qkv.bias", (6144,)) == [
        ("transformer_blocks.3.attn.add_q_proj.bias", [(0, 2048)]),
        ("transformer_blocks.3.attn.add_k_proj.bias", [(2048, 2048)]),
        ("transformer_blocks.3.attn.add_v_proj.bias", [(4096, 2048)]),
    ]
    assert m("double_blocks.0.img_mod.linear.weight", (12288, 2048)) == [
        ("transformer_blocks.0.norm1.linear.weight", None)
    ]
    assert m("double_blocks.0.txt_mlp.fc2.weight", (2048, 8192)) == [
        ("transformer_blocks.0.ff_context.net.2.weight", None)
    ]
    assert m("txt_in.individual_token_refiner.blocks.1.self_attn_qkv.weight", (6144, 2048))[2] == (
        "context_embedder.token_refiner.refiner_blocks.1.attn.to_v.weight",
        [(4096, 2048)],
    )
    assert m("txt_in.individual_token_refiner.blocks.1.adaLN_modulation.1.bias", (4096,)) == [
        ("context_embedder.token_refiner.refiner_blocks.1.norm_out.linear.bias", None)
    ]
    # [shift; scale] -> diffusers' [scale; shift]
    assert m("final_layer.adaLN_modulation.1.weight", (4096, 2048)) == [
        ("norm_out.linear.weight", [(2048, 2048), (0, 2048)])
    ]
    assert m("vision_in.proj.3.weight", (2048, 1152)) == [("image_embedder.linear_2.weight", None)]
    assert m("img_in.proj.weight", (2048, 65, 1, 1, 1)) == [("x_embedder.proj.weight", None)]
    assert m("cond_type_embedding.weight", (3, 2048)) == [("cond_type_embed.weight", None)]
    assert m("transformer_blocks.0.attn.to_q.weight", (2048, 2048)) == [
        ("transformer_blocks.0.attn.to_q.weight", None)
    ]
    with pytest.raises(ValueError, match = "not a HunyuanVideo-1.5"):
        m("double_blocks.0.mystery.weight", (4, 4))
    with pytest.raises(ValueError, match = "not a HunyuanVideo-1.5"):
        m("something_else.weight", (4, 4))


def test_hv15_is_the_family_with_a_key_map():
    assert (
        vid._video_comfy_key_map(
            types.SimpleNamespace(transformer_class = "HunyuanVideo15Transformer3DModel")
        )
        is not None
    )
    assert (
        vid._video_comfy_key_map(types.SimpleNamespace(transformer_class = "WanTransformer3DModel"))
        is None
    )


def test_original_layouts_are_found_by_class_name(tmp_path):
    hv = type("HunyuanVideo15Transformer3DModel", (), {})
    from core.inference.video_hv15_comfy import hv15_comfy_key_map

    assert cq.original_layout(hv, "x.safetensors")["key_map"] is hv15_comfy_key_map
    path = _save(tmp_path / "h3.safetensors", {"adaln_t_table": torch.zeros(1025, 8)})
    h3 = cq.original_layout(type("MiniMaxH3Transformer3DModel", (), {}), path)
    assert h3["key_map"] is h3c.h3_comfy_key_map
    assert h3["keep_dtype"].func is h3c.h3_comfy_keep_dtype
    assert h3["keep_dtype"].keywords == {"pruned": True}
    assert callable(h3["prepare_model"])
    assert cq.original_layout(type("WanTransformer3DModel", (), {}), path) is None


def test_a_class_without_a_converter_uses_its_registered_layout(h3like_file, monkeypatch):
    """Any caller (a hosted ComfyUI-format twin included) gets the layout without passing a key map."""
    path, dense, _ = h3like_file
    monkeypatch.setattr(
        cq,
        "original_layout",
        lambda cls, p: {"key_map": _h3like_map, "keep_dtype": None, "prepare_model": None}
        if cls is _H3Like
        else None,
    )
    model = cq.load_comfy_quant_transformer(
        _H3Like,
        path,
        cq.refuse_comfy_quant(path),
        {"torch_dtype": torch.float32, "config": "base/repo"},
        int8_backend = None,
    )
    assert _cos(model.ff.proj.weight, dense.ff.proj.weight) > 0.9999


def test_h3_single_file_task_comes_from_the_file_name_unless_requested():
    from core.inference.video import _h3_single_file_task
    from core.inference.video_families import detect_video_family

    h3 = detect_video_family("MiniMaxAI/MiniMax-H3")
    ref = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    assert _h3_single_file_task(h3, "single_file", ref, None) == "ref2va"
    assert _h3_single_file_task(h3, "single_file", ref, "fl2va") == "fl2va"
    assert _h3_single_file_task(h3, "pipeline", None, None) is None
    wan = detect_video_family("Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    assert _h3_single_file_task(wan, "single_file", ref, None) is None


def test_h3_single_file_is_refused_on_metal(monkeypatch):
    from core.inference.diffusion_device import DiffusionDeviceTarget
    from core.inference.video import VideoBackend

    monkeypatch.setattr(
        "core.inference.video.resolve_diffusion_device_target",
        lambda: DiffusionDeviceTarget(
            device = "mps",
            dtype = None,
            backend = "mps",
            vendor = None,
            supports_model_cpu_offload = False,
            supports_default_torch_compile = False,
            supports_pinned_transfer = False,
            supports_float64 = False,
        ),
    )
    with pytest.raises(ValueError, match = "cannot run on Apple Silicon"):
        VideoBackend().validate_load_request(
            "MiniMaxAI/MiniMax-H3",
            gguf_filename = "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            model_kind = "single_file",
        )


def test_h3_single_file_conditioner_skips_the_generic_precision_gate(monkeypatch):
    from core.inference.video import assert_video_precision_available
    from core.inference.video_families import detect_video_family

    fam = detect_video_family("MiniMaxAI/MiniMax-H3")
    monkeypatch.setattr(
        "core.inference.video.precision_fallback_allowed", lambda: False, raising = False
    )
    monkeypatch.setattr(
        "core.inference.video.te_quant_supported", lambda *_a, **_k: False, raising = False
    )
    assert_video_precision_available(
        fam,
        model_kind = "single_file",
        text_encoder_quant = "int8",
        checkpoint_filename = "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    )


def test_resident_mib_prices_layers_the_runtime_filter_skips_at_bf16(tmp_path):
    big, small, ragged = (2048, 1024), (2048, 64), (2048, 1000)
    tensors = {}
    for name, shape, fmt, dtype in (
        ("i_big", big, "int8_tensorwise", torch.int8),
        ("i_small", small, "int8_tensorwise", torch.int8),
        ("f_ragged", ragged, "float8_e4m3fn", torch.float8_e4m3fn),
    ):
        tensors[f"{name}.weight"] = torch.zeros(shape, dtype = dtype)
        tensors[f"{name}.weight_scale"] = torch.ones(shape[0], 1)
        tensors[f"{name}.comfy_quant"] = _conf(format = fmt)
    path = _save(tmp_path / "filter.safetensors", tensors)
    scan = cq.refuse_comfy_quant(path)
    kw = dict(keep_int8 = True, keep_fp8 = True)
    loose = cq.comfy_resident_mib(path, scan, **kw)
    strict = cq.comfy_resident_mib(path, scan, **kw, min_features = 128, fp8_divisible = 16)
    extra = 2048 * 64 + 2048 * 1000  # i_small and f_ragged: stored 1 B, priced 2 B
    assert strict * 1024 * 1024 - loose * 1024 * 1024 >= extra - 1024 * 1024
    assert strict > loose


def test_a_checkpoint_buffer_keeps_its_keep_dtype(h3like_file, tmp_path):
    path, _dense, tensors = h3like_file
    table = torch.randn(9, 4, dtype = torch.float64)
    path = _save(tmp_path / "h3like_table.safetensors", {**tensors, "table": table.half()})

    def add_table(model):
        model.register_buffer("table", torch.empty(9, 4, dtype = torch.float32))

    model = cq.load_comfy_quant_transformer(
        _H3Like,
        path,
        cq.refuse_comfy_quant(path),
        {"torch_dtype": torch.bfloat16, "config": "base/repo"},
        family = "minimax-h3",
        key_map = _h3like_map,
        prepare_model = add_table,
        keep_dtype = lambda key: torch.float32
        if key in ("table",) or key.startswith("head.")
        else None,
        int8_backend = None,
    )
    assert model.table.dtype is torch.float32
    assert torch.equal(model.table, table.half().float())


def test_offloaded_plan_size_prices_fp8_that_only_runs_resident(monkeypatch, tmp_path):
    rows = cols = 1024
    tensors = {
        "x.weight": torch.zeros(rows, cols, dtype = torch.float8_e4m3fn),
        "x.weight_scale": torch.ones(()),
        "x.comfy_quant": _conf(format = "float8_e4m3fn"),
    }
    path = _save(tmp_path / "f.safetensors", tensors)
    scan = cq.refuse_comfy_quant(path)
    fam = types.SimpleNamespace(name = "wan2.2-ti2v-5b")
    monkeypatch.setattr(vid, "comfy_int8_backend", lambda *a, **k: None)
    monkeypatch.setattr(
        vid, "comfy_fp8_backend", lambda *a, offload = False, **k: None if offload else "torchao"
    )
    assert vid._video_comfy_resident_mib(fam, "b", None, path, scan) == 2
    assert vid._video_comfy_resident_mib(fam, "b", None, path, scan, offload = True) == 3


def test_dense_h3_maps_its_timestep_mlp_and_keeps_adaln_at_compute_dtype():
    m = h3c.h3_comfy_key_map
    assert m("time_embedder.proj_in.weight", (5376, 256)) == [
        ("time_embedder.linear_1.weight", None)
    ]
    assert m("time_embedder.proj_out.bias", (2688,)) == [("time_embedder.linear_2.bias", None)]
    keep = h3c.h3_comfy_keep_dtype
    assert keep("time_embedder.proj_in.weight", pruned = False) is torch.float32
    assert keep("blocks.3.adaln_proj.linear.weight", pruned = False) is None
    assert keep("final_layer.adaln_proj.linear.bias", pruned = False) is None
    assert keep("blocks.3.adaln_proj.linear.weight") is torch.float32
    assert keep("final_layer.video_out.weight", pruned = False) is torch.float32


def test_h3_single_file_refuses_a_conflicting_partition_before_staging():
    from core.inference.video import VideoBackend
    with pytest.raises(ValueError, match = "ref2va partition"):
        VideoBackend().validate_load_request(
            "MiniMaxAI/MiniMax-H3",
            gguf_filename = "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            model_kind = "single_file",
            h3_task = "fl2va",
        )


def test_resident_mib_excludes_by_the_converted_name(tmp_path):
    from core.inference.video_hv15_comfy import hv15_comfy_key_map

    shape = (2048, 1024)
    tensors = {
        "time_in.mlp.0.weight": torch.zeros(shape, dtype = torch.int8),
        "time_in.mlp.0.weight_scale": torch.ones(shape[0], 1),
        "time_in.mlp.0.comfy_quant": _conf(format = "int8_tensorwise"),
    }
    path = _save(tmp_path / "hv.safetensors", tensors)
    scan = cq.refuse_comfy_quant(path)
    kw = dict(keep_int8 = True, keep_fp8 = False, exclude_tokens = ("timestep_embed",))
    plain = cq.comfy_resident_mib(path, scan, **kw)
    mapped = cq.comfy_resident_mib(path, scan, **kw, key_map = hv15_comfy_key_map)
    assert (
        mapped > plain
    )  # time_embed.timestep_embedder.linear_1 is excluded, so it is priced at bf16


def test_offload_only_int8_is_priced_dense_for_a_resident_plan(monkeypatch, tmp_path):
    rows = cols = 1024
    tensors = {
        "x.weight": torch.zeros(rows, cols, dtype = torch.int8),
        "x.weight_scale": torch.ones(rows, 1),
        "x.comfy_quant": _conf(format = "int8_tensorwise"),
    }
    path = _save(tmp_path / "i.safetensors", tensors)
    scan = cq.refuse_comfy_quant(path)
    fam = types.SimpleNamespace(name = "wan2.2-ti2v-5b")
    monkeypatch.setattr(
        vid, "comfy_int8_backend", lambda *a, offload = False, **k: "native" if offload else None
    )
    monkeypatch.setattr(vid, "comfy_fp8_backend", lambda *a, **k: None)
    assert vid._video_comfy_resident_mib(fam, "b", None, path, scan) == 3
    assert vid._video_comfy_resident_mib(fam, "b", None, path, scan, offload = True) == 2
