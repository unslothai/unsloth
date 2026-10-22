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
        "fp8_storage": False,
        "fp8_widen_chosen": False,
    }
    # either plan offloading the DiT means the hooks move it: int8 takes the torchao-free twin, fp8 has no GEMM
    # and stays stored as fp8, widened per layer
    assert vid._video_comfy_backends(fam, "b", None, _plan(True), _plan(False)) == {
        "int8_backend": "native",
        "fp8_backend": None,
        "fp8_storage": True,
        "fp8_widen_chosen": False,
    }
    assert vid._video_comfy_backends(fam, "b", None, _plan(True), keep = False) == {
        "int8_backend": None,
        "fp8_backend": None,
        "fp8_storage": False,
        "fp8_widen_chosen": False,
    }
    h3 = types.SimpleNamespace(name = "minimax-h3")
    assert (
        vid._video_comfy_backends(h3, "b", None, _plan(True), _plan(False))["fp8_storage"] is False
    )
    monkeypatch.setenv(cq.FP8_STORAGE_ENV, "0")
    assert (
        vid._video_comfy_backends(fam, "b", None, _plan(True), _plan(False))["fp8_storage"] is False
    )


def test_a_smaller_offloaded_price_replans_and_keeps_the_offloaded_backends(monkeypatch):
    """fp8 stored under offload prices below the widened resident estimate: re-plan at it, and a plan that now fits
    resident keeps fp8 storage (else the load widens past the size it was planned at)."""
    fam = types.SimpleNamespace(name = "wan2.2-ti2v-5b")
    assert vid._video_offload_reprice(fam, 2000, 1000) == (1000, True)
    assert vid._video_offload_reprice(fam, 1000, 2000) == (2000, False)
    assert vid._video_offload_reprice(fam, 1000, 1000) == (1000, False)
    assert vid._video_offload_reprice(fam, None, 1000) == (None, False)
    h3 = types.SimpleNamespace(name = "minimax-h3")
    assert vid._video_offload_reprice(h3, 2000, 1000) == (2000, False)
    assert vid._video_offload_reprice(h3, 1000, 2000) == (2000, False)
    monkeypatch.setattr(vid, "plan_keeps_transformer_resident", lambda p: p.resident)
    monkeypatch.setattr(
        vid, "comfy_int8_backend", lambda t, f, b, *, offload = False: "native" if offload else None
    )
    monkeypatch.setattr(vid, "comfy_fp8_backend", lambda t, f, b, *, offload = False: None)
    resident = vid._video_comfy_backends(fam, "b", None, _plan(True), _plan(True))
    assert resident["fp8_storage"] is False and resident["int8_backend"] is None
    kept = vid._video_comfy_backends(fam, "b", None, _plan(True), _plan(True), priced_offload = True)
    assert kept == {
        "int8_backend": "native",
        "fp8_backend": None,
        "fp8_storage": True,
        "fp8_widen_chosen": False,
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
    # no fp8 GEMM under offload: the codes stay stored as fp8 (widened per layer), priced at their stored size
    assert vid._video_comfy_resident_mib(fam, "b", None, path, scan, offload = True) == 2
    monkeypatch.setenv(cq.FP8_STORAGE_ENV, "1")
    assert vid._video_comfy_resident_mib(fam, "b", None, path, scan) == 2
    monkeypatch.setenv(cq.FP8_STORAGE_ENV, "0")
    assert vid._video_comfy_resident_mib(fam, "b", None, path, scan, offload = True) == 3
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
    assert vid._video_comfy_resident_mib(fam, "b", None, path, scan, offload = True) == 2
    monkeypatch.setenv(cq.FP8_STORAGE_ENV, "0")
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


def _record_host_ram_checks(monkeypatch):
    """Every pre-allocation host RAM check a load makes, by size, on a host with room for all of them."""
    import core.inference.diffusion_single_file_converters as conv
    import core.inference.diffusion_small_host as small_host

    monkeypatch.setattr(conv, "HOST_RAM_CHECK_MIN_MIB", 0)
    monkeypatch.delenv(small_host.HOST_RAM_CHECK_ENV, raising = False)
    monkeypatch.setattr(small_host, "host_ram_mib", lambda: (1 << 20, 1 << 20))
    seen = []
    monkeypatch.setattr(conv.HostRamBudget, "charge", lambda self, n: seen.append(int(n)))
    return seen


def test_a_key_map_row_reorder_is_checked_before_it_is_copied(h3like_file, monkeypatch):
    """MiniMax-H3's fc1 swaps its [gate; value] halves, so the kept int8 codes are rebuilt with
    torch.cat: a full copy of the layer that must be checked against free host RAM first."""
    seen = _record_host_ram_checks(monkeypatch)
    path, _dense, tensors = h3like_file
    _load_h3like(path, int8_backend = "native")
    fc1 = tensors["fc1.weight"]
    assert fc1.numel() * fc1.element_size() in seen


def test_a_stock_converter_row_reorder_of_kept_codes_is_checked(h3like_file, monkeypatch):
    """No key map: a diffusers-style converter that swaps a gated layer's halves (Mochi's mlp_x.w1) makes
    the loader rebuild the kept int8 codes with torch.cat, a full copy checked before it is made."""

    def convert(checkpoint = None, **_kwargs):
        out = {}
        for key, value in checkpoint.items():
            if key == "attn.qkv.weight":
                for part, chunk in zip(("to_q", "to_k", "to_v"), value.chunk(3)):
                    out[f"attn.{part}.weight"] = chunk
            elif key == "fc1.weight":
                gate, val = value.chunk(2)
                out["ff.proj.weight"] = torch.cat([val, gate])
            else:
                out[key] = value
        return out

    monkeypatch.setattr(
        cq,
        "_mapping",
        lambda cls: (
            convert,
            types.SimpleNamespace(
                _should_convert_state_dict_to_diffusers = lambda a, b: set(a) != set(b),
                _get_mapping_function_kwargs = lambda fn, **kw: {},
            ),
        ),
    )
    seen = _record_host_ram_checks(monkeypatch)
    path, _dense, tensors = h3like_file
    model = cq.load_comfy_quant_transformer(
        _H3Like,
        path,
        cq.refuse_comfy_quant(path),
        {"torch_dtype": torch.bfloat16, "config": "base/repo"},
        family = "minimax-h3",
        int8_backend = "native",
        keep_dtype = lambda key: torch.float32 if key.startswith("head.") else None,
    )
    assert model._unsloth_comfy_quant["int8"] == 4
    fc1 = tensors["fc1.weight"]
    assert any(n >= fc1.numel() * fc1.element_size() for n in seen), seen


def test_a_host_ram_refusal_in_the_narrow_pass_is_not_retried_at_full_width(
    h3like_file, monkeypatch
):
    """The narrow-tag conversion pass retries at full width on any failure it cannot prove, but a host RAM
    refusal is final: the full-width float64 tags (and a converter's unchecked joins over them) would only
    allocate more."""
    import core.inference.diffusion_single_file_converters as conv
    import core.inference.diffusion_small_host as small_host

    calls = []

    def convert(checkpoint = None, **_kwargs):
        calls.append(1)
        out = {}
        for key, value in checkpoint.items():
            if key == "attn.qkv.weight":
                for part, chunk in zip(("to_q", "to_k", "to_v"), value.chunk(3)):
                    out[f"attn.{part}.weight"] = chunk
            elif key == "fc1.weight":
                gate, val = value.chunk(2)
                out["ff.proj.weight"] = torch.cat([val, gate])
            else:
                out[key] = value
        return out

    monkeypatch.setattr(
        cq,
        "_mapping",
        lambda cls: (
            convert,
            types.SimpleNamespace(
                _should_convert_state_dict_to_diffusers = lambda a, b: set(a) != set(b),
                _get_mapping_function_kwargs = lambda fn, **kw: {},
            ),
        ),
    )
    monkeypatch.setattr(conv, "HOST_RAM_CHECK_MIN_MIB", 0)
    monkeypatch.delenv(small_host.HOST_RAM_CHECK_ENV, raising = False)
    monkeypatch.setattr(small_host, "host_ram_mib", lambda: (1 << 20, 1 << 20))
    path, _dense, tensors = h3like_file
    fc1_bytes = tensors["fc1.weight"].numel()

    def charge(self, n):
        if n >= fc1_bytes:  # the kept fc1 reorder does not fit
            raise conv.HostRamShortfallError("Not enough system RAM to load h3like.safetensors")

    monkeypatch.setattr(conv.HostRamBudget, "charge", charge)
    with pytest.raises(conv.HostRamShortfallError):
        cq.load_comfy_quant_transformer(
            _H3Like,
            path,
            cq.refuse_comfy_quant(path),
            {"torch_dtype": torch.bfloat16, "config": "base/repo"},
            family = "minimax-h3",
            int8_backend = "native",
        )
    assert len(calls) == 1  # refused in the narrow pass, never converted again at full width


def test_the_up_front_estimate_skips_tensors_the_key_filter_never_reads(tmp_path, monkeypatch):
    """A bundled checkpoint (LTX-2.3 / LTX-2.5) is read through ``keep_key``: only the DiT. Its VAE stored in
    float32 must not be priced as widened to the compute dtype and refuse a DiT load that fits."""
    import core.inference.diffusion_single_file_converters as conv
    import core.inference.diffusion_small_host as small_host

    torch.manual_seed(3)
    dense = _H3Like()
    tensors = {"head.weight": dense.head.weight.detach(), "head.bias": dense.head.bias.detach()}
    for name, w in (("attn.to_q", dense.attn.to_q.weight), ("ff.proj", dense.ff.proj.weight)):
        q, s = _int8(w.detach())
        tensors[f"{name}.weight"], tensors[f"{name}.weight_scale"] = q, s
        tensors[f"{name}.comfy_quant"] = _conf(format = "int8_tensorwise")
    tensors["attn.to_k.weight"] = dense.attn.to_k.weight.detach()
    tensors["attn.to_v.weight"] = dense.attn.to_v.weight.detach()
    tensors["vae.decoder.weight"] = torch.zeros(2048, 2048)  # 16 MiB fp32, 8 MiB widened to bf16
    path = _save(tmp_path / "bundled.safetensors", tensors)
    monkeypatch.setattr(
        cq,
        "_mapping",
        lambda cls: (
            lambda checkpoint = None, **_kw: checkpoint,
            types.SimpleNamespace(
                _should_convert_state_dict_to_diffusers = lambda a, b: False,
                _get_mapping_function_kwargs = lambda fn, **kw: {},
            ),
        ),
    )
    monkeypatch.setattr(conv, "HOST_RAM_CHECK_MIN_MIB", 0)
    monkeypatch.delenv(small_host.HOST_RAM_CHECK_ENV, raising = False)
    total = 1 << 20
    monkeypatch.setattr(
        small_host, "host_ram_mib", lambda: (total, small_host.host_reserve_mib(total) + 4)
    )
    model = cq.load_comfy_quant_transformer(
        _H3Like,
        path,
        cq.refuse_comfy_quant(path),
        {"torch_dtype": torch.bfloat16, "config": "base/repo"},
        int8_backend = None,
        keep_key = lambda key: not key.startswith("vae."),
    )
    assert model.attn.to_q.weight.dtype == torch.bfloat16


def test_a_full_width_tag_retry_that_does_not_fit_is_refused_before_converting(
    h3like_file, monkeypatch
):
    """The narrow-tag pass failing for an ordinary reason retries at full float64 width, which a converter
    can materialize at 8 bytes per weight element before any checked helper runs: that peak is checked
    first, and a retry that does not fit is refused without calling the converter again."""
    import core.inference.diffusion_single_file_converters as conv
    import core.inference.diffusion_small_host as small_host

    calls = []

    def convert(checkpoint = None, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("narrow tags could not be proven")
        out = {}
        for key, value in checkpoint.items():
            if key == "attn.qkv.weight":
                for part, chunk in zip(("to_q", "to_k", "to_v"), value.chunk(3)):
                    out[f"attn.{part}.weight"] = chunk
            elif key == "fc1.weight":
                out["ff.proj.weight"] = value
            else:
                out[key] = value
        return out

    monkeypatch.setattr(
        cq,
        "_mapping",
        lambda cls: (
            convert,
            types.SimpleNamespace(
                _should_convert_state_dict_to_diffusers = lambda a, b: set(a) != set(b),
                _get_mapping_function_kwargs = lambda fn, **kw: {},
            ),
        ),
    )
    monkeypatch.setattr(conv, "HOST_RAM_CHECK_MIN_MIB", 0)
    monkeypatch.delenv(small_host.HOST_RAM_CHECK_ENV, raising = False)
    monkeypatch.setattr(small_host, "host_ram_mib", lambda: (1 << 20, 1 << 20))
    path, _dense, tensors = h3like_file
    tagged = tensors["attn.qkv.weight"].numel() + tensors["fc1.weight"].numel()

    def charge(self, n):
        if n >= 8 * tagged:  # the full-width tags do not fit
            raise conv.HostRamShortfallError("Not enough system RAM to load h3like.safetensors")

    monkeypatch.setattr(conv.HostRamBudget, "charge", charge)
    with pytest.raises(conv.HostRamShortfallError):
        cq.load_comfy_quant_transformer(
            _H3Like,
            path,
            cq.refuse_comfy_quant(path),
            {"torch_dtype": torch.bfloat16, "config": "base/repo"},
            family = "minimax-h3",
            int8_backend = "native",
        )
    assert len(calls) == 1  # refused before the full-width conversion ran


def test_a_resident_video_dit_that_fits_only_stored_gets_fp8_storage(tmp_path, monkeypatch):
    """A resident ComfyUI fp8 video DiT whose widened form would leave under the host reserve stores fp8, as an
    image single file does: free RAM between the stored and the widened need loads, and is refused when
    UNSLOTH_DIFFUSION_FP8_STORAGE=0 makes the load widen."""
    import core.inference.diffusion_single_file_converters as conv
    import core.inference.diffusion_small_host as small_host

    torch.manual_seed(4)
    dense = _H3Like()
    tensors = {"head.weight": dense.head.weight.detach(), "head.bias": dense.head.bias.detach()}
    quantized = {
        "attn.to_q": dense.attn.to_q.weight,
        "attn.to_k": dense.attn.to_k.weight,
        "attn.to_v": dense.attn.to_v.weight,
        "ff.proj": dense.ff.proj.weight,
    }
    for name, w in quantized.items():
        w = w.detach().float()
        scale = w.abs().max() / torch.finfo(torch.float8_e4m3fn).max
        tensors[f"{name}.weight"] = (w / scale).to(torch.float8_e4m3fn)
        tensors[f"{name}.weight_scale"] = scale.reshape(1)
        tensors[f"{name}.comfy_quant"] = _conf(format = "float8_e4m3fn")
    path = _save(tmp_path / "wan_fp8.safetensors", tensors)
    monkeypatch.setattr(
        cq,
        "_mapping",
        lambda cls: (
            lambda checkpoint = None, **_kw: checkpoint,
            types.SimpleNamespace(
                _should_convert_state_dict_to_diffusers = lambda a, b: False,
                _get_mapping_function_kwargs = lambda fn, **kw: {},
            ),
        ),
    )
    monkeypatch.setattr(vid, "plan_keeps_transformer_resident", lambda p: p.resident)
    monkeypatch.setattr(vid, "comfy_int8_backend", lambda *a, **k: None)
    monkeypatch.setattr(vid, "comfy_fp8_backend", lambda *a, **k: None)
    monkeypatch.delenv(cq.FP8_STORAGE_ENV, raising = False)
    # Price in bytes, no reserves: the tiny model's needs are a few hundred KB apart.
    monkeypatch.setattr(conv, "HOST_RAM_CHECK_MIN_MIB", 0)
    monkeypatch.delenv(small_host.HOST_RAM_CHECK_ENV, raising = False)
    monkeypatch.setattr(conv, "_MIB", 1)
    monkeypatch.setattr(small_host, "host_reserve_mib", lambda total: 0)
    monkeypatch.setattr(cq, "FP8_STORAGE_HOST_RESERVE_MIB", 0)
    keys = [n + ".weight" for n in quantized]
    widened = conv.converted_host_bytes(
        path, torch.bfloat16, dequantized = dict.fromkeys(keys, 1).get
    )
    stored = conv.converted_host_bytes(
        path, torch.bfloat16, dequantized = dict.fromkeys(keys, 0).get
    ) + 4 * sum(int(tensors[k].shape[0]) for k in keys)
    assert 0 < stored < widened
    available = (stored + widened) // 2
    monkeypatch.setattr(small_host, "host_ram_mib", lambda: (1 << 40, available))
    fam = types.SimpleNamespace(name = "wan2.2-ti2v-5b")

    def backends(**kw):
        return vid._video_comfy_backends(fam, "b", None, _plan(True), _plan(True), **kw)

    def load(**kw):
        return cq.load_comfy_quant_transformer(
            _H3Like,
            path,
            cq.refuse_comfy_quant(path),
            {"torch_dtype": torch.bfloat16, "config": "base/repo"},
            family = fam.name,
            **kw,
        )

    picked = backends(widened_mib = widened)
    assert picked == {
        "int8_backend": None,
        "fp8_backend": None,
        "fp8_storage": True,
        "fp8_widen_chosen": False,
    }
    model = load(**picked)
    assert model._unsloth_comfy_quant["fp8_stored"] == 4
    assert model.attn.to_q.weight_q.dtype == torch.uint8
    # room to widen: the faster widened load; H3 keeps its torchao-or-dequantize rule
    monkeypatch.setattr(small_host, "host_ram_mib", lambda: (1 << 40, 4 * widened))
    assert backends(widened_mib = widened)["fp8_storage"] is False
    # that widen was the rule's choice, so its load refuses against the rule's reserve; no host price, no choice
    assert backends(widened_mib = widened)["fp8_widen_chosen"] is True
    assert backends()["fp8_widen_chosen"] is False
    monkeypatch.setattr(small_host, "host_ram_mib", lambda: (1 << 40, available))
    h3 = types.SimpleNamespace(name = "minimax-h3")
    assert (
        vid._video_comfy_backends(h3, "b", None, _plan(True), _plan(True), widened_mib = widened)[
            "fp8_storage"
        ]
        is False
    )
    # the kill switch widens, and the host RAM check refuses that load
    monkeypatch.setenv(cq.FP8_STORAGE_ENV, "0")
    widened_pick = backends(widened_mib = widened)
    assert widened_pick["fp8_storage"] is False and widened_pick["fp8_widen_chosen"] is False
    with pytest.raises(RuntimeError, match = "Not enough system RAM"):
        load(**widened_pick)
