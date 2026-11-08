# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 int8 / fp8 files where torchao does not run: routing per host, the 1-byte twins, hosted fp8."""

from __future__ import annotations

import json
import types

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")
diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "MiniMaxH3Transformer3DModel"):
    pytest.skip("diffusers has no MiniMaxH3Transformer3DModel", allow_module_level = True)

from core.inference import diffusion_comfy_quant as cq  # noqa: E402
from core.inference import diffusion_transformer_quant as dtq  # noqa: E402
from core.inference import video as vid  # noqa: E402
from core.inference import video_minimax_h3_native_quant as nq  # noqa: E402
from core.inference.diffusion_native_quant import native_linear_class  # noqa: E402
from core.inference.video_minimax_h3_comfy import _RENAMES, load_h3_comfy_transformer  # noqa: E402

# Wide enough that Studio's int8 filter (512 features) keeps the block Linears.
TINY = dict(
    num_attention_heads = 4,
    attention_head_dim = 128,
    hidden_size = 512,
    num_layers = 2,
    num_refiner_layers = 1,
    ffn_dim = 1024,
    in_channels = 24,
    audio_in_channels = 32,
    text_dim = 16,
    freq_dim = 32,
    time_embed_hidden_dim = 32,
    time_embed_dim = 16,
    rope_freq_dim = 2,
)
FP8_NAME = "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
INT8_NAME = "minimax_h3_fl2va_pruned_int8.safetensors"
H3 = types.SimpleNamespace(name = "minimax-h3")


@pytest.fixture(autouse = True)
def _switch_default(monkeypatch):
    monkeypatch.delenv(nq.H3_NATIVE_QUANT_ENV, raising = False)
    monkeypatch.delenv(cq.FP8_STORAGE_ENV, raising = False)
    monkeypatch.delenv(cq.COMFY_FP8_ENV, raising = False)
    monkeypatch.delenv(cq.COMFY_INT8_ENV, raising = False)


def _target(device = "cuda"):
    return types.SimpleNamespace(device = device, dtype = torch.bfloat16)


def _host(monkeypatch, kind):
    """Studio's host predicates as ``kind`` answers them."""
    rocm = kind == "rocm"
    monkeypatch.setattr(dtq, "torch_is_rocm", lambda: rocm)
    monkeypatch.setattr(dtq, "is_stubbed", lambda name: False)
    runs = {
        "sm89": {dtq.TQ_INT8, dtq.TQ_FP8},
        "sm80": {dtq.TQ_INT8},  # no fp8 GEMM below sm_89
        "rocm": set(),
        "mps": set(),
    }[kind]

    def select(
        target,
        requested,
        family = None,
        **_k,
    ):
        scheme = dtq.normalize_transformer_quant(requested)
        return scheme if scheme in runs else None

    monkeypatch.setattr(dtq, "select_transformer_quant_scheme", select)
    return _target("mps" if kind == "mps" else "cuda")


# ── routing per host ──


@pytest.mark.parametrize(
    "kind,int8,fp8,storage",
    [
        ("sm89", "torchao", "torchao", False),  # unchanged: torchao for both
        ("sm80", "torchao", None, True),  # A100 / RTX 3090: fp8 stored at 1 byte
        ("rocm", "native", "native", False),  # the torchao-free twin for both
        ("mps", "native", None, True),  # Metal int8; fp8 codes stored (no fp8 cast on MPS)
    ],
)
def test_h3_backends_per_host(monkeypatch, kind, int8, fp8, storage):
    target = _host(monkeypatch, kind)
    got = nq.h3_comfy_backends(target)
    assert got == {"int8_backend": int8, "fp8_backend": fp8, "fp8_storage": storage}
    assert vid._video_fp8_storage(H3, target, False) is storage
    assert vid._video_fp8_storage(H3, target, True) is storage
    assert vid._video_fp8_storage(H3, target, False, widen_chosen = True) is False


@pytest.mark.parametrize("kind", ["sm89", "sm80", "rocm", "mps"])
def test_kill_switch_restores_torchao_only(monkeypatch, kind):
    target = _host(monkeypatch, kind)
    monkeypatch.setenv(nq.H3_NATIVE_QUANT_ENV, "0")
    got = nq.h3_comfy_backends(target)
    want = "torchao" if kind in ("sm89", "sm80") else None
    assert got["int8_backend"] == want
    assert got["fp8_backend"] == ("torchao" if kind == "sm89" else None)
    assert got["fp8_storage"] is False
    assert vid._video_fp8_storage(H3, target, False) is False


def test_storage_off_switch_and_cpu_never_store(monkeypatch):
    target = _host(monkeypatch, "sm80")
    monkeypatch.setenv(cq.FP8_STORAGE_ENV, "0")
    assert nq.h3_comfy_backends(target)["fp8_storage"] is False
    monkeypatch.delenv(cq.FP8_STORAGE_ENV)
    assert nq.h3_comfy_backends(_target("cpu"))["fp8_storage"] is False
    assert nq.h3_fp8_storage(None) is False


@pytest.mark.parametrize("kind,native", [("sm89", False), ("sm80", True), ("rocm", False)])
def test_hosted_fp8_native_only_where_torchao_fp8_does_not_run_on_cuda(monkeypatch, kind, native):
    target = _host(monkeypatch, kind)
    # ROCm keeps torchao's own answer: this rule is for CUDA cards without an fp8 GEMM.
    if kind == "rocm":
        monkeypatch.setattr(dtq, "select_transformer_quant_scheme", lambda *a, **k: dtq.TQ_FP8)
    assert nq.h3_hosted_fp8_native(target) is native
    assert vid._h3_hosted_fp8_native(H3, target, "fp8") is native
    assert vid._h3_hosted_fp8_native(H3, target, "int8") is False
    assert vid._h3_hosted_fp8_native(types.SimpleNamespace(name = "ltx-2"), target, "fp8") is False
    monkeypatch.setenv(nq.H3_NATIVE_QUANT_ENV, "0")
    assert nq.h3_hosted_fp8_native(target) is False


# ── a tiny H3 single file through the real loader ──


def _tiny(seed = 0):
    torch.manual_seed(seed)
    model = diffusers.MiniMaxH3Transformer3DModel(**TINY)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.05)
    return model.eval()


def _comfy_key(name):
    if name.startswith("transformer_blocks."):
        name = "blocks." + name[len("transformer_blocks.") :]
    elif name.startswith("token_refiner.refiner_blocks."):
        name = "token_refiner.blocks." + name[len("token_refiner.refiner_blocks.") :]
    for old, new in reversed(_RENAMES):
        if new in name and (not new.startswith(("proj_", "audio_proj")) or name.startswith(new)):
            name = name.replace(new, old, 1)
            break
    return name


def _quant(w, fmt):
    w = w.float()
    if fmt == "int8_tensorwise":
        scale = w.abs().amax().clamp_min(1e-8) / 127.0
        return torch.round(w / scale).clamp(-127, 127).to(torch.int8), scale.reshape(())
    scale = w.abs().amax().clamp_min(1e-8) / 448.0
    return (w / scale).to(torch.float8_e4m3fn), scale.reshape(())


def _write(tmp_path, model, name, fmt):
    """An original-layout H3 file (fused qkv, [gate; value] fc1), block Linears as ``fmt`` codes."""
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    out = {}
    for key in list(sd):
        if ".attn.to_q.weight" in key:
            stem = key[: -len("to_q.weight")]
            fused = torch.cat([sd.pop(stem + p + ".weight") for p in ("to_q", "to_k", "to_v")])
            out[_comfy_key(stem) + "qkv_proj.weight"] = fused
    for key, value in sd.items():
        if key.endswith(".ff.net.0.proj.weight"):
            value_half, gate = value.chunk(2)
            out[_comfy_key(key.replace(".ff.net.0.proj.weight", ".mlp.fc1.weight"))] = torch.cat(
                [gate, value_half]
            )
            continue
        out[_comfy_key(key)] = value
    quantized = [
        k
        for k in out
        if k.startswith(("blocks.", "token_refiner.blocks."))
        and k.endswith(("qkv_proj.weight", "out_proj.weight", "fc1.weight", "fc2.weight"))
    ]
    for key in quantized:
        codes, scale = _quant(out[key], fmt)
        out[key] = codes
        out[key[: -len(".weight")] + ".weight_scale"] = scale
        out[key[: -len(".weight")] + ".comfy_quant"] = torch.tensor(
            list(json.dumps({"format": fmt}).encode()), dtype = torch.uint8
        )
    path = tmp_path / name
    safetensors_torch.save_file({k: v.contiguous() for k, v in out.items()}, str(path))
    cfg = tmp_path / "base" / "transformer"
    cfg.mkdir(parents = True, exist_ok = True)
    (cfg / "config.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3Transformer3DModel", **TINY})
    )
    return str(path), str(tmp_path / "base"), len(quantized)


def _load(
    path,
    base,
    target,
    dtype = torch.float32,
    **backends,
):
    return load_h3_comfy_transformer(
        diffusers.MiniMaxH3Transformer3DModel,
        path,
        cq.refuse_comfy_quant(path),
        config = base,
        subfolder = "transformer",
        dtype = dtype,
        hf_token = None,
        cache_dir = None,
        local_files_only = True,
        target = target,
        **backends,
    )


def _inputs(
    seed = 0,
    n_text = 4,
    n_video = 6,
    n_audio = 3,
):
    g = torch.Generator().manual_seed(seed)
    s = n_text + n_video + n_audio
    text = torch.arange(n_text)
    video = torch.arange(n_text, n_text + n_video)
    audio = torch.arange(n_text + n_video, s)
    tags = torch.zeros(s, dtype = torch.long)
    tags[text], tags[audio] = 1, 2
    row_t = torch.full((s,), 0.6)
    row_t[audio] = 0.4
    uniq, inv = torch.unique(row_t, sorted = True, return_inverse = True)
    pos = torch.stack(
        [torch.arange(s), torch.zeros(s, dtype = torch.long), torch.arange(s) % 3], 1
    ).float()
    return dict(
        hidden_states = torch.randn(1, n_video, 96, generator = g),
        audio_hidden_states = torch.randn(1, n_audio, 32, generator = g),
        encoder_hidden_states = torch.randn(1, n_text, 16, generator = g),
        timestep = uniq,
        timestep_indices = inv,
        token_tags = tags,
        position_ids = pos,
        video_indices = video,
        audio_indices = audio,
        text_indices = text,
        return_dict = False,
    )


def _forward(model):
    with torch.no_grad():
        out = model(**_inputs())
    return torch.cat([t.float().reshape(-1) for t in out if torch.is_tensor(t)])


def _twins(model):
    cls = native_linear_class()
    return [m for m in model.modules() if isinstance(m, cls)]


def test_fp8_file_on_ampere_loads_stored_fp8_and_matches_the_dequantized_model(
    tmp_path, monkeypatch
):
    path, base, n_quant = _write(tmp_path, _tiny(), FP8_NAME, "float8_e4m3fn")
    target = _host(monkeypatch, "sm80")
    backends = nq.h3_comfy_backends(target)
    model = _load(path, base, target, **backends)
    info = model._unsloth_comfy_quant
    # qkv splits into three layers
    assert info["fp8_stored"] == n_quant + 2 * TINY["num_layers"] + 2 * TINY["num_refiner_layers"]
    assert info["dequantized"] == 0
    twins = _twins(model)
    assert len(twins) == info["fp8_stored"]
    assert all(t.weight_q.element_size() == 1 and t.scheme == "fp8" for t in twins)
    reference = _load(path, base, target, int8_backend = None, fp8_backend = None, fp8_storage = False)
    assert not _twins(reference)
    torch.testing.assert_close(_forward(model), _forward(reference), rtol = 0, atol = 0)


def test_fp8_stored_bytes_are_half_the_widened_bf16_bytes(tmp_path, monkeypatch):
    path, base, _ = _write(tmp_path, _tiny(), FP8_NAME, "float8_e4m3fn")
    target = _host(monkeypatch, "sm80")
    stored = _load(path, base, target, dtype = torch.bfloat16, **nq.h3_comfy_backends(target))
    widened = _load(path, base, target, dtype = torch.bfloat16, int8_backend = None, fp8_backend = None)

    def block_bytes(model):
        total = 0
        for name, t in list(model.named_parameters()) + list(model.named_buffers()):
            if name.startswith("transformer_blocks."):
                total += t.numel() * t.element_size()
        return total

    assert block_bytes(stored) < 0.55 * block_bytes(widened)
    scan = cq.refuse_comfy_quant(path)
    priced = cq.comfy_resident_mib(path, scan, keep_int8 = False, keep_fp8 = False, fp8_stored = True)
    widened_mib = cq.comfy_resident_mib(
        path, scan, keep_int8 = False, keep_fp8 = False, fp8_stored = False
    )
    assert priced is not None and widened_mib is not None and priced <= widened_mib


def test_a_single_file_keeps_the_fp32_rope_the_hosted_checkpoint_keeps(tmp_path, monkeypatch):
    path, base, _ = _write(tmp_path, _tiny(), FP8_NAME, "float8_e4m3fn")
    target = _host(monkeypatch, "sm80")
    model = _load(path, base, target, dtype = torch.bfloat16, **nq.h3_comfy_backends(target))
    fresh = diffusers.MiniMaxH3Transformer3DModel(**TINY)
    assert model.rope.inv_freq.dtype == torch.float32
    assert torch.equal(model.rope.inv_freq, fresh.rope.inv_freq)


@pytest.mark.parametrize("kind", ["rocm", "mps"])
def test_int8_file_on_native_hosts_keeps_int8_codes(tmp_path, monkeypatch, kind):
    path, base, _ = _write(tmp_path, _tiny(), INT8_NAME, "int8_tensorwise")
    target = _host(monkeypatch, kind)
    backends = nq.h3_comfy_backends(target)
    assert backends["int8_backend"] == "native"
    model = _load(path, base, target, **backends)
    info = model._unsloth_comfy_quant
    assert info["int8"] > 0 and info["backend"] == "native"
    twins = _twins(model)
    assert twins and all(t.weight_q.dtype == torch.int8 for t in twins)
    reference = _load(path, base, target, int8_backend = None, fp8_backend = None)
    torch.testing.assert_close(_forward(model), _forward(reference), rtol = 1e-5, atol = 1e-5)


def test_fp8_file_on_rocm_keeps_fp8_codes(tmp_path, monkeypatch):
    path, base, _ = _write(tmp_path, _tiny(), FP8_NAME, "float8_e4m3fn")
    target = _host(monkeypatch, "rocm")
    # Studio's fp8 runtimes assert bf16 weights (an fp32 load dequantizes them)
    model = _load(path, base, target, dtype = torch.bfloat16, **nq.h3_comfy_backends(target))
    info = model._unsloth_comfy_quant
    assert info["fp8"] > 0 and info["fp8_backend"] == "native" and info["dequantized"] == 0
    reference = _load(path, base, target, dtype = torch.bfloat16, int8_backend = None, fp8_backend = None)
    torch.testing.assert_close(_forward(model), _forward(reference), rtol = 0, atol = 0)


# ── the hosted torchao fp8 checkpoint ──


def test_hosted_float8tensor_layers_become_stored_fp8_with_identical_output():
    quant = pytest.importorskip("torchao.quantization")
    model = _tiny().to(torch.bfloat16)
    for name, module in model.named_modules():
        if (
            isinstance(module, torch.nn.Linear)
            and name.startswith("transformer_blocks.")
            and module.in_features >= 512
        ):
            weight = quant.Float8Tensor.from_hp(
                module.weight.detach(), float8_dtype = torch.float8_e4m3fn, granularity = quant.PerRow()
            )
            module.weight = torch.nn.Parameter(weight, requires_grad = False)
    fp8 = {
        n: m.weight
        for n, m in model.named_modules()
        if type(getattr(m, "weight", None)).__name__ == "Float8Tensor"
    }
    if not fp8:
        pytest.skip("torchao produced no Float8Tensor weights")
    reference = {n: w.dequantize().to(torch.bfloat16) for n, w in fp8.items()}
    built = nq.h3_hosted_fp8_to_native(model)
    assert built == len(fp8)
    modules = dict(model.named_modules())
    for name, want in reference.items():
        layer = modules[name]
        assert layer.weight_q.element_size() == 1
        torch.testing.assert_close(layer.weight, want, rtol = 0, atol = 0)
    assert getattr(model, cq.FP8_STORED_ATTR) == built
    _forward(model)  # runs with no torchao fp8 GEMM


def test_hosted_conversion_refuses_a_layout_it_cannot_hold():
    class Float8Tensor:  # a block-scaled stand-in: neither per-row nor per-tensor
        qdata = torch.zeros(8, 16).to(torch.float8_e4m3fn)
        scale = torch.ones(8, 2)
        dtype = torch.bfloat16

    lin = torch.nn.Linear(16, 8)
    lin._parameters["weight"] = Float8Tensor()
    with pytest.raises(ValueError, match = "cannot hold"):
        nq.h3_hosted_fp8_to_native(torch.nn.Sequential(lin))


# ── on this host's real GPU (CUDA / ROCm / MPS): the backends it picks, run there ──


def _gpu_target():
    from core.inference.diffusion_device import resolve_diffusion_device_target

    target = resolve_diffusion_device_target()
    if getattr(target, "device", None) not in ("cuda", "mps"):
        pytest.skip("no GPU here")
    return target


@pytest.mark.parametrize("fmt,name", [("float8_e4m3fn", FP8_NAME), ("int8_tensorwise", INT8_NAME)])
def test_the_host_backends_run_on_its_gpu_like_the_dequantized_model(tmp_path, fmt, name):
    target = _gpu_target()
    device = target.torch_device
    path, base, _ = _write(tmp_path, _tiny(), name, fmt)
    backends = nq.h3_comfy_backends(target)
    scheme_runs = (
        backends["int8_backend"]
        if fmt.startswith("int8")
        else (backends["fp8_backend"] or backends["fp8_storage"])
    )
    if not scheme_runs:
        pytest.skip(f"{fmt} has no runtime on this host: {backends}")
    model = _load(path, base, target, dtype = torch.bfloat16, **backends).to(device)
    info = model._unsloth_comfy_quant
    kept = info["int8"] if fmt.startswith("int8") else info["fp8"] + info.get("fp8_stored", 0)
    assert kept > 0 and info["dequantized"] == 0, info
    reference = _load(
        path, base, target, dtype = torch.bfloat16, int8_backend = None, fp8_backend = None
    ).to(device)
    kw = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in _inputs().items()}
    with torch.no_grad():
        got = torch.cat([t.float().reshape(-1) for t in model(**kw) if torch.is_tensor(t)])
        want = torch.cat([t.float().reshape(-1) for t in reference(**kw) if torch.is_tensor(t)])
    assert torch.isfinite(got).all()
    # W8A8 int8 adds activation rounding
    rel = (got - want).norm() / want.norm().clamp_min(1e-12)
    assert rel < (5e-2 if fmt.startswith("int8") else 1e-2), (backends, float(rel))
