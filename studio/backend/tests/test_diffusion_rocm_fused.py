# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for ``diffusion_rocm_fused.py``: the switches (auto = ROCm only), install / uninstall wiring, and on a GPU the
kernels against diffusers (RoPE bit-identical, AdaLN within one rounding of stock) plus a tiny FLUX.1 transformer."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_eager_patches as ep  # noqa: E402
from core.inference import diffusion_rocm_fused as rf  # noqa: E402

fmod = pytest.importorskip("diffusers.models.transformers.transformer_flux")
from diffusers.models import normalization as nm  # noqa: E402
from diffusers.models.embeddings import apply_rotary_emb as stock_rope  # noqa: E402

# Another test file in the same session may leave Studio's eager AdaLN patch live on the classes: start from stock.
ep.uninstall_patches()
_STOCK_FWD = {name: getattr(nm, name).forward for name in rf._ADALN_CLASSES}

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or rf._rope_kernel() is None or rf._adaln_kernel() is None,
    reason = "needs a CUDA / ROCm GPU and Triton",
)


class _PipeLike:
    def __init__(self, module = "diffusers.models.transformers.transformer_flux"):
        self.transformer = type("FluxTransformer2DModel", (), {"__module__": module})()


@pytest.fixture(autouse = True)
def _restore(monkeypatch):
    monkeypatch.delenv(rf.FUSED_ROPE_ENV, raising = False)
    monkeypatch.delenv(rf.FUSED_ADALN_ENV, raising = False)
    rf.uninstall()
    ep.uninstall_patches()
    yield
    rf.uninstall()
    assert fmod.apply_rotary_emb is stock_rope
    for name, fwd in _STOCK_FWD.items():
        assert getattr(nm, name).forward is fwd


@pytest.fixture
def fake_kernels(monkeypatch):
    monkeypatch.setattr(rf, "_rope_kernel", lambda: (lambda x, cos, sin: None))
    monkeypatch.setattr(rf, "_adaln_kernel", lambda: (lambda x, scale, shift, eps: None))


def test_auto_is_inert_off_rocm(fake_kernels, monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    assert rf.install_for_pipe(_PipeLike(), torch.bfloat16, "cuda") == {"rope": False, "adaln": 0}
    assert not rf.is_installed()


def test_auto_engages_rope_only_on_rocm(fake_kernels, monkeypatch):
    # auto = the bit-identical RoPE kernel on ROCm; fused AdaLN bought no speed on gfx1151 and is opt-in.
    monkeypatch.setattr(torch.version, "hip", "7.2.0", raising = False)
    got = rf.install_for_pipe(_PipeLike(), torch.bfloat16, "cuda")
    assert got == {"rope": True, "adaln": 0}
    assert fmod.apply_rotary_emb is rf._ROPE_FNS[rf._MODULES[0]]
    assert nm.AdaLayerNormZero.forward is _STOCK_FWD["AdaLayerNormZero"]


def test_adaln_opt_in_on_rocm(fake_kernels, monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.2.0", raising = False)
    monkeypatch.setenv(rf.FUSED_ADALN_ENV, "1")
    got = rf.install_for_pipe(_PipeLike(), torch.bfloat16, "cuda")
    assert got == {"rope": True, "adaln": 3}
    assert nm.AdaLayerNormZero.forward is rf._adaln_zero_forward


@pytest.mark.parametrize("value", ["0", "off", "false"])
def test_kill_switches_on_rocm(fake_kernels, monkeypatch, value):
    monkeypatch.setattr(torch.version, "hip", "7.2.0", raising = False)
    monkeypatch.setenv(rf.FUSED_ROPE_ENV, value)
    monkeypatch.setenv(rf.FUSED_ADALN_ENV, value)
    assert rf.install_for_pipe(_PipeLike(), torch.bfloat16, "cuda") == {"rope": False, "adaln": 0}


def test_force_on_engages_on_cuda(fake_kernels, monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setenv(rf.FUSED_ROPE_ENV, "1")
    monkeypatch.setenv(rf.FUSED_ADALN_ENV, "1")
    assert rf.install_for_pipe(_PipeLike(), torch.float16, "cuda") == {"rope": True, "adaln": 3}


def test_switches_are_independent(fake_kernels, monkeypatch):
    monkeypatch.setenv(rf.FUSED_ROPE_ENV, "1")
    monkeypatch.setenv(rf.FUSED_ADALN_ENV, "0")
    assert rf.install_for_pipe(_PipeLike(), torch.bfloat16, "cuda") == {"rope": True, "adaln": 0}


def test_fp32_cpu_and_other_families_stay_stock(fake_kernels, monkeypatch):
    monkeypatch.setenv(rf.FUSED_ROPE_ENV, "1")
    monkeypatch.setenv(rf.FUSED_ADALN_ENV, "1")
    assert rf.install_for_pipe(_PipeLike(), torch.float32, "cuda") == {"rope": False, "adaln": 0}
    assert rf.install_for_pipe(_PipeLike(), torch.bfloat16, "cpu") == {"rope": False, "adaln": 0}
    other = _PipeLike("diffusers.models.transformers.transformer_qwenimage")
    assert rf.install_for_pipe(other, torch.bfloat16, "cuda") == {"rope": False, "adaln": 0}


def test_no_triton_keeps_stock(monkeypatch):
    monkeypatch.setenv(rf.FUSED_ROPE_ENV, "1")
    monkeypatch.setenv(rf.FUSED_ADALN_ENV, "1")
    monkeypatch.setattr(rf, "_rope_kernel", lambda: None)
    monkeypatch.setattr(rf, "_adaln_kernel", lambda: None)
    assert rf.install_for_pipe(_PipeLike(), torch.bfloat16, "cuda") == {"rope": False, "adaln": 0}


def test_reinstall_is_clean(fake_kernels, monkeypatch):
    monkeypatch.setenv(rf.FUSED_ROPE_ENV, "1")
    monkeypatch.setenv(rf.FUSED_ADALN_ENV, "1")
    for _ in range(2):
        rf.install_for_pipe(_PipeLike(), torch.bfloat16, "cuda")
    # the second install must not have stashed the first install's fused forward as "previous"
    assert all(prev not in rf._FORWARDS.values() for prev in rf._ADALN_PREV.values())


def test_cpu_tensors_fall_through(fake_kernels, monkeypatch):
    monkeypatch.setenv(rf.FUSED_ROPE_ENV, "1")
    monkeypatch.setenv(rf.FUSED_ADALN_ENV, "1")
    rf.install_for_pipe(_PipeLike(), torch.bfloat16, "cuda")
    x = torch.randn(1, 6, 2, 8)
    pos = torch.randn(6, 4, dtype = torch.float64)
    freqs = (pos.cos().repeat_interleave(2, -1).float(), pos.sin().repeat_interleave(2, -1).float())
    assert torch.equal(
        fmod.apply_rotary_emb(x, freqs, sequence_dim = 1), stock_rope(x, freqs, sequence_dim = 1)
    )
    torch.manual_seed(0)
    norm = nm.AdaLayerNormZero(16)
    xs, emb = torch.randn(2, 5, 16), torch.randn(2, 16)
    got = norm(xs, emb = emb)
    want = _STOCK_FWD["AdaLayerNormZero"](norm, xs, emb = emb)
    assert all(torch.equal(a, b) for a, b in zip(got, want))


def test_load_and_teardown_wiring():
    src = (Path(__file__).resolve().parents[1] / "core" / "inference" / "diffusion.py").read_text(
        encoding = "utf-8"
    )
    tree = ast.parse(src)
    teardown = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_uninstall_fused_dit_patches"
    )
    assert "uninstall_rocm_fused()" in ast.unparse(teardown)
    # Both unload paths unwind the fused AdaLN again after the eager layer (deferred profile installs it on top).
    assert src.count("_uninstall_fused_dit_patches()") >= 4
    load = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "load_pipeline"
    )
    assert "install_rocm_fused(pipe, dtype, device, logger)" in ast.unparse(load)


def test_stock_lines_still_present_in_diffusers():
    import inspect
    for name in rf._ADALN_CLASSES:
        src = inspect.getsource(_STOCK_FWD[name])
        assert all(line in src for line in rf._STOCK_LINES[name]), name
    assert "apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)" in Path(
        fmod.__file__
    ).read_text("utf-8")


# ---------------------------------------------------------------------------------------------------------------- GPU
def _freqs(S, D, device):
    pos = torch.randn(S, D // 2, dtype = torch.float64, device = device) * 50
    return (pos.cos().repeat_interleave(2, -1).float(), pos.sin().repeat_interleave(2, -1).float())


@needs_gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(1, 4608, 24, 128), (2, 333, 3, 64)])
def test_rope_bit_identical(monkeypatch, dtype, shape):
    monkeypatch.setenv(rf.FUSED_ROPE_ENV, "1")
    dev = "cuda"
    assert rf.install_for_pipe(_PipeLike(), dtype, dev)["rope"]
    B, S, H, D = shape
    x = torch.randn(B, S, H, D, device = dev).to(dtype)
    freqs = _freqs(S, D, dev)
    before = rf.COUNTS["rope_fused"]
    with torch.no_grad():
        got = fmod.apply_rotary_emb(x, freqs, sequence_dim = 1)
        want = stock_rope(x, freqs, sequence_dim = 1)
    assert rf.COUNTS["rope_fused"] == before + 1
    assert torch.equal(got, want)


def _bf16_ulp(t):
    a = t.float().abs().clamp_min(2.0**-126)
    return torch.exp2(torch.floor(torch.log2(a)) - (7 if t.dtype is torch.bfloat16 else 10))


@needs_gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "cls_name,D,S",
    [
        ("AdaLayerNormZero", 3072, 4096),
        ("AdaLayerNormZeroSingle", 3072, 777),
        ("AdaLayerNormContinuous", 256, 300),
    ],
)
def test_adaln_matches_stock(monkeypatch, dtype, cls_name, D, S):
    monkeypatch.setenv(rf.FUSED_ADALN_ENV, "1")
    dev = "cuda"
    torch.manual_seed(0)
    cls = getattr(nm, cls_name)
    if cls_name == "AdaLayerNormContinuous":
        mod = cls(D, D, elementwise_affine = False, eps = 1e-6)
    else:
        mod = cls(D)
    mod = mod.to(dev, dtype).eval()
    assert rf.install_for_pipe(_PipeLike(), dtype, dev)["adaln"] == 3
    x = (torch.randn(2, S, D, device = dev) * 3 + 0.5).to(dtype)
    emb = torch.randn(2, D, device = dev).to(dtype)
    before = rf.COUNTS["adaln_fused"]
    with torch.no_grad():
        got = mod(x, emb) if cls_name == "AdaLayerNormContinuous" else mod(x, emb = emb)
        want = (
            _STOCK_FWD[cls_name](mod, x, emb)
            if cls_name == "AdaLayerNormContinuous"
            else _STOCK_FWD[cls_name](mod, x, emb = emb)
        )
    assert rf.COUNTS["adaln_fused"] == before + 1
    got0, want0 = (got, want) if torch.is_tensor(got) else (got[0], want[0])
    if not torch.is_tensor(got):  # gates / mlp modulation pass through untouched
        assert all(torch.equal(a, b) for a, b in zip(got[1:], want[1:]))
    diff = (got0.float() - want0.float()).abs()
    # The stock rounding chain is reproduced; only the row reduction order differs, which can flip the rounding of the
    # normalised value by one ULP. Bound: that flip carried through the product, plus one rounding of product and sum.
    with torch.no_grad():
        e = mod.linear(mod.silu(emb))
        scale = (
            e.chunk(2, dim = 1)[0]
            if cls_name == "AdaLayerNormContinuous"
            else e.chunk(6 if cls_name == "AdaLayerNormZero" else 3, dim = 1)[1]
        )
        n = mod.norm(x)
        m = (1 + scale)[:, None, :]
        p = n * m
    bound = 2 * (_bf16_ulp(n) * m.float().abs() + _bf16_ulp(p) + _bf16_ulp(want0))
    assert (diff <= bound).all(), float((diff / bound).max())
    assert (diff > 0).float().mean().item() < 0.01, (diff > 0).float().mean().item()


@needs_gpu
def test_tiny_flux1_transformer_end_to_end(monkeypatch):
    from diffusers import FluxTransformer2DModel

    monkeypatch.setenv(rf.FUSED_ROPE_ENV, "1")
    monkeypatch.setenv(rf.FUSED_ADALN_ENV, "1")
    dev, dtype = "cuda", torch.bfloat16
    torch.manual_seed(0)
    model = (
        FluxTransformer2DModel(
            patch_size = 1,
            in_channels = 16,
            num_layers = 2,
            num_single_layers = 2,
            attention_head_dim = 32,
            num_attention_heads = 4,
            joint_attention_dim = 64,
            pooled_projection_dim = 32,
            axes_dims_rope = (8, 12, 12),
        )
        .to(dev, dtype)
        .eval()
    )
    B, img, txt = 1, 256, 32
    inputs = dict(
        hidden_states = torch.randn(B, img, 16, device = dev, dtype = dtype),
        encoder_hidden_states = torch.randn(B, txt, 64, device = dev, dtype = dtype),
        pooled_projections = torch.randn(B, 32, device = dev, dtype = dtype),
        timestep = torch.tensor([0.5], device = dev, dtype = dtype),
        img_ids = torch.randint(0, 16, (img, 3), device = dev).float(),
        txt_ids = torch.zeros(txt, 3, device = dev),
        return_dict = False,
    )
    with torch.no_grad():
        want = model(**inputs)[0]
        rf.install_for_pipe(type("P", (), {"transformer": model})(), dtype, dev)
        r0, a0 = rf.COUNTS["rope_fused"], rf.COUNTS["adaln_fused"]
        got = model(**inputs)[0]
    # 2 double blocks (img + txt AdaLN, q + k RoPE) + 2 single blocks + norm_out
    assert rf.COUNTS["rope_fused"] - r0 == 2 * 2 + 2 * 2
    assert rf.COUNTS["adaln_fused"] - a0 == 2 * 2 + 2 + 1
    rf.uninstall()
    rel = (got.float() - want.float()).norm() / want.float().norm()
    assert rel < 1e-2, float(rel)


def test_teardown_survives_a_later_eager_layer(fake_kernels, monkeypatch):
    # Deferred speed profile: fused AdaLN at load, Studio's eager patch layered on top at the 3rd image, then unload
    # in the order diffusion.py uses (fused teardown first, eager after).
    monkeypatch.setenv(rf.FUSED_ADALN_ENV, "1")
    try:
        assert rf.install_adaln(torch.bfloat16, "cuda") == len(rf._ADALN_CLASSES)
        ep.install_compile_safe_patches()
        assert nm.AdaLayerNormZero.forward is not rf._FORWARDS["AdaLayerNormZero"]
        rf.uninstall()
        ep.uninstall_patches()
        rf.uninstall()
        assert {n: getattr(nm, n).forward for n in rf._ADALN_CLASSES} == _STOCK_FWD
        torch.manual_seed(0)
        norm = nm.AdaLayerNormZero(16)
        xs, emb = torch.randn(2, 5, 16), torch.randn(2, 16)
        got = norm(xs, emb = emb)
        want = _STOCK_FWD["AdaLayerNormZero"](norm, xs, emb = emb)
        assert all(torch.equal(a, b) for a, b in zip(got, want))
    finally:
        ep.uninstall_patches()
        rf.uninstall()
        for n, fwd in _STOCK_FWD.items():
            getattr(nm, n).forward = fwd


def test_stock_branch_never_loses_its_previous_forward(fake_kernels, monkeypatch):
    monkeypatch.setenv(rf.FUSED_ADALN_ENV, "1")
    rf.install_adaln(torch.bfloat16, "cuda")
    cls = nm.AdaLayerNormZero
    rf._ADALN_PREV.clear()
    try:
        assert rf._prev(cls) is _STOCK_FWD["AdaLayerNormZero"]
    finally:
        for n, fwd in _STOCK_FWD.items():
            getattr(nm, n).forward = fwd
