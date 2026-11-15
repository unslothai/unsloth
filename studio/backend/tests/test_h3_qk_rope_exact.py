# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The fused q/k RMSNorm + RoPE op is bit-identical to the eager chain: form selection, the per-device check and its
fallbacks on CPU; the kernel forms against eager on CUDA (skipped without one)."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from core.inference import video_minimax_h3_qknorm as Q  # noqa: E402

cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason = "the Triton kernel runs on CUDA only"
)


@pytest.fixture(autouse = True)
def _fresh(monkeypatch):
    monkeypatch.setattr(Q, "_VERDICTS", {})
    monkeypatch.setattr(Q, "STATS", {"fused": 0, "rope": 0, "stock": 0})
    monkeypatch.delenv(Q.QK_ROPE_MODE_ENV, raising = False)


def _inputs(
    device = "cpu",
    dtype = torch.bfloat16,
    seq = 37,
    heads = 3,
    dim = 128,
    rot = 96,
    strided = False,
    seed = 3,
):
    g = torch.Generator(device = "cpu").manual_seed(seed)
    if strided:
        qkv = torch.randn(1, seq, 3 * heads * dim, generator = g).to(dtype).to(device)
        x = qkv[..., : heads * dim].unflatten(-1, (heads, dim))
    else:
        x = torch.randn(1, seq, heads, dim, generator = g).to(dtype).to(device)
    w = (1 + 0.2 * torch.randn(dim, generator = g)).to(dtype).to(device)
    ang = torch.rand(seq, rot // 2, generator = g) * 30
    ang = torch.cat((ang, ang), dim = -1).to(device)
    return x, w, ang.cos(), ang.sin()


def test_off_cuda_is_the_stock_math():
    x, w, cos, sin = _inputs()
    assert torch.equal(
        Q.qk_norm_rope_impl(x, w, cos, sin, 1e-5), Q.reference_qk_norm_rope(x, w, cos, sin, 1e-5)
    )
    assert Q.STATS == {"fused": 0, "rope": 0, "stock": 1}


@pytest.mark.parametrize("value, want", [(None, ("fused", "rope")), ("auto", ("fused", "rope")), ("rope", ("rope",)),
                                         ("FUSED", ("fused",)), ("junk", ("fused", "rope"))])  # fmt: skip
def test_mode_switch(monkeypatch, value, want):
    if value is not None:
        monkeypatch.setenv(Q.QK_ROPE_MODE_ENV, value)
    assert Q.allowed_modes() == want


def test_fused_form_only_where_torch_takes_its_vectorized_row_kernel():
    x, w, _cos, _sin = _inputs()
    assert Q._fused_row_path(x, w)
    assert not Q._fused_row_path(x, w.float())  # weight dtype differs: torch's composite norm
    buf = torch.empty(w.numel() + 1, dtype = w.dtype)
    assert not Q._fused_row_path(x, buf[1:])  # weight 2 bytes off the 8-byte vector alignment
    flat = torch.empty(x.numel() + 1, dtype = x.dtype)
    assert not Q._fused_row_path(
        flat[1:].view(x.shape), w
    )  # contiguous, misaligned: torch reads it in place
    xs, _, _, _ = _inputs(strided = True)
    assert not xs.is_contiguous() and Q._fused_row_path(
        xs, w
    )  # torch copies a strided input to a fresh buffer
    odd = torch.empty(1, 37, 3 * 3 * 128 + 8, dtype = x.dtype)[..., : 3 * 128].unflatten(-1, (3, 128))
    assert not Q._fused_row_path(
        odd, w
    )  # token stride 1160: a specialization the check never compiled


@pytest.mark.parametrize(
    "passed, env, aligned, want",
    [
        ({"fused": True, "rope": True, "order": ["fused", "rope"]}, None, True, "fused"),
        (
            {"fused": True, "rope": True, "order": ["rope", "fused"]},
            None,
            True,
            "rope",
        ),  # the race decides
        ({"fused": True, "rope": True, "order": ["fused", "rope"]}, "rope", True, "rope"),
        ({"fused": True, "rope": True, "order": ["fused", "rope"]}, None, False, "rope"),
        (
            {"fused": True, "rope": True, "order": []},
            None,
            True,
            None,
        ),  # neither beat the stock chain
        ({"fused": True, "rope": True, "order": []}, "fused", True, "fused"),  # forced, still exact
        ({"fused": True, "rope": False, "order": ["fused"]}, None, False, None),
        ({"fused": False, "rope": True, "order": ["rope"]}, "fused", True, None),
        ({"fused": False, "rope": False, "order": []}, None, True, None),
    ],
)
def test_form_selection(monkeypatch, passed, env, aligned, want):
    if env:
        monkeypatch.setenv(Q.QK_ROPE_MODE_ENV, env)
    monkeypatch.setattr(Q, "_triton_ok", lambda x: True)
    monkeypatch.setattr(Q, "verdict", lambda x, rot: passed)
    monkeypatch.setattr(Q, "_fused_row_path", lambda x, w: aligned)
    x, w, cos, _sin = _inputs()
    assert Q._mode(x, w, cos) == want


def test_check_runs_once_per_key_and_never_during_capture(monkeypatch):
    calls = []
    monkeypatch.setattr(
        Q, "_check", lambda dev, dt, d, r: calls.append((dt, d, r)) or {"fused": True, "rope": True}
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device", lambda d: __import__("contextlib").nullcontext())
    capturing = {"on": True}
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing["on"])
    x, _w, _c, _s = _inputs()
    assert Q.verdict(x, 96) is None and calls == []
    capturing["on"] = False
    assert Q.verdict(x, 96) == {"fused": True, "rope": True}
    capturing["on"] = True
    assert Q.verdict(x, 96) == {"fused": True, "rope": True}  # cached: no new check while capturing
    assert Q.verdict(x, 32) is None
    capturing["on"] = False
    Q.verdict(x, 32)
    assert calls == [(torch.bfloat16, 128, 96), (torch.bfloat16, 128, 32)]


def test_a_failed_launch_falls_back_to_the_stock_math(monkeypatch):
    monkeypatch.setattr(Q, "_mode", lambda x, w, c: "fused")

    def boom(*a, **k):
        raise RuntimeError("no kernel")

    monkeypatch.setattr(Q, "_launch", boom)
    x, w, cos, sin = _inputs()
    assert torch.equal(
        Q.qk_norm_rope_impl(x, w, cos, sin, 1e-5), Q.reference_qk_norm_rope(x, w, cos, sin, 1e-5)
    )
    assert Q.STATS["stock"] == 1


def test_a_failed_check_launch_fails_only_that_form(monkeypatch):
    real = Q._launch

    def launch(
        x,
        w,
        cos,
        sin,
        eps,
        *a,
        norm = True,
        **k,
    ):
        if norm:
            raise RuntimeError("fused JIT failed")
        return Q._reference_rope(x, cos, sin)  # stands in for the rope kernel on CPU

    monkeypatch.setattr(Q, "_launch", launch)
    monkeypatch.setattr(Q, "torch_fused_rms_norm_on_cuda", lambda: True)
    monkeypatch.setattr(Q, "_race_ms", lambda form, *a, **k: {"stock": 1.0, "rope": 0.5}[form])
    got = Q._check("cpu", torch.bfloat16, 128, 96)
    assert got == {
        "fused": False,
        "rope": True,
        "order": ["rope"],
        "ms": {"stock": 1.0, "rope": 0.5},
    }
    monkeypatch.setattr(Q, "_race_ms", lambda form, *a, **k: {"stock": 1.0, "rope": 0.99}[form])
    assert Q._check("cpu", torch.bfloat16, 128, 96)["order"] == []  # a tie keeps the stock chain
    monkeypatch.setattr(Q, "_launch", real)


# ------------------------------------------------------------------------------------------------------- CUDA


def _forms(x, w, cos, sin, eps):
    import torch.nn.functional as F

    out = {"rope": Q._launch(F.rms_norm(x, (x.shape[-1],), w, eps), w, cos, sin, eps, norm = False)}
    if Q.torch_fused_rms_norm_on_cuda():
        out["fused"] = Q._launch(x, w, cos, sin, eps)
    return out


@cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dim, rot", [(128, 96), (64, 32), (256, 192), (128, 128)])
@pytest.mark.parametrize("strided", [False, True])
def test_both_forms_are_bit_identical_to_eager_over_seeds(dtype, dim, rot, strided):
    for seed in range(8):
        x, w, cos, sin = _inputs(
            "cuda", dtype, seq = 331, heads = 7, dim = dim, rot = rot, strided = strided, seed = seed
        )
        x = x * (10.0 ** (seed - 4))  # row magnitudes 1e-4 .. 1e3
        eps = 1e-6 if seed % 2 else 1e-5
        ref = Q.reference_qk_norm_rope(x, w, cos, sin, eps)
        for form, got in _forms(x, w, cos, sin, eps).items():
            assert torch.equal(got, ref), (form, seed, int((got != ref).sum()))


@cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_edge_rows(dtype):
    x, w, cos, sin = _inputs("cuda", dtype, seq = 64, heads = 4)
    x[0, 0] = 0  # all-zero row: rstd = rsqrt(eps)
    x[0, 1, 0] = torch.finfo(dtype).max / 64  # one huge element
    x[0, 2, :, ::2] = 0
    x[0, 3] = x[0, 3].sign() * torch.finfo(dtype).tiny * 4  # tiny but normal
    ref = Q.reference_qk_norm_rope(x, w, cos, sin, 1e-5)
    for form, got in _forms(x, w, cos, sin, 1e-5).items():
        assert torch.equal(got, ref), form


@cuda
def test_the_device_check_picks_the_fused_form_and_a_perturbed_kernel_fails_it():
    x, w, cos, sin = _inputs("cuda", torch.bfloat16, strided = True)
    passed = Q.verdict(x, 96)
    assert passed["rope"]
    assert passed["fused"] is Q.torch_fused_rms_norm_on_cuda()
    assert passed["order"], passed["ms"]  # some exact form beats the stock chain on this card
    got = Q.qk_norm_rope_impl(x, w, cos, sin, 1e-5)
    assert torch.equal(got, Q.reference_qk_norm_rope(x, w, cos, sin, 1e-5))
    assert Q.STATS[passed["order"][0]] == 1
    bad = (got.float() * (1 + 2**-7)).to(got.dtype)
    assert int((bad != got).sum()) > got.numel() // 2


@cuda
def test_strided_processor_on_cuda_is_bit_identical_and_engages():
    pytest.importorskip("diffusers")
    pytest.importorskip("accelerate")
    import _minimax_h3_image_tiny as tiny
    import test_diffusion_minimax_h3_image_speed as T

    from core.inference import diffusion_minimax_h3_image as h3i
    from core.inference import video_minimax_h3_attn as A

    native = dict(tiny.TINY_TRANSFORMER_NATIVE, num_attention_heads = 1, attention_head_dim = 64)
    torch.manual_seed(0)
    model = h3i.transformer_class()(**h3i.transformer_config_from_native(native)).eval()
    tiny.randomize(model, 0)
    model = model.to(torch.bfloat16).cuda()
    model.set_attention_backend("_native_math")
    stock = copy.deepcopy(model)
    assert A.install_strided_attention(model, qk_rope = True) == 4
    inputs = {
        k: (v.cuda() if torch.is_tensor(v) else v)
        for k, v in T._inputs(model, torch.bfloat16).items()
    }
    with torch.no_grad():
        want = stock(**inputs)[0]
        got = model(**inputs)[0]
    assert torch.equal(got, want)
    # q and k of the 2 DiT blocks (the refiner attention has no rotary embedding)
    assert Q.STATS["fused"] + Q.STATS["rope"] == 4 and Q.STATS["stock"] == 0


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_probe_inputs_stay_finite_in_the_target_dtype(dtype_name):
    """An fp16 probe past 65504 turns into inf, RMSNorm makes NaN on both sides, and torch.equal then rejects an
    exact form; the cached verdict would switch the op off for every later finite input."""
    dtype = getattr(torch, dtype_name)
    for seed in (0, 1):
        for x, w, cos, sin, _ in Q._probe_inputs(torch.device("cpu"), dtype, 128, 96, seed):
            assert torch.isfinite(x).all() and torch.isfinite(w).all()
            ref = Q.reference_qk_norm_rope(x.float(), w.float(), cos, sin, 1e-6)
            assert torch.isfinite(ref).all()
