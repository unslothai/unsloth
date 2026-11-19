# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""16-bit VAE decode safety net: a finite decode is the stock one exactly, an injected NaN decode re-decodes once in
fp32 from the same latents and stays fp32, non-finite latents name the denoiser, GPUs without tensor cores decode
fp32 from the start, and the weights stay stored in 16 bits throughout."""

from __future__ import annotations

import copy
import logging
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from core.inference import decode_layers  # noqa: E402
from core.inference import diffusion_vae_finite as vfin  # noqa: E402

CUDA = types.SimpleNamespace(device = "cuda", backend = "cuda", ordinal = 0)


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(vfin.VAE_FP32_ENV, raising = False)
    monkeypatch.setattr(vfin, "target_lacks_tensor_cores", lambda target: False)
    for k in vfin.STATS:
        monkeypatch.setitem(vfin.STATS, k, 0)


def _tiny_vae(cls = "AutoencoderKL"):
    torch.manual_seed(0)
    if cls == "AutoencoderKLFlux2":
        vae = diffusers.AutoencoderKLFlux2(
            down_block_types = ("DownEncoderBlock2D",) * 2,
            up_block_types = ("UpDecoderBlock2D",) * 2,
            block_out_channels = (16, 32),
            layers_per_block = 1,
            latent_channels = 4,
            norm_num_groups = 8,
        )
    else:
        vae = diffusers.AutoencoderKL(
            in_channels = 3,
            out_channels = 3,
            down_block_types = ("DownEncoderBlock2D",) * 2,
            up_block_types = ("UpDecoderBlock2D",) * 2,
            block_out_channels = (16, 32),
            layers_per_block = 1,
            latent_channels = 4,
            norm_num_groups = 8,
        )
    return vae.eval()


def _pipe(vae):
    return types.SimpleNamespace(vae = vae)


def _poison_slot(vae, when = torch.float16):
    """Fault injection below the check: the 16-bit decode comes back all NaN, fp32 is untouched."""
    slot = vae.decode._unsloth_decode_slot
    inner, calls = slot.decode, []

    def poisoned(z, *args, **kwargs):
        calls.append(z.dtype)
        out = inner(z, *args, **kwargs)
        if z.dtype is when:
            out[0].fill_(float("nan"))
        return out

    slot.decode = poisoned
    return calls


def _fp32_reference(cls, z):
    """The same seeded weights, rounded to fp16 and held in fp32 (a copy of the wrapped VAE would share its closures)."""
    ref = _tiny_vae(cls).to(torch.float16).to(torch.float32)
    with torch.no_grad():
        return ref.decode(z.float(), return_dict = False)[0]


@pytest.mark.parametrize(
    "name, cap, want",
    [
        ("NVIDIA GeForce GTX 1650", (7, 5), True),
        ("NVIDIA GeForce GTX 1650 Ti with Max-Q Design", (7, 5), True),
        ("NVIDIA GeForce GTX 1660 SUPER", (7, 5), True),
        ("NVIDIA GeForce GTX 1660 Ti", (7, 5), True),
        ("NVIDIA GeForce GTX 1630", (7, 5), True),
        ("NVIDIA GeForce MX450", (7, 5), True),
        ("NVIDIA GeForce MX550", (7, 5), True),
        ("Quadro T1000", (7, 5), True),
        ("Quadro T2000 with Max-Q Design", (7, 5), True),
        ("NVIDIA T400 4GB", (7, 5), True),
        ("NVIDIA T600 Laptop GPU", (7, 5), True),
        ("NVIDIA T1200 Laptop GPU", (7, 5), True),
        ("NVIDIA T550 Laptop GPU", (7, 5), True),
        ("Tesla T4", (7, 5), False),
        ("NVIDIA GeForce RTX 2060", (7, 5), False),
        ("Quadro RTX 4000", (7, 5), False),
        ("NVIDIA TITAN RTX", (7, 5), False),
        ("Tesla V100-SXM2-16GB", (7, 0), False),
        ("NVIDIA GeForce GTX 1080 Ti", (6, 1), True),
        ("NVIDIA GeForce RTX 3060", (8, 6), False),
        ("NVIDIA RTX A1000 Laptop GPU", (8, 6), False),
        ("NVIDIA GeForce GTX 1650", None, False),
    ],
)
def test_tensor_core_detection(name, cap, want):
    assert vfin.lacks_tensor_cores(name, cap) is want


def test_target_detection_reads_the_selected_card(monkeypatch):
    monkeypatch.undo()  # the real target_lacks_tensor_cores
    seen = []
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda i: seen.append(i) or "NVIDIA GeForce GTX 1650"
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: (7, 5))
    assert vfin.target_lacks_tensor_cores(
        types.SimpleNamespace(device = "cuda", backend = "cuda", ordinal = 1)
    )
    assert seen == [1]
    assert not vfin.target_lacks_tensor_cores(
        types.SimpleNamespace(device = "cuda", backend = "rocm", ordinal = 0)
    )
    assert not vfin.target_lacks_tensor_cores(
        types.SimpleNamespace(device = "xpu", backend = "xpu", ordinal = 0)
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "Tesla T4")
    assert not vfin.target_lacks_tensor_cores(CUDA)


@pytest.mark.parametrize(
    "raw, mode",
    [
        (None, "auto"),
        ("", "auto"),
        ("auto", "auto"),
        ("1", "always"),
        ("fp32", "always"),
        ("0", "off"),
        ("off", "off"),
    ],
)
def test_mode(monkeypatch, raw, mode):
    if raw is not None:
        monkeypatch.setenv(vfin.VAE_FP32_ENV, raw)
    assert vfin.vae_fp32_mode() == mode


@pytest.mark.parametrize("cls", ["AutoencoderKL", "AutoencoderKLFlux2"])
def test_finite_decode_is_stock_exactly(cls):
    vae = _tiny_vae(cls).to(torch.float16)
    stock = copy.deepcopy(vae)
    assert vfin.install_finite_decode(_pipe(vae), CUDA) == "retry"
    z = torch.randn(1, 4, 8, 8).to(torch.float16)
    with torch.no_grad():
        got = vae.decode(z, return_dict = False)[0]
        want = stock.decode(z, return_dict = False)[0]
    assert got.dtype is torch.float16 and torch.equal(got, want)
    assert vfin.STATS == {"retries": 0, "fp32_decodes": 0, "nonfinite_latents": 0}


@pytest.mark.parametrize("cls", ["AutoencoderKL", "AutoencoderKLFlux2"])
def test_injected_nan_redecodes_fp32_from_the_same_latents_and_stays(cls, caplog):
    vae = _tiny_vae(cls).to(torch.float16)
    vfin.install_finite_decode(_pipe(vae), CUDA, logger = logging.getLogger("t"))
    calls = _poison_slot(vae)
    z = torch.randn(1, 4, 8, 8).to(torch.float16)
    want = _fp32_reference(cls, z)
    with caplog.at_level(logging.WARNING), torch.no_grad():
        got = vae.decode(z, return_dict = False)[0]
    assert calls == [torch.float16, torch.float32]
    assert got.dtype is torch.float32 and bool(torch.isfinite(got).all())
    torch.testing.assert_close(got, want, rtol = 0, atol = 0)
    assert vfin.STATS["retries"] == 1 and "not finite" in caplog.text
    # Stored weights never left 16 bits.
    assert {p.dtype for p in vae.parameters()} == {torch.float16}
    with torch.no_grad():
        vae.decode(z, return_dict = False)
    assert calls[2:] == [torch.float32] and vfin.STATS["retries"] == 1


def test_nonfinite_latents_name_the_denoiser():
    vae = _tiny_vae().to(torch.float16)
    vfin.install_finite_decode(_pipe(vae), CUDA)
    calls = _poison_slot(vae)
    z = torch.randn(1, 4, 8, 8).to(torch.float16)
    z[0, 0, 0, 0] = float("inf")
    with (
        pytest.raises(RuntimeError, match = "denoiser produced NaN or infinite latents"),
        torch.no_grad(),
    ):
        vae.decode(z, return_dict = False)
    assert calls == [torch.float16] and vfin.STATS["nonfinite_latents"] == 1


@pytest.mark.parametrize("how", ["env", "card"])
def test_fp32_from_the_start(monkeypatch, how):
    if how == "env":
        monkeypatch.setenv(vfin.VAE_FP32_ENV, "1")
    else:
        monkeypatch.setattr(vfin, "target_lacks_tensor_cores", lambda target: True)
    vae = _tiny_vae().to(torch.float16)
    ref = copy.deepcopy(vae).to(torch.float32)
    assert vfin.prefers_fp32_decode(CUDA)
    assert vfin.install_finite_decode(_pipe(vae), CUDA) == "fp32"
    calls = _poison_slot(vae)
    z = torch.randn(1, 4, 8, 8).to(torch.float16)
    x = torch.randn(1, 3, 16, 16).to(torch.float16)
    with torch.no_grad():
        got = vae.decode(z, return_dict = False)[0]
        dist = vae.encode(x).latent_dist
        tup = vae.encode(x, return_dict = False)[0]
        want = ref.decode(z.float(), return_dict = False)[0]
        want_enc = ref.encode(x.float()).latent_dist.mean
    assert calls == [torch.float32] and vfin.STATS["retries"] == 0
    torch.testing.assert_close(got, want, rtol = 0, atol = 0)
    # Encoded in fp32, handed back in the VAE's 16-bit dtype (img2img feeds the latents to a 16-bit denoiser).
    torch.testing.assert_close(dist.mean, want_enc.to(torch.float16), rtol = 0, atol = 0)
    for d in (dist, tup):
        assert {d.parameters.dtype, d.mean.dtype, d.std.dtype, d.var.dtype} == {torch.float16}
        assert d.sample(generator = torch.Generator().manual_seed(0)).dtype == torch.float16
    assert {p.dtype for p in vae.parameters()} == {torch.float16}


def test_off_leaves_the_vae_alone(monkeypatch):
    monkeypatch.setenv(vfin.VAE_FP32_ENV, "0")
    vae = _tiny_vae().to(torch.float16)
    assert vfin.install_finite_decode(_pipe(vae), CUDA) is None
    assert "decode" not in vae.__dict__ and not vfin.prefers_fp32_decode(CUDA)


def test_fp32_vae_passes_through_unchecked():
    vae = _tiny_vae()
    vfin.install_finite_decode(_pipe(vae), CUDA)
    calls = _poison_slot(vae, when = torch.float32)
    z = torch.randn(1, 4, 8, 8)
    with torch.no_grad():
        got = vae.decode(z, return_dict = False)[0]
    # Nothing to fall back to: the stock result (here poisoned) is returned, the postprocess check reports it.
    assert calls == [torch.float32] and bool(torch.isnan(got).all()) and vfin.STATS["retries"] == 0


def test_bf16_vae_is_guarded_too():
    vae = _tiny_vae().to(torch.bfloat16)
    vfin.install_finite_decode(_pipe(vae), CUDA)
    calls = _poison_slot(vae, when = torch.bfloat16)
    z = torch.randn(1, 4, 8, 8).to(torch.bfloat16)
    with torch.no_grad():
        got = vae.decode(z, return_dict = False)[0]
    assert calls == [torch.bfloat16, torch.float32] and bool(torch.isfinite(got).all())


def test_idempotent_and_yields_to_owning_fp16_layers():
    vae = _tiny_vae().to(torch.float16)
    assert vfin.install_finite_decode(_pipe(vae), CUDA) == "retry"
    first = vae.decode
    assert vfin.install_finite_decode(_pipe(vae), CUDA) == "retry" and vae.decode is first
    for mark in ("_unsloth_fp16_decode", "_unsloth_half_decode"):
        other = _tiny_vae().to(torch.float16)
        setattr(other, mark, True)
        assert (
            vfin.install_finite_decode(_pipe(other), CUDA) is None
            and "decode" not in other.__dict__
        )


def test_decode_compile_finds_the_slot():
    """diffusion_speed._compile_vae_decode compiles what sits in the slot, so the check stays eager outside it."""
    vae = _tiny_vae().to(torch.float16)
    vfin.install_finite_decode(_pipe(vae), CUDA)
    outer = vae.__dict__["decode"]
    assert decode_layers.next_layer(outer) is outer._unsloth_decode_slot.decode


def test_load_installs_before_the_speed_layer():
    src = (Path(__file__).resolve().parents[1] / "core" / "inference" / "diffusion.py").read_text()
    install_at = src.index("install_finite_decode(pipe, target")
    assert (
        src.index("enable_fp16_vae_decode(pipe, target")
        < install_at
        < src.index("autoopt.run_speed_layer(")
    )
    assert "and not prefers_fp32_decode(target)" in src
