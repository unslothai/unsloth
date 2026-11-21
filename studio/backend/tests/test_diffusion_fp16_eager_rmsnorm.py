# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The fused RMSNorm on the fp16 eager (``off``) tier for Z-Image (diffusion_eager_patches.install_fp16_rmsnorm)."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from core.inference import diffusion_eager_patches as ep  # noqa: E402
from core.inference.diffusion_autoopt import apply as aa  # noqa: E402
from core.inference.diffusion_autoopt import patches as ap  # noqa: E402
from diffusers.models.normalization import AdaLayerNormZero, RMSNorm  # noqa: E402


class ZImageTransformer2DModel(torch.nn.Module):  # name-only stand-in for the class gate
    pass


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(ep.FP16_RMSNORM_ENV, raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_EAGER_PATCHES", raising = False)
    ep.uninstall_patches()
    yield
    ep.uninstall_patches()


def test_wanted_only_for_fp16_off_tier_zimage(monkeypatch):
    z = ZImageTransformer2DModel()
    assert ep.fp16_rmsnorm_wanted(z, torch.float16, "off")
    assert not ep.fp16_rmsnorm_wanted(z, torch.bfloat16, "off")
    assert not ep.fp16_rmsnorm_wanted(z, torch.float16, "default")
    assert not ep.fp16_rmsnorm_wanted(torch.nn.Linear(2, 2), torch.float16, "off")
    monkeypatch.setenv(ep.FP16_RMSNORM_ENV, "0")
    assert not ep.fp16_rmsnorm_wanted(z, torch.float16, "off")


def test_install_is_rmsnorm_only_and_reversible():
    orig_rms, orig_ada = RMSNorm.forward, AdaLayerNormZero.forward
    assert ep.install_fp16_rmsnorm() and ep.install_fp16_rmsnorm()
    assert RMSNorm.forward is not orig_rms and AdaLayerNormZero.forward is orig_ada
    assert ep.fp16_rmsnorm_installed()
    ep.uninstall_fp16_rmsnorm()
    assert RMSNorm.forward is orig_rms and not ep.fp16_rmsnorm_installed()


def test_full_set_replaces_the_rmsnorm_only_patch():
    orig_ada = AdaLayerNormZero.forward
    assert ep.install_fp16_rmsnorm()
    assert ep.install_compile_safe_patches() >= 2
    assert AdaLayerNormZero.forward is not orig_ada and not ep.fp16_rmsnorm_installed()
    ep.uninstall_fp16_rmsnorm()
    assert ep.is_installed() and AdaLayerNormZero.forward is not orig_ada


def _ctx(
    speed,
    dtype = torch.float16,
    denoiser = None,
):
    return dict(
        pipe = types.SimpleNamespace(transformer = denoiser or ZImageTransformer2DModel()),
        dtype = dtype,
        speed_mode = speed,
        backend = "image",
    )


def test_registry_installs_on_off_and_reverts_on_the_next_load():
    orig = RMSNorm.forward
    aa.install_patches(ap.STAGE_LOAD_EAGER, **(_ctx("off")))
    assert ep.fp16_rmsnorm_installed() and RMSNorm.forward is not orig
    aa.install_patches(ap.STAGE_LOAD_EAGER, **(_ctx("off", denoiser = torch.nn.Linear(2, 2))))
    assert not ep.fp16_rmsnorm_installed() and RMSNorm.forward is orig
    aa.install_patches(ap.STAGE_LOAD_EAGER, **(_ctx("off")))
    aa.install_patches(ap.STAGE_LOAD_EAGER, **(_ctx("default")))
    assert not ep.fp16_rmsnorm_installed() and ep.is_installed()
    ap.uninstall_group(ap.GROUP_EAGER)
    assert RMSNorm.forward is orig


def test_registry_kill_switch(monkeypatch):
    orig = RMSNorm.forward
    monkeypatch.setenv(ep.FP16_RMSNORM_ENV, "0")
    aa.install_patches(ap.STAGE_LOAD_EAGER, **(_ctx("off")))
    assert RMSNorm.forward is orig


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA for fp16 kernels")
def test_fused_fp16_is_at_least_as_close_to_fp64():
    torch.manual_seed(0)
    m = RMSNorm(3840, eps = 1e-6).to("cuda", torch.float16)
    with torch.no_grad():
        m.weight.copy_(torch.randn(3840) * 0.2 + 1)
    x = (torch.randn(4, 4096, 3840) * 3).to("cuda", torch.float16)
    ref = (
        x.double() * torch.rsqrt(x.double().pow(2).mean(-1, keepdim = True) + 1e-6)
    ) * m.weight.double()
    with torch.no_grad():
        stock = m(x)
        ep.install_fp16_rmsnorm()
        fused = m(x)
    err_stock = (stock.double() - ref).abs().mean().item()
    err_fused = (fused.double() - ref).abs().mean().item()
    assert err_fused <= err_stock * 1.001, (err_fused, err_stock)
    assert (fused.float() - stock.float()).abs().max().item() <= 2**-8 * ref.abs().max().item()
