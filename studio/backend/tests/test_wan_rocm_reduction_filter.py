# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Wan2.2-TI2V-5B pins inductor's reduction-config filter on gfx1151 (Strix Halo), so one seed renders one clip in
every fresh Studio process there.

Two reduction kernels of its compiled block get an R0_BLOCK 2048 and a 4096 config that sum in different orders, and
a cold-cache process benchmarked them on near-equal timings: fresh gfx1151 processes rendered one seed into up to four
clips (the frames matched exactly the two kernels' picks). With the filter every process matched bitwise. The opt-in is keyed on
the ROCm gfx name, so NVIDIA (already bit-exact) and other AMD archs keep inductor's pick."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_compile_config as cc  # noqa: E402
from core.inference import diffusion_speed as ds  # noqa: E402

_WAN = "wan2.2-ti2v-5b"


def _family(name):
    from core.inference import diffusion_families, video_families
    for fam in (*video_families._FAMILIES, *diffusion_families._FAMILIES):
        if fam.name == name:
            return fam
    raise AssertionError(f"no family {name}")


def _on_rocm(monkeypatch, arch):
    """A ROCm torch whose current device reports ``arch`` (gcnArchName carries the feature suffix)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "7.13.99004", raising = False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    props = types.SimpleNamespace(gcnArchName = f"{arch}:sramecc-:xnack-", major = 11, minor = 5)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a, **k: props)
    # A ROCm device answers a capability too: it must never be read as an NVIDIA one.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (11, 5))


def _filter_reaching_compile(monkeypatch, family):
    seen = {}

    def fake_compile(pipe, logger, **kwargs):
        seen.update(kwargs)
        return False

    monkeypatch.setattr(ds, "compile_eligible", lambda *a, **k: True)
    monkeypatch.setattr(ds, "_compile_repeated_blocks", fake_compile)
    target = types.SimpleNamespace(device = "cpu", dtype = torch.bfloat16, backend = "cuda")
    ds.apply_speed_optims(
        types.SimpleNamespace(), target, is_gguf = False, family = family, speed_mode = ds.SPEED_DEFAULT
    )
    return seen["filter_reductions"]


def test_wan_ti2v_5b_pins_the_filter_on_gfx1151(monkeypatch):
    _on_rocm(monkeypatch, "gfx1151")
    fam = _family(_WAN)
    assert cc.family_filters_reductions(fam) is True
    assert _filter_reaching_compile(monkeypatch, fam) is True


@pytest.mark.parametrize("arch", ["gfx1100", "gfx1201", "gfx942", "gfx90a"])
def test_wan_ti2v_5b_keeps_inductor_pick_on_other_amd_archs(monkeypatch, arch):
    _on_rocm(monkeypatch, arch)
    assert cc.family_filters_reductions(_family(_WAN)) is False


@pytest.mark.parametrize("cap", [(7, 5), (8, 0), (8, 9), (9, 0), (10, 0), (12, 0), None])
def test_wan_ti2v_5b_keeps_inductor_pick_on_nvidia(monkeypatch, cap):
    monkeypatch.setattr(cc, "_device_capability", lambda: cap)
    monkeypatch.setattr(cc, "_rocm_gfx_arch", lambda: None, raising = False)
    assert cc.family_filters_reductions(_family(_WAN)) is False


def test_cuda_arch_lists_never_match_a_rocm_device(monkeypatch):
    # FLUX.1 / Qwen-Image list CUDA capabilities only: a gfx1151 card stays on inductor's pick.
    _on_rocm(monkeypatch, "gfx1151")
    for name in ("flux.1", "qwen-image", "z-image"):
        assert cc.family_filters_reductions(_family(name)) is False


def test_gfx_names_match_case_and_suffix_insensitively(monkeypatch):
    _on_rocm(monkeypatch, "gfx1151")
    for archs in (("gfx1151",), ("GFX1151",), [(12, 0), "gfx1151"]):
        assert (
            cc.family_filters_reductions(
                types.SimpleNamespace(filter_reduction_configs_archs = archs)
            )
            is True
        )
    assert (
        cc.family_filters_reductions(
            types.SimpleNamespace(filter_reduction_configs_archs = ("gfx115",))
        )
        is False
    )


def test_a_cuda_device_never_matches_a_gfx_name(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(torch, "__version__", "2.11.0+cu130", raising = False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (12, 0))
    fam = types.SimpleNamespace(filter_reduction_configs_archs = ("gfx1151",))
    assert cc.family_filters_reductions(fam) is False
    assert (
        cc.family_filters_reductions(
            types.SimpleNamespace(filter_reduction_configs_archs = ("gfx1151", (12, 0)))
        )
        is True
    )


def test_unreadable_rocm_device_pins_nothing(monkeypatch):
    _on_rocm(monkeypatch, "gfx1151")

    def boom(*a, **k):
        raise RuntimeError("no device")

    monkeypatch.setattr(torch.cuda, "get_device_properties", boom)
    assert cc.family_filters_reductions(_family(_WAN)) is False
