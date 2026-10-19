# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.1, Z-Image and Qwen-Image pin inductor's reduction-config filter on sm120, sm80 and sm89, so one seed renders
one image in every fresh server there.

A few of their norm reductions get several configs that sum in different orders, and a cold-cache server benchmarks
them in its own process on near-equal timings: fresh servers rendered 2-4 variants per seed on an RTX PRO 6000, A100
and L4. B200 servers were already deterministic for these families, so the opt-in is per (family, compute
capability)."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_compile_config as cc  # noqa: E402
from core.inference import diffusion_speed as ds  # noqa: E402

_SM120 = (12, 0)
_MEASURED_ARCHS = ((8, 0), (8, 9), (12, 0))
_IMAGE_FAMILIES = ("flux.1", "z-image", "qwen-image", "flux.1-kontext", "qwen-image-edit")


def _family(name):
    from core.inference import diffusion_families, video_families
    for fam in (*video_families._FAMILIES, *diffusion_families._FAMILIES):
        if fam.name == name:
            return fam
    raise AssertionError(f"no family {name}")


def _on(monkeypatch, cap):
    monkeypatch.setattr(cc, "_device_capability", lambda: cap)


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


@pytest.mark.parametrize("cap", _MEASURED_ARCHS)
@pytest.mark.parametrize("name", _IMAGE_FAMILIES)
def test_image_family_opts_in_on_the_measured_archs(monkeypatch, name, cap):
    fam = _family(name)
    assert cap in fam.filter_reduction_configs_archs
    _on(monkeypatch, cap)
    assert cc.family_filters_reductions(fam) is True
    assert _filter_reaching_compile(monkeypatch, fam) is True


def test_the_race_families_share_one_arch_list():
    from core.inference import diffusion_families
    for name in _IMAGE_FAMILIES:
        assert (
            _family(name).filter_reduction_configs_archs == diffusion_families._REDUCTION_RACE_ARCHS
        )
    assert set(diffusion_families._REDUCTION_RACE_ARCHS) == set(_MEASURED_ARCHS)


@pytest.mark.parametrize("cap", [(10, 0), (10, 3), (9, 0), (7, 5), (12, 1), None])
@pytest.mark.parametrize("name", _IMAGE_FAMILIES)
def test_image_family_keeps_inductor_pick_elsewhere(monkeypatch, name, cap):
    # B200 (measured deterministic), B300, H100, T4, other Blackwell parts (unmeasured), no CUDA / ROCm.
    _on(monkeypatch, cap)
    fam = _family(name)
    assert cc.family_filters_reductions(fam) is False
    assert _filter_reaching_compile(monkeypatch, fam) is False


@pytest.mark.parametrize("name", ["sdxl", "hunyuanvideo-1.5", "wan2.2-ti2v-5b"])
def test_unmeasured_families_stay_off_on_sm120(monkeypatch, name):
    _on(monkeypatch, _SM120)
    assert cc.family_filters_reductions(_family(name)) is False


def test_all_arch_opt_in_still_applies_everywhere(monkeypatch):
    for cap in (_SM120, (10, 0), None):
        _on(monkeypatch, cap)
        assert cc.family_filters_reductions(_family("ltx-2")) is True


def test_no_family_or_plain_object(monkeypatch):
    _on(monkeypatch, _SM120)
    assert cc.family_filters_reductions(None) is False
    assert cc.family_filters_reductions(types.SimpleNamespace()) is False
    assert (
        cc.family_filters_reductions(
            types.SimpleNamespace(filter_reduction_configs_archs = [[12, 0]])
        )
        is True
    )


def test_capability_query_never_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)

    def boom(*a, **k):
        raise RuntimeError("no device")

    monkeypatch.setattr(torch.cuda, "get_device_capability", boom)
    assert cc._device_capability() is None
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (12, 0))
    assert cc._device_capability() == _SM120
    monkeypatch.setattr(torch.version, "hip", "6.4", raising = False)
    assert cc._device_capability() is None


def test_load_paths_key_the_bundle_on_the_same_decision():
    # The bundle key follows the arch-scoped decision, else a load reuses a bundle compiled the other way.
    import inspect

    from core.inference import diffusion

    src = inspect.getsource(diffusion)
    assert src.count("reduction_filter = family_filters_reductions(") == 2
    assert 'getattr(fam, "filter_reduction_configs"' not in src
    assert 'getattr(state.family, "filter_reduction_configs"' not in src
