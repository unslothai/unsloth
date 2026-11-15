# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A UNet pipeline (SDXL) has no dense quant scheme and no hosted seed, so planning its load must not run the
transformer-quant scheme probe (a spawned child process, about 3 s of every SDXL load on an L4)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


def _fam(name):
    from core.inference.diffusion_families import _FAMILIES
    return next(f for f in _FAMILIES if f.name == name)


def test_sdxl_plans_no_seed_without_probing(monkeypatch):
    from core.inference import diffusion as D

    probed = []

    def boom(*a, **k):
        # The planner swallows exceptions, so record the call instead of raising.
        probed.append(True)
        return None

    monkeypatch.setattr(D, "_planned_quant_scheme", boom)
    monkeypatch.setattr(
        D, "dense_transformer_supported", lambda *a, **k: probed.append(True) or True
    )
    monkeypatch.setattr(D, "_pipeline_quant_uncompilable_reason", lambda *a, **k: None)
    monkeypatch.setattr(D, "_auto_keeps_bf16_reason", lambda *a, **k: None)
    backend = D.DiffusionBackend.__new__(D.DiffusionBackend)
    backend._target_for_ordinal = lambda *a, **k: object()
    fam = _fam("sdxl")
    assert fam.denoiser_attr == "unet" and not fam.prequant_repos
    got = backend._pipeline_planned_denoiser_scheme(
        fam,
        base = fam.base_repo,
        kind = "pipeline",
        transformer_quant = None,
        speed_mode = None,
        repo_id = fam.base_repo,
    )
    assert got is None
    assert probed == []


def test_a_transformer_family_still_reaches_the_scheme_pick(monkeypatch):
    from core.inference import diffusion as D

    seen = []

    def stop(*a, **k):
        seen.append(True)
        return False

    monkeypatch.setattr(D, "dense_transformer_supported", stop)
    backend = D.DiffusionBackend.__new__(D.DiffusionBackend)
    backend._target_for_ordinal = lambda *a, **k: object()
    fam = _fam("z-image")
    backend._pipeline_planned_denoiser_scheme(
        fam,
        base = fam.base_repo,
        kind = "pipeline",
        transformer_quant = None,
        speed_mode = None,
        repo_id = fam.base_repo,
    )
    assert seen == [True]
