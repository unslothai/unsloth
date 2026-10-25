# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An automatic int8 text-encoder pick is budgeted at its hosted int8 ConvRot file, not the fp8 ceiling.

The auto pick loads the int8 file or the dense encoder, never the fp8 file (``te_prequant_sources`` drops it), so
``TE_PREQUANT_BUDGET_SCALE`` (the fp8 files' 0.65 ceiling) over-stated every auto load on CUDA. The Studio picker
sizes the same file (model-catalog.ts ``autoQuantTextEncoderGb``), so the two now agree.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_te_prequant as tpq  # noqa: E402
from core.inference.diffusion_families import _FAMILIES  # noqa: E402
from core.inference.diffusion_precision import resolve_te_quant_request  # noqa: E402

CUDA_BF16 = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)

# family -> (bf16 encoder bytes, hosted int8 ConvRot file bytes), from the Hub listing of the family's FP8 repo.
_HOSTED_INT8 = {
    "flux.3-klein": (8_875_719_344, 5_247_076_320),
    "flux.3-klein-base": (8_875_719_344, 5_247_076_320),
    "qwen-image-2.1": (17_534_340_584, 9_349_769_248),
}


def _fam(name):
    return next(f for f in _FAMILIES if f.name == name)


@pytest.fixture
def cuda_env(monkeypatch):
    import core.inference.diffusion as D
    import core.inference.diffusion_precision as prec
    from core.inference.diffusion_text_encoder_trim import KEEP_LM_HEAD_ENV

    monkeypatch.delenv(KEEP_LM_HEAD_ENV, raising = False)
    supported = lambda target, mode: getattr(target, "device", None) == "cuda"
    monkeypatch.setattr(prec, "te_quant_supported", supported)
    monkeypatch.setattr(D, "te_quant_supported", supported, raising = False)
    monkeypatch.setattr(tpq, "te_prequant_sources_confirmed", lambda sources, **k: True)
    return CUDA_BF16


def _auto_mode(fam):
    mode, auto = resolve_te_quant_request(None, fam.te_quant_auto)
    assert auto and mode == "int8", fam.name
    return mode


@pytest.mark.parametrize("name", sorted(_HOSTED_INT8))
def test_auto_int8_budgets_the_int8_file(name, cuda_env):
    fam = _fam(name)
    scale = tpq.te_prequant_budget_scale(
        fam, te_quant_mode = _auto_mode(fam), target = cuda_env, base = fam.base_repo
    )
    assert scale == tpq.te_int8_convrot_budget_scale(fam)
    assert scale < tpq.TE_PREQUANT_BUDGET_SCALE
    dense, hosted = _HOSTED_INT8[name]
    # Never below the file it loads: an under-estimate lets an oversized load through.
    assert scale >= hosted / dense


@pytest.mark.parametrize("name", sorted(_HOSTED_INT8))
def test_explicit_int8_keeps_the_fp8_fallback_budget(name, cuda_env):
    # An explicit int8 may fall back to the fp8 file, so it keeps the fp8 ceiling.
    fam = _fam(name)
    scale = tpq.te_prequant_budget_scale(
        fam, te_quant_mode = "int8", target = cuda_env, base = fam.base_repo
    )
    assert scale == tpq.TE_PREQUANT_BUDGET_SCALE


def test_auto_fp8_pick_is_unchanged(cuda_env):
    fam = _fam("flux.1-kontext")
    mode, auto = resolve_te_quant_request(None, fam.te_quant_auto)
    assert auto and mode == "fp8"
    explicit = tpq.te_prequant_budget_scale(
        fam, te_quant_mode = "fp8", target = cuda_env, base = fam.base_repo
    )
    scale = tpq.te_prequant_budget_scale(
        fam, te_quant_mode = mode, target = cuda_env, base = fam.base_repo
    )
    assert explicit == tpq.TE_PREQUANT_BUDGET_SCALE
    # Only the CLIP-L sibling stays dense; the hosted T5 share is still priced at the fp8 ceiling.
    assert tpq.TE_PREQUANT_BUDGET_SCALE < scale < 1.0


def test_an_unconfirmed_auto_int8_file_still_budgets_dense(cuda_env, monkeypatch):
    monkeypatch.setattr(tpq, "te_prequant_sources_confirmed", lambda sources, **k: False)
    fam = _fam("qwen-image-2.1")
    scale = tpq.te_prequant_budget_scale(
        fam, te_quant_mode = _auto_mode(fam), target = cuda_env, base = fam.base_repo
    )
    assert scale == 1.0


def test_the_uncached_precast_price_of_an_auto_int8_pick_is_the_int8_file(cuda_env, monkeypatch):
    from core.inference import diffusion_prequant as pq
    from core.inference.diffusion import DiffusionBackend
    from core.inference.diffusion_auto_policy import family_bf16_components_gb

    monkeypatch.setattr(pq, "prefer_cached_pickle_twins", lambda repo, names, **k: names)
    monkeypatch.setattr(pq, "first_cached_as_resolved", lambda *a, **k: None)
    fam = _fam("qwen-image-2.1")
    mib, _components, exact = DiffusionBackend._precast_text_encoder_mib(
        fam, fam.base_repo, cuda_env, _auto_mode(fam)
    )
    encoders = family_bf16_components_gb(fam, fam.base_repo)[1]
    assert not exact
    assert mib == int(encoders * 1000**3 * tpq.te_int8_convrot_budget_scale(fam)) // (1024 * 1024)
    explicit, _c, _e = DiffusionBackend._precast_text_encoder_mib(
        fam, fam.base_repo, cuda_env, "int8"
    )
    assert explicit == int(encoders * 1000**3 * tpq.TE_PREQUANT_BUDGET_SCALE) // (1024 * 1024)


# Hosted int8 ConvRot file bytes of every automatic int8 pick, from the Hub listing of the family's FP8 repo.
_AUTO_INT8_FILE_BYTES = {
    "qwen-image": 8_974_915_168,
    "qwen-image-edit": 8_974_915_168,
    "hunyuanimage-2.1": 8_974_915_168,
    "krea-2": 5_247_054_376,
    "flux.2-dev": 24_464_184_096,
    "flux.3-klein": 5_247_076_320,
    "flux.3-klein-base": 5_247_076_320,
    "qwen-image-2.1": 9_349_769_248,
    "moss-image": 11_488_171_720,
}


def test_every_auto_int8_family_is_listed():
    auto = {f.name for f in _FAMILIES if getattr(f, "te_quant_auto", None) == "int8"}
    assert auto <= set(_AUTO_INT8_FILE_BYTES)


@pytest.mark.parametrize("name", sorted(_AUTO_INT8_FILE_BYTES))
def test_the_auto_int8_budget_covers_the_file_and_dense_siblings(name, cuda_env):
    # table bf16 term * scale must cover the int8 file plus dense siblings, or an oversized load slips through.
    from core.inference.diffusion_auto_policy import family_bf16_components_gb

    fam = _fam(name)
    scale = tpq.te_prequant_budget_scale(
        fam, te_quant_mode = _auto_mode(fam), target = cuda_env, base = fam.base_repo
    )
    table_gb = family_bf16_components_gb(fam, fam.base_repo)[1]
    sibling_gb = tpq._TE_DENSE_SIBLING_GB.get(name, 0.0)
    assert table_gb * 1e9 * scale >= _AUTO_INT8_FILE_BYTES[name] + sibling_gb * 1e9
