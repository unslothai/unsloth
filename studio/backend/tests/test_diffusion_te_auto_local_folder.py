# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An unset text-encoder request on a local folder named like the upstream repo keeps the folder's own encoder."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_te_prequant as tpq  # noqa: E402
from core.inference.diffusion_families import detect_family  # noqa: E402
from core.inference.diffusion_precision import resolve_te_quant_request  # noqa: E402

CUDA_BF16 = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)

# family -> (folder named like the upstream repo, component the family hosts a file for)
FAMILIES = {
    "flux.1-kontext": ("FLUX.1-Kontext-dev", "text_encoder_2"),
    "qwen-image-layered": ("Qwen-Image-Layered", "text_encoder"),
    "qwen-image": ("Qwen-Image", "text_encoder"),
    "flux.2-dev": ("FLUX.2-dev", "text_encoder"),
}


def _auto_mode(fam):
    mode, auto = resolve_te_quant_request(None, getattr(fam, "te_quant_auto", None))
    assert auto and mode is not None
    return mode


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_unset_request_on_local_folder_takes_no_hosted_encoder(family, tmp_path):
    folder, component = FAMILIES[family]
    fam = detect_family("", override = family)
    local = tmp_path / folder
    local.mkdir()
    mode = _auto_mode(fam)
    hub = tpq.te_prequant_sources_for_base(fam, fam.base_repo, te_quant_mode = mode, target = CUDA_BF16)
    assert component in hub
    assert (
        tpq.te_prequant_sources_for_base(fam, str(local), te_quant_mode = mode, target = CUDA_BF16)
        == {}
    )
    # Neither does a request made inside a load whose raw request was unset (plain string mode).
    with tpq.auto_te_request(True):
        assert (
            tpq.te_prequant_sources_for_base(
                fam, str(local), te_quant_mode = str(mode), target = CUDA_BF16
            )
            == {}
        )


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_explicit_request_on_local_folder_is_unchanged(family, tmp_path):
    folder, component = FAMILIES[family]
    fam = detect_family("", override = family)
    local = tmp_path / folder
    local.mkdir()
    explicit = str(_auto_mode(fam))
    hub = tpq.te_prequant_sources_for_base(
        fam, fam.base_repo, te_quant_mode = explicit, target = CUDA_BF16
    )
    got = tpq.te_prequant_sources_for_base(
        fam, str(local), te_quant_mode = explicit, target = CUDA_BF16
    )
    assert got == hub and component in got


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_unset_request_on_local_folder_injects_nothing(family, tmp_path, monkeypatch):
    folder, component = FAMILIES[family]
    fam = detect_family("", override = family)
    local = tmp_path / folder
    local.mkdir()
    sentinel = object()
    calls = []

    def fake_load(base, comp, source, **kwargs):
        calls.append((base, comp, source.location))
        return sentinel

    monkeypatch.setattr(tpq, "load_prequant_text_encoder", fake_load)
    monkeypatch.setattr(tpq, "supplied_component_pipe_kwargs", lambda *a, **k: {})
    mode = _auto_mode(fam)
    with tpq.auto_te_request(True):
        local_kwargs = tpq.te_prequant_pipe_kwargs(
            fam,
            str(local),
            te_quant_mode = mode,
            target = CUDA_BF16,
            dtype = torch.bfloat16,
            local_files_only = True,
            dense_source = str(local),
        )
        hub_kwargs = tpq.te_prequant_pipe_kwargs(
            fam,
            fam.base_repo,
            te_quant_mode = mode,
            target = CUDA_BF16,
            dtype = torch.bfloat16,
            local_files_only = True,
        )
    assert component not in local_kwargs
    assert hub_kwargs.get(component) is sentinel
    assert all(base == fam.base_repo for base, _c, _l in calls)


def test_non_local_ids_still_resolve(tmp_path):
    fam = detect_family("", override = "qwen-image")
    mode = _auto_mode(fam)
    assert tpq.te_prequant_sources_for_base(
        fam, "Qwen/Qwen-Image", te_quant_mode = mode, target = CUDA_BF16
    )
    assert not tpq._is_local_base("Qwen/Qwen-Image")
    assert tpq._is_local_base(str(tmp_path))
    assert not tpq._is_local_base("")
