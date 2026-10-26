# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Ideogram 4 takes the hosted pre-cast fp8 text encoder: the per-component branch hands it to the pipeline and
reads the dense vendor-fp8 encoder only when there is none. CPU only, torch / diffusers stubbed."""

from __future__ import annotations

import dataclasses
import sys
import types

import pytest

from core.inference import diffusion as dmod
from core.inference import diffusion_families as families
from core.inference import diffusion_ideogram4 as ideogram
from core.inference.diffusion import DiffusionBackend
from .test_diffusion_backend import (  # noqa: F401 -- fixtures
    _FakePipe,
    _unmeasured_torchao,
    _write_pipeline,
    fake_runtime,
)

_HOSTED = "unsloth/ideogram-4-FP8"


def _with_hosted_te(monkeypatch):
    """The rows the family table carries once the artifact is published."""
    rows = []
    for fam in families._FAMILIES:
        if fam.name == families.IDEOGRAM4_FAMILY_NAME:
            fam = dataclasses.replace(
                fam,
                te_prequant_repos = (("fp8", "text_encoder", _HOSTED),),
                te_quant_auto = "fp8",
            )
        rows.append(fam)
    monkeypatch.setattr(families, "_FAMILIES", tuple(rows))
    return next(f for f in rows if f.name == families.IDEOGRAM4_FAMILY_NAME)


def _stub_components(monkeypatch, calls, built):
    def component(name):
        calls.append(name)
        return _FakePipe()

    monkeypatch.setattr(dmod, "load_ideogram4_pipeline", ideogram.load_ideogram4_pipeline)
    monkeypatch.setattr(ideogram, "_patch_create_causal_mask", lambda: None)
    monkeypatch.setattr(
        ideogram, "load_ideogram4_text_encoder", lambda *a, **k: component("dense_text_encoder")
    )
    monkeypatch.setattr(ideogram, "load_krea2_tokenizer", lambda *a, **k: component("tokenizer"))
    monkeypatch.setattr(
        ideogram,
        "load_ideogram4_transformer",
        lambda repo, subfolder, *a, **k: component(subfolder),
    )
    diffusers = sys.modules["diffusers"]

    def pipeline(**kwargs):
        built.update(kwargs)
        return _FakePipe()

    diffusers.Ideogram4Pipeline = pipeline
    diffusers.AutoencoderKLFlux2 = types.SimpleNamespace(
        from_pretrained = lambda *a, **k: component("vae")
    )
    diffusers.FlowMatchEulerDiscreteScheduler = types.SimpleNamespace(
        from_pretrained = lambda *a, **k: component("scheduler")
    )


@pytest.mark.parametrize("precast", [True, False])
def test_ideogram_branch_takes_precast_text_encoder_else_dense(
    fake_runtime, tmp_path, monkeypatch, precast
):
    calls: list = []
    built: dict = {}
    asked: list = []
    precast_te = object()
    _stub_components(monkeypatch, calls, built)

    def te_kwargs(fam, base, **kwargs):
        asked.append((fam.name, kwargs.get("te_quant_mode")))
        return {"text_encoder": precast_te} if precast else {}

    monkeypatch.setattr(dmod, "te_prequant_pipe_kwargs", te_kwargs)
    # The stubbed host cannot run the later runtime cast; this test is only about what assembly is handed.
    from core.inference import diffusion_auto_policy

    monkeypatch.setenv(diffusion_auto_policy._PRECISION_FALLBACK_ENV, "1")
    _write_pipeline(tmp_path, weight = "pytorch_model.bin")
    assert DiffusionBackend().load_pipeline(
        str(tmp_path),
        family_override = "ideogram-4",
        local_files_only = True,
        text_encoder_quant = "fp8",
    )["loaded"]

    assert asked == [("ideogram-4", "fp8")]
    if precast:
        assert built["text_encoder"] is precast_te
        assert "dense_text_encoder" not in calls
    else:
        assert calls.count("dense_text_encoder") == 1
        assert built["text_encoder"] is not precast_te and built["text_encoder"] is not None
    assert {"transformer", "unconditional_transformer", "vae", "scheduler"} <= set(calls)


def test_ideogram_hosted_te_resolves_for_the_upstream_and_mirror_base(monkeypatch):
    """The plan / prefetch / injection all key off te_prequant_sources_for_base: the mirror the load fetches from must
    match the registered upstream base, or the dense shards would be staged and the artifact never read."""
    from core.inference import diffusion_precision, diffusion_te_prequant as tp

    fam = _with_hosted_te(monkeypatch)
    monkeypatch.setattr(diffusion_precision, "te_quant_supported", lambda target, mode: True)
    for base in ("ideogram-ai/ideogram-4-fp8", "unsloth/ideogram-4-fp8"):
        sources = tp.te_prequant_sources_for_base(fam, base, te_quant_mode = "fp8", target = object())
        assert set(sources) == {"text_encoder"}
        assert sources["text_encoder"].location == _HOSTED
        assert sources["text_encoder"].filename == "ideogram-4-text_encoder-FP8.safetensors"
    # Dense (none) never resolves a pre-cast source, so the dense shards stay in the plan.
    assert (
        tp.te_prequant_sources_for_base(
            fam, "unsloth/ideogram-4-fp8", te_quant_mode = None, target = object()
        )
        == {}
    )
