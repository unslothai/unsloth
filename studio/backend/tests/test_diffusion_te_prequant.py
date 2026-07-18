# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hermetic CPU tests for the pre-cast text-encoder load path.

Mirrors tests/test_diffusion_prequant.py: resolution priority, checkpoint validation,
fallback behaviour, the local-path allowlist gate, and the pipeline-assembly injection
gating -- all without CUDA, the Hub, or a real transformers model."""

from __future__ import annotations

import types

import pytest

import core.inference.diffusion_te_prequant as tpq
from core.inference.diffusion_te_prequant import (
    TE_PREQUANT_FORMAT,
    TePrequantSource,
    family_te_prequant_repo,
    resolve_te_prequant_source,
    te_prequant_pipe_kwargs,
    te_prequant_repo_filename,
)


def _fam(te_prequant_repos = (), name = "ltx-2"):
    return types.SimpleNamespace(name = name, te_prequant_repos = te_prequant_repos)


# ── resolution ───────────────────────────────────────────────────────────────
def test_repo_filename_convention():
    assert (
        te_prequant_repo_filename("unsloth/LTX-2-FP8", "text_encoder", "fp8")
        == "LTX-2-text_encoder-FP8.pt"
    )
    assert (
        te_prequant_repo_filename("org/Some-Model-quantized", "text_encoder_2", "fp8")
        == "Some-Model-text_encoder_2-FP8.pt"
    )
    assert (
        te_prequant_repo_filename("org/PlainRepo", "text_encoder", "fp8")
        == "PlainRepo-text_encoder-FP8.pt"
    )


def test_family_repo_by_scheme_and_component():
    fam = _fam(
        te_prequant_repos = (
            ("fp8", "text_encoder", "org/hosted-fp8"),
            ("fp8", "text_encoder_2", "org/hosted-2-fp8"),
        )
    )
    assert family_te_prequant_repo(fam, "fp8", "text_encoder") == "org/hosted-fp8"
    assert family_te_prequant_repo(fam, "fp8", "text_encoder_2") == "org/hosted-2-fp8"
    assert family_te_prequant_repo(fam, "fp8", "text_encoder_3") is None
    assert family_te_prequant_repo(fam, "int8", "text_encoder") is None
    # A malformed entry is skipped, not fatal.
    assert family_te_prequant_repo(_fam(te_prequant_repos = (("bad",),)), "fp8", "text_encoder") is None
    # Families without the field resolve to None (both dataclasses default it, but a fake
    # or an older family object must not break).
    assert family_te_prequant_repo(types.SimpleNamespace(name = "x"), "fp8", "text_encoder") is None


def test_resolve_priority_and_scheme_gate():
    fam = _fam(te_prequant_repos = (("fp8", "text_encoder", "org/hosted-fp8"),))
    # Path override wins.
    src = resolve_te_prequant_source(fam, "text_encoder", "fp8", path_override = "/tmp/te.pt")
    assert src == TePrequantSource(kind = "path", location = "/tmp/te.pt", filename = None)
    # Hosted repo second.
    src = resolve_te_prequant_source(fam, "text_encoder", "fp8")
    assert src.kind == "repo" and src.location == "org/hosted-fp8"
    assert src.filename == "hosted-text_encoder-FP8.pt"
    # Nothing configured -> None.
    assert resolve_te_prequant_source(_fam(), "text_encoder", "fp8") is None
    # v1 hosts the layerwise fp8 storage scheme only.
    assert resolve_te_prequant_source(fam, "text_encoder", "int8") is None
    assert resolve_te_prequant_source(fam, "text_encoder", "fp8_dynamic") is None


# ── checkpoint validation ────────────────────────────────────────────────────
def _good_ckpt(scheme = "fp8", component = "text_encoder", base = "Lightricks/LTX-2"):
    return {
        "format": TE_PREQUANT_FORMAT,
        "metadata": {
            "scheme": scheme,
            "component": component,
            "base_model_id": base,
            "te_class": "Gemma3ForConditionalGeneration",
        },
        "state_dict": {"weight": object()},
    }


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (lambda c: c.update(format = "other"), "format"),
        (lambda c: c.pop("state_dict"), "state_dict"),
        (lambda c: c["metadata"].update(scheme = "int8"), "scheme"),
        (lambda c: c["metadata"].update(component = "text_encoder_2"), "component"),
        (lambda c: c["metadata"].update(base_model_id = "other/repo"), "base"),
        (lambda c: c["metadata"].pop("base_model_id"), "missing base"),
    ],
)
def test_validate_rejects_mismatches(mutate, reason):
    ckpt = _good_ckpt()
    mutate(ckpt)
    assert (
        tpq._validate_checkpoint(ckpt, "fp8", "text_encoder", "Lightricks/LTX-2", None) is False
    ), reason


def test_validate_accepts_good_checkpoint_and_base_case_folding():
    assert tpq._validate_checkpoint(_good_ckpt(), "fp8", "text_encoder", "Lightricks/LTX-2", None)
    # _same_base_model folds case like the DiT module.
    assert tpq._validate_checkpoint(_good_ckpt(), "fp8", "text_encoder", "lightricks/ltx-2", None)


# ── loader fallback behaviour ────────────────────────────────────────────────
def test_load_refuses_unallowlisted_local_path(monkeypatch, tmp_path):
    from core.inference.diffusion_prequant import ALLOW_LOCAL_PREQUANT_PATH_ENV

    monkeypatch.delenv(ALLOW_LOCAL_PREQUANT_PATH_ENV, raising = False)
    path = tmp_path / "te.pt"
    path.write_bytes(b"x")
    out = tpq.load_prequant_text_encoder(
        "Lightricks/LTX-2",
        "text_encoder",
        TePrequantSource(kind = "path", location = str(path)),
        dtype = None,
    )
    assert out is None  # refused, caller falls back to dense


def test_load_missing_file_returns_none(monkeypatch, tmp_path):
    from core.inference.diffusion_prequant import ALLOW_LOCAL_PREQUANT_PATH_ENV

    monkeypatch.setenv(ALLOW_LOCAL_PREQUANT_PATH_ENV, str(tmp_path))
    out = tpq.load_prequant_text_encoder(
        "Lightricks/LTX-2",
        "text_encoder",
        TePrequantSource(kind = "path", location = str(tmp_path / "absent.pt")),
        dtype = None,
    )
    assert out is None


# ── pipeline-assembly injection gating ───────────────────────────────────────
def _target():
    return types.SimpleNamespace(device = "cuda", dtype = None)


def test_pipe_kwargs_empty_when_mode_not_fp8(monkeypatch):
    fam = _fam(te_prequant_repos = (("fp8", "text_encoder", "org/hosted"),))
    for mode in (None, "", "off", "int8", "fp8_dynamic"):
        assert te_prequant_pipe_kwargs(
            fam, "Lightricks/LTX-2", te_quant_mode = mode, target = _target(), dtype = None
        ) == {}


def test_pipe_kwargs_empty_without_hosted_entry(monkeypatch):
    import core.inference.diffusion_precision as precision

    monkeypatch.setattr(precision, "te_quant_supported", lambda target, mode: True)
    assert te_prequant_pipe_kwargs(
        _fam(), "Lightricks/LTX-2", te_quant_mode = "fp8", target = _target(), dtype = None
    ) == {}


def test_pipe_kwargs_empty_when_device_unsupported(monkeypatch):
    import core.inference.diffusion_precision as precision

    fam = _fam(te_prequant_repos = (("fp8", "text_encoder", "org/hosted"),))
    monkeypatch.setattr(precision, "te_quant_supported", lambda target, mode: False)
    assert te_prequant_pipe_kwargs(
        fam, "Lightricks/LTX-2", te_quant_mode = "fp8", target = _target(), dtype = None
    ) == {}


def test_pipe_kwargs_respects_family_deny(monkeypatch):
    import core.inference.diffusion_precision as precision

    fam = _fam(te_prequant_repos = (("fp8", "text_encoder", "org/hosted"),))
    monkeypatch.setattr(precision, "te_quant_supported", lambda target, mode: True)
    # The deny helper ships on the video branch's precision module; simulate it here.
    monkeypatch.setattr(
        precision, "_te_family_denied", lambda family, mode: family == "ltx-2", raising = False
    )
    assert te_prequant_pipe_kwargs(
        fam, "Lightricks/LTX-2", te_quant_mode = "fp8", target = _target(), dtype = None
    ) == {}


def test_pipe_kwargs_injects_loaded_encoder(monkeypatch):
    import core.inference.diffusion_precision as precision

    fam = _fam(te_prequant_repos = (("fp8", "text_encoder", "org/hosted"),))
    monkeypatch.setattr(precision, "te_quant_supported", lambda target, mode: True)
    marker = object()
    seen = {}

    def fake_load(base, component, source, **kw):
        seen.update(base = base, component = component, source = source)
        return marker

    monkeypatch.setattr(tpq, "load_prequant_text_encoder", fake_load)
    out = te_prequant_pipe_kwargs(
        fam, "Lightricks/LTX-2", te_quant_mode = "fp8", target = _target(), dtype = None
    )
    assert out == {"text_encoder": marker}
    assert seen["base"] == "Lightricks/LTX-2"
    assert seen["source"].location == "org/hosted"


def test_pipe_kwargs_empty_when_load_fails(monkeypatch):
    import core.inference.diffusion_precision as precision

    fam = _fam(te_prequant_repos = (("fp8", "text_encoder", "org/hosted"),))
    monkeypatch.setattr(precision, "te_quant_supported", lambda target, mode: True)
    monkeypatch.setattr(tpq, "load_prequant_text_encoder", lambda *a, **k: None)
    assert te_prequant_pipe_kwargs(
        fam, "Lightricks/LTX-2", te_quant_mode = "fp8", target = _target(), dtype = None
    ) == {}


# ── family field wiring ──────────────────────────────────────────────────────
def test_family_dataclasses_declare_te_prequant_field():
    from core.inference.diffusion_families import DiffusionFamily, detect_family
    from core.inference.video_families import VideoFamily

    assert DiffusionFamily.__dataclass_fields__["te_prequant_repos"].default_factory is tuple
    assert VideoFamily.__dataclass_fields__["te_prequant_repos"].default_factory is tuple
    # No family ships a hosted TE checkpoint yet: the campaign wires entries after the
    # artifacts are gate-validated and uploaded.
    fam = detect_family("unsloth/FLUX.1-schnell-GGUF")
    assert fam.te_prequant_repos == ()
