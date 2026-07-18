# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hermetic CPU tests for the pre-cast text-encoder load path.

Mirrors tests/test_diffusion_prequant.py: resolution priority, checkpoint validation,
fallback behaviour, the local-path allowlist gate, and the pipeline-assembly injection
gating -- all without CUDA, the Hub, or a real transformers model."""

from __future__ import annotations

import types
from pathlib import Path

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
    # Families without a hosted TE checkpoint keep the empty default.
    fam = detect_family("unsloth/FLUX.1-schnell-GGUF")
    assert fam.te_prequant_repos == ()


def test_hosted_te_prequant_entries():
    """The hosted pre-cast fp8 text encoders live in the family's own -FP8 repos."""
    from core.inference.diffusion_families import detect_family
    from core.inference.video_families import detect_video_family

    assert detect_family("Qwen/Qwen-Image").te_prequant_repos == (
        ("fp8", "text_encoder", "unsloth/Qwen-Image-FP8"),
    )
    assert detect_family("black-forest-labs/FLUX.2-dev").te_prequant_repos == (
        ("fp8", "text_encoder", "unsloth/FLUX.2-dev-FP8"),
    )
    assert detect_video_family("Lightricks/LTX-2").te_prequant_repos == (
        ("fp8", "text_encoder", "unsloth/LTX-2-FP8"),
    )
    # The hosted filenames follow the repo naming convention the resolver derives.
    assert te_prequant_repo_filename(
        "unsloth/Qwen-Image-FP8", "text_encoder", "fp8"
    ) == "Qwen-Image-text_encoder-FP8.pt"
    assert te_prequant_repo_filename(
        "unsloth/FLUX.2-dev-FP8", "text_encoder", "fp8"
    ) == "FLUX.2-dev-text_encoder-FP8.pt"
    assert te_prequant_repo_filename(
        "unsloth/LTX-2-FP8", "text_encoder", "fp8"
    ) == "LTX-2-text_encoder-FP8.pt"


def test_cast_fp8_is_idempotent_on_precast_encoder():
    """A pre-cast encoder arrives with the layerwise hooks installed; the runtime re-apply in
    quantize_text_encoders must be a no-op (re-registering the hook name raises, which made
    the engaged cast report as failed and status show no TE quant)."""
    import torch

    from core.inference.diffusion_precision import _cast_fp8

    target = types.SimpleNamespace(dtype = torch.bfloat16)
    enc = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.LayerNorm(64))
    _cast_fp8(enc, target)
    assert enc[0].weight.dtype == torch.float8_e4m3fn
    _cast_fp8(enc, target)  # must not raise
    assert enc[0].weight.dtype == torch.float8_e4m3fn


def test_builder_metadata_survives_weights_only_load(tmp_path):
    """The builder's checkpoint must load with torch.load(weights_only=True): version
    metadata has to be plain str (a pickled TorchVersion object gets the whole artifact
    rejected and the loader would silently fall back to the dense download)."""
    import sys

    import torch

    scripts = Path(__file__).resolve().parents[3] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        import build_te_prequant_checkpoint  # noqa: F401  (import proves the module parses)
    finally:
        sys.path.remove(str(scripts))
    ckpt = {
        "format": TE_PREQUANT_FORMAT,
        "metadata": {
            "scheme": "fp8",
            "component": "text_encoder",
            "base_model_id": "Lightricks/LTX-2",
            "te_class": "Gemma3ForConditionalGeneration",
            "torch_version": str(torch.__version__),
            "transformers_version": "0.0.0",
        },
        "state_dict": {"weight": torch.zeros(1)},
    }
    path = tmp_path / "te.pt"
    torch.save(ckpt, path)
    loaded = torch.load(path, weights_only = True, map_location = "cpu")
    assert tpq._validate_checkpoint(loaded, "fp8", "text_encoder", "Lightricks/LTX-2", None)
    # The regression: an unstringified TorchVersion in metadata must fail weights_only.
    bad = dict(ckpt, metadata = dict(ckpt["metadata"], torch_version = torch.__version__))
    bad_path = tmp_path / "bad.pt"
    torch.save(bad, bad_path)
    if not isinstance(torch.__version__, str):
        with pytest.raises(Exception):
            torch.load(bad_path, weights_only = True, map_location = "cpu")
