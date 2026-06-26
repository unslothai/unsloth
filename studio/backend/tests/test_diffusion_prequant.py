# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hermetic CPU tests for the pre-quantized transformer load path.

torch / accelerate are stubbed via ``sys.modules`` (the module under test imports them
lazily), and ``transformer_cls`` is a fake that records calls -- so the resolver, the
meta-init + ``load_state_dict(assign=True)`` flow, and the validation/fallback behaviour
are all exercised without CUDA, torchao, or a real diffusers model.
"""

from __future__ import annotations

import contextlib
import sys
import types

import core.inference.diffusion_prequant as pq
from core.inference.diffusion_families import DiffusionFamily
from core.inference.diffusion_prequant import (
    PREQUANT_FORMAT,
    PrequantSource,
    load_prequantized_transformer,
    resolve_prequant_source,
)


# ── resolve_prequant_source ──────────────────────────────────────────────────────
def _fam(prequant_repos = ()):
    return DiffusionFamily(
        name = "z-image",
        pipeline_class = "ZImagePipeline",
        transformer_class = "ZImageTransformer2DModel",
        base_repo = "Tongyi-MAI/Z-Image-Turbo",
        prequant_repos = prequant_repos,
    )


def test_resolve_path_override_wins():
    fam = _fam(prequant_repos = (("fp8", "org/hosted-fp8"),))
    src = resolve_prequant_source(fam, "fp8", path_override = "/tmp/local.pt")
    assert src == PrequantSource(kind = "path", location = "/tmp/local.pt", filename = None)


def test_resolve_family_repo_by_scheme():
    fam = _fam(prequant_repos = (("fp8", "org/hosted-fp8"), ("int8", "org/hosted-int8")))
    src = resolve_prequant_source(fam, "int8")
    assert src.kind == "repo" and src.location == "org/hosted-int8"
    assert src.filename == "transformer_int8.pt"


def test_resolve_wrong_scheme_is_none():
    fam = _fam(prequant_repos = (("fp8", "org/hosted-fp8"),))
    assert resolve_prequant_source(fam, "int8") is None


def test_resolve_nothing_configured_is_none():
    assert resolve_prequant_source(_fam(), "fp8") is None
    assert resolve_prequant_source(_fam(), "fp8", path_override = "") is None


# ── load_prequantized_transformer ────────────────────────────────────────────────
class _FakeTransformer:
    calls: dict = {}

    def __init__(self):
        self.assigned = None
        self.moved = None

    @classmethod
    def load_config(cls, base, **kw):
        cls.calls["load_config"] = {"base": base, **kw}
        return {"cfg": True}

    @classmethod
    def from_config(cls, config):
        cls.calls["from_config"] = config
        return cls()

    @classmethod
    def from_pretrained(cls, *a, **k):  # the dense path -- must never run here
        cls.calls["from_pretrained"] = True
        raise AssertionError("from_pretrained must not be called on the prequant path")

    def load_state_dict(
        self,
        sd,
        strict = True,
        assign = False,
    ):
        _FakeTransformer.calls["load_state_dict"] = {"strict": strict, "assign": assign}
        self.assigned = sd

    def parameters(self):
        return []

    def buffers(self):
        return []

    def to(self, device):
        self.moved = device
        return self


def _stub_torch_accelerate(
    monkeypatch,
    ckpt,
    *,
    load_raises = False,
):
    torch = types.ModuleType("torch")

    def _load(
        path,
        weights_only = False,
        map_location = None,
    ):
        if load_raises:
            raise RuntimeError("corrupt checkpoint")
        return ckpt

    torch.load = _load
    monkeypatch.setitem(sys.modules, "torch", torch)

    accelerate = types.ModuleType("accelerate")
    accelerate.init_empty_weights = lambda: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "accelerate", accelerate)


def _good_ckpt(scheme = "fp8", base = "Tongyi-MAI/Z-Image-Turbo"):
    return {
        "format": PREQUANT_FORMAT,
        "metadata": {"scheme": scheme, "base_model_id": base},
        "state_dict": {"weight": object()},
    }


def _load(
    monkeypatch,
    tmp_path,
    ckpt,
    *,
    scheme = "fp8",
    load_raises = False,
    exists = True,
):
    _FakeTransformer.calls = {}
    _stub_torch_accelerate(monkeypatch, ckpt, load_raises = load_raises)
    path = tmp_path / "ckpt.pt"
    if exists:
        path.write_bytes(b"x")
    source = PrequantSource(kind = "path", location = str(path), filename = None)
    return load_prequantized_transformer(
        _FakeTransformer,
        "Tongyi-MAI/Z-Image-Turbo",
        source,
        device = "cuda",
        dtype = "bfloat16",
        hf_token = None,
        scheme = scheme,
        logger = None,
    )


def test_load_meta_init_and_assign(monkeypatch, tmp_path):
    t = _load(monkeypatch, tmp_path, _good_ckpt())
    assert t is not None
    # meta-init path was used, not the dense from_pretrained.
    assert "from_config" in _FakeTransformer.calls
    assert "from_pretrained" not in _FakeTransformer.calls
    # assign=True is the whole point (copy into meta is a no-op).
    assert _FakeTransformer.calls["load_state_dict"] == {"strict": True, "assign": True}
    assert t.moved == "cuda"
    assert t._unsloth_runtime_quant == "fp8"


def test_load_missing_file_is_none(monkeypatch, tmp_path):
    assert _load(monkeypatch, tmp_path, _good_ckpt(), exists = False) is None


def test_load_torch_load_raises_is_none(monkeypatch, tmp_path):
    assert _load(monkeypatch, tmp_path, _good_ckpt(), load_raises = True) is None


def test_load_format_mismatch_is_none(monkeypatch, tmp_path):
    bad = _good_ckpt()
    bad["format"] = "something_else"
    assert _load(monkeypatch, tmp_path, bad) is None


def test_load_scheme_mismatch_is_none(monkeypatch, tmp_path):
    # checkpoint built for int8, but fp8 was requested.
    assert _load(monkeypatch, tmp_path, _good_ckpt(scheme = "int8"), scheme = "fp8") is None


def test_load_base_mismatch_is_none(monkeypatch, tmp_path):
    assert _load(monkeypatch, tmp_path, _good_ckpt(base = "other/model")) is None
