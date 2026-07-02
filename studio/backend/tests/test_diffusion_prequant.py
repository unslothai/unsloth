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
        self.eval_called = False

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

    def eval(self):
        self.eval_called = True
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
    meta = {"scheme": scheme, "base_model_id": base}
    # fp8 checkpoints must record per-row granularity or the loader rejects them as stale.
    if scheme == "fp8":
        meta["fp8_granularity"] = "per_row"
    return {
        "format": PREQUANT_FORMAT,
        "metadata": meta,
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
    allow_local = True,
    fast_accum = None,
):
    _FakeTransformer.calls = {}
    _stub_torch_accelerate(monkeypatch, ckpt, load_raises = load_raises)
    # The local-path branch is opt-in via a directory ALLOWLIST (it unpickles an arbitrary
    # file); these tests exercise the load mechanics, so allowlist tmp_path (where ckpt.pt
    # lives) unless a test is checking the gate.
    if allow_local:
        monkeypatch.setenv(pq.ALLOW_LOCAL_PREQUANT_PATH_ENV, str(tmp_path))
    else:
        monkeypatch.delenv(pq.ALLOW_LOCAL_PREQUANT_PATH_ENV, raising = False)
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
        fast_accum = fast_accum,
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


def test_load_puts_transformer_in_eval_mode(monkeypatch, tmp_path):
    # Built via from_config (not from_pretrained), so the loader must eval() it to match
    # the dense/GGUF paths; otherwise train-mode dropout makes inference nondeterministic.
    t = _load(monkeypatch, tmp_path, _good_ckpt())
    assert t is not None
    assert t.eval_called is True


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


def test_load_fp8_stale_per_tensor_is_rejected(monkeypatch, tmp_path):
    # A pre-fix fp8 checkpoint has no fp8_granularity (old per-tensor layout); it must be
    # rejected so the loader rebuilds instead of reproducing the noise failure.
    stale = _good_ckpt(scheme = "fp8")
    del stale["metadata"]["fp8_granularity"]
    assert _load(monkeypatch, tmp_path, stale, scheme = "fp8") is None
    # An explicit per-tensor granularity is likewise rejected.
    per_tensor = _good_ckpt(scheme = "fp8")
    per_tensor["metadata"]["fp8_granularity"] = "per_tensor"
    assert _load(monkeypatch, tmp_path, per_tensor, scheme = "fp8") is None


def test_load_int8_ignores_fp8_granularity(monkeypatch, tmp_path):
    # The granularity gate is fp8-only: an int8 checkpoint without it still loads.
    assert _load(monkeypatch, tmp_path, _good_ckpt(scheme = "int8"), scheme = "int8") is not None


def test_load_missing_base_metadata_is_none(monkeypatch, tmp_path):
    # A checkpoint whose keys happen to match a different base can load strict=True and then
    # render from the wrong weights, so a base was requested but none recorded must be refused.
    ckpt = _good_ckpt()
    del ckpt["metadata"]["base_model_id"]
    assert _load(monkeypatch, tmp_path, ckpt) is None


def test_load_fast_accum_mismatch_is_none(monkeypatch, tmp_path):
    # fp8 fast-accum is baked into the saved kernels; an explicit request that contradicts
    # the recorded value must fall to the dense path (which honors it), not silently use it.
    ckpt = _good_ckpt()
    ckpt["metadata"]["fast_accum"] = True
    assert _load(monkeypatch, tmp_path, ckpt, fast_accum = False) is None


def test_load_fast_accum_match_ok(monkeypatch, tmp_path):
    ckpt = _good_ckpt()
    ckpt["metadata"]["fast_accum"] = True
    assert _load(monkeypatch, tmp_path, ckpt, fast_accum = True) is not None


def test_load_fast_accum_auto_ignores_baked(monkeypatch, tmp_path):
    # An auto (None) request must accept whatever the checkpoint baked, on any GPU class.
    ckpt = _good_ckpt()
    ckpt["metadata"]["fast_accum"] = True
    assert _load(monkeypatch, tmp_path, ckpt, fast_accum = None) is not None


def test_load_exclude_tokens_mismatch_is_none(monkeypatch, tmp_path):
    # An int8 checkpoint recording a stale exclusion set (would bake M=1 modulation linears
    # as int8 and crash) must be rejected rather than loaded.
    ckpt = _good_ckpt(scheme = "int8")
    ckpt["metadata"]["exclude_name_tokens"] = ["stale_token"]
    assert _load(monkeypatch, tmp_path, ckpt, scheme = "int8") is None


def test_load_exclude_tokens_match_ok(monkeypatch, tmp_path):
    from core.inference.diffusion_transformer_quant import exclude_tokens_for_scheme

    ckpt = _good_ckpt(scheme = "int8")
    ckpt["metadata"]["exclude_name_tokens"] = list(exclude_tokens_for_scheme("int8"))
    assert _load(monkeypatch, tmp_path, ckpt, scheme = "int8") is not None


def test_resolve_checkpoint_path_expands_user(monkeypatch, tmp_path):
    # The allowlist gate expands ~, so the existence check must too, or a "~/..." checkpoint
    # that passed the gate is silently skipped.
    import os

    real = tmp_path / "transformer_fp8.pt"
    real.write_bytes(b"x")
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(real) if p == "~/ckpt.pt" else p)
    source = PrequantSource(kind = "path", location = "~/ckpt.pt", filename = None)
    assert pq._resolve_checkpoint_path(source, None) == str(real)


# ── local-path opt-in gate (RCE guard) ───────────────────────────────────────────
def test_load_local_path_refused_by_default(monkeypatch, tmp_path):
    # A valid checkpoint at a real file is still refused: torch.load must never run on a
    # request-supplied path without the operator opt-in.
    called = {"load": False}

    def _explode(*a, **k):
        called["load"] = True
        raise AssertionError("torch.load must not run on a refused local path")

    torch = types.ModuleType("torch")
    torch.load = _explode
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.delenv(pq.ALLOW_LOCAL_PREQUANT_PATH_ENV, raising = False)

    path = tmp_path / "ckpt.pt"
    path.write_bytes(b"x")
    source = PrequantSource(kind = "path", location = str(path), filename = None)
    result = load_prequantized_transformer(
        _FakeTransformer,
        "Tongyi-MAI/Z-Image-Turbo",
        source,
        device = "cuda",
        dtype = "bfloat16",
        hf_token = None,
        scheme = "fp8",
        logger = None,
    )
    assert result is None
    assert called["load"] is False


def test_load_local_path_allowed_with_optin(monkeypatch, tmp_path):
    assert _load(monkeypatch, tmp_path, _good_ckpt(), allow_local = True) is not None


def test_load_repo_source_allowed_without_optin(monkeypatch, tmp_path):
    # The hosted-repo branch is first-party and trusted: it loads with no opt-in env set.
    _FakeTransformer.calls = {}
    _stub_torch_accelerate(monkeypatch, _good_ckpt())
    monkeypatch.delenv(pq.ALLOW_LOCAL_PREQUANT_PATH_ENV, raising = False)

    downloaded = tmp_path / "transformer_fp8.pt"
    downloaded.write_bytes(b"x")
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda repo_id, filename, token = None: str(downloaded)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    source = PrequantSource(kind = "repo", location = "org/hosted-fp8", filename = "transformer_fp8.pt")
    result = load_prequantized_transformer(
        _FakeTransformer,
        "Tongyi-MAI/Z-Image-Turbo",
        source,
        device = "cuda",
        dtype = "bfloat16",
        hf_token = None,
        scheme = "fp8",
        logger = None,
    )
    assert result is not None


def test_load_local_path_outside_allowlist_refused(monkeypatch, tmp_path):
    # Even with the opt-in set, a path OUTSIDE every allowlisted directory must not be
    # unpickled: enabling one trusted dir is not a wildcard for arbitrary request paths.
    called = {"load": False}

    def _explode(*a, **k):
        called["load"] = True
        raise AssertionError("torch.load must not run on a path outside the allowlist")

    torch = types.ModuleType("torch")
    torch.load = _explode
    monkeypatch.setitem(sys.modules, "torch", torch)

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setenv(pq.ALLOW_LOCAL_PREQUANT_PATH_ENV, str(allowed))

    outside = tmp_path / "evil.pt"  # a real file, but outside the allowlisted dir
    outside.write_bytes(b"x")
    source = PrequantSource(kind = "path", location = str(outside), filename = None)
    result = load_prequantized_transformer(
        _FakeTransformer,
        "Tongyi-MAI/Z-Image-Turbo",
        source,
        device = "cuda",
        dtype = "bfloat16",
        hf_token = None,
        scheme = "fp8",
        logger = None,
    )
    assert result is None
    assert called["load"] is False


def test_load_min_features_mismatch_is_none(monkeypatch, tmp_path):
    # A checkpoint built with a different --min-features quantises a different Linear set,
    # so it must be rejected when the runtime threshold is supplied.
    ckpt = _good_ckpt()
    ckpt["metadata"]["min_features"] = 256  # built with 256, runtime asks for 512
    _FakeTransformer.calls = {}
    _stub_torch_accelerate(monkeypatch, ckpt)
    monkeypatch.setenv(pq.ALLOW_LOCAL_PREQUANT_PATH_ENV, str(tmp_path))
    path = tmp_path / "ckpt.pt"
    path.write_bytes(b"x")
    source = PrequantSource(kind = "path", location = str(path), filename = None)
    result = load_prequantized_transformer(
        _FakeTransformer,
        "Tongyi-MAI/Z-Image-Turbo",
        source,
        device = "cuda",
        dtype = "bfloat16",
        hf_token = None,
        scheme = "fp8",
        min_features = 512,
        logger = None,
    )
    assert result is None


def test_load_base_fork_tail_matches(monkeypatch, tmp_path):
    # A local path / fork id with the same final segment as the canonical base is accepted.
    ckpt = _good_ckpt(base = "Tongyi-MAI/Z-Image-Turbo")
    _FakeTransformer.calls = {}
    _stub_torch_accelerate(monkeypatch, ckpt)
    monkeypatch.setenv(pq.ALLOW_LOCAL_PREQUANT_PATH_ENV, str(tmp_path))
    path = tmp_path / "ckpt.pt"
    path.write_bytes(b"x")
    source = PrequantSource(kind = "path", location = str(path), filename = None)
    result = load_prequantized_transformer(
        _FakeTransformer,
        "/local/models/Z-Image-Turbo",  # different prefix, same tail
        source,
        device = "cuda",
        dtype = "bfloat16",
        hf_token = None,
        scheme = "fp8",
        logger = None,
    )
    assert result is not None
