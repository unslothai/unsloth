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

import pytest

import core.inference.diffusion_prequant as pq
from core.inference.diffusion_families import DiffusionFamily
from core.inference.diffusion_prequant import (
    PREQUANT_FORMAT,
    PrequantSource,
    load_prequantized_transformer,
    resolve_prequant_source,
)


# ── resolve_prequant_source ──────────────────────────────────────────────────────
def _fam(prequant_repos = (), prequant_variant_repos = ()):
    return DiffusionFamily(
        name = "z-image",
        pipeline_class = "ZImagePipeline",
        transformer_class = "ZImageTransformer2DModel",
        base_repo = "Tongyi-MAI/Z-Image-Turbo",
        prequant_repos = prequant_repos,
        prequant_variant_repos = prequant_variant_repos,
    )


def test_resolve_path_override_wins():
    fam = _fam(prequant_repos = (("fp8", "org/hosted-fp8"),))
    src = resolve_prequant_source(fam, "fp8", path_override = "/tmp/local.pt")
    assert src == PrequantSource(kind = "path", location = "/tmp/local.pt", filename = None)


def test_resolve_family_repo_by_scheme():
    fam = _fam(prequant_repos = (("fp8", "org/hosted-fp8"), ("int8", "org/hosted-int8")))
    src = resolve_prequant_source(fam, "int8")
    assert src.kind == "repo" and src.location == "org/hosted-int8"
    # Model-name convention first (repo scheme suffix stripped), legacy name as fallback.
    assert src.filename == "hosted-INT8.pt"
    assert src.fallback_filename == "transformer_int8.pt"


def test_prequant_repo_filename_convention():
    from core.inference.diffusion_prequant import prequant_repo_filename

    assert prequant_repo_filename("unsloth/Z-Image-Turbo-FP8", "int8") == "Z-Image-Turbo-INT8.pt"
    assert prequant_repo_filename("unsloth/Z-Image-Turbo-FP8", "fp8") == "Z-Image-Turbo-FP8.pt"
    assert (
        prequant_repo_filename("unsloth/Qwen-Image-2512-INT8", "int8") == "Qwen-Image-2512-INT8.pt"
    )
    assert prequant_repo_filename("org/Some-Model-quantized", "fp8") == "Some-Model-FP8.pt"
    assert prequant_repo_filename("org/PlainRepo", "int8") == "PlainRepo-INT8.pt"


def test_resolve_variant_base_picks_variant_repo():
    # A base with its own baked checkpoint resolves to the variant repo; case-insensitive.
    fam = _fam(
        prequant_repos = (("int8", "org/default-fp8"),),
        prequant_variant_repos = (("org/model-dev", "int8", "org/dev-fp8"),),
    )
    src = resolve_prequant_source(fam, "int8", base_repo = "Org/Model-DEV")
    assert src.kind == "repo" and src.location == "org/dev-fp8"
    assert src.filename == "dev-INT8.pt"


def test_resolve_variant_base_falls_back_to_default():
    # An unknown variant base (or none) keeps the family default entry: base_model_id validation then refuses it and dense-quantises.
    fam = _fam(
        prequant_repos = (("int8", "org/default-fp8"),),
        prequant_variant_repos = (("org/model-dev", "int8", "org/dev-fp8"),),
    )
    assert resolve_prequant_source(fam, "int8").location == "org/default-fp8"
    assert (
        resolve_prequant_source(fam, "int8", base_repo = "org/other-variant").location
        == "org/default-fp8"
    )
    # Scheme still has to match within the variant table.
    assert resolve_prequant_source(fam, "int8", base_repo = "org/model-dev").location == "org/dev-fp8"


def test_flux1_variant_prequant_wiring():
    # The real flux.1 entry serves schnell by default and dev / Krea-dev via variants.
    from core.inference.diffusion_families import detect_family, family_prequant_repo
    fam = detect_family("black-forest-labs/FLUX.1-schnell")
    for scheme in ("int8", "fp8"):
        assert family_prequant_repo(fam, scheme) == "unsloth/FLUX.1-schnell-FP8"
        assert (
            family_prequant_repo(fam, scheme, base_repo = "black-forest-labs/FLUX.1-dev")
            == "unsloth/FLUX.1-dev-FP8"
        )
        assert (
            family_prequant_repo(fam, scheme, base_repo = "black-forest-labs/FLUX.1-Krea-dev")
            == "unsloth/FLUX.1-Krea-dev-FP8"
        )


def test_resolve_wrong_scheme_is_none():
    fam = _fam(prequant_repos = (("fp8", "org/hosted-fp8"),))
    assert resolve_prequant_source(fam, "int8") is None


def test_resolve_nothing_configured_is_none():
    assert resolve_prequant_source(_fam(), "fp8") is None
    assert resolve_prequant_source(_fam(), "fp8", path_override = "") is None


def test_local_prequant_path_ready(tmp_path, monkeypatch):
    # The auto-policy planner budgets the small prequant plan only when a request-supplied path would actually load: present
    # AND inside an allowlisted root. Otherwise the loader refuses it and rebuilds dense after evicting.
    import os

    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"x")
    root = os.path.realpath(str(tmp_path))
    monkeypatch.setattr(pq, "_allowed_prequant_roots", lambda: [root])
    assert pq.local_prequant_path_ready(str(ckpt)) is True
    assert pq.local_prequant_path_ready(str(tmp_path / "missing.pt")) is False
    monkeypatch.setattr(pq, "_allowed_prequant_roots", lambda: [])
    assert pq.local_prequant_path_ready(str(ckpt)) is False


# ── usable_prequant_source ───────────────────────────────────────────────────────
def test_usable_source_missing_path_is_none(tmp_path, monkeypatch):
    # An allowlisted but ABSENT path is not a prequant source: load_prequantized_transformer would find no file and fall back
    # to the dense bf16 build after evicting, so the planner must run the dense fit checks up front.
    import os

    monkeypatch.setattr(pq, "_allowed_prequant_roots", lambda: [os.path.realpath(str(tmp_path))])
    fam = _fam(prequant_repos = (("fp8", "org/hosted-fp8"),))
    missing = str(tmp_path / "missing.pt")
    assert pq.usable_prequant_source(fam, "fp8", path_override = missing) is None


def test_usable_source_disallowed_path_is_none(tmp_path, monkeypatch):
    # A path OUTSIDE the UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH allowlist (including the empty default) is refused by the loader, so it resolves to None even when it exists.
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"x")
    monkeypatch.setattr(pq, "_allowed_prequant_roots", lambda: [])
    fam = _fam(prequant_repos = (("fp8", "org/hosted-fp8"),))
    assert pq.usable_prequant_source(fam, "fp8", path_override = str(ckpt)) is None


def test_usable_source_allowed_present_path_wins(tmp_path, monkeypatch):
    # Allowlisted AND present: the override is usable and beats the hosted repo, exactly like resolve_prequant_source.
    import os

    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"x")
    monkeypatch.setattr(pq, "_allowed_prequant_roots", lambda: [os.path.realpath(str(tmp_path))])
    fam = _fam(prequant_repos = (("fp8", "org/hosted-fp8"),))
    src = pq.usable_prequant_source(fam, "fp8", path_override = str(ckpt))
    assert src == PrequantSource(kind = "path", location = str(ckpt), filename = None)


def test_usable_source_repo_unaffected_by_allowlist(monkeypatch):
    # Hosted-repo sources are first-party and keep resolving with no allowlist at all.
    monkeypatch.setattr(pq, "_allowed_prequant_roots", lambda: [])
    fam = _fam(prequant_repos = (("fp8", "org/hosted-fp8"),))
    src = pq.usable_prequant_source(fam, "fp8")
    assert src is not None and src.kind == "repo" and src.location == "org/hosted-fp8"


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
    # The local-path branch is opt-in via a directory ALLOWLIST (it unpickles an arbitrary file); allowlist tmp_path unless a test checks the gate.
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
    # Built via from_config, so the loader must eval() it like the dense/GGUF paths or train-mode dropout makes inference nondeterministic.
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
    # A pre-fix fp8 checkpoint has no fp8_granularity (old per-tensor layout), so it must be rejected and rebuilt rather than reproduce the noise failure.
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
    # A checkpoint whose keys match a different base loads strict=True and renders from the wrong weights, so one requested with a base but recording none is refused.
    ckpt = _good_ckpt()
    del ckpt["metadata"]["base_model_id"]
    assert _load(monkeypatch, tmp_path, ckpt) is None


def test_load_fast_accum_mismatch_is_none(monkeypatch, tmp_path):
    # fp8 fast-accum is baked into the saved kernels, so a request contradicting the recorded value falls to the dense path, which honors it.
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
    # An int8 checkpoint recording a stale exclusion set would bake M=1 modulation linears as int8 and crash, so it is rejected.
    ckpt = _good_ckpt(scheme = "int8")
    ckpt["metadata"]["exclude_name_tokens"] = ["stale_token"]
    assert _load(monkeypatch, tmp_path, ckpt, scheme = "int8") is None


def test_load_exclude_tokens_match_ok(monkeypatch, tmp_path):
    from core.inference.diffusion_transformer_quant import exclude_tokens_for_scheme

    ckpt = _good_ckpt(scheme = "int8")
    ckpt["metadata"]["exclude_name_tokens"] = list(exclude_tokens_for_scheme("int8"))
    assert _load(monkeypatch, tmp_path, ckpt, scheme = "int8") is not None


def test_load_exclude_tokens_need_the_recorded_family(monkeypatch, tmp_path):
    # int8 carries PER-FAMILY exclusions (Qwen's unpadded text stream runs at M = prompt tokens, under _int_mm's floor of 16), so an artifact
    # recording the family but building its set with family=None is rejected. Pins the offline builder to exclude_tokens_for_scheme(scheme, fam.name).
    from core.inference.diffusion_transformer_quant import exclude_tokens_for_scheme
    for family in ("qwen-image", "qwen-image-edit"):
        family_less = _good_ckpt(scheme = "int8")
        family_less["metadata"]["family"] = family
        family_less["metadata"]["exclude_name_tokens"] = list(exclude_tokens_for_scheme("int8"))
        assert _load(monkeypatch, tmp_path, family_less, scheme = "int8") is None

        family_aware = _good_ckpt(scheme = "int8")
        family_aware["metadata"]["family"] = family
        family_aware["metadata"]["exclude_name_tokens"] = list(
            exclude_tokens_for_scheme("int8", family)
        )
        assert _load(monkeypatch, tmp_path, family_aware, scheme = "int8") is not None


def test_load_require_bf16_mismatch_is_none(monkeypatch, tmp_path):
    # An fp8 (scaled_mm) checkpoint built WITHOUT the bf16 gate quantised a different layer set than the runtime filter produces, so it is rejected.
    ckpt = _good_ckpt(scheme = "fp8")
    ckpt["metadata"]["require_bf16"] = False
    assert _load(monkeypatch, tmp_path, ckpt, scheme = "fp8") is None


def test_load_require_bf16_match_ok(monkeypatch, tmp_path):
    ckpt = _good_ckpt(scheme = "fp8")
    ckpt["metadata"]["require_bf16"] = True
    assert _load(monkeypatch, tmp_path, ckpt, scheme = "fp8") is not None


def test_load_require_bf16_int8_true_is_none(monkeypatch, tmp_path):
    # int8 (torch._int_mm) tolerates non-bf16 weights so it never sets the gate; a checkpoint claiming it did is rejected.
    ckpt = _good_ckpt(scheme = "int8")
    ckpt["metadata"]["require_bf16"] = True
    assert _load(monkeypatch, tmp_path, ckpt, scheme = "int8") is None


def test_load_require_bf16_nvfp4_false_ok(monkeypatch, tmp_path):
    # nvfp4 quantises fp32 fine, so the runtime filter leaves the bf16 gate off and a checkpoint built the same way matches.
    ckpt = _good_ckpt(scheme = "nvfp4")
    ckpt["metadata"]["require_bf16"] = False
    assert _load(monkeypatch, tmp_path, ckpt, scheme = "nvfp4") is not None


def test_load_require_bf16_nvfp4_true_is_none(monkeypatch, tmp_path):
    # An nvfp4 checkpoint claiming the bf16 gate quantised a different layer set, so it is rejected.
    ckpt = _good_ckpt(scheme = "nvfp4")
    ckpt["metadata"]["require_bf16"] = True
    assert _load(monkeypatch, tmp_path, ckpt, scheme = "nvfp4") is None


def test_resolve_checkpoint_path_expands_user(monkeypatch, tmp_path):
    # The allowlist gate expands ~, so the existence check must too, or a "~/..." checkpoint that passed the gate is silently skipped.
    import os

    real = tmp_path / "transformer_fp8.pt"
    real.write_bytes(b"x")
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(real) if p == "~/ckpt.pt" else p)
    source = PrequantSource(kind = "path", location = "~/ckpt.pt", filename = None)
    assert pq._resolve_checkpoint_path(source, None) == str(real)


# ── local-path opt-in gate (RCE guard) ───────────────────────────────────────────
def test_load_local_path_refused_by_default(monkeypatch, tmp_path):
    # Even a valid checkpoint is refused: torch.load must never run on a request-supplied path without the operator opt-in.
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
    roots: list = []

    def _dl(
        repo_id,
        filename,
        token = None,
        cache_dir = None,
    ):
        roots.append(cache_dir)
        return str(downloaded)

    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = _dl
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
        cache_dir = "/live-hub",
        logger = None,
    )
    assert result is not None
    # The loader pins the caller's live root, so the fetch cannot split across two.
    assert roots == ["/live-hub"]


def test_load_repo_source_falls_back_to_legacy_filename(monkeypatch, tmp_path):
    # A repo still carrying the legacy transformer_<scheme>.pt name serves the download after the model-name filename 404s; both are requested in order.
    _FakeTransformer.calls = {}
    _stub_torch_accelerate(monkeypatch, _good_ckpt())
    monkeypatch.delenv(pq.ALLOW_LOCAL_PREQUANT_PATH_ENV, raising = False)

    downloaded = tmp_path / "transformer_fp8.pt"
    downloaded.write_bytes(b"x")

    class _NotFound(Exception):
        pass

    errors = types.ModuleType("huggingface_hub.errors")
    errors.EntryNotFoundError = _NotFound
    requested = []

    def _dl(
        repo_id,
        filename,
        token = None,
        cache_dir = None,
    ):
        requested.append(filename)
        if filename != "transformer_fp8.pt":
            raise _NotFound(filename)
        return str(downloaded)

    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = _dl
    hub.errors = errors
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)

    source = PrequantSource(
        kind = "repo",
        location = "org/Z-Image-Turbo-FP8",
        filename = "Z-Image-Turbo-FP8.pt",
        fallback_filename = "transformer_fp8.pt",
    )
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
    assert requested == ["Z-Image-Turbo-FP8.pt", "transformer_fp8.pt"]


def test_load_local_path_outside_allowlist_refused(monkeypatch, tmp_path):
    # Even with the opt-in set, a path outside every allowlisted directory must not be unpickled: one trusted dir is not a wildcard.
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
    # A checkpoint built with a different --min-features quantises a different Linear set, so it is rejected when the runtime threshold is supplied.
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


# ── kernel preference ────────────────────────────────────────────────────────


class _FakeKernelPreference:
    """Stand-in for torchao's enum: only identity and attribute access matter here."""

    TORCH = "KernelPreference.TORCH"
    AUTO = "KernelPreference.AUTO"


def _stub_kernel_preference(monkeypatch):
    mod = types.ModuleType("torchao.quantization.quantize_.common.kernel_preference")
    mod.KernelPreference = _FakeKernelPreference
    for name in (
        "torchao",
        "torchao.quantization",
        "torchao.quantization.quantize_",
        "torchao.quantization.quantize_.common",
    ):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "torchao.quantization.quantize_.common.kernel_preference", mod)


class _FakeFp8Weight:
    def __init__(self, pref):
        self.kernel_preference = pref


def test_pin_kernel_preference_rewrites_auto(monkeypatch):
    # The published checkpoints serialize KernelPreference.AUTO on every fp8 weight, re-arming the MSLK
    # kernel _fp8_config pins away from; mslk.f8f8bf16_rowwise has no fake impl, so a COMPILED generate
    # then fails to trace. Loading must normalise it.
    _stub_kernel_preference(monkeypatch)
    sd = {
        "a.weight": _FakeFp8Weight(_FakeKernelPreference.AUTO),
        "b.weight": _FakeFp8Weight(_FakeKernelPreference.AUTO),
        "c.weight": _FakeFp8Weight(_FakeKernelPreference.TORCH),  # already pinned
        "d.bias": object(),  # plain tensor: no preference, untouched
    }
    pinned = pq._pin_kernel_preference(sd, logger = None)
    assert pinned == 2
    assert all(
        getattr(t, "kernel_preference", _FakeKernelPreference.TORCH) == _FakeKernelPreference.TORCH
        for t in sd.values()
    )


def test_pin_kernel_preference_survives_a_frozen_weight(monkeypatch):
    # A subclass that refuses the assignment must not sink the load: the checkpoint is still usable eagerly, and raising here would lose it entirely.
    _stub_kernel_preference(monkeypatch)

    class _Frozen:
        kernel_preference = _FakeKernelPreference.AUTO

        def __setattr__(self, name, value):
            raise AttributeError("frozen")

    sd = {"a.weight": _Frozen(), "b.weight": _FakeFp8Weight(_FakeKernelPreference.AUTO)}
    assert pq._pin_kernel_preference(sd, logger = None) == 1


def test_pin_kernel_preference_no_torchao(monkeypatch):
    # Without the enum there is nothing to pin to; leave the checkpoint exactly as saved.
    monkeypatch.setitem(
        sys.modules, "torchao.quantization.quantize_.common.kernel_preference", None
    )
    sd = {"a.weight": _FakeFp8Weight(_FakeKernelPreference.AUTO)}
    assert pq._pin_kernel_preference(sd, logger = None) == 0
    assert sd["a.weight"].kernel_preference == _FakeKernelPreference.AUTO


# ── local cache lookup (no network) ──────────────────────────────────────────────
def test_prequant_checkpoint_cached_reads_only_the_cache(monkeypatch, tmp_path):
    # Memory planning asks this, so it must be a pure lookup: no Hub call, no raise.
    from core.inference.diffusion_prequant import prequant_checkpoint_cached

    ckpt = tmp_path / "Z-Image-Turbo-FP8.pt"
    ckpt.write_bytes(b"weights")
    legacy = tmp_path / "transformer_fp8.pt"
    legacy.write_bytes(b"weights")
    source = PrequantSource(
        kind = "repo",
        location = "unsloth/Z-Image-Turbo-FP8",
        filename = "Z-Image-Turbo-FP8.pt",
        fallback_filename = "transformer_fp8.pt",
    )
    asked: list = []

    def _cache(
        repo_id,
        filename,
        cache_dir = None,
    ):
        asked.append((repo_id, filename, cache_dir))
        return str(tmp_path / filename) if (tmp_path / filename).is_file() else None

    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", _cache)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda *a, **k: pytest.fail("the cache probe must never download"),
    )

    assert prequant_checkpoint_cached(source, cache_dir = "/models/hub") is True
    # The live root is asked first, and the model-name file resolves, so no legacy lookup.
    assert asked == [("unsloth/Z-Image-Turbo-FP8", "Z-Image-Turbo-FP8.pt", "/models/hub")]

    # Only the legacy name on disk does NOT count: whether the repo publishes the canonical one
    # needs a network call, so this reads as "would download" and the GGUF runs.
    ckpt.unlink()
    assert prequant_checkpoint_cached(source) is False
    # Neither name cached -> same answer, for the ordinary reason.
    legacy.unlink()
    assert prequant_checkpoint_cached(source) is False


def test_a_live_root_hit_still_goes_through_the_hub_so_it_revalidates(monkeypatch, tmp_path):
    # hf_hub_download(cache_dir = live) reuses that root's blob AND revalidates, so a hit there
    # must not be short-circuited: returning it raw would pin a stale checkpoint past a republish.
    live = tmp_path / "live-hub"
    live.mkdir()
    ckpt = live / "Z-Image-Turbo-FP8.pt"
    ckpt.write_bytes(b"weights")
    source = PrequantSource(
        kind = "repo",
        location = "unsloth/Z-Image-Turbo-FP8",
        filename = "Z-Image-Turbo-FP8.pt",
        fallback_filename = "transformer_fp8.pt",
    )

    def _cache(
        repo_id,
        filename,
        cache_dir = None,
    ):
        path = live / filename
        return str(path) if cache_dir == str(live) and path.is_file() else None

    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", _cache)
    asked: list = []

    def _download(
        repo_id,
        filename,
        token = None,
        cache_dir = None,
    ):
        asked.append((filename, cache_dir))
        return str(ckpt)  # the same blob, revalidated

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _download)

    # Still "cached" for the decline gate: this costs a HEAD, not a multi-GB fetch.
    assert pq.prequant_checkpoint_cached(source, cache_dir = str(live)) is True
    assert pq._resolve_checkpoint_path(source, None, str(live)) == str(ckpt)
    assert asked == [("Z-Image-Turbo-FP8.pt", str(live))]  # pinned to the root holding the blob


def _other_root_source():
    return PrequantSource(
        kind = "repo",
        location = "unsloth/Z-Image-Turbo-FP8",
        filename = "Z-Image-Turbo-FP8.pt",
        fallback_filename = "transformer_fp8.pt",
    )


def test_a_hit_only_in_the_other_root_is_revalidated_through_that_root(monkeypatch, tmp_path):
    # hf_hub_download(cache_dir = live) would not look in huggingface_hub's import-time root and
    # would re-fetch multiple GB, while returning the cached path raw would pin it past a
    # republish. So the download is re-run THROUGH the root that holds the copy: one HEAD.
    default_root = tmp_path / "default-hub"
    default_root.mkdir()
    ckpt = default_root / "Z-Image-Turbo-FP8.pt"
    ckpt.write_bytes(b"weights")
    source = _other_root_source()

    def _cache(
        repo_id,
        filename,
        cache_dir = None,
    ):
        # Only the import-time default (cache_dir None) holds it; the live root is a miss.
        path = default_root / filename
        return str(path) if cache_dir is None and path.is_file() else None

    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", _cache)
    asked: list = []

    def _download(
        repo_id,
        filename,
        token = None,
        cache_dir = None,
    ):
        asked.append((filename, cache_dir))
        return str(ckpt)  # unchanged upstream: the same blob, revalidated

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _download)

    assert pq.prequant_checkpoint_cached(source, cache_dir = "/models/live") is True
    assert pq._resolve_checkpoint_path(source, None, "/models/live") == str(ckpt)
    # cache_dir None, never the live root: the live root is empty, so pinning it there would be
    # the multi-GB re-fetch the caller declined on.
    assert asked == [("Z-Image-Turbo-FP8.pt", None)]


def test_a_republished_checkpoint_in_the_other_root_is_picked_up(monkeypatch, tmp_path):
    # Revalidation earns its HEAD here: the repo replaced the file under the same name, so the
    # cached pointer is stale and the loader must follow the new snapshot instead.
    default_root = tmp_path / "default-hub"
    default_root.mkdir()
    stale = default_root / "Z-Image-Turbo-FP8.pt"
    stale.write_bytes(b"old weights")
    fresh = default_root / "snapshots-new" / "Z-Image-Turbo-FP8.pt"
    fresh.parent.mkdir()
    fresh.write_bytes(b"corrected weights")

    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache",
        lambda repo_id, filename, cache_dir = None, **k: (
            str(stale) if cache_dir is None and filename == "Z-Image-Turbo-FP8.pt" else None
        ),
    )
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kwargs: str(fresh))

    assert pq._resolve_checkpoint_path(_other_root_source(), None, "/models/live") == str(fresh)


def test_other_root_revalidation_never_breaks_a_load_that_works(monkeypatch, tmp_path):
    # Offline hf_hub_download already serves the cached pointer, but anything else it raises (a hub
    # layout change, a read-only root) must fall back to the copy already located.
    default_root = tmp_path / "default-hub"
    default_root.mkdir()
    ckpt = default_root / "Z-Image-Turbo-FP8.pt"
    ckpt.write_bytes(b"weights")

    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache",
        lambda repo_id, filename, cache_dir = None, **k: (
            str(default_root / filename)
            if cache_dir is None and (default_root / filename).is_file()
            else None
        ),
    )
    asked: list = []

    def _download(**kwargs):
        asked.append(kwargs["cache_dir"])
        raise RuntimeError("hub is unhappy")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _download)

    assert pq._resolve_checkpoint_path(_other_root_source(), None, "/models/live") == str(ckpt)
    assert asked == [None]  # it was tried, through the root holding the copy, and forgiven


def test_an_uncached_checkpoint_downloads_into_the_live_root(monkeypatch):
    # The other half: a real fetch must land where Studio is reading, not under the stale constant.
    asked: list = []
    source = PrequantSource(
        kind = "repo", location = "unsloth/Z-Image-Turbo-FP8", filename = "Z-Image-Turbo-FP8.pt"
    )

    def _dl(**kwargs):
        asked.append(kwargs)
        return "/live-hub/blobs/abc"

    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _dl)

    assert pq._resolve_checkpoint_path(source, "tok", "/live-hub") == "/live-hub/blobs/abc"
    assert asked == [
        {
            "repo_id": "unsloth/Z-Image-Turbo-FP8",
            "filename": "Z-Image-Turbo-FP8.pt",
            "token": "tok",
            "cache_dir": "/live-hub",
        }
    ]


def test_prequant_checkpoint_cached_never_raises(monkeypatch):
    # An unanswerable probe is "not cached", never an exception into the memory planner.
    from core.inference.diffusion_prequant import prequant_checkpoint_cached

    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache",
        lambda *a, **k: (_ for _ in ()).throw(OSError("unreadable cache")),
    )
    source = PrequantSource(kind = "repo", location = "org/hosted-fp8", filename = "hosted-FP8.pt")
    assert prequant_checkpoint_cached(source) is False
    # A local override is the operator's own file, so it is not a cache question.
    assert prequant_checkpoint_cached(PrequantSource(kind = "path", location = "/tmp/x.pt")) is False
    assert prequant_checkpoint_cached(None) is False


def test_a_cached_legacy_file_does_not_pre_empt_the_canonical_one(monkeypatch, tmp_path):
    """fallback_filename is primary-first by contract, reached only once the canonical name is
    absent remotely. Short-circuiting on a cached legacy artifact would pin a stale
    transformer_<scheme>.pt forever, even after the repo publishes the real name."""
    from core.inference.diffusion_prequant import _resolve_checkpoint_path

    legacy = tmp_path / "transformer_fp8.pt"
    legacy.write_bytes(b"stale")
    source = PrequantSource(
        kind = "repo",
        location = "unsloth/Z-Image-Turbo-FP8",
        filename = "Z-Image-Turbo-FP8.pt",
        fallback_filename = "transformer_fp8.pt",
    )
    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache",
        lambda repo_id, filename, cache_dir = None: (
            str(tmp_path / filename) if (tmp_path / filename).is_file() else None
        ),
    )
    asked: list = []

    def _download(
        repo_id,
        filename,
        token = None,
        cache_dir = None,
    ):
        asked.append(filename)
        return str(tmp_path / "downloaded-canonical.pt")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _download)

    assert _resolve_checkpoint_path(source, None) == str(tmp_path / "downloaded-canonical.pt")
    assert asked == ["Z-Image-Turbo-FP8.pt"]  # the canonical name, not the cached legacy one


def test_the_legacy_name_is_still_used_once_the_canonical_one_is_absent(monkeypatch, tmp_path):
    """The other direction, which is what fallback_filename exists for: a repo that only ever
    shipped the legacy artifact must still load."""
    from huggingface_hub.errors import EntryNotFoundError

    from core.inference.diffusion_prequant import _resolve_checkpoint_path

    source = PrequantSource(
        kind = "repo",
        location = "unsloth/Z-Image-Turbo-FP8",
        filename = "Z-Image-Turbo-FP8.pt",
        fallback_filename = "transformer_fp8.pt",
    )
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    asked: list = []

    def _download(
        repo_id,
        filename,
        token = None,
        cache_dir = None,
    ):
        asked.append(filename)
        if filename == "Z-Image-Turbo-FP8.pt":
            raise EntryNotFoundError("404")
        return str(tmp_path / filename)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _download)

    assert _resolve_checkpoint_path(source, None) == str(tmp_path / "transformer_fp8.pt")
    assert asked == ["Z-Image-Turbo-FP8.pt", "transformer_fp8.pt"]  # primary first, then legacy


def test_a_legacy_copy_in_the_other_root_is_reused_after_the_primary_404s(monkeypatch, tmp_path):
    """Once the primary is absent remotely the legacy name IS the artifact, so it needs the same
    other-root treatment: revalidate through the root holding the copy rather than re-fetch multiple
    GB into the live one, or pin a stale file by returning it raw."""
    from huggingface_hub.errors import EntryNotFoundError

    default_root = tmp_path / "default-hub"
    default_root.mkdir()
    legacy = default_root / "transformer_fp8.pt"
    legacy.write_bytes(b"weights")
    source = PrequantSource(
        kind = "repo",
        location = "unsloth/Z-Image-Turbo-FP8",
        filename = "Z-Image-Turbo-FP8.pt",
        fallback_filename = "transformer_fp8.pt",
    )

    def _cache(
        repo_id,
        filename,
        cache_dir = None,
    ):
        # Only the import-time default holds the legacy file; the live root holds nothing.
        path = default_root / filename
        return str(path) if cache_dir is None and path.is_file() else None

    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", _cache)
    asked: list = []

    def _download(
        repo_id,
        filename,
        token = None,
        cache_dir = None,
    ):
        asked.append((filename, cache_dir))
        if filename == "Z-Image-Turbo-FP8.pt":
            raise EntryNotFoundError("404")
        return str(legacy)  # unchanged upstream: the same blob, revalidated

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _download)

    assert pq._resolve_checkpoint_path(source, None, "/models/live") == str(legacy)
    # The primary was tried in the live root and 404'd; the legacy was revalidated through the
    # root that actually holds it, never re-fetched into the live one.
    assert asked == [("Z-Image-Turbo-FP8.pt", "/models/live"), ("transformer_fp8.pt", None)]


def test_a_primary_404_during_revalidation_still_reaches_the_legacy_fallback(monkeypatch, tmp_path):
    """The other-root revalidation must not swallow the primary's 404: when the live root misses
    but the import-time root still holds an OLD canonical checkpoint, a blanket catch returns that
    stale file and the fallback-name branch is never reached."""
    from huggingface_hub.errors import EntryNotFoundError

    default_root = tmp_path / "default-hub"
    default_root.mkdir()
    stale = default_root / "Z-Image-Turbo-FP8.pt"
    stale.write_bytes(b"obsolete")
    live = str(tmp_path / "live")

    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache",
        lambda repo_id, filename, cache_dir = None, **k: (
            str(default_root / filename)
            if cache_dir is None and (default_root / filename).is_file()
            else None
        ),
    )
    asked: list = []

    def _download(
        repo_id,
        filename,
        token = None,
        cache_dir = None,
    ):
        asked.append((filename, cache_dir))
        if filename == "Z-Image-Turbo-FP8.pt":
            raise EntryNotFoundError("404: the repo no longer publishes this name")
        return str(tmp_path / "fresh-legacy.pt")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _download)

    assert pq._resolve_checkpoint_path(_other_root_source(), None, live) == str(
        tmp_path / "fresh-legacy.pt"
    )
    # The 404 reached the fallback branch instead of pinning the stale canonical copy.
    assert asked == [("Z-Image-Turbo-FP8.pt", None), ("transformer_fp8.pt", live)]


def test_offline_revalidation_still_returns_the_other_root_copy(monkeypatch, tmp_path):
    """A LOCAL cache miss is not "absent remotely": huggingface_hub raises LocalEntryNotFoundError
    (a subclass of EntryNotFoundError on both majors) when it cannot reach the Hub, so telling it
    apart from a real 404 is what keeps an offline load from dropping the prequant."""
    from huggingface_hub.errors import LocalEntryNotFoundError

    default_root = tmp_path / "default-hub"
    default_root.mkdir()
    ckpt = default_root / "Z-Image-Turbo-FP8.pt"
    ckpt.write_bytes(b"weights")

    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache",
        lambda repo_id, filename, cache_dir = None, **k: (
            str(default_root / filename)
            if cache_dir is None and (default_root / filename).is_file()
            else None
        ),
    )
    asked: list = []

    def _download(
        repo_id,
        filename,
        token = None,
        cache_dir = None,
    ):
        asked.append((filename, cache_dir))
        raise LocalEntryNotFoundError("offline and not in this root")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _download)

    assert pq._resolve_checkpoint_path(_other_root_source(), None, "/models/live") == str(ckpt)
    assert asked == [("Z-Image-Turbo-FP8.pt", None)]  # no doomed fallback fetch


def test_with_no_fallback_name_a_404_keeps_the_other_root_copy(monkeypatch, tmp_path):
    """Propagating the 404 only makes sense while another filename is left to try. With none, the
    copy already located is still the best answer, so revalidation stays a bonus."""
    from huggingface_hub.errors import EntryNotFoundError

    default_root = tmp_path / "default-hub"
    default_root.mkdir()
    ckpt = default_root / "X-FP8.pt"
    ckpt.write_bytes(b"weights")
    source = PrequantSource(kind = "repo", location = "unsloth/X-FP8", filename = "X-FP8.pt")

    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache",
        lambda repo_id, filename, cache_dir = None, **k: (
            str(default_root / filename)
            if cache_dir is None and (default_root / filename).is_file()
            else None
        ),
    )
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda **k: (_ for _ in ()).throw(EntryNotFoundError("404")),
    )

    assert pq._resolve_checkpoint_path(source, None, "/models/live") == str(ckpt)


def test_load_config_reads_the_same_cache_root_as_the_checkpoint(monkeypatch, tmp_path):
    """load_config forwards cache_dir to hf_hub_download, so unpinned it reads huggingface_hub's
    import-time constant. After a mid-session cache change that root may be gone or read-only, and
    the raise is swallowed into a None return: the prequant is silently dropped."""
    import torch

    from core.inference import diffusion_prequant as P

    seen: list = []
    live_root = tmp_path / "live"
    live_root.mkdir()
    live = str(live_root)
    ckpt = live_root / "ck.pt"  # the checkpoint came from the LIVE root
    ckpt.write_bytes(b"x")

    monkeypatch.setattr(P, "_resolve_checkpoint_path", lambda s, t, c = None: str(ckpt))
    monkeypatch.setattr(P, "_validate_checkpoint", lambda *a, **k: True)
    monkeypatch.setattr(P, "_pin_kernel_preference", lambda *a, **k: 0)
    monkeypatch.setattr(torch, "load", lambda *a, **k: {"state_dict": {}, "scheme": "int8"})

    class _Cls:
        @staticmethod
        def load_config(
            base,
            subfolder = None,
            token = None,
            cache_dir = None,
        ):
            seen.append(cache_dir)
            raise RuntimeError("stop right after the config fetch")

    src = P.PrequantSource(kind = "repo", location = "unsloth/X-FP8", filename = "X-FP8.pt")
    assert (
        P.load_prequantized_transformer(
            _Cls,
            "org/base",
            src,
            device = "cpu",
            dtype = torch.bfloat16,
            scheme = "int8",
            cache_dir = live,
        )
        is None
    )

    assert seen[0] == live  # the checkpoint's own root is read first


def test_the_config_follows_the_checkpoint_into_the_other_cache_root(monkeypatch, tmp_path):
    """``_resolve_checkpoint_path`` can answer from huggingface_hub's import-time root while Studio
    pins its live one, so a config pinned to the live root misses in exactly the cache-moved case
    the checkpoint lookup just accepted -- silently, as the raise becomes a None return."""
    import torch

    from core.inference import diffusion_prequant as P

    seen: list = []
    live = str(tmp_path / "live")  # the moved-to root: nothing is cached under it yet
    other_root = tmp_path / "default-hub"
    other_root.mkdir()
    ckpt = other_root / "ck.pt"
    ckpt.write_bytes(b"x")

    monkeypatch.setattr(P, "_resolve_checkpoint_path", lambda s, t, c = None: str(ckpt))
    monkeypatch.setattr(P, "_validate_checkpoint", lambda *a, **k: True)
    monkeypatch.setattr(P, "_pin_kernel_preference", lambda *a, **k: 0)
    monkeypatch.setattr(torch, "load", lambda *a, **k: {"state_dict": {}, "scheme": "int8"})
    monkeypatch.setattr(P, "_has_meta_tensors", lambda *a, **k: False)
    accelerate = types.ModuleType("accelerate")
    accelerate.init_empty_weights = lambda: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "accelerate", accelerate)

    class _Transformer:
        def load_state_dict(self, *a, **k):
            return None

        def to(self, device):
            return self

        def eval(self):
            return self

    class _Cls:
        @staticmethod
        def load_config(
            base,
            subfolder = None,
            token = None,
            cache_dir = None,
        ):
            seen.append(cache_dir)
            if cache_dir is not None:
                raise OSError("the live root has no cached config for this base")
            return {"ok": True}

        @staticmethod
        def from_config(config):
            return _Transformer()

    src = P.PrequantSource(kind = "repo", location = "unsloth/X-FP8", filename = "X-FP8.pt")
    out = P.load_prequantized_transformer(
        _Cls,
        "org/base",
        src,
        device = "cpu",
        dtype = torch.bfloat16,
        scheme = "int8",
        cache_dir = live,
    )

    assert isinstance(out, _Transformer)  # the prequant loaded instead of being dropped
    assert seen == [None]  # read straight through the root that supplied the checkpoint


# ── fp8 activation scale floor ──────────────────────────────────────────────────


class _FakeFp8Tensor:
    """Stands in for a torchao Float8Tensor: only act_quant_kwargs.hp_value_lb is read."""

    def __init__(self, hp_value_lb):
        self.act_quant_kwargs = types.SimpleNamespace(hp_value_lb = hp_value_lb)


def test_an_fp8_checkpoint_without_the_activation_floor_is_rejected():
    # A checkpoint built before activation_value_lb bakes hp_value_lb=None into every quantised
    # tensor, and stays broken however it is loaded: torchao's per-row activation quantiser divides
    # by the row amax, so qwen's all-zero text rows give scale 0 and NaN. The metadata checks around
    # this one all accept an absent field for back-compat, which is exactly wrong here, so the floor
    # is read off the TENSORS instead. Measured: 412 of 512 rows non-finite without it, 0 with it.
    floored = {"blocks.0.attn.to_q.weight": _FakeFp8Tensor(1e-12)}
    unfloored = {"blocks.0.attn.to_q.weight": _FakeFp8Tensor(None)}
    assert pq._fp8_activation_floor_present(floored, None) is True
    assert pq._fp8_activation_floor_present(unfloored, None) is False
    # Zero is not a floor either: it is what an unclamped amax divide produces.
    assert pq._fp8_activation_floor_present({"w": _FakeFp8Tensor(0.0)}, None) is False


def test_the_floor_check_ignores_dense_and_unreadable_state_dicts():
    # A dense tensor carries no act_quant_kwargs, so it is not evidence of a missing floor; the
    # scheme / granularity checks own that case. Same for a state dict this cannot walk at all:
    # failing closed here would reject every int8 checkpoint too.
    assert pq._fp8_activation_floor_present({"w": object()}, None) is True
    assert pq._fp8_activation_floor_present(None, None) is True
    assert pq._fp8_activation_floor_present({}, None) is True
