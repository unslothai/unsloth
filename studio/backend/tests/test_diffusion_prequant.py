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
import dataclasses
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


@pytest.fixture(autouse = True)
def _pin_prequant_safe_globals(real_prequant_safe_globals):
    """Apply the shared stand-in allowlist (see conftest) to every test in this module."""
    return real_prequant_safe_globals


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


def test_resolve_prefers_a_family_declared_filename():
    # A family may host a SECOND artifact for the same repo and scheme. Naming it makes it the
    # primary and demotes the derived name to the fallback, so a build that knows the new name
    # gets it while an older one still resolves the artifact it already understands.
    fam = _fam(prequant_repos = (("int8", "unsloth/Model-FP8"),))
    fam = dataclasses.replace(fam, prequant_filenames = (("int8", "Model-INT8-ConvRot.pt"),))
    src = resolve_prequant_source(fam, "int8")
    assert src.filename == "Model-INT8-ConvRot.pt"
    assert src.fallback_filename == "Model-INT8.pt"
    # Only for the scheme that declares one; everything else keeps today's derived/legacy pair.
    other = resolve_prequant_source(
        dataclasses.replace(fam, prequant_repos = (("fp8", "unsloth/Model-FP8"),)), "fp8"
    )
    assert other.filename == "Model-FP8.pt"
    assert other.fallback_filename == "transformer_fp8.pt"


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
@pytest.fixture
def restricted_load_available(monkeypatch):
    """Whether this install could open a checkpoint depends on the host's torchao. The resolution
    tests below are not about that, so pin it on."""
    monkeypatch.setattr(pq, "restricted_prequant_load_supported", lambda scheme = None: True)


def test_usable_source_missing_path_is_none(tmp_path, monkeypatch, restricted_load_available):
    # An allowlisted but ABSENT path is not a prequant source: load_prequantized_transformer would find no file and fall back
    # to the dense bf16 build after evicting, so the planner must run the dense fit checks up front.
    import os

    monkeypatch.setattr(pq, "_allowed_prequant_roots", lambda: [os.path.realpath(str(tmp_path))])
    fam = _fam(prequant_repos = (("fp8", "org/hosted-fp8"),))
    missing = str(tmp_path / "missing.pt")
    assert pq.usable_prequant_source(fam, "fp8", path_override = missing) is None


def test_usable_source_disallowed_path_is_none(tmp_path, monkeypatch, restricted_load_available):
    # A path OUTSIDE the UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH allowlist (including the empty default) is refused by the loader, so it resolves to None even when it exists.
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"x")
    monkeypatch.setattr(pq, "_allowed_prequant_roots", lambda: [])
    fam = _fam(prequant_repos = (("fp8", "org/hosted-fp8"),))
    assert pq.usable_prequant_source(fam, "fp8", path_override = str(ckpt)) is None


def test_usable_source_allowed_present_path_wins(tmp_path, monkeypatch, restricted_load_available):
    # Allowlisted, present AND baked for this scheme: the override is usable and beats the hosted repo, exactly like resolve_prequant_source.
    import os

    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"x")
    monkeypatch.setattr(pq, "_allowed_prequant_roots", lambda: [os.path.realpath(str(tmp_path))])
    monkeypatch.setattr(pq, "local_prequant_scheme", lambda _p: "fp8")
    fam = _fam(prequant_repos = (("fp8", "org/hosted-fp8"),))
    src = pq.usable_prequant_source(fam, "fp8", path_override = str(ckpt))
    assert src == PrequantSource(kind = "path", location = str(ckpt), filename = None)


def test_usable_source_rejects_an_override_baked_for_another_scheme(
    tmp_path, monkeypatch, restricted_load_available
):
    """An int8 checkpoint must not read as an available fp8 pre-quant.

    resolve_prequant_source hands back a path source for ANY override without inspecting the file,
    so under `auto` (which picks a scheme the user never named) planning would skip staging the
    dense transformer, the loader would hit the same metadata.scheme check it runs at load time,
    refuse the file, and with no dense fallback the pick silently drops to GGUF.
    """
    import os

    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"x")
    monkeypatch.setattr(pq, "_allowed_prequant_roots", lambda: [os.path.realpath(str(tmp_path))])
    monkeypatch.setattr(pq, "local_prequant_scheme", lambda _p: "int8")
    fam = _fam(prequant_repos = (("fp8", "org/hosted-fp8"),))
    assert pq.usable_prequant_source(fam, "fp8", path_override = str(ckpt)) is None
    # The same file IS usable for the scheme it was actually baked for.
    src = pq.usable_prequant_source(fam, "int8", path_override = str(ckpt))
    assert src == PrequantSource(kind = "path", location = str(ckpt), filename = None)


def test_an_unreadable_override_is_not_usable(tmp_path, monkeypatch, restricted_load_available):
    # A file we cannot parse as a pre-quant checkpoint is "unknown", and the loader would reject it
    # too, so planning must budget dense rather than assume a shortcut it will not get.
    import os

    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"not a checkpoint")
    monkeypatch.setattr(pq, "_allowed_prequant_roots", lambda: [os.path.realpath(str(tmp_path))])
    fam = _fam(prequant_repos = (("fp8", "org/hosted-fp8"),))
    assert pq.local_prequant_scheme(str(ckpt)) is None
    assert pq.usable_prequant_source(fam, "fp8", path_override = str(ckpt)) is None


def test_the_local_scheme_cache_survives_a_same_second_swap(tmp_path):
    """Two checkpoints of one model differ only in scheme, so an atomic swap is same-size.

    The memo key used int(st_mtime), which truncates to seconds: replacing an int8 override with
    the fp8 bake of the same model inside one second left the key unchanged, so every later probe
    in that process reported the OLD scheme. Under `auto` that is the exact failure the scheme
    check exists to stop, only inverted: planning trusts a scheme the file no longer records.
    """
    import os

    import torch

    ckpt = tmp_path / "model.pt"

    def _write(scheme, pad):
        torch.save(
            {"format": pq.PREQUANT_FORMAT, "metadata": {"scheme": scheme, "pad": "x" * pad}},
            str(ckpt),
        )

    _write("int8", 8)
    stamp = os.stat(str(ckpt)).st_mtime_ns
    size = os.stat(str(ckpt)).st_size
    assert pq.local_prequant_scheme(str(ckpt)) == "int8"

    # Same size, and a timestamp inside the same second as the first write.
    _write("fp8", 9)
    assert os.stat(str(ckpt)).st_size == size, "the two bakes must be same-size for this to bite"
    os.utime(str(ckpt), ns = (stamp + 1, stamp + 1))
    assert int(os.stat(str(ckpt)).st_mtime) == int(stamp / 1e9)

    assert pq.local_prequant_scheme(str(ckpt)) == "fp8"


def test_usable_source_repo_unaffected_by_allowlist(monkeypatch, restricted_load_available):
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

    def named_modules(self):
        # Real nn.Modules have this, and the small-M padding pass walks it.
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
    # Registration is version-gated (2.6+), so the stub needs a version or every load declines.
    torch.__version__ = "2.9.1+cu128"
    seen = {"weights_only": None, "safe_globals": None}

    def _load(
        path,
        weights_only = False,
        map_location = None,
        **kwargs,
    ):
        seen["weights_only"] = weights_only
        if load_raises:
            raise RuntimeError("corrupt checkpoint")
        return ckpt

    def _add_safe_globals(entries):
        seen["safe_globals"] = list(entries)

    torch.load = _load
    # A stub without this namespace would let a regression to an unrestricted load pass silently.
    torch.serialization = types.SimpleNamespace(add_safe_globals = _add_safe_globals)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.serialization", torch.serialization)

    accelerate = types.ModuleType("accelerate")
    accelerate.init_empty_weights = lambda: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "accelerate", accelerate)
    return seen


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
    # The local-path branch is opt-in via a directory ALLOWLIST (it loads arbitrary weights); allowlist tmp_path unless a test checks the gate.
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


# ── deserialization gate (RCE guard) ─────────────────────────────────────────────
def test_the_checkpoint_is_deserialized_under_an_allowlist(monkeypatch):
    """A pre-quant checkpoint is a pickle, so the load has to be ``weights_only``.

    It is a mutable remote file reached WITHOUT anyone asking for it (auto resolves an unset
    precision to a hosted checkpoint), so "the repo is first-party" is not a reason to run
    whatever bytes arrive. Everything the format needs is allowlisted instead."""
    seen = _stub_torch_accelerate(monkeypatch, _good_ckpt())
    monkeypatch.setattr(pq, "_SAFE_GLOBALS_REGISTERED", None)
    monkeypatch.setattr(pq, "_resolve_checkpoint_path", lambda *a, **k: "/cache/x.pt")
    load_prequantized_transformer(
        _FakeTransformer,
        "Tongyi-MAI/Z-Image-Turbo",
        PrequantSource(kind = "repo", location = "org/hosted-fp8", filename = "x.pt"),
        device = "cpu",
        dtype = "bf16",
        hf_token = None,
        scheme = "fp8",
    )
    assert seen["weights_only"] is True
    assert seen["safe_globals"], "the allowlist has to be registered before the load"


def test_the_allowlist_names_every_constructor_the_hosted_checkpoints_use(
    monkeypatch, real_prequant_safe_globals
):
    """The exact set read out of the pickles Unsloth actually resolves.

    Surveyed with ``pickletools`` (no unpickling) over every hosted prequant repo the family
    tables name -- image and video, fp8 and int8, rotated and not -- so a checkpoint naming
    anything beyond this is not one of ours. Torch's defaults cover the storages, dtypes,
    ``_rebuild_*``, ``OrderedDict``, ``torch.device`` and ``_get_layout``; what is left is
    torchao's subclasses plus ``TorchVersion``."""
    listed = {f"{module}.{name}" for module, name in pq._PREQUANT_SAFE_GLOBALS}
    required = {
        # every TQ_SCHEME the builder can bake, not just the two the hosted repos ship
        "torchao.prototype.mx_formats.mx_tensor.MXTensor",
        "torchao.prototype.mx_formats.nvfp4_tensor.NVFP4Tensor",
        "torchao.dtypes.affine_quantized_tensor.AffineQuantizedTensor",
        "torchao.dtypes.uintx.plain_layout.PlainAQTTensorImpl",
        "torchao.dtypes.utils.PlainLayout",
        "torchao.quantization.linear_activation_quantized_tensor.LinearActivationQuantizedTensor",
        "torchao.quantization.quant_api._int8_symm_per_token_reduced_range_quant",
        "torchao.quantization.quant_primitives.ZeroPointDomain",
        "torchao.quantization.Float8Tensor",
        "torchao.quantization.quantize_.workflows.float8.float8_tensor"
        ".QuantizeTensorToFloat8Kwargs",
        "torchao.quantization.quantize_.common.kernel_preference.KernelPreference",
        "torchao.quantization.granularity.PerRow",
        "torchao.float8.inference.Float8MMConfig",
        "torch.torch_version.TorchVersion",
    }
    assert required <= listed, sorted(required - listed)
    # Nothing outside torch/torchao, so the allowlist cannot grow a general-purpose callable.
    assert all(
        module.split(".")[0] in ("torch", "torchao") for module, _ in pq._PREQUANT_SAFE_GLOBALS
    )
    # A name a given torchao release does not ship is skipped, not raised.
    monkeypatch.setattr(
        pq, "_PREQUANT_SAFE_GLOBALS", (("torchao.nowhere", "Nope"), ("collections", "OrderedDict"))
    )
    assert [name for _obj, name in real_prequant_safe_globals()] == ["collections.OrderedDict"]


def test_the_registration_floor_needs_a_real_torchao(real_prequant_safe_globals):
    """On a host that HAS torchao, the real resolution must clear the floor.

    Every other test in this file stands in for the names the host cannot import, which is what
    lets them run on the torchao-free CI image. That stand-in would also hide a torchao release
    that renamed or retired the constructors out from under us -- ``AffineQuantizedTensor`` is
    already deprecated upstream (pytorch/ao#2752). So this one asks the unpatched resolver, and
    skips where there is nothing to ask."""
    pytest.importorskip("torchao")
    resolved = {name for _obj, name in real_prequant_safe_globals()}
    assert "torch.torch_version.TorchVersion" in resolved
    assert [name for name in resolved if name.startswith("torchao.")], (
        "no torchao constructor resolved, so no pre-quant checkpoint could be opened: "
        + repr(sorted(resolved))
    )


def test_the_registration_refuses_when_nothing_resolves(monkeypatch):
    """And where there IS nothing to ask, the load refuses rather than unpickling unrestricted.

    This is the CI image's own situation -- torch installed, torchao not -- so it is worth
    pinning directly: an install that cannot express the allowlist gets a raise and a dense
    fallback, never a ``torch.load`` with the restriction dropped."""
    monkeypatch.setattr(pq, "_prequant_safe_globals", list)
    monkeypatch.setattr(pq, "_SAFE_GLOBALS_REGISTERED", None)
    assert pq._register_prequant_safe_globals() is False
    assert pq.restricted_prequant_load_supported("fp8") is False
    with pytest.raises(RuntimeError, match = "allowlist"):
        pq._torch_load_prequant("/x.pt", map_location = "cpu")


def test_a_malicious_checkpoint_is_refused_before_it_executes():
    """The whole point, against real torch: a checkpoint carrying a ``__reduce__`` payload must
    fail to load rather than run. The unrestricted load this replaces executes it."""
    import os

    torch = pytest.importorskip("torch")
    if not hasattr(torch.serialization, "safe_globals"):
        pytest.skip("torch < 2.6 has no safe_globals")
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        marker = os.path.join(tmp, "executed")

        class _Payload:
            # mkdir rather than a command: observable proof of execution, confined to the temp dir.
            def __reduce__(self):
                return (os.mkdir, (marker,))

        path = os.path.join(tmp, "ckpt.pt")
        torch.save(
            {"format": PREQUANT_FORMAT, "metadata": {"scheme": "int8"}, "state_dict": _Payload()},
            path,
        )
        with pytest.raises(Exception):
            pq._torch_load_prequant(path, map_location = "cpu")
        assert not os.path.exists(marker)


def test_the_scheme_probe_does_not_execute_the_checkpoint_either(tmp_path, monkeypatch):
    """The probe is the WIDER reach of the two, so it gets its own test.

    ``local_prequant_scheme`` runs during download PLANNING, not only during a load, so it is
    reached for a request that never loads anything. An unreadable checkpoint is "unknown", which
    the caller already handles, and nothing in it runs."""
    import os

    torch = pytest.importorskip("torch")
    if not hasattr(torch.serialization, "safe_globals"):
        pytest.skip("torch < 2.6 has no safe_globals")
    marker = tmp_path / "executed"

    class _Payload:
        def __reduce__(self):
            return (os.mkdir, (str(marker),))

    ckpt = tmp_path / "malicious.pt"
    torch.save(
        {"format": PREQUANT_FORMAT, "metadata": {"scheme": "fp8"}, "payload": _Payload()}, ckpt
    )
    monkeypatch.setattr(pq, "_allowed_prequant_roots", lambda: [os.path.realpath(str(tmp_path))])

    assert pq.local_prequant_scheme(str(ckpt)) is None
    assert not marker.exists()
    # And the planning entry point declines the override, as for any unreadable scheme.
    fam = _fam(prequant_repos = (("fp8", "org/hosted-fp8"),))
    assert pq.usable_prequant_source(fam, "fp8", path_override = str(ckpt)) is None
    assert not marker.exists()


class _UnknownConstructor:
    """A constructor no allowlist entry names. Module level so it pickles by reference."""


def test_the_probe_and_the_loader_agree_on_what_is_readable(tmp_path, monkeypatch):
    """One mechanism on both sites, so the two cannot drift apart.

    ``usable_prequant_source`` treats an unreadable scheme as not usable "since the loader would
    reject it too", which only holds while both answer the same question: a probe reading MORE
    than the loader accepts would let planning skip the dense shards for a checkpoint the load
    then drops -- the GGUF silent downgrade the scheme check exists to prevent."""
    import os

    torch = pytest.importorskip("torch")
    if not hasattr(torch.serialization, "safe_globals"):
        pytest.skip("torch < 2.6 has no safe_globals")

    path = tmp_path / "off-allowlist.pt"
    torch.save(
        {
            "format": PREQUANT_FORMAT,
            "metadata": {"scheme": "int8", "base_model_id": "Tongyi-MAI/Z-Image-Turbo"},
            "state_dict": {"weight": _UnknownConstructor()},
        },
        path,
    )
    monkeypatch.setattr(pq, "_allowed_prequant_roots", lambda: [os.path.realpath(str(tmp_path))])

    # The probe: unknown, so planning budgets the dense build.
    assert pq.local_prequant_scheme(str(path)) is None
    fam = _fam(prequant_repos = (("int8", "org/hosted-int8"),))
    assert pq.usable_prequant_source(fam, "int8", path_override = str(path)) is None
    # And the loader agrees rather than installing it: same allowlist, same verdict.
    assert (
        load_prequantized_transformer(
            _FakeTransformer,
            "Tongyi-MAI/Z-Image-Turbo",
            PrequantSource(kind = "path", location = str(path), filename = None),
            device = "cpu",
            dtype = "bf16",
            hf_token = None,
            scheme = "int8",
        )
        is None
    )


def test_a_torch_without_safe_globals_refuses_rather_than_reopening_the_pickle(monkeypatch):
    """No allowlist support means no load. Falling back to an unrestricted one would put the
    sink back on exactly the installs least able to defend it."""
    torch = types.ModuleType("torch")
    torch.load = lambda *a, **k: pytest.fail("torch.load must not run without add_safe_globals")
    torch.serialization = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(pq, "_SAFE_GLOBALS_REGISTERED", None)
    with pytest.raises(RuntimeError, match = "add_safe_globals"):
        pq._torch_load_prequant("/nonexistent.pt", map_location = "cpu")


def test_an_old_torch_registers_nothing_at_all(monkeypatch):
    """2.4/2.5 take the (object, name) pairs without looking at them and only fail later, in
    ``_get_user_allowed_globals``, which reads ``f.__module__`` off every entry of a PROCESS-WIDE
    list -- so a tuple left there breaks every OTHER weights_only load in Unsloth too. Hence:
    decide by version first, register nothing below 2.6."""
    torch = types.ModuleType("torch")
    torch.serialization = types.SimpleNamespace(
        add_safe_globals = lambda entries: pytest.fail("nothing may be registered below 2.6")
    )
    torch.load = lambda *a, **k: pytest.fail("and no load may run")
    for version in ("2.4.0", "2.5.1+cu124", "2.5.0"):
        torch.__version__ = version
        monkeypatch.setitem(sys.modules, "torch", torch)
        monkeypatch.setattr(pq, "_SAFE_GLOBALS_REGISTERED", None)
        assert pq._tuple_safe_globals_supported() is False, version
        assert pq.restricted_prequant_load_supported() is False, version
    torch.__version__ = "2.6.0"
    assert pq._tuple_safe_globals_supported() is True


def test_a_stubbed_torchao_cannot_open_a_checkpoint(monkeypatch):
    """Windows ROCm runs on the torchao IMPORT STUB, which fabricates a class for every name asked
    of it. The allowlist would register those fakes and answer yes for an install that cannot
    rebuild a single quantized tensor, and the H3 auto fallback treats ROCm as eligible, so the
    plan would drop the dense denoiser shards for a checkpoint nothing can open."""
    import core._torchao_stub as stub

    monkeypatch.setattr(pq, "_SAFE_GLOBALS_REGISTERED", None)
    monkeypatch.setattr(stub, "is_stubbed", lambda package: package == "torchao")
    torch = types.ModuleType("torch")
    torch.__version__ = "2.9.1"
    torch.serialization = types.SimpleNamespace(
        add_safe_globals = lambda entries: pytest.fail("a stub must never be registered")
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert pq.restricted_prequant_load_supported() is False


def test_a_torchao_that_resolves_nothing_reports_no_support(monkeypatch):
    """Registering successfully is not the same as being able to open a checkpoint.

    A missing or skewed torchao leaves nothing to register but the torch entries, which
    ``add_safe_globals`` accepts happily -- while the load then refuses the first torchao global
    the file names, after planning already dropped the dense shards for it. So the answer requires
    the two entries every artifact needs whatever its scheme."""
    registered = []
    torch = types.ModuleType("torch")
    torch.__version__ = "2.9.1"
    torch.serialization = types.SimpleNamespace(add_safe_globals = registered.append)
    monkeypatch.setitem(sys.modules, "torch", torch)

    def only(names):
        return [(object(), n) for n in names]

    # torchao contributes nothing: the version string alone opens no checkpoint.
    monkeypatch.setattr(pq, "_SAFE_GLOBALS_REGISTERED", None)
    monkeypatch.setattr(
        pq, "_prequant_safe_globals", lambda: only(["torch.torch_version.TorchVersion"])
    )
    assert pq.restricted_prequant_load_supported() is False
    assert registered == [], "and nothing is registered on the way to saying no"

    # torchao is there but the version stamp is not: every torchao checkpoint carries one.
    monkeypatch.setattr(pq, "_SAFE_GLOBALS_REGISTERED", None)
    monkeypatch.setattr(
        pq, "_prequant_safe_globals", lambda: only(["torchao.quantization.Float8Tensor"])
    )
    assert pq.restricted_prequant_load_supported() is False

    # Both present: supported, and registered exactly once.
    monkeypatch.setattr(pq, "_SAFE_GLOBALS_REGISTERED", None)
    monkeypatch.setattr(
        pq,
        "_prequant_safe_globals",
        lambda: only(["torch.torch_version.TorchVersion", "torchao.quantization.Float8Tensor"]),
    )
    assert pq.restricted_prequant_load_supported() is True
    assert len(registered) == 1


def test_support_is_answered_per_scheme(monkeypatch):
    """The schemes do not share constructors, and torchao does not retire them together.

    AffineQuantizedTensor and its layout carry every int8 checkpoint and are already deprecated
    upstream (pytorch/ao#2752), so a release that drops them while keeping Float8Tensor leaves fp8
    loadable and int8 not. One answer for both would drop the dense shards for an int8 pick this
    install cannot open."""
    torch = types.ModuleType("torch")
    torch.__version__ = "2.9.1"
    torch.serialization = types.SimpleNamespace(add_safe_globals = lambda entries: None)
    monkeypatch.setitem(sys.modules, "torch", torch)

    fp8_only = set(pq._SCHEME_REQUIRED_GLOBALS["fp8"])
    monkeypatch.setattr(pq, "_SAFE_GLOBALS_REGISTERED", None)
    monkeypatch.setattr(pq, "_RESOLVED_SAFE_GLOBALS", set())
    monkeypatch.setattr(pq, "_prequant_safe_globals", lambda: [(object(), n) for n in fp8_only])

    assert pq.restricted_prequant_load_supported("fp8") is True
    assert pq.restricted_prequant_load_supported("int8") is False
    assert pq.restricted_prequant_load_supported("nvfp4") is False
    # An unnamed or unknown scheme gets the floor answer the registration already checked.
    assert pq.restricted_prequant_load_supported() is True
    assert pq.restricted_prequant_load_supported("something-else") is True
    # And the source resolver carries the scheme through, so an int8 pick is not offered.
    fam = _fam(prequant_repos = (("int8", "org/hosted-int8"), ("fp8", "org/hosted-fp8")))
    assert pq.usable_prequant_source(fam, "int8") is None
    assert pq.usable_prequant_source(fam, "fp8") is not None


def test_the_required_sets_are_a_subset_of_the_allowlist():
    """A required name the allowlist never registers would refuse its scheme forever."""
    listed = {f"{module}.{name}" for module, name in pq._PREQUANT_SAFE_GLOBALS}
    for scheme, required in pq._SCHEME_REQUIRED_GLOBALS.items():
        assert required <= listed, (scheme, sorted(required - listed))


def test_an_install_that_cannot_restrict_the_load_offers_no_prequant_source(monkeypatch, tmp_path):
    """Planning has to ask the loader's question BEFORE it sizes the load.

    A plan that keeps a hosted pre-quant source, drops the dense shards and evicts the resident
    pipeline has nothing left when the loader then refuses every checkpoint: the dense build it
    now needs was never budgeted or staged."""
    import os

    fam = _fam(prequant_repos = (("int8", "org/hosted-int8"),))
    monkeypatch.setattr(pq, "restricted_prequant_load_supported", lambda scheme = None: True)
    assert pq.usable_prequant_source(fam, "int8") is not None
    monkeypatch.setattr(pq, "restricted_prequant_load_supported", lambda scheme = None: False)
    # Hosted and local alike: the loader refuses both, so neither is usable.
    assert pq.usable_prequant_source(fam, "int8") is None
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"x")
    monkeypatch.setattr(pq, "_allowed_prequant_roots", lambda: [os.path.realpath(str(tmp_path))])
    assert pq.usable_prequant_source(fam, "int8", path_override = str(ckpt)) is None


def test_the_allowlist_is_registered_once_and_never_withdrawn(monkeypatch):
    """The registration must OUTLIVE the load that installed it.

    ``safe_globals`` as a context manager adds on entry and removes on exit, against a
    process-wide table that is not refcounted. Two overlapping reads (a download-plan probe beside
    a load, both on the route's thread pool) then let whichever finishes first strip the allowlist
    out from under the other's ``torch.load``, dropping a good checkpoint to dense."""
    torch = pytest.importorskip("torch")
    if not hasattr(torch.serialization, "add_safe_globals"):
        pytest.skip("torch without add_safe_globals")
    calls = []
    monkeypatch.setattr(pq, "_SAFE_GLOBALS_REGISTERED", None)
    monkeypatch.setattr(
        torch.serialization, "add_safe_globals", lambda entries: calls.append(list(entries))
    )
    # No remove_safe_globals / safe_globals context: nothing gives the entries back.
    monkeypatch.setattr(
        torch.serialization,
        "safe_globals",
        lambda *a, **k: pytest.fail("the load must not scope the allowlist to itself"),
    )
    monkeypatch.setattr(torch, "load", lambda *a, **k: {"ok": True})
    assert pq._torch_load_prequant("/x.pt", map_location = "cpu") == {"ok": True}
    assert pq._torch_load_prequant("/x.pt", map_location = "cpu") == {"ok": True}
    assert len(calls) == 1, "registered once for the process, not per load"
    assert calls[0], "and with the allowlist, not an empty list"


# ── local-path opt-in gate ───────────────────────────────────────────────────────
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
        local_files_only = False,
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
        local_files_only = False,
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
        local_files_only = False,
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
        local_files_only = False,
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
    # The other half: a real fetch must land where Unsloth is reading, not under the stale constant.
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
            # An ordinary user-initiated load still downloads; only an API-initiated one is pinned
            # to the cache, and that is the caller's flag rather than this helper's default.
            "local_files_only": False,
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
        local_files_only = False,
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
        local_files_only = False,
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
        local_files_only = False,
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
        local_files_only = False,
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
        local_files_only = False,
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

    monkeypatch.setattr(P, "_resolve_checkpoint_path", lambda s, t, c = None, **_: str(ckpt))
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
            local_files_only = False,
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
    """``_resolve_checkpoint_path`` can answer from huggingface_hub's import-time root while Unsloth
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

    monkeypatch.setattr(P, "_resolve_checkpoint_path", lambda s, t, c = None, **_: str(ckpt))
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
            local_files_only = False,
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


# ── small-M activation padding on the hosted path ────────────────────────────────


def test_load_pads_the_small_m_linears_with_the_recorded_family(monkeypatch, tmp_path):
    """A checkpoint built under the current exclusion set QUANTISES its family's small-M linears,
    so the loader must wrap them exactly as the runtime dense-quantise path does. The family comes
    from the checkpoint's own metadata, not from the caller: it is the same field
    ``_validate_checkpoint`` derives the expected exclusion set from, so the two cannot disagree
    about which model this is."""
    from core.inference.diffusion_transformer_quant import exclude_tokens_for_scheme

    seen = {}

    def _spy(
        transformer,
        scheme,
        family = None,
        logger = None,
    ):
        seen["args"] = (scheme, family)
        return ("context_embedder",)

    monkeypatch.setattr("core.inference.diffusion_transformer_quant.apply_small_m_padding", _spy)
    ckpt = _good_ckpt(scheme = "int8")
    ckpt["metadata"]["family"] = "minimax-h3"
    ckpt["metadata"]["exclude_name_tokens"] = list(exclude_tokens_for_scheme("int8", "minimax-h3"))
    assert _load(monkeypatch, tmp_path, ckpt, scheme = "int8") is not None
    assert seen["args"] == ("int8", "minimax-h3")


def test_load_is_dropped_when_the_padding_cannot_be_proven(monkeypatch, tmp_path):
    """Half-padded is the worst outcome: it compiles on the modules that were wrapped and crashes
    inside ``_int_mm`` on the ones that were not. A raise must drop the prequant so the caller
    falls back to dense-quantise, rather than returning a transformer that renders once and dies
    the moment the compiled scope reaches the text stream."""

    def _boom(
        transformer,
        scheme,
        family = None,
        logger = None,
    ):
        raise RuntimeError("cannot prove per-row granularity")

    monkeypatch.setattr("core.inference.diffusion_transformer_quant.apply_small_m_padding", _boom)
    ckpt = _good_ckpt(scheme = "int8")
    ckpt["metadata"]["family"] = "minimax-h3"
    assert _load(monkeypatch, tmp_path, ckpt, scheme = "int8") is None


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
