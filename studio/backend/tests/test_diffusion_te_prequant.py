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
    te_prequant_repo_filenames,
)


def _fam(
    te_prequant_repos = (),
    name = "ltx-2",
    base_repo = "Lightricks/LTX-2",
):
    return types.SimpleNamespace(
        name = name,
        base_repo = base_repo,
        te_prequant_repos = te_prequant_repos,
    )


# ── resolution ───────────────────────────────────────────────────────────────
def test_repo_filename_convention():
    # safetensors first, the historical pickle second: both are live and the reader takes either.
    assert te_prequant_repo_filenames("unsloth/LTX-2-FP8", "text_encoder", "fp8") == (
        "LTX-2-text_encoder-FP8.safetensors",
        "LTX-2-text_encoder-FP8.pt",
    )
    assert te_prequant_repo_filenames("org/Some-Model-quantized", "text_encoder_2", "fp8") == (
        "Some-Model-text_encoder_2-FP8.safetensors",
        "Some-Model-text_encoder_2-FP8.pt",
    )
    assert te_prequant_repo_filenames("org/PlainRepo", "text_encoder", "fp8") == (
        "PlainRepo-text_encoder-FP8.safetensors",
        "PlainRepo-text_encoder-FP8.pt",
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
    assert (
        family_te_prequant_repo(_fam(te_prequant_repos = (("bad",),)), "fp8", "text_encoder") is None
    )
    # Families without the field resolve to None (both dataclasses default it, but a fake or older family object must not break).
    assert family_te_prequant_repo(types.SimpleNamespace(name = "x"), "fp8", "text_encoder") is None


def test_resolve_priority_and_scheme_gate():
    fam = _fam(te_prequant_repos = (("fp8", "text_encoder", "org/hosted-fp8"),))
    # Path override wins.
    src = resolve_te_prequant_source(fam, "text_encoder", "fp8", path_override = "/tmp/te.pt")
    assert src == TePrequantSource(kind = "path", location = "/tmp/te.pt", filename = None)
    # Hosted repo second.
    src = resolve_te_prequant_source(fam, "text_encoder", "fp8")
    assert src.kind == "repo" and src.location == "org/hosted-fp8"
    assert src.filename == "hosted-text_encoder-FP8.safetensors"
    assert src.fallback_filenames == ("hosted-text_encoder-FP8.pt",)
    # Nothing configured -> None.
    assert resolve_te_prequant_source(_fam(), "text_encoder", "fp8") is None
    # v1 hosts the layerwise fp8 storage scheme only.
    assert resolve_te_prequant_source(fam, "text_encoder", "int8") is None
    assert resolve_te_prequant_source(fam, "text_encoder", "fp8_dynamic") is None


def test_a_hosted_safetensors_encoder_is_asked_for_and_a_pickle_repo_still_resolves(monkeypatch):
    """The regression this exists for: the resolver only ever asked for ``.pt``.

    A repo hosting the encoder as safetensors then 404'd, and because the loader is best effort the
    user silently got the dense encoder instead (17.5 GB against 9.4 GB on Qwen-Image-2.1) with no
    error anywhere. Both extensions have to be reachable, and the repos that still host a pickle
    have to keep working, so this drives the download path with each in turn.
    """
    from huggingface_hub.errors import EntryNotFoundError

    asked = []

    def fake_download(*, repo_id, filename, token, cache_dir, local_files_only):
        asked.append(filename)
        if filename not in hosted:
            raise EntryNotFoundError(f"no {filename} in {repo_id}")
        return f"/cache/{filename}"

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    src = resolve_te_prequant_source(
        _fam(te_prequant_repos = (("fp8", "text_encoder", "org/hosted-fp8"),)),
        "text_encoder",
        "fp8",
    )

    # A safetensors repo is found first, with no wasted request for the pickle.
    hosted = {"hosted-text_encoder-FP8.safetensors"}
    asked.clear()
    assert tpq._resolve_checkpoint_path(src, None, cache_dir = "/cache") == (
        "/cache/hosted-text_encoder-FP8.safetensors"
    )
    assert asked == ["hosted-text_encoder-FP8.safetensors"]

    # A repo that still hosts the pickle resolves through the fallback.
    hosted = {"hosted-text_encoder-FP8.pt"}
    asked.clear()
    assert tpq._resolve_checkpoint_path(src, None, cache_dir = "/cache") == (
        "/cache/hosted-text_encoder-FP8.pt"
    )
    assert asked == ["hosted-text_encoder-FP8.safetensors", "hosted-text_encoder-FP8.pt"]

    # Neither present: the miss surfaces rather than being swallowed into a None the caller
    # cannot tell apart from "this family hosts nothing".
    hosted = set()
    with pytest.raises(EntryNotFoundError):
        tpq._resolve_checkpoint_path(src, None, cache_dir = "/cache")


def test_a_transport_failure_is_not_mistaken_for_a_missing_file(monkeypatch):
    """Only "this name is not in this repo" may advance to the next candidate.

    An auth failure or a dead network must not be retried as a different extension and then
    reported as an absent artifact: that would turn a fixable error into a silent 17.5 GB
    download on every load.
    """

    def fake_download(*, repo_id, filename, token, cache_dir, local_files_only):
        raise PermissionError("401 unauthorized")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    src = resolve_te_prequant_source(
        _fam(te_prequant_repos = (("fp8", "text_encoder", "org/hosted-fp8"),)),
        "text_encoder",
        "fp8",
    )
    with pytest.raises(PermissionError):
        tpq._resolve_checkpoint_path(src, None, cache_dir = "/cache")


# ── checkpoint validation ────────────────────────────────────────────────────
def _good_ckpt(
    scheme = "fp8",
    component = "text_encoder",
    base = "Lightricks/LTX-2",
):
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


def test_hosted_checkpoint_and_config_honor_cache_only_and_the_active_root(monkeypatch, tmp_path):
    import huggingface_hub
    import torch
    import transformers
    from utils import hf_cache_settings

    seen: dict = {"download": {}, "config": {}}

    def fake_download(**kwargs):
        seen["download"].update(kwargs)
        return "/cache/encoder.pt"

    def fake_config(repo_id, **kwargs):
        seen["config"] = {"repo_id": repo_id, **kwargs}
        raise FileNotFoundError("stop after config lookup")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(torch, "load", lambda *_a, **_k: _good_ckpt())
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", fake_config)
    monkeypatch.setattr(hf_cache_settings, "active_hf_hub_cache", lambda: str(tmp_path))
    out = tpq.load_prequant_text_encoder(
        "Lightricks/LTX-2",
        "text_encoder",
        TePrequantSource(kind = "repo", location = "org/hosted", filename = "encoder.pt"),
        dtype = None,
        local_files_only = True,
    )
    assert out is None
    assert seen["download"]["local_files_only"] is True
    assert seen["download"]["cache_dir"] == str(tmp_path)
    assert seen["config"]["repo_id"] == "Lightricks/LTX-2"
    assert seen["config"]["subfolder"] == "text_encoder"
    assert seen["config"]["local_files_only"] is True
    assert seen["config"]["cache_dir"] == str(tmp_path)


# ── pipeline-assembly injection gating ───────────────────────────────────────
def _target():
    return types.SimpleNamespace(device = "cuda", dtype = None)


def _budget_scale(
    fam,
    mode = "fp8",
    *,
    base = None,
):
    return tpq.te_prequant_budget_scale(
        fam, te_quant_mode = mode, target = _target(), base = base or fam.base_repo
    )


def test_pipe_kwargs_empty_when_mode_not_fp8(monkeypatch):
    fam = _fam(te_prequant_repos = (("fp8", "text_encoder", "org/hosted"),))
    for mode in (None, "", "off", "int8", "fp8_dynamic"):
        assert (
            te_prequant_pipe_kwargs(
                fam, "Lightricks/LTX-2", te_quant_mode = mode, target = _target(), dtype = None
            )
            == {}
        )


def test_pipe_kwargs_empty_without_hosted_entry(monkeypatch):
    import core.inference.diffusion_precision as precision
    monkeypatch.setattr(precision, "te_quant_supported", lambda target, mode: True)
    assert (
        te_prequant_pipe_kwargs(
            _fam(), "Lightricks/LTX-2", te_quant_mode = "fp8", target = _target(), dtype = None
        )
        == {}
    )


def test_pipe_kwargs_empty_when_device_unsupported(monkeypatch):
    import core.inference.diffusion_precision as precision

    fam = _fam(te_prequant_repos = (("fp8", "text_encoder", "org/hosted"),))
    monkeypatch.setattr(precision, "te_quant_supported", lambda target, mode: False)
    assert (
        te_prequant_pipe_kwargs(
            fam, "Lightricks/LTX-2", te_quant_mode = "fp8", target = _target(), dtype = None
        )
        == {}
    )


def test_pipe_kwargs_respects_family_deny(monkeypatch):
    import core.inference.diffusion_precision as precision

    fam = _fam(te_prequant_repos = (("fp8", "text_encoder", "org/hosted"),))
    monkeypatch.setattr(precision, "te_quant_supported", lambda target, mode: True)
    # The deny helper ships on the video branch's precision module; simulate it here.
    monkeypatch.setattr(
        precision, "_te_family_denied", lambda family, mode: family == "ltx-2", raising = False
    )
    assert (
        te_prequant_pipe_kwargs(
            fam, "Lightricks/LTX-2", te_quant_mode = "fp8", target = _target(), dtype = None
        )
        == {}
    )


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


def test_pipe_kwargs_does_not_download_a_checkpoint_for_a_custom_base(monkeypatch):
    import core.inference.diffusion_precision as precision

    fam = _fam(te_prequant_repos = (("fp8", "text_encoder", "org/hosted"),))
    monkeypatch.setattr(precision, "te_quant_supported", lambda target, mode: True)

    def unexpected_load(*_args, **_kwargs):
        raise AssertionError("an incompatible hosted checkpoint must not be opened")

    monkeypatch.setattr(tpq, "load_prequant_text_encoder", unexpected_load)
    assert (
        te_prequant_pipe_kwargs(
            fam,
            "someone/custom-ltx-2",
            te_quant_mode = "fp8",
            target = _target(),
            dtype = None,
        )
        == {}
    )


def test_pipe_kwargs_empty_when_load_fails(monkeypatch):
    import core.inference.diffusion_precision as precision

    fam = _fam(te_prequant_repos = (("fp8", "text_encoder", "org/hosted"),))
    monkeypatch.setattr(precision, "te_quant_supported", lambda target, mode: True)
    monkeypatch.setattr(tpq, "load_prequant_text_encoder", lambda *a, **k: None)
    assert (
        te_prequant_pipe_kwargs(
            fam, "Lightricks/LTX-2", te_quant_mode = "fp8", target = _target(), dtype = None
        )
        == {}
    )


def test_pipe_kwargs_injects_every_hosted_component(monkeypatch):
    """A family hosting several TE components (flux.1: T5 as text_encoder_2) gets each
    one injected under its own attr; unhosted components stay dense."""
    import core.inference.diffusion_precision as precision

    fam = _fam(
        te_prequant_repos = (
            ("fp8", "text_encoder", "org/hosted"),
            ("fp8", "text_encoder_2", "org/hosted-2"),
        )
    )
    monkeypatch.setattr(precision, "te_quant_supported", lambda target, mode: True)
    markers = {"text_encoder": object(), "text_encoder_2": object()}
    monkeypatch.setattr(
        tpq,
        "load_prequant_text_encoder",
        lambda base, component, source, **kw: markers[component],
    )
    out = te_prequant_pipe_kwargs(
        fam, "Lightricks/LTX-2", te_quant_mode = "fp8", target = _target(), dtype = None
    )
    assert out == markers


# ── base equivalence ─────────────────────────────────────────────────────────
def test_te_base_equivalent_groups():
    from core.inference.diffusion_te_prequant import te_base_equivalent

    # Same repo (case-folded) always matches.
    assert te_base_equivalent("Qwen/Qwen-Image", "qwen/qwen-image")
    # Verified byte-identical groups match across repos, both directions.
    assert te_base_equivalent(
        "Qwen/Qwen-Image", "hunyuanvideo-community/HunyuanImage-2.1-Diffusers"
    )
    assert te_base_equivalent("black-forest-labs/FLUX.1-schnell", "black-forest-labs/FLUX.1-dev")
    assert te_base_equivalent(
        "black-forest-labs/FLUX.1-Krea-dev", "black-forest-labs/FLUX.1-schnell"
    )
    # Z-Image ships one Qwen3-4B encoder for the distilled Turbo and the undistilled base, so the
    # Turbo-baked artifact serves both and training on the base does not re-pull it dense.
    assert te_base_equivalent("Tongyi-MAI/Z-Image-Turbo", "Tongyi-MAI/Z-Image")
    assert te_base_equivalent("Tongyi-MAI/Z-Image", "Tongyi-MAI/Z-Image-Turbo")
    # Unrelated bases stay refused, including across groups.
    assert not te_base_equivalent("Qwen/Qwen-Image", "black-forest-labs/FLUX.1-schnell")
    assert not te_base_equivalent("Tongyi-MAI/Z-Image-Turbo", "black-forest-labs/FLUX.2-klein-4B")
    assert not te_base_equivalent("Tongyi-MAI/Z-Image", "Qwen/Qwen-Image")


def test_validate_accepts_equivalent_base():
    ckpt = {
        "format": TE_PREQUANT_FORMAT,
        "state_dict": {},
        "metadata": {
            "scheme": "fp8",
            "component": "text_encoder",
            "base_model_id": "Qwen/Qwen-Image",
        },
    }
    assert tpq._validate_checkpoint(
        ckpt,
        "fp8",
        "text_encoder",
        "hunyuanvideo-community/HunyuanImage-2.1-Diffusers",
        None,
    )
    assert not tpq._validate_checkpoint(
        ckpt, "fp8", "text_encoder", "black-forest-labs/FLUX.1-schnell", None
    )


# ── family field wiring ──────────────────────────────────────────────────────
def test_family_dataclasses_declare_te_prequant_field():
    from core.inference.diffusion_families import DiffusionFamily, detect_family
    from core.inference.video_families import VideoFamily

    assert DiffusionFamily.__dataclass_fields__["te_prequant_repos"].default_factory is tuple
    assert VideoFamily.__dataclass_fields__["te_prequant_repos"].default_factory is tuple
    # Families without a hosted TE checkpoint keep the empty default (sdxl's CLIPs stay dense; flux.1 hosts its T5, asserted below).
    fam = detect_family("stabilityai/stable-diffusion-xl-base-1.0")
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
    assert te_prequant_repo_filenames("unsloth/Qwen-Image-FP8", "text_encoder", "fp8") == (
        "Qwen-Image-text_encoder-FP8.safetensors",
        "Qwen-Image-text_encoder-FP8.pt",
    )
    assert te_prequant_repo_filenames("unsloth/FLUX.2-dev-FP8", "text_encoder", "fp8") == (
        "FLUX.2-dev-text_encoder-FP8.safetensors",
        "FLUX.2-dev-text_encoder-FP8.pt",
    )
    assert te_prequant_repo_filenames("unsloth/LTX-2-FP8", "text_encoder", "fp8") == (
        "LTX-2-text_encoder-FP8.safetensors",
        "LTX-2-text_encoder-FP8.pt",
    )
    # HiDream's heavyweight is TE4 (Llama-3.1-8B), engaged via hidream_te4_kwargs since the generic pass only covers text_encoder.._3.
    assert detect_family("HiDream-ai/HiDream-I1-Full").te_prequant_repos == (
        ("fp8", "text_encoder_4", "unsloth/HiDream-I1-Full-FP8"),
    )
    assert te_prequant_repo_filenames("unsloth/HiDream-I1-Full-FP8", "text_encoder_4", "fp8") == (
        "HiDream-I1-Full-text_encoder_4-FP8.safetensors",
        "HiDream-I1-Full-text_encoder_4-FP8.pt",
    )
    # Round 2: T5-XXL for every flux.1 base (byte-identical, one artifact), Gemma2-2B, Qwen3-4B, Qwen3-VL-4B, and hunyuanimage reusing the Qwen-Image artifact.
    assert detect_family("black-forest-labs/FLUX.1-schnell").te_prequant_repos == (
        ("fp8", "text_encoder_2", "unsloth/FLUX.1-schnell-FP8"),
    )
    assert te_prequant_repo_filenames("unsloth/FLUX.1-schnell-FP8", "text_encoder_2", "fp8") == (
        "FLUX.1-schnell-text_encoder_2-FP8.safetensors",
        "FLUX.1-schnell-text_encoder_2-FP8.pt",
    )
    assert detect_family("Alpha-VLLM/Lumina-Image-2.0").te_prequant_repos == (
        ("fp8", "text_encoder", "unsloth/Lumina-Image-2.0-FP8"),
    )
    assert detect_family("Tongyi-MAI/Z-Image-Turbo").te_prequant_repos == (
        ("fp8", "text_encoder", "unsloth/Z-Image-Turbo-FP8"),
    )
    assert detect_family("krea/Krea-2-Turbo").te_prequant_repos == (
        ("fp8", "text_encoder", "unsloth/Krea-2-Turbo-FP8"),
    )
    assert detect_family("hunyuanvideo-community/HunyuanImage-2.1-Diffusers").te_prequant_repos == (
        ("fp8", "text_encoder", "unsloth/Qwen-Image-FP8"),
    )
    # flux.2-klein-4B hosts NO TE entry: its Qwen3-4B retrained layer 35's MLP, so the z-image artifact must not serve it (maxdiff 0.86).
    assert detect_family("black-forest-labs/FLUX.2-klein-4B").te_prequant_repos == ()


def _hidream_transformers_stub(monkeypatch, recorder):
    """Fake transformers surface for hidream_te4_kwargs: records from_pretrained calls."""
    import sys

    class _FakeLlama:
        def __init__(self, tag):
            self.tag = tag

    class _LlamaCls:
        @staticmethod
        def from_pretrained(repo, **kwargs):
            recorder.append(("llama_from_pretrained", repo))
            return _FakeLlama(f"dense{len(recorder)}")

    class _TokCls:
        @staticmethod
        def from_pretrained(repo, **kwargs):
            recorder.append(("tokenizer", repo))
            return "tok4"

    fake = types.ModuleType("transformers")
    fake.AutoTokenizer = _TokCls
    fake.LlamaForCausalLM = _LlamaCls
    monkeypatch.setitem(sys.modules, "transformers", fake)
    return _FakeLlama


def test_hidream_te4_stays_dense_without_fp8(monkeypatch):
    from core.inference.diffusion_hidream import hidream_te4_kwargs

    recorder: list = []
    _hidream_transformers_stub(monkeypatch, recorder)
    out = hidream_te4_kwargs(
        None, None, fam = _fam(name = "hidream-i1"), te_quant_mode = None, target = _target()
    )
    assert out["tokenizer_4"] == "tok4"
    assert getattr(out["text_encoder_4"], "tag", "").startswith("dense")
    # No cast attempted: mode None normalises to no TE quant.
    assert ("llama_from_pretrained", "unsloth/Meta-Llama-3.1-8B-Instruct") in recorder


def test_hidream_te4_prefers_precast_checkpoint(monkeypatch):
    import core.inference.diffusion_hidream as dh
    import core.inference.diffusion_precision as precision

    recorder: list = []
    _hidream_transformers_stub(monkeypatch, recorder)
    monkeypatch.setattr(precision, "te_quant_supported", lambda target, mode: True)
    precast = object()
    calls: dict = {}

    def _fake_load(base, component, source, **kwargs):
        calls["base"] = base
        calls["component"] = component
        calls["config_subfolder"] = kwargs.get("config_subfolder")
        calls["config_overrides"] = kwargs.get("config_overrides")
        calls["local_files_only"] = kwargs.get("local_files_only")
        return precast

    monkeypatch.setattr(tpq, "load_prequant_text_encoder", _fake_load)
    fam = _fam(
        te_prequant_repos = (("fp8", "text_encoder_4", "unsloth/HiDream-I1-Full-FP8"),),
        name = "hidream-i1",
    )
    out = dh.hidream_te4_kwargs(
        None,
        None,
        fam = fam,
        te_quant_mode = "fp8",
        target = _target(),
        local_files_only = True,
    )
    assert out["text_encoder_4"] is precast
    assert calls["base"] == "unsloth/Meta-Llama-3.1-8B-Instruct"
    assert calls["component"] == "text_encoder_4"
    # Standalone repo: config at the root, forward flags the pipeline needs applied.
    assert calls["config_subfolder"] == ""
    assert calls["config_overrides"] == {"output_hidden_states": True, "output_attentions": True}
    assert calls["local_files_only"] is True
    # The dense Llama download never ran.
    assert ("llama_from_pretrained", "unsloth/Meta-Llama-3.1-8B-Instruct") not in recorder


def test_hidream_te4_falls_back_to_dense_cast(monkeypatch):
    import core.inference.diffusion_hidream as dh
    import core.inference.diffusion_precision as precision

    recorder: list = []
    _hidream_transformers_stub(monkeypatch, recorder)
    monkeypatch.setattr(precision, "te_quant_supported", lambda target, mode: True)
    monkeypatch.setattr(tpq, "load_prequant_text_encoder", lambda *a, **k: None)
    cast: list = []
    monkeypatch.setattr(precision, "_cast_fp8", lambda enc, tgt: cast.append(enc))
    fam = _fam(
        te_prequant_repos = (("fp8", "text_encoder_4", "unsloth/HiDream-I1-Full-FP8"),),
        name = "hidream-i1",
    )
    out = dh.hidream_te4_kwargs(None, None, fam = fam, te_quant_mode = "fp8", target = _target())
    assert cast == [out["text_encoder_4"]]
    assert ("llama_from_pretrained", "unsloth/Meta-Llama-3.1-8B-Instruct") in recorder


def test_hidream_te4_partial_cast_reloads_dense(monkeypatch):
    """A mid-pass TE4 cast failure must ship a FRESH dense encoder, not partial fp8 state."""
    import core.inference.diffusion_hidream as dh
    import core.inference.diffusion_precision as precision

    recorder: list = []
    _hidream_transformers_stub(monkeypatch, recorder)
    monkeypatch.setattr(precision, "te_quant_supported", lambda target, mode: True)

    def _boom(enc, tgt):
        raise RuntimeError("cast failed mid-pass")

    monkeypatch.setattr(precision, "_cast_fp8", _boom)
    fam = _fam(name = "hidream-i1")  # no hosted entry -> dense + cast path
    out = dh.hidream_te4_kwargs(None, None, fam = fam, te_quant_mode = "fp8", target = _target())
    dense_loads = [r for r in recorder if r[0] == "llama_from_pretrained"]
    assert len(dense_loads) == 2  # initial load + the fail-safe reload
    assert getattr(out["text_encoder_4"], "tag", "").startswith("dense")


def test_assemble_pipe_injects_precast_te(monkeypatch):
    """The dense transformer_quant fast path assembles companions through _assemble_pipe,
    which must inject the hosted pre-cast TE like the full-pipeline and GGUF branches."""
    import core.inference.diffusion as dif

    seen: dict = {}

    class FakePipe:
        def to(self, device):
            return self

    class FakePipelineCls:
        @staticmethod
        def from_pretrained(base, **kw):
            seen.update(kw)
            return FakePipe()

    monkeypatch.setattr(dif, "te_prequant_pipe_kwargs", lambda *a, **k: {"text_encoder": "PRECAST"})
    dif.DiffusionBackend._assemble_pipe(
        FakePipelineCls,
        "org/base",
        "TR",
        None,
        None,
        "cpu",
        None,
        fam = None,
        te_quant_mode = "fp8",
        target = object(),
    )
    assert seen["text_encoder"] == "PRECAST"
    seen.clear()
    # No target (defensive default) keeps the assembly unchanged.
    dif.DiffusionBackend._assemble_pipe(
        FakePipelineCls,
        "org/base",
        "TR",
        None,
        None,
        "cpu",
        None,
        fam = None,
    )
    assert "text_encoder" not in seen


def test_cast_fp8_is_idempotent_on_precast_encoder():
    """A pre-cast encoder arrives with the layerwise hooks installed; the runtime re-apply in
    quantize_text_encoders must be a no-op (re-registering the hook name raises, which made
    the engaged cast report as failed and status show no TE quant)."""
    import torch

    pytest.importorskip("diffusers")  # _cast_fp8 installs diffusers' layerwise hooks
    from core.inference.diffusion_precision import _cast_fp8

    target = types.SimpleNamespace(dtype = torch.bfloat16)
    enc = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.LayerNorm(64))
    _cast_fp8(enc, target)
    assert enc[0].weight.dtype == torch.float8_e4m3fn
    # Module.dtype must report the COMPUTE dtype: pipelines derive tensor dtypes from it (Flux2 feeds it to randn_tensor, which has no fp8 kernel).
    assert enc.dtype == torch.bfloat16
    # EXACT class identity: a dynamic-subclass swap broke transformers' kwargs-based output recording (Qwen3VLModel returned hidden_states=None).
    assert type(enc) is torch.nn.Sequential
    # An uncast sibling of the same (now property-patched) class keeps original behaviour.
    sibling = torch.nn.Sequential(torch.nn.Linear(8, 8))
    with pytest.raises(AttributeError):
        sibling.dtype
    _cast_fp8(enc, target)  # must not raise
    assert enc[0].weight.dtype == torch.float8_e4m3fn
    assert enc.dtype == torch.bfloat16


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


# ── memory budgeting ─────────────────────────────────────────────────────────
# Hosted checkpoint bytes over bf16-equivalent dense bytes, read from Hub file metadata on
# 2026-08-07. The budget constant is a CEILING over these, so it can never under-state a
# pre-cast encoder; PR #8213 gates a hard load refusal on the number this feeds.
_MEASURED_FP8_RATIOS = {
    "flux.2-dev/text_encoder": (24_683_130_873, 48_022_800_560),
    "hidream-i1-full/text_encoder_4": (8_555_963_320, 16_060_556_376),
    "qwen-image/text_encoder": (8_839_210_073, 16_584_414_544),
    "ltx-2/text_encoder": (13_205_302_695, 24_374_720_836),
    "krea-2-turbo/text_encoder": (4_831_262_424, 8_875_715_136),
    "z-image-turbo/text_encoder": (4_411_751_967, 8_044_982_000),
    "lumina-image-2.0/text_encoder": (3_204_501_909, 5_228_699_608),
    "flux.1-schnell/text_encoder_2": (5_900_818_800, 9_524_648_584),
}


def test_budget_scale_over_states_every_measured_artifact():
    worst = max(fp8 / dense for fp8, dense in _MEASURED_FP8_RATIOS.values())
    # Conservative by construction: budget at or above the largest realized artifact...
    assert tpq.TE_PREQUANT_BUDGET_SCALE >= worst
    # ...and still below bf16, or the fix does nothing.
    assert tpq.TE_PREQUANT_BUDGET_SCALE < 1.0
    # fp8 storage is one byte per parameter against bf16's two, so nothing can come in under 0.5.
    assert min(fp8 / dense for fp8, dense in _MEASURED_FP8_RATIOS.values()) > 0.5


def test_budget_scale_applies_only_when_a_pre_cast_checkpoint_resolves(monkeypatch):
    import core.inference.diffusion_precision as precision

    monkeypatch.setattr(precision, "te_quant_supported", lambda target, mode: True)
    hosted = _fam(te_prequant_repos = (("fp8", "text_encoder", "org/hosted"),))
    assert _budget_scale(hosted) == tpq.TE_PREQUANT_BUDGET_SCALE
    assert _budget_scale(hosted, base = "someone/custom-ltx-2") == 1.0
    # No hosted checkpoint: the encoder is downloaded dense and cast in place AFTER assembly, so
    # its peak is bf16 and the budget must stay bf16.
    assert _budget_scale(_fam()) == 1.0
    # Not requested, or a scheme with no hosted artifact.
    for mode in (None, "", "off", "int8", "fp8_dynamic", "nvfp4"):
        assert _budget_scale(hosted, mode) == 1.0


def test_budget_scale_is_bf16_when_the_device_cannot_quantise(monkeypatch):
    import core.inference.diffusion_precision as precision

    hosted = _fam(te_prequant_repos = (("fp8", "text_encoder", "org/hosted"),))
    monkeypatch.setattr(precision, "te_quant_supported", lambda target, mode: False)
    assert _budget_scale(hosted) == 1.0


def test_budget_scale_fails_open_to_bf16(monkeypatch):
    # An unresolvable pick keeps today's (larger) budget rather than guessing small.
    def _boom(*args, **kwargs):
        raise RuntimeError("hub down")

    monkeypatch.setattr(tpq, "te_prequant_sources", _boom)
    assert _budget_scale(_fam()) == 1.0


def test_shipped_video_and_image_families_resolve_the_scale(monkeypatch):
    import core.inference.diffusion_precision as precision
    from core.inference.diffusion_families import detect_family
    from core.inference.video_families import detect_video_family

    monkeypatch.setattr(precision, "te_quant_supported", lambda target, mode: True)
    scale = tpq.TE_PREQUANT_BUDGET_SCALE
    # ltx-2 hosts its Gemma3-12B encoder pre-cast; the Wan families do not.
    for repo, expected in (
        ("Lightricks/LTX-2", scale),
        ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", 1.0),
        ("Wan-AI/Wan2.2-T2V-A14B-Diffusers", 1.0),
    ):
        fam = detect_video_family(repo)
        assert _budget_scale(fam, base = repo) == expected, repo
    assert _budget_scale(detect_family("Qwen/Qwen-Image"), base = "Qwen/Qwen-Image") == scale


def test_a_sibling_release_keeps_the_pre_cast_encoder(monkeypatch):
    """The base gate must not refuse a release that republishes the SAME encoder.

    Qwen-Image-2512 and Krea-2-Raw ship their sibling's text encoder byte for byte (shard
    LFS sha256 compared 2026-08-25), so dropping the hosted pre-cast artifact for them
    would stage 16.6 GB / 8.9 GB of dense encoder the load never opens -- and would widen
    the memory budget that the pre-download unified-memory guard is sized against."""
    import core.inference.diffusion_precision as precision
    from core.inference.diffusion_families import detect_family_for_pick

    monkeypatch.setattr(precision, "te_quant_supported", lambda target, mode: True)
    for base in (
        "Qwen/Qwen-Image",
        "Qwen/Qwen-Image-2512",
        "unsloth/Qwen-Image-2512",
        "krea/Krea-2-Turbo",
        "krea/Krea-2-Raw",
    ):
        fam = detect_family_for_pick(base, None, None)
        assert fam is not None, base
        sources = tpq.te_prequant_sources_for_base(fam, base, te_quant_mode = "fp8", target = _target())
        assert "text_encoder" in sources, base
        assert _budget_scale(fam, base = base) == tpq.TE_PREQUANT_BUDGET_SCALE, base


def test_an_unrelated_custom_base_still_loses_it(monkeypatch):
    """The other half of the same gate: a base nobody has compared keeps the strict
    refusal, because the hosted artifact would otherwise download before its metadata
    could reject it."""
    import core.inference.diffusion_precision as precision
    from core.inference.diffusion_families import detect_family_for_pick

    monkeypatch.setattr(precision, "te_quant_supported", lambda target, mode: True)
    fam = detect_family_for_pick("Qwen/Qwen-Image", None, None)
    for base in ("someone/my-qwen-image-finetune", "randomuser/qwen-image-merged"):
        assert (
            tpq.te_prequant_sources_for_base(fam, base, te_quant_mode = "fp8", target = _target()) == {}
        ), base


def test_the_plan_recognises_the_pt_repos_that_already_exist(monkeypatch):
    """Preferring safetensors may only ADD a name, and the resolver is not the only reader.

    Every pre-cast encoder repo published so far hosts a ``.pt``. The download plan matched the
    PRIMARY name alone, so the moment the preferred spelling became safetensors it reported those
    repos as having no pre-cast encoder: the dense shards went back into the pull AND the loader
    still fetched the ``.pt``, so the user downloaded both. That is the opposite of what this
    change is for, and no test covered it because the plan lives beside the resolver, not in it.
    """
    src = TePrequantSource(
        kind = "repo",
        location = "unsloth/LTX-2-FP8",
        filename = "LTX-2-text_encoder-FP8.safetensors",
        fallback_filenames = ("LTX-2-text_encoder-FP8.pt",),
    )

    class _Sib:
        def __init__(self, name, size):
            self.rfilename, self.size = name, size

    class _Api:
        def __init__(self, files):
            self._files = files

        def model_info(self, *a, **k):
            outer = self

            class _I:
                siblings = outer._files

            return _I()

    def plan(hosted):
        return tpq.te_prequant_hub_files({"text_encoder": src}, _Api(hosted), None)

    pt, st = "LTX-2-text_encoder-FP8.pt", "LTX-2-text_encoder-FP8.safetensors"
    # A .pt-only repo: recognised, and sized from the file that will really be fetched.
    assert plan([_Sib(pt, 9_000_000_000)]) == {"text_encoder": [(pt, 9_000_000_000)]}
    # A safetensors-only repo: the artifact this change exists for.
    assert plan([_Sib(st, 9_400_000_000)]) == {"text_encoder": [(st, 9_400_000_000)]}
    # Both hosted: the preferred one wins, matching the order the resolver downloads in.
    assert plan([_Sib(pt, 9_000_000_000), _Sib(st, 9_400_000_000)]) == {
        "text_encoder": [(st, 9_400_000_000)]
    }
    # Neither: no pre-cast, so the dense encoder stays in the plan.
    assert plan([_Sib("unrelated.bin", 1)]) == {}

    # An install that cannot read safetensors takes the pickle rather than planning for a file it
    # would then refuse, which would leave the load with no encoder at all.
    monkeypatch.setattr(tpq, "te_candidate_is_readable", lambda n: bool(n) and n.endswith(".pt"))
    assert plan([_Sib(pt, 9_000_000_000), _Sib(st, 9_400_000_000)]) == {
        "text_encoder": [(pt, 9_000_000_000)]
    }


def test_the_te_capability_question_is_not_the_transformers_one():
    """A pre-cast encoder pickle is plain tensors read under a bare ``weights_only`` load, so it
    needs no constructor allowlist. Asking the DiT's question would refuse a ``.pt`` encoder on
    every install whose torchao lacks some DiT scheme's constructors."""
    assert tpq.te_candidate_is_readable("X-text_encoder-FP8.pt") is True
    assert tpq.te_candidate_is_readable(None) is False


def test_the_candidate_accessor_tolerates_a_planner_stand_in():
    """Planners pass lightweight objects carrying only ``filename``; reading the chain off one
    must not raise, or the whole pre-cast plan is swallowed into a silent dense fallback."""
    import types

    assert tpq.te_candidate_filenames(types.SimpleNamespace(filename = "a.pt")) == ("a.pt",)
    assert tpq.te_candidate_filenames(types.SimpleNamespace()) == ()


def test_an_unreachable_hub_is_not_a_missing_filename(monkeypatch):
    """``LocalEntryNotFoundError`` means two different things and only one of them is a miss.

    huggingface_hub documents it as "not on the disk when network is disabled OR UNAVAILABLE
    (connection issue). The entry may exist on the Hub", and it SUBCLASSES
    ``EntryNotFoundError``. Treating it as a candidate miss online spends a second full attempt on
    the next name and reports that one's error instead of the connection failure that happened.
    """
    import huggingface_hub
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    assert issubclass(
        LocalEntryNotFoundError, EntryNotFoundError
    ), "if this stops holding the ordering below is no longer load-bearing"
    src = TePrequantSource(
        kind = "repo",
        location = "org/hosted-fp8",
        filename = "hosted-text_encoder-FP8.safetensors",
        fallback_filenames = ("hosted-text_encoder-FP8.pt",),
    )
    asked: list = []

    def unreachable(**kw):
        asked.append(kw["filename"])
        raise LocalEntryNotFoundError("Hub unreachable")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", unreachable)
    # ONLINE: surfaces as itself, and the second candidate is never attempted.
    with pytest.raises(LocalEntryNotFoundError):
        tpq._resolve_checkpoint_path(src, None, cache_dir = "/tmp/x", local_files_only = False)
    assert asked == ["hosted-text_encoder-FP8.safetensors"], asked

    # OFFLINE: a cache miss is the only verdict there is, so the chain is walked.
    asked.clear()
    with pytest.raises(LocalEntryNotFoundError):
        tpq._resolve_checkpoint_path(src, None, cache_dir = "/tmp/x", local_files_only = True)
    assert asked == ["hosted-text_encoder-FP8.safetensors", "hosted-text_encoder-FP8.pt"], asked

    # A real 404 still advances online, which is the whole point of the chain.
    asked.clear()

    def only_pt(**kw):
        asked.append(kw["filename"])
        if kw["filename"].endswith(".safetensors"):
            raise EntryNotFoundError("404")
        return "/cache/hosted-text_encoder-FP8.pt"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", only_pt)
    got = tpq._resolve_checkpoint_path(src, None, cache_dir = "/tmp/x", local_files_only = False)
    assert got == "/cache/hosted-text_encoder-FP8.pt", got
    assert len(asked) == 2, asked


def test_the_video_prefetch_advances_only_on_a_missing_name():
    """The prefetch plan and the load have to agree about what a failure MEANS.

    The resolver advances to the next spelling only for "this name is absent"; if the prefetch
    advanced on an unreachable Hub too, it could report a legacy artifact as fetched, the plan would
    drop the dense encoder, and the load would then refuse to advance past the same error and have
    neither.
    """
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    from core.inference.video import VideoBackend

    miss = VideoBackend._te_fetch_miss
    # A real 404 is a miss in both modes: try the next name.
    assert miss(EntryNotFoundError("404"), local_files_only = False) is True
    assert miss(EntryNotFoundError("404"), local_files_only = True) is True
    # Offline, a cache miss is the only verdict there is.
    assert miss(LocalEntryNotFoundError("no local copy"), local_files_only = True) is True
    # Online, the same exception means the Hub could not be reached: stop, do not blame the name.
    assert miss(LocalEntryNotFoundError("connection error"), local_files_only = False) is False
    # Anything else is about the repo, not the name.
    assert miss(PermissionError("401"), local_files_only = False) is False
    assert miss(OSError("corrupt cache"), local_files_only = True) is False


# ── the family opt-in ────────────────────────────────────────────────────────────


def test_every_family_default_scheme_is_one_we_actually_host_for_that_family():
    """``te_quant_auto`` promises a scheme an UNSET request gets. If the family has no hosted
    pre-cast encoder for it, that promise costs a dense download and an in-place cast on every
    default load, which is the opposite of why the field exists. Catches a family opting in
    before its artifact is published, and a scheme/component pair that does not line up."""
    from core.inference.diffusion_families import _FAMILIES
    from core.inference.diffusion_te_prequant import family_te_prequant_repo

    offenders = []
    for fam in _FAMILIES:
        scheme = getattr(fam, "te_quant_auto", None)
        if scheme is None:
            continue
        if not any(
            family_te_prequant_repo(fam, scheme, component)
            for component in tpq.TE_PREQUANT_COMPONENTS
        ):
            offenders.append(
                f"{fam.name}: te_quant_auto={scheme!r} with no hosted {scheme} encoder"
            )
    assert not offenders, "\n  ".join(
        ["families default to an unhosted encoder scheme:", *offenders]
    )


def test_qwen_image_2_1_defaults_to_the_hosted_fp8_encoder():
    """The family this was built for. Its encoder (Qwen3-VL-8B, 16.33 GiB dense) is bigger than
    its INT8 denoiser (6.76 GiB), so leaving it dense is what made a quantised pick still cost
    ~26 GB. Named rather than covered only by the sweep above, because the whole change is
    pointless if this one row regresses."""
    from core.inference.diffusion_families import detect_family
    from core.inference.diffusion_te_prequant import resolve_te_prequant_source

    fam = detect_family("Qwen/Qwen-Image-2.1")
    assert fam.te_quant_auto == "fp8"
    source = resolve_te_prequant_source(fam, "text_encoder", "fp8")
    assert source is not None and source.kind == "repo"
    assert source.location == "unsloth/Qwen-Image-2.1-FP8"
    assert source.filename == "Qwen-Image-2.1-text_encoder-FP8.safetensors"
