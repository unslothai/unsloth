# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the native sd.cpp diffusion backend (the no-GPU engine)."""

from __future__ import annotations

import threading

import pytest
from PIL import Image

from core.inference import sd_cpp_backend as bk
from core.inference.diffusion_families import detect_family
from core.inference.sd_cpp_args import SdCppGenParams, SdCppModelFiles
from core.inference.sd_cpp_backend import (
    SdCppDiffusionBackend,
    _map_guidance,
    ensure_sd_cpp_binary,
)
from core.inference.sd_cpp_engine import SdCppCancelled


class _FakeEngine:
    """Stands in for SdCppEngine: writes a 1x1 PNG and records the args."""

    def __init__(
        self,
        *,
        fail = None,
        cancel_on_call = False,
    ):
        self.calls = []
        self.fail = fail
        self.cancel_on_call = cancel_on_call

    def is_available(self):
        return True

    def version(self, **_):
        return "fake sd-cli"

    def generate(
        self,
        files,
        params,
        *,
        output_path,
        cancel_event = None,
        **kw,
    ):
        self.calls.append((files, params, output_path, kw))
        if self.cancel_on_call and cancel_event is not None:
            cancel_event.set()
        if self.fail is not None:
            raise self.fail
        if cancel_event is not None and cancel_event.is_set():
            raise SdCppCancelled("cancelled")
        Image.new("RGB", (1, 1), (10, 20, 30)).save(output_path)
        from pathlib import Path

        return Path(output_path)


def _loaded_backend(fam_name = "z-image", engine = None):
    b = SdCppDiffusionBackend(engine = engine or _FakeEngine())
    fam = detect_family(fam_name)
    b._state = bk._SdState(
        repo_id = "unsloth/Z-Image-Turbo-GGUF",
        base_repo = fam.base_repo,
        family = fam,
        device = "cpu",
        files = SdCppModelFiles(
            diffusion_model = "/m/z.gguf", vae = "/m/vae.safetensors", llm = "/m/llm.safetensors"
        ),
        vae_format = fam.sd_cpp_vae_format,
        sampling_method = fam.sd_cpp_sampling_method,
        flow_shift = fam.sd_cpp_flow_shift,
    )
    return b


# ── asset resolution ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fam_name,expect_kinds",
    [
        ("flux.1", {"diffusion_model", "vae", "clip_l", "t5xxl"}),
        ("z-image", {"diffusion_model", "vae", "llm"}),
        ("qwen-image", {"diffusion_model", "vae", "qwen2vl"}),
        ("flux.2-klein", {"diffusion_model", "vae", "llm"}),
    ],
)
def test_asset_specs_cover_required_files(fam_name, expect_kinds):
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    fam = detect_family(fam_name)
    specs = b._asset_specs("unsloth/x-GGUF", "x-Q4_K_M.gguf", fam)
    kinds = {kind for _, _, kind in specs}
    assert kinds == expect_kinds
    # Every spec has a non-empty repo + filename.
    assert all(repo and fn for repo, fn, _ in specs)
    # The transformer reuses the requested GGUF, not a registry file.
    tr = [s for s in specs if s[2] == "diffusion_model"][0]
    assert tr[0] == "unsloth/x-GGUF" and tr[1] == "x-Q4_K_M.gguf"


# ── guidance mapping ──────────────────────────────────────────────────────────


def test_map_guidance_flux_uses_distilled_guidance():
    cfg, g = _map_guidance(detect_family("flux.1"), 3.5)
    assert cfg is None and g == 3.5


def test_map_guidance_cfg_family_off_when_distilled():
    # qwen-image uses real CFG; a distilled 0 -> CFG off (1.0), a >1 value passes through.
    assert _map_guidance(detect_family("qwen-image"), 0.0) == (1.0, None)
    assert _map_guidance(detect_family("qwen-image"), 4.0) == (4.0, None)


# ── status ────────────────────────────────────────────────────────────────────


def test_status_unloaded_reports_sd_cpp_engine():
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    st = b.status()
    assert st["loaded"] is False and st["engine"] == "sd_cpp"


def test_status_loaded_shape():
    b = _loaded_backend()
    st = b.status()
    assert st["loaded"] is True
    assert st["engine"] == "sd_cpp"
    assert st["family"] == "z-image"
    assert st["device"] == "cpu"
    # diffusers-only fields are present (route response parity) but null.
    for k in ("transformer_quant", "attention_backend", "transformer_cache", "text_encoder_quant"):
        assert st[k] is None


# ── generate ──────────────────────────────────────────────────────────────────


def test_generate_returns_images_and_seed():
    eng = _FakeEngine()
    b = _loaded_backend(engine = eng)
    out = b.generate(prompt = "a fox", width = 64, height = 64, steps = 8, seed = 123, batch_size = 2)
    assert out["seed"] == 123
    assert out["repo_id"] == "unsloth/Z-Image-Turbo-GGUF"
    assert len(out["images"]) == 2
    assert all(isinstance(im, Image.Image) for im in out["images"])
    # One sd-cli run per batch image, each a distinct seed from the base.
    assert len(eng.calls) == 2
    seeds = [params.seed for _, params, _, _ in eng.calls]
    assert seeds == [123, 124]
    # The per-image seeds are returned so the route can persist each one.
    assert out["seeds"] == [123, 124]


def test_generate_qwen_passes_sampling_args():
    eng = _FakeEngine()
    b = _loaded_backend(fam_name = "qwen-image", engine = eng)
    b.generate(prompt = "x", steps = 20, guidance = 4.0, seed = 1)
    _, params, _, kw = eng.calls[0]
    assert params.sampling_method == "euler"  # Qwen's supported sd.cpp sampler
    assert "--flow-shift" in (kw.get("extra_args") or [])


def test_generate_raises_when_not_loaded():
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    with pytest.raises(RuntimeError, match = "No diffusion model is loaded"):
        b.generate(prompt = "x")


def test_generate_passes_vae_format_for_flux2():
    eng = _FakeEngine()
    b = _loaded_backend(fam_name = "flux.2-klein", engine = eng)
    b.generate(prompt = "x", steps = 4, seed = 1)
    _, _, _, kw = eng.calls[0]
    assert kw.get("extra_args") == ["--vae-format", "flux2"]


def test_generate_cancellation_raises_cancelled_not_failure():
    # The engine cancels mid-run; the backend surfaces a cancellation, not a crash.
    eng = _FakeEngine(cancel_on_call = True)
    b = _loaded_backend(engine = eng)
    with pytest.raises(RuntimeError, match = "cancelled"):
        b.generate(prompt = "x", steps = 8, seed = 5)


def test_generate_progress_tracks_parsed_steps():
    b = _loaded_backend()
    b._gen = bk._SdGen(total_steps = 8)
    b._on_log("  sampling 4/8 done")
    p = b.generate_progress()
    assert p["active"] is True and p["step"] == 4 and p["total_steps"] == 8
    # A fraction with a different denominator must not move the bar.
    b._on_log("loaded 1/3 tensors")
    assert b.generate_progress()["step"] == 4


# ── load validation + binary install ──────────────────────────────────────────


def test_begin_load_rejects_unsupported_family(monkeypatch):
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    # A family with no native asset mapping must be rejected (router falls back).
    monkeypatch.setattr(bk, "family_sd_cpp_supported", lambda fam: False)
    with pytest.raises(ValueError, match = "no native sd.cpp asset mapping"):
        b.begin_load("unsloth/Z-Image-Turbo-GGUF", gguf_filename = "z.gguf")


def test_begin_load_requires_gguf_filename():
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    with pytest.raises(ValueError, match = "gguf_filename is required"):
        b.begin_load("unsloth/Z-Image-Turbo-GGUF")


def test_ensure_binary_returns_found(monkeypatch):
    monkeypatch.setattr(bk, "find_sd_cpp_binary", lambda: "/usr/bin/sd-cli")
    assert ensure_sd_cpp_binary() == "/usr/bin/sd-cli"


def test_ensure_binary_install_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(bk, "find_sd_cpp_binary", lambda: None)
    assert ensure_sd_cpp_binary(allow_install = False) is None


def test_unload_clears_state_and_signals_cancel():
    cancel = threading.Event()
    b = _loaded_backend()
    b._active_generate_cancel = cancel
    st = b.unload()
    assert st["loaded"] is False
    assert cancel.is_set()
    assert b._cancel_event.is_set()
