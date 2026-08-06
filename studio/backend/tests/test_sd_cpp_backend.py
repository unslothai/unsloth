# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the native sd.cpp diffusion backend (the no-GPU engine)."""

from __future__ import annotations

import threading
import types

import pytest
from PIL import Image

from core.inference import sd_cpp_backend as bk
from core.inference.diffusion_families import (
    _FLUX2_KLEIN_9B_SD_CPP_TEXT_ENCODERS,
    detect_family,
)
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
        mode = "oneshot",  # this fixture injects an engine, so it exercises the one-shot path
    )
    return b


def test_loaded_repo_ids_includes_native_companions():
    # The one-shot native engine re-reads its companion VAE / text-encoder files every generation, so the delete-cached guard
    # queries loaded_repo_ids(). It must surface the family's VAE + encoder repos, not just the GGUF, and be empty once unloaded.
    b = _loaded_backend("flux.1")
    ids = set(b.loaded_repo_ids())
    fam = detect_family("flux.1")
    assert "unsloth/Z-Image-Turbo-GGUF" in ids  # the main GGUF repo
    assert fam.base_repo in ids
    assert fam.sd_cpp_vae[0] in ids  # black-forest-labs/FLUX.1-schnell (VAE)
    for terepo, _f, _k in fam.sd_cpp_text_encoders:
        assert terepo in ids  # unsloth/flux-text-encoders
    b._state = None
    assert b.loaded_repo_ids() == ()


def test_loaded_repo_ids_tracks_variant_encoder_by_gguf_filename():
    # A local *klein-9B*.gguf carries the variant keyword only in the basename, so loaded_repo_ids() must include the filename or the guard protects the wrong repo.
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    fam = detect_family("flux.2-klein")
    b._state = bk._SdState(
        repo_id = "local/my-klein-checkpoints",  # no variant keyword; it lives in the filename
        base_repo = fam.base_repo,
        family = fam,
        device = "cpu",
        files = SdCppModelFiles(diffusion_model = "/m/FLUX.2-klein-9B-Q4_K_M.gguf"),
        vae_format = fam.sd_cpp_vae_format,
        sampling_method = fam.sd_cpp_sampling_method,
        flow_shift = fam.sd_cpp_flow_shift,
        mode = "oneshot",
        gguf_filename = "FLUX.2-klein-9B-Q4_K_M.gguf",
    )
    ids = set(b.loaded_repo_ids())
    assert "unsloth/FLUX.2-klein-9B-ComfyUI" in ids  # the 8B encoder this load pulled
    # The 4B default must not be protected instead.
    assert "unsloth/Z-Image-Turbo-ComfyUI" not in ids


class _FakeServer:
    """Stands in for SdCppServer: records the spawn + one img_gen per whole batch."""

    def __init__(self, binary):
        self.binary = binary
        self.started = None
        self.stopped = False
        self.payloads = []
        self.timeouts = []
        self.alive = True
        self.lora_dir = None  # set by a test to the server's --lora-model-dir scratch dir
        # Raised by the next img_gen (one shot) so a test can stage a mid-generation server death.
        self.img_gen_error = None

    def is_alive(self):
        return self.alive and not self.stopped

    def start(
        self,
        files,
        *,
        vae_format = None,
        offload = None,
        native_speed = None,
        threads = None,
        extra_args = None,
    ):
        self.started = dict(
            files = files,
            vae_format = vae_format,
            offload = offload,
            native_speed = native_speed,
            threads = threads,
            extra_args = list(extra_args or []),
        )

    def img_gen(
        self,
        payload,
        *,
        on_step = None,
        cancel_event = None,
        total_timeout = None,
    ):
        import io as _io

        self.payloads.append(payload)
        self.timeouts.append(total_timeout)
        if self.img_gen_error is not None:
            err, self.img_gen_error = self.img_gen_error, None
            self.alive = False  # a ggml abort takes the process down
            raise err
        if on_step is not None:
            steps = payload.get("sample_params", {}).get("sample_steps", 0)
            on_step(f"  {steps}/{steps}")
        n = int(payload.get("batch_count", 1))
        blobs = []
        for i in range(n):
            buf = _io.BytesIO()
            Image.new("RGB", (1, 1), (i, i, i)).save(buf, format = "PNG")
            blobs.append(buf.getvalue())
        return blobs

    def stop(self):
        self.stopped = True


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


def test_download_plan_stages_exactly_what_sd_cli_opens(monkeypatch):
    # The plan feeds the Hub download manager. Native reads single-file assets, so a native-routed pick must be staged from the asset specs.
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    sizes = {
        ("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q4_K_M.gguf"): 4_000,
        ("unsloth/Z-Image-Turbo-ComfyUI", "split_files/vae/ae.safetensors"): 300,
        ("unsloth/Z-Image-Turbo-ComfyUI", "split_files/text_encoders/qwen_3_4b.safetensors"): 8_000,
    }
    monkeypatch.setattr(
        SdCppDiffusionBackend,
        "_plan_file_sizes",
        staticmethod(lambda by_repo, token: sizes),
    )

    plan = b.download_plan(
        "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z-image-turbo-Q4_K_M.gguf",
        model_kind = "gguf",
        # diffusers-only knobs are accepted and ignored, exactly as begin_load accepts them.
        transformer_quant = "int8",
        memory_mode = "low_vram",
    )

    fam = detect_family("z-image")
    expected = {
        (r, f)
        for r, f, _k in b._asset_specs(
            "unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q4_K_M.gguf", fam
        )
    }
    listed = {(e["repo_id"], f) for e in plan["entries"] for f in e["files"]}
    assert listed == expected
    assert plan["total_bytes"] == 12_300
    # The transformer entry is the only one carrying the GGUF filename; the VAE + encoder share one repo entry.
    tr = [e for e in plan["entries"] if e["gguf_filename"]]
    assert len(tr) == 1 and tr[0]["repo_id"] == "unsloth/Z-Image-Turbo-GGUF"
    assert len([e for e in plan["entries"] if e["repo_id"] == "unsloth/Z-Image-Turbo-ComfyUI"]) == 1


def test_download_plan_skips_a_local_transformer_but_still_stages_the_assets(monkeypatch, tmp_path):
    # A local GGUF folder is already on disk; its VAE + encoder still have to come from the Hub.
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    (tmp_path / "z-image-turbo-Q4_K_M.gguf").write_bytes(b"gguf")
    monkeypatch.setattr(
        SdCppDiffusionBackend, "_plan_file_sizes", staticmethod(lambda by_repo, token: {})
    )

    plan = b.download_plan(
        str(tmp_path), gguf_filename = "z-image-turbo-Q4_K_M.gguf", model_kind = "gguf"
    )

    assert str(tmp_path) not in {e["repo_id"] for e in plan["entries"]}
    assert {e["repo_id"] for e in plan["entries"]} == {"unsloth/Z-Image-Turbo-ComfyUI"}
    # An unreadable size understates the total; it must never fail the plan.
    assert plan["total_bytes"] == 0


def _no_cache(monkeypatch):
    """Report every upstream as uncached, so a local cache cannot mask the mirror decision."""
    monkeypatch.setattr(
        "core.inference.diffusion_families._upstream_is_cached", lambda repo_id, files = None: False
    )
    monkeypatch.delenv("UNSLOTH_DIFFUSION_NO_MIRROR", raising = False)


def test_download_plan_stages_the_mirrored_asset_repo(monkeypatch):
    """STAGED before the load runs, so a gated asset repo left here 401s an anonymous user and
    _fetch_assets' swap is never reached. FLUX.1's VAE lives in the gated FLUX.1-schnell."""
    _no_cache(monkeypatch)
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    monkeypatch.setattr(
        SdCppDiffusionBackend, "_plan_file_sizes", staticmethod(lambda by_repo, token: {})
    )

    plan = b.download_plan(
        "unsloth/FLUX.1-dev-GGUF", gguf_filename = "flux1-dev-Q4_K_M.gguf", model_kind = "gguf"
    )

    staged = {e["repo_id"] for e in plan["entries"]}
    assert "unsloth/FLUX.1-schnell" in staged
    assert "black-forest-labs/FLUX.1-schnell" not in staged
    # The GGUF entry is untouched and still the only one carrying the filename.
    tr = [e for e in plan["entries"] if e["gguf_filename"]]
    assert len(tr) == 1 and tr[0]["repo_id"] == "unsloth/FLUX.1-dev-GGUF"


def test_download_plan_and_fetch_assets_pick_the_same_repo(monkeypatch):
    """Staging one repo and then downloading from the other is the failure this feature removes,
    so both sides take the decision from the same per-repo file list."""
    _no_cache(monkeypatch)
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    monkeypatch.setattr(
        SdCppDiffusionBackend, "_plan_file_sizes", staticmethod(lambda by_repo, token: {})
    )
    pulled: list = []
    monkeypatch.setattr(
        "utils.hf_xet_fallback.hf_hub_download_with_xet_fallback",
        lambda repo, fn, tok, **k: (pulled.append((repo, fn)), f"/cache/{fn}")[1],
    )

    fam = detect_family("flux.1")
    specs = b._asset_specs("unsloth/FLUX.1-dev-GGUF", "flux1-dev-Q4_K_M.gguf", fam)
    b._fetch_assets(specs, None)
    plan = b.download_plan(
        "unsloth/FLUX.1-dev-GGUF", gguf_filename = "flux1-dev-Q4_K_M.gguf", model_kind = "gguf"
    )

    assert {(e["repo_id"], f) for e in plan["entries"] for f in e["files"]} == set(pulled)


def test_delete_guard_covers_the_mirrored_asset_repos(monkeypatch):
    """The bytes land under whichever of the pair was fetched, so guarding only the upstream leaves
    the mirror cache deletable under a one-shot sd-cli that re-reads it."""
    b = _loaded_backend("flux.1")
    ids = set(b.loaded_repo_ids())
    assert "black-forest-labs/FLUX.1-schnell" in ids
    assert "unsloth/FLUX.1-schnell" in ids

    b._state = None
    b._loading = bk._SdLoading(
        repo_id = "unsloth/FLUX.1-dev-GGUF",
        base_repo = "black-forest-labs/FLUX.1-dev",
        asset_repos = ("black-forest-labs/FLUX.1-schnell",),
    )
    loading = set(b.loading_repo_ids())
    assert {"black-forest-labs/FLUX.1-schnell", "unsloth/FLUX.1-schnell"} <= loading
    assert {"black-forest-labs/FLUX.1-dev", "unsloth/FLUX.1-dev"} <= loading


def test_download_plan_refuses_a_pick_native_cannot_serve():
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    with pytest.raises(ValueError, match = "gguf_filename is required"):
        b.download_plan("unsloth/Z-Image-Turbo-GGUF", model_kind = "gguf")
    with pytest.raises(ValueError, match = "native sd.cpp asset mapping"):
        b.download_plan(
            "stabilityai/sdxl-turbo", gguf_filename = "sdxl-Q4_K_M.gguf", model_kind = "gguf"
        )


def test_asset_specs_flux2_klein_selects_encoder_by_variant():
    # FLUX.2-klein 4B pairs with Qwen3-4B, 9B with Qwen3-8B, so the encoder must come from the load identity, not the family default.
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    fam = detect_family("flux.2-klein")

    specs_4b = b._asset_specs("unsloth/FLUX.2-klein-4B-GGUF", "FLUX.2-klein-4B-Q4_K_M.gguf", fam)
    te_4b = [(r, f) for r, f, k in specs_4b if k == "llm"]
    assert te_4b == [
        ("unsloth/Z-Image-Turbo-ComfyUI", "split_files/text_encoders/qwen_3_4b.safetensors")
    ]

    specs_9b = b._asset_specs("unsloth/FLUX.2-klein-9B-GGUF", "FLUX.2-klein-9B-Q4_K_M.gguf", fam)
    te_9b = [(r, f) for r, f, k in specs_9b if k == "llm"]
    assert te_9b == [
        (
            "unsloth/FLUX.2-klein-9B-ComfyUI",
            "split_files/text_encoders/qwen_3_8b.safetensors",
        )
    ]


# A rename or takedown in a community repack breaks every no-GPU load needing the file, so each
# one Studio depended on now has a byte-identical unsloth mirror.
_REPACKER_ORGS = frozenset(
    {"comfy-org", "comfyanonymous", "quantstack", "city96", "calcuis", "orabazes"}
)


def test_no_sd_cpp_asset_comes_from_a_community_repack():
    # Vendor repos are fine (they are the source of truth); repacks are not, so assert on the org.
    from core.inference.diffusion_families import _FAMILIES

    offenders = []
    for fam in _FAMILIES:
        specs = list(fam.sd_cpp_text_encoders)
        if fam.sd_cpp_vae:
            specs.append((*fam.sd_cpp_vae, "vae"))
        specs.extend(_FLUX2_KLEIN_9B_SD_CPP_TEXT_ENCODERS)
        for repo, _f, _k in specs:
            if repo.split("/", 1)[0].lower() in _REPACKER_ORGS:
                offenders.append((fam.name, repo))
    assert offenders == []


def test_the_moved_sd_cpp_assets_keep_their_upstream_relative_paths():
    # The mirrors kept the upstream paths, so the swap is an id change only; a reorganised mirror
    # would 404 sd-cli at load time.
    want = {
        "flux.1": [
            ("unsloth/flux-text-encoders", "clip_l.safetensors"),
            ("unsloth/flux-text-encoders", "t5xxl_fp16.safetensors"),
        ],
        "flux.2-klein": [
            ("unsloth/Z-Image-Turbo-ComfyUI", "split_files/text_encoders/qwen_3_4b.safetensors"),
        ],
        "flux.2-dev": [
            (
                "unsloth/FLUX.2-dev-ComfyUI",
                "split_files/text_encoders/mistral_3_small_flux2_bf16.safetensors",
            ),
        ],
        "z-image": [
            ("unsloth/Z-Image-Turbo-ComfyUI", "split_files/text_encoders/qwen_3_4b.safetensors"),
        ],
    }
    for name, encoders in want.items():
        fam = detect_family(name)
        assert [(r, f) for r, f, _k in fam.sd_cpp_text_encoders] == encoders, name
    assert detect_family("qwen-image").sd_cpp_vae == (
        "unsloth/Qwen-Image-ComfyUI",
        "split_files/vae/qwen_image_vae.safetensors",
    )
    assert detect_family("z-image").sd_cpp_vae == (
        "unsloth/Z-Image-Turbo-ComfyUI",
        "split_files/vae/ae.safetensors",
    )
    # The FLUX.2 autoencoder is Apache-2.0 while the conditioner beside it in the source repack is
    # not, so it is mirrored on its own.
    for name in ("flux.2-klein", "flux.2-dev"):
        assert detect_family(name).sd_cpp_vae == (
            "unsloth/FLUX.2-VAE",
            "split_files/vae/flux2-vae.safetensors",
        ), name


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


def test_generate_publishes_progress_before_lora_resolution(monkeypatch):
    # LoRA resolution runs during pre-generate setup while _generate_lock is held, so a progress probe then must read ACTIVE.
    from core.inference import diffusion_lora

    eng = _FakeEngine()
    b = _loaded_backend(engine = eng)
    monkeypatch.setattr(diffusion_lora, "supports_lora", lambda **_k: True)

    seen: dict = {}

    def _resolve(
        active,
        *,
        family = None,
        hf_token = None,
        cancel_event = None,
    ):
        # Mid-setup: the in-flight generation must already be reported as active.
        seen["progress"] = b.generate_progress()
        return []

    monkeypatch.setattr(diffusion_lora, "resolve_specs", _resolve)

    out = b.generate(prompt = "a fox", width = 64, height = 64, steps = 8, loras = [("some/lora", 1.0)])
    assert out["images"]
    assert seen["progress"]["active"] is True
    assert seen["progress"]["total_steps"] == 8


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


def test_begin_load_resolves_family_from_filename_only(monkeypatch):
    # A local .gguf pick whose family keyword lives only in the basename must resolve via the same filename fallback the route used.
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    monkeypatch.setattr(b, "_run_load", lambda **kwargs: None)  # skip the download thread
    b.begin_load("/models/gguf-store", gguf_filename = "Z-Image-Turbo-Q4_K_M.gguf")
    # Validation passed (no ValueError) and the family was inferred from the filename.
    assert b._loading is not None and b._loading.repo_id == "/models/gguf-store"


def test_each_load_owns_its_cancel_event(monkeypatch):
    # Same contract as the diffusers backend: unload() cancels the running asset pull by setting the event that worker holds and
    # drops _loading. A clear() of one shared event would un-cancel it; a fresh Event per load leaves the superseded worker cancelled.
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    started = threading.Event()
    seen: list[threading.Event] = []

    def _capture(**kwargs):
        seen.append(kwargs["_cancel_event"])
        started.set()

    monkeypatch.setattr(b, "_run_load", _capture)  # skip the download thread's work

    b.begin_load("unsloth/Z-Image-Turbo-GGUF", gguf_filename = "z.gguf")
    assert started.wait(5)
    first = seen[0]
    b.unload()
    assert first.is_set()

    started.clear()
    b.begin_load("unsloth/Z-Image-Turbo-GGUF", gguf_filename = "z.gguf")
    assert started.wait(5)
    second = seen[1]
    assert second is not first, "each load needs its own event, not a clear() of the shared one"
    assert not second.is_set()
    assert first.is_set(), "the superseded worker's event must stay set"
    # The superseded worker's fetch bails on ITS event, not on the live one.
    with pytest.raises(SdCppCancelled):
        b._fetch_assets(
            [("unsloth/Z-Image-Turbo-GGUF", "z.gguf", "diffusion_model")],
            None,
            cancel_event = first,
        )


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


def test_status_reports_offload_when_flags_active():
    # status must reflect the offload flags actually passed to sd-cli, not always "none".
    b = _loaded_backend()
    # No flags (CPU default) -> none.
    assert b.status()["offload_policy"] == "none" and b.status()["cpu_offload"] is False
    # Flags present (off-CPU offload) -> reported active.
    s = b._state
    b._state = bk._SdState(
        repo_id = s.repo_id,
        base_repo = s.base_repo,
        family = s.family,
        device = "cuda",
        files = s.files,
        offload_flags = ("--vae-on-cpu", "--clip-on-cpu"),
    )
    st = b.status()
    assert st["cpu_offload"] is True and st["offload_policy"] == "active"


def test_run_load_cancels_and_waits_for_inflight_generation(monkeypatch):
    # A generation started during the asset download still runs against the OLD model, so _run_load must cancel it AND wait on _generate_lock before committing.
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    fam = detect_family("z-image")
    monkeypatch.setattr(b, "_asset_specs", lambda *a, **k: [])
    monkeypatch.setattr(b, "_set_expected_bytes", lambda *a, **k: None)
    monkeypatch.setattr(
        b,
        "_fetch_assets",
        lambda *a, **k: {"diffusion_model": "/m/z.gguf", "vae": "/m/vae.sft", "llm": "/m/llm.sft"},
    )
    # Avoid importing torch from the worker thread (its first import deadlocks off the main thread); the device only needs to be CPU.
    monkeypatch.setattr(
        bk, "resolve_diffusion_device_target", lambda: types.SimpleNamespace(device = "cpu")
    )

    b._load_token = 5
    cancel = threading.Event()
    b._active_generate_cancel = cancel  # a generation is "in flight"

    committed = threading.Event()

    def _load():
        b._run_load(
            repo_id = "unsloth/Z-Image-Turbo-GGUF",
            gguf_filename = "z.gguf",
            base = fam.base_repo,
            fam = fam,
            hf_token = None,
            _load_token = 5,
        )
        committed.set()

    b._generate_lock.acquire()  # simulate the live denoise holding _generate_lock
    try:
        threading.Thread(target = _load, daemon = True).start()
        # The commit must block behind the live generation and not publish, but must already have signalled the cancel.
        assert not committed.wait(0.5)
        assert b._state is None
        assert cancel.is_set()
    finally:
        b._generate_lock.release()
    assert committed.wait(5)  # only now does the commit run
    assert b._state is not None and b._state.repo_id == "unsloth/Z-Image-Turbo-GGUF"


# ── persistent sd-server mode ──────────────────────────────────────────────────


def test_resolve_backend_prefers_server(monkeypatch):
    b = SdCppDiffusionBackend()  # no injected engine
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: "/x/sd-server")
    mode, binary, engine = b._resolve_backend()
    assert mode == "server" and binary == "/x/sd-server" and engine is None


def test_resolve_backend_injected_engine_forces_oneshot():
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    mode, binary, engine = b._resolve_backend()
    assert mode == "oneshot" and binary is None and engine is not None


def test_resolve_backend_falls_back_to_oneshot_without_server(monkeypatch):
    b = SdCppDiffusionBackend()
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: None)
    monkeypatch.setattr(bk, "_install_allowed", lambda: False)  # don't attempt a real install
    monkeypatch.setattr(bk, "find_sd_cpp_binary", lambda: "/usr/bin/sd-cli")
    mode, binary, engine = b._resolve_backend()
    assert mode == "oneshot" and engine is not None


def test_resolve_backend_cached_fallback_engine_does_not_pin_oneshot(monkeypatch):
    # A lazily cached fallback engine (NOT an explicit injection) must not force one-shot: a later load can use a server again.
    b = SdCppDiffusionBackend()  # no injected engine
    b._engine = _FakeEngine()  # simulate a prior lazy one-shot fallback caching the engine
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: "/x/sd-server")
    mode, binary, engine = b._resolve_backend()
    assert mode == "server" and binary == "/x/sd-server" and engine is None


def _run_server_load(
    monkeypatch,
    b,
    servers,
    fam_name = "z-image",
    device = "cpu",
):
    fam = detect_family(fam_name)
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: "/x/sd-server")
    # The fake binary path is not a real executable; skip the up-front runnability probe.
    monkeypatch.setattr(bk, "_server_binary_runnable", lambda *_a, **_k: True)

    def _factory(binary):
        s = _FakeServer(binary)
        servers.append(s)
        return s

    monkeypatch.setattr(bk, "SdCppServer", _factory)
    monkeypatch.setattr(b, "_asset_specs", lambda *a, **k: [])
    monkeypatch.setattr(b, "_set_expected_bytes", lambda *a, **k: None)
    monkeypatch.setattr(
        b,
        "_fetch_assets",
        lambda *a, **k: {"diffusion_model": "/m/z.gguf", "vae": "/m/vae.sft", "llm": "/m/llm.sft"},
    )
    monkeypatch.setattr(
        bk, "resolve_diffusion_device_target", lambda: types.SimpleNamespace(device = device)
    )
    b._load_token = 1
    b._run_load(
        repo_id = "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z.gguf",
        base = fam.base_repo,
        fam = fam,
        hf_token = None,
        _load_token = 1,
    )


def test_server_load_spawns_once_and_status_reports_mode(monkeypatch):
    b = SdCppDiffusionBackend()
    servers: list = []
    _run_server_load(monkeypatch, b, servers)
    assert len(servers) == 1
    assert servers[0].started is not None  # the model is loaded once, at spawn
    assert b._state is not None and b._state.mode == "server" and b._state.server is servers[0]
    assert b.status()["native_mode"] == "server"


def test_server_generate_uses_one_request_for_whole_batch(monkeypatch):
    b = SdCppDiffusionBackend()
    servers: list = []
    _run_server_load(monkeypatch, b, servers)
    out = b.generate(prompt = "a fox", width = 64, height = 64, steps = 8, seed = 7, batch_size = 3)
    assert len(out["images"]) == 3
    assert all(isinstance(im, Image.Image) for im in out["images"])
    # ONE job for the whole batch (no per-image model reload), unlike the one-shot path.
    assert len(servers[0].payloads) == 1
    assert servers[0].payloads[0]["batch_count"] == 3
    assert out["seed"] == 7 and out["seeds"] == [7, 8, 9]
    # step progress was driven from the server's stdout line.
    assert b._gen is None  # cleared after generate


_GGML_ABORT = (
    "sd-server connection lost during img_gen poll (process exited, code -6)\n"
    "Last output:\n"
    "[ERROR] ggml_extend.hpp:70   - ggml_metal_op_encode_impl: error: unsupported op 'MUL_MAT'\n"
    "1   sd-server   0x00000001044f8df4 ggml_abort + 156\n"
    "10  sd-server   0x00000001043f6ce8 StableDiffusionGGML::sample"
)


def test_server_generation_restarts_on_the_cpu_backend_after_a_ggml_abort(monkeypatch):
    # ggml calls GGML_ABORT on an unimplemented op, killing sd-server mid-generation with no per-op CPU fallback, so the load is restarted with --backend cpu instead of failing.
    b = SdCppDiffusionBackend()
    servers: list = []
    _run_server_load(monkeypatch, b, servers, device = "mps")
    servers[0].img_gen_error = RuntimeError(_GGML_ABORT)

    out = b.generate(prompt = "a fox", width = 64, height = 64, steps = 4, seed = 3)

    assert len(out["images"]) == 1  # the retry produced the image
    assert len(servers) == 2 and servers[0].stopped is True
    assert servers[1].started["extra_args"] == ["--backend", "cpu"]
    # The same checkpoint and run settings are reused; only the backend placement changed.
    assert servers[1].started["files"] is servers[0].started["files"]
    assert servers[1].started["native_speed"] == servers[0].started["native_speed"]
    # The live state points at the replacement, so the next generation does not touch the dead one.
    assert b._state is not None and b._state.server is servers[1]


def test_cpu_backend_restart_happens_once_per_load(monkeypatch):
    # The restart is a one-shot rescue: if the CPU backend aborts too, the error surfaces rather than spawning servers forever.
    b = SdCppDiffusionBackend()
    servers: list = []
    _run_server_load(monkeypatch, b, servers, device = "mps")
    servers[0].img_gen_error = RuntimeError(_GGML_ABORT)
    b.generate(prompt = "x", steps = 4, seed = 1)
    assert len(servers) == 2

    servers[1].img_gen_error = RuntimeError(_GGML_ABORT)
    with pytest.raises(RuntimeError, match = "unsupported op"):
        b.generate(prompt = "x", steps = 4, seed = 1)
    assert len(servers) == 2  # no third spawn


def test_server_death_without_the_abort_signature_is_not_retried(monkeypatch):
    # An OOM kill, a corrupt checkpoint or a genuine bug must not be silently retried on another backend: only the unsupported-op abort earns the CPU restart.
    b = SdCppDiffusionBackend()
    servers: list = []
    _run_server_load(monkeypatch, b, servers, device = "mps")
    servers[0].img_gen_error = RuntimeError(
        "sd-server connection lost during img_gen poll (process exited, code -9)"
    )
    with pytest.raises(RuntimeError, match = "code -9"):
        b.generate(prompt = "x", steps = 4, seed = 1)
    assert len(servers) == 1


def test_cpu_device_does_not_restart_on_an_abort(monkeypatch):
    # Already on CPU: the abort is not a backend-placement problem, so restarting would just repeat it. Surface the error instead.
    b = SdCppDiffusionBackend()
    servers: list = []
    _run_server_load(monkeypatch, b, servers, device = "cpu")
    servers[0].img_gen_error = RuntimeError(_GGML_ABORT)
    with pytest.raises(RuntimeError, match = "unsupported op"):
        b.generate(prompt = "x", steps = 4, seed = 1)
    assert len(servers) == 1


def test_server_generate_splits_batches_above_server_limit(monkeypatch):
    # A batch above the server's per-job limit is chunked; each chunk gets a timeout proportional to its image count.
    b = SdCppDiffusionBackend()
    servers: list = []
    _run_server_load(monkeypatch, b, servers)
    out = b.generate(prompt = "x", width = 64, height = 64, steps = 4, seed = 100, batch_size = 10)
    assert len(out["images"]) == 10
    counts = [p["batch_count"] for p in servers[0].payloads]
    assert counts == [bk._MAX_SERVER_BATCH, 10 - bk._MAX_SERVER_BATCH]  # [8, 2]
    # Chunks share ONE request deadline rather than each getting a full budget: a batch is split only because the server caps
    # images per job, so per-chunk budgets would let it outlive the window the page is waiting on. Each gets what is left.
    assert servers[0].timeouts[0] <= bk.NATIVE_GENERATION_TIMEOUT_S
    assert servers[0].timeouts[1] <= servers[0].timeouts[0]
    # A single slow image can still use the whole window (the old per-image cap was 30 minutes).
    assert servers[0].timeouts[-1] > 1800.0
    # Seeds run contiguously across chunks (chunk 2 submitted at base + 8).
    assert out["seeds"] == list(range(100, 110))
    assert servers[0].payloads[1]["seed"] == 108


def test_server_generate_masks_large_seed(monkeypatch):
    # sd.cpp's image seed is signed int64, so a larger explicit seed must be masked before it reaches the server.
    b = SdCppDiffusionBackend()
    servers: list = []
    _run_server_load(monkeypatch, b, servers)
    out = b.generate(prompt = "x", width = 64, height = 64, steps = 4, seed = 2**64 - 1, batch_size = 1)
    assert servers[0].payloads[0]["seed"] <= (1 << 63) - 1
    assert all(s <= (1 << 63) - 1 for s in out["seeds"])


def test_status_clears_when_server_died(monkeypatch):
    b = SdCppDiffusionBackend()
    servers: list = []
    _run_server_load(monkeypatch, b, servers)
    assert b.status()["loaded"] is True
    servers[0].alive = False  # the resident server crashed / was OOM-killed
    st = b.status()
    assert st["loaded"] is False
    assert b._state is None  # stale state was dropped so clients reload


def test_server_generate_progress_from_stdout(monkeypatch):
    b = SdCppDiffusionBackend()
    servers: list = []
    _run_server_load(monkeypatch, b, servers)

    seen = {}

    class _WatchServer(_FakeServer):
        def img_gen(
            self,
            payload,
            *,
            on_step = None,
            cancel_event = None,
            total_timeout = None,
        ):
            on_step("  4/8")
            seen["mid"] = b.generate_progress()
            return super().img_gen(
                payload, on_step = on_step, cancel_event = cancel_event, total_timeout = total_timeout
            )

    b._state = bk._SdState(
        repo_id = b._state.repo_id,
        base_repo = b._state.base_repo,
        family = b._state.family,
        device = b._state.device,
        files = b._state.files,
        vae_format = b._state.vae_format,
        sampling_method = b._state.sampling_method,
        flow_shift = b._state.flow_shift,
        server = _WatchServer("/x/sd-server"),
        mode = "server",
    )
    b.generate(prompt = "x", steps = 8, seed = 1)
    assert seen["mid"]["step"] == 4 and seen["mid"]["total_steps"] == 8


def test_server_unload_stops_server(monkeypatch):
    b = SdCppDiffusionBackend()
    servers: list = []
    _run_server_load(monkeypatch, b, servers)
    st = b.unload()
    assert st["loaded"] is False
    assert servers[0].stopped is True
    assert b._state is None


def test_server_reload_stops_old_server_before_new(monkeypatch):
    b = SdCppDiffusionBackend()
    servers: list = []
    _run_server_load(monkeypatch, b, servers)
    # A second load must tear down the first server and start a fresh one.
    b._load_token = 2
    fam = detect_family("z-image")
    b._run_load(
        repo_id = "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z.gguf",
        base = fam.base_repo,
        fam = fam,
        hf_token = None,
        _load_token = 2,
    )
    assert len(servers) == 2
    assert servers[0].stopped is True  # old server stopped
    assert b._state.server is servers[1] and servers[1].stopped is False


def test_server_start_failure_falls_back_to_oneshot(monkeypatch):
    # A present-but-broken sd-server must not fail the load when sd-cli works.
    b = SdCppDiffusionBackend()
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: "/x/sd-server")
    # The probe passes; the failure exercised here is in start(), not the up-front probe.
    monkeypatch.setattr(bk, "_server_binary_runnable", lambda *_a, **_k: True)

    class _BadServer:
        def __init__(self, binary):
            self.stopped = False

        def start(self, *a, **k):
            raise RuntimeError("sd-server broken")

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(bk, "SdCppServer", _BadServer)
    fake = _FakeEngine()
    monkeypatch.setattr(b, "_resolve_engine", lambda: fake)
    monkeypatch.setattr(b, "_asset_specs", lambda *a, **k: [])
    monkeypatch.setattr(b, "_set_expected_bytes", lambda *a, **k: None)
    monkeypatch.setattr(
        b,
        "_fetch_assets",
        lambda *a, **k: {"diffusion_model": "/m/z.gguf", "vae": "/m/vae.sft", "llm": "/m/llm.sft"},
    )
    monkeypatch.setattr(
        bk, "resolve_diffusion_device_target", lambda: types.SimpleNamespace(device = "cpu")
    )
    fam = detect_family("z-image")
    b._load_token = 1
    b._run_load(
        repo_id = "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z.gguf",
        base = fam.base_repo,
        fam = fam,
        hf_token = None,
        _load_token = 1,
    )
    assert b._state is not None and b._state.mode == "oneshot" and b._state.server is None
    # and it can still generate via the one-shot engine
    out = b.generate(prompt = "x", steps = 4, seed = 1)
    assert len(out["images"]) == 1 and len(fake.calls) == 1


def test_run_load_redacts_paths_in_progress_error(monkeypatch):
    # A load failure surfaced via load_progress() must run through redact_native_paths, the same scrub the diffusers path applies.
    from utils import native_path_leases as npl

    secret_root = "/managed/native/root"
    npl._remember_native_path_for_redaction(secret_root, "model dir")
    try:
        b = SdCppDiffusionBackend(engine = _FakeEngine())
        fam = detect_family("z-image")
        monkeypatch.setattr(b, "_asset_specs", lambda *a, **k: [])
        monkeypatch.setattr(b, "_set_expected_bytes", lambda *a, **k: None)

        def _boom(*a, **k):
            raise RuntimeError(f"failed to read {secret_root}/z.gguf")

        monkeypatch.setattr(b, "_fetch_assets", _boom)

        b._load_token = 1
        b._loading = bk._SdLoading(repo_id = "unsloth/Z-Image-Turbo-GGUF", base_repo = fam.base_repo)
        b._run_load(
            repo_id = "unsloth/Z-Image-Turbo-GGUF",
            gguf_filename = "z.gguf",
            base = fam.base_repo,
            fam = fam,
            hf_token = None,
            _load_token = 1,
        )
        err = b.load_progress()["error"]
        assert err and secret_root not in err and "<native_path>" in err
    finally:
        with npl._REDACTION_LOCK:
            if secret_root in npl._NATIVE_PATH_REDACTIONS:
                npl._NATIVE_PATH_REDACTIONS.remove(secret_root)


# ── LoRA (native engine) ────────────────────────────────────────────────────────


def _fake_materialize(resolved, dest):
    """Stand-in for diffusion_lora.materialize_native_dir: write a stub file per adapter
    into ``dest`` and return the resolved list pointing at the written paths (mirroring the
    real helper's contract without touching the Hub / real weights)."""
    from pathlib import Path as _P

    from core.inference import diffusion_lora as dl

    dest.mkdir(parents = True, exist_ok = True)
    out = []
    for r in resolved:
        p = _P(dest) / f"{r.alias}.safetensors"
        p.write_bytes(b"stub")
        out.append(dl.ResolvedLora(r.id, r.alias, str(p), r.fmt, r.weight))
    return out


def _patch_lora(
    monkeypatch,
    resolved,
    supported = True,
):
    from core.inference import diffusion_lora as dl

    monkeypatch.setattr(dl, "supports_lora", lambda **k: supported)
    monkeypatch.setattr(dl, "resolve_specs", lambda specs, **k: list(resolved))
    monkeypatch.setattr(dl, "materialize_native_dir", _fake_materialize)


def test_generate_oneshot_applies_loras_via_prompt_tags(monkeypatch):
    # One-shot sd-cli LoRA: adapters materialized into a --lora-model-dir and selected with <lora:ALIAS:w> prompt tags.
    from core.inference import diffusion_lora as dl

    eng = _FakeEngine()
    b = _loaded_backend(engine = eng)  # mode = "oneshot"
    _patch_lora(
        monkeypatch, [dl.ResolvedLora("id1", "myalias", "/x/a.safetensors", "safetensors", 0.8)]
    )
    b.generate(prompt = "a fox", steps = 4, seed = 1, loras = [("id1", 0.8)])
    _, params, _, _ = eng.calls[0]
    assert params.lora_dir is not None and params.lora_apply_mode == "auto"
    assert "<lora:myalias:0.8>" in params.prompt


def test_generate_server_stages_loras_and_sends_structured_field(monkeypatch, tmp_path):
    # Server-mode LoRA rides the structured `lora` field (the sdcpp API ignores prompt tags): staged into --lora-model-dir and referenced by relative path.
    from pathlib import Path as _P

    from core.inference import diffusion_lora as dl

    b = SdCppDiffusionBackend()
    servers: list = []
    _run_server_load(monkeypatch, b, servers)
    servers[0].lora_dir = str(tmp_path)
    _patch_lora(
        monkeypatch, [dl.ResolvedLora("id1", "myalias", "/x/a.safetensors", "safetensors", 0.7)]
    )
    b.generate(prompt = "x", steps = 4, seed = 1, batch_size = 1, loras = [("id1", 0.7)])
    payload = servers[0].payloads[0]
    assert "lora" in payload and len(payload["lora"]) == 1
    assert payload["lora"][0]["multiplier"] == 0.7
    assert payload["lora"][0]["path"].endswith("myalias.safetensors")
    assert "<lora:" not in payload["prompt"]  # no prompt-tag mechanism on the server
    # The per-request stage subdir under the server's lora dir is removed after the batch.
    assert not list(_P(tmp_path).glob("gen_*"))


def test_generate_rejects_loras_on_unsupported_family(monkeypatch):
    b = _loaded_backend(engine = _FakeEngine())
    _patch_lora(monkeypatch, [], supported = False)
    with pytest.raises(ValueError, match = "LoRA is not supported"):
        b.generate(prompt = "x", steps = 4, seed = 1, loras = [("id1", 1.0)])


def test_generate_zero_weight_loras_are_noop(monkeypatch):
    # weight-0 rows are dropped BEFORE the support gate, so an only-disabled request stays a no-op even where native LoRA is unsupported.
    eng = _FakeEngine()
    b = _loaded_backend(engine = eng)
    _patch_lora(monkeypatch, [], supported = False)  # would raise if the gate were reached
    b.generate(prompt = "x", steps = 4, seed = 1, loras = [("id1", 0.0)])
    _, params, _, _ = eng.calls[0]
    assert params.lora_dir is None  # nothing applied


def test_generate_rejects_controlnet_on_native_engine():
    # ControlNet is diffusers-only, so the native backend must reject it with a clean ValueError (400), not a TypeError (500).
    b = _loaded_backend(engine = _FakeEngine())
    with pytest.raises(ValueError, match = "ControlNet is not yet supported on the native"):
        b.generate(prompt = "x", steps = 4, seed = 1, controlnet = ("id", "img", "canny", 1.0, 0.0, 1.0))


@pytest.mark.parametrize("cn_strength", [0, 0.0, None])
def test_generate_treats_zero_strength_controlnet_as_disabled(cn_strength):
    # strength 0 (or None) disables ControlNet, so a strength-0 spec must succeed on the native engine too.
    eng = _FakeEngine()
    b = _loaded_backend(engine = eng)
    out = b.generate(
        prompt = "x",
        steps = 4,
        seed = 1,
        controlnet = ("id", "img", "canny", cn_strength, 0.0, 1.0),
    )
    assert len(out["images"]) == 1


def test_generate_rejects_image_conditioned_on_native_engine():
    # img2img / inpaint / reference / upscale are diffusers-only; a native call with an init image gets a clean ValueError, not a silent txt2img.
    b = _loaded_backend(engine = _FakeEngine())
    with pytest.raises(ValueError, match = "not yet supported on the native"):
        b.generate(prompt = "x", steps = 4, seed = 1, init_image = "data:image/png;base64,AAAA")


def test_status_native_reports_supports_controlnet_false():
    b = _loaded_backend()
    assert b.status()["supports_controlnet"] is False
