# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the native sd.cpp diffusion backend (the no-GPU engine)."""

from __future__ import annotations

import threading
import types

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
        mode = "oneshot",  # this fixture injects an engine, so it exercises the one-shot path
    )
    return b


def test_loaded_repo_ids_includes_native_companions():
    # The one-shot native engine re-reads its companion VAE / text-encoder files from the HF
    # cache on every generation, so the delete-cached guard queries loaded_repo_ids() to refuse
    # deleting an in-use companion repo. It must surface the committed family's VAE + text-encoder
    # repos (plus the main + base repos), not just the loaded GGUF, and be empty once unloaded.
    b = _loaded_backend("flux.1")
    ids = set(b.loaded_repo_ids())
    fam = detect_family("flux.1")
    assert "unsloth/Z-Image-Turbo-GGUF" in ids  # the main GGUF repo
    assert fam.base_repo in ids
    assert fam.sd_cpp_vae[0] in ids  # black-forest-labs/FLUX.1-schnell (VAE)
    for terepo, _f, _k in fam.sd_cpp_text_encoders:
        assert terepo in ids  # comfyanonymous/flux_text_encoders
    b._state = None
    assert b.loaded_repo_ids() == ()


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
    ):
        self.started = dict(
            files = files,
            vae_format = vae_format,
            offload = offload,
            native_speed = native_speed,
            threads = threads,
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
    # Native LoRA resolution (listing/downloading a not-yet-cached adapter) happens during the
    # pre-generate setup while _generate_lock is already held. A reload/progress probe in that
    # window must read ACTIVE, not idle, or the UI queues a second generate behind the first.
    # So _gen is published before LoRA resolution, mirroring the diffusers path.
    from core.inference import diffusion_lora

    eng = _FakeEngine()
    b = _loaded_backend(engine = eng)
    monkeypatch.setattr(diffusion_lora, "supports_lora", lambda **_k: True)

    seen: dict = {}

    def _resolve(
        active,
        *,
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
    # A local .gguf pick whose family keyword lives only in the basename (parent dir
    # carries none) must resolve via the same filename fallback the route validated
    # with -- not dead-end with "Could not infer" on a native (no-GPU) host.
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    monkeypatch.setattr(b, "_run_load", lambda **kwargs: None)  # skip the download thread
    b.begin_load("/models/gguf-store", gguf_filename = "Z-Image-Turbo-Q4_K_M.gguf")
    # Validation passed (no ValueError) and the family was inferred from the filename.
    assert b._loading is not None and b._loading.repo_id == "/models/gguf-store"


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
    # status must reflect the offload flags actually passed to sd-cli, not always "none",
    # so a balanced/low_vram (or cpu_offload) load is verifiable.
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
    # A generation that started during the asset download is still running against the OLD
    # model. _run_load must cancel it AND wait on _generate_lock before committing the new
    # state, or a stale sd-cli run finishes afterward and persists an image from the previous
    # model once the new load reports ready.
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    fam = detect_family("z-image")
    monkeypatch.setattr(b, "_asset_specs", lambda *a, **k: [])
    monkeypatch.setattr(b, "_set_expected_bytes", lambda *a, **k: None)
    monkeypatch.setattr(
        b,
        "_fetch_assets",
        lambda *a, **k: {"diffusion_model": "/m/z.gguf", "vae": "/m/vae.sft", "llm": "/m/llm.sft"},
    )
    # Avoid importing torch from the worker thread (its first import deadlocks off the main
    # thread -- a test artifact, not a production path); the device only needs to be CPU here.
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
        # The commit must block behind the live generation and not publish the new state,
        # but must already have signalled the in-flight cancel.
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
    # A lazily cached fallback engine (NOT an explicit injection) must not force one-shot:
    # once a server is available again, the next load can use it.
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
        bk, "resolve_diffusion_device_target", lambda: types.SimpleNamespace(device = "cpu")
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


def test_server_generate_splits_batches_above_server_limit(monkeypatch):
    # A batch above the server's per-job limit is chunked (the one-shot path did these
    # image-by-image); each chunk gets a timeout proportional to its image count.
    b = SdCppDiffusionBackend()
    servers: list = []
    _run_server_load(monkeypatch, b, servers)
    out = b.generate(prompt = "x", width = 64, height = 64, steps = 4, seed = 100, batch_size = 10)
    assert len(out["images"]) == 10
    counts = [p["batch_count"] for p in servers[0].payloads]
    assert counts == [bk._MAX_SERVER_BATCH, 10 - bk._MAX_SERVER_BATCH]  # [8, 2]
    # Each chunk's timeout scales with its image count, not one fixed batch deadline.
    assert servers[0].timeouts == [
        bk._SERVER_PER_IMAGE_TIMEOUT_S * 8,
        bk._SERVER_PER_IMAGE_TIMEOUT_S * 2,
    ]
    # Seeds run contiguously across chunks (chunk 2 submitted at base + 8).
    assert out["seeds"] == list(range(100, 110))
    assert servers[0].payloads[1]["seed"] == 108


def test_server_generate_masks_large_seed(monkeypatch):
    # sd.cpp's image seed is signed int64; a larger explicit seed must be masked before it
    # reaches the server (the request model / diffusers accept up to 2**64 - 1).
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
    # Probe passes; the failure we exercise here is in start(), not the up-front probe.
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
    # A load failure surfaced via load_progress() must run through redact_native_paths, the
    # same scrub the diffusers load path applies, so a registered native path can't leak.
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
    # One-shot sd-cli LoRA: adapters materialized into a --lora-model-dir and selected with
    # <lora:ALIAS:w> tags injected into the prompt (the real inject_prompt_tags runs here).
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
    # Server-mode LoRA rides the structured `lora` request field (the sdcpp API ignores
    # <lora:> prompt tags): adapters staged into the server's --lora-model-dir, referenced
    # by their path relative to it + the validated multiplier.
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
    # weight-0 rows are dropped BEFORE the support gate, so a request carrying only disabled
    # adapters stays a no-op even on a family where native LoRA is unsupported.
    eng = _FakeEngine()
    b = _loaded_backend(engine = eng)
    _patch_lora(monkeypatch, [], supported = False)  # would raise if the gate were reached
    b.generate(prompt = "x", steps = 4, seed = 1, loras = [("id1", 0.0)])
    _, params, _, _ = eng.calls[0]
    assert params.lora_dir is None  # nothing applied


def test_generate_rejects_controlnet_on_native_engine():
    # ControlNet is diffusers-only. The route passes `controlnet` to whichever engine is
    # active, so the native backend must reject it with a clean ValueError (-> 400) rather
    # than TypeError on an unexpected kwarg (-> opaque 500).
    b = _loaded_backend(engine = _FakeEngine())
    with pytest.raises(ValueError, match = "ControlNet is not yet supported on the native"):
        b.generate(prompt = "x", steps = 4, seed = 1, controlnet = ("id", "img", "canny", 1.0, 0.0, 1.0))


@pytest.mark.parametrize("cn_strength", [0, 0.0, None])
def test_generate_treats_zero_strength_controlnet_as_disabled(cn_strength):
    # strength 0 (or None) disables ControlNet -- the diffusers path treats it as plain
    # txt2img and the request model documents it -- so a strength-0 spec must succeed on the
    # native engine too, not 400. Only a genuinely active (strength > 0) ControlNet is rejected.
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
    # img2img / inpaint / reference / upscale are likewise diffusers-only; a direct API call
    # with an init image on the native engine gets a clean ValueError, not a silent txt2img.
    b = _loaded_backend(engine = _FakeEngine())
    with pytest.raises(ValueError, match = "not yet supported on the native"):
        b.generate(prompt = "x", steps = 4, seed = 1, init_image = "data:image/png;base64,AAAA")


def test_status_native_reports_supports_controlnet_false():
    b = _loaded_backend()
    assert b.status()["supports_controlnet"] is False
