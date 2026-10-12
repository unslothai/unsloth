# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Router matrix for the opt-in MLX image engine: defaults unchanged, refusals fall back, predict == select."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from core.inference import diffusion_engine_router as r
from core.inference.diffusion_families import detect_family
from core.inference.sd_cpp_engine import ENGINE_DIFFUSERS, ENGINE_MLX, ENGINE_SD_CPP

_ENVS = (
    "UNSLOTH_DIFFUSION_ENGINE",
    "UNSLOTH_DIFFUSION_SD_CPP",
    "UNSLOTH_DIFFUSION_SD_CPP_MPS",
    "UNSLOTH_DIFFUSION_SD_CPP_INSTALL",
    "UNSLOTH_DIFFUSION_MLX",
)
_Q4 = "z-image-turbo-Q4_0.gguf"


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    r._supported_family_capabilities.cache_clear()
    for e in _ENVS:
        monkeypatch.delenv(e, raising = False)
    monkeypatch.setattr(
        r,
        "get_active_diffusion_engine",
        lambda: SimpleNamespace(
            status = lambda: {"loaded": False, "repo_id": None}, unload = lambda: None
        ),
    )
    # No sd.cpp binary anywhere and no install: today's MPS / CPU choice is then diffusers, never a real exec.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_INSTALL", "0")
    monkeypatch.setattr(r, "ensure_sd_server_binary", lambda **_: None)
    monkeypatch.setattr(r, "ensure_sd_cpp_binary", lambda **_: None)
    monkeypatch.setattr(r, "_server_binary_runnable", lambda *_a, **_k: False)
    saved = (r._active_engine_name, r._fallback_reason, r._mlx_fallback_reason)
    try:
        yield
    finally:
        r._active_engine_name, r._fallback_reason, r._mlx_fallback_reason = saved
        r._supported_family_capabilities.cache_clear()


def _device(monkeypatch, backend):
    monkeypatch.setattr(
        r,
        "resolve_diffusion_device_target",
        lambda: SimpleNamespace(backend = backend, device = backend),
    )


def _host(monkeypatch, *, system = "darwin", machine = "arm64", macos = "15.1"):
    monkeypatch.setattr(r, "sys", SimpleNamespace(platform = system, modules = sys.modules))
    monkeypatch.setattr(
        r,
        "platform",
        SimpleNamespace(machine = lambda: machine, mac_ver = lambda: (macos, ("", "", ""), "")),
    )


def _runtime(monkeypatch, *, mlx = True, mflux = True):
    present = {"mlx": mlx, "mflux": mflux}
    real_find_spec = r.importlib.util.find_spec
    monkeypatch.setattr(
        r,
        "importlib",
        SimpleNamespace(
            util = SimpleNamespace(
                find_spec = lambda name, *a, **k: (
                    (object() if present[name] else None)
                    if name in present
                    else real_find_spec(name, *a, **k)
                )
            )
        ),
    )


def _apple(monkeypatch, backend = "mps"):
    _host(monkeypatch)
    _device(monkeypatch, backend)
    _runtime(monkeypatch)


def _select(fam = "z-image", gguf = _Q4, kind = "gguf"):
    r.select_and_activate_engine(detect_family(fam), model_kind = kind, gguf_filename = gguf)
    return r.active_engine_name()


def _predict(fam = "z-image", gguf = _Q4, kind = "gguf"):
    return r.predict_engine(detect_family(fam), model_kind = kind, gguf_filename = gguf)


# ── opt-in on a capable Mac picks MLX ────────────────────────────────────────


@pytest.mark.parametrize("value", ["1", "on", "true", "yes"])
@pytest.mark.parametrize(
    "gguf",
    [
        _Q4,
        "z-image-turbo-Q4_1.gguf",
        "z-image-turbo-Q8_0.gguf",
        "z-image-turbo-BF16.gguf",
    ],
)
def test_opt_in_on_apple_silicon_picks_mlx(monkeypatch, value, gguf):
    _apple(monkeypatch)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX", value)
    assert _predict(gguf = gguf) == ENGINE_MLX
    assert _select(gguf = gguf) == ENGINE_MLX
    status = r.annotate_status({"loaded": False})
    assert status["engine"] == ENGINE_MLX
    assert status["mlx_fallback_reason"] is None
    assert status["fallback_reason"] is None


def test_opt_in_on_the_mps_less_cpu_backend_still_picks_mlx(monkeypatch):
    # A Mac whose torch has no MPS resolves "cpu"; MLX drives the GPU itself.
    _apple(monkeypatch, backend = "cpu")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX", "1")
    assert _predict() == _select() == ENGINE_MLX


# ── default unchanged: unset / 0 / auto never consult MLX ────────────────────


@pytest.mark.parametrize("value", [None, "0", "off", "auto", "AUTO", "garbage"])
def test_default_and_auto_keep_todays_engine(monkeypatch, value):
    _apple(monkeypatch)
    if value is not None:
        monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX", value)

    def _no_probe(*_a, **_k):
        raise AssertionError("MLX must not be probed when it was not asked for")

    monkeypatch.setattr(r, "_mlx_runtime_refusal", _no_probe)
    monkeypatch.setattr(r, "_mlx_host_refusal", _no_probe)
    assert _predict() == ENGINE_DIFFUSERS
    assert _select() == ENGINE_DIFFUSERS
    status = r.annotate_status({})
    assert status["mlx_fallback_reason"] is None
    # Today's reason, untouched.
    assert status["fallback_reason"] == "GPU backend 'mps' uses diffusers"


def test_default_keeps_sd_cpp_where_sd_cpp_is_chosen_today(monkeypatch):
    _host(monkeypatch)
    _device(monkeypatch, "mps")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_MPS", "1")
    monkeypatch.setattr(r, "ensure_sd_cpp_binary", lambda **_: "/usr/bin/sd-cli")
    monkeypatch.setattr(r, "SdCppEngine", lambda **_: SimpleNamespace(version = lambda: "v0"))
    assert _predict() == ENGINE_SD_CPP
    assert _select() == ENGINE_SD_CPP


# ── refusals fall back to today's engine and say why ─────────────────────────


@pytest.mark.parametrize(
    "fam, gguf, kind, needle",
    [
        ("sdxl", "sdxl-Q4_0.gguf", "gguf", "no MLX implementation"),
        ("qwen-image", "qwen-image-Q4_0.gguf", "gguf", "no MLX implementation"),
        ("z-image", "z-image-turbo-Q4_K_M.gguf", "gguf", "Q4_K_M"),
        ("z-image", "z-image-turbo-Q6_K.gguf", "gguf", "Q6_K"),
        ("z-image", "z-image-turbo-UD-Q4_K_XL.gguf", "gguf", "not convertible"),
        ("z-image", None, "gguf", "unknown"),
        ("z-image", None, "pipeline", "GGUF picks only"),
        ("z-image", "model.safetensors", "single_file", "GGUF picks only"),
    ],
)
def test_unsupported_family_quant_or_kind_falls_back_with_reason(monkeypatch, fam, gguf, kind, needle):
    _apple(monkeypatch)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX", "1")
    predicted = _predict(fam, gguf, kind)
    chosen = _select(fam, gguf, kind)
    assert predicted == chosen == ENGINE_DIFFUSERS
    reason = r.annotate_status({})["mlx_fallback_reason"] or ""
    assert needle in reason


@pytest.mark.parametrize(
    "host, needle",
    [
        ({"system": "linux", "machine": "x86_64"}, "macOS only"),
        ({"system": "linux", "machine": "aarch64"}, "macOS only"),
        ({"system": "win32", "machine": "AMD64"}, "macOS only"),
        ({"system": "darwin", "machine": "x86_64"}, "Apple Silicon"),
        ({"system": "darwin", "machine": "arm64", "macos": "13.6"}, "macOS 14"),
        ({"system": "darwin", "machine": "arm64", "macos": ""}, "macOS 14"),
    ],
)
def test_non_apple_silicon_hosts_never_get_mlx(monkeypatch, host, needle):
    _host(monkeypatch, **host)
    _device(monkeypatch, "cpu")
    _runtime(monkeypatch)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX", "1")
    assert _predict() == _select() == ENGINE_DIFFUSERS
    assert needle in (r.annotate_status({})["mlx_fallback_reason"] or "")


@pytest.mark.parametrize("gpu", ["cuda", "rocm", "xpu"])
def test_gpu_backends_never_get_mlx(monkeypatch, gpu):
    _host(monkeypatch)
    _device(monkeypatch, gpu)
    _runtime(monkeypatch)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX", "1")
    assert _predict() == _select() == ENGINE_DIFFUSERS
    assert "not Apple MPS" in (r.annotate_status({})["mlx_fallback_reason"] or "")


def test_real_linux_host_never_gets_mlx(monkeypatch):
    # No host stubs at all: this suite runs on Linux CI.
    if sys.platform == "darwin":
        pytest.skip("checks the real non-macOS host")
    _device(monkeypatch, "cpu")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX", "1")
    assert _predict() != ENGINE_MLX
    assert _select() != ENGINE_MLX
    assert "macOS only" in (r.annotate_status({})["mlx_fallback_reason"] or "")


@pytest.mark.parametrize(
    "mlx, mflux, needle",
    [(False, True, "mlx not installed"), (True, False, "mflux not installed"), (False, False, "mlx and mflux")],
)
def test_missing_runtime_falls_back(monkeypatch, mlx, mflux, needle):
    _host(monkeypatch)
    _device(monkeypatch, "mps")
    _runtime(monkeypatch, mlx = mlx, mflux = mflux)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX", "1")
    assert _predict() == _select() == ENGINE_DIFFUSERS
    assert needle in (r.annotate_status({})["mlx_fallback_reason"] or "")


@pytest.mark.parametrize("forced", ["diffusers", "sd_cpp"])
def test_forced_engine_overrides_the_opt_in(monkeypatch, forced):
    _apple(monkeypatch)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX", "1")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ENGINE", forced)
    assert _predict() != ENGINE_MLX
    assert _select() != ENGINE_MLX
    assert "overrides" in (r.annotate_status({})["mlx_fallback_reason"] or "")


def test_refusal_falls_through_to_sd_cpp_when_that_is_todays_choice(monkeypatch):
    # A K-quant on a CPU-backend Mac with a runnable sd-cli stays on sd.cpp, exactly as without the opt-in.
    _apple(monkeypatch, backend = "cpu")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX", "1")
    monkeypatch.setattr(r, "ensure_sd_cpp_binary", lambda **_: "/usr/bin/sd-cli")
    monkeypatch.setattr(r, "SdCppEngine", lambda **_: SimpleNamespace(version = lambda: "v0"))
    gguf = "z-image-turbo-Q4_K_M.gguf"
    assert _predict(gguf = gguf) == _select(gguf = gguf) == ENGINE_SD_CPP
    assert "Q4_K_M" in (r.annotate_status({})["mlx_fallback_reason"] or "")


def test_a_later_default_selection_clears_the_mlx_reason(monkeypatch):
    _apple(monkeypatch)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX", "1")
    _select(gguf = "z-image-turbo-Q4_K_M.gguf")
    assert r.annotate_status({})["mlx_fallback_reason"]
    monkeypatch.delenv("UNSLOTH_DIFFUSION_MLX")
    _select()
    assert r.annotate_status({})["mlx_fallback_reason"] is None


# ── predict == select on the whole matrix ────────────────────────────────────


@pytest.mark.parametrize("env", [None, "0", "auto", "1"])
@pytest.mark.parametrize("backend", ["mps", "cpu", "cuda"])
@pytest.mark.parametrize("gguf", [_Q4, "z-image-turbo-Q4_K_M.gguf", None])
@pytest.mark.parametrize("fam", ["z-image", "flux.1"])
def test_predict_matches_selection(monkeypatch, env, backend, gguf, fam):
    _host(monkeypatch)
    _device(monkeypatch, backend)
    _runtime(monkeypatch)
    if env is not None:
        monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX", env)
    predicted = _predict(fam, gguf)
    r._active_engine_name = ENGINE_DIFFUSERS
    assert _select(fam, gguf) == predicted
    expect_mlx = env == "1" and backend != "cuda" and gguf == _Q4 and fam == "z-image"
    assert (predicted == ENGINE_MLX) is expect_mlx


# ── engine plumbing ──────────────────────────────────────────────────────────


def test_engine_for_mlx_is_the_mlx_singleton():
    from core.inference import mlx_diffusion_backend as mb

    engine = r.engine_for(ENGINE_MLX)
    assert isinstance(engine, mb.MlxDiffusionBackend)
    assert engine is mb.get_mlx_diffusion_backend()


def test_cancel_and_retire_scan_the_mlx_singleton(monkeypatch):
    from core.inference import mlx_diffusion_backend as mb

    calls = []
    fake = SimpleNamespace(
        _active_generate_account = "acct",
        cancel_generate = lambda expected_account = None: calls.append(expected_account) or True,
    )
    monkeypatch.setattr(mb, "_mlx_diffusion_backend", fake)
    assert ("core.inference.mlx_diffusion_backend", "_mlx_diffusion_backend") in r._ENGINE_SINGLETONS
    assert r.cancel_generation_for_account("acct") is True
    assert calls == ["acct"]


def test_switching_away_from_mlx_unloads_it(monkeypatch):
    unloaded = []
    monkeypatch.setattr(
        r,
        "get_active_diffusion_engine",
        lambda: SimpleNamespace(status = lambda: {}, unload = lambda: unloaded.append(True)),
    )
    r._active_engine_name = ENGINE_MLX
    _host(monkeypatch)
    _device(monkeypatch, "mps")
    assert _select() == ENGINE_DIFFUSERS
    assert unloaded == [True]


def test_family_buildable_here_counts_mlx_only_with_a_convertible_gguf(monkeypatch):
    _apple(monkeypatch)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX", "1")
    fam = detect_family("z-image")
    monkeypatch.setattr(r, "family_pipeline_available", lambda _f: False)
    assert r.family_buildable_here(fam, model_kind = "gguf", gguf_filename = _Q4) is True
    assert r.family_buildable_here(fam, model_kind = "gguf") is False
    monkeypatch.delenv("UNSLOTH_DIFFUSION_MLX")
    assert r.family_buildable_here(fam, model_kind = "gguf", gguf_filename = _Q4) is False
