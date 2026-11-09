# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Router matrix for the Apple Silicon MLX image engine."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from core.inference import diffusion_engine_router as r
from core.inference.diffusion_families import detect_family
from core.inference.sd_cpp_engine import ENGINE_DIFFUSERS, ENGINE_MLX, ENGINE_SD_CPP


def _mlx_without_lora(monkeypatch):
    """Z-Image's MLX build applies LoRAs; the fallback rules below are for a family whose build does not."""
    from core.inference import mlx_diffusion_routing as routing
    monkeypatch.setitem(routing.MLX_CAPABILITIES, "z-image", routing.MlxCapabilities(loras = False))


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


def _host(
    monkeypatch,
    *,
    system = "darwin",
    machine = "arm64",
    macos = "15.1",
):
    monkeypatch.setattr(r, "sys", SimpleNamespace(platform = system, modules = sys.modules))
    monkeypatch.setattr(
        r,
        "platform",
        SimpleNamespace(machine = lambda: machine, mac_ver = lambda: (macos, ("", "", ""), "")),
    )


_GOOD_VERSIONS = {
    "mflux": "0.22.0",
    "mlx": "0.32.3",
    "piexif": "1.1.3",
    "platformdirs": "4.3.8",
    "toml": "0.10.2",
    "pillow": "12.3.0",
}


def _runtime(
    monkeypatch,
    *,
    mlx = True,
    mflux = True,
    versions = None,
):
    present = {"mlx": mlx, "mflux": mflux}
    versions = {**_GOOD_VERSIONS, **(versions or {})}
    monkeypatch.setattr(r, "_installed_version", versions.get)
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


def _select(
    fam = "z-image",
    gguf = _Q4,
    kind = "gguf",
):
    r.select_and_activate_engine(detect_family(fam), model_kind = kind, gguf_filename = gguf)
    return r.active_engine_name()


def _predict(
    fam = "z-image",
    gguf = _Q4,
    kind = "gguf",
):
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


@pytest.mark.parametrize("value", [None, "", "auto", "AUTO"])
def test_default_and_auto_pick_mlx_on_apple_silicon(monkeypatch, value):
    _apple(monkeypatch)
    if value is not None:
        monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX", value)
    assert _predict() == ENGINE_MLX
    assert _select() == ENGINE_MLX
    status = r.annotate_status({"loaded": False})
    assert status["engine"] == ENGINE_MLX
    assert status["mlx_fallback_reason"] is None


@pytest.mark.parametrize("value", [None, "auto"])
@pytest.mark.parametrize(
    "host",
    [
        {"system": "linux", "machine": "x86_64"},
        {"system": "win32", "machine": "AMD64"},
        {"system": "darwin", "machine": "x86_64"},
        {"system": "darwin", "machine": "arm64", "macos": "13.6"},
    ],
)
def test_auto_on_other_hosts_is_todays_engine_and_records_nothing(monkeypatch, value, host):
    _host(monkeypatch, **host)
    _device(monkeypatch, "cpu")

    def _no_probe(*_a, **_k):
        raise AssertionError("auto must not probe the MLX runtime off Apple Silicon")

    monkeypatch.setattr(r, "_mlx_runtime_refusal", _no_probe)
    if value is not None:
        monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX", value)
    assert _predict() == _select() == ENGINE_DIFFUSERS
    assert r.annotate_status({})["mlx_fallback_reason"] is None


@pytest.mark.parametrize(
    "blocker_kwargs, needle",
    [
        ({"component_files": {"vae_file": "vae.safetensors"}}, "text-encoder / VAE"),
        ({"transformer_quant": "int8"}, "transformer precision (int8)"),
        ({"text_encoder_quant": "fp8"}, "text-encoder precision (fp8)"),
    ],
)
def test_auto_keeps_a_load_mlx_cannot_honour_on_diffusers(monkeypatch, blocker_kwargs, needle):
    _apple(monkeypatch)
    blocker = r.mlx_load_blocker(**blocker_kwargs)
    assert needle in blocker
    fam = detect_family("z-image")
    assert (
        r.predict_engine(fam, model_kind = "gguf", gguf_filename = _Q4, mlx_blocker = blocker)
        == ENGINE_DIFFUSERS
    )
    r.select_and_activate_engine(fam, model_kind = "gguf", gguf_filename = _Q4, mlx_blocker = blocker)
    assert r.active_engine_name() == ENGINE_DIFFUSERS
    assert needle in (r.annotate_status({})["mlx_fallback_reason"] or "")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX", "1")
    assert (
        r.predict_engine(fam, model_kind = "gguf", gguf_filename = _Q4, mlx_blocker = blocker)
        == ENGINE_MLX
    )


@pytest.mark.parametrize("value", [None, "auto", "1"])
def test_native_lora_pick_on_a_mac_is_never_an_mlx_refusal(monkeypatch, value):
    _apple(monkeypatch)
    if value is not None:
        monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX", value)
    monkeypatch.setattr(r, "ensure_sd_cpp_binary", lambda **_: "/usr/bin/sd-cli")
    monkeypatch.setattr(r, "SdCppEngine", lambda **_: SimpleNamespace(version = lambda: "v0"))
    blocker = r.mlx_load_blocker()
    fam = detect_family("z-image")
    use_mlx, reason = r.mlx_selection(
        fam, model_kind = "gguf", gguf_filename = _Q4, engine = ENGINE_SD_CPP, load_blocker = blocker
    )
    assert use_mlx is False
    assert "sd_cpp" in reason
    assert (
        r.predict_engine(
            fam,
            model_kind = "gguf",
            gguf_filename = _Q4,
            engine = ENGINE_SD_CPP,
            mlx_blocker = blocker,
        )
        == ENGINE_SD_CPP
    )


def test_delegating_precisions_do_not_block_mlx():
    for quant in (None, "", "auto", "AUTO"):
        assert r.mlx_load_blocker(transformer_quant = quant, text_encoder_quant = quant) is None


def test_pinned_original_precisions_block_mlx():
    for quant in ("none", "off"):
        assert r.mlx_load_blocker(transformer_quant = quant) is not None
        assert r.mlx_load_blocker(text_encoder_quant = quant) is not None


@pytest.mark.parametrize("value", ["0", "off", "false", "no", "garbage"])
def test_kill_switch_keeps_todays_engine(monkeypatch, value):
    _apple(monkeypatch)
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


@pytest.mark.parametrize("runtime", [False, True])
def test_default_keeps_sd_cpp_where_sd_cpp_is_chosen_today(monkeypatch, runtime):
    _host(monkeypatch)
    _device(monkeypatch, "mps")
    if runtime:
        _runtime(monkeypatch)
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
        ("z-image", "z-image-turbo-Q3_K_M.gguf", "gguf", "Q3_K_M"),
        ("z-image", "z-image-turbo-Q2_K.gguf", "gguf", "Q2_K"),
        ("z-image", "z-image-turbo-UD-Q4_K_XL.gguf", "gguf", "not convertible"),
        ("z-image", None, "gguf", "unknown"),
        ("z-image", None, "pipeline", "GGUF picks only"),
        ("z-image", "model.safetensors", "single_file", "GGUF picks only"),
    ],
)
def test_unsupported_family_quant_or_kind_falls_back_with_reason(
    monkeypatch, fam, gguf, kind, needle
):
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
    [
        (False, True, "mlx not installed"),
        (True, False, "mflux not installed"),
        (False, False, "mlx and mflux"),
    ],
)
def test_missing_runtime_falls_back(monkeypatch, mlx, mflux, needle):
    _host(monkeypatch)
    _device(monkeypatch, "mps")
    _runtime(monkeypatch, mlx = mlx, mflux = mflux)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX", "1")
    assert _predict() == _select() == ENGINE_DIFFUSERS
    assert needle in (r.annotate_status({})["mlx_fallback_reason"] or "")


@pytest.mark.parametrize(
    "versions, needle",
    [
        ({"mflux": "0.17.0"}, "mflux==0.22.0 (found 0.17.0)"),
        ({"mlx": "0.29.1"}, "mlx>=0.32.0,<0.33.0 (found 0.29.1)"),
        ({"mlx": None}, "mlx>=0.32.0,<0.33.0 (found none)"),
        ({"toml": None}, "toml>=0.10.2,<1 (found none)"),
        ({"platformdirs": "5.0.0"}, "platformdirs>=4,<5 (found 5.0.0)"),
        ({"pillow": "13.0.0"}, "pillow>=12.3.0,<13 (found 13.0.0)"),
    ],
)
def test_an_incompatible_installed_runtime_falls_back(monkeypatch, versions, needle):
    # An older mflux left behind by a failed upgrade still imports; selection must read the installed versions.
    _host(monkeypatch)
    _device(monkeypatch, "mps")
    _runtime(monkeypatch, versions = versions)
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
    gguf = "z-image-turbo-Q3_K_M.gguf"
    assert _predict(gguf = gguf) == _select(gguf = gguf) == ENGINE_SD_CPP
    assert "Q3_K_M" in (r.annotate_status({})["mlx_fallback_reason"] or "")


def test_a_later_default_selection_clears_the_mlx_reason(monkeypatch):
    _apple(monkeypatch)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX", "1")
    _select(gguf = "z-image-turbo-Q3_K_M.gguf")
    assert r.annotate_status({})["mlx_fallback_reason"]
    monkeypatch.delenv("UNSLOTH_DIFFUSION_MLX")
    _select()
    assert r.annotate_status({})["mlx_fallback_reason"] is None


# ── predict == select on the whole matrix ────────────────────────────────────


@pytest.mark.parametrize("env", [None, "0", "auto", "1"])
@pytest.mark.parametrize("backend", ["mps", "cpu", "cuda"])
@pytest.mark.parametrize(
    "gguf", [_Q4, "z-image-turbo-Q4_K_M.gguf", "z-image-turbo-Q3_K_M.gguf", None]
)
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
    convertible = gguf in (_Q4, "z-image-turbo-Q4_K_M.gguf")
    expect_mlx = (
        env in (None, "auto", "1") and backend != "cuda" and convertible and fam == "z-image"
    )
    assert (predicted == ENGINE_MLX) is expect_mlx


# ── engine plumbing ──────────────────────────────────────────────────────────


def test_engine_for_forced_mlx_is_the_mlx_singleton(monkeypatch):
    from core.inference import mlx_diffusion_backend as mb

    monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX", "1")
    engine = r.engine_for(ENGINE_MLX)
    assert isinstance(engine, mb.MlxDiffusionBackend)
    assert engine is mb.get_mlx_diffusion_backend()


@pytest.mark.parametrize("value", [None, "auto"])
def test_engine_for_auto_mlx_is_the_routed_singleton(monkeypatch, value):
    from core.inference import mlx_diffusion_routing as routing

    if value is not None:
        monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX", value)
    engine = r.engine_for(ENGINE_MLX)
    assert isinstance(engine, routing.RoutedMlxEngine)
    assert engine is routing.get_routed_mlx_engine()


def test_cancel_scans_the_routed_engine_while_it_loads_its_fallback(monkeypatch):
    from core.inference import mlx_diffusion_routing as routing

    calls = []
    fake = SimpleNamespace(
        _active_generate_account = "acct",
        cancel_generate = lambda expected_account = None: calls.append(expected_account) or True,
    )
    monkeypatch.setattr(routing, "_routed_mlx_engine", fake)
    assert r.cancel_generation_for_account("acct") is True
    assert calls == ["acct"]
    assert r.cancel_generation_for_account("someone-else") is False


def test_cancel_and_retire_scan_the_mlx_singleton(monkeypatch):
    from core.inference import mlx_diffusion_backend as mb

    calls = []
    fake = SimpleNamespace(
        _active_generate_account = "acct",
        cancel_generate = lambda expected_account = None: calls.append(expected_account) or True,
    )
    monkeypatch.setattr(mb, "_mlx_diffusion_backend", fake)
    assert (
        "core.inference.mlx_diffusion_backend",
        "_mlx_diffusion_backend",
    ) in r._ENGINE_SINGLETONS
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
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX", "0")
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
    assert r.family_buildable_here(fam, model_kind = "gguf", gguf_filename = _Q4) is True
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX", "0")
    assert r.family_buildable_here(fam, model_kind = "gguf", gguf_filename = _Q4) is False


def test_runtime_requirements_match_the_installer():
    import ast
    from pathlib import Path

    from core.inference.mlx_diffusion_convert import MFLUX_VERSION

    source = (Path(__file__).resolve().parents[2] / "install_python_stack.py").read_text(
        encoding = "utf-8"
    )
    literals = {
        node.targets[0].id if isinstance(node, ast.Assign) else node.target.id: node.value
        for node in ast.parse(source).body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and isinstance(node.targets[0] if isinstance(node, ast.Assign) else node.target, ast.Name)
    }
    assert ast.literal_eval(literals["_MLX_DIFFUSION_PINS"]) == (f"mflux=={MFLUX_VERSION}",)
    deps = set(ast.literal_eval(literals["_MLX_DIFFUSION_DEPS"]))
    assert deps == {s for s in r._MLX_RUNTIME_REQUIREMENTS if not s.startswith("mlx")}


def test_auto_switch_planner_judges_mlx_with_the_picks_gguf(monkeypatch):
    from core.inference import media_locality
    from core.inference.gpu_arbiter import DIFFUSION
    from core.inference.media_model_index import MediaModelPick

    _apple(monkeypatch)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX", "1")
    monkeypatch.setattr(r, "engine_for", lambda name: name)
    pick = MediaModelPick("unsloth/Z-Image-Turbo-GGUF", "unsloth/Z-Image-Turbo-GGUF", _Q4, "gguf")
    assert media_locality.planners_for(DIFFUSION, pick) == [ENGINE_MLX]


@pytest.mark.parametrize("value", [None, "auto"])
@pytest.mark.parametrize("gguf", [_Q4, "z-image-turbo-Q4_K_M.gguf", "z-image-turbo-Q5_1.gguf"])
def test_a_lora_load_stays_on_mlx_when_its_build_applies_loras(monkeypatch, value, gguf):
    _apple(monkeypatch)
    if value is not None:
        monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX", value)
    fam = detect_family("z-image")
    kw = dict(model_kind = "gguf", gguf_filename = gguf, with_loras = True)
    assert r.predict_engine(fam, **kw) == ENGINE_MLX
    r.select_and_activate_engine(fam, **kw)
    assert r.active_engine_name() == ENGINE_MLX
    assert r.annotate_status({})["mlx_fallback_reason"] is None


@pytest.mark.parametrize("value", [None, "auto"])
@pytest.mark.parametrize("gguf", [_Q4, "z-image-turbo-Q4_K_M.gguf", "z-image-turbo-Q5_1.gguf"])
def test_a_lora_load_declines_an_mlx_build_without_loras_under_default_routing(
    monkeypatch, value, gguf
):
    _apple(monkeypatch)
    _mlx_without_lora(monkeypatch)
    if value is not None:
        monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX", value)
    fam = detect_family("z-image")
    assert _predict(gguf = gguf) == _select(gguf = gguf) == ENGINE_MLX
    kw = dict(model_kind = "gguf", gguf_filename = gguf, with_loras = True)
    assert r.predict_engine(fam, **kw) == ENGINE_DIFFUSERS
    r.select_and_activate_engine(fam, **kw)
    assert r.active_engine_name() == ENGINE_DIFFUSERS
    assert "LoRA" in (r.annotate_status({})["mlx_fallback_reason"] or "")


@pytest.mark.parametrize("sd_cpp_off", [False, True])
def test_a_forced_mlx_lora_load_falls_back_to_diffusers_which_applies_them_to_the_gguf(
    monkeypatch, sd_cpp_off
):
    _apple(monkeypatch)
    _mlx_without_lora(monkeypatch)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX", "1")
    if sd_cpp_off:
        monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP", "0")
    fam = detect_family("z-image")
    kw = dict(model_kind = "gguf", gguf_filename = "z-image-turbo-Q4_K_M.gguf", with_loras = True)
    assert r.predict_engine(fam, **kw) == ENGINE_DIFFUSERS
    r.select_and_activate_engine(fam, **kw)
    assert r.active_engine_name() == ENGINE_DIFFUSERS


@pytest.mark.parametrize("sd_cpp_off", [False, True])
def test_a_forced_mlx_lora_load_with_no_lora_capable_fallback_is_refused_before_the_switch(
    monkeypatch, sd_cpp_off
):
    from core.inference import diffusion_lora

    _apple(monkeypatch)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX", "1")
    if sd_cpp_off:
        monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP", "0")
    _mlx_without_lora(monkeypatch)
    monkeypatch.setattr(diffusion_lora, "supports_lora", lambda **_: False)
    fam = detect_family("z-image")
    kw = dict(model_kind = "gguf", gguf_filename = "z-image-turbo-Q4_K_M.gguf", with_loras = True)
    r._active_engine_name = ENGINE_MLX
    with pytest.raises(ValueError, match = "LoRA"):
        r.predict_engine(fam, **kw)
    with pytest.raises(ValueError, match = "LoRA"):
        r.select_and_activate_engine(fam, **kw)
    assert r.active_engine_name() == ENGINE_MLX


@pytest.mark.parametrize("value", [None, "1"])
def test_a_lora_load_stays_on_mlx_over_the_native_mps_knob_only_when_forced(monkeypatch, value):
    _apple(monkeypatch)
    if value is not None:
        monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX", value)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_MPS", "1")
    monkeypatch.setattr(r, "ensure_sd_cpp_binary", lambda **_: "/usr/bin/sd-cli")
    monkeypatch.setattr(r, "SdCppEngine", lambda **_: SimpleNamespace(version = lambda: "v0"))
    fam = detect_family("z-image")
    kw = dict(model_kind = "gguf", gguf_filename = _Q4, with_loras = True)
    # The knob already sends an auto load to sd.cpp; a forced MLX engine now keeps a LoRA load.
    expected = ENGINE_MLX if value == "1" else ENGINE_SD_CPP
    assert r.predict_engine(fam, **kw) == expected
    r.select_and_activate_engine(fam, **kw)
    assert r.active_engine_name() == expected


def test_a_forced_mlx_lora_load_keeps_a_lora_capable_sd_cpp(monkeypatch):
    _apple(monkeypatch)
    _mlx_without_lora(monkeypatch)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX", "1")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_MPS", "1")
    monkeypatch.setattr(r, "ensure_sd_cpp_binary", lambda **_: "/usr/bin/sd-cli")
    monkeypatch.setattr(r, "SdCppEngine", lambda **_: SimpleNamespace(version = lambda: "v0"))
    fam = detect_family("z-image")
    kw = dict(model_kind = "gguf", gguf_filename = _Q4, with_loras = True)
    assert r.predict_engine(fam, **kw) == ENGINE_SD_CPP
    r.select_and_activate_engine(fam, **kw)
    assert r.active_engine_name() == ENGINE_SD_CPP


@pytest.mark.parametrize("value", [None, "auto", "1"])
def test_a_lora_load_off_apple_silicon_is_unchanged(monkeypatch, value):
    _host(monkeypatch, system = "linux", machine = "x86_64")
    _device(monkeypatch, "cpu")
    if value is not None:
        monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX", value)
    fam = detect_family("z-image")
    kw = dict(model_kind = "gguf", gguf_filename = _Q4, with_loras = True)
    assert r.predict_engine(fam, **kw) == ENGINE_DIFFUSERS
    r.select_and_activate_engine(fam, **kw)
    assert r.active_engine_name() == ENGINE_DIFFUSERS


def test_a_lora_load_keeps_sd_cpp_under_default_routing_with_the_native_mps_knob(monkeypatch):
    _apple(monkeypatch)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_MPS", "1")
    monkeypatch.setattr(r, "ensure_sd_cpp_binary", lambda **_: "/usr/bin/sd-cli")
    monkeypatch.setattr(r, "SdCppEngine", lambda **_: SimpleNamespace(version = lambda: "v0"))
    fam = detect_family("z-image")
    kw = dict(model_kind = "gguf", gguf_filename = _Q4, with_loras = True)
    assert r.predict_engine(fam, **kw) == ENGINE_SD_CPP
    r.select_and_activate_engine(fam, **kw)
    assert r.active_engine_name() == ENGINE_SD_CPP


def test_every_load_route_and_the_residency_estimate_pass_loras_to_engine_selection():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for rel, minimum in (("routes/inference.py", 3), ("core/inference/gpu_residency.py", 1)):
        source = (root / rel).read_text(encoding = "utf-8")
        calls = [
            c
            for c in ast.walk(ast.parse(source))
            if isinstance(c, ast.Call)
            and (
                getattr(c.func, "id", None) == "predict_engine"
                or (
                    getattr(c.func, "id", None) == "partial"
                    or getattr(c.func, "attr", None) == "partial"
                )
                and c.args
                and getattr(c.args[0], "id", None) == "select_and_activate_engine"
            )
        ]
        assert len(calls) >= minimum, rel
        for call in calls:
            kw = {k.arg: k.value for k in call.keywords}
            assert {"with_loras", "mlx_blocker"} <= set(kw), f"{rel}: {ast.unparse(call)[:120]}"
            assert getattr(kw["with_loras"].func, "id", None) in (
                "_has_active_loras",
                "has_active_loras",
            ), f"{rel}: {ast.unparse(call)[:120]}"


@pytest.mark.parametrize(
    "gguf",
    [
        "z-image-turbo-Q4_K_M.gguf",
        "z-image-turbo-Q5_K_M.gguf",
        "z-image-turbo-Q6_K.gguf",
        "z-image-turbo-Q5_0.gguf",
        "z-image-turbo-Q5_1.gguf",
        _Q4,
    ],
)
def test_a_lora_load_keeps_sd_cpp_under_the_mlx_opt_in(monkeypatch, gguf):
    _apple(monkeypatch)
    _mlx_without_lora(monkeypatch)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX", "1")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_MPS", "1")
    monkeypatch.setattr(r, "ensure_sd_cpp_binary", lambda **_: "/usr/bin/sd-cli")
    monkeypatch.setattr(r, "SdCppEngine", lambda **_: SimpleNamespace(version = lambda: "v0"))
    fam = detect_family("z-image")
    assert _predict(gguf = gguf) == _select(gguf = gguf) == ENGINE_MLX
    kw = dict(model_kind = "gguf", gguf_filename = gguf, with_loras = True)
    assert r.predict_engine(fam, **kw) == ENGINE_SD_CPP
    r.select_and_activate_engine(fam, **kw)
    assert r.active_engine_name() == ENGINE_SD_CPP
    assert "LoRA" in (r.annotate_status({})["mlx_fallback_reason"] or "")


def test_a_lora_load_without_the_mlx_opt_in_still_falls_back_to_diffusers(monkeypatch):
    _apple(monkeypatch)
    _mlx_without_lora(monkeypatch)
    fam = detect_family("z-image")
    kw = dict(model_kind = "gguf", gguf_filename = "z-image-turbo-Q4_K_M.gguf", with_loras = True)
    assert r.predict_engine(fam, **kw) == ENGINE_DIFFUSERS
    r.select_and_activate_engine(fam, **kw)
    assert r.active_engine_name() == ENGINE_DIFFUSERS


@pytest.mark.parametrize(
    "loras, active",
    [
        (None, False),
        ([], False),
        ([("a", 0.0)], False),
        ([("a", 0), ("b", 0.0)], False),
        ([SimpleNamespace(id = "a", weight = 0.0)], False),
        ([("a", 0.0), ("b", 0.5)], True),
        ([SimpleNamespace(id = "a", weight = 1.0)], True),
        ([object()], True),
    ],
)
def test_has_active_loras_ignores_zero_weight_entries(loras, active):
    assert r.has_active_loras(loras) is active


def test_a_zero_weight_lora_load_still_lands_on_mlx(monkeypatch):
    _apple(monkeypatch)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MLX", "1")
    gguf = "z-image-turbo-Q4_K_M.gguf"
    fam = detect_family("z-image")
    kw = dict(model_kind = "gguf", gguf_filename = gguf, with_loras = r.has_active_loras([("a", 0.0)]))
    assert r.predict_engine(fam, **kw) == ENGINE_MLX
