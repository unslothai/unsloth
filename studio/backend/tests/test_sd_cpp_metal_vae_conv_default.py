# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Metal VAE direct conv is the default on Apple Silicon for the families whose VAE was measured there (the FLUX.2 VAE
and the FLUX.1 ``ae``), in every speed mode. ``UNSLOTH_SD_CPP_METAL_VAE_CONV_DIRECT=0`` turns it off. CUDA, Vulkan,
ROCm, CPU, Intel Macs, macOS VMs and every other family keep their argv."""

from __future__ import annotations

import json
import stat
import sys
from types import SimpleNamespace

import pytest

import core.inference.sd_cpp_args as sd_cpp_args
from core.inference.diffusion_families import _FAMILIES, detect_family
from core.inference.sd_cpp_args import (
    SdCppGenParams,
    SdCppModelFiles,
    build_sd_cpp_server_command,
    native_speed_flags,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason = "fake sd.cpp binaries are shell scripts"
)

_HELP = (
    "  --diffusion-conv-direct   deprecated; use --conv-direct auto,diffusion=on,controlnet=on\n"
    "  --conv-direct             on|off|auto or diffusion/vae/controlnet assignments (default: auto)\n"
)
GEMM_TAG = "master-982-0518b34-u9455ee1"
SCALAR_TAG = "master-982-0518b34-u27b9ddc"
_ENV = "UNSLOTH_SD_CPP_METAL_VAE_CONV_DIRECT"
_VAE_ON = ["--conv-direct", "off,vae=on"]
# The two VAE files the default covers: FLUX.2 (measured on an M5 Max) and FLUX.1 ``ae`` (Z-Image, FLUX.1, Kontext).
_MEASURED_VAES = {
    ("black-forest-labs/FLUX.2-dev", "vae/diffusion_pytorch_model.safetensors"),
    ("black-forest-labs/FLUX.1-schnell", "ae.safetensors"),
}
_DEFAULT_FAMILIES = {"flux.2-klein", "flux.2-dev", "flux.1", "flux.1-kontext", "z-image"}


def _binary(
    tmp_path,
    tag,
    name = "sd-cli",
    help_text = _HELP,
):
    root = tmp_path / f"{tag or 'unrecorded'}_{name}_{len(help_text)}"
    root.mkdir(exist_ok = True)
    if tag:
        (root / ".unsloth-sd-cpp-install.json").write_text(json.dumps({"tag": tag}))
    path = root / name
    path.write_text("#!/bin/sh\ncat <<'EOF'\n" + help_text + "EOF\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def _clear_caches():
    sd_cpp_args._conv_direct_in_help.cache_clear()
    sd_cpp_args._installed_sd_cpp_tag.cache_clear()


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    for env in ("UNSLOTH_SD_CPP_CONV_DIRECT", _ENV, "UNSLOTH_DIFFUSION_SD_CPP_MMAP"):
        monkeypatch.delenv(env, raising = False)
    monkeypatch.setattr("sys.platform", "linux")
    _clear_caches()
    yield
    _clear_caches()


def _mac(
    monkeypatch,
    arm64 = True,
    paravirtual = False,
):
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setattr(sd_cpp_args, "apple_silicon_machine", lambda: arm64)
    monkeypatch.setattr(sd_cpp_args, "metal_is_paravirtual", lambda: paravirtual)


def _server_cmd(binary, accelerator, speed, family_default):
    return build_sd_cpp_server_command(
        binary = binary,
        files = SdCppModelFiles(diffusion_model = "/m/x.gguf", vae = "/m/vae.safetensors"),
        host = "127.0.0.1",
        port = 1234,
        native_speed = speed,
        accelerator = accelerator,
        metal_vae_conv_direct = family_default,
    )


def _pair(cmd, flag):
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


@pytest.mark.parametrize("speed", ["off", None])
def test_metal_default_speed_off_gets_vae_direct_conv(tmp_path, monkeypatch, speed):
    _mac(monkeypatch)
    binary = _binary(tmp_path, GEMM_TAG)
    assert native_speed_flags(speed, [], "metal", binary, metal_vae_conv_direct = True) == _VAE_ON
    assert (
        native_speed_flags(speed, ["--clip-on-cpu"], None, binary, metal_vae_conv_direct = True)
        == _VAE_ON
    )
    # A family outside the measured VAEs keeps the old argv (no conv flag at all).
    assert native_speed_flags(speed, [], "metal", binary) == []
    server = _server_cmd(_binary(tmp_path, GEMM_TAG, "sd-server"), "metal", speed, True)
    assert _pair(server, "--conv-direct") == "off,vae=on" and server.count("--conv-direct") == 1


@pytest.mark.parametrize("speed", ["default", "max"])
def test_metal_speed_mode_keeps_vae_direct_conv(tmp_path, monkeypatch, speed):
    _mac(monkeypatch)
    binary = _binary(tmp_path, GEMM_TAG)
    assert native_speed_flags(speed, [], "metal", binary, metal_vae_conv_direct = True) == [
        "--diffusion-fa",
        *_VAE_ON,
    ]
    assert native_speed_flags(speed, [], "metal", binary)[-1] == "off"
    server = _server_cmd(_binary(tmp_path, GEMM_TAG, "sd-server"), "metal", speed, True)
    assert _pair(server, "--conv-direct") == "off,vae=on" and server.count("--conv-direct") == 1


@pytest.mark.parametrize("value", ["0", "off", "false", "no"])
def test_kill_switch_restores_the_old_metal_argv(tmp_path, monkeypatch, value):
    _mac(monkeypatch)
    monkeypatch.setenv(_ENV, value)
    binary = _binary(tmp_path, GEMM_TAG)
    assert native_speed_flags("off", [], "metal", binary, metal_vae_conv_direct = True) == []
    assert (
        native_speed_flags("default", [], "metal", binary, metal_vae_conv_direct = True)[-1] == "off"
    )
    on, reason = sd_cpp_args.metal_vae_conv_direct_decision([], binary, family_default = True)
    assert not on and _ENV in reason


def test_global_conv_direct_off_wins(tmp_path, monkeypatch):
    _mac(monkeypatch)
    monkeypatch.setenv("UNSLOTH_SD_CPP_CONV_DIRECT", "0")
    binary = _binary(tmp_path, GEMM_TAG)
    assert native_speed_flags("off", [], "metal", binary, metal_vae_conv_direct = True) == []
    assert (
        native_speed_flags("default", [], "metal", binary, metal_vae_conv_direct = True)[-1] == "off"
    )


@pytest.mark.parametrize(
    "tag,backend_args",
    [
        (SCALAR_TAG, []),
        ("master-982-0518b34", []),
        (None, []),
        (GEMM_TAG, ["--vae-on-cpu"]),
        (GEMM_TAG, ["--backend", "diffusion=MTL0,vae=cpu"]),
        (GEMM_TAG, ["--conv-direct", "off"]),
        (GEMM_TAG, ["--backend", "cpu"]),
    ],
)
def test_metal_default_needs_the_kernel_a_metal_vae_and_no_user_flag(
    tmp_path, monkeypatch, tag, backend_args
):
    _mac(monkeypatch)
    binary = _binary(tmp_path, tag)
    assert (
        native_speed_flags("off", backend_args, "metal", binary, metal_vae_conv_direct = True) == []
    )


def test_build_without_module_conv_direct_gets_nothing(tmp_path, monkeypatch):
    _mac(monkeypatch)
    binary = _binary(tmp_path, GEMM_TAG, help_text = "  --diffusion-fa   use flash attention\n")
    assert native_speed_flags("off", [], "metal", binary, metal_vae_conv_direct = True) == []


@pytest.mark.parametrize("speed", ["off", "default", "max"])
@pytest.mark.parametrize(
    "platform,accelerator,arm64,paravirtual",
    [
        ("linux", "cuda", True, False),
        ("linux", "vulkan", True, False),
        ("linux", "rocm", True, False),
        ("linux", "cpu", True, False),
        ("win32", "cuda", True, False),
        # Intel Mac (explicit Metal backend) and a macOS VM (CPU backend) never take it.
        ("darwin", "metal", False, False),
        ("darwin", "metal", True, True),
    ],
)
def test_other_platforms_ignore_the_family_default(
    tmp_path, monkeypatch, speed, platform, accelerator, arm64, paravirtual
):
    _mac(monkeypatch, arm64 = arm64, paravirtual = paravirtual)
    monkeypatch.setattr("sys.platform", platform)
    binary = _binary(tmp_path, GEMM_TAG, "sd-server")
    args = ["--backend", "MTL0"] if (platform == "darwin" and not arm64) else []
    base = native_speed_flags(speed, args, accelerator, binary)
    assert native_speed_flags(speed, args, accelerator, binary, metal_vae_conv_direct = True) == base
    assert "vae=on" not in " ".join(base)
    a = [c for c in _server_cmd(binary, accelerator, speed, True) if c != binary]
    b = [c for c in _server_cmd(binary, accelerator, speed, False) if c != binary]
    assert a == b


def test_engine_generate_passes_the_family_default(tmp_path, monkeypatch):
    import core.inference.sd_cpp_engine as sd_cpp_engine
    from core.inference.sd_cpp_engine import SdCppEngine

    _mac(monkeypatch)
    binary = _binary(tmp_path, GEMM_TAG)
    monkeypatch.setattr(sd_cpp_engine, "binary_accelerator", lambda b: "metal")
    eng = SdCppEngine(binary = binary)
    monkeypatch.setattr(eng, "_require_binary", lambda: binary)
    monkeypatch.setattr(eng, "_prepare_out", lambda p: p)
    cmds = []
    monkeypatch.setattr(eng, "_run", lambda cmd, *a, **k: cmds.append(cmd))
    for family_default in (True, False):
        eng.generate(
            SdCppModelFiles(diffusion_model = "/m/x.gguf"),
            SdCppGenParams(prompt = "x"),
            output_path = str(tmp_path / "x.png"),
            native_speed = "off",
            metal_vae_conv_direct = family_default,
        )
    assert _pair(cmds[0], "--conv-direct") == "off,vae=on" and cmds[0].count("--conv-direct") == 1
    assert "--conv-direct" not in cmds[1]


def test_families_on_the_default_share_a_measured_vae():
    on = {f.name for f in _FAMILIES if getattr(f, "sd_cpp_metal_vae_conv_direct", False)}
    assert on == _DEFAULT_FAMILIES
    for fam in _FAMILIES:
        if fam.name in on:
            assert fam.sd_cpp_vae in _MEASURED_VAES, fam.name


def test_backend_start_kwargs_follow_the_family(monkeypatch):
    from core.inference.sd_cpp_backend import _family_start_kwargs

    # Off macOS the kwargs stay exactly as before.
    for repo in ("unsloth/FLUX.2-klein-4B-GGUF", "unsloth/Z-Image-Turbo-GGUF"):
        assert _family_start_kwargs(detect_family(repo)) == {}
    monkeypatch.setattr("sys.platform", "darwin")
    klein = detect_family("unsloth/FLUX.2-klein-4B-GGUF")
    zimage = detect_family("unsloth/Z-Image-Turbo-GGUF")
    qwen = detect_family("unsloth/Qwen-Image-2512-GGUF")
    assert _family_start_kwargs(klein) == {"metal_vae_conv_direct": True}
    assert _family_start_kwargs(zimage) == {"metal_vae_conv_direct": True}
    assert _family_start_kwargs(qwen) == {}
    h3 = detect_family("unsloth/MiniMax-H3-Image-GGUF")
    assert _family_start_kwargs(h3) == {"rng": "cpu"}


def test_status_reports_the_family_default(tmp_path, monkeypatch):
    from core.inference.sd_cpp_backend import SdCppDiffusionBackend

    backend = SdCppDiffusionBackend.__new__(SdCppDiffusionBackend)
    backend._engine = None
    server = SimpleNamespace(binary = _binary(tmp_path, GEMM_TAG, "sd-server"))
    klein = detect_family("unsloth/FLUX.2-klein-4B-GGUF")
    qwen = detect_family("unsloth/Qwen-Image-2512-GGUF")
    _mac(monkeypatch)
    for speed in ("off", "default"):
        state = SimpleNamespace(offload_flags = (), server = server, native_speed = speed, family = klein)
        got = backend._metal_vae_conv_status(state)["metal_vae_conv_direct"]
        assert got["enabled"] is True, (speed, got)
        state.family = qwen
        assert backend._metal_vae_conv_status(state)["metal_vae_conv_direct"]["enabled"] is False
    monkeypatch.setenv(_ENV, "0")
    state = SimpleNamespace(offload_flags = (), server = server, native_speed = "off", family = klein)
    got = backend._metal_vae_conv_status(state)["metal_vae_conv_direct"]
    assert got["enabled"] is False and _ENV in got["reason"]
