# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Metal VAE direct conv: off by default. ``UNSLOTH_SD_CPP_METAL_VAE_CONV_DIRECT=1`` opts in, and then only an Apple
Silicon Mac on a build with the implicit GEMM Metal CONV_2D (unslothai/stable-diffusion.cpp#103) gets
``--conv-direct off,vae=on``; Intel Macs, older builds and every other backend keep their argv."""

from __future__ import annotations

import json
import stat
import sys

import pytest

import core.inference.sd_cpp_args as sd_cpp_args
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
# First Unsloth prebuilt carrying ggml patches 0072 / 0073 (the fork's master at 9455ee1, no pinned PRs).
GEMM_TAG = "master-982-0518b34-u9455ee1"
SCALAR_TAG = "master-982-0518b34-u27b9ddc"
UPSTREAM_TAG = "master-982-0518b34"
_ENV = "UNSLOTH_SD_CPP_METAL_VAE_CONV_DIRECT"


def _binary(
    tmp_path,
    tag,
    name = "sd-cli",
):
    root = tmp_path / (tag or "unrecorded")
    root.mkdir(exist_ok = True)
    if tag:
        (root / ".unsloth-sd-cpp-install.json").write_text(json.dumps({"tag": tag}))
    path = root / name
    path.write_text("#!/bin/sh\ncat <<'EOF'\n" + _HELP + "EOF\n")
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


def _metal(
    monkeypatch,
    paravirtual = False,
    opt_in = True,
    arm64 = True,
):
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setattr(sd_cpp_args, "apple_silicon_machine", lambda: arm64)
    if opt_in:
        monkeypatch.setenv(_ENV, "1")
    monkeypatch.setattr(sd_cpp_args, "metal_is_paravirtual", lambda: paravirtual)


def _server_cmd(
    binary,
    accelerator,
    speed = "default",
):
    return build_sd_cpp_server_command(
        binary = binary,
        files = SdCppModelFiles(diffusion_model = "/m/x.gguf", vae = "/m/vae.safetensors"),
        host = "127.0.0.1",
        port = 1234,
        native_speed = speed,
        accelerator = accelerator,
    )


def _pair(cmd, flag):
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


# Off Metal the argv is the same whichever build is installed (the pre-change argv for each backend).
_NON_METAL = [
    ("cuda", "default", ["--diffusion-fa", "--diffusion-conv-direct"]),
    ("cuda", "max", ["--diffusion-fa", "--diffusion-conv-direct"]),
    ("cpu", "default", ["--diffusion-fa", "--diffusion-conv-direct"]),
    ("vulkan", "default", ["--diffusion-fa", "--conv-direct", "off,vae=auto"]),
    ("rocm", "default", ["--diffusion-fa", "--conv-direct", "off"]),
    ("hip", "default", ["--diffusion-fa", "--conv-direct", "off"]),
    ("cuda", "off", []),
]


@pytest.mark.parametrize("accelerator,speed,flags", _NON_METAL)
def test_non_metal_argv_is_unchanged_by_the_build(tmp_path, accelerator, speed, flags):
    cmds = []
    for tag in (GEMM_TAG, SCALAR_TAG, None):
        binary = _binary(tmp_path, tag, "sd-server")
        assert native_speed_flags(speed, [], accelerator, binary) == flags
        cmd = _server_cmd(binary, accelerator, speed)
        cmds.append([c for c in cmd if c != binary])
    assert cmds[0] == cmds[1] == cmds[2]
    assert "vae=on" not in " ".join(cmds[0])


@pytest.mark.parametrize(
    "tag,backend_args,mode",
    [
        (GEMM_TAG, [], "off,vae=on"),
        (GEMM_TAG, ["--clip-on-cpu"], "off,vae=on"),
        (GEMM_TAG, ["--backend", "MTL0"], "off,vae=on"),
        (GEMM_TAG, ["--backend", "diffusion=MTL0,te=cpu,vae=MTL0"], "off,vae=on"),
        # A CPU VAE is not the Metal kernel.
        (GEMM_TAG, ["--vae-on-cpu"], "off"),
        (GEMM_TAG, ["--backend", "diffusion=MTL0,vae=cpu"], "off"),
        # Builds with the scalar Metal kernel, upstream fallbacks and unrecorded binaries keep im2col.
        (SCALAR_TAG, [], "off"),
        ("master-982-0518b34-uaba3a1c", [], "off"),
        ("master-975-1a2b3c4-u1234567", [], "off"),
        (UPSTREAM_TAG, [], "off"),
        (None, [], "off"),
        # A later Unsloth prebuilt keeps the kernel (the patches stay in the fork's tree).
        ("master-990-abcdef1-u7654321", [], "off,vae=on"),
    ],
)
def test_metal_vae_direct_conv_table(tmp_path, monkeypatch, tag, backend_args, mode):
    # Opted in on Apple Silicon: the build and the VAE placement decide.
    _metal(monkeypatch)
    binary = _binary(tmp_path, tag)
    assert native_speed_flags("default", backend_args, "metal", binary) == [
        "--diffusion-fa",
        "--conv-direct",
        mode,
    ]
    assert native_speed_flags("max", backend_args, None, binary)[-1] == mode


@pytest.mark.parametrize("tag", [GEMM_TAG, "master-990-abcdef1-u7654321", None])
@pytest.mark.parametrize("backend_args", [[], ["--backend", "MTL0"]])
def test_metal_vae_direct_conv_is_off_by_default(tmp_path, monkeypatch, tag, backend_args):
    # No physical Apple Silicon measurement yet: without the opt-in every build keeps im2col.
    _metal(monkeypatch, opt_in = False)
    binary = _binary(tmp_path, tag)
    assert native_speed_flags("default", backend_args, "metal", binary)[-1] == "off"
    assert (
        _pair(_server_cmd(_binary(tmp_path, tag, "sd-server"), "metal"), "--conv-direct") == "off"
    )
    on, reason = sd_cpp_args.metal_vae_conv_direct_decision(backend_args, binary)
    assert not on and _ENV in reason


@pytest.mark.parametrize("value", ["1", "force"])
def test_intel_mac_never_gets_metal_vae_direct_conv(tmp_path, monkeypatch, value):
    # x86_64 macOS with an explicit Metal backend: the scalar CONV_2D there tripped the GPU watchdog.
    _metal(monkeypatch, arm64 = False)
    monkeypatch.setenv(_ENV, value)
    binary = _binary(tmp_path, GEMM_TAG)
    assert native_speed_flags("default", ["--backend", "MTL0"], "metal", binary)[-1] == "off"
    on, reason = sd_cpp_args.metal_vae_conv_direct_decision(["--backend", "MTL0"], binary)
    assert not on and "Apple Silicon" in reason


@pytest.mark.parametrize("value", ["1", "force"])
def test_non_metal_ignores_the_opt_in(tmp_path, monkeypatch, value):
    monkeypatch.setenv(_ENV, value)
    for accelerator, speed, flags in _NON_METAL:
        assert native_speed_flags(speed, [], accelerator, _binary(tmp_path, GEMM_TAG)) == flags


def test_opt_in_decision_is_logged(tmp_path, monkeypatch, caplog):
    import logging

    _metal(monkeypatch)
    monkeypatch.setattr(sd_cpp_args, "_METAL_VAE_CONV_LOGGED", set())
    with caplog.at_level(logging.INFO, logger = sd_cpp_args.__name__):
        native_speed_flags("default", [], "metal", _binary(tmp_path, GEMM_TAG))
        native_speed_flags("default", [], "metal", _binary(tmp_path, GEMM_TAG))
    lines = [r.getMessage() for r in caplog.records if "Metal VAE direct conv" in r.getMessage()]
    assert lines == ["sd.cpp Metal VAE direct conv: on: opted in, Apple Silicon, VAE on Metal"]


def test_metal_server_command_carries_vae_direct_conv(tmp_path, monkeypatch):
    _metal(monkeypatch)
    cmd = _server_cmd(_binary(tmp_path, GEMM_TAG, "sd-server"), "metal")
    assert _pair(cmd, "--conv-direct") == "off,vae=on"
    assert cmd.count("--conv-direct") == 1
    assert "--diffusion-conv-direct" not in cmd and "--vae-conv-direct" not in cmd
    old = _server_cmd(_binary(tmp_path, SCALAR_TAG, "sd-server"), "metal")
    assert _pair(old, "--conv-direct") == "off"


def test_metal_engine_generate_carries_vae_direct_conv(tmp_path, monkeypatch):
    import core.inference.sd_cpp_engine as sd_cpp_engine
    from core.inference.sd_cpp_engine import SdCppEngine

    _metal(monkeypatch)
    binary = _binary(tmp_path, GEMM_TAG)
    monkeypatch.setattr(sd_cpp_engine, "binary_accelerator", lambda b: "metal")
    captured = {}
    eng = SdCppEngine(binary = binary)
    monkeypatch.setattr(eng, "_require_binary", lambda: binary)
    monkeypatch.setattr(eng, "_prepare_out", lambda p: p)
    monkeypatch.setattr(eng, "_run", lambda cmd, *a, **k: captured.setdefault("cmd", cmd))
    eng.generate(
        SdCppModelFiles(diffusion_model = "/m/x.gguf"),
        SdCppGenParams(prompt = "x"),
        output_path = str(tmp_path / "x.png"),
        native_speed = "default",
    )
    assert _pair(captured["cmd"], "--conv-direct") == "off,vae=on"
    assert captured["cmd"].count("--conv-direct") == 1


@pytest.mark.parametrize("value", ["0", "off", "false"])
def test_metal_vae_kill_switch(tmp_path, monkeypatch, value):
    _metal(monkeypatch)
    monkeypatch.setenv(_ENV, value)
    assert native_speed_flags("default", [], "metal", _binary(tmp_path, GEMM_TAG))[-1] == "off"


def test_global_conv_direct_off_also_keeps_metal_vae_off(tmp_path, monkeypatch):
    _metal(monkeypatch)
    monkeypatch.setenv("UNSLOTH_SD_CPP_CONV_DIRECT", "0")
    assert native_speed_flags("default", [], "metal", _binary(tmp_path, GEMM_TAG))[-1] == "off"


def test_metal_vae_force_on_for_an_unrecorded_build(tmp_path, monkeypatch):
    _metal(monkeypatch)
    # ``1`` needs a build known to carry the kernel; ``force`` skips that check (source builds).
    assert native_speed_flags("default", [], "metal", _binary(tmp_path, None))[-1] == "off"
    monkeypatch.setenv(_ENV, "force")
    assert native_speed_flags("default", [], "metal", _binary(tmp_path, None))[-1] == "off,vae=on"
    # Forcing never moves a CPU VAE onto direct conv.
    assert (
        native_speed_flags("default", ["--vae-on-cpu"], "metal", _binary(tmp_path, None))[-1]
        == "off"
    )


def test_paravirtual_mac_keeps_the_cpu_path(tmp_path, monkeypatch):
    # A macOS VM runs sd.cpp on the CPU backend (``--backend cpu``); that argv does not change.
    _metal(monkeypatch, paravirtual = True)
    cmds = []
    for tag in (GEMM_TAG, SCALAR_TAG):
        binary = _binary(tmp_path, tag, "sd-server")
        cmds.append([c for c in _server_cmd(binary, "metal") if c != binary])
    assert cmds[0] == cmds[1]
    assert "vae=on" not in " ".join(cmds[0])


def test_a_user_conv_direct_flag_still_wins_on_metal(tmp_path, monkeypatch):
    _metal(monkeypatch)
    binary = _binary(tmp_path, GEMM_TAG)
    assert native_speed_flags("default", ["--conv-direct", "off"], "metal", binary) == [
        "--diffusion-fa"
    ]


def test_status_reports_the_resolved_metal_vae_state(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from core.inference.sd_cpp_backend import SdCppDiffusionBackend

    backend = SdCppDiffusionBackend.__new__(SdCppDiffusionBackend)
    backend._engine = None
    server = SimpleNamespace(binary = _binary(tmp_path, GEMM_TAG, "sd-server"))
    state = SimpleNamespace(offload_flags = (), server = server, native_speed = "default")
    # Off Metal the field is absent.
    assert backend._metal_vae_conv_status(state) == {}
    _metal(monkeypatch, opt_in = False)
    got = backend._metal_vae_conv_status(state)["metal_vae_conv_direct"]
    assert got["enabled"] is False and _ENV in got["reason"]
    monkeypatch.setenv(_ENV, "1")
    assert backend._metal_vae_conv_status(state)["metal_vae_conv_direct"]["enabled"] is True
    state.native_speed = "off"
    assert backend._metal_vae_conv_status(state)["metal_vae_conv_direct"]["enabled"] is False


def test_status_follows_the_effective_conv_direct_policy(tmp_path, monkeypatch):
    # Status must agree with the argv: the global kill switch and a user flag both win over the opt-in.
    from types import SimpleNamespace

    from core.inference.sd_cpp_backend import SdCppDiffusionBackend

    backend = SdCppDiffusionBackend.__new__(SdCppDiffusionBackend)
    backend._engine = None
    binary = _binary(tmp_path, GEMM_TAG, "sd-server")
    state = SimpleNamespace(
        offload_flags = (), server = SimpleNamespace(binary = binary), native_speed = "default"
    )
    _metal(monkeypatch)
    monkeypatch.setenv(_ENV, "1")
    assert backend._metal_vae_conv_status(state)["metal_vae_conv_direct"]["enabled"] is True
    monkeypatch.setenv("UNSLOTH_SD_CPP_CONV_DIRECT", "0")
    got = backend._metal_vae_conv_status(state)["metal_vae_conv_direct"]
    assert got["enabled"] is False and "UNSLOTH_SD_CPP_CONV_DIRECT" in got["reason"]
    assert native_speed_flags("default", [], "metal", binary)[-1] == "off"
    monkeypatch.delenv("UNSLOTH_SD_CPP_CONV_DIRECT")
    state.offload_flags = ("--conv-direct", "off")
    got = backend._metal_vae_conv_status(state)["metal_vae_conv_direct"]
    assert got == {"enabled": False, "reason": "user --conv-direct off"}
