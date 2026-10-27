# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Direct conv rejected by Studio's policy must reach sd.cpp as an explicit ``--conv-direct off``.

sd.cpp master-963+ reads an omitted ``--conv-direct`` as auto, and auto turns direct conv on for implicit-GEMM
backends (HIP MMA; Vulkan when its timing probe wins), so leaving the legacy flag out no longer means off."""

from __future__ import annotations

import os
import stat
import sys

import pytest

import core.inference.sd_cpp_args as sd_cpp_args
from core.inference.sd_cpp_args import (
    SdCppGenParams,
    SdCppModelFiles,
    build_sd_cpp_server_command,
    native_speed_flags,
    speed_flags_without,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason = "fake sd.cpp binaries are shell scripts"
)

_NEW_HELP = (
    "  --diffusion-conv-direct   deprecated; use --conv-direct auto,diffusion=on,controlnet=on\n"
    "  --conv-direct             on|off|auto or diffusion/vae/controlnet assignments (default: auto)\n"
)
_OLD_HELP = "  --diffusion-conv-direct   use ggml_conv2d_direct in the diffusion model\n"


def _fake_binary(tmp_path, name, help_text):
    path = tmp_path / name
    path.write_text("#!/bin/sh\ncat <<'EOF'\n" + help_text + "EOF\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv("UNSLOTH_SD_CPP_CONV_DIRECT", raising = False)
    monkeypatch.setattr("sys.platform", "linux")
    sd_cpp_args._conv_direct_in_help.cache_clear()
    yield
    sd_cpp_args._conv_direct_in_help.cache_clear()


def _pair(cmd, flag):
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


@pytest.mark.parametrize("accelerator", ["rocm", "vulkan", "hip"])
def test_rejected_backend_sends_explicit_off_to_an_auto_build(tmp_path, accelerator):
    binary = _fake_binary(tmp_path, "sd-server", _NEW_HELP)
    cmd = build_sd_cpp_server_command(
        binary = binary,
        files = SdCppModelFiles(diffusion_model = "/m/x.gguf"),
        host = "127.0.0.1",
        port = 1234,
        native_speed = "default",
        accelerator = accelerator,
    )
    assert "--diffusion-fa" in cmd
    assert _pair(cmd, "--conv-direct") == "off"
    assert "--diffusion-conv-direct" not in cmd


def test_force_off_override_is_explicit_on_cuda(tmp_path, monkeypatch):
    binary = _fake_binary(tmp_path, "sd-cli", _NEW_HELP)
    assert native_speed_flags("max", [], "cuda", binary) == [
        "--diffusion-fa",
        "--diffusion-conv-direct",
    ]
    monkeypatch.setenv("UNSLOTH_SD_CPP_CONV_DIRECT", "0")
    assert native_speed_flags("max", [], "cuda", binary) == [
        "--diffusion-fa",
        "--conv-direct",
        "off",
    ]


def test_older_build_without_the_flag_keeps_the_omission(tmp_path):
    # Pre-963 builds reject an unknown flag; there an omitted legacy flag already means off.
    binary = _fake_binary(tmp_path, "sd-cli", _OLD_HELP)
    assert native_speed_flags("default", [], "rocm", binary) == ["--diffusion-fa"]
    assert native_speed_flags("default", [], "rocm", str(tmp_path / "missing")) == [
        "--diffusion-fa"
    ]
    assert native_speed_flags("default", [], "rocm") == ["--diffusion-fa"]


@pytest.mark.parametrize("user_flag", [["--conv-direct", "on"], ["--diffusion-conv-direct"]])
def test_a_user_conv_direct_flag_wins(tmp_path, user_flag):
    binary = _fake_binary(tmp_path, "sd-cli", _NEW_HELP)
    assert native_speed_flags("default", user_flag, "rocm", binary) == ["--diffusion-fa"]


def test_speed_off_adds_nothing(tmp_path):
    binary = _fake_binary(tmp_path, "sd-cli", _NEW_HELP)
    assert native_speed_flags("off", [], "rocm", binary) == []


def test_dedupe_keeps_the_flag_and_value_together():
    speed = ["--diffusion-fa", "--conv-direct", "off"]
    # A bare "off" elsewhere in the offload argv must not strip the value from its flag.
    assert speed_flags_without(speed, ["--flash-attn", "off"]) == speed
    assert speed_flags_without(speed, ["--diffusion-fa"]) == ["--conv-direct", "off"]
    assert speed_flags_without(speed, ["--conv-direct", "auto"]) == ["--diffusion-fa"]


def test_engine_generate_sends_explicit_off(tmp_path, monkeypatch):
    import core.inference.sd_cpp_engine as sd_cpp_engine
    from core.inference.sd_cpp_engine import SdCppEngine

    binary = _fake_binary(tmp_path, "sd-cli", _NEW_HELP)
    monkeypatch.setattr(sd_cpp_engine, "binary_accelerator", lambda b: "rocm")
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
    assert _pair(captured["cmd"], "--conv-direct") == "off"
    assert "--diffusion-conv-direct" not in captured["cmd"]
