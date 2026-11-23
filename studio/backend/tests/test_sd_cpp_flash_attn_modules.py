# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""--diffusion-fa on a build with per-module --flash-attn means "off,diffusion=on": text-encoder flash attention off.
The managed flags must carry the module form instead, so the text encoder keeps flash attention."""

import pytest

import core.inference.sd_cpp_args as sd_cpp_args
from core.inference.sd_cpp_args import (
    OFFLOAD_GROUP,
    SdCppGenParams,
    SdCppModelFiles,
    SdCppVideoGenParams,
    build_sd_cpp_command,
    build_sd_cpp_server_command,
    build_sd_cpp_video_command,
    offload_flags,
)

FLASH_ATTN_MODULES = "off,diffusion=on,te=auto"


def _clear_help_cache():
    cache = getattr(sd_cpp_args, "_flash_attn_modules_in_help", None)
    if cache is not None:
        cache.cache_clear()


# The pinned fork's sd-cli / sd-server --help line (examples/common/common.cpp) and the deprecated alias.
MODULE_HELP = (
    "  --flash-attn                             on|off|auto or module assignments (default: auto), e.g. "
    "auto,te=off,vae=on. Last repeated option wins\n"
    "  --diffusion-fa                           deprecated; use --flash-attn off,diffusion=on\n"
)
LEGACY_HELP = (
    "  --diffusion-fa                           use flash attention in the diffusion model\n"
)


@pytest.fixture
def binary(tmp_path, monkeypatch):
    def make(help_text):
        _clear_help_cache()
        path = tmp_path / f"sd-{abs(hash(help_text))}"
        path.write_text("#!/bin/sh\n")
        monkeypatch.setattr(
            "core.inference.sd_cpp_backend._sd_cpp_probe_output",
            lambda b, *a: help_text if b == str(path) else None,
        )
        return str(path)

    yield make
    _clear_help_cache()


FILES = SdCppModelFiles(diffusion_model = "/m/x.gguf", llm = "/m/te.gguf", vae = "/m/ae.safetensors")


def _flash_attn_values(cmd):
    return [cmd[i + 1] for i, f in enumerate(cmd[:-1]) if f == "--flash-attn"]


def test_server_speed_flag_keeps_text_encoder_flash_attention(binary):
    b = binary(MODULE_HELP)
    cmd = build_sd_cpp_server_command(b, FILES, host = "127.0.0.1", port = 1, native_speed = "default")
    assert "--diffusion-fa" not in cmd
    assert _flash_attn_values(cmd) == [FLASH_ATTN_MODULES]


def test_offload_and_speed_give_one_flash_attn(binary):
    b = binary(MODULE_HELP)
    offload = offload_flags(OFFLOAD_GROUP)
    assert "--diffusion-fa" in offload
    cmd = build_sd_cpp_server_command(
        b, FILES, host = "127.0.0.1", port = 1, offload = offload, native_speed = "default"
    )
    assert "--diffusion-fa" not in cmd and _flash_attn_values(cmd) == [FLASH_ATTN_MODULES]
    assert "--offload-to-cpu" in cmd


def test_sd_cli_image_and_video_offload_take_the_module_form(binary):
    b = binary(MODULE_HELP)
    offload = offload_flags(OFFLOAD_GROUP)
    img = build_sd_cpp_command(
        b, FILES, SdCppGenParams(prompt = "x"), output_path = "/o/x.png", offload = offload
    )
    assert "--diffusion-fa" not in img and _flash_attn_values(img) == [FLASH_ATTN_MODULES]
    vid = build_sd_cpp_video_command(
        b,
        FILES,
        SdCppVideoGenParams(prompt = "x", width = 256, height = 256, num_frames = 9),
        output_path = "/o/x.avi",
        offload = offload,
    )
    assert "--diffusion-fa" not in vid and _flash_attn_values(vid) == [FLASH_ATTN_MODULES]


def test_engine_generate_rewrites_managed_flags_and_keeps_user_args(binary, monkeypatch):
    from core.inference.sd_cpp_engine import SdCppEngine

    b = binary(MODULE_HELP)
    eng = SdCppEngine(binary = b)
    captured = {}
    monkeypatch.setattr(eng, "_require_binary", lambda: b)
    monkeypatch.setattr(eng, "_prepare_out", lambda p: p)
    monkeypatch.setattr(eng, "_run", lambda cmd, *a, **k: captured.setdefault("cmd", cmd))
    eng.generate(FILES, SdCppGenParams(prompt = "x"), output_path = "/o/x.png", native_speed = "default")
    cmd = captured["cmd"]
    assert "--diffusion-fa" not in cmd and _flash_attn_values(cmd) == [FLASH_ATTN_MODULES]

    captured.clear()
    eng.generate(
        FILES,
        SdCppGenParams(prompt = "x"),
        output_path = "/o/x.png",
        native_speed = "default",
        extra_args = ["--flash-attn", "on"],
    )
    # last-wins parser: the user's own value comes after the managed one
    assert _flash_attn_values(captured["cmd"])[-1] == "on"


def test_builds_without_module_syntax_keep_diffusion_fa(binary):
    b = binary(LEGACY_HELP)
    cmd = build_sd_cpp_server_command(b, FILES, host = "127.0.0.1", port = 1, native_speed = "default")
    assert "--diffusion-fa" in cmd and "--flash-attn" not in cmd


def test_unknown_binary_keeps_diffusion_fa():
    from core.inference.sd_cpp_args import with_module_flash_attn

    assert with_module_flash_attn(["--diffusion-fa"], None) == ["--diffusion-fa"]
    assert with_module_flash_attn(["--diffusion-fa"], "/does/not/exist") == ["--diffusion-fa"]
    assert with_module_flash_attn(["--offload-to-cpu"], "/does/not/exist") == ["--offload-to-cpu"]
