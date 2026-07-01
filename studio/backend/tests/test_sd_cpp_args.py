# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the sd-cli command builder (``sd_cpp_args.py``).

Pure: no torch, no subprocess, no files. Just argv construction and the
policy -> offload-flag / family -> text-encoder-flag mappings.
"""

from __future__ import annotations

import pytest

from core.inference.diffusion_memory import (
    OFFLOAD_GROUP,
    OFFLOAD_MODEL,
    OFFLOAD_NONE,
    OFFLOAD_SEQUENTIAL,
)
from core.inference.sd_cpp_args import (
    SdCppGenParams,
    SdCppModelFiles,
    build_sd_cpp_command,
    offload_flags,
    text_encoder_flags_for_family,
)


def _pair(cmd: list[str], flag: str):
    """Value following ``flag`` in ``cmd``, or None if the flag is absent."""
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


# ── family text-encoder wiring ──────────────────────────────────────────────


def test_te_flags_by_family():
    assert text_encoder_flags_for_family("z-image") == ("--llm",)
    assert text_encoder_flags_for_family("qwen-image") == ("--qwen2vl",)
    assert text_encoder_flags_for_family("flux.1") == ("--clip_l", "--t5xxl")
    assert text_encoder_flags_for_family("flux.2-klein") == ("--llm",)
    assert text_encoder_flags_for_family("unknown") == ()


# ── offload policy -> sd-cli flags ──────────────────────────────────────────


def test_offload_none_is_empty():
    assert offload_flags(OFFLOAD_NONE) == []


def test_offload_group_streams_with_flash_attention():
    flags = offload_flags(OFFLOAD_GROUP)
    assert "--offload-to-cpu" in flags
    assert "--diffusion-fa" in flags
    # group keeps CLIP/VAE resident -> no per-component cpu flags
    assert "--clip-on-cpu" not in flags
    assert "--vae-on-cpu" not in flags


def test_offload_model_pushes_everything_to_cpu_and_tiles():
    flags = offload_flags(OFFLOAD_MODEL)
    for expected in (
        "--offload-to-cpu",
        "--clip-on-cpu",
        "--vae-on-cpu",
        "--vae-tiling",
        "--diffusion-fa",
    ):
        assert expected in flags
    # sequential maps the same as model
    assert offload_flags(OFFLOAD_SEQUENTIAL) == flags


def test_offload_forced_flags_dedup():
    # vae_tiling/diffusion_fa forced on with a policy that already sets them
    flags = offload_flags(OFFLOAD_MODEL, vae_tiling = True, diffusion_fa = True)
    assert flags.count("--vae-tiling") == 1
    assert flags.count("--diffusion-fa") == 1
    # forced on top of a no-offload policy
    none_forced = offload_flags(OFFLOAD_NONE, vae_tiling = True, diffusion_fa = True)
    assert none_forced == ["--diffusion-fa", "--vae-tiling"]


# ── full command construction ───────────────────────────────────────────────


def test_build_zimage_command_minimal():
    files = SdCppModelFiles(
        diffusion_model = "/m/z.gguf",
        vae = "/m/ae.sft",
        llm = "/m/qwen3.gguf",
    )
    params = SdCppGenParams(prompt = "a cat", width = 512, height = 768, steps = 8, cfg_scale = 1.0, seed = 42)
    cmd = build_sd_cpp_command("/bin/sd-cli", files, params, output_path = "/out/x.png")

    assert cmd[0] == "/bin/sd-cli"
    assert _pair(cmd, "--mode") == "img_gen"
    assert _pair(cmd, "--diffusion-model") == "/m/z.gguf"
    assert _pair(cmd, "--vae") == "/m/ae.sft"
    assert _pair(cmd, "--llm") == "/m/qwen3.gguf"
    assert _pair(cmd, "--prompt") == "a cat"
    assert _pair(cmd, "--width") == "512"
    assert _pair(cmd, "--height") == "768"
    assert _pair(cmd, "--steps") == "8"
    assert _pair(cmd, "--cfg-scale") == "1"  # 1.0 -> "1"
    assert _pair(cmd, "--seed") == "42"
    assert _pair(cmd, "--output") == "/out/x.png"
    # unset encoders are omitted
    assert "--t5xxl" not in cmd
    assert "--qwen2vl" not in cmd


def test_build_flux1_dual_text_encoders():
    files = SdCppModelFiles(
        diffusion_model = "/m/flux.gguf",
        vae = "/m/ae.sft",
        clip_l = "/m/clip_l.sft",
        t5xxl = "/m/t5.gguf",
    )
    params = SdCppGenParams(prompt = "x", guidance = 3.5)
    cmd = build_sd_cpp_command("/bin/sd-cli", files, params, output_path = "/o.png")
    assert _pair(cmd, "--clip_l") == "/m/clip_l.sft"
    assert _pair(cmd, "--t5xxl") == "/m/t5.gguf"
    assert _pair(cmd, "--guidance") == "3.5"
    assert "--llm" not in cmd


def test_build_appends_offload_and_extra_args_last():
    files = SdCppModelFiles(diffusion_model = "/m/z.gguf")
    params = SdCppGenParams(prompt = "x")
    off = offload_flags(OFFLOAD_GROUP)
    cmd = build_sd_cpp_command(
        "/bin/sd-cli",
        files,
        params,
        output_path = "/o.png",
        offload = off,
        threads = 8,
        verbose = True,
        extra_args = ["--rng", "cuda"],
    )
    assert "--offload-to-cpu" in cmd
    assert _pair(cmd, "--threads") == "8"
    assert "-v" in cmd
    # extra args come after everything Studio set (last-wins for power users)
    assert cmd[-2:] == ["--rng", "cuda"]


def test_build_negative_prompt_and_batch():
    files = SdCppModelFiles(diffusion_model = "/m/z.gguf")
    params = SdCppGenParams(prompt = "x", negative_prompt = "blurry", batch_count = 3)
    cmd = build_sd_cpp_command("/bin/sd-cli", files, params, output_path = "/o.png")
    assert _pair(cmd, "--negative-prompt") == "blurry"
    assert _pair(cmd, "--batch-count") == "3"


def test_build_omits_unset_optional_params():
    files = SdCppModelFiles(diffusion_model = "/m/z.gguf")
    params = SdCppGenParams(prompt = "x")  # no steps/cfg/seed/sampler
    cmd = build_sd_cpp_command("/bin/sd-cli", files, params, output_path = "/o.png")
    for flag in (
        "--steps",
        "--cfg-scale",
        "--guidance",
        "--seed",
        "--sampling-method",
        "--batch-count",
        "--threads",
        "-v",
    ):
        assert flag not in cmd


def test_build_requires_diffusion_model_and_prompt():
    with pytest.raises(ValueError):
        build_sd_cpp_command(
            "/bin/sd-cli",
            SdCppModelFiles(diffusion_model = ""),
            SdCppGenParams(prompt = "x"),
            output_path = "/o.png",
        )
    with pytest.raises(ValueError):
        build_sd_cpp_command(
            "/bin/sd-cli",
            SdCppModelFiles(diffusion_model = "/m/z.gguf"),
            SdCppGenParams(prompt = "   "),
            output_path = "/o.png",
        )
