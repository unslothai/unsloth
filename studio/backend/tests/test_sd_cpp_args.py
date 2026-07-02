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
    SdCppUpscaleParams,
    build_sd_cpp_command,
    build_sd_cpp_upscale_command,
    native_speed_flags,
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


def test_native_speed_flags():
    assert native_speed_flags(None) == []
    assert native_speed_flags("off") == []
    assert native_speed_flags("") == []
    assert native_speed_flags("default") == ["--diffusion-fa"]
    assert native_speed_flags("max") == ["--diffusion-fa", "--diffusion-conv-direct"]
    with pytest.raises(ValueError):
        native_speed_flags("ludicrous")


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


# ── img2img / inpaint / edit / LoRA (Phase 6) ───────────────────────────────


def test_build_img2img_adds_init_and_strength():
    files = SdCppModelFiles(diffusion_model = "/m/z.gguf", vae = "/m/ae.sft", llm = "/m/q.gguf")
    params = SdCppGenParams(prompt = "make it autumn", init_img = "/in/src.png", strength = 0.6)
    cmd = build_sd_cpp_command("/bin/sd-cli", files, params, output_path = "/o.png")
    assert _pair(cmd, "--init-img") == "/in/src.png"
    assert _pair(cmd, "--strength") == "0.6"
    assert _pair(cmd, "--mode") == "img_gen"  # img2img is still img_gen mode


def test_build_inpaint_adds_mask():
    files = SdCppModelFiles(diffusion_model = "/m/z.gguf")
    params = SdCppGenParams(prompt = "x", init_img = "/in/src.png", mask = "/in/mask.png", strength = 0.8)
    cmd = build_sd_cpp_command("/bin/sd-cli", files, params, output_path = "/o.png")
    assert _pair(cmd, "--mask") == "/in/mask.png"
    assert _pair(cmd, "--init-img") == "/in/src.png"


def test_build_inpaint_mask_without_init_img_rejected():
    # sd-cli inpaint needs a source image; a --mask with no --init-img is invalid argv,
    # so the builder must reject it up front instead of emitting a doomed command.
    files = SdCppModelFiles(diffusion_model = "/m/z.gguf")
    params = SdCppGenParams(prompt = "x", mask = "/in/mask.png")
    with pytest.raises(ValueError, match = "init_img is required"):
        build_sd_cpp_command("/bin/sd-cli", files, params, output_path = "/o.png")


def test_build_rejects_none_prompt():
    # A None prompt must be rejected, not coerced to the literal string "None" and
    # forwarded into argv.
    files = SdCppModelFiles(diffusion_model = "/m/z.gguf")
    with pytest.raises(ValueError, match = "prompt is required"):
        build_sd_cpp_command(
            "/bin/sd-cli", files, SdCppGenParams(prompt = None), output_path = "/o.png"
        )


def test_build_edit_repeats_ref_image():
    files = SdCppModelFiles(diffusion_model = "/m/flux.gguf")
    params = SdCppGenParams(prompt = "add a hat", ref_images = ("/r/a.png", "/r/b.png"))
    cmd = build_sd_cpp_command("/bin/sd-cli", files, params, output_path = "/o.png")
    # each ref image gets its own --ref-image flag
    idxs = [i for i, t in enumerate(cmd) if t == "--ref-image"]
    assert len(idxs) == 2
    assert [cmd[i + 1] for i in idxs] == ["/r/a.png", "/r/b.png"]


def test_img2img_unset_dims_lets_sdcpp_derive_from_source():
    # img2img/inpaint/edit with dims left unset must NOT force --width/--height,
    # so sd.cpp derives the size from the input image instead of resizing it to 1024.
    files = SdCppModelFiles(diffusion_model = "/m/z.gguf")
    cmd = build_sd_cpp_command(
        "/bin/sd-cli",
        files,
        SdCppGenParams(prompt = "x", init_img = "/in/src.png"),
        output_path = "/o.png",
    )
    assert "--width" not in cmd and "--height" not in cmd
    # an edit (ref-image) run derives its size too
    cmd2 = build_sd_cpp_command(
        "/bin/sd-cli",
        files,
        SdCppGenParams(prompt = "x", ref_images = ("/r/a.png",)),
        output_path = "/o.png",
    )
    assert "--width" not in cmd2 and "--height" not in cmd2


def test_img2img_explicit_dims_are_emitted():
    files = SdCppModelFiles(diffusion_model = "/m/z.gguf")
    cmd = build_sd_cpp_command(
        "/bin/sd-cli",
        files,
        SdCppGenParams(prompt = "x", init_img = "/in/src.png", width = 768, height = 512),
        output_path = "/o.png",
    )
    assert _pair(cmd, "--width") == "768" and _pair(cmd, "--height") == "512"


def test_txt2img_unset_dims_keep_1024_default():
    # A plain txt2img run with no dims keeps the prior 1024x1024 default.
    files = SdCppModelFiles(diffusion_model = "/m/z.gguf")
    cmd = build_sd_cpp_command(
        "/bin/sd-cli", files, SdCppGenParams(prompt = "x"), output_path = "/o.png"
    )
    assert _pair(cmd, "--width") == "1024" and _pair(cmd, "--height") == "1024"


def test_build_lora_dir_and_apply_mode():
    files = SdCppModelFiles(diffusion_model = "/m/z.gguf")
    params = SdCppGenParams(
        prompt = "a portrait <lora:mystyle:0.8>",
        lora_dir = "/loras",
        lora_apply_mode = "at_runtime",
    )
    cmd = build_sd_cpp_command("/bin/sd-cli", files, params, output_path = "/o.png")
    assert _pair(cmd, "--lora-model-dir") == "/loras"
    assert _pair(cmd, "--lora-apply-mode") == "at_runtime"
    # the <lora:...> tag rides in the prompt unchanged
    assert _pair(cmd, "--prompt") == "a portrait <lora:mystyle:0.8>"


def test_txt2img_omits_image_conditioning_flags():
    files = SdCppModelFiles(diffusion_model = "/m/z.gguf")
    cmd = build_sd_cpp_command(
        "/bin/sd-cli", files, SdCppGenParams(prompt = "x"), output_path = "/o.png"
    )
    for flag in ("--init-img", "--strength", "--mask", "--ref-image", "--lora-model-dir"):
        assert flag not in cmd


# ── upscale mode ────────────────────────────────────────────────────────────


def test_build_upscale_command():
    params = SdCppUpscaleParams(
        input_image = "/in/small.png", upscale_model = "/m/esrgan.pth", repeats = 2
    )
    cmd = build_sd_cpp_upscale_command("/bin/sd-cli", params, output_path = "/out/big.png")
    assert _pair(cmd, "--mode") == "upscale"
    assert _pair(cmd, "--init-img") == "/in/small.png"
    assert _pair(cmd, "--upscale-model") == "/m/esrgan.pth"
    assert _pair(cmd, "--upscale-repeats") == "2"
    assert _pair(cmd, "--output") == "/out/big.png"
    # no prompt / text-encoder flags in upscale mode
    assert "--prompt" not in cmd and "--llm" not in cmd


def test_build_upscale_rejects_non_positive_repeats():
    # repeats=0 must not be silently swallowed into sd-cli's default of one pass.
    with pytest.raises(ValueError, match = "repeats"):
        build_sd_cpp_upscale_command(
            "/bin/sd-cli",
            SdCppUpscaleParams(input_image = "/i.png", upscale_model = "/m/e.pth", repeats = 0),
            output_path = "/o.png",
        )


def test_build_upscale_default_repeats_omits_flag():
    cmd = build_sd_cpp_upscale_command(
        "/bin/sd-cli",
        SdCppUpscaleParams(input_image = "/i.png", upscale_model = "/m/e.pth"),  # repeats=1
        output_path = "/o.png",
    )
    assert "--upscale-repeats" not in cmd


def test_build_upscale_requires_input_and_model():
    with pytest.raises(ValueError):
        build_sd_cpp_upscale_command(
            "/bin/sd-cli",
            SdCppUpscaleParams(input_image = "", upscale_model = "/m/e.pth"),
            output_path = "/o.png",
        )
    with pytest.raises(ValueError):
        build_sd_cpp_upscale_command(
            "/bin/sd-cli",
            SdCppUpscaleParams(input_image = "/i.png", upscale_model = ""),
            output_path = "/o.png",
        )
