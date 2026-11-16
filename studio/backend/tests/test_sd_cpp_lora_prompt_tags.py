# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A <lora:...> tag typed into a prompt must never reach sd-cli unless the account selected that adapter."""

from __future__ import annotations

import re

import pytest

from core.inference import diffusion_lora as dl
from core.inference.sd_cpp_args import (
    SdCppGenParams,
    SdCppModelFiles,
    SdCppVideoGenParams,
    build_sd_cpp_command,
    build_sd_cpp_video_command,
)

_SDCLI_RE = re.compile(r"<lora:([^:>]+):([^>]+)>")

_TAGS = [
    "<lora:/victim/bob-private:1>",
    "<lora:|high_noise|/victim/bob-private:1>",
    "<lora:../../victim/bob-private:1>",
    # Split tags that reassemble after a single-pass strip.
    "<lo<lora:x:1>ra:/victim/bob-private:1>",
    "<lora<lora:x:1>:/victim/bob-private:1>",
]


def _sdcli_lora_names(prompt: str) -> list[str]:
    """Mirrors sd-cli's extract_and_remove_lora: search, record, drop the first match, repeat."""
    names = []
    while (m := _SDCLI_RE.search(prompt)) is not None:
        names.append(m.group(1))
        prompt = _SDCLI_RE.sub("", prompt, count = 1)
    return names


def _prompt_arg(cmd: list[str]) -> str:
    return cmd[cmd.index("--prompt") + 1]


def _video_cmd(prompt: str) -> list[str]:
    files = SdCppModelFiles(
        diffusion_model = "/m/h3.gguf", vae = "/m/vae.safetensors", llm = "/m/te.gguf"
    )
    params = SdCppVideoGenParams(
        prompt = prompt, width = 512, height = 512, num_frames = 25, ref_audios = ("/s/a.wav",)
    )
    return build_sd_cpp_video_command("sd-cli", files, params, output_path = "/s/out.webm")


def _image_cmd(prompt: str, lora_dir = None) -> list[str]:
    params = SdCppGenParams(prompt = prompt, lora_dir = lora_dir)
    return build_sd_cpp_command(
        "sd-cli", SdCppModelFiles(diffusion_model = "/m/z.gguf"), params, output_path = "/s/o.png"
    )


@pytest.mark.parametrize("tag", _TAGS)
def test_video_argv_drops_prompt_lora_tags(tag):
    assert _sdcli_lora_names(_prompt_arg(_video_cmd(f"a cat {tag} running"))) == []


@pytest.mark.parametrize("tag", _TAGS)
def test_image_argv_without_selected_lora_drops_prompt_tags(tag):
    assert _sdcli_lora_names(_prompt_arg(_image_cmd(f"a cat {tag} running"))) == []


@pytest.mark.parametrize("tag", _TAGS)
def test_selected_lora_keeps_only_the_validated_tag(tag):
    r = dl.ResolvedLora("style", "style", "/x/style.safetensors", "safetensors", 0.8)
    prompt = dl.inject_prompt_tags(f"a cat {tag}", [r])
    cmd = _image_cmd(prompt, lora_dir = "/tmp/run/loras")
    assert _sdcli_lora_names(_prompt_arg(cmd)) == ["style"]
    assert cmd[cmd.index("--lora-model-dir") + 1] == "/tmp/run/loras"


def test_dangling_tag_opener_cannot_pair_with_the_injected_tag():
    r = dl.ResolvedLora("style", "style", "/x/style.safetensors", "safetensors", 1.0)
    assert _sdcli_lora_names(dl.inject_prompt_tags("a cat <lora:/victim/x", [r])) == ["style"]


def test_tag_free_prompt_passes_through_unchanged():
    prompt = "  a cat:  <b>bold</b> lora  "
    assert _prompt_arg(_video_cmd(prompt)) == prompt
    assert _prompt_arg(_image_cmd(prompt)) == prompt
