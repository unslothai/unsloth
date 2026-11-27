# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""sd.cpp Qwen-Image-Edit sends a 384 x 384 ref area on both transports; nothing otherwise."""

from __future__ import annotations

import base64
import io
import math

import pytest
from PIL import Image

from core.inference import sd_cpp_backend as bk
from core.inference.diffusion_families import detect_family
from core.inference.sd_cpp_args import (
    QWEN_EDIT_REF_AREA_ENV,
    QWEN_EDIT_REF_IMAGE_ARGS,
    SdCppModelFiles,
    build_img_gen_request,
    ref_image_args_for_family,
)
from core.inference.sd_cpp_backend import SdCppDiffusionBackend

from .test_sd_cpp_backend import _FakeEngine, _FakeServer

FILES = SdCppModelFiles(
    diffusion_model = "/m/qwen-image-edit-2511-Q8_0.gguf",
    vae = "/m/vae.safetensors",
    llm = "/m/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
    llm_vision = "/m/mmproj-F16.gguf",
)


@pytest.fixture(autouse = True)
def _no_kill_switch(monkeypatch):
    monkeypatch.delenv(QWEN_EDIT_REF_AREA_ENV, raising = False)


def _png(size = (64, 32), color = (200, 10, 10, 255)) -> str:
    buf = io.BytesIO()
    Image.new("RGBA", size, color).save(buf, format = "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _state(
    repo,
    mode = "oneshot",
    server = None,
):
    fam = detect_family(repo)
    return bk._SdState(
        repo_id = repo,
        base_repo = fam.base_repo,
        family = fam,
        device = "cpu",
        files = FILES,
        sampling_method = fam.sd_cpp_sampling_method,
        mode = mode,
        server = server,
    )


def _generate(b, monkeypatch, edit):
    monkeypatch.setattr(SdCppDiffusionBackend, "_native_edit_ready", lambda self, st: True)
    if edit:
        b.generate(prompt = "p", steps = 2, workflow = "edit", width = 256, height = 256, init_image = _png())
    else:
        b.generate(prompt = "p", steps = 2, width = 256, height = 256, seed = 3)


@pytest.mark.parametrize(
    "repo", ["unsloth/Qwen-Image-Edit-2511-GGUF", "unsloth/Qwen-Image-Edit-2509-GGUF"]
)
def test_server_edit_sends_the_384_area(monkeypatch, repo):
    server = _FakeServer("sd-server")
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    b._state = _state(repo, mode = "server", server = server)
    _generate(b, monkeypatch, edit = True)
    assert server.payloads[-1]["ref_image_args"] == QWEN_EDIT_REF_IMAGE_ARGS


def test_oneshot_edit_passes_the_384_area(monkeypatch):
    engine = _FakeEngine()
    seen = {}

    def _capture(files, params, output_path, **kw):
        seen["extra"] = list(kw.get("extra_args") or [])
        Image.new("RGBA", (1, 1), (1, 2, 3, 4)).save(output_path)

    monkeypatch.setattr(engine, "generate", _capture)
    b = SdCppDiffusionBackend(engine = engine)
    b._state = _state("unsloth/Qwen-Image-Edit-2511-GGUF")
    _generate(b, monkeypatch, edit = True)
    i = seen["extra"].index("--ref-image-args")
    assert seen["extra"][i + 1] == QWEN_EDIT_REF_IMAGE_ARGS


def test_text_to_image_and_other_families_send_nothing(monkeypatch):
    server = _FakeServer("sd-server")
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    b._state = _state("unsloth/Qwen-Image-GGUF", mode = "server", server = server)
    _generate(b, monkeypatch, edit = False)
    assert "ref_image_args" not in server.payloads[-1]
    assert ref_image_args_for_family("qwen-image-edit", False) is None
    for fam in ("qwen-image", "qwen-image-2.1", "flux.1-kontext", "minimax-h3-image", None):
        assert ref_image_args_for_family(fam, True) is None


@pytest.mark.parametrize("value", ["0", "off", "false", "no"])
def test_kill_switch_keeps_the_binary_default(monkeypatch, value):
    monkeypatch.setenv(QWEN_EDIT_REF_AREA_ENV, value)
    assert ref_image_args_for_family("qwen-image-edit", True) is None
    server = _FakeServer("sd-server")
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    b._state = _state("unsloth/Qwen-Image-Edit-2511-GGUF", mode = "server", server = server)
    _generate(b, monkeypatch, edit = True)
    assert "ref_image_args" not in server.payloads[-1]


def test_request_builder_only_sends_args_beside_references():
    assert "ref_image_args" not in build_img_gen_request(prompt = "p", ref_image_args = "vlm_size=1")
    req = build_img_gen_request(prompt = "p", ref_images = ["A"], ref_image_args = "vlm_size=1")
    assert req["ref_image_args"] == "vlm_size=1"


def test_the_area_is_384_squared_and_lands_on_diffusers_grid():
    # sd.cpp (min == max, area mode): side = max(28, round(side * sqrt(area / (h * w)) / 28) * 28).
    # QwenImageEditPlusPipeline: CONDITION_IMAGE_SIZE = 384 * 384, then Qwen2-VL smart_resize to 28 px.
    area = int(QWEN_EDIT_REF_IMAGE_ARGS.split("=", 1)[1])
    assert area == 384 * 384
    for h, w, expect in (
        (1024, 1024, (392, 392)),
        (720, 1280, (280, 504)),
        (4096, 4096, (392, 392)),
    ):
        beta = math.sqrt(area / (h * w))
        got = (max(28, round(h * beta / 28) * 28), max(28, round(w * beta / 28) * 28))
        assert got == expect
