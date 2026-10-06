# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Native (sd.cpp) route for Qwen-Image-Layered: the asset mapping, the layer count on the one-shot
argv and in the sd-server body, the canvas the input is decomposed at, and the layers + 1 images
sd.cpp decodes, of which the first (its reconstruction of the input) is dropped as the diffusers
pipeline drops it."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest
from PIL import Image

from core.inference import sd_cpp_backend as bk
from core.inference.diffusion_families import (
    detect_family,
    detect_family_for_pick,
    family_sd_cpp_supported,
)
from core.inference.sd_cpp_args import (
    SdCppGenParams,
    SdCppModelFiles,
    build_img_gen_request,
    build_sd_cpp_command,
    sd_cli_output_paths,
    text_encoder_flags_for_family,
)
from core.inference.sd_cpp_backend import SdCppDiffusionBackend

from .test_sd_cpp_backend import _FakeEngine, _FakeServer

LAYERED = detect_family("unsloth/Qwen-Image-Layered-GGUF")
FILES = SdCppModelFiles(
    diffusion_model = "/m/qwen-image-layered-Q4_K_M.gguf",
    vae = "/m/qwen_image_layered_vae.safetensors",
    qwen2vl = "/m/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
)


class _LayerServer(_FakeServer):
    """A server answering with the blobs a test set: sd.cpp's layers + 1 images per generation."""

    next_blobs: list = []

    def img_gen(self, payload, **kw):
        super().img_gen(payload, **kw)
        return list(self.next_blobs)


def _png(size = (900, 600), color = (200, 10, 10)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format = "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _blob(color) -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", (4, 4), color).save(buf, format = "PNG")
    return buf.getvalue()


def _state(*, mode = "oneshot", server = None):
    return bk._SdState(
        repo_id = "unsloth/Qwen-Image-Layered-GGUF",
        base_repo = LAYERED.base_repo,
        family = LAYERED,
        device = "cpu",
        files = FILES,
        sampling_method = LAYERED.sd_cpp_sampling_method,
        flow_shift = LAYERED.sd_cpp_flow_shift,
        mode = mode,
        server = server,
    )


def test_layered_maps_to_its_own_vae_and_the_qwen_encoder():
    fam = detect_family_for_pick(
        "unsloth/Qwen-Image-Layered-GGUF", "qwen-image-layered-Q4_K_M.gguf"
    )
    assert fam is LAYERED and family_sd_cpp_supported(fam)
    specs = SdCppDiffusionBackend(engine = _FakeEngine())._asset_specs(
        "unsloth/Qwen-Image-Layered-GGUF", "qwen-image-layered-Q4_K_M.gguf", fam
    )
    # The vendor base's own VAE file (no community repack), the one the diffusers route fetches too.
    assert specs[1] == ("Qwen/Qwen-Image-Layered", "vae/diffusion_pytorch_model.safetensors", "vae")
    assert fam.sd_cpp_vae[0] == fam.base_repo
    # sd.cpp never runs the vision tower for this model, so no projector is fetched.
    assert [k for _r, _f, k in specs] == ["diffusion_model", "vae", "qwen2vl"]
    assert text_encoder_flags_for_family(fam.name) == ("--qwen2vl",)
    # The diffusers route's ComfyUI shift, so both engines sample the same schedule.
    assert (fam.sd_cpp_sampling_method, fam.sd_cpp_flow_shift) == ("euler", fam.comfy_flow_shift)


def test_both_builders_carry_the_layer_count_only_when_set():
    params = SdCppGenParams(prompt = "p", width = 640, height = 640, ref_images = ("/r.png",))
    argv = build_sd_cpp_command("sd-cli", FILES, params, output_path = "/o.png")
    assert "--qwen-image-layers" not in argv
    layered = SdCppGenParams(
        prompt = "p", width = 640, height = 640, ref_images = ("/r.png",), qwen_image_layers = 2
    )
    argv = build_sd_cpp_command("sd-cli", FILES, layered, output_path = "/o.png")
    assert argv[argv.index("--qwen-image-layers") + 1] == "2"
    assert "qwen_image_layers" not in build_img_gen_request(prompt = "p")
    assert build_img_gen_request(prompt = "p", qwen_image_layers = 2)["qwen_image_layers"] == 2


def test_sd_cli_output_paths_follow_its_naming():
    assert sd_cli_output_paths("/t/img_0.png", 1) == ["/t/img_0.png"]
    assert sd_cli_output_paths("/t/img_0.png", 3) == [
        "/t/img_0_0.png",
        "/t/img_0_1.png",
        "/t/img_0_2.png",
    ]


def test_canvas_matches_the_diffusers_engine():
    from core.inference.diffusion import _layered_canvas
    for size in ((900, 600), (768, 768), (600, 1400), (1024, 1000)):
        assert bk._layered_canvas_size(LAYERED, size) == _layered_canvas(LAYERED, size), size


def test_workflow_is_edit_only_and_ready_without_a_projector():
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    b._state = _state()
    assert b.status()["workflows"] == ["edit"]


def test_server_body_sends_the_layer_count_and_drops_the_reconstruction():
    server = _LayerServer("sd-server")
    layers = LAYERED.layer_count
    colors = [(i * 10, 0, 0, 255) for i in range(layers + 1)]
    server.next_blobs = [_blob(c) for c in colors]
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    b._state = _state(mode = "server", server = server)
    out = b.generate(prompt = "p", steps = 20, guidance = 2.5, seed = 5, init_image = _png())
    body = server.payloads[-1]
    assert body["qwen_image_layers"] == layers
    assert (body["width"], body["height"]) == bk._layered_canvas_size(LAYERED, (900, 600))
    assert body["sample_params"]["guidance"] == {"txt_cfg": 2.5}
    assert body["sample_params"]["flow_shift"] == LAYERED.sd_cpp_flow_shift
    ref = Image.open(io.BytesIO(base64.b64decode(body["ref_images"][0].split(",", 1)[1])))
    assert ref.size == (body["width"], body["height"])
    assert [im.getpixel((0, 0)) for im in out["images"]] == colors[1:]
    assert all(im.mode == "RGBA" for im in out["images"])
    assert out["seeds"] == [5] * layers and out["workflow"] == "edit"


def test_server_refuses_a_short_layer_set():
    server = _LayerServer("sd-server")
    server.next_blobs = [_blob((1, 1, 1, 255))]
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    b._state = _state(mode = "server", server = server)
    with pytest.raises(RuntimeError, match = r"returned 1 of \d+ requested images"):
        b.generate(prompt = "p", steps = 2, init_image = _png())


def test_oneshot_collects_every_numbered_layer_file(monkeypatch):
    engine = _FakeEngine()
    layers = LAYERED.layer_count
    colors = [(0, i * 10, 0, 255) for i in range(layers + 1)]
    seen = {}

    def _write(files, params, output_path, **kw):
        seen["argv"] = build_sd_cpp_command("sd-cli", files, params, output_path = output_path)
        for path, color in zip(sd_cli_output_paths(output_path, layers + 1), colors):
            Path(path).write_bytes(_blob(color))
        return Path(output_path)

    monkeypatch.setattr(engine, "generate", _write)
    b = SdCppDiffusionBackend(engine = engine)
    b._state = _state()
    out = b.generate(prompt = "p", steps = 20, guidance = 2.5, seed = 9, init_image = _png((768, 768)))
    argv = seen["argv"]
    assert argv[argv.index("--qwen-image-layers") + 1] == str(layers)
    assert argv.count("--ref-image") == 1 and argv[argv.index("--width") + 1] == "640"
    assert [im.getpixel((0, 0)) for im in out["images"]] == colors[1:]
    assert out["seeds"] == [9] * layers


@pytest.mark.parametrize("size", [(500, 521), (770, 500), (901, 603), (768, 768)])
def test_off_grid_source_gets_the_same_canvas_as_the_diffusers_engine(size):
    # The diffusers engine snaps the source to 16 px before the pipeline picks its canvas (500x521 -> 608x672, not
    # the 640x640 the raw size gives), so the native engine must pick from the snapped size too.
    from core.inference.diffusion import _layered_canvas, _snap_to_multiple

    buf = io.BytesIO()
    Image.new("RGBA", size, (10, 20, 30, 255)).save(buf, format = "PNG")
    src = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    w, h, _blobs = bk._native_condition_images(
        LAYERED,
        src,
        None,
        None,
        None,
        None,
        full_fidelity = True,
        pad_to_output = False,
        source_sized = True,
    )
    assert (w, h) == _layered_canvas(LAYERED, _snap_to_multiple(Image.new("RGB", size), 16).size)


def test_layered_needs_a_build_that_carries_layered_support(tmp_path):
    # Layered landed upstream in master-744; an older reused build must not be picked for it.
    old = tmp_path / "sd-cli-old"
    old.write_bytes(b"stable-diffusion.cpp qwen_image_2_1 --ref-image")
    new = tmp_path / "sd-cli-new"
    new.write_bytes(b"stable-diffusion.cpp --qwen-image-layers qwen_image_layers")
    assert not bk.sd_cpp_binary_runs_family(str(old), LAYERED)
    assert bk.sd_cpp_binary_runs_family(str(new), LAYERED)
