# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Native (sd.cpp) Qwen-Image-2.1 editing: the vision projector asset, the --llm_vision and
reference-image transport on both the one-shot and the server path, the readiness gate, and the
pinned build's alpha and crop workarounds."""

from __future__ import annotations

import base64
import dataclasses
import io
import types

import pytest
from PIL import Image

from core.inference import sd_cpp_backend as bk
from core.inference.diffusion_conditioning import LocalizedEdit
from core.inference.diffusion_families import detect_family, sd_cpp_text_encoders_for
from core.inference.sd_cpp_args import (
    SdCppGenParams,
    SdCppModelFiles,
    build_img_gen_request,
    build_sd_cpp_command,
    build_sd_cpp_server_command,
)
from core.inference.sd_cpp_backend import SdCppDiffusionBackend

from .test_sd_cpp_backend import _FakeEngine, _FakeServer

FAM = detect_family("unsloth/Qwen-Image-2.1-GGUF")
FILES = SdCppModelFiles(
    diffusion_model = "/m/q.gguf",
    vae = "/m/vae.safetensors",
    llm = "/m/llm.gguf",
    llm_vision = "/m/mmproj.gguf",
)


def _png(size = (64, 32), color = (200, 10, 10, 255)) -> str:
    buf = io.BytesIO()
    Image.new("RGBA", size, color).save(buf, format = "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _decode(blob: bytes) -> Image.Image:
    return Image.open(io.BytesIO(blob))


def test_the_projector_is_a_planned_companion_asset():
    specs = SdCppDiffusionBackend(engine = _FakeEngine())._asset_specs(
        "unsloth/Qwen-Image-2.1-GGUF", "qwen-image-2.1-Q4_K_M.gguf", FAM
    )
    assert ("unsloth/Qwen3-VL-8B-Instruct-GGUF", "mmproj-F16.gguf", "llm_vision") in specs
    assert any(k == "llm_vision" for _r, _f, k in sd_cpp_text_encoders_for(FAM))


def test_both_command_builders_pass_the_projector_and_the_cli_keeps_ref_order():
    cli = build_sd_cpp_command(
        "sd-cli",
        FILES,
        SdCppGenParams(prompt = "p", width = 512, height = 512, ref_images = ("/a.png", "/b.png")),
        output_path = "/o.png",
    )
    assert cli[cli.index("--llm_vision") + 1] == "/m/mmproj.gguf"
    refs = [cli[i + 1] for i, a in enumerate(cli) if a == "--ref-image"]
    assert refs == ["/a.png", "/b.png"]
    assert "--init-img" not in cli and "--mask" not in cli
    server = build_sd_cpp_server_command("sd-server", FILES, host = "127.0.0.1", port = 1)
    assert server[server.index("--llm_vision") + 1] == "/m/mmproj.gguf"


def test_the_server_request_carries_refs_and_nothing_img2img():
    req = build_img_gen_request(prompt = "p", ref_images = ["data:image/png;base64,AA", "B"])
    assert req["ref_images"] == ["data:image/png;base64,AA", "B"]
    assert "init_image" not in req and "strength" not in req and "mask_image" not in req
    assert "ref_images" not in build_img_gen_request(prompt = "p")


def test_condition_images_flatten_alpha_over_white_and_pad_only_for_the_server():
    clear = _png(size = (64, 32), color = (0, 0, 255, 0))
    w, h, blobs = bk._native_condition_images(
        FAM, _png(size = (64, 32)), [clear], None, None, None, full_fidelity = False, pad_to_output = True
    )
    # Image 1's 2:1 aspect ratio at the 1024 area, the same size the diffusers engine picks.
    assert (w, h) == (1440, 736)
    ref = _decode(blobs[1])
    assert ref.mode == "RGB" and ref.getpixel((32, 16)) == (255, 255, 255)
    # A reference with another aspect ratio is padded to the output's instead of being cropped.
    _w, _h, padded = bk._native_condition_images(
        FAM,
        _png(size = (64, 32)),
        [_png(size = (32, 32))],
        None,
        1024,
        512,
        full_fidelity = False,
        pad_to_output = True,
    )
    assert _decode(padded[1]).size == (64, 32)
    _w, _h, plain = bk._native_condition_images(
        FAM,
        _png(size = (64, 32)),
        [_png(size = (32, 32))],
        None,
        1024,
        512,
        full_fidelity = False,
        pad_to_output = False,
    )
    assert _decode(plain[1]).size == (32, 32)


def test_native_outputs_keep_alpha_for_the_family_only():
    rgba = Image.new("RGBA", (2, 2), (1, 2, 3, 100))
    assert bk._native_output_image(FAM, rgba).mode == "RGBA"
    assert bk._native_output_image(detect_family("z-image"), rgba).mode == "RGB"


def _state(
    files = FILES,
    mode = "oneshot",
    server = None,
):
    return bk._SdState(
        repo_id = "unsloth/Qwen-Image-2.1-GGUF",
        base_repo = FAM.base_repo,
        family = FAM,
        device = "cpu",
        files = files,
        sampling_method = FAM.sd_cpp_sampling_method,
        mode = mode,
        server = server,
    )


def test_edit_is_advertised_only_with_the_projector_and_an_edit_capable_build(tmp_path):
    capable = tmp_path / "sd-new"
    capable.write_bytes(b"xx" + FAM.sd_cpp_edit_marker.encode() + b"yy qwen_image_2_1")
    older = tmp_path / "sd-old"
    older.write_bytes(b"qwen_image_2_1 only")
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    for binary, files, expect in (
        (capable, FILES, True),
        (older, FILES, False),
        (capable, SdCppModelFiles(diffusion_model = "/m/q.gguf", llm = "/m/llm.gguf"), False),
    ):
        b._state = _state(
            files,
            mode = "server",
            server = types.SimpleNamespace(binary = str(binary), is_alive = lambda: True),
        )
        status = b.status()
        assert ("edit" in status["workflows"]) is expect, binary
        assert ("reference" in status["workflows"]) is expect
        if expect:
            c = status["conditioning"]
            assert c["reference_resolutions"] == [] and c["alpha"] is False
            assert any("padded to its aspect ratio" in n for n in c["notes"])
            assert any("raise Guidance above 1" in n for n in c["notes"])


def test_oneshot_edit_stages_ordered_pngs_and_records_the_workflow(monkeypatch):
    engine = _FakeEngine()
    b = SdCppDiffusionBackend(engine = engine)
    b._state = _state()
    monkeypatch.setattr(SdCppDiffusionBackend, "_native_edit_ready", lambda self, st: True)
    seen = {}

    def _capture(files, params, output_path, **kw):
        seen["refs"] = [Image.open(p).getpixel((0, 0)) for p in params.ref_images]
        seen["size"] = (params.width, params.height)
        Image.new("RGBA", (1, 1), (1, 2, 3, 4)).save(output_path)

    monkeypatch.setattr(engine, "generate", _capture)
    out = b.generate(
        prompt = "p",
        steps = 2,
        workflow = "edit",
        width = None,
        height = None,
        init_image = _png(color = (10, 0, 0, 255)),
        reference_images = [_png(color = (20, 0, 0, 255)), _png(color = (30, 0, 0, 255))],
        localized_edit = None,
    )
    assert [c[:3] for c in seen["refs"]] == [(10, 0, 0), (20, 0, 0), (30, 0, 0)]
    assert seen["size"][0] / seen["size"][1] == pytest.approx(2.0, rel = 0.05)
    assert out["workflow"] == "edit" and out["images"][0].mode == "RGBA"


def test_server_edit_sends_the_mask_as_image_2(monkeypatch):
    server = _FakeServer("sd-server")
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    b._state = _state(mode = "server", server = server)
    monkeypatch.setattr(SdCppDiffusionBackend, "_native_edit_ready", lambda self, st: True)
    mask = io.BytesIO()
    Image.new("L", (64, 32), 255).save(mask, format = "PNG")
    b.generate(
        prompt = "p",
        steps = 2,
        workflow = "edit",
        width = 512,
        height = 256,
        init_image = _png(),
        localized_edit = LocalizedEdit("mask", base64.b64encode(mask.getvalue()).decode()),
    )
    refs = server.payloads[-1]["ref_images"]
    assert len(refs) == 2
    second = _decode(base64.b64decode(refs[1].split(",", 1)[1]))
    assert second.getpixel((0, 0)) == (255, 255, 255)


def test_native_refuses_what_it_cannot_honour(monkeypatch):
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    b._state = _state()
    with pytest.raises(ValueError, match = "Reference detail is not adjustable"):
        b.generate(prompt = "p", workflow = "edit", init_image = _png(), reference_resolution = 1024)
    with pytest.raises(ValueError, match = "not yet supported"):
        b.generate(prompt = "p", init_image = _png())  # an omitted workflow keeps its refusal
    # Same refusal as the diffusers engine: a localized edit belongs to the edit workflow only.
    with pytest.raises(ValueError, match = "localized_edit needs the edit workflow"):
        b.generate(
            prompt = "p",
            workflow = "reference",
            init_image = _png(),
            width = 512,
            height = 256,
            localized_edit = LocalizedEdit("paint", _png()),
        )
    # Without an edit-capable build and projector, the workflow is refused rather than sent.
    monkeypatch.setattr(SdCppDiffusionBackend, "_native_edit_ready", lambda self, st: False)
    with pytest.raises(ValueError, match = "Image editing is not available"):
        b.generate(prompt = "p", workflow = "edit", init_image = _png(), width = 512, height = 512)


@pytest.mark.parametrize(
    "family, width, height, match",
    [
        ("z-image", 2560, 1024, "at most 2048px"),
        ("qwen-image-2.1", 1040, 1024, "multiples of 32"),
        ("qwen-image-2.1", 2752, 2752, "pixels"),
    ],
)
def test_native_text_to_image_keeps_the_family_size_bounds(family, width, height, match):
    # The request schema admits 2752 for Qwen-Image-2.1's 2K presets; every other bound is the family's.
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    b._state = dataclasses.replace(_state(), family = detect_family(family))
    with pytest.raises(ValueError, match = match):
        b.generate(prompt = "p", steps = 2, width = width, height = height)


def test_a_padded_mask_adds_no_region_and_stays_aligned_with_the_source():
    """Padding the separate mask with white marked the added border as part of the region to
    edit. It is padded black, by the same amount as the source, so the region stays where it was
    drawn."""
    mask = Image.new("L", (64, 32), 0)
    for x in range(8):
        for y in range(8):
            mask.putpixel((x, y), 255)
    buf = io.BytesIO()
    mask.save(buf, format = "PNG")
    _w, _h, blobs = bk._native_condition_images(
        FAM,
        _png(size = (64, 32), color = (9, 9, 9, 255)),
        None,
        LocalizedEdit("mask", base64.b64encode(buf.getvalue()).decode()),
        512,
        512,
        full_fidelity = False,
        pad_to_output = True,
    )
    source, padded = _decode(blobs[0]), _decode(blobs[1]).convert("L")
    assert source.size == padded.size == (64, 64)
    # The border the padding added is not part of the region...
    assert padded.getpixel((32, 2)) == 0 and padded.getpixel((32, 60)) == 0
    # ...the drawn region moved with the source, and the source border is white as before.
    assert padded.getpixel((2, 18)) == 255 and padded.getpixel((2, 26)) == 0
    assert source.getpixel((32, 2))[:3] == (255, 255, 255)
    assert source.getpixel((2, 18))[:3] == (9, 9, 9)


def test_a_full_fidelity_build_gets_the_images_as_decoded(tmp_path):
    """A build carrying the upstream reference fixes reads alpha and keeps each image's shape, so
    neither workaround applies and status reports alpha."""
    clear = _png(size = (32, 32), color = (0, 0, 255, 0))
    _w, _h, blobs = bk._native_condition_images(
        FAM, _png(size = (64, 32)), [clear], None, 1024, 512, full_fidelity = True, pad_to_output = True
    )
    ref = _decode(blobs[1])
    assert ref.mode == "RGBA" and ref.size == (32, 32) and ref.getpixel((4, 4)) == (0, 0, 255, 0)

    binary = tmp_path / "sd-server"
    binary.write_bytes(
        b"qwen_image_2_1 "
        + FAM.sd_cpp_edit_marker.encode()
        + b" "
        + bk._REFERENCE_FIDELITY_MARKER.encode()
    )
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    b._state = _state(
        mode = "server", server = types.SimpleNamespace(binary = str(binary), is_alive = lambda: True)
    )
    c = b.status()["conditioning"]
    assert c["alpha"] is True
    assert c["notes"] == [
        "For transparent output on the native engine, raise Guidance above 1 (upstream uses 6)."
    ]
