# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Native edit families (Qwen-Image-Edit, FLUX.1-Kontext): asset map, guidance and --ref-image wiring."""

from __future__ import annotations

import base64
import io
import types

import pytest
from PIL import Image

from core.inference import sd_cpp_backend as bk
from core.inference.diffusion_families import (
    detect_family,
    detect_family_for_pick,
    family_sd_cpp_supported,
    sd_cpp_companion_only_repo_ids,
)
from core.inference.sd_cpp_args import (
    SdCppModelFiles,
    build_sd_cpp_command,
    text_encoder_flags_for_family,
)
from core.inference.sd_cpp_backend import SdCppDiffusionBackend

from .test_sd_cpp_backend import _FakeEngine, _FakeServer

QWEN_EDIT = detect_family("unsloth/Qwen-Image-Edit-2511-GGUF")
KONTEXT = detect_family("unsloth/FLUX.1-Kontext-dev-GGUF")

QWEN_PICKS = (
    ("unsloth/Qwen-Image-Edit-GGUF", "qwen-image-edit-Q4_K_M.gguf"),
    ("unsloth/Qwen-Image-Edit-2509-GGUF", "qwen-image-edit-2509-Q4_K_M.gguf"),
    ("unsloth/Qwen-Image-Edit-2511-GGUF", "qwen-image-edit-2511-Q4_K_M.gguf"),
)

QWEN_FILES = SdCppModelFiles(
    diffusion_model = "/m/qwen-image-edit-2511-Q4_K_M.gguf",
    vae = "/m/qwen_image_vae.safetensors",
    qwen2vl = "/m/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
    llm_vision = "/m/mmproj-F16.gguf",
)
KONTEXT_FILES = SdCppModelFiles(
    diffusion_model = "/m/flux1-kontext-dev-Q4_K_M.gguf",
    vae = "/m/ae.safetensors",
    clip_l = "/m/clip_l.safetensors",
    t5xxl = "/m/t5xxl_fp16.safetensors",
)


def _png(size = (770, 500), color = (200, 10, 10)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format = "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _state(
    fam,
    files,
    *,
    mode = "oneshot",
    server = None,
):
    return bk._SdState(
        repo_id = "unsloth/x-GGUF",
        base_repo = fam.base_repo,
        family = fam,
        device = "cpu",
        files = files,
        sampling_method = fam.sd_cpp_sampling_method,
        flow_shift = fam.sd_cpp_flow_shift,
        mode = mode,
        server = server,
    )


def _server_state(fam, files):
    server = _FakeServer("sd-server")
    return _state(fam, files, mode = "server", server = server), server


@pytest.mark.parametrize("repo, filename", QWEN_PICKS)
def test_every_qwen_edit_checkpoint_maps_to_vae_encoder_and_projector(repo, filename):
    fam = detect_family_for_pick(repo, filename)
    assert fam is QWEN_EDIT and family_sd_cpp_supported(fam)
    specs = SdCppDiffusionBackend(engine = _FakeEngine())._asset_specs(repo, filename, fam)
    assert specs == [
        (repo, filename, "diffusion_model"),
        ("unsloth/Qwen-Image-ComfyUI", "split_files/vae/qwen_image_vae.safetensors", "vae"),
        (
            "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
            "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
            "qwen2vl",
        ),
        ("unsloth/Qwen2.5-VL-7B-Instruct-GGUF", "mmproj-F16.gguf", "llm_vision"),
    ]
    assert (fam.sd_cpp_sampling_method, fam.sd_cpp_flow_shift) == ("euler", 3.0)
    assert text_encoder_flags_for_family(fam.name) == ("--qwen2vl",)


def test_kontext_maps_to_the_flux1_assets():
    fam = detect_family_for_pick("unsloth/FLUX.1-Kontext-dev-GGUF", "flux1-kontext-dev-Q4_K_M.gguf")
    assert fam is KONTEXT and family_sd_cpp_supported(fam)
    flux1 = detect_family("unsloth/FLUX.1-dev-GGUF")
    assert fam.sd_cpp_vae == flux1.sd_cpp_vae
    assert fam.sd_cpp_text_encoders == flux1.sd_cpp_text_encoders
    specs = SdCppDiffusionBackend(engine = _FakeEngine())._asset_specs(
        "unsloth/FLUX.1-Kontext-dev-GGUF", "flux1-kontext-dev-Q4_K_M.gguf", fam
    )
    assert [k for _r, _f, k in specs] == ["diffusion_model", "vae", "clip_l", "t5xxl"]
    assert text_encoder_flags_for_family(fam.name) == ("--clip_l", "--t5xxl")


def test_the_new_companions_stay_fetch_only_and_bases_stay_loadable():
    companions = sd_cpp_companion_only_repo_ids()
    assert "unsloth/qwen-image-comfyui" in companions
    assert "unsloth/qwen2.5-vl-7b-instruct-gguf" in companions
    assert "black-forest-labs/flux.1-schnell" not in companions


def test_inpaint_and_other_layered_variants_are_still_refused():
    # Qwen-Image-Layered has a family of its own; a layered or inpaint variant of any other family does not.
    assert detect_family("unsloth/Qwen-Image-Layered-GGUF").name == "qwen-image-layered"
    assert detect_family("someone/FLUX.1-dev-layered-GGUF") is None
    assert detect_family("someone/FLUX.1-dev-inpaint-GGUF") is None


def test_kontext_runs_one_pass_with_embedded_guidance():
    assert bk._map_guidance(KONTEXT, 2.5) == (1.0, 2.5)
    assert bk._map_guidance(KONTEXT, None) == (1.0, None)


@pytest.mark.parametrize("guidance, cfg", [(4.0, 4.0), (2.5, 2.5), (1.0, 1.0), (0.0, 1.0)])
def test_qwen_edit_uses_real_cfg_like_the_diffusers_true_cfg_scale(guidance, cfg):
    assert bk._map_guidance(QWEN_EDIT, guidance) == (cfg, None)


def test_edit_only_families_advertise_edit_and_never_txt2img():
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    for fam, files in ((QWEN_EDIT, QWEN_FILES), (KONTEXT, KONTEXT_FILES)):
        b._state = _state(fam, files)
        status = b.status()
        assert status["workflows"] == ["edit"], fam.name
        assert status["conditioning"]["unified_edit"] is False
        assert status["conditioning"]["localized_edit_modes"] == []
        assert status["conditioning"]["notes"] == []


def test_qwen_edit_without_its_projector_is_not_edit_ready():
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    no_vision = SdCppModelFiles(
        diffusion_model = "/m/q.gguf", vae = "/m/vae.safetensors", qwen2vl = "/m/llm.gguf"
    )
    b._state = _state(QWEN_EDIT, no_vision)
    assert b.status()["workflows"] == []
    with pytest.raises(ValueError, match = "Image editing is not available"):
        b.generate(prompt = "p", steps = 2, init_image = _png())


@pytest.mark.parametrize("workflow", [None, "edit"])
def test_oneshot_kontext_sends_the_source_as_a_ref_image_at_its_own_size(monkeypatch, workflow):
    engine = _FakeEngine()
    b = SdCppDiffusionBackend(engine = engine)
    b._state = _state(KONTEXT, KONTEXT_FILES)
    seen = {}
    real_generate = engine.generate

    def _capture(files, params, output_path, **kw):
        seen["ref"] = Image.open(params.ref_images[0]).copy()
        seen["argv"] = build_sd_cpp_command(
            "sd-cli", files, params, output_path = output_path, extra_args = kw.get("extra_args")
        )
        return real_generate(files, params, output_path = output_path, **kw)

    monkeypatch.setattr(engine, "generate", _capture)
    out = b.generate(
        prompt = "make it night",
        steps = 20,
        guidance = 2.5,
        seed = 7,
        width = 1024,
        height = 1024,
        init_image = _png(),
        workflow = workflow,
    )
    argv = seen["argv"]
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "--ref-image"] == [
        engine.calls[0][1].ref_images[0]
    ]
    # Source-sized as on the diffusers engine: 770x500 snaps to 768x496, the requested 1024 is ignored.
    assert argv[argv.index("--width") + 1] == "768" and argv[argv.index("--height") + 1] == "496"
    assert seen["ref"].size == (768, 496)
    assert (
        argv[argv.index("--cfg-scale") + 1] == "1" and argv[argv.index("--guidance") + 1] == "2.5"
    )
    for flag, path in (
        ("--vae", "/m/ae.safetensors"),
        ("--clip_l", "/m/clip_l.safetensors"),
        ("--t5xxl", "/m/t5xxl_fp16.safetensors"),
    ):
        assert argv[argv.index(flag) + 1] == path
    assert "--init-img" not in argv and "--strength" not in argv and "--llm_vision" not in argv
    assert out["workflow"] == "edit"


def test_oneshot_qwen_edit_sends_projector_cfg_and_flow_shift(monkeypatch):
    engine = _FakeEngine()
    b = SdCppDiffusionBackend(engine = engine)
    b._state = _state(QWEN_EDIT, QWEN_FILES)
    b.generate(prompt = "p", steps = 20, guidance = 4.0, seed = 1, init_image = _png((768, 768)))
    files, params, out_path, kw = engine.calls[0]
    argv = build_sd_cpp_command(
        "sd-cli", files, params, output_path = out_path, extra_args = kw.get("extra_args")
    )
    assert argv[argv.index("--qwen2vl") + 1] == QWEN_FILES.qwen2vl
    assert argv[argv.index("--llm_vision") + 1] == QWEN_FILES.llm_vision
    assert argv.count("--ref-image") == 1
    assert argv[argv.index("--cfg-scale") + 1] == "4" and "--guidance" not in argv
    assert argv[argv.index("--sampling-method") + 1] == "euler"
    assert argv[argv.index("--flow-shift") + 1] == "3.0"
    assert (params.width, params.height) == (768, 768)


@pytest.mark.parametrize(
    "fam, files, guidance, expect_guidance, expect_sampling",
    [
        (KONTEXT, KONTEXT_FILES, 2.5, {"txt_cfg": 1.0, "distilled_guidance": 2.5}, {}),
        (
            QWEN_EDIT,
            QWEN_FILES,
            4.0,
            {"txt_cfg": 4.0},
            {"sample_method": "euler", "flow_shift": 3.0},
        ),
    ],
)
def test_server_body_carries_the_source_as_the_one_ref_image(
    fam, files, guidance, expect_guidance, expect_sampling
):
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    b._state, server = _server_state(fam, files)
    b.generate(prompt = "p", steps = 20, guidance = guidance, seed = 3, init_image = _png(), workflow = "edit")
    body = server.payloads[-1]
    assert (body["width"], body["height"]) == (768, 496)
    assert body["sample_params"]["guidance"] == expect_guidance
    for key, value in expect_sampling.items():
        assert body["sample_params"][key] == value
    assert body["seed"] == 3 and body["batch_count"] == 1
    assert len(body["ref_images"]) == 1
    head, b64 = body["ref_images"][0].split(",", 1)
    assert head == "data:image/png;base64"
    ref = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert ref.size == (768, 496) and ref.mode == "RGB"
    assert "init_image" not in body and "mask_image" not in body and "strength" not in body


@pytest.mark.parametrize("fam, files", [(QWEN_EDIT, QWEN_FILES), (KONTEXT, KONTEXT_FILES)])
def test_edit_only_families_refuse_what_the_diffusers_engine_refuses(fam, files):
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    b._state = _state(fam, files)
    with pytest.raises(ValueError, match = "is an image-editing model: provide an input image"):
        b.generate(prompt = "p", steps = 2, width = 512, height = 512)
    with pytest.raises(ValueError, match = "reference workflow is not supported"):
        b.generate(prompt = "p", steps = 2, init_image = _png(), workflow = "reference")
    with pytest.raises(ValueError, match = "Reference images are not supported"):
        b.generate(prompt = "p", steps = 2, init_image = _png(), reference_images = [_png()])
    with pytest.raises(ValueError, match = "mask_image is not supported"):
        b.generate(prompt = "p", steps = 2, init_image = _png(), mask_image = _png())


def test_a_text_to_image_family_keeps_refusing_an_input_image():
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    flux1 = detect_family("unsloth/FLUX.1-dev-GGUF")
    b._state = _state(flux1, KONTEXT_FILES)
    with pytest.raises(ValueError, match = "not yet supported on the native"):
        b.generate(prompt = "p", steps = 2, init_image = _png())
    assert b.status()["workflows"] == ["txt2img"]


def test_unified_edit_readiness_is_unchanged(tmp_path):
    q21 = detect_family("unsloth/Qwen-Image-2.1-GGUF")
    files = SdCppModelFiles(diffusion_model = "/m/q.gguf", llm = "/m/llm.gguf", llm_vision = "/m/v.gguf")
    binary = tmp_path / "sd-server"
    binary.write_bytes(b"no marker here")
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    b._state = _state(
        q21,
        files,
        mode = "server",
        server = types.SimpleNamespace(binary = str(binary), is_alive = lambda: True),
    )
    assert b.status()["workflows"] == ["txt2img"]
    binary.write_bytes(q21.sd_cpp_edit_marker.encode())
    assert b.status()["workflows"] == ["txt2img", "reference", "edit"]


def test_source_size_snaps_like_the_diffusers_engine_and_fits_an_oversized_source():
    from core.inference.diffusion import _snap_to_multiple

    for size in ((770, 500), (776, 520), (1000, 1000), (513, 1999)):
        w, h, _blobs = bk._native_condition_images(
            KONTEXT,
            _png(size),
            None,
            None,
            1024,
            1024,
            full_fidelity = False,
            pad_to_output = True,
            source_sized = True,
        )
        assert (w, h) == _snap_to_multiple(Image.new("RGB", size)).size, size
    # 4000x3000 exceeds the 2048 side / 2048*2048 pixel bounds: scaled to fit, aspect kept, never refused.
    w, h, blobs = bk._native_condition_images(
        QWEN_EDIT,
        _png((4000, 3000)),
        None,
        None,
        None,
        None,
        full_fidelity = False,
        pad_to_output = False,
        source_sized = True,
    )
    assert (w, h) == (2048, 1536)
    assert Image.open(io.BytesIO(blobs[0])).size == (2048, 1536)


@pytest.mark.parametrize(
    "size,expected", [((64, 64), (256, 256)), ((200, 100), (512, 256)), ((100, 300), (256, 768))]
)
def test_a_source_below_the_minimum_side_is_scaled_up_not_refused(size, expected):
    # An edit-only call has no width/height, so a small source is scaled up to the minimum side.
    w, h, blobs = bk._native_condition_images(
        QWEN_EDIT,
        _png(size),
        None,
        None,
        None,
        None,
        full_fidelity = False,
        pad_to_output = False,
        source_sized = True,
    )
    assert (w, h) == expected
    assert Image.open(io.BytesIO(blobs[0])).size == expected


def test_an_uncached_projector_fails_an_offline_edit_only_load(monkeypatch, tmp_path):
    # An edit-only family has no workflow without its projector, so a cache-only load must not succeed without it.
    from huggingface_hub.errors import LocalEntryNotFoundError

    import utils.hf_xet_fallback as xet

    cached = tmp_path / "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
    cached.write_bytes(b"")

    def _download(repo_id, filename, token, **kwargs):
        if filename == "mmproj-F16.gguf":
            raise LocalEntryNotFoundError("Cannot find the requested files in the disk cache")
        return str(cached)

    monkeypatch.setattr(xet, "hf_hub_download_with_xet_fallback", _download)
    repo = "unsloth/Qwen2.5-VL-7B-Instruct-GGUF"
    assets = [(repo, cached.name, "qwen2vl"), (repo, "mmproj-F16.gguf", "llm_vision")]
    with pytest.raises(RuntimeError, match = "mmproj-F16.gguf"):
        SdCppDiffusionBackend(engine = None)._fetch_assets(
            assets, None, local_files_only = True, vision_optional = False
        )
    assert SdCppDiffusionBackend(engine = None)._fetch_assets(
        assets, None, local_files_only = True
    ) == {"qwen2vl": str(cached)}
