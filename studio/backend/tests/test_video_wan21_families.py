# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Wan2.1 text-to-video families: detection by repo name, alias, file name and checkpoint header,
ComfyUI defaults, the refusals for Wan2.1 checkpoints no family serves, and a table proving every
repo the other video families already claimed keeps its family.

The header fixture holds real tensor names / shapes (first block of each stack only) read from the
Comfy-Org repackage, city96, samuelchristlie and QuantStack files on the Hub."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from core.inference import diffusion_content as dc
from core.inference import video_families as vf

_HEADERS = json.loads(
    (Path(__file__).parent / "fixtures" / "wan21_checkpoint_headers.json").read_text(
        encoding = "utf-8"
    )
)

_I2V = "image-to-video"
_VARIANT = "conditioning variant"
_FASTWAN = "FastWan"

# (identifier, family on this branch). Rows above the marker are what main already detected and must not move; the
# Wan2.1 rows below it all detected None on main.
_TABLE = [
    ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "wan2.2-ti2v-5b"),
    ("Wan-AI/Wan2.2-TI2V-5B", "wan2.2-ti2v-5b"),
    ("unsloth/Wan2.2-TI2V-5B-GGUF", "wan2.2-ti2v-5b"),
    ("QuantStack/Wan2.2-TI2V-5B-GGUF/Wan2.2-TI2V-5B-Q4_K_M.gguf", "wan2.2-ti2v-5b"),
    ("wan2.2_ti2v_5B_fp16.safetensors", "wan2.2-ti2v-5b"),
    ("Wan-AI/Wan2.2-T2V-A14B-Diffusers", "wan2.2-t2v-a14b"),
    ("Wan-AI/Wan2.2-T2V-A14B", "wan2.2-t2v-a14b"),
    (
        "QuantStack/Wan2.2-T2V-A14B-GGUF/HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf",
        "wan2.2-t2v-a14b",
    ),
    ("wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", "wan2.2-t2v-a14b"),
    ("Wan2_2-T2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors", "wan2.2-t2v-a14b"),
    ("Wan-AI/Wan2.2-I2V-A14B-Diffusers", None),
    ("FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers", None),
    ("Lightricks/LTX-2", "ltx-2"),
    ("Lightricks/LTX-2.3", "ltx-2"),
    ("unsloth/LTX-2.3-GGUF", "ltx-2"),
    ("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v", "hunyuanvideo-1.5"),
    ("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v", "hunyuanvideo-1.5-720p"),
    ("hunyuanvideo1.5_720p_t2v_fp16.safetensors", "hunyuanvideo-1.5-720p"),
    ("tencent/HunyuanVideo", None),
    ("MiniMaxAI/MiniMax-H3", "minimax-h3"),
    ("unsloth/MiniMax-H3-GGUF", "minimax-h3"),
    ("minimax_h3_fl2va_int8_convrot.safetensors", "minimax-h3"),
    # ---- Wan2.1 (None on main for every row) ----
    ("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "wan2.1-t2v-1.3b"),
    ("Wan-AI/Wan2.1-T2V-1.3B", "wan2.1-t2v-1.3b"),
    ("Wan-AI/Wan2.1-T2V-14B-Diffusers", "wan2.1-t2v-14b"),
    ("Wan-AI/Wan2.1-T2V-14B", "wan2.1-t2v-14b"),
    (
        "Comfy-Org/Wan_2.1_ComfyUI_repackaged/split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors",
        "wan2.1-t2v-1.3b",
    ),
    (
        "Comfy-Org/Wan_2.1_ComfyUI_repackaged/split_files/diffusion_models/wan2.1_t2v_14B_fp8_scaled.safetensors",
        "wan2.1-t2v-14b",
    ),
    ("city96/Wan2.1-T2V-14B-gguf", "wan2.1-t2v-14b"),
    ("city96/Wan2.1-T2V-14B-gguf/wan2.1-t2v-14b-Q4_K_M.gguf", "wan2.1-t2v-14b"),
    ("samuelchristlie/Wan2.1-T2V-1.3B-GGUF/Wan2.1-T2V-1.3B-Q8_0.gguf", "wan2.1-t2v-1.3b"),
    (
        "QuantStack/Wan2.1_T2V_14B_FusionX-GGUF/Wan2.1_T2V_14B_FusionX-Q4_K_M.gguf",
        "wan2.1-t2v-14b",
    ),
    ("Kijai/WanVideo_comfy/Wan2_1-T2V-1_3B_bf16.safetensors", "wan2.1-t2v-1.3b"),
    ("Kijai/WanVideo_comfy/Wan2_1-T2V-14B_fp8_e4m3fn.safetensors", "wan2.1-t2v-14b"),
    ("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", None),
    ("Wan-AI/Wan2.1-I2V-14B-720P-Diffusers", None),
    ("city96/Wan2.1-I2V-14B-480P-gguf/wan2.1-i2v-14b-480p-Q4_K_M.gguf", None),
    ("Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers", None),
    ("Wan-AI/Wan2.1-VACE-14B-diffusers", None),
    ("QuantStack/Wan2.1_T2V_14B_FusionX_VACE-GGUF", None),
    ("lym00/Wan2.1_T2V_1.3B_SelfForcing_VACE-GGUF", None),
    ("TheDenk/wan2.1-t2v-1.3b-controlnet-canny-v1", None),
    ("FastVideo/FastWan2.1-T2V-1.3B-Diffusers", None),
    ("FastVideo/FastWan2.1-T2V-14B-Diffusers", None),
]


@pytest.mark.parametrize("identifier, expected", _TABLE)
def test_detection_table(identifier, expected):
    fam = vf.detect_video_family(identifier)
    assert (fam.name if fam else None) == expected


@pytest.mark.parametrize(
    "identifier, reason",
    [
        ("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", _I2V),
        ("Wan-AI/Wan2.1-I2V-14B-720P", _I2V),
        ("wan2.1_i2v_480p_14B_fp8_scaled.safetensors", _I2V),
        ("city96/Wan2.1-I2V-14B-720P-gguf", _I2V),
        ("Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers", _I2V),
        ("Wan-AI/Wan2.1-VACE-1.3B", _VARIANT),
        ("wan2.1_fun_inp_1.3B_bf16.safetensors", _VARIANT),
        ("QuantStack/Wan2.1_T2V_14B_FusionX_VACE-GGUF", _VARIANT),
        ("FastVideo/FastWan2.1-T2V-1.3B-Diffusers", _FASTWAN),
        ("FastVideo/FastWan-QAD-1.3B", _FASTWAN),
    ],
)
def test_unsupported_wan21_checkpoints_name_their_reason(identifier, reason):
    message = vf.unsupported_video_checkpoint_reason(identifier)
    assert message is not None and reason in message
    assert vf.detect_video_family(identifier) is None


@pytest.mark.parametrize("identifier, expected", [row for row in _TABLE if row[1] is not None])
def test_supported_names_carry_no_refusal(identifier, expected):
    assert vf.unsupported_video_checkpoint_reason(identifier) is None


def test_comfyui_template_defaults():
    for name, presets, steps, shift in (
        ("wan2.1-t2v-1.3b", ((832, 480), (480, 832)), 30, 8.0),
        ("wan2.1-t2v-14b", ((832, 480), (1280, 720), (480, 832), (720, 1280)), 20, 5.0),
    ):
        fam = vf.detect_video_family("", override = name)
        assert fam.pipeline_class == "WanPipeline"
        assert fam.transformer_class == "WanTransformer3DModel"
        assert not fam.is_moe and fam.transformer2_class is None
        assert (fam.default_steps, fam.default_guidance) == (steps, 6.0)
        assert fam.comfy_flow_shift == shift
        assert (fam.default_num_frames, fam.default_fps, fam.frame_step) == (33, 16, 4)
        assert fam.resolution_presets == presets and fam.resolution_multiple == 16
        assert fam.vae_force_fp32 and not fam.cudnn_benchmark
        vf.validate_video_request_shape(fam, 832, 480, 33)
        with pytest.raises(vf.VideoShapeError):
            vf.validate_video_request_shape(fam, 832, 480, 32)
    assert vf.detect_video_family("", override = "wan2.1-t2v-14b").base_repo == (
        "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    )
    assert vf.detect_video_family("", override = "wan2_1_t2v_1_3b").name == "wan2.1-t2v-1.3b"


@pytest.mark.parametrize(
    "ids, expected",
    [
        (("Wan-AI/Wan2.1-T2V-1.3B-Diffusers",), (30, 6.0)),
        (("wan2.1-t2v-14b-Q4_K_M.gguf", "city96/Wan2.1-T2V-14B-gguf"), (20, 6.0)),
        (("wan2.1_t2v_14B_fp8_scaled.safetensors",), (20, 6.0)),
        ((None, "Wan-AI/Wan2.1-T2V-14B-Diffusers"), (20, 6.0)),
        (("Wan2_1-T2V-1_3B_bf16.safetensors",), (30, 6.0)),
        ((None, "Wan-AI/Wan2.2-TI2V-5B-Diffusers"), (20, 5.0)),
        ((None, "Wan-AI/Wan2.2-T2V-A14B-Diffusers"), (20, 3.5)),
        (("wan2.2-ti2v-5b-Q4_K_M.gguf",), (20, 5.0)),
        (("Lightricks/LTX-2",), (40, 4.0)),
    ],
)
def test_generation_defaults(ids, expected):
    assert vf.default_video_generation_params(*ids) == expected


def test_registry_lists_the_new_families_after_the_existing_ones():
    names = vf.supported_video_family_names()
    assert names[-2:] == ("hunyuanvideo-1.5", "hunyuanvideo-1.5-720p")
    assert "wan2.1-t2v-1.3b" in names and "wan2.1-t2v-14b" in names
    for fam in ("wan2.1-t2v-1.3b", "wan2.1-t2v-14b"):
        assert dc.family_page(fam) == dc.PAGE_VIDEO


def _row(suffix: str) -> dict:
    return next(v for k, v in _HEADERS.items() if k.endswith(suffix))


@pytest.mark.parametrize(
    "suffix, role, family",
    [
        ("wan2.1_t2v_1.3B_bf16.safetensors", dc.ROLE_DIT, "wan2.1-t2v-1.3b"),
        ("Wan2.1-T2V-1.3B-Q8_0.gguf", dc.ROLE_DIT, "wan2.1-t2v-1.3b"),
        # The 14B stack has an A14B expert's keys exactly: the header alone keeps main's answer.
        ("wan2.1_t2v_14B_fp8_scaled.safetensors", dc.ROLE_DIT, "wan2.2-t2v-a14b"),
        ("wan2.1-t2v-14b-Q4_K_M.gguf", dc.ROLE_DIT, "wan2.2-t2v-a14b"),
        ("Wan2.1_T2V_14B_FusionX-Q4_K_M.gguf", dc.ROLE_DIT, "wan2.2-t2v-a14b"),
        ("wan2.1_i2v_480p_14B_fp8_scaled.safetensors", dc.ROLE_DIT, None),
        ("wan2.1-i2v-14b-480p-Q4_K_M.gguf", dc.ROLE_DIT, None),
        ("wan2.1_vace_1.3B_fp16.safetensors", dc.ROLE_DIT, None),
        ("wan2.1_fun_inp_1.3B_bf16.safetensors", dc.ROLE_DIT, None),
        ("wan_2.1_vae.safetensors", dc.ROLE_VAE, None),
    ],
)
def test_header_classification(suffix, role, family):
    row = _row(suffix)
    info = dc.classify_tensors(row["shapes"], row["meta"])
    assert (info.role, info.family) == (role, family)


def _write_safetensors(path: Path, shapes: dict) -> Path:
    header = {k: {"dtype": "BF16", "shape": v, "data_offsets": [0, 0]} for k, v in shapes.items()}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    return path


def _write_gguf(path: Path, shapes: dict, arch: str) -> Path:
    def s(text):
        b = text.encode()
        return struct.pack("<Q", len(b)) + b

    out = struct.pack("<IIQQ", 0x46554747, 3, len(shapes), 1)
    out += s("general.architecture") + struct.pack("<I", 8) + s(arch)
    for name, shape in shapes.items():
        dims = list(reversed(shape))
        out += s(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        out += struct.pack("<IQ", 0, 0)
    path.write_bytes(out)
    return path


def test_name_picks_wan21_14b_over_the_a14b_header(tmp_path):
    shapes = _row("wan2.1_t2v_14B_fp8_scaled.safetensors")["shapes"]
    _write_safetensors(tmp_path / "wan2.1_t2v_14B_fp8_scaled.safetensors", shapes)
    _write_safetensors(tmp_path / "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors", shapes)
    _write_safetensors(tmp_path / "renamed.safetensors", shapes)
    resolve = lambda f: dc.resolve_family_with_content(  # noqa: E731
        (vf.detect_video_family(f).name if vf.detect_video_family(f) else None),
        str(tmp_path / f),
        dc.PAGE_VIDEO,
    )[0]
    assert resolve("wan2.1_t2v_14B_fp8_scaled.safetensors") == "wan2.1-t2v-14b"
    assert resolve("wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors") == "wan2.2-t2v-a14b"
    assert resolve("renamed.safetensors") == "wan2.2-t2v-a14b"


def test_renamed_wan21_1_3b_gguf_resolves_by_header(tmp_path):
    from core.inference.video import _detect_load_family

    shapes = _row("Wan2.1-T2V-1.3B-Q8_0.gguf")["shapes"]
    _write_gguf(tmp_path / "model-Q8_0.gguf", shapes, "wan")
    fam = _detect_load_family(str(tmp_path), "model-Q8_0.gguf", None)
    assert fam is not None and fam.name == "wan2.1-t2v-1.3b"


def test_comfy_folder_hides_refused_names_with_a_t2v_header(tmp_path):
    # FastWan 1.3B and a renamed variant carry the plain 1.3B T2V layout; the loader refuses them by name, so the
    # listing must not offer them.
    from hub.utils import comfy_models

    folder = tmp_path / "diffusion_models"
    folder.mkdir()
    shapes = _row("wan2.1_t2v_1.3B_bf16.safetensors")["shapes"]
    for name in (
        "wan2.1_t2v_1.3B_bf16.safetensors",
        "FastWan2.1-T2V-1.3B_bf16.safetensors",
        "wan2.1_fun_1.3B_bf16.safetensors",
    ):
        _write_safetensors(folder / name, shapes)
    names = {p.name for p in comfy_models.loose_diffusion_checkpoints(folder)}
    assert names == {"wan2.1_t2v_1.3B_bf16.safetensors"}


def test_comfy_folder_lists_wan21_t2v_files(tmp_path):
    from hub.utils import comfy_models

    folder = tmp_path / "diffusion_models"
    folder.mkdir()
    for suffix in (
        "wan2.1_t2v_1.3B_bf16.safetensors",
        "wan2.1_t2v_14B_fp8_scaled.safetensors",
        "wan2.1_i2v_480p_14B_fp8_scaled.safetensors",
        "wan2.1_vace_1.3B_fp16.safetensors",
    ):
        _write_safetensors(folder / suffix, _row(suffix)["shapes"])
    # An unpaired A14B expert stays hidden, as on main.
    _write_safetensors(
        folder / "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
        _row("wan2.1_t2v_14B_fp8_scaled.safetensors")["shapes"],
    )
    names = {p.name for p in comfy_models.loose_diffusion_checkpoints(folder)}
    assert names == {"wan2.1_t2v_1.3B_bf16.safetensors", "wan2.1_t2v_14B_fp8_scaled.safetensors"}


@pytest.mark.parametrize(
    "hints, video",
    [
        (("city96/Wan2.1-T2V-14B-gguf", "wan2.1-t2v-14b-Q4_K_M.gguf"), True),
        (("samuelchristlie/Wan2.1-T2V-1.3B-GGUF", "Wan2.1-T2V-1.3B-Q8_0.gguf"), True),
        (("city96/Wan2.1-I2V-14B-480P-gguf", "wan2.1-i2v-14b-480p-Q4_K_M.gguf"), False),
        (("QuantStack/Wan2.1_T2V_14B_FusionX_VACE-GGUF", "x-Q4_K_M.gguf"), False),
        (("unsloth/Wan2.2-TI2V-5B-GGUF", "Wan2.2-TI2V-5B-Q4_K_M.gguf"), True),
        (("QuantStack/Wan2.2-T2V-A14B-GGUF", "Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf"), True),
    ],
)
def test_wan_gguf_listing_task(hints, video):
    from hub.services.models.catalog_classification import _VIDEO_GEN_TASK, _arch_to_task

    assert (_arch_to_task("wan", hints) == _VIDEO_GEN_TASK) is video


def test_load_validation_refuses_with_the_named_reason():
    from core.inference.video import VideoBackend

    backend = VideoBackend()
    with pytest.raises(ValueError, match = _I2V):
        backend.validate_load_request("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
    with pytest.raises(ValueError, match = _I2V):
        backend.validate_load_request(
            "city96/Wan2.1-I2V-14B-480P-gguf",
            gguf_filename = "wan2.1-i2v-14b-480p-Q4_K_M.gguf",
            model_kind = "gguf",
        )
    with pytest.raises(ValueError, match = _FASTWAN):
        backend.validate_load_request("FastVideo/FastWan2.1-T2V-1.3B-Diffusers")
    with pytest.raises(ValueError, match = "not a supported text-to-video model"):
        backend.validate_load_request("someorg/some-image-model")


@pytest.mark.parametrize(
    "repo, kwargs, reason",
    [
        (
            "city96/Wan2.1-T2V-14B-gguf",
            {"gguf_filename": "wan2.1-i2v-14b-480p-Q4_K_M.gguf", "model_kind": "gguf"},
            _I2V,
        ),
        (
            "Wan-AI/Wan2.1-T2V-1.3B",
            {"gguf_filename": "wan2.1_vace_1.3B_fp16.safetensors", "model_kind": "single_file"},
            _VARIANT,
        ),
        (
            "someorg/Wan2.1-T2V-1.3B-mix",
            {"gguf_filename": "FastWan2.1-T2V-1.3B-Q8_0.gguf", "model_kind": "gguf"},
            _FASTWAN,
        ),
    ],
)
def test_unsupported_file_under_a_t2v_repo_is_refused(repo, kwargs, reason):
    # The repo id alone resolves a T2V family; the picked file must still be judged.
    from core.inference.video import VideoBackend

    assert vf.detect_video_family(repo) is not None
    with pytest.raises(ValueError, match = reason):
        VideoBackend().validate_load_request(repo, **kwargs)


@pytest.mark.parametrize(
    "repo, kwargs, family",
    [
        ("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", {}, "wan2.1-t2v-1.3b"),
        ("Wan-AI/Wan2.1-T2V-14B-Diffusers", {}, "wan2.1-t2v-14b"),
        (
            "city96/Wan2.1-T2V-14B-gguf",
            {"gguf_filename": "wan2.1-t2v-14b-Q4_K_M.gguf", "model_kind": "gguf"},
            "wan2.1-t2v-14b",
        ),
        (
            "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            {
                "gguf_filename": "split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors",
                "model_kind": "single_file",
            },
            "wan2.1-t2v-1.3b",
        ),
    ],
)
def test_load_validation_accepts_wan21_t2v(repo, kwargs, family):
    from core.inference.video import VideoBackend

    assert VideoBackend().validate_load_request(repo, **kwargs).name == family


def test_non_diffusers_upstream_repo_stays_untrusted():
    from core.inference.video import VideoBackend

    # The original-format repo has no model_index.json; like Wan-AI/Wan2.2-TI2V-5B it is not a pipeline load.
    with pytest.raises(ValueError, match = "limited to"):
        VideoBackend().validate_load_request("Wan-AI/Wan2.1-T2V-1.3B")


def test_vae_untiled_tables_cover_wan21():
    from core.inference import video_vae_untiled as vu

    for fam in ("wan2.1-t2v-1.3b", "wan2.1-t2v-14b"):
        assert vu._BYTES_PER_LATENT_PIXEL[fam] == vu._BYTES_PER_LATENT_PIXEL["wan2.2-t2v-a14b"]
        assert vu._OUTPUT_SCALE[fam] == (8, 4)
