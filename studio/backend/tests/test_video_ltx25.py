# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The ltx-2.5 video family: routing (names, model_index.json, headers), defaults, the base-pack file selection, the
Gemma 4 encoder worker path, the euler-ancestral sampler and the two-stage render orchestration. LTX-2.0 / 2.3 keep
resolving to ltx-2, and neither family ever loads the other's weights."""

from __future__ import annotations

import copy
import json
import queue
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.inference import diffusion_content as dc
from core.inference import video_ltx25 as v25
from core.inference.video_families import (
    LTX25_FAMILY,
    assert_ltx_pick_supported,
    assert_supported_ltx_version,
    default_video_generation_params,
    detect_video_family,
    ltx25_is_dev,
    ltx_family_by_name,
    ltx_family_for_version,
    snap_num_frames,
    snap_video_size,
    unsupported_ltx_version,
)

_FIXTURES = Path(__file__).parent / "fixtures"
_FIXTURE = json.loads((_FIXTURES / "comfy_checkpoint_headers.json").read_text(encoding = "utf-8"))
# Range-read header of the official distilled DiT (Lightricks/LTX-2.5, via an ungated mirror), trimmed to the
# non-block tensors and block 0. Shapes only.
_LTX25_REAL = json.loads(
    (_FIXTURES / "ltx25_checkpoint_headers.json").read_text(encoding = "utf-8")
)
_LTX25_REAL_KEY = (
    "Lightricks/LTX-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
)

LTX25_NAMES = [
    "Lightricks/LTX-2.5-Diffusers",
    "Lightricks/LTX-2.5",
    "lightricks/ltx-2.5",
    "someone/LTX-2.5-GGUF",
    "Abiray/LTX-2.5-Distilled-GGUF",
    "ltx2.5",
    "LTX25",
    "LTX-Video-2.5",
]

LTX25_FILES = [
    ("Lightricks/LTX-2.5", "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"),
    ("Lightricks/LTX-2.5", "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"),
    ("Lightricks/LTX-2.5", "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"),
    ("vantagewithai/LTX-2.5-GGUF", "distilled/ltx-2.5-22b-distilled-transformer-Q4_K_M.gguf"),
    ("Abiray/LTX-2.5-Distilled-GGUF", "LTX-2.5-Distilled-Q8_0.gguf"),
    ("someorg/quants", "ltx-2.5-22b-dev-transformer-Q4_K_M.gguf"),
]

LTX2X_NAMES = [
    "Lightricks/LTX-2",
    "Lightricks/LTX-2.3",
    "Lightricks/LTX-2.3-fp8",
    "unsloth/LTX-2-FP8",
    "unsloth/LTX-2.3-FP8",
    "unsloth/LTX-2.3-GGUF",
    "ltx-2-19b-dev.safetensors",
    "ltx-2.3-22b-distilled.safetensors",
    "LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf",
    "ltxv",
    "ltx2",
]


# ── release -> family ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "version, family",
    [("2.0", "ltx-2"), ("2.3", "ltx-2"), ("2.4", "ltx-2.5"), ("2.5", "ltx-2.5"), ("2.6", None), ("2.9", None)],
)
def test_release_maps_to_one_family(version, family):
    assert ltx_family_for_version(version) == family



@pytest.mark.parametrize(
    "name, version",
    [("Lightricks/LTX-2.10", "2.10"), ("ltx-2.12-22b-distilled.safetensors", "2.12"), ("ltx_2_10-dev", "2.10")],
)
def test_multi_digit_releases_are_refused_not_routed_to_ltx2(name, version):
    from core.inference.video import _detect_load_family

    assert unsupported_ltx_version(name) == version
    assert ltx_family_by_name(name) is None
    with pytest.raises(ValueError, match = f"LTX-{version}, which Studio does not support yet"):
        _detect_load_family(name, None, None)


@pytest.mark.parametrize("name", LTX25_NAMES)
def test_ltx25_names_resolve_to_the_ltx25_family(name):
    from core.inference.video import _detect_load_family

    assert unsupported_ltx_version(name) is None
    assert ltx_family_by_name(name) == LTX25_FAMILY
    if name.lower() == v25.LTX25_SINGLE_FILE_REPO.lower():
        # The split release has no Diffusers pipeline: a load without a DiT file is refused before any download.
        with pytest.raises(ValueError, match = "single files, not a Diffusers pipeline"):
            _detect_load_family(name, None, None)
        with pytest.raises(ValueError, match = "single files, not a Diffusers pipeline"):
            _detect_load_family(name, "diffusion_models/x.safetensors", None, None, "pipeline")
        return
    assert _detect_load_family(name, None, None).name == LTX25_FAMILY


@pytest.mark.parametrize("repo, filename", LTX25_FILES)
def test_ltx25_files_resolve_to_the_ltx25_family(repo, filename):
    from core.inference.video import _detect_load_family

    assert _detect_load_family(repo, filename, None).name == LTX25_FAMILY


@pytest.mark.parametrize("name", LTX2X_NAMES)
def test_ltx2_and_ltx23_names_still_resolve_to_ltx2(name):
    from core.inference.video import _detect_load_family

    assert detect_video_family(name).name == "ltx-2"
    assert _detect_load_family(name, None, None).name == "ltx-2"


@pytest.mark.parametrize("name", ["ltx-24fps-clip", "voltx-2.5", "ltx-23", "Lightricks/LTX-Video"])
def test_lookalikes_are_not_ltx25(name):
    assert ltx_family_by_name(name) != LTX25_FAMILY


def test_later_releases_are_refused_by_name():
    from core.inference.video import _detect_load_family

    assert unsupported_ltx_version("ltx-2.6-test") == "2.6"
    with pytest.raises(ValueError, match = "LTX-2.6, which Studio does not support"):
        _detect_load_family("someone/LTX-2.6-GGUF", "model.gguf", None)
    with pytest.raises(ValueError, match = "LTX-2.6"):
        assert_supported_ltx_version("/models/x", "ltx-2.6-22b.safetensors")


def test_picked_file_name_outranks_the_folder_both_ways():
    from core.inference.video import _detect_load_family

    assert (
        _detect_load_family("/models/ltx-2.5-tests", "ltx-2.3-22b-distilled.safetensors", None).name
        == "ltx-2"
    )
    assert (
        _detect_load_family(
            "/models/ltx-2.3-tests", "ltx-2.5-22b-distilled-transformer-bf16.safetensors", None
        ).name
        == LTX25_FAMILY
    )
    # A version-free file defers to the repo.
    assert _detect_load_family("Lightricks/LTX-2.5", "model.safetensors", None).name == LTX25_FAMILY


def test_an_override_never_crosses_releases():
    from core.inference.video import _detect_load_family

    for override in ("ltx-2", "ltx-2.3", "ltxv"):
        with pytest.raises(ValueError, match = "loads as the 'ltx-2.5' family, not 'ltx-2'"):
            _detect_load_family("Lightricks/LTX-2.5-Diffusers", None, override)
    with pytest.raises(ValueError, match = "loads as the 'ltx-2' family, not 'ltx-2.5'"):
        _detect_load_family("Lightricks/LTX-2.3", "ltx-2.3-22b-distilled.safetensors", "ltx-2.5")
    assert _detect_load_family("Lightricks/LTX-2.5-Diffusers", None, "ltx-2.5").name == LTX25_FAMILY
    # A non-LTX family never trips on an LTX token.
    fam = _detect_load_family(
        "someone/wan-ltx-2.5-style", "wan-ltx-2.5-style.safetensors", "wan2.2-ti2v-5b", None, "single_file"
    )
    assert fam.name == "wan2.2-ti2v-5b"
    assert_ltx_pick_supported("wan2.2-ti2v-5b", "someone/ltx-2.5")


def test_validate_load_request_accepts_the_official_picks():
    from core.inference.video import get_video_backend

    backend = get_video_backend()
    assert (
        backend.validate_load_request("Lightricks/LTX-2.5-Diffusers", model_kind = "pipeline").name
        == LTX25_FAMILY
    )
    fam = backend.validate_load_request(
        "Lightricks/LTX-2.5",
        gguf_filename = "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        model_kind = "single_file",
    )
    assert fam.name == LTX25_FAMILY
    with pytest.raises(ValueError, match = "'ltx-2.5' family, not 'ltx-2'"):
        backend.validate_load_request(
            "Lightricks/LTX-2.5",
            gguf_filename = "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
            model_kind = "single_file",
            family_override = "ltx-2",
        )


def test_companion_base_must_be_the_same_release(tmp_path, monkeypatch):
    import core.inference.diffusion as diffusion
    from core.inference.video import get_video_backend

    monkeypatch.setattr(diffusion, "_assert_local_base_is_pipeline", lambda *a, **k: None)
    backend = get_video_backend()
    ltx23 = dict(gguf_filename = "distilled/ltx-2.3-22b-distilled-Q4_K_M.gguf", model_kind = "gguf")
    with pytest.raises(ValueError, match = "LTX-2.5"):
        backend.validate_load_request(
            "unsloth/LTX-2.3-GGUF", base_repo = "unsloth/LTX-2.5-Diffusers", **ltx23
        )
    ltx25 = dict(gguf_filename = "distilled/ltx-2.5-22b-distilled-transformer-Q4_K_M.gguf", model_kind = "gguf")
    with pytest.raises(ValueError, match = "'ltx-2' family, not 'ltx-2.5'"):
        backend.validate_load_request("vantagewithai/LTX-2.5-GGUF", base_repo = "Lightricks/LTX-2", **ltx25)
    folder = tmp_path / "my-base"
    folder.mkdir()
    _write_index(folder, "Gemma4UnifiedForConditionalGeneration")
    with pytest.raises(ValueError, match = r"by its model_index\.json\) is LTX-2\.5"):
        backend.validate_load_request("unsloth/LTX-2.3-GGUF", base_repo = str(folder), **ltx23)
    backend.validate_load_request("vantagewithai/LTX-2.5-GGUF", base_repo = str(folder), **ltx25)


def test_training_still_refuses_ltx25(tmp_path):
    from core.training.diffusion_train_common import resolve_trainable_family

    for family in (None, "ltx-2", "ltx-2.5"):
        with pytest.raises(ValueError, match = r"ltx-2\.5|LTX-2\.5"):
            resolve_trainable_family("Lightricks/LTX-2.5-Diffusers", family)
    folder = tmp_path / "my-video-base"
    folder.mkdir()
    _write_index(folder, "Gemma4UnifiedForConditionalGeneration")
    for family in (None, "ltx-2"):
        with pytest.raises(ValueError, match = "model_index.json"):
            resolve_trainable_family(str(folder), family)
    _write_index(folder, "Gemma3ForConditionalGeneration")
    assert resolve_trainable_family(str(folder), "ltx-2") == "ltx-2"


# ── model_index.json ─────────────────────────────────────────────────────────


def _write_index(folder: Path, encoder: str, **extra) -> None:
    index = {
        "_class_name": "LTX2Pipeline",
        "text_encoder": ["transformers", encoder],
        "transformer": ["diffusers", "LTX2VideoTransformer3DModel"],
        **extra,
    }
    (folder / "model_index.json").write_text(json.dumps(index), encoding = "utf-8")


def test_local_diffusers_folder_answers_by_its_model_index(tmp_path):
    from core.inference.video import _detect_load_family

    folder = tmp_path / "my-ltx-video"
    folder.mkdir()
    _write_index(folder, "Gemma4UnifiedForConditionalGeneration")
    assert _detect_load_family(str(folder), None, None).name == LTX25_FAMILY
    with pytest.raises(ValueError, match = "model_index.json"):
        _detect_load_family(str(folder), None, "ltx-2")
    _write_index(folder, "Gemma3ForConditionalGeneration")
    assert _detect_load_family(str(folder), None, None).name == "ltx-2"
    with pytest.raises(ValueError, match = "model_index.json"):
        _detect_load_family(str(folder), None, "ltx-2.5")
    _write_index(folder, "Gemma3ForConditionalGeneration", duration_head = ["ltx2", "LTX2DurationHead"])
    assert _detect_load_family(str(folder), None, None).name == LTX25_FAMILY
    _write_index(folder, "Gemma3ForConditionalGeneration", duration_head = [None, None])
    assert _detect_load_family(str(folder), None, None).name == "ltx-2"
    # A folder named after 2.3 still answers by its index.
    named = tmp_path / "ltx-2.3-copy"
    named.mkdir()
    _write_index(named, "Gemma4UnifiedForConditionalGeneration")
    assert _detect_load_family(str(named), None, None).name == LTX25_FAMILY


# ── headers ──────────────────────────────────────────────────────────────────

_LTX23_KEYS = [
    "Lightricks/LTX-2.3/ltx-2.3-22b-distilled.safetensors",
    "Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors",
    "Lightricks/LTX-2.3-fp8/ltx-2.3-22b-distilled-fp8.safetensors",
    "Kijai/LTX2.3_comfy/diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot.safetensors",
    "Lightricks/LTX-2/ltx-2-19b-dev-fp4.safetensors",
    "QuantStack/LTX-2.3-GGUF/LTX-2.3-distilled-1.1/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf",
    "diffusers/ltx-2/LTX2VideoTransformer3DModel",
]


def _shapes(key):
    entry = _FIXTURE[key]
    return (entry["shapes"], entry.get("meta")) if "shapes" in entry else (entry, None)


def _as_ltx25(shapes):
    """A 2.x header with the 2.5 deltas: no video FFN bias, a keyframe position embedding."""
    prefix = "model.diffusion_model." if any(k.startswith("model.diffusion_model.") for k in shapes) else ""
    out = {
        k: v
        for k, v in shapes.items()
        if not (".ff.net." in k and k.endswith(".bias") and ".audio_ff." not in k)
    }
    out[f"{prefix}keyframes_abs_pos_embedding"] = [1, 4096]
    return out


@pytest.mark.parametrize("key", _LTX23_KEYS)
def test_every_ltx2x_header_still_classifies_as_ltx2(key):
    shapes, meta = _shapes(key)
    assert dc.classify_tensors(shapes, meta).family == "ltx-2"


@pytest.mark.parametrize("key", _LTX23_KEYS)
def test_ltx25_header_classifies_as_ltx25(key):
    shapes, meta = _shapes(key)
    info = dc.classify_tensors(_as_ltx25(shapes), meta)
    assert info.role == dc.ROLE_DIT and info.family == LTX25_FAMILY and info.page == dc.PAGE_VIDEO
    assert "LTX-2.5" in info.what


def test_the_real_ltx25_header_classifies_as_ltx25():
    entry = _LTX25_REAL[_LTX25_REAL_KEY]
    shapes = entry["shapes"]
    assert "model.diffusion_model.keyframes_abs_pos_embedding" in shapes
    assert "model.diffusion_model.transformer_blocks.0.ff.net.0.proj.bias" not in shapes
    info = dc.classify_tensors(shapes, entry.get("meta"))
    assert info.family == LTX25_FAMILY and info.layout == "comfy"
    assert dc.is_ltx25_dit_keys([k[len("model.diffusion_model.") :] for k in shapes])


def _write_header_only_safetensors(path: Path, shapes: dict) -> None:
    header = {k: {"dtype": "BF16", "shape": v, "data_offsets": [0, 0]} for k, v in shapes.items()}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


def test_renamed_local_ltx25_file_loads_as_ltx25_by_its_header(tmp_path):
    from core.inference.video import _detect_load_family
    from core.inference.video_ltx2 import assert_not_ltx25_checkpoint

    shapes = _LTX25_REAL[_LTX25_REAL_KEY]["shapes"]
    bad = tmp_path / "ltx-2.3-but-really-2.5.safetensors"
    _write_header_only_safetensors(bad, shapes)
    assert dc.resolve_family_with_content("ltx-2", str(bad), "video") == (LTX25_FAMILY, True)
    assert _detect_load_family(str(tmp_path), bad.name, None, None, "single_file").name == LTX25_FAMILY
    # Forced into ltx-2 it still never reaches the 2.3 assembly.
    with pytest.raises(ValueError, match = "'ltx-2.5' family, not 'ltx-2'"):
        assert_not_ltx25_checkpoint(bad)
    good23, _ = _shapes("Lightricks/LTX-2.3/ltx-2.3-22b-distilled.safetensors")
    good = tmp_path / "ltx-2.5-but-really-2.3.safetensors"
    _write_header_only_safetensors(good, good23)
    assert _detect_load_family(str(tmp_path), good.name, None, None, "single_file").name == "ltx-2"


def test_a_forced_ltx_family_still_reads_a_local_header(tmp_path):
    from core.inference.video import _detect_load_family

    shapes23, _ = _shapes("Lightricks/LTX-2.3/ltx-2.3-22b-distilled.safetensors")
    v23 = tmp_path / "v23"
    v23.mkdir()
    _write_header_only_safetensors(v23 / "model.safetensors", shapes23)
    with pytest.raises(ValueError, match = "loads as the 'ltx-2' family, not 'ltx-2.5'"):
        _detect_load_family(str(v23), "model.safetensors", "ltx-2.5", None, "single_file")
    assert _detect_load_family(str(v23), "model.safetensors", "ltx-2", None, "single_file").name == "ltx-2"
    v25_dir = tmp_path / "v25"
    v25_dir.mkdir()
    _write_header_only_safetensors(v25_dir / "model.safetensors", _LTX25_REAL[_LTX25_REAL_KEY]["shapes"])
    with pytest.raises(ValueError, match = "loads as the 'ltx-2.5' family, not 'ltx-2'"):
        _detect_load_family(str(v25_dir), "model.safetensors", "ltx-2", None, "single_file")
    assert (
        _detect_load_family(str(v25_dir), "model.safetensors", "ltx-2.5", None, "single_file").name
        == LTX25_FAMILY
    )


# ── registry + defaults ──────────────────────────────────────────────────────


def test_family_carries_the_reference_defaults():
    fam = detect_video_family("", override = LTX25_FAMILY)
    assert fam.base_repo == "Lightricks/LTX-2.5-Diffusers"
    assert fam.pipeline_class == "LTX2Pipeline" and fam.transformer_class == "LTX2VideoTransformer3DModel"
    assert fam.has_audio and fam.supports_keyframes and fam.supports_cfg
    assert (fam.default_steps, fam.default_guidance) == (8, 1.0)
    assert (fam.default_num_frames, fam.default_fps) == (121, 24)
    assert fam.resolution_presets[0] == (1280, 704)
    assert all(w % 64 == 0 and h % 64 == 0 for w, h in fam.resolution_presets)
    assert snap_video_size(fam, 1280, 720) == (1280, 704)
    assert snap_num_frames(fam, 121) == 121
    # Never the LTX-2 registry entry.
    assert detect_video_family("", override = "ltx-2").base_repo == "Lightricks/LTX-2"


@pytest.mark.parametrize(
    "ids, expected",
    [
        (("Lightricks/LTX-2.5-Diffusers",), (8, 1.0)),
        ((None, "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors", "Lightricks/LTX-2.5"), (8, 1.0)),
        ((None, "dev/ltx-2.5-22b-dev-transformer-Q4_K_M.gguf", "vantagewithai/LTX-2.5-GGUF"), (30, 3.0)),
        ((None, "LTX-2.5-Distilled-Q4_K_M.gguf", "Abiray/LTX-2.5-Distilled-GGUF"), (8, 1.0)),
        ((None, "ltx-2.5-22b-dev-transformer-bf16.safetensors", "/home/me/distilled-models"), (30, 3.0)),
    ],
)
def test_ltx25_generation_defaults(ids, expected):
    assert default_video_generation_params(*ids, fallback = (8, 1.0), family = LTX25_FAMILY) == expected


def test_other_families_keep_their_defaults():
    assert default_video_generation_params("Lightricks/LTX-2") == (40, 4.0)
    assert default_video_generation_params("ltx-2.3-22b-distilled.safetensors", "Lightricks/LTX-2.3") == (8, 1.0)
    assert default_video_generation_params("Lightricks/LTX-2", family = "ltx-2") == (40, 4.0)


def test_dev_detection_reads_the_leaf_only():
    assert ltx25_is_dev("ltx-2.5-22b-dev-transformer-bf16.safetensors")
    assert ltx25_is_dev("transformer_full")
    assert not ltx25_is_dev("/home/dev/models/ltx-2.5-22b-distilled-transformer-bf16.safetensors")
    assert not ltx25_is_dev("Lightricks/LTX-2.5-Diffusers")
    assert not ltx25_is_dev(None, "devices/LTX-2.5")


def test_keyframe_canvas_and_fit():
    from PIL import Image

    # Rounding both sides up must not leave the pixel cap (an aspect of ~0.3284 used to give 576x1664).
    for i in range(2500, 40001):
        ratio = i / 10000
        w, h = v25.ltx25_canvas_for_aspect(ratio, 1.0)
        assert w * h <= v25.LTX25_CANVAS_MAX_PIXELS, (ratio, w, h)
        assert w % 64 == 0 and h % 64 == 0
    assert v25.ltx25_canvas_for_aspect(1920, 1080) == (1280, 704)
    assert v25.ltx25_canvas_for_aspect(1080, 1920) == (704, 1280)
    w, h = v25.ltx25_canvas_for_aspect(1000, 1000)
    assert w % 64 == 0 and h % 64 == 0 and w == h
    with pytest.raises(ValueError, match = "1:4 to 4:1"):
        v25.ltx25_canvas_for_aspect(5000, 100)
    # The start image keeps its aspect: the condition pipeline crops it to each stage's canvas after the CRF.
    assert v25.prepare_ltx25_first_frame(Image.new("RGB", (300, 100))).size == (1536, 512)
    assert v25.prepare_ltx25_first_frame(Image.new("RGB", (1080, 3840))).size == (432, 1536)
    same = Image.new("RGB", (1536, 864))
    assert v25.prepare_ltx25_first_frame(same) is same


def test_status_canvas_rule_per_family():
    from core.inference.video import _keyframe_canvas

    assert _keyframe_canvas(detect_video_family("", override = LTX25_FAMILY)) == (704, 1280 * 704)
    assert _keyframe_canvas(detect_video_family("", override = "ltx-2")) == (None, None)
    h3 = detect_video_family("", override = "minimax-h3")
    from core.inference.video_minimax_h3 import H3_CANVAS_MAX_PIXELS, H3_CANVAS_SHORT_EDGE

    assert _keyframe_canvas(h3) == (H3_CANVAS_SHORT_EDGE, H3_CANVAS_MAX_PIXELS)


def test_last_frame_is_refused_and_canvas_follows_the_image():
    import base64
    import io

    from PIL import Image

    from core.inference.video import VideoBackend

    fam = detect_video_family("", override = LTX25_FAMILY)
    buf = io.BytesIO()
    Image.new("RGB", (1920, 1080), (10, 20, 30)).save(buf, format = "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    with pytest.raises(ValueError, match = "takes no last frame"):
        VideoBackend._resolve_keyframes(fam, None, b64, b64, None, None)
    # The same pure check the generate route applies before switching models; a first frame alone passes.
    from core.inference.video_families import (
        validate_video_keyframe_conditioning,
        video_family_accepts_first_frame,
        video_family_accepts_last_frame,
        video_family_uses_i2v_pipeline,
    )

    with pytest.raises(ValueError, match = "takes no last frame"):
        validate_video_keyframe_conditioning(fam, None, has_keyframes = True, has_last_frame = True)
    validate_video_keyframe_conditioning(fam, None, has_keyframes = True, has_last_frame = False)
    assert video_family_accepts_first_frame(fam) and not video_family_accepts_last_frame(fam)
    # Its first frame goes through its own two-stage render, not the generic image-to-video pipeline.
    assert not video_family_uses_i2v_pipeline(fam)
    h3 = detect_video_family("", override = "minimax-h3")
    assert video_family_accepts_last_frame(h3)
    first, last, w, h, mode = VideoBackend._resolve_keyframes(fam, None, b64, None, None, None)
    # The canvas is 1280x704; the image itself only gets its longer side at 1536, for the condition pipeline to fit.
    assert (w, h) == (1280, 704) and first.size == (1536, 864) and last is None and mode == "i2va"
    first, _, w, h, mode = VideoBackend._resolve_keyframes(fam, None, None, None, None, None)
    assert first is None and (w, h) == (1280, 704) and mode == "t2va"


# ── base-pack file selection ─────────────────────────────────────────────────

# The real Lightricks/LTX-2.5-Diffusers listing (model_info, readable without accepting the gate).
_PACK = [
    ("README.md", 29971),
    ("audio_vae/config.json", 505),
    ("audio_vae/diffusion_pytorch_model.safetensors", 106507972),
    ("connectors/config.json", 817),
    ("connectors/diffusion_pytorch_model-00001-of-00002.safetensors", 4999998520),
    ("connectors/diffusion_pytorch_model-00002-of-00002.safetensors", 1344490576),
    ("connectors/diffusion_pytorch_model.safetensors", 6344489120),
    ("connectors/diffusion_pytorch_model.safetensors.index.json", 30629),
    ("diffusion_decoder/config.json", 1089),
    ("diffusion_decoder/diffusion_pytorch_model.safetensors", 834313392),
    ("duration_head/config.json", 251),
    ("duration_head/diffusion_pytorch_model.safetensors", 3808466),
    ("latent_upsampler/config.json", 301),
    ("latent_upsampler/diffusion_pytorch_model.safetensors", 995743320),
    ("ltx-2.5-22b-distilled-lora-450-bf16.safetensors", 9699368752),
    ("model_index.json", 971),
    ("processor/tokenizer.json", 32169626),
    ("prompt_enhancer/config.json", 5278),
    ("prompt_enhancer/model-00001-of-00003.safetensors", 1422130776),
    ("scheduler/scheduler_config.json", 489),
    ("temporal_latent_upsampler/config.json", 299),
    ("temporal_latent_upsampler/diffusion_pytorch_model.safetensors", 261943760),
    ("text_encoder/config.json", 4594),
    ("text_encoder/model-00001-of-00005.safetensors", 4961189118),
    ("text_encoder/model-00002-of-00005.safetensors", 4966688198),
    ("text_encoder/model-00003-of-00005.safetensors", 4884113678),
    ("text_encoder/model-00004-of-00005.safetensors", 4986341908),
    ("text_encoder/model-00005-of-00005.safetensors", 4121215522),
    ("text_encoder/model.safetensors.index.json", 66306),
    ("tokenizer/tokenizer.json", 32169626),
    ("tokenizer/tokenizer_config.json", 3735),
    ("transformer/config.json", 1440),
    *[(f"transformer/diffusion_pytorch_model-0000{i}-of-00004.safetensors", 9_900_000_000) for i in range(1, 5)],
    *[(f"transformer/diffusion_pytorch_model-0000{i}-of-00008.safetensors", 4_950_000_000) for i in range(1, 9)],
    ("transformer/diffusion_pytorch_model.safetensors.index.json", 446505),
    ("transformer_full/config.json", 1440),
    ("transformer_full/diffusion_pytorch_model-00001-of-00004.safetensors", 9932336016),
    ("vae/config.json", 1480),
    ("vae/diffusion_pytorch_model.safetensors", 1452233194),
    ("vocoder/config.json", 1478),
    ("vocoder/diffusion_pytorch_model.safetensors", 258305056),
    # The real pack also ships a modular index (MiniMax-H3's marker elsewhere in the plan).
    ("modular_model_index.json", 3093),
]


def _info(files):
    return SimpleNamespace(siblings = [SimpleNamespace(rfilename = n, size = s) for n, s in files])


def test_pack_download_keeps_only_what_a_load_reads():
    from core.inference.video import VideoBackend

    assert v25.is_ltx25_pack(n for n, _ in _PACK)
    names = {n for n, _ in VideoBackend._base_download_files(_info(_PACK), "pipeline")}
    for gone in (
        "transformer_full/",
        "prompt_enhancer/",
        "processor/",
        "diffusion_decoder/",
        "temporal_latent_upsampler/",
        "duration_head/",
    ):
        assert not any(n.startswith(gone) for n in names), gone
    assert "ltx-2.5-22b-distilled-lora-450-bf16.safetensors" not in names
    assert "connectors/diffusion_pytorch_model.safetensors" not in names
    assert "connectors/diffusion_pytorch_model-00001-of-00002.safetensors" in names
    transformer = sorted(n for n in names if n.startswith("transformer/") and n.endswith(".safetensors"))
    # No index read: the coarsest set, the one the official pack's index names.
    assert len(transformer) == 4 and all("-of-00004" in n for n in transformer)
    for kept in (
        "model_index.json",
        "connectors/config.json",
        "vocoder/diffusion_pytorch_model.safetensors",
        "audio_vae/diffusion_pytorch_model.safetensors",
        "latent_upsampler/diffusion_pytorch_model.safetensors",
        "text_encoder/model-00005-of-00005.safetensors",
        "tokenizer/tokenizer.json",
        "vae/diffusion_pytorch_model.safetensors",
        "transformer/config.json",
    ):
        assert kept in names


def test_pack_download_follows_the_shard_index():
    from core.inference.video import VideoBackend

    shards = v25.ltx25_shards_from_indexes(
        {
            "transformer/diffusion_pytorch_model.safetensors.index.json": {
                "weight_map": {
                    f"w{i}": f"diffusion_pytorch_model-0000{i}-of-00004.safetensors" for i in range(1, 5)
                }
            },
            "connectors/diffusion_pytorch_model.safetensors.index.json": {
                "weight_map": {"a": "diffusion_pytorch_model-00001-of-00002.safetensors"}
            },
        }
    )
    names = {n for n, _ in VideoBackend._base_download_files(_info(_PACK), "pipeline", ltx25_shards = shards)}
    transformer = sorted(n for n in names if n.startswith("transformer/") and n.endswith(".safetensors"))
    assert transformer == [f"transformer/diffusion_pytorch_model-0000{i}-of-00004.safetensors" for i in range(1, 5)]
    assert {n for n in names if n.startswith("connectors/")} == {
        "connectors/config.json",
        "connectors/diffusion_pytorch_model-00001-of-00002.safetensors",
        "connectors/diffusion_pytorch_model.safetensors.index.json",
    }


def test_download_plan_reads_the_pack_indexes(monkeypatch, tmp_path):
    from core.inference.video import VideoBackend
    from utils import hf_xet_fallback

    pack = _PACK
    indexes = {
        # Pointing at the 8-shard set proves the plan follows the index rather than the listing-only choice.
        "transformer/diffusion_pytorch_model.safetensors.index.json": {
            "weight_map": {f"w{i}": f"diffusion_pytorch_model-0000{i}-of-00008.safetensors" for i in range(1, 9)}
        },
        "connectors/diffusion_pytorch_model.safetensors.index.json": {
            "weight_map": {"a": "diffusion_pytorch_model-00001-of-00002.safetensors"}
        },
    }
    fetched = []

    def fake_download(repo, name, token, **_k):
        fetched.append((repo, name))
        component = name.split("/", 1)[0]
        payload = indexes.get(name) or {
            "weight_map": {
                n: n.split("/", 1)[1]
                for n, _ in pack
                if n.startswith(component + "/") and n.endswith(".safetensors") and "-of-" in n
            }
        }
        path = tmp_path / name.replace("/", "__")
        path.write_text(json.dumps(payload))
        return str(path)

    monkeypatch.setattr(hf_xet_fallback, "hf_hub_download_with_xet_fallback", fake_download)
    info = _info(pack)
    shards = VideoBackend._ltx25_plan_shards(info, "Lightricks/LTX-2.5-Diffusers", None)
    # Only the components a load stages: never transformer_full/.
    assert ("Lightricks/LTX-2.5-Diffusers", "transformer/diffusion_pytorch_model.safetensors.index.json") in fetched
    assert not any(n.startswith("transformer_full/") for _, n in fetched)
    names = {n for n, _ in VideoBackend._base_download_files(info, "pipeline", ltx25_shards = shards)}
    transformer = sorted(n for n in names if n.startswith("transformer/") and n.endswith(".safetensors"))
    assert transformer == [f"transformer/diffusion_pytorch_model-0000{i}-of-00008.safetensors" for i in range(1, 9)]
    assert "text_encoder/model-00005-of-00005.safetensors" in names

    def offline(*_a, **_k):
        raise OSError("offline")

    monkeypatch.setattr(hf_xet_fallback, "hf_hub_download_with_xet_fallback", offline)
    assert VideoBackend._ltx25_plan_shards(info, "Lightricks/LTX-2.5-Diffusers", None) is None
    # Another family's repo never reads anything.
    fetched.clear()
    monkeypatch.setattr(hf_xet_fallback, "hf_hub_download_with_xet_fallback", fake_download)
    ltx2 = _info([(n, sz) for n, sz in pack if not n.startswith(("duration_head/", "diffusion_decoder/"))])
    assert VideoBackend._ltx25_plan_shards(ltx2, "Lightricks/LTX-2", None) is None and fetched == []


def test_single_file_pick_skips_the_pack_dit():
    from core.inference.video import VideoBackend

    names = {n for n, _ in VideoBackend._base_download_files(_info(_PACK), "gguf")}
    assert "transformer/config.json" in names
    assert not any(n.startswith("transformer/") and n.endswith(".safetensors") for n in names)
    assert "latent_upsampler/diffusion_pytorch_model.safetensors" in names


def test_dev_loads_skip_the_latent_upsampler(monkeypatch):
    from core.inference.video import VideoBackend, _ltx25_dev_load

    upsampler = "latent_upsampler/diffusion_pytorch_model.safetensors"
    dev = {n for n, _ in VideoBackend._base_download_files(_info(_PACK), "gguf", ltx25_dev = True)}
    assert upsampler not in dev and "vae/diffusion_pytorch_model.safetensors" in dev
    single = [n for n, _ in VideoBackend._base_download_files(_info(_SINGLE_REPO), "single_file", ltx25_dev = True)]
    assert sorted(single) == sorted(set(v25.LTX25_SINGLE_COMPANIONS) - {v25.LTX25_SINGLE_UPSAMPLER})
    # The two-stage switch off: no load attaches the upsampler, so no plan stages it.
    monkeypatch.setenv(v25.TWO_STAGE_ENV, "0")
    assert upsampler not in {n for n, _ in VideoBackend._base_download_files(_info(_PACK), "gguf")}
    monkeypatch.delenv(v25.TWO_STAGE_ENV)
    fam = detect_video_family("", override = LTX25_FAMILY)
    assert _ltx25_dev_load(fam, "single_file", "ltx-2.5-22b-dev-bf16.safetensors", "Lightricks/LTX-2.5")
    assert not _ltx25_dev_load(fam, "single_file", "ltx-2.5-22b-distilled-bf16.safetensors", "Lightricks/LTX-2.5")
    # A pipeline decides by its display / repo id; the official pack is distilled.
    assert not _ltx25_dev_load(fam, "pipeline", None, "Lightricks/LTX-2.5-Diffusers")
    assert _ltx25_dev_load(fam, "pipeline", None, "/models/opaque", "someone/LTX-2.5-dev-Diffusers")
    assert not _ltx25_dev_load(detect_video_family("", override = "ltx-2"), "gguf", "ltx-2.3-dev.gguf", "x")


def test_other_packs_are_untouched():
    from core.inference.video import VideoBackend

    ltx2 = [
        (n, s)
        for n, s in _PACK
        if not n.startswith(("duration_head/", "diffusion_decoder/")) and n != "modular_model_index.json"
    ]
    assert not v25.is_ltx25_pack(n for n, _ in ltx2)
    names = {n for n, _ in VideoBackend._base_download_files(_info(ltx2), "pipeline")}
    assert "transformer_full/config.json" in names
    assert "connectors/diffusion_pytorch_model.safetensors" in names


# ── the Gemma 4 encoder path ─────────────────────────────────────────────────


def test_missing_sidecar_is_a_named_refusal(monkeypatch):
    import utils.transformers_version as tv

    monkeypatch.setattr(tv, "_ensure_venv_t5_510_exists", lambda: False)
    with pytest.raises(RuntimeError) as exc:
        v25.ensure_text_encoder_sidecar()
    text = str(exc.value)
    assert "Gemma4UnifiedForConditionalGeneration" in text and ".venv_t5_510" in text
    assert "5.10.1" in text


def test_worker_reports_a_missing_sidecar_as_a_message(tmp_path):
    from core.inference import video_ltx25_te_worker as worker

    cmd, resp = queue.Queue(), queue.Queue()
    worker.run_text_encoder_worker(
        cmd_queue = cmd, resp_queue = resp, config = {"venv_dir": str(tmp_path / "nope"), "repo": "x"}
    )
    msg = resp.get_nowait()
    assert msg["type"] == "error" and "sidecar is missing" in msg["error"]
    assert worker._version_tuple("5.10.2") >= worker._version_tuple("5.10.1")
    assert worker._version_tuple("5.5.0") < worker._version_tuple("5.10.1")


def test_worker_serves_encodes_until_shutdown(monkeypatch, tmp_path):
    import torch

    from core.inference import video_ltx25_te_worker as worker

    model = SimpleNamespace(config = SimpleNamespace(model_type = "gemma4_unified", text_config = SimpleNamespace(hidden_size = 4)))
    monkeypatch.setattr(worker, "_activate_sidecar", lambda venv: None)
    monkeypatch.setattr(worker, "_load", lambda config: ("tok", model))
    monkeypatch.setattr(
        worker,
        "_encode_on",
        lambda tok, m, config, prompts, n, server_reserved_mib = None: (
            torch.ones(len(prompts), n, 8),
            torch.ones(len(prompts), n),
            "cpu",
        ),
    )
    cmd, resp = queue.Queue(), queue.Queue()
    cmd.put({"type": "encode", "id": 3, "prompts": ["a", "b"], "max_sequence_length": 5})
    cmd.put({"type": "shutdown"})
    worker.run_text_encoder_worker(cmd_queue = cmd, resp_queue = resp, config = {"venv_dir": str(tmp_path)})
    ready = resp.get_nowait()
    assert ready["type"] == "ready" and ready["model_type"] == "gemma4_unified" and ready["hidden_size"] == 4
    out = resp.get_nowait()
    assert out["type"] == "embeds" and out["id"] == 3 and tuple(out["embeds"].shape) == (2, 5, 8)
    assert out["device"] == "cpu" and out.get("host_rss_mib", 0) > 0


class _FakeClient:
    def __init__(self):
        self.calls = []

    def encode(self, texts, max_len, cancel = None):
        import torch

        self.calls.append((list(texts), max_len))
        n = len(texts)
        embeds = torch.arange(n, dtype = torch.float32).view(n, 1, 1).expand(n, max_len, 6).clone()
        return embeds, torch.ones(n, max_len, dtype = torch.long)


def test_remote_encode_prompt_keeps_diffusers_semantics():
    import torch

    pipe = SimpleNamespace(_execution_device = torch.device("cpu"), transformer = SimpleNamespace(dtype = torch.bfloat16))
    client = _FakeClient()
    v25.install_remote_encode_prompt(pipe, client)
    assert pipe._unsloth_ltx25_encoder is client
    pe, pm, ne, nm = pipe.encode_prompt(
        prompt = "  a fox ", do_classifier_free_guidance = False, num_videos_per_prompt = 2, max_sequence_length = 4
    )
    assert client.calls == [(["a fox"], 4)]
    assert pe.shape == (2, 4, 6) and pe.dtype == torch.bfloat16 and pm.shape == (2, 4)
    assert ne is None and nm is None
    client.calls.clear()
    pe, pm, ne, nm = pipe.encode_prompt(
        prompt = ["x"], negative_prompt = "blurry", do_classifier_free_guidance = True, max_sequence_length = 3
    )
    assert client.calls == [(["x"], 3), (["blurry"], 3)]
    assert ne.shape == (1, 3, 6)


def test_client_caches_recent_encodes_and_reports_a_dead_worker(monkeypatch):
    import torch

    client = v25.Ltx25TextEncoderClient(
        repo = "r", hf_token = None, cache_dir = None, local_files_only = True, device = "cpu", dtype = "bfloat16",
        resident = False, venv_dir = "/x",
    )
    client._cmd, client._resp = queue.Queue(), queue.Queue()
    client._proc = SimpleNamespace(is_alive = lambda: True, exitcode = None)
    client._ready = True

    def fake_put(msg):
        client._resp.put({"type": "embeds", "id": msg["id"], "embeds": torch.zeros(1, 2, 3), "mask": torch.ones(1, 2)})

    client._cmd.put = fake_put
    a = client.encode(["p"], 2)
    b = client.encode(["p"], 2)
    assert client._seq == 1 and torch.equal(a[0], b[0])
    client._proc = SimpleNamespace(is_alive = lambda: False, exitcode = -9)
    client._cmd.put = lambda msg: None
    monkeypatch.setattr(client, "start", lambda: None)
    with pytest.raises(v25.TextEncoderWorkerError, match = "exited"):
        client._get(None, "encoding")



def test_an_abandoned_encodes_error_does_not_fail_the_next_one():
    import torch

    client = v25.Ltx25TextEncoderClient(
        repo = "r", hf_token = None, cache_dir = None, local_files_only = True, device = "cpu", dtype = "bfloat16",
        resident = False, venv_dir = "/x",
    )
    client._cmd, client._resp = queue.Queue(), queue.Queue()
    client._proc = SimpleNamespace(is_alive = lambda: True, exitcode = None)
    client._ready = True
    # Request 1 was abandoned by a cancel; its error arrives after request 2 was sent.
    client._seq = 1
    client._resp.put({"type": "error", "id": 1, "error": "LTX-2.5 text encoder failed: CUDA out of memory"})

    def fake_put(msg):
        client._resp.put({"type": "embeds", "id": msg["id"], "embeds": torch.zeros(1, 2, 3), "mask": torch.ones(1, 2)})

    client._cmd.put = fake_put
    embeds, _mask = client.encode(["p"], 2)
    assert embeds.shape == (1, 2, 3) and client._seq == 2
    # Its own error still fails it.
    client._cmd.put = lambda msg: client._resp.put({"type": "error", "id": msg["id"], "error": "boom"})
    with pytest.raises(v25.TextEncoderWorkerError, match = "boom"):
        client.encode(["q"], 2)


# ── sampler ──────────────────────────────────────────────────────────────────


def test_ancestral_step_matches_diffusers_rf_reference():
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.schedulers.scheduling_ltx_euler_ancestral_rf import LTXEulerAncestralRFScheduler

    base = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting = False, shift = 1.0, shift_terminal = None)
    ours = v25.make_ancestral_scheduler(base)
    ref = LTXEulerAncestralRFScheduler(eta = 1.0, s_noise = 1.0)
    sigmas = list(v25.LTX25_STAGE2_SIGMAS)
    ours.set_timesteps(sigmas = sigmas, device = "cpu")
    ref.set_timesteps(sigmas = sigmas + [0.0], device = "cpu")
    torch.manual_seed(0)
    sample = torch.randn(1, 6, 4)
    a = b = sample
    g1, g2 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    ours.noise_generator = g1
    for i in range(len(sigmas)):
        out = torch.randn(1, 6, 4, generator = torch.Generator().manual_seed(100 + i))
        a = ours.step(out, ours.timesteps[i], a, return_dict = False)[0]
        b = ref.step(out, ref.timesteps[i], b, generator = g2, return_dict = False)[0]
        torch.testing.assert_close(a, b)


def test_keep_tokens_step_deterministically_and_copies_share_the_generator():
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler

    sched = v25.make_ancestral_scheduler(FlowMatchEulerDiscreteScheduler())
    sched.set_timesteps(sigmas = [0.85, 0.725, 0.421875], device = "cpu")
    gen = torch.Generator().manual_seed(1)
    sched.noise_generator = gen
    sched.keep_tokens = (2, 6)
    sample = torch.randn(1, 6, 3)
    out = torch.randn(1, 6, 3)
    nxt = sched.step(out, sched.timesteps[0], sample, return_dict = False)[0]
    expected = sample[:, :2] + (0.725 - 0.85) * out[:, :2]
    torch.testing.assert_close(nxt[:, :2], expected)
    assert not torch.allclose(nxt[:, 2:], sample[:, 2:] + (0.725 - 0.85) * out[:, 2:])
    clone = copy.deepcopy(sched)
    assert clone.noise_generator is gen and clone.keep_tokens == (2, 6)


def test_prepare_scheduler_per_variant(monkeypatch):
    from diffusers import FlowMatchEulerDiscreteScheduler

    pipe = SimpleNamespace(scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting = False))
    assert v25.prepare_scheduler(pipe, dev = False) == "euler_ancestral"
    assert hasattr(pipe.scheduler, "keep_tokens")
    pipe = SimpleNamespace(scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting = False))
    assert v25.prepare_scheduler(pipe, dev = True) == "euler"
    assert pipe.scheduler.config.use_dynamic_shifting and pipe.scheduler.config.shift_terminal == 0.1
    monkeypatch.setenv(v25.SAMPLER_ENV, "euler")
    pipe = SimpleNamespace(scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting = False))
    assert v25.prepare_scheduler(pipe, dev = False) == "euler"


# ── render orchestration ─────────────────────────────────────────────────────


class _FakePipe:
    """Records each call; returns latents shaped like the real stage-1 output."""

    vae_spatial_compression_ratio = 32
    vae_temporal_compression_ratio = 8

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.calls = []
        self.freed = 0

    def __call__(
        self,
        prompt = None,
        width = None,
        height = None,
        num_frames = None,
        sigmas = None,
        num_inference_steps = None,
        output_type = None,
        return_dict = True,
        latents = None,
        audio_latents = None,
        noise_scale = None,
        conditions = None,
        callback_on_step_end = None,
        guidance_scale = None,
        stg_scale = None,
        modality_scale = None,
        audio_guidance_scale = None,
        generator = None,
        frame_rate = None,
    ):
        import torch

        self.calls.append(
            dict(
                width = width,
                height = height,
                sigmas = sigmas,
                steps = num_inference_steps,
                output_type = output_type,
                noise_scale = noise_scale,
                latents = None if latents is None else tuple(latents.shape),
                conditions = conditions,
                keep = getattr(self.scheduler, "keep_tokens", None),
                stg = stg_scale,
                audio_cfg = audio_guidance_scale,
            )
        )
        if callback_on_step_end is not None:
            for i in range(len(sigmas) if sigmas else num_inference_steps):
                callback_on_step_end(self, i, None, {})
        if output_type == "latent":
            lat = torch.zeros(1, 8, 2, height // 32, width // 32)
            return lat, torch.zeros(1, 2, 3, 4)
        return SimpleNamespace(frames = ["final"], audio = None)

    def maybe_free_model_hooks(self):
        self.freed += 1


class _Upsampler:
    def __init__(self):
        import torch

        self.weight = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        yield self.weight

    def to(self, *_a, **_k):
        return self

    def __call__(self, latents):
        import torch

        return torch.nn.functional.interpolate(latents, scale_factor = (1, 2, 2))


def _kwargs():
    return dict(
        prompt = "p", num_inference_steps = 8, width = 1280, height = 704, num_frames = 121, generator = None,
        guidance_scale = 1.0, frame_rate = 24.0,
    )


def test_two_stage_t2v_follows_the_reference_recipe():
    pipe = _FakePipe(SimpleNamespace(keep_tokens = None))
    seen = []
    out = v25.render(
        pipe, _kwargs(), dev = False, upsampler = _Upsampler(), on_step = lambda p, i, t, kw: seen.append(i) or kw
    )
    assert out.frames == ["final"]
    s1, s2 = pipe.calls
    assert (s1["width"], s1["height"]) == (640, 352) and s1["output_type"] == "latent"
    assert s1["sigmas"] == list(v25.LTX25_STAGE1_SIGMAS) and s1["stg"] == 0.0 and s1["audio_cfg"] == 1.0
    assert (s2["width"], s2["height"]) == (1280, 704)
    assert s2["sigmas"] == [0.85, 0.725, 0.421875] and s2["noise_scale"] == 0.85
    assert s2["latents"] == (1, 8, 2, 22, 40)
    assert s1["conditions"] is None and s1["keep"] is None
    assert seen == list(range(11))
    assert v25.stage_steps(8, dev = False) == (8, 3)


def test_two_stage_i2v_conditions_both_stages(monkeypatch):
    pipe = _FakePipe(SimpleNamespace(keep_tokens = None))
    cond = _FakePipe(pipe.scheduler)
    pipe._unsloth_ltx25_condition_pipe = cond
    image = object()
    v25.render(pipe, _kwargs(), dev = False, image = image, upsampler = _Upsampler())
    assert pipe.calls == [] and len(cond.calls) == 2
    c1, c2 = (call["conditions"][0] for call in cond.calls)
    assert (c1.strength, c1.crf, c1.index) == (0.7, 18, 0) and c1.frames is image
    assert (c2.strength, c2.crf) == (1.0, 18)
    # keep tokens = the first latent frame at each stage's latent size
    assert cond.calls[0]["keep"] == (11 * 20, 11 * 20 * 16)
    assert cond.calls[1]["keep"] == (22 * 40, 22 * 40 * 16)
    assert pipe.freed == 1


def test_every_image_path_resets_the_offload_hooks(monkeypatch):
    import threading

    image = object()
    # dev, single stage
    pipe = _FakePipe(SimpleNamespace(keep_tokens = None))
    pipe._unsloth_ltx25_condition_pipe = _FakePipe(pipe.scheduler)
    v25.render(pipe, dict(_kwargs(), num_inference_steps = 30), dev = True, image = image)
    assert pipe.freed == 1
    # a cancel during the upsample
    cancel = threading.Event()

    class _CancellingUpsampler(_Upsampler):
        def __call__(self, latents):
            cancel.set()
            return super().__call__(latents)

    pipe = _FakePipe(SimpleNamespace(keep_tokens = None))
    pipe._unsloth_ltx25_condition_pipe = _FakePipe(pipe.scheduler)
    with pytest.raises(RuntimeError, match = "cancelled"):
        v25.render(pipe, _kwargs(), dev = False, image = image, upsampler = _CancellingUpsampler(), cancel = cancel)
    assert pipe.freed == 1
    # distilled with the two-stage recipe off
    monkeypatch.setenv(v25.TWO_STAGE_ENV, "0")
    pipe = _FakePipe(SimpleNamespace(keep_tokens = None))
    pipe._unsloth_ltx25_condition_pipe = _FakePipe(pipe.scheduler)
    v25.render(pipe, _kwargs(), dev = False, image = image)
    assert pipe.freed == 1
    # text-to-video runs pipe itself, whose own call resets its hooks
    pipe = _FakePipe(SimpleNamespace(keep_tokens = None))
    v25.render(pipe, _kwargs(), dev = False)
    assert pipe.freed == 0


def test_dev_and_single_stage_paths(monkeypatch):
    pipe = _FakePipe(SimpleNamespace())
    v25.render(pipe, dict(_kwargs(), num_inference_steps = 30, guidance_scale = 3.0), dev = True)
    (call,) = pipe.calls
    assert call["steps"] == 30 and call["sigmas"] is None and call["stg"] is None
    assert v25.stage_steps(30, dev = True) == (30, 0)
    monkeypatch.setenv(v25.TWO_STAGE_ENV, "0")
    pipe = _FakePipe(SimpleNamespace())
    v25.render(pipe, _kwargs(), dev = False)
    (call,) = pipe.calls
    assert (call["width"], call["height"]) == (1280, 704) and call["sigmas"] == list(v25.LTX25_STAGE1_SIGMAS)
    assert v25.stage_steps(8, dev = False) == (8, 0)


def test_two_stage_without_an_upsampler_is_a_clear_error():
    with pytest.raises(RuntimeError, match = "latent upsampler"):
        v25.render(_FakePipe(SimpleNamespace()), _kwargs(), dev = False, upsampler = None)


class _OffloadedUpsampler(_Upsampler):
    """Starts off the latents' device; ``to`` swaps the parameter's device in place, as Module.to() does."""

    def __init__(self):
        import torch

        super().__init__()
        self.param = SimpleNamespace(device = torch.device("meta"), dtype = torch.float32)
        self.moves = []

    def parameters(self):
        yield self.param

    def to(self, device, *_a, **_k):
        import torch

        self.moves.append(torch.device(device))
        self.param.device = torch.device(device)
        return self


def test_a_cancel_during_the_upsample_skips_stage_two():
    import threading

    cancel = threading.Event()

    class _CancellingUpsampler(_Upsampler):
        def __call__(self, latents):
            cancel.set()
            return super().__call__(latents)

    pipe = _FakePipe(SimpleNamespace(keep_tokens = None))
    with pytest.raises(RuntimeError, match = "cancelled"):
        v25.render(pipe, _kwargs(), dev = False, upsampler = _CancellingUpsampler(), cancel = cancel)
    assert len(pipe.calls) == 1


def test_an_offloaded_upsampler_goes_back_where_it_was():
    import torch

    upsampler = _OffloadedUpsampler()
    v25.render(_FakePipe(SimpleNamespace(keep_tokens = None)), _kwargs(), dev = False, upsampler = upsampler)
    assert upsampler.moves == [torch.device("cpu"), torch.device("meta")]
    assert upsampler.param.device == torch.device("meta")


# ── the real 2.5 DiT layout lands on the 2.5 diffusers transformer ───────────

# diffusers' scripts/convert_ltx2_to_diffusers.py, version "2.5" (what Lightricks/LTX-2.5-Diffusers transformer/ holds).
_LTX25_TRANSFORMER_CONFIG = {
    "in_channels": 128, "out_channels": 128, "patch_size": 1, "patch_size_t": 1, "num_attention_heads": 32,
    "attention_head_dim": 128, "cross_attention_dim": 4096, "vae_scale_factors": (8, 32, 32), "pos_embed_max_pos": 20,
    "base_height": 2048, "base_width": 2048, "gated_attn": True, "cross_attn_mod": True, "audio_in_channels": 128,
    "audio_out_channels": 128, "audio_patch_size": 1, "audio_patch_size_t": 1, "audio_num_attention_heads": 32,
    "audio_attention_head_dim": 64, "audio_cross_attention_dim": 2048, "audio_scale_factor": 4,
    "audio_pos_embed_max_pos": 20, "audio_sampling_rate": 16000, "audio_hop_length": 160, "audio_gated_attn": True,
    "audio_cross_attn_mod": True, "num_layers": 48, "activation_fn": "gelu-approximate",
    "qk_norm": "rms_norm_across_heads", "norm_elementwise_affine": False, "norm_eps": 1e-6, "caption_channels": 3840,
    "attention_bias": True, "attention_out_bias": True, "rope_theta": 10000.0, "rope_double_precision": True,
    "causal_offset": 1, "timestep_scale_multiplier": 1000, "cross_attn_timestep_scale_multiplier": 1000,
    "rope_type": "split", "use_prompt_embeddings": False, "perturbed_attn": True, "ff_bias": False,
    "use_keyframes_abs_pos_embedding": True,
}


@pytest.mark.parametrize(
    "key",
    [_LTX25_REAL_KEY, "vantagewithai/LTX-2.5-GGUF/distilled/ltx-2.5-22b-distilled-transformer-Q4_K_M.gguf"],
)
def test_real_ltx25_header_converts_onto_the_ltx25_transformer(key):
    """Every DiT tensor of the official file and of a community GGUF (non-block tensors + block 0, by name and logical
    shape; range-read headers) lands on a parameter of
    the same shape after load_ltx25_transformer's key handling (2.3 pre-rename + diffusers' converter), with nothing
    missing; the connector stacks in the file are dropped (they come from the pack)."""
    import torch
    from accelerate import init_empty_weights
    from diffusers import LTX2VideoTransformer3DModel
    from diffusers.loaders.single_file_utils import convert_ltx2_transformer_to_diffusers

    from core.inference.video_ltx2 import _ltx23_pre_convert, ltx23_is_dit_key

    with init_empty_weights():
        model = LTX2VideoTransformer3DModel.from_config(_LTX25_TRANSFORMER_CONFIG)
    expected = {
        k: tuple(v.shape)
        for k, v in model.state_dict().items()
        if not k.startswith("transformer_blocks.") or k.startswith("transformer_blocks.0.")
    }
    shapes = _LTX25_REAL[key]["shapes"]
    state = {k: torch.empty(v, device = "meta") for k, v in shapes.items() if ltx23_is_dit_key(k)}
    converted = convert_ltx2_transformer_to_diffusers(_ltx23_pre_convert(state))
    got = {k: tuple(v.shape) for k, v in converted.items()}
    assert got == expected


# ── the split single-file release as its own base ───────────────────────────

# Lightricks/LTX-2.5's listing (sizes rounded).
_SINGLE_REPO = [
    (".gitattributes", 1600),
    ("README.md", 9000),
    ("diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors", 42_020_000_000),
    ("diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors", 42_020_000_000),
    ("diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors", 21_500_000_000),
    ("diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors", 18_720_000_000),
    ("hf-hero-web.webp", 90_000),
    ("latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors", 1_000_000_000),
    ("latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors", 260_000_000),
    ("loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors", 8_900_000_000),
    ("model_patches/ltx-2.5-duration-head-bf16.safetensors", 4_000_000),
    ("text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors", 26_260_000_000),
    ("text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors", 15_370_000_000),
    ("vae/ltx-2.5-audio-vae-bf16.safetensors", 360_000_000),
    ("vae/ltx-2.5-video-vae-bf16.safetensors", 1_470_000_000),
    ("vae/ltx-2.5-video-vae-conv-bf16.safetensors", 1_450_000_000),
]


def test_single_file_release_is_its_own_base(tmp_path):
    from core.inference.video_families import resolve_video_base_repo

    fam = detect_video_family("Lightricks/LTX-2.5-Diffusers")
    assert fam.name == LTX25_FAMILY
    assert v25.is_ltx25_single_file_base("Lightricks/LTX-2.5")
    assert v25.is_ltx25_single_file_base("lightricks/ltx-2.5")
    assert not v25.is_ltx25_single_file_base("Lightricks/LTX-2.5-Diffusers")
    assert not v25.is_ltx25_single_file_base("")
    # A DiT picked from the release defaults to it; an explicit base wins; any other pick keeps the pack.
    assert resolve_video_base_repo(fam, None, "Lightricks/LTX-2.5") == "Lightricks/LTX-2.5"
    assert resolve_video_base_repo(fam, "Lightricks/LTX-2.5-Diffusers", "Lightricks/LTX-2.5") == (
        "Lightricks/LTX-2.5-Diffusers"
    )
    assert resolve_video_base_repo(fam, None, "someone/ltx-2.5-gguf") == "Lightricks/LTX-2.5-Diffusers"
    ltx2 = detect_video_family("Lightricks/LTX-2")
    assert resolve_video_base_repo(ltx2, None, "Lightricks/LTX-2.5") == ltx2.base_repo
    # A local folder counts once the encoder file and the conv VAE sit in it.
    assert not v25.is_ltx25_single_file_base(str(tmp_path))
    for name in (v25.LTX25_SINGLE_TEXT_ENCODER, v25.LTX25_SINGLE_VIDEO_VAE):
        (tmp_path / name).parent.mkdir(parents = True, exist_ok = True)
        (tmp_path / name).write_bytes(b"x")
    assert v25.is_ltx25_single_file_base(str(tmp_path))
    assert resolve_video_base_repo(fam, None, str(tmp_path)) == str(tmp_path)
    assert v25.ltx25_companion_path(str(tmp_path), v25.LTX25_SINGLE_VIDEO_VAE, None) == str(
        tmp_path / v25.LTX25_SINGLE_VIDEO_VAE
    )
    with pytest.raises(FileNotFoundError, match = "audio-vae"):
        v25.ltx25_companion_path(str(tmp_path), v25.LTX25_SINGLE_AUDIO_VAE, None)


def test_single_file_release_downloads_only_the_companions():
    from core.inference.video import VideoBackend

    assert v25.is_ltx25_single_file_listing(n for n, _ in _SINGLE_REPO)
    assert not v25.is_ltx25_pack(n for n, _ in _SINGLE_REPO)
    names = [n for n, _ in VideoBackend._base_download_files(_info(_SINGLE_REPO), "single_file")]
    assert sorted(names) == sorted(v25.LTX25_SINGLE_COMPANIONS)
    assert not v25.is_ltx25_single_file_listing(n for n, _ in _PACK)


def test_text_encoder_key_mapping_reaches_the_gemma4_names():
    import re

    def mapped(key):
        for pattern, repl in v25.LTX25_TEXT_ENCODER_KEY_MAPPING.items():
            key = re.sub(pattern, repl, key)
        return key

    # Real names from the encoder file -> the pack's text_encoder/ names (checked on the full headers).
    assert mapped("model.layers.0.mlp.down_proj.weight") == "model.language_model.layers.0.mlp.down_proj.weight"
    assert mapped("model.embed_tokens.weight") == "model.language_model.embed_tokens.weight"
    assert mapped("model.norm.weight") == "model.language_model.norm.weight"
    assert mapped("audio_projector.embedding_projection.weight") == "model.embed_audio.embedding_projection.weight"
    assert mapped("multi_modal_projector.embedding_projection.weight") == (
        "model.embed_vision.embedding_projection.weight"
    )
    assert mapped("vision_model.pos_embedding") == "model.vision_embedder.pos_embedding"
    assert mapped("model.language_model.norm.weight") == "model.language_model.norm.weight"
    assert mapped("text_embedding_projection.video_aggregate_embed.weight").startswith("text_embedding_projection.")


def test_config_dir_survives_a_concurrent_writer(monkeypatch, tmp_path):
    import os
    import tempfile

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    real_replace = os.replace
    nested = []

    def replace(src, dst):
        # Another load in this process writes the same config between this one's scratch write and its replace.
        if not nested:
            nested.append(True)
            v25.ltx25_transformer_config_dir()
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    folder = Path(v25.ltx25_transformer_config_dir())
    config = folder / "transformer" / "config.json"
    assert nested and json.loads(config.read_text(encoding = "utf-8"))["ff_bias"] is False
    assert [p.name for p in config.parent.iterdir()] == ["config.json"]


def test_embedded_transformer_config_matches_the_tested_layout():
    # The config the real-header conversion test builds on, plus the two keys the pack's config spells out.
    expected = {k: (list(v) if isinstance(v, tuple) else v) for k, v in _LTX25_TRANSFORMER_CONFIG.items()}
    extra = {k: v for k, v in v25.LTX25_TRANSFORMER_CONFIG.items() if k not in expected}
    assert extra == {"audio_ff_bias": True, "use_prompt_adaln_single": True}
    assert {k: v for k, v in v25.LTX25_TRANSFORMER_CONFIG.items() if k in expected} == expected
    folder = Path(v25.ltx25_transformer_config_dir())
    written = json.loads((folder / "transformer" / "config.json").read_text(encoding = "utf-8"))
    assert written["_class_name"] == "LTX2VideoTransformer3DModel" and written["ff_bias"] is False
    assert v25._transformer_config_source("Lightricks/LTX-2.5") == str(folder)
    assert v25._transformer_config_source("Lightricks/LTX-2.5-Diffusers") == "Lightricks/LTX-2.5-Diffusers"


def test_single_file_assembly_refuses_quantized_connectors(tmp_path):
    import torch
    from safetensors.torch import save_file

    path = tmp_path / "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
    save_file(
        {
            "model.diffusion_model.video_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight": torch.zeros(
                2, 2, dtype = torch.int8
            ),
            "model.diffusion_model.video_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight_scale": (
                torch.ones(1)
            ),
            "model.diffusion_model.proj_out.bias": torch.zeros(2),
        },
        str(path),
    )
    with pytest.raises(ValueError, match = "connectors quantized"):
        v25.assemble_ltx25_single_file_pipeline(
            object(), path, base = "Lightricks/LTX-2.5", torch_dtype = torch.bfloat16, hf_token = None
        )


def test_encoder_worker_reads_the_single_file_release(tmp_path):
    import os

    import numpy as np
    import torch
    from safetensors.torch import save_file

    from core.inference import video_ltx25_te_worker as worker

    name = v25.LTX25_SINGLE_TEXT_ENCODER
    path = tmp_path / name
    path.parent.mkdir(parents = True)
    tokenizer_json = b'{"version": "1.0"}'
    tokenizer_config = b'{"tokenizer_class": "GemmaTokenizer"}'
    save_file(
        {
            "model.norm.weight": torch.ones(4),
            "tokenizer_json": torch.from_numpy(np.frombuffer(tokenizer_json, dtype = np.uint8).copy()),
            "hf_asset__tokenizer_config.json": torch.from_numpy(
                np.frombuffer(tokenizer_config, dtype = np.uint8).copy()
            ),
        },
        str(path),
        metadata = {"gemma_config": json.dumps({"model_type": "gemma4_unified"}), "format": "pt"},
    )
    client = v25.Ltx25TextEncoderClient(
        repo = str(tmp_path),
        hf_token = None,
        cache_dir = None,
        local_files_only = True,
        device = "cpu",
        dtype = "bfloat16",
        resident = False,
        venv_dir = "/nonexistent",
        single_file = name,
    )
    assert client.config["single_file"] == name
    assert client.config["key_mapping"] == v25.LTX25_TEXT_ENCODER_KEY_MAPPING
    folder = Path(worker._single_file_folder(client.config))
    try:
        assert json.loads((folder / "config.json").read_text()) == {"model_type": "gemma4_unified"}
        assert (folder / "tokenizer.json").read_bytes() == tokenizer_json
        assert (folder / "tokenizer_config.json").read_bytes() == tokenizer_config
        assert os.path.realpath(folder / "model.safetensors") == os.path.realpath(path)
    finally:
        import shutil

        shutil.rmtree(folder)
    # A pack base keeps the subfolder layout.
    pack_client = v25.Ltx25TextEncoderClient(
        repo = "Lightricks/LTX-2.5-Diffusers",
        hf_token = None,
        cache_dir = None,
        local_files_only = True,
        device = "cpu",
        dtype = "bfloat16",
        resident = False,
        venv_dir = "/nonexistent",
    )
    assert pack_client.config["single_file"] is None and pack_client.config["key_mapping"] is None


def _write_single_file_encoder(folder: Path) -> Path:
    import numpy as np
    import torch
    from safetensors.torch import save_file

    path = folder / v25.LTX25_SINGLE_TEXT_ENCODER
    path.parent.mkdir(parents = True, exist_ok = True)
    save_file(
        {
            "model.norm.weight": torch.ones(4),
            "tokenizer_json": torch.from_numpy(np.frombuffer(b"{}", dtype = np.uint8).copy()),
            "hf_asset__tokenizer_config.json": torch.from_numpy(np.frombuffer(b"{}", dtype = np.uint8).copy()),
        },
        str(path),
        metadata = {"gemma_config": json.dumps({"model_type": "gemma4_unified"}), "format": "pt"},
    )
    return path


def test_encoder_worker_links_without_symlink_rights(monkeypatch, tmp_path):
    import os
    import shutil

    from core.inference import video_ltx25_te_worker as worker

    path = _write_single_file_encoder(tmp_path / "weights")
    config = {"repo": str(tmp_path / "weights"), "single_file": v25.LTX25_SINGLE_TEXT_ENCODER}

    def no_symlink(*_a, **_k):
        raise OSError(1314, "A required privilege is not held by the client")

    monkeypatch.setattr(worker.os, "symlink", no_symlink)
    folder = Path(worker._single_file_folder(config))
    try:
        linked = folder / "model.safetensors"
        assert not linked.is_symlink() and os.path.samefile(linked, path)
    finally:
        shutil.rmtree(folder)
    assert path.is_file()

    # The temp dir on another volume: the scratch folder goes beside the weights, and is the caller's to remove.
    real_link = os.link

    def link_beside_only(src, dst):
        if Path(dst).parent.parent != path.parent:
            raise OSError(18, "Invalid cross-device link")
        real_link(src, dst)

    monkeypatch.setattr(worker.os, "link", link_beside_only)
    folder = Path(worker._single_file_folder(config))
    try:
        assert folder.parent == path.parent and os.path.samefile(folder / "model.safetensors", path)
        assert json.loads((folder / "config.json").read_text()) == {"model_type": "gemma4_unified"}
    finally:
        shutil.rmtree(folder)
    assert path.is_file() and sorted(p.name for p in path.parent.iterdir()) == [path.name]


def test_a_failed_load_stops_only_its_own_encoder_worker():
    from core.inference import video as vid

    class _Client:
        closed = False

        def close(self):
            self.closed = True

    backend = vid.VideoBackend()
    newer = _Client()
    backend._ltx25_pending_encoder = (2, newer)
    # A superseded load (token 1) unwinding after the newer load started its worker leaves that worker alone.
    backend._close_pending_ltx25_encoder(1)
    assert not newer.closed and backend._ltx25_pending_encoder == (2, newer)
    backend._close_pending_ltx25_encoder(2)
    assert newer.closed and backend._ltx25_pending_encoder is None
    # Untokened, whichever load started it.
    other = _Client()
    backend._ltx25_pending_encoder = (3, other)
    backend._close_pending_ltx25_encoder()
    assert other.closed and backend._ltx25_pending_encoder is None



def test_unload_stops_a_pending_encoder_worker():
    from core.inference import video as vid

    class _Client:
        closed = False

        def close(self):
            self.closed = True

    backend = vid.VideoBackend()
    pending = _Client()
    backend._ltx25_pending_encoder = (backend._load_token, pending)
    backend.unload()
    # Stopped by unload itself, not when the cancelled load thread finally unwinds.
    assert pending.closed and backend._ltx25_pending_encoder is None


def test_encoder_worker_is_spawned_on_the_lifetime_thread(monkeypatch):
    """The child arms PR_SET_PDEATHSIG, which follows the spawning thread; a worker started from the (short-lived)
    load thread died as soon as the load returned, so the first generate restarted it."""
    import multiprocessing
    import threading

    import utils.process_lifetime as lifetime

    spawned_via = []

    class _Proc:
        pid = 4242

        def __init__(self, **kwargs):
            self.started_on = None

        def start(self):
            self.started_on = threading.current_thread().name

    class _Ctx:
        Queue = queue.Queue

        def Process(self, **kwargs):  # noqa: N802 -- mirrors the multiprocessing API
            return _Proc(**kwargs)

    def _spawn(fn):
        spawned_via.append(fn)
        return fn()

    monkeypatch.setattr(multiprocessing, "get_context", lambda method: _Ctx())
    monkeypatch.setattr(lifetime, "spawn_on_lifetime_thread", _spawn)
    monkeypatch.setattr(lifetime, "adopt_pid", lambda pid: None)
    client = v25.Ltx25TextEncoderClient(
        repo = "Lightricks/LTX-2.5-Diffusers",
        hf_token = None,
        cache_dir = None,
        local_files_only = True,
        device = "cpu",
        dtype = "bfloat16",
        resident = False,
        venv_dir = "/nonexistent",
    )
    worker = threading.Thread(target = client.start, name = "short-lived-load")
    worker.start()
    worker.join()
    assert len(spawned_via) == 1 and spawned_via[0] == client._proc.start


# ── the encoder worker inside the memory plan ────────────────────────────────


@pytest.mark.parametrize(
    "policy, stream_te, resident",
    [
        ("none", False, True),
        ("group", False, True),  # companions resident: the plan budgeted the encoder on the card
        ("group", True, False),  # streamed text encoders: the plan did not
        ("model", False, False),  # one module on the card at a time
        ("streaming", True, False),
        ("sequential", False, False),
    ],
)
def test_encoder_placement_follows_the_plan_tier(policy, stream_te, resident):
    plan = SimpleNamespace(offload_policy = policy, stream_text_encoders = stream_te)
    assert v25.encoder_resident_for_plan(plan) is resident


def test_family_table_prices_the_worker_encoder():
    """The plan counts the worker's Gemma 4 as the family's text encoder term: the pack's text_encoder/ weights are
    22.28 GiB (23.9 GB), the table's second entry."""
    fam = detect_video_family("Lightricks/LTX-2.5-Diffusers")
    assert fam.bf16_components_gb[1] == pytest.approx(22.28 * 2**30 / 1e9, rel = 0.01)


class _FakeEncoderModel:
    def __init__(self):
        import torch

        self.weight = torch.zeros(1024, 1024, dtype = torch.bfloat16)  # 2 MiB
        self.dtype = torch.bfloat16
        self.moves = []

    def parameters(self):
        return [self.weight]

    def buffers(self):
        return []

    def to(self, device):
        self.moves.append(str(device))
        return self

    def __call__(self, input_ids, attention_mask, output_hidden_states):
        import torch

        b, n = input_ids.shape
        return SimpleNamespace(hidden_states = tuple(torch.full((b, n, 4), float(i)) for i in range(3)))


class _FakeTokenizer:
    padding_side = "right"
    pad_token = "<pad>"

    def __call__(self, prompts, **kwargs):
        import torch

        n = kwargs["max_length"]
        return SimpleNamespace(
            input_ids = torch.ones(len(prompts), n, dtype = torch.long),
            attention_mask = torch.ones(len(prompts), n, dtype = torch.long),
        )


def _patch_free(monkeypatch, free_mib):
    import torch

    def _mem_get_info(device = None):
        if free_mib is None:
            raise RuntimeError("no device")
        return free_mib << 20, 80 << 30

    monkeypatch.setattr(torch.cuda, "mem_get_info", _mem_get_info)


def test_offloaded_encoder_onloads_only_when_the_card_has_room(monkeypatch):
    from core.inference import video_ltx25_te_worker as worker

    model = _FakeEncoderModel()
    monkeypatch.setattr(worker, "ENCODE_MARGIN_MIB", 10)
    offloaded = {"device": "cuda:0", "resident": False}
    _patch_free(monkeypatch, 12)  # weights 2 MiB + margin 10 MiB
    assert worker._encode_device(offloaded, model) == "cuda:0"
    _patch_free(monkeypatch, 11)
    assert worker._encode_device(offloaded, model) == "cpu"
    _patch_free(monkeypatch, None)  # unreadable: keep the planned onload
    assert worker._encode_device(offloaded, model) == "cuda:0"
    _patch_free(monkeypatch, 0)
    # A resident encoder is already on the card; other devices are not probed.
    assert worker._encode_device({"device": "cuda:0", "resident": True}, model) == "cuda:0"
    assert worker._encode_device({"device": "mps", "resident": False}, model) == "mps"



def test_offloaded_encoder_on_xpu_is_budgeted_too(monkeypatch):
    import torch

    from core.inference import video_ltx25_te_worker as worker

    model = _FakeEncoderModel()
    monkeypatch.setattr(worker, "ENCODE_MARGIN_MIB", 10)
    offloaded = {"device": "xpu:0", "resident": False}
    free = {"mib": 12}

    def mem_get_info(_device):
        if free["mib"] is None:
            raise RuntimeError("no reading")
        return free["mib"] << 20, 80 << 30

    monkeypatch.setattr(torch, "xpu", SimpleNamespace(mem_get_info = mem_get_info), raising = False)
    assert worker._encode_device(offloaded, model) == "xpu:0"
    free["mib"] = 11
    assert worker._encode_device(offloaded, model) == "cpu"
    # Neither a free reading nor a budget: an XPU cannot be judged, so the encode stays on the CPU.
    free["mib"] = None
    assert worker._encode_device(offloaded, model) == "cpu"
    budgeted = dict(offloaded, budget_mib = 1000)
    monkeypatch.setattr(worker, "WORKER_CONTEXT_MIB", 100)
    assert worker._encode_device(budgeted, model, server_reserved_mib = 888) == "xpu:0"
    assert worker._encode_device(budgeted, model, server_reserved_mib = 889) == "cpu"


def test_short_card_encodes_on_the_cpu_without_moving_the_encoder(monkeypatch):
    import torch

    from core.inference import video_ltx25_te_worker as worker

    model = _FakeEncoderModel()
    _patch_free(monkeypatch, 1)
    embeds, mask, device = worker._encode_on(
        _FakeTokenizer(), model, {"device": "cuda:0", "resident": False}, ["a fox"], 6
    )
    assert device == "cpu" and model.moves == []
    # Same packing as the GPU path: every hidden state stacked on a last axis, flattened.
    assert embeds.shape == (1, 6, 4 * 3) and embeds.dtype == torch.bfloat16
    assert mask.shape == (1, 6)
    assert worker._encode(_FakeTokenizer(), model, {"device": "cpu", "resident": False}, ["a"], 2)[0].shape == (
        1,
        2,
        12,
    )


def test_worker_reports_its_own_memory():
    from core.inference import video_ltx25_te_worker as worker

    stats = worker._memory_stats("cpu")
    assert stats.get("host_rss_mib", 0) > 0 and "gpu_peak_allocated_mib" not in stats


class _Planned(Exception):
    pass


def _ltx25_pack_plan(monkeypatch, total_gib):
    """Drive ``load_pipeline`` for the LTX-2.5 pack on a card of ``total_gib`` up to the encoder worker's start; return
    the plan it is placed by, the plan inputs and the placement."""
    import core.inference.diffusion_auto_policy as auto_policy
    import core.inference.diffusion_te_prequant as te_prequant
    import core.inference.video as vid
    from core.inference.diffusion_memory import DeviceMemory

    total = int(total_gib * 1024)
    target = SimpleNamespace(
        device = "cuda", dtype = None, torch_device = "cuda", ordinal = None, supports_model_cpu_offload = True
    )
    backend = vid.VideoBackend()
    monkeypatch.setattr(backend, "_device_target", lambda *a, **k: target)
    monkeypatch.setattr(
        vid,
        "settled_snapshot_device_memory",
        lambda _t: DeviceMemory("cuda", "cuda", "discrete_vram", total - 600, total),
    )
    monkeypatch.setattr(te_prequant, "te_prequant_budget_scale", lambda *a, **k: 1.0)
    # The family table, not whatever pack sizes this machine's cache holds.
    monkeypatch.setattr(auto_policy, "base_repo_bf16_components_gb", lambda *_a, **_k: None)
    inputs = []
    real = vid.plan_diffusion_memory

    def capture(**kw):
        inputs.append(kw)
        return real(**kw)

    monkeypatch.setattr(vid, "plan_diffusion_memory", capture)
    seen = {}

    def placement(plan):
        seen["plan"], seen["resident"] = plan, v25.encoder_resident_for_plan.__wrapped__(plan)
        raise _Planned()

    placement.__wrapped__ = v25.encoder_resident_for_plan
    monkeypatch.setattr(v25, "encoder_resident_for_plan", placement)
    with pytest.raises(_Planned):
        backend.load_pipeline("Lightricks/LTX-2.5-Diffusers", local_files_only = True, model_kind = "pipeline")
    return seen["plan"], inputs[-1], seen["resident"]


@pytest.mark.parametrize(
    "total_gib, policy, stream_te, stream_dit, resident",
    [
        # Companions alone exceed the card: the DiT and the encoder both stream; the worker never holds Gemma 4.
        (24, "group", True, True, False),
        # The DiT streams, the companions (Gemma 4 included) stay: the worker keeps it on the card.
        (48, "group", False, True, True),
        # The DiT fits once the encoder is not resident: the worker onloads only for an encode, when there is room.
        (80, "group", True, False, False),
        (180, "none", False, True, True),
    ],
)
def test_plan_counts_the_encoder_worker_per_card(monkeypatch, total_gib, policy, stream_te, stream_dit, resident):
    from core.inference.video_ltx25 import LTX25_WORKER_RESIDENT_GB

    plan, inputs, placed_resident = _ltx25_pack_plan(monkeypatch, total_gib)
    mib = 1000.0**3 / 2**20
    te_mib = int(23.9 * mib)
    # The worker's weights are the encoder term, its context the always-resident companion share.
    assert inputs["text_encoder_dense_mib"] == te_mib
    assert inputs["companion_dense_mib"] == int((23.9 + 9.2 + LTX25_WORKER_RESIDENT_GB) * mib)
    assert inputs["model_dense_mib"] == int((42.0 + 23.9 + 9.2 + LTX25_WORKER_RESIDENT_GB) * mib)
    assert (plan.offload_policy, plan.stream_text_encoders, plan.stream_transformer) == (
        policy,
        stream_te,
        stream_dit,
    )
    assert placed_resident is resident
    if resident:
        # Whatever keeps the encoder resident budgeted its bytes on the card.
        budget = plan.estimates["safe_device_budget_mib"]
        floor = plan.estimates["resident_required_mib"] if policy == "none" else plan.estimates["group_floor_mib"]
        assert floor <= budget and inputs["text_encoder_dense_mib"] <= floor


class _DeviceTensor:
    """Input ids / mask that accept a move to any device (no accelerator in CPU CI)."""

    def __init__(self, real):
        self.real = real
        self.shape = real.shape

    def to(self, device):
        return self if not str(device).startswith("cpu") else self.real


class _AnyDeviceTokenizer(_FakeTokenizer):
    def __call__(self, prompts, **kwargs):
        out = super().__call__(prompts, **kwargs)
        return SimpleNamespace(input_ids = _DeviceTensor(out.input_ids), attention_mask = _DeviceTensor(out.attention_mask))


def test_offloaded_encoder_is_released_after_each_encode_and_reloaded_for_the_next(monkeypatch):
    from core.inference import video_ltx25_te_worker as worker

    monkeypatch.setattr(worker, "ENCODE_MARGIN_MIB", 0)
    _patch_free(monkeypatch, 1 << 20)
    model = _FakeEncoderModel()
    config = {"device": "cuda:0", "resident": False}
    for _ in range(2):
        _embeds, _mask, device = worker._encode_on(_AnyDeviceTokenizer(), model, config, ["a fox"], 4)
        assert device == "cuda:0"
    # Onto the card for each encode, back to the host after it.
    assert model.moves == ["cuda:0", "cpu", "cuda:0", "cpu"]
    resident = _FakeEncoderModel()
    worker._encode_on(_AnyDeviceTokenizer(), resident, {"device": "cuda:0", "resident": True}, ["a fox"], 4)
    assert resident.moves == []  # loaded onto the card once, at worker start


def test_offloaded_encoder_borrows_only_budget_the_server_leaves(monkeypatch):
    """A free card is not enough: the plan did not budget the encoder, so it may only use the load's budget minus what
    the server holds at encode time."""
    from core.inference import video_ltx25_te_worker as worker

    model = _FakeEncoderModel()  # 2 MiB
    monkeypatch.setattr(worker, "ENCODE_MARGIN_MIB", 10)
    monkeypatch.setattr(worker, "WORKER_CONTEXT_MIB", 100)
    _patch_free(monkeypatch, 1 << 20)  # the card itself is wide open (a shared or larger card)
    config = {"device": "cuda:0", "resident": False, "budget_mib": 1000}
    assert worker._encode_device(config, model, server_reserved_mib = 888) == "cuda:0"  # 1000 - 888 - 100 = 12
    assert worker._encode_device(config, model, server_reserved_mib = 889) == "cpu"
    # No server figure: the card's free memory decides, as before.
    assert worker._encode_device(config, model, server_reserved_mib = None) == "cuda:0"
    # No free figure: the budget alone decides.
    _patch_free(monkeypatch, None)
    assert worker._encode_device(config, model, server_reserved_mib = 889) == "cpu"
    assert worker._encode_device(config, model, server_reserved_mib = 0) == "cuda:0"


def test_client_sends_the_budget_and_the_servers_share():
    client = v25.Ltx25TextEncoderClient(
        repo = "Lightricks/LTX-2.5-Diffusers",
        hf_token = None,
        cache_dir = None,
        local_files_only = True,
        device = "cpu",
        dtype = "bfloat16",
        resident = False,
        venv_dir = "/nonexistent",
        budget_mib = 43622,
    )
    assert client.config["budget_mib"] == 43622
    assert v25._server_reserved_mib("cpu") is None
