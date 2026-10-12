# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LTX-2.5 is refused instead of being loaded as LTX-2; LTX-2.0 / 2.3 still resolve to ltx-2."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from core.inference import diffusion_content as dc
from core.inference.video_families import (
    assert_supported_ltx_version,
    detect_video_family,
    unsupported_ltx_version,
)

_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "comfy_checkpoint_headers.json").read_text(
        encoding = "utf-8"
    )
)

LTX25_NAMES = [
    "Lightricks/LTX-2.5-Diffusers",
    "Lightricks/LTX-2.5",
    "lightricks/ltx-2.5",
    "ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
    "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
    "ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
    "someone/LTX-2.5-GGUF",
    "ltx2.5",
    "ltx2_5",
    "LTX25",
    "LTX-Video-2.5",
]

LTX2X_NAMES = [
    "Lightricks/LTX-2",
    "Lightricks/LTX-2.3",
    "Lightricks/LTX-2.3-fp8",
    "unsloth/LTX-2-FP8",
    "unsloth/LTX-2.3-FP8",
    "unsloth/LTX-2.3-GGUF",
    "ltx-2-19b-dev.safetensors",
    "ltx-2-19b-distilled-fp8.safetensors",
    "ltx-2.3-22b-distilled.safetensors",
    "ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf",
    "LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf",
    "ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot.safetensors",
    "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors",
    "ltxv",
    "ltx2",
]


@pytest.mark.parametrize("name", LTX25_NAMES)
def test_ltx25_names_are_recognised(name):
    assert unsupported_ltx_version(name) == "2.5"


@pytest.mark.parametrize("name", LTX2X_NAMES)
def test_ltx2_and_ltx23_names_are_untouched(name):
    assert unsupported_ltx_version(name) is None
    assert detect_video_family(name).name == "ltx-2"


@pytest.mark.parametrize("name", ["ltx-24fps-clip", "voltx-2.5", "ltx-23", "Lightricks/LTX-Video"])
def test_lookalikes_are_not_ltx25(name):
    assert unsupported_ltx_version(name) is None


def test_prerelease_name_is_named_as_such():
    assert unsupported_ltx_version("ltx-2.4-test") == "2.4"


def test_refusal_names_the_pick_and_the_alternative():
    with pytest.raises(ValueError) as exc:
        assert_supported_ltx_version("Lightricks/LTX-2.5-Diffusers")
    text = str(exc.value)
    assert "'Lightricks/LTX-2.5-Diffusers' is LTX-2.5" in text
    assert "Gemma 4" in text
    assert "LTX-2.3" in text


def test_picked_file_name_outranks_the_folder():
    # An ltx-2.3 file in a folder someone named after 2.5 still loads...
    assert_supported_ltx_version("/models/ltx-2.5-tests", "ltx-2.3-22b-distilled.safetensors")
    # ... while a 2.5 file is refused whatever the folder is called...
    with pytest.raises(ValueError, match = "ltx-2.5-22b-distilled-transformer-bf16"):
        assert_supported_ltx_version(
            "Lightricks/LTX-2.5", "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
        )
    # ... and a generically named file defers to the repo / display id.
    with pytest.raises(ValueError, match = "LTX-2.5"):
        assert_supported_ltx_version("Lightricks/LTX-2.5", "model.safetensors")
    with pytest.raises(ValueError, match = "LTX-2.5"):
        assert_supported_ltx_version("/tmp/x", "model.safetensors", "Lightricks/LTX-2.5")


def test_detect_load_family_refuses_ltx25_even_with_an_override():
    from core.inference.video import _detect_load_family

    for override in (None, "ltx-2", "ltx-2.3"):
        with pytest.raises(ValueError, match = "LTX-2.5"):
            _detect_load_family("Lightricks/LTX-2.5-Diffusers", None, override)
    with pytest.raises(ValueError, match = "LTX-2.5"):
        _detect_load_family(
            "Lightricks/LTX-2.5", "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors", None
        )
    assert _detect_load_family("Lightricks/LTX-2.3", "ltx-2.3-22b-distilled.safetensors", None).name == "ltx-2"
    assert _detect_load_family("Lightricks/LTX-2", None, None).name == "ltx-2"
    assert _detect_load_family("unsloth/LTX-2.3-GGUF", "distilled/ltx-2.3-22b-distilled-Q4_K_M.gguf", None).name == "ltx-2"


def test_validate_load_request_refuses_before_any_download():
    from core.inference.video import get_video_backend

    backend = get_video_backend()
    with pytest.raises(ValueError, match = "LTX-2.5"):
        backend.validate_load_request("Lightricks/LTX-2.5-Diffusers", model_kind = "pipeline")
    with pytest.raises(ValueError, match = "LTX-2.5"):
        backend.validate_load_request(
            "Lightricks/LTX-2.5",
            gguf_filename = "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
            model_kind = "single_file",
            family_override = "ltx-2",
        )


def test_training_refuses_ltx25_base():
    from core.training.diffusion_train_common import resolve_trainable_family

    for family in (None, "ltx-2"):
        with pytest.raises(ValueError, match = "LTX-2.5"):
            resolve_trainable_family("Lightricks/LTX-2.5-Diffusers", family)



_LTX23_KEYS = [
    "Lightricks/LTX-2.3/ltx-2.3-22b-distilled.safetensors",
    "Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors",
    "Lightricks/LTX-2.3-fp8/ltx-2.3-22b-distilled-fp8.safetensors",
    "Kijai/LTX2.3_comfy/diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot.safetensors",
    "Kijai/LTX2.3_comfy/diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_mxfp8_block32.safetensors",
    "Lightricks/LTX-2/ltx-2-19b-dev-fp4.safetensors",
    "QuantStack/LTX-2.3-GGUF/LTX-2.3-distilled-1.1/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf",
    "diffusers/ltx-2/LTX2VideoTransformer3DModel",
]


def _shapes(key):
    entry = _FIXTURE[key]
    return (entry["shapes"], entry.get("meta")) if "shapes" in entry else (entry, None)


def _as_ltx25(shapes):
    """The 2.3 header with diffusers' 2.5 deltas applied: no video FFN bias, a keyframe position embedding."""
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
def test_ltx25_header_is_recognised_but_unsupported(key):
    shapes, meta = _shapes(key)
    info = dc.classify_tensors(_as_ltx25(shapes), meta)
    assert info.role == dc.ROLE_DIT and info.family is None
    assert "LTX-2.5" in info.what


def test_ltx25_header_without_the_embedding_is_still_caught():
    # Fixture is trimmed past the FFN: re-add block 0's video FFN.
    shapes, meta = _shapes("Lightricks/LTX-2.3/ltx-2.3-22b-distilled.safetensors")
    ff = "model.diffusion_model.transformer_blocks.0.ff.net.0.proj"
    no_bias = {**shapes, f"{ff}.weight": [16384, 4096]}
    assert dc.classify_tensors(no_bias, meta).family is None
    assert dc.classify_tensors({**no_bias, f"{ff}.bias": [16384]}, meta).family == "ltx-2"


def _write_header_only_safetensors(path: Path, shapes: dict) -> None:
    header = {k: {"dtype": "BF16", "shape": v, "data_offsets": [0, 0]} for k, v in shapes.items()}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


def test_renamed_local_ltx25_file_is_refused_by_its_header(tmp_path):
    from core.inference.video import get_video_backend
    from core.inference.video_ltx2 import assert_not_ltx25_checkpoint

    shapes, _ = _shapes("Lightricks/LTX-2.3/ltx-2.3-22b-distilled.safetensors")
    good = tmp_path / "ltx-2.3-renamed.safetensors"
    bad = tmp_path / "ltx-2.3-but-really-2.5.safetensors"
    _write_header_only_safetensors(good, shapes)
    _write_header_only_safetensors(bad, _as_ltx25(shapes))
    assert_not_ltx25_checkpoint(good)
    with pytest.raises(ValueError, match = "LTX-2.5"):
        assert_not_ltx25_checkpoint(bad)
    message = dc.refusal_for(str(bad), "video")
    assert message and "LTX-2.5" in message
    assert dc.resolve_family_with_content("ltx-2", str(bad), "video") == (None, True)
    with pytest.raises(ValueError, match = "LTX-2.5"):
        get_video_backend().validate_load_request(
            str(tmp_path), gguf_filename = bad.name, model_kind = "single_file"
        )


def test_local_ltx25_diffusers_folder_is_refused_by_model_index(tmp_path):
    from core.inference.video import _detect_load_family

    folder = tmp_path / "my-ltx-video"
    folder.mkdir()
    index = {
        "_class_name": "LTX2Pipeline",
        "text_encoder": ["transformers", "Gemma4UnifiedForConditionalGeneration"],
        "transformer": ["diffusers", "LTX2VideoTransformer3DModel"],
    }
    (folder / "model_index.json").write_text(json.dumps(index), encoding = "utf-8")
    with pytest.raises(ValueError, match = "model_index.json"):
        _detect_load_family(str(folder), None, None)
    index["text_encoder"] = ["transformers", "Gemma3ForConditionalGeneration"]
    (folder / "model_index.json").write_text(json.dumps(index), encoding = "utf-8")
    assert _detect_load_family(str(folder), None, None).name == "ltx-2"
    index["duration_head"] = ["diffusers", "LTX2DurationHead"]
    (folder / "model_index.json").write_text(json.dumps(index), encoding = "utf-8")
    with pytest.raises(ValueError, match = "LTX-2.5"):
        _detect_load_family(str(folder), None, None)
    index["duration_head"] = [None, None]
    (folder / "model_index.json").write_text(json.dumps(index), encoding = "utf-8")
    assert _detect_load_family(str(folder), None, None).name == "ltx-2"


@pytest.mark.parametrize("filename", ["ltx-transformer.safetensors", "ltxv_dit.safetensors", "LTX.safetensors"])
def test_unversioned_ltx_file_name_defers_to_the_repo(filename):
    from core.inference.video import get_video_backend

    with pytest.raises(ValueError, match = "LTX-2.5"):
        assert_supported_ltx_version("Lightricks/LTX-2.5", filename)
    with pytest.raises(ValueError, match = "LTX-2.5"):
        get_video_backend().validate_load_request(
            "Lightricks/LTX-2.5", gguf_filename = filename, model_kind = "single_file", family_override = "ltx-2"
        )
    # A version-free name is no evidence either way in a 2.0 / 2.3 repo.
    assert_supported_ltx_version("Lightricks/LTX-2.3", filename)


@pytest.mark.parametrize(
    "filename", ["ltx-2-19b-dev.safetensors", "ltx2_dit.safetensors", "ltx-video-2.3-x.safetensors"]
)
def test_versioned_supported_file_name_still_outranks_the_folder(filename):
    assert_supported_ltx_version("/models/ltx-2.5-tests", filename)


def test_training_refuses_renamed_local_ltx25_pipeline(tmp_path):
    from core.training.diffusion_train_common import resolve_trainable_family

    folder = tmp_path / "my-video-base"
    folder.mkdir()
    index = {
        "_class_name": "LTX2Pipeline",
        "text_encoder": ["transformers", "Gemma4UnifiedForConditionalGeneration"],
        "transformer": ["diffusers", "LTX2VideoTransformer3DModel"],
    }
    (folder / "model_index.json").write_text(json.dumps(index), encoding = "utf-8")
    for family in (None, "ltx-2"):
        with pytest.raises(ValueError, match = "model_index.json"):
            resolve_trainable_family(str(folder), family)
    index["text_encoder"] = ["transformers", "Gemma3ForConditionalGeneration"]
    (folder / "model_index.json").write_text(json.dumps(index), encoding = "utf-8")
    assert resolve_trainable_family(str(folder), "ltx-2") == "ltx-2"


def test_explicit_pipeline_load_ignores_the_unused_filename(tmp_path):
    from core.inference.video import _detect_load_family, get_video_backend

    with pytest.raises(ValueError, match = "LTX-2.5"):
        get_video_backend().validate_load_request(
            "Lightricks/LTX-2.5",
            gguf_filename = "ltx-2.3-22b-dev.safetensors",
            model_kind = "pipeline",
            family_override = "ltx-2",
        )
    folder = tmp_path / "opaque"
    folder.mkdir()
    index = {
        "_class_name": "LTX2Pipeline",
        "text_encoder": ["transformers", "Gemma4UnifiedForConditionalGeneration"],
    }
    (folder / "model_index.json").write_text(json.dumps(index), encoding = "utf-8")
    with pytest.raises(ValueError, match = "model_index.json"):
        _detect_load_family(str(folder), "ltx-2.3.safetensors", "ltx-2", None, "pipeline")
    # A real single-file pick still lets its own ltx-2.3 file decide.
    assert _detect_load_family(str(folder), "ltx-2.3.safetensors", "ltx-2", None, "single_file").name == "ltx-2"


def test_training_name_guard_is_scoped_to_ltx(tmp_path):
    from core.training.diffusion_train_common import resolve_trainable_family

    folder = tmp_path / "sdxl-ltx-2.5-style"
    folder.mkdir()
    (folder / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionXLPipeline"}), encoding = "utf-8"
    )
    assert resolve_trainable_family(str(folder), "sdxl") == "sdxl"
    with pytest.raises(ValueError, match = "LTX-2.5"):
        resolve_trainable_family(str(folder), "ltx-2")


@pytest.mark.parametrize("name", ["ltxv-2.5.safetensors", "ltxv25", "LTXV_2_5-dev.safetensors", "ltxv2.4"])
def test_versioned_ltxv_alias_is_refused(name):
    assert unsupported_ltx_version(name) is not None
    with pytest.raises(ValueError, match = "LTX-2"):
        assert_supported_ltx_version("/models/x", name)


@pytest.mark.parametrize("filename", ["ltx-24fps-clip.safetensors", "ltx-23.safetensors", "ltx-2.10-x.safetensors", "ltx2fast.safetensors"])
def test_numeric_lookalike_file_name_defers_to_the_repo(filename):
    with pytest.raises(ValueError, match = "LTX-2.5"):
        assert_supported_ltx_version("Lightricks/LTX-2.5", filename)
    assert_supported_ltx_version("Lightricks/LTX-2.3", filename)


def test_inference_name_guard_is_scoped_to_ltx():
    from core.inference.video import _detect_load_family

    fam = _detect_load_family(
        "someone/wan-ltx-2.5-style", "wan-ltx-2.5-style.safetensors", "wan2.2-ti2v-5b", None, "single_file"
    )
    assert fam is not None and fam.name == "wan2.2-ti2v-5b"
    with pytest.raises(ValueError, match = "LTX-2.5"):
        _detect_load_family("someone/wan-ltx-2.5-style", "wan-ltx-2.5-style.safetensors", "ltx-2", None, "single_file")


def test_ltx25_companion_base_is_refused(tmp_path, monkeypatch):
    import core.inference.diffusion as diffusion
    from core.inference.video import get_video_backend

    # Isolate the LTX-2.5 check from the pipeline-completeness check on the stub folder.
    monkeypatch.setattr(diffusion, "_assert_local_base_is_pipeline", lambda *a, **k: None)

    backend = get_video_backend()
    pick = dict(gguf_filename = "distilled/ltx-2.3-22b-distilled-Q4_K_M.gguf", model_kind = "gguf")
    with pytest.raises(ValueError, match = "LTX-2.5"):
        backend.validate_load_request("unsloth/LTX-2.3-GGUF", base_repo = "unsloth/LTX-2.5-Diffusers", **pick)
    folder = tmp_path / "my-base"
    folder.mkdir()
    index = {
        "_class_name": "LTX2Pipeline",
        "text_encoder": ["transformers", "Gemma4UnifiedForConditionalGeneration"],
    }
    (folder / "model_index.json").write_text(json.dumps(index), encoding = "utf-8")
    with pytest.raises(ValueError, match = r"by its model_index\.json\) is LTX-2\.5"):
        backend.validate_load_request("unsloth/LTX-2.3-GGUF", base_repo = str(folder), **pick)
