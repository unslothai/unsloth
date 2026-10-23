# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Wan2.2-A14B single-file picks: either expert of a high/low noise pair names both, the high one
loads as ``transformer``, and validation, the download plan and the base pull cover the pair."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from core.inference.video import VideoBackend, _checkpoint_files
from core.inference.video_families import detect_video_family
from core.inference.video_moe_pair import (
    moe_expert_of,
    moe_expert_pair,
    moe_partner_filename,
    moe_pick_pairs,
)

COMFY_HIGH = "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
COMFY_LOW = "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
GGUF_HIGH = "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf"
GGUF_LOW = "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf"


@pytest.mark.parametrize(
    "name, partner",
    [
        (COMFY_HIGH, COMFY_LOW),
        (COMFY_LOW, COMFY_HIGH),
        # Folder and file tokens both swap, case kept.
        (GGUF_HIGH, GGUF_LOW),
        (GGUF_LOW, GGUF_HIGH),
        ("WAN_HIGH_NOISE_Q8.gguf", "WAN_LOW_NOISE_Q8.gguf"),
        ("wan2.2-t2v-a14b-high-noise.safetensors", "wan2.2-t2v-a14b-low-noise.safetensors"),
        ("Wan2.2-T2V-A14B-Q4_K_M.gguf", None),
        # Both tokens: ambiguous, never paired.
        ("high_noise_to_low_noise.safetensors", None),
        ("highlight_noisy.safetensors", None),
        ("", None),
        (None, None),
    ],
)
def test_partner_swaps_every_noise_token(name, partner):
    assert moe_partner_filename(name) == partner
    if partner is not None:
        assert moe_partner_filename(partner) == name


def test_pair_orders_high_first_whichever_was_picked():
    assert moe_expert_pair(COMFY_LOW) == (COMFY_HIGH, COMFY_LOW)
    assert moe_expert_pair(GGUF_HIGH) == (GGUF_HIGH, GGUF_LOW)
    assert moe_expert_of(GGUF_LOW) == "low"
    assert moe_expert_pair("model.gguf") is None


def test_comfy_expert_names_resolve_the_a14b_family_and_pair():
    fam = detect_video_family(f"Comfy-Org/Wan_2.2_ComfyUI_Repackaged/{COMFY_HIGH}")
    assert fam is not None and fam.name == "wan2.2-t2v-a14b"
    assert moe_pick_pairs(fam, COMFY_HIGH)
    assert not moe_pick_pairs(fam, "Wan2.2-T2V-A14B-Q4_K_M.gguf")
    # The I2V experts are a different model (36-channel input) and must not resolve to T2V.
    i2v = detect_video_family("wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors")
    assert i2v is None or i2v.name != "wan2.2-t2v-a14b"
    # A single-DiT family never needs a pair.
    assert moe_pick_pairs(detect_video_family("Wan-AI/Wan2.2-TI2V-5B-Diffusers"), "x.gguf")


def test_checkpoint_files_name_both_experts():
    assert _checkpoint_files(GGUF_HIGH, "gguf") == (GGUF_HIGH, GGUF_LOW)
    assert _checkpoint_files("Wan2.2-TI2V-5B-Q4_K_M.gguf", "gguf") == (
        "Wan2.2-TI2V-5B-Q4_K_M.gguf",
    )
    assert _checkpoint_files(None, "gguf") == ()


def _sibling(name: str, size: int):
    return types.SimpleNamespace(rfilename = name, size = size)


def test_a_paired_pick_drops_both_dense_experts_from_the_base_pull():
    info = types.SimpleNamespace(
        siblings = [
            _sibling("model_index.json", 1),
            _sibling("transformer/config.json", 1),
            _sibling("transformer/diffusion_pytorch_model-00001-of-00002.safetensors", 50),
            _sibling("transformer_2/config.json", 1),
            _sibling("transformer_2/diffusion_pytorch_model-00001-of-00002.safetensors", 50),
            _sibling("text_encoder/model-00001-of-00002.safetensors", 5),
            _sibling("vae/diffusion_pytorch_model.safetensors", 1),
        ]
    )
    for kind in ("gguf", "single_file"):
        names = {n for n, _ in VideoBackend._base_download_files(info, kind)}
        # Both configs stay (from_single_file reads config = <base>, subfolder = transformer / transformer_2).
        assert {"transformer/config.json", "transformer_2/config.json"} <= names
        assert not any(n.endswith(".safetensors") and n.startswith("transformer") for n in names)
        assert "text_encoder/model-00001-of-00002.safetensors" in names
    pipeline = {n for n, _ in VideoBackend._base_download_files(info, "pipeline")}
    assert "transformer_2/diffusion_pytorch_model-00001-of-00002.safetensors" in pipeline


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(b"\0")
    return path


def test_partner_path_resolves_in_a_local_folder_and_next_to_a_local_file(tmp_path):
    high = _touch(tmp_path / "HighNoise" / "Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf")
    low = _touch(tmp_path / "LowNoise" / "Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf")
    assert VideoBackend._resolve_moe_partner_path(str(tmp_path), GGUF_HIGH, None) == low.resolve()
    assert VideoBackend._resolve_moe_partner_path(str(tmp_path), GGUF_LOW, None) == high.resolve()
    flat_high = _touch(tmp_path / "flat" / "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors")
    with pytest.raises(FileNotFoundError):
        VideoBackend._resolve_moe_partner_path(str(flat_high), None, None)
    flat_low = _touch(flat_high.with_name("wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"))
    assert VideoBackend._resolve_moe_partner_path(str(flat_high), None, None) == flat_low
    with pytest.raises(ValueError, match = "high_noise / low_noise"):
        VideoBackend._resolve_moe_partner_path(str(tmp_path), "model.gguf", None)


def test_validation_needs_the_partner_on_disk_for_a_local_pick(tmp_path, monkeypatch):
    import core.inference.video as video_module

    monkeypatch.setattr(video_module, "_ensure_mp4_encoder_available", lambda: None)
    monkeypatch.setattr(
        "core.inference.diffusion._assert_local_base_is_pipeline", lambda *a, **k: None
    )
    backend = VideoBackend()
    high = _touch(tmp_path / "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors")
    with pytest.raises(ValueError, match = "low_noise"):
        backend.validate_load_request(
            str(tmp_path), gguf_filename = high.name, family_override = "wan2.2-t2v-a14b"
        )
    with pytest.raises(ValueError, match = "partner"):
        backend.validate_load_request(
            str(high), gguf_filename = high.name, family_override = "wan2.2-t2v-a14b"
        )
    _touch(tmp_path / "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors")
    for repo in (str(tmp_path), str(high)):
        fam = backend.validate_load_request(
            repo, gguf_filename = high.name, family_override = "wan2.2-t2v-a14b"
        )
        assert fam.name == "wan2.2-t2v-a14b"
    # An unpaired name is still refused, with the way out.
    lone = _touch(tmp_path / "solo" / "wan2.2_t2v_14B_fp8_scaled.safetensors")
    with pytest.raises(ValueError, match = "dual-expert"):
        backend.validate_load_request(
            str(lone), gguf_filename = lone.name, family_override = "wan2.2-t2v-a14b"
        )
