# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only unit tests for the diffusion LoRA trainer's pure helpers.

The training loop needs a GPU + weights, but dataset discovery, config normalisation,
the SDXL add-time-ids, and the dict->config adapter are pure and tested here.
"""

from __future__ import annotations

import json

import pytest

from core.training.diffusion_lora_trainer import (
    DEFAULT_LORA_TARGETS,
    DiffusionLoraConfig,
    _coerce_gradient_checkpointing,
    _config_from_dict,
    compute_sdxl_add_time_ids,
    discover_image_caption_pairs,
)


def _touch(p):
    p.write_bytes(b"")


def test_discover_prefers_metadata_then_sidecar_then_instance(tmp_path):
    _touch(tmp_path / "a.png")
    _touch(tmp_path / "b.jpg")
    _touch(tmp_path / "c.webp")
    # a.png captioned via metadata.jsonl
    (tmp_path / "metadata.jsonl").write_text(
        json.dumps({"file_name": "a.png", "text": "from metadata"}) + "\n", encoding = "utf-8"
    )
    # b.jpg captioned via sidecar
    (tmp_path / "b.txt").write_text("from sidecar", encoding = "utf-8")
    # c.webp falls back to the instance prompt
    pairs = dict(discover_image_caption_pairs(tmp_path, instance_prompt = "from instance"))
    assert pairs[str(tmp_path / "a.png")] == "from metadata"
    assert pairs[str(tmp_path / "b.jpg")] == "from sidecar"
    assert pairs[str(tmp_path / "c.webp")] == "from instance"


def test_discover_skips_uncaptioned_without_instance_prompt(tmp_path):
    _touch(tmp_path / "cap.png")
    _touch(tmp_path / "nocap.png")
    (tmp_path / "cap.caption").write_text("a caption", encoding = "utf-8")
    pairs = discover_image_caption_pairs(tmp_path)
    assert pairs == [(str(tmp_path / "cap.png"), "a caption")]


def test_discover_captions_jsonl_and_image_key(tmp_path):
    _touch(tmp_path / "x.png")
    (tmp_path / "captions.jsonl").write_text(
        json.dumps({"image": "x.png", "text": "hi"}) + "\n", encoding = "utf-8"
    )
    assert discover_image_caption_pairs(tmp_path) == [(str(tmp_path / "x.png"), "hi")]


def test_discover_custom_caption_column(tmp_path):
    _touch(tmp_path / "x.png")
    (tmp_path / "metadata.jsonl").write_text(
        json.dumps({"file_name": "x.png", "caption": "col"}) + "\n", encoding = "utf-8"
    )
    assert discover_image_caption_pairs(tmp_path, caption_column = "caption")[0][1] == "col"


def test_discover_empty_raises(tmp_path):
    _touch(tmp_path / "x.png")  # no captions anywhere, no instance prompt
    with pytest.raises(ValueError, match = "No captioned images"):
        discover_image_caption_pairs(tmp_path)


def test_discover_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_image_caption_pairs(tmp_path / "nope")


def test_config_normalized_defaults():
    cfg = DiffusionLoraConfig(base_model = "b", data_dir = "d", output_dir = "o").normalized()
    assert cfg.lora_alpha == cfg.lora_rank  # alpha defaults to rank
    assert cfg.lora_target_modules == DEFAULT_LORA_TARGETS


@pytest.mark.parametrize(
    "kw",
    [
        {"train_steps": 0},
        {"train_batch_size": 0},
        {"gradient_accumulation_steps": 0},
        {"lora_rank": 0},
        {"resolution": 100},  # not a multiple of 8
        {"resolution": 32},  # too small
        {"mixed_precision": "int4"},
    ],
)
def test_config_normalized_validation(kw):
    with pytest.raises(ValueError):
        DiffusionLoraConfig(base_model = "b", data_dir = "d", output_dir = "o", **kw).normalized()


def test_compute_sdxl_add_time_ids():
    assert compute_sdxl_add_time_ids(1024) == (1024, 1024, 0, 0, 1024, 1024)


def test_config_from_dict_ignores_unknown_and_tuples_targets():
    cfg = _config_from_dict(
        {
            "base_model": "b",
            "data_dir": "d",
            "output_dir": "o",
            "lora_target_modules": ["to_q", "to_v"],
            "unknown_field": 123,  # must be ignored, not crash
        }
    )
    assert cfg.lora_target_modules == ("to_q", "to_v")
    assert not hasattr(cfg, "unknown_field")


def test_config_rejects_zero_lora_alpha():
    # An explicit zero alpha would scale the adapter to nothing; reject it.
    with pytest.raises(ValueError, match = "lora_alpha"):
        DiffusionLoraConfig(base_model = "b", data_dir = "d", output_dir = "o", lora_alpha = 0).normalized()


def test_config_coerces_string_learning_rate():
    # The Studio config path preserves learning_rate as a string; normalize to float.
    cfg = DiffusionLoraConfig(
        base_model = "b", data_dir = "d", output_dir = "o", learning_rate = "1e-4"
    ).normalized()
    assert cfg.learning_rate == 1e-4
    with pytest.raises(ValueError, match = "learning_rate"):
        DiffusionLoraConfig(
            base_model = "b", data_dir = "d", output_dir = "o", learning_rate = "abc"
        ).normalized()


def test_config_blank_hf_token_is_anonymous():
    cfg = DiffusionLoraConfig(
        base_model = "b", data_dir = "d", output_dir = "o", hf_token = "   "
    ).normalized()
    assert cfg.hf_token is None


def test_config_from_dict_aliases_generic_studio_keys():
    # The generic Studio training payload uses different key names; alias them.
    cfg = _config_from_dict(
        {
            "model_name": "b",
            "data_dir": "d",
            "output_dir": "o",
            "max_steps": 25,
            "batch_size": 3,
            "lora_r": 8,
            "lr_scheduler_type": "cosine",
            "random_seed": 7,
        }
    )
    assert cfg.base_model == "b"
    assert cfg.train_steps == 25
    assert cfg.train_batch_size == 3
    assert cfg.lora_rank == 8
    assert cfg.lr_scheduler == "cosine"
    assert cfg.seed == 7


def test_config_from_dict_canonical_key_beats_alias():
    cfg = _config_from_dict(
        {"base_model": "canon", "model_name": "alias", "data_dir": "d", "output_dir": "o"}
    )
    assert cfg.base_model == "canon"


def test_gradient_checkpointing_string_coercion():
    # Studio sends a string; the disable words are False, everything else truthy True.
    for off in ("none", "None", "false", "0", "no", "off", ""):
        assert _coerce_gradient_checkpointing(off) is False
    for on in ("true", "unsloth", "yes"):
        assert _coerce_gradient_checkpointing(on) is True
    assert _coerce_gradient_checkpointing(True) is True
    assert _coerce_gradient_checkpointing(False) is False
    cfg = _config_from_dict(
        {"base_model": "b", "data_dir": "d", "output_dir": "o", "gradient_checkpointing": "none"}
    )
    assert cfg.gradient_checkpointing is False


def test_config_rejects_nonpositive_learning_rate():
    with pytest.raises(ValueError, match = "learning_rate"):
        DiffusionLoraConfig(
            base_model = "b", data_dir = "d", output_dir = "o", learning_rate = 0
        ).normalized()


def test_config_rejects_known_non_sdxl_base_models():
    # Known DiT families and GGUF checkpoints must fail at normalise time (an instant
    # 400 via the API) instead of minutes later inside from_pretrained.
    for bad in (
        "unsloth/FLUX.1-dev-GGUF",
        "black-forest-labs/FLUX.1-schnell",
        "unsloth/Qwen-Image-2512-unsloth-bnb-4bit",
        "Tongyi-MAI/Z-Image-Turbo",
        "stabilityai/stable-diffusion-3-medium",
        "unsloth/FLUX.1-Kontext-dev",
        "z-image-turbo-Q4_K_M.gguf",
    ):
        with pytest.raises(ValueError, match = "SDXL"):
            DiffusionLoraConfig(base_model = bad, data_dir = "d", output_dir = "o").normalized()


def test_config_accepts_sdxl_and_unknown_base_models():
    # SDXL names and unclassifiable custom names/paths must pass the guard (a wrong
    # custom pick still fails cleanly in from_pretrained).
    for ok in (
        "stabilityai/stable-diffusion-xl-base-1.0",
        "stabilityai/sdxl-turbo",
        "/data/checkpoints/my-custom-sdxl",
        "my-finetune",
    ):
        cfg = DiffusionLoraConfig(base_model = ok, data_dir = "d", output_dir = "o").normalized()
        assert cfg.base_model == ok
