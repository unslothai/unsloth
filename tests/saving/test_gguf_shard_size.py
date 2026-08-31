# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

from __future__ import annotations

import pytest

import unsloth.save as save_mod


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, "50GB"),
        ("", "0"),
        ("  ", "0"),
        ("0", "0"),
        ("none", "0"),
        ("NONE", "0"),
        ("1MB", "1MB"),
        ("256m", "256MB"),
        ("512 MB", "512MB"),
        ("4G", "4GB"),
        (" 8gb ", "8GB"),
    ],
)
def test_resolve_gguf_shard_size(value, expected):
    assert save_mod._resolve_gguf_shard_size(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "0MB",
        "0GB",
        "-1GB",
        "1.5GB",
        "512",
        "64KB",
        "GB",
        "2TB",
        "2GBx",
    ],
)
def test_invalid_gguf_shard_sizes_fail_before_conversion(value):
    with pytest.raises(ValueError, match = "gguf_shard_size"):
        save_mod._resolve_gguf_shard_size(value)


@pytest.mark.parametrize("value", [0, False, 512, object()])
def test_non_string_gguf_shard_sizes_are_rejected(value):
    with pytest.raises(TypeError, match = "string or None"):
        save_mod._resolve_gguf_shard_size(value)


def test_oversized_gguf_shard_size_is_rejected(monkeypatch):
    monkeypatch.setattr(save_mod.sys, "maxsize", 2_147_483_647)
    with pytest.raises(ValueError, match = "too large"):
        save_mod._resolve_gguf_shard_size("3GB")


def test_public_save_rejects_invalid_size_before_model_work():
    with pytest.raises(ValueError, match = "gguf_shard_size"):
        save_mod.unsloth_save_pretrained_gguf(
            object(),
            "unused",
            tokenizer = object(),
            gguf_shard_size = "1.5GB",
        )


@pytest.mark.parametrize(
    "first_conversion, methods, is_vlm, expected",
    [
        ("f16", ["f16"], False, "256MB"),
        ("bf16", ["bf16", "q4_k_m"], False, "256MB"),
        ("f32", ["f32"], False, "256MB"),
        ("q8_0", ["q8_0"], False, "0"),
        ("bf16", ["q4_k_m"], False, "0"),
        ("f16", ["f16"], True, "0"),
    ],
)
def test_converter_only_shards_final_full_precision_outputs(
    first_conversion,
    methods,
    is_vlm,
    expected,
):
    assert (
        save_mod._converter_gguf_shard_size(
            "256MB",
            first_conversion,
            methods,
            is_vlm,
        )
        == expected
    )
