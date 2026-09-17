# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import types

from hub.utils.snapshot_filters import (
    resolve_snapshot_ignore_patterns_for_files,
    snapshot_download_siblings,
    snapshot_download_size,
)


def _siblings(sizes: dict[str, int]) -> list:
    return [types.SimpleNamespace(rfilename = name, size = size) for name, size in sizes.items()]


def _kept(sizes: dict[str, int]) -> set[str]:
    return {s.rfilename for s in snapshot_download_siblings(_siblings(sizes))}


GPT_OSS = {
    "config.json": 2,
    "tokenizer.json": 27,
    "model.safetensors.index.json": 36,
    "model-00000-of-00002.safetensors": 4_792,
    "model-00001-of-00002.safetensors": 4_798,
    "model-00002-of-00002.safetensors": 4_170,
    "original/config.json": 1,
    "original/dtypes.json": 13,
    "original/model.safetensors": 13_761,
    "metal/model.bin": 13_750,
}

# bigscience/bloom shards its root safetensors with an underscore, and ships the same
# 72 shards again as pytorch_model_000NN-of-00072.bin.
BLOOM = {
    "config.json": 2,
    "tokenizer.json": 14,
    "model_00001-of-00072.safetensors": 4_900,
    "model_00002-of-00072.safetensors": 4_900,
    "model.safetensors.index.json": 1,
    "pytorch_model_00001-of-00072.bin": 4_900,
    "pytorch_model_00002-of-00072.bin": 4_900,
    "pytorch_model.bin.index.json": 1,
}

WHISPER = {
    "config.json": 2,
    "tokenizer.json": 2,
    "training_args.bin": 3,
    "model.safetensors": 967,
    "pytorch_model.bin": 967,
    "tf_model.h5": 968,
    "flax_model.msgpack": 967,
}


def test_gpt_oss_downloads_only_the_root_safetensors():
    assert snapshot_download_size(_siblings(GPT_OSS)) == 2 + 27 + 36 + 4_792 + 4_798 + 4_170


def test_whisper_downloads_only_the_safetensors_copy():
    assert snapshot_download_size(_siblings(WHISPER)) == 2 + 2 + 3 + 967
    assert "training_args.bin" in _kept(WHISPER)


def test_sharded_bin_checkpoint_beside_safetensors_is_skipped():
    kept = _kept(
        {
            "model.safetensors": 100,
            "pytorch_model-00001-of-00002.bin": 60,
            "pytorch_model-00002-of-00002.bin": 60,
            "pytorch_model.bin.index.json": 1,
            "rust_model.ot": 100,
            "coreml/model.mlpackage/weights.bin": 100,
            "tokenizer.bin": 5,
            "adapter_model.bin": 7,
            "2_Dense/pytorch_model.bin": 9,
        }
    )
    assert kept == {
        "model.safetensors",
        "tokenizer.bin",
        "adapter_model.bin",
        "2_Dense/pytorch_model.bin",
    }


def test_sibling_formats_are_kept_without_root_safetensors():
    bin_only = {
        "config.json": 2,
        "pytorch_model.bin": 967,
        "tf_model.h5": 968,
        "flax_model.msgpack": 967,
        "original/consolidated.00.pth": 900,
        "metal/model.bin": 900,
    }
    assert _kept(bin_only) == set(bin_only)

    nested_only = {
        "unet/diffusion_pytorch_model.safetensors": 100,
        "adapter_model.safetensors": 10,
        "pytorch_model.bin": 50,
        "original/model.safetensors": 50,
    }
    assert _kept(nested_only) == set(nested_only)
    assert "pytorch_model*.bin" not in resolve_snapshot_ignore_patterns_for_files(nested_only)


def test_underscore_sharded_safetensors_still_skip_the_bin_copy():
    assert snapshot_download_size(_siblings(BLOOM)) == 2 + 14 + 4_900 + 4_900 + 1
    assert _kept(BLOOM) == {
        "config.json",
        "tokenizer.json",
        "model_00001-of-00072.safetensors",
        "model_00002-of-00072.safetensors",
        "model.safetensors.index.json",
    }
