# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""CPU-only tests for gguf_shard_size resolution.

The value ends up as llama.cpp's ``--split-max-size``, which takes a whole
number plus a K/M/G unit. unsloth-zoo's check parses with ``float``, so a
decimal gets through and only fails in the converter subprocess -- after the
16-bit merge. ``_resolve_gguf_shard_size`` is the fail-fast gate, so its
accept/reject/normalize behaviour is what these tests pin down.
"""

from __future__ import annotations

import pytest

import unsloth.save as save_mod


NO_SHARDING = save_mod._GGUF_NO_SHARDING


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, "50GB"),  # unset keeps the historical default
        ("", NO_SHARDING),
        ("0", NO_SHARDING),
        ("none", NO_SHARDING),
        ("NONE", NO_SHARDING),
        ("0MB", NO_SHARDING),  # a zero magnitude is still "one file"
    ],
)
def test_sentinels_disable_sharding(value, expected):
    assert save_mod._resolve_gguf_shard_size(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("256MB", "256MB"),
        ("128mb", "128MB"),
        ("  2GB  ", "2GB"),
        ("512 MB", "512MB"),
        ("4G", "4GB"),  # the trailing B is optional
        ("64KB", "64KB"),
    ],
)
def test_valid_sizes_normalize(value, expected):
    assert save_mod._resolve_gguf_shard_size(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "1.5GB",  # zoo's float check passes it; llama.cpp's int() parse does not
        "512",  # a bare number would be read as bytes -> thousands of shards
        "GB",
        "abc",
        "2GBx",
        "-4GB",
    ],
)
def test_invalid_sizes_raise_before_conversion(value):
    with pytest.raises(ValueError, match = "not a valid shard size"):
        save_mod._resolve_gguf_shard_size(value)
