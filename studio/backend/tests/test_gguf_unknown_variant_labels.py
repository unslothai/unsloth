# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unrecognised GGUF builds in one folder are labelled by the part of the name that differs."""

from __future__ import annotations

from hub.utils.gguf import GgufVariantInfo, _apply_gguf_display_labels


def _labels(*filenames: str, name_tails: bool = True) -> list[str | None]:
    variants = [GgufVariantInfo(filename = f, quant = f, size_bytes = 1) for f in filenames]
    _apply_gguf_display_labels(variants, name_tails = name_tails)
    return [v.display_label for v in variants]


def test_shared_name_is_dropped_from_each_label():
    assert _labels(
        "Qwen3.8-27B-NVFP4-MTP-HIGH.gguf",
        "Qwen3.8-27B-NVFP4-MTP-LOW.gguf",
        "Qwen3.8-27B-NVFP4-MTP-MID.gguf",
    ) == ["GGUF · HIGH", "GGUF · LOW", "GGUF · MID"]


def test_prefix_cut_stays_on_a_word_boundary():
    assert _labels("Model-v2.gguf", "Model-v3.gguf") == ["GGUF · v2", "GGUF · v3"]
    assert _labels("Model.gguf", "Model-v2.gguf") == [
        "GGUF · Model.gguf",
        "GGUF · Model-v2.gguf",
    ]


def test_unrelated_names_keep_the_whole_filename():
    assert _labels("foo.gguf", "bar.gguf") == ["GGUF · foo.gguf", "GGUF · bar.gguf"]


def test_a_single_unknown_build_is_plain_gguf():
    assert _labels("Model-HIGH.gguf") == ["GGUF"]


def test_hub_repos_keep_whole_filenames():
    assert _labels("Model-HIGH.gguf", "Model-LOW.gguf", name_tails = False) == [
        "GGUF · Model-HIGH.gguf",
        "GGUF · Model-LOW.gguf",
    ]
