# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression for #11343: PQ2_0 labels and non-conventional GGUF filenames."""

from __future__ import annotations

import pytest

from hub.utils.gguf import extract_quant_token, gguf_variant_key
from core.inference import llama_cpp as llama_cpp_module

_BONSAI_FILE = "Bonsai-2-27B-PQ2_0-CRACK.gguf"
_BONSAI_REPO = "dealignai/Bonsai-2-27B-Ternary-CRACK-GGUF"


def test_pq2_0_is_not_advertised_as_q2_0():
    assert extract_quant_token(_BONSAI_FILE) == "PQ2_0"
    assert gguf_variant_key(_BONSAI_FILE) == "PQ2_0"


def test_packed_and_grouped_q2_labels_stay_distinct():
    assert extract_quant_token("Ternary-Bonsai-1.7B-PQ2_0.gguf") == "PQ2_0"
    assert extract_quant_token("Ternary-Bonsai-1.7B-Q2_0.gguf") == "Q2_0"
    assert extract_quant_token("Ternary-Bonsai-1.7B-Q2_0_g64.gguf") == "Q2_0_g64"
    keys = {
        gguf_variant_key("Ternary-Bonsai-1.7B-PQ2_0.gguf").casefold(),
        gguf_variant_key("Ternary-Bonsai-1.7B-Q2_0.gguf").casefold(),
        gguf_variant_key("Ternary-Bonsai-1.7B-Q2_0_g64.gguf").casefold(),
    }
    assert keys == {"pq2_0", "q2_0", "q2_0_g64"}


def test_resolve_pq2_0_from_live_listing(monkeypatch):
    monkeypatch.setattr(
        "huggingface_hub.list_repo_files",
        lambda repo_id, token = None: [_BONSAI_FILE],
    )

    filename, shards = llama_cpp_module._resolve_variant_gguf_files(
        _BONSAI_REPO,
        "PQ2_0",
    )
    assert filename == _BONSAI_FILE
    assert shards == []


def test_a_selection_saved_as_q2_0_still_resolves_the_packed_file(monkeypatch):
    # Before PQ2_0 had its own label the menu offered this file as Q2_0.
    monkeypatch.setattr(
        "huggingface_hub.list_repo_files",
        lambda repo_id, token = None: [_BONSAI_FILE],
    )

    filename, shards = llama_cpp_module._resolve_variant_gguf_files(_BONSAI_REPO, "Q2_0")
    assert filename == _BONSAI_FILE
    assert shards == []


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("Q2_0", "Ternary-Bonsai-1.7B-Q2_0.gguf"),
        ("PQ2_0", "Ternary-Bonsai-1.7B-PQ2_0.gguf"),
        ("Q2_0_g64", "Ternary-Bonsai-1.7B-Q2_0_g64.gguf"),
    ],
)
def test_packed_plain_and_grouped_q2_files_each_resolve_to_their_own(
    monkeypatch, variant, expected
):
    files = [
        "Ternary-Bonsai-1.7B-PQ2_0.gguf",
        "Ternary-Bonsai-1.7B-Q2_0.gguf",
        "Ternary-Bonsai-1.7B-Q2_0_g64.gguf",
    ]
    monkeypatch.setattr("huggingface_hub.list_repo_files", lambda repo_id, token = None: files)

    filename, _ = llama_cpp_module._resolve_variant_gguf_files(
        "prism-ml/Ternary-Bonsai-1.7B-gguf", variant
    )
    assert filename == expected


def test_an_ordinary_q2_0_repo_is_unchanged(monkeypatch):
    files = ["Tiny-Q2_0.gguf", "Tiny-Q4_0.gguf"]
    monkeypatch.setattr("huggingface_hub.list_repo_files", lambda repo_id, token = None: files)

    assert extract_quant_token("Tiny-Q2_0.gguf") == "Q2_0"
    filename, _ = llama_cpp_module._resolve_variant_gguf_files("org/Tiny-GGUF", "Q2_0")
    assert filename == "Tiny-Q2_0.gguf"
