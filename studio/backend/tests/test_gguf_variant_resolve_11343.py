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


def _write_gguf(path):
    import struct
    path.write_bytes(b"GGUF" + struct.pack("<I", 3) + b"\0" * 64)
    return path


def test_a_local_selection_saved_as_q2_0_still_resolves_the_packed_file(tmp_path):
    from utils.models.model_config import _find_local_gguf_by_variant

    packed = _write_gguf(tmp_path / _BONSAI_FILE)
    assert _find_local_gguf_by_variant(str(tmp_path), "Q2_0") == str(packed)
    assert _find_local_gguf_by_variant(str(tmp_path), "PQ2_0") == str(packed)


def test_a_local_q2_0_file_keeps_its_own_label_beside_a_packed_one(tmp_path):
    from utils.models.model_config import _find_local_gguf_by_variant

    _write_gguf(tmp_path / "Ternary-Bonsai-1.7B-PQ2_0.gguf")
    plain = _write_gguf(tmp_path / "Ternary-Bonsai-1.7B-Q2_0.gguf")
    assert _find_local_gguf_by_variant(str(tmp_path), "Q2_0") == str(plain)


def test_a_models_pin_on_q2_0_aliases_the_packed_quant():
    from types import SimpleNamespace

    from core.inference.local_model_resolver import _legacy_variant_aliases

    packed = SimpleNamespace(quant = "PQ2_0", filename = _BONSAI_FILE)
    assert ("q2_0", "PQ2_0") in _legacy_variant_aliases([packed])
    plain = SimpleNamespace(quant = "Q2_0", filename = "Ternary-Bonsai-1.7B-Q2_0.gguf")
    assert all(legacy != "q2_0" for legacy, _ in _legacy_variant_aliases([packed, plain]))
