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


def test_wrong_variant_does_not_synthesize_when_listing_succeeds(monkeypatch):
    monkeypatch.setattr(
        "huggingface_hub.list_repo_files",
        lambda repo_id, token = None: [_BONSAI_FILE],
    )
    monkeypatch.setattr(
        llama_cpp_module,
        "_cached_variant_resolution",
        lambda *_a, **_k: (None, []),
    )

    filename, shards = llama_cpp_module._resolve_variant_gguf_files(
        _BONSAI_REPO,
        "Q2_0",
    )
    assert filename is None
    assert shards == []


def test_requirements_cache_resolves_when_listing_fails(monkeypatch):
    import sys
    import types

    def _fail_list(*_a, **_k):
        raise OSError("hub down")

    monkeypatch.setattr("huggingface_hub.list_repo_files", _fail_list)
    monkeypatch.setattr(
        llama_cpp_module,
        "_cached_variant_resolution",
        lambda *_a, **_k: (None, []),
    )

    class _Req:
        main_filenames = frozenset({_BONSAI_FILE})
        target_filenames = (_BONSAI_FILE,)

    fake = types.ModuleType("hub.services.models.gguf_variants")
    fake.gguf_variant_requirements = lambda *_a, **_k: _Req()
    monkeypatch.setitem(sys.modules, "hub.services.models.gguf_variants", fake)

    filename, shards = llama_cpp_module._resolve_variant_gguf_files(
        _BONSAI_REPO,
        "PQ2_0",
    )
    assert filename == _BONSAI_FILE
    assert shards == []


def test_does_not_synthesize_when_listing_and_cache_miss(monkeypatch):
    import sys
    import types

    def _fail_list(*_a, **_k):
        raise OSError("hub down")

    monkeypatch.setattr("huggingface_hub.list_repo_files", _fail_list)
    monkeypatch.setattr(
        llama_cpp_module,
        "_cached_variant_resolution",
        lambda *_a, **_k: (None, []),
    )

    fake = types.ModuleType("hub.services.models.gguf_variants")
    fake.gguf_variant_requirements = lambda *_a, **_k: None
    monkeypatch.setitem(sys.modules, "hub.services.models.gguf_variants", fake)
    monkeypatch.setattr(
        "hub.utils.gguf.list_gguf_variants_from_hf_cache",
        lambda *_a, **_k: None,
    )

    filename, shards = llama_cpp_module._resolve_variant_gguf_files(
        _BONSAI_REPO,
        "PQ2_0",
    )
    assert filename is None
    assert shards == []
