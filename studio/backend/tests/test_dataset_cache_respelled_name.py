# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The cached-preview gate must see the folder datasets opens: it snake-cases the dataset name,
so acme/SecretData reads acme___secret_data. The AI assist card fetch goes through the same gate."""

from __future__ import annotations

import json

from hub.utils import dataset_cache, llm_assist


def _processed_cache(root, folder):
    leaf = root / folder / "default" / "0.0.0" / "abc123"
    leaf.mkdir(parents = True)
    (leaf / "dataset_info.json").write_text(json.dumps({"description": ""}))
    (leaf / "data-00000-of-00001.arrow").write_bytes(b"\x01")


def test_a_respelled_name_still_counts_as_cached(monkeypatch, tmp_path):
    _processed_cache(tmp_path, "acme___secret_data")
    monkeypatch.setattr(dataset_cache, "hf_datasets_cache_roots", lambda: [tmp_path])
    assert dataset_cache.dataset_cache_can_answer("acme/secret_data")
    assert dataset_cache.dataset_cache_can_answer("acme/SecretData")
    assert not dataset_cache.dataset_cache_can_answer("acme/other_data")


def test_ai_assist_skips_a_cached_card_the_caller_cannot_reach(monkeypatch):
    import hub.utils.hf_tokens as hf_tokens

    loads = []
    monkeypatch.setattr(hf_tokens, "cached_read_refused", lambda *a, **k: True)
    monkeypatch.setattr("huggingface_hub.DatasetCard.load", lambda *a, **k: loads.append(a) or None)
    assert llm_assist._fetch_hf_dataset_card("acme/evals", False) == (None, None)
    assert loads == []
