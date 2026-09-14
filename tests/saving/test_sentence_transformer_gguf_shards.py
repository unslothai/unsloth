# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

from __future__ import annotations

import os
import types

import pytest


@pytest.fixture
def sentence_transformer_module():
    return pytest.importorskip("unsloth.models.sentence_transformer")


class _FakeSentenceTransformer:
    tokenizer = object()

    def save_pretrained(self, path):
        os.makedirs(os.path.join(path, "0_Transformer"), exist_ok = True)

    def __getitem__(self, index):
        assert index == 0
        return types.SimpleNamespace(auto_model = object())


def test_sentence_transformer_save_forwards_gguf_shard_size(
    sentence_transformer_module, tmp_path, monkeypatch
):
    seen = {}
    monkeypatch.setattr(
        sentence_transformer_module,
        "unsloth_save_pretrained_gguf",
        lambda *args, **kwargs: (seen.update(kwargs), {"gguf_files": []})[1],
    )

    sentence_transformer_module._save_pretrained_gguf(
        _FakeSentenceTransformer(),
        str(tmp_path / "sentence model ü"),
        gguf_shard_size = "512MB",
    )

    assert seen["gguf_shard_size"] == "512MB"
    assert seen["merge_is_disposable"] is False


def test_sentence_transformer_push_forwards_gguf_shard_size(
    sentence_transformer_module, monkeypatch
):
    seen = {}

    class FakeApi:
        def __init__(self, token):
            assert token == "token"

        def create_repo(self, **kwargs):
            return None

        def upload_file(self, **kwargs):
            return None

        def add_tags(self, **kwargs):
            return None

    monkeypatch.setattr(sentence_transformer_module, "HfApi", FakeApi)
    monkeypatch.setattr(
        sentence_transformer_module,
        "_save_pretrained_gguf",
        lambda *args, **kwargs: (
            seen.update(kwargs),
            {
                "gguf_files": [],
                "modelfile_location": None,
                "is_vlm": False,
                "fix_bos_token": False,
            },
        )[1],
    )

    result = sentence_transformer_module._push_to_hub_gguf(
        object(),
        "owner/model",
        token = "token",
        gguf_shard_size = "2GB",
    )

    assert result == "owner/model"
    assert seen["gguf_shard_size"] == "2GB"
