# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from types import SimpleNamespace

import pytest


def test_unmapped_lora_base_skips_the_remote_mapper(monkeypatch):
    pytest.importorskip("unsloth")
    from unsloth.models import loader_utils
    from core.inference import worker

    fetches = []

    def _fetch():
        fetches.append(1)
        return ({}, {}, {}, {}, {})

    monkeypatch.setattr(loader_utils, "_get_new_mapper", _fetch)
    monkeypatch.setattr(loader_utils, "_env_says_offline", lambda: False)
    cuda = SimpleNamespace(device = "cuda")

    unmapped = SimpleNamespace(identifier = "owner/adapter", base_model = "owner/unmapped-base")
    for flag in (True, False):
        assert worker._load_download_repos(unmapped, flag, cuda) == [
            "owner/adapter",
            "owner/unmapped-base",
        ]
    assert fetches == []

    # A base only the remote mapper knows raised before, so it never added a repo either.
    monkeypatch.setattr(
        loader_utils, "_get_new_mapper", lambda: ({}, {"owner/unmapped-base": "x/y"}, {}, {}, {})
    )
    assert worker._load_download_repos(unmapped, True, cuda) == [
        "owner/adapter",
        "owner/unmapped-base",
    ]

    name = next(iter(loader_utils.FLOAT_TO_INT_MAPPER))
    mapped = SimpleNamespace(identifier = "owner/adapter", base_model = name)
    from unsloth.models import loader

    expected = loader_utils.get_model_name(name, load_in_4bit = loader.ALLOW_BITSANDBYTES)
    if not loader.ALLOW_PREQUANTIZED_MODELS:
        expected = loader._strip_unsloth_bnb_4bit_suffix(expected)
    assert worker._load_download_repos(mapped, True, cuda) == list(
        dict.fromkeys(["owner/adapter", name, expected])
    )
    assert fetches == []
