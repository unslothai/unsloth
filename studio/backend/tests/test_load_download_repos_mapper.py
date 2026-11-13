# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from types import SimpleNamespace

import pytest


def test_unmapped_lora_base_skips_the_remote_mapper(monkeypatch):
    # Not importorskip("unsloth"): an earlier file in the worker may have stubbed it as a plain module.
    # exc_type: the real package raises a plain ImportError when unsloth_zoo is missing.
    loader_utils = pytest.importorskip("unsloth.models.loader_utils", exc_type = ImportError)
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

    monkeypatch.setattr(
        loader_utils, "_get_new_mapper", lambda: ({}, {"owner/unmapped-base": "x/y"}, {}, {}, {})
    )
    assert worker._load_download_repos(unmapped, True, cuda) == [
        "owner/adapter",
        "owner/unmapped-base",
    ]

    name = next(iter(loader_utils.FLOAT_TO_INT_MAPPER))
    mapped = SimpleNamespace(identifier = "owner/adapter", base_model = name)
    loader = pytest.importorskip("unsloth.models.loader", exc_type = ImportError)

    expected = loader_utils.get_model_name(name, load_in_4bit = loader.ALLOW_BITSANDBYTES)
    if not loader.ALLOW_PREQUANTIZED_MODELS:
        expected = loader._strip_unsloth_bnb_4bit_suffix(expected)
    assert worker._load_download_repos(mapped, True, cuda) == list(
        dict.fromkeys(["owner/adapter", name, expected])
    )
    assert fetches == []


def test_unmapped_prequantized_base_still_lists_the_stripped_repo(monkeypatch):
    loader = pytest.importorskip("unsloth.models.loader", exc_type = ImportError)
    loader_utils = pytest.importorskip("unsloth.models.loader_utils", exc_type = ImportError)
    from core.inference import worker

    monkeypatch.setattr(loader_utils, "_get_new_mapper", lambda: ({}, {}, {}, {}, {}))
    monkeypatch.setattr(loader, "ALLOW_PREQUANTIZED_MODELS", False)
    mc = SimpleNamespace(identifier = "owner/adapter", base_model = "owner/unmapped-bnb-4bit")
    assert worker._load_download_repos(mc, True, SimpleNamespace(device = "cuda")) == [
        "owner/adapter",
        "owner/unmapped-bnb-4bit",
        "owner/unmapped",
    ]
