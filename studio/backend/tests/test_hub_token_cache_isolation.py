# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A per-token cache is only a boundary if the two callers hash to different tokens.

``token_fingerprint`` folded every falsy token into ``""``, so ``False`` shared an identity
with ``None`` and the caches leaked both ways: a UI session's private-repo metadata read
back by an API key, and an API key's 403 blanking the UI's next lookup. The dataset cache
is sharpest -- its own ``not restricted or cached_fp == token_fp`` guard became ``"" == ""``.
These drive the real cache, not the fingerprint in isolation.
"""

import asyncio
import time
from types import SimpleNamespace

import pytest

from hub.services.datasets import downloads as dataset_downloads
from hub.services.models import gguf_variants
from hub.utils.inventory_scan import token_fingerprint


def _sibling(name: str, size: int, sha: str):
    return SimpleNamespace(rfilename = name, size = size, lfs = SimpleNamespace(sha256 = sha))


def _clear_dataset_caches():
    with dataset_downloads._dataset_size_cache_lock:
        dataset_downloads._dataset_size_cache.clear()
        dataset_downloads._dataset_size_neg_cache.clear()


@pytest.fixture(autouse = True)
def _isolated_caches():
    _clear_dataset_caches()
    yield
    _clear_dataset_caches()


def _stub_dataset_info(monkeypatch, *, private: bool, calls: list):
    """Answer dataset_info only for a token that is actually allowed to see the repo."""

    def _api(*_args, **kwargs):
        token = kwargs.get("token")
        calls.append(token)

        def _dataset_info(repo_id, **_kw):
            if private and not isinstance(token, str):
                # Anonymous or ambient-less: the Hub 404s a private repo.
                raise RuntimeError("401 Unauthorized")
            return SimpleNamespace(
                siblings = [_sibling("data/train.parquet", 4096, "sha-private")],
                private = private,
                gated = False,
            )

        return SimpleNamespace(dataset_info = _dataset_info)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _api)


def test_a_ui_session_private_dataset_is_not_served_to_an_anonymous_caller(monkeypatch):
    calls: list = []
    _stub_dataset_info(monkeypatch, private = True, calls = calls)

    size, hashes = dataset_downloads.get_dataset_snapshot_metadata_cached(
        "org/private-set", "hf_operator_token"
    )
    assert size == 4096 and "sha-private" in hashes

    anon_size, anon_hashes = dataset_downloads.get_dataset_snapshot_metadata_cached(
        "org/private-set", False
    )

    assert (anon_size, anon_hashes) == (0, frozenset()), (
        "the anonymous caller was served a private dataset's size and blob hashes "
        "out of the cache the UI session filled"
    )
    assert calls[-1] is False


def test_an_ambient_ui_entry_is_not_served_to_an_anonymous_caller(monkeypatch):
    """``None`` (ambient install) and ``False`` shared one identity: the exact disclosure."""
    calls: list = []

    def _api(*_args, **kwargs):
        token = kwargs.get("token")
        calls.append(token)

        def _dataset_info(repo_id, **_kw):
            # `None` carries the ambient HF_TOKEN and succeeds; `False` sends none.
            if token is False:
                raise RuntimeError("401 Unauthorized")
            return SimpleNamespace(
                siblings = [_sibling("data/train.parquet", 4096, "sha-private")],
                private = True,
                gated = False,
            )

        return SimpleNamespace(dataset_info = _dataset_info)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _api)

    size, hashes = dataset_downloads.get_dataset_snapshot_metadata_cached("org/private-set", None)
    assert size == 4096 and "sha-private" in hashes

    anon = dataset_downloads.get_dataset_snapshot_metadata_cached("org/private-set", False)

    assert anon == (0, frozenset()), (
        "an API key read a private dataset's metadata out of the slot a UI session "
        "filled with the backend's ambient credential"
    )
    assert calls == [None, False]


def test_an_anonymous_denial_does_not_blank_a_later_ui_lookup(monkeypatch):
    """The negative direction: denial-of-service rather than disclosure."""
    calls: list = []
    _stub_dataset_info(monkeypatch, private = True, calls = calls)

    refused = dataset_downloads.get_dataset_snapshot_metadata_cached("org/private-set", False)
    assert refused == (0, frozenset())

    size, hashes = dataset_downloads.get_dataset_snapshot_metadata_cached(
        "org/private-set", "hf_operator_token"
    )

    assert (
        size == 4096 and "sha-private" in hashes
    ), "an API key's anonymous denial poisoned the UI session's cache slot"


def test_the_same_caller_still_gets_a_cache_hit(monkeypatch):
    """The fix must not turn every lookup into a miss."""
    calls: list = []
    _stub_dataset_info(monkeypatch, private = True, calls = calls)

    first = dataset_downloads.get_dataset_snapshot_metadata_cached(
        "org/private-set", "hf_operator_token"
    )
    second = dataset_downloads.get_dataset_snapshot_metadata_cached(
        "org/private-set", "hf_operator_token"
    )

    assert first == second
    assert len(calls) == 1, "the second identical lookup should have been a cache hit"


def test_concurrent_callers_of_different_credentials_do_not_share_one_scan():
    """Not TTL-bounded: whichever credential started the shared task decided both answers."""
    computed: list = []

    def _key(token):
        return ("org/repo", False, False, "", token_fingerprint(token), "cache")

    async def _drive():
        def _compute(token):
            # Sync: _shared_variants_scan runs it in a thread. The sleep holds it in flight.
            computed.append(token)
            time.sleep(0.05)
            return f"result-for-{token}"

        async def _run(token):
            return await gguf_variants._shared_variants_scan(_key(token), lambda: _compute(token))

        return await asyncio.gather(_run(None), _run(False))

    ambient_result, anon_result = asyncio.run(_drive())

    assert ambient_result == "result-for-None"
    assert (
        anon_result == "result-for-False"
    ), "the anonymous caller received the scan computed under the ambient credential"
    assert sorted(map(str, computed)) == [
        "False",
        "None",
    ], "both credentials must run their own scan"


def test_concurrent_callers_of_the_same_credential_still_dedupe():
    """The fix must not defeat the deduplication the in-flight map exists for."""
    computed: list = []

    async def _drive():
        def _compute():
            computed.append(1)
            time.sleep(0.05)
            return "shared"

        key = ("org/repo", False, False, "", token_fingerprint(False), "cache")
        return await asyncio.gather(
            gguf_variants._shared_variants_scan(key, _compute),
            gguf_variants._shared_variants_scan(key, _compute),
        )

    first, second = asyncio.run(_drive())

    assert first == second == "shared"
    assert len(computed) == 1, "two identical-scope requests should share one scan"
