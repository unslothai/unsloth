# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A per-token cache is only a boundary if the two callers hash to different tokens.

Studio's Hub metadata caches are keyed by ``token_fingerprint``, which folded every falsy
token into ``""``. Once ``hf_token_arg`` began returning ``False`` for an API-key caller
that identity became shared with ``None`` -- a UI session that may use the backend's
ambient credential -- so the caches leaked in both directions:

* a UI session warms a private repo's size and blob hashes under the operator's token,
  and an API key denied that token reads them straight back out;
* an API key's anonymous 403 writes a negative entry that blanks the UI's next lookup.

The dataset cache is the sharpest case because it already carries the defence explicitly
-- ``not restricted or cached_fp == token_fp`` -- which the collision turned into
``"" == ""``. These tests drive the real cache, not the fingerprint in isolation.
"""

import asyncio
import time
from types import SimpleNamespace

import pytest

from hub.services.datasets import downloads as dataset_downloads
from hub.services.models import gguf_variants
from hub.utils.inventory_scan import token_fingerprint


def _sibling(name: str, size: int, sha: str):
    return SimpleNamespace(
        rfilename = name, size = size, lfs = SimpleNamespace(sha256 = sha)
    )


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

    # The UI session holds an explicit token and warms the cache for a private dataset.
    size, hashes = dataset_downloads.get_dataset_snapshot_metadata_cached(
        "org/private-set", "hf_operator_token"
    )
    assert size == 4096 and "sha-private" in hashes

    # The API-key caller is forced anonymous. It must NOT read that entry back.
    anon_size, anon_hashes = dataset_downloads.get_dataset_snapshot_metadata_cached(
        "org/private-set", False
    )

    assert (anon_size, anon_hashes) == (0, frozenset()), (
        "the anonymous caller was served a private dataset's size and blob hashes "
        "out of the cache the UI session filled"
    )
    # And it genuinely tried anonymously rather than being answered from the cache.
    assert calls[-1] is False


def test_an_ambient_ui_entry_is_not_served_to_an_anonymous_caller(monkeypatch):
    """The same leak with the ambient credential rather than an explicit one.

    ``None`` is what a UI session resolves to on an install whose HF_TOKEN lives in the
    environment, and it is the value this PR is careful to keep working. Before the fix
    it and ``False`` shared one cache identity, so this is the exact disclosure the
    boundary exists to prevent.
    """
    calls: list = []

    def _api(*_args, **kwargs):
        token = kwargs.get("token")
        calls.append(token)

        def _dataset_info(repo_id, **_kw):
            # `None` reaches the Hub with the process's ambient HF_TOKEN and succeeds;
            # `False` sends no credential at all and is refused.
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

    size, hashes = dataset_downloads.get_dataset_snapshot_metadata_cached(
        "org/private-set", None
    )
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

    # The API key probes first and is refused. That writes a negative entry.
    refused = dataset_downloads.get_dataset_snapshot_metadata_cached(
        "org/private-set", False
    )
    assert refused == (0, frozenset())

    # The UI session, which does have the credential, must still get a real answer
    # rather than reading the API key's 403 back out of the shared negative slot.
    size, hashes = dataset_downloads.get_dataset_snapshot_metadata_cached(
        "org/private-set", "hf_operator_token"
    )

    assert size == 4096 and "sha-private" in hashes, (
        "an API key's anonymous denial poisoned the UI session's cache slot"
    )


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


# --- in-flight deduplication ---------------------------------------------------------


def test_concurrent_callers_of_different_credentials_do_not_share_one_scan():
    """The in-flight key is not TTL-bounded: two concurrent requests join one task.

    A UI scan started under the ambient token and an API-key scan forced anonymous used
    to produce the same key, so whichever arrived first decided the answer for both.
    """
    computed: list = []

    def _key(token):
        # The same shape as the real in-flight key, with the token identity in it.
        return ("org/repo", False, False, "", token_fingerprint(token), "cache")

    async def _drive():
        def _compute(token):
            # Run in a worker thread by _shared_variants_scan, so this is sync. The sleep
            # holds the task in flight long enough for the second caller to try to join.
            computed.append(token)
            time.sleep(0.05)
            return f"result-for-{token}"

        async def _run(token):
            return await gguf_variants._shared_variants_scan(
                _key(token), lambda: _compute(token)
            )

        return await asyncio.gather(_run(None), _run(False))

    ambient_result, anon_result = asyncio.run(_drive())

    assert ambient_result == "result-for-None"
    assert anon_result == "result-for-False", (
        "the anonymous caller received the scan computed under the ambient credential"
    )
    assert sorted(map(str, computed)) == ["False", "None"], (
        "both credentials must run their own scan"
    )


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
