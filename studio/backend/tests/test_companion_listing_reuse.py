# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One GGUF load asks the Hub for the repo listing once for its mmproj / MTP / DSpark / DFlash lookups."""

from __future__ import annotations

import threading

import pytest

import core.inference.llama_cpp as llama_cpp


class _NoWaitEvent(threading.Event):
    def wait(self, timeout = None):
        return self.is_set()


class _Hub:
    def __init__(
        self,
        files,
        error = None,
    ):
        self.files = list(files)
        self.error = error
        self.calls = []
        self.downloads = []

    def list_repo_files(
        self,
        repo_id,
        *,
        token = None,
        **kwargs,
    ):
        self.calls.append((repo_id, token))
        if self.error is not None:
            raise self.error
        return list(self.files)

    def download(
        self,
        repo,
        name,
        token,
        *,
        cancel_event = None,
        cache_dir = None,
        **kwargs,
    ):
        self.downloads.append(name)
        return f"/hub/{repo}/{name}"


@pytest.fixture
def hub(monkeypatch):
    state = _Hub(["Model-Q4_K_M.gguf", "README.md"])
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "list_repo_files", state.list_repo_files)
    monkeypatch.setattr(llama_cpp, "_hub_download_in_flight", lambda repo: False)
    monkeypatch.setattr(llama_cpp, "_hf_env_offline", lambda: False)
    monkeypatch.setattr(llama_cpp, "hf_hub_download_with_xet_fallback", state.download)
    return state


def _backend():
    backend = llama_cpp.LlamaCppBackend.__new__(llama_cpp.LlamaCppBackend)
    backend._cancel_event = threading.Event()
    return backend


def _lookup(
    backend,
    label,
    pick,
    token = None,
):
    outcome: dict = {}
    found = backend._download_companion_gguf(
        hf_repo = "org/Model-GGUF",
        hf_token = token,
        pick = pick,
        label = label,
        cancel_event = _NoWaitEvent(),
        outcome = outcome,
    )
    return found, outcome


def _none(names):
    return None


def test_lookups_in_one_load_share_one_listing(hub):
    backend = _backend()
    with llama_cpp._load_probe_memo_scope():
        answers = [
            _lookup(backend, label, _none) for label in ("mmproj", "mtp", "dspark", "dflash")
        ]
    assert len(hub.calls) == 1
    assert answers == [(None, {"listed": False, "listed_live": False})] * 4


def test_each_lookup_sees_the_whole_listing(hub):
    hub.files = ["Model-Q4_K_M.gguf", "mtp-Model-Q8_0.gguf"]
    seen = []
    backend = _backend()
    with llama_cpp._load_probe_memo_scope():
        for _ in range(3):
            _lookup(backend, "x", lambda names: seen.append(list(names)) or None)
    assert seen == [["Model-Q4_K_M.gguf", "mtp-Model-Q8_0.gguf"]] * 3
    assert len(hub.calls) == 1


def test_a_companion_download_makes_the_next_lookup_list_again(hub):
    hub.files = ["Model-Q4_K_M.gguf", "mtp-Model-Q8_0.gguf"]
    backend = _backend()
    with llama_cpp._load_probe_memo_scope():
        found, _ = _lookup(backend, "mtp", llama_cpp._pick_mtp)
        _lookup(backend, "dspark", _none)
        _lookup(backend, "dflash", _none)
    assert found == "/hub/org/Model-GGUF/mtp-Model-Q8_0.gguf"
    assert hub.downloads == ["mtp-Model-Q8_0.gguf"]
    # Listed for MTP, again after its download for DSpark, reused by DFlash.
    assert len(hub.calls) == 2


def test_failed_listings_are_never_reused(hub):
    class RepositoryNotFoundError(Exception):
        pass

    hub.error = RepositoryNotFoundError("missing")
    backend = _backend()
    with llama_cpp._load_probe_memo_scope():
        answers = [_lookup(backend, label, _none) for label in ("mtp", "dspark", "dflash")]
    assert len(hub.calls) == 3
    assert [found for found, _ in answers] == [None, None, None]


def test_transient_failures_still_retry_per_lookup(hub, monkeypatch):
    attempts = {"n": 0}

    def flaky(
        repo_id,
        *,
        token = None,
        **kwargs,
    ):
        hub.calls.append((repo_id, token))
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ConnectionError("reset")
        return list(hub.files)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "list_repo_files", flaky)
    backend = _backend()
    with llama_cpp._load_probe_memo_scope():
        first = _lookup(backend, "mtp", _none)
        second = _lookup(backend, "dspark", _none)
    assert first == second == (None, {"listed": False, "listed_live": False})
    assert len(hub.calls) == 2


def test_outside_a_load_every_lookup_lists(hub):
    backend = _backend()
    for label in ("mtp", "dspark", "dflash"):
        _lookup(backend, label, _none)
    assert len(hub.calls) == 3


def test_tokens_and_loads_do_not_share_a_listing(hub):
    backend = _backend()
    with llama_cpp._load_probe_memo_scope():
        _lookup(backend, "mtp", _none, token = None)
        _lookup(backend, "mtp", _none, token = "hf_other")
        _lookup(backend, "dspark", _none, token = "hf_other")
    with llama_cpp._load_probe_memo_scope():
        _lookup(backend, "mtp", _none)
    assert hub.calls == [
        ("org/Model-GGUF", None),
        ("org/Model-GGUF", "hf_other"),
        ("org/Model-GGUF", None),
    ]


def test_nested_scope_restores_the_outer_listing_and_threads_stay_apart(hub):
    backend = _backend()
    with llama_cpp._load_probe_memo_scope():
        _lookup(backend, "mtp", _none)
        with llama_cpp._load_probe_memo_scope():
            _lookup(backend, "mtp", _none)
        _lookup(backend, "dspark", _none)
        worker = threading.Thread(target = lambda: _lookup(backend, "dflash", _none))
        worker.start()
        worker.join()
    # Outer listing, inner listing, outer reuse, the other thread's own listing.
    assert len(hub.calls) == 3
    assert getattr(llama_cpp._LOAD_PROBE_STATE, "companion_listings", None) is None
