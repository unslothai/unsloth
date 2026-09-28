# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A token the Hub rejects must not block downloads of repos anyone may read.

An expired or revoked token (an ``hf_oauth_`` one especially) makes huggingface.co answer 401
to every read, public repos included, and huggingface_hub reports that as "repository not
found" (#11551). The download path retries such a read once without the token. A private or
gated repo still refuses the anonymous read, so nothing the token alone could reach is served.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

from hub.utils import hf_tokens
from hub.utils.hf_tokens import (
    call_hub_with_anonymous_retry,
    is_token_rejection,
    saved_token_rejected,
    token_rejection_scope,
)

BAD = "hf_oauth_" + "x" * 40
REPO = "unsloth/Qwen3-0.6B-GGUF"
VARIANT = "UD-Q4_K_XL"
MAIN = f"{VARIANT}/Qwen3-0.6B-{VARIANT}-00001-of-00002.gguf"
SHARD = f"{VARIANT}/Qwen3-0.6B-{VARIANT}-00002-of-00002.gguf"


def _http_error(cls, status: int):
    request = httpx.Request("GET", f"https://huggingface.co/api/models/{REPO}")
    response = httpx.Response(status, request = request)
    return cls(f"{status} Client Error. (Request ID: test)", response = response)


def _rejected():
    return _http_error(RepositoryNotFoundError, 401)


class _Hub:
    """Answers anonymous reads; refuses any read carrying a token with the OAuth 401."""

    def __init__(
        self,
        *,
        private: bool = False,
        anonymous_error = None,
    ):
        self.private = private
        self.anonymous_error = anonymous_error
        self.tokens: list = []

    def __call__(
        self,
        *args,
        token = None,
        **kwargs,
    ):
        self.tokens.append(token)
        if token is False:
            if self.private:
                raise _rejected()
            if self.anonymous_error is not None:
                raise self.anonymous_error
            return "ok"
        raise _rejected()


@pytest.fixture(autouse = True)
def _fresh_scope():
    with token_rejection_scope():
        yield


# The helper


def test_rejected_token_retries_once_anonymously_and_is_remembered():
    hub = _Hub()
    assert call_hub_with_anonymous_retry(hub, BAD) == "ok"
    assert hub.tokens == [BAD, False]
    assert saved_token_rejected(BAD)
    # The next read in the same request skips the doomed attempt.
    assert call_hub_with_anonymous_retry(hub, BAD) == "ok"
    assert hub.tokens == [BAD, False, False]


def test_private_repo_keeps_the_original_refusal():
    hub = _Hub(private = True)
    with pytest.raises(RepositoryNotFoundError) as info:
        call_hub_with_anonymous_retry(hub, BAD)
    assert hub.tokens == [BAD, False]
    assert info.value.response.status_code == 401
    assert not saved_token_rejected(BAD)


def test_after_a_rejection_a_private_read_still_asks_with_the_token():
    hf_tokens.note_saved_token_rejected(BAD)
    hub = _Hub(private = True)
    with pytest.raises(RepositoryNotFoundError):
        call_hub_with_anonymous_retry(hub, BAD)
    assert hub.tokens == [False, BAD]


def test_gated_repo_is_not_opened_by_the_retry():
    hub = _Hub(anonymous_error = _http_error(GatedRepoError, 401))
    with pytest.raises(RepositoryNotFoundError):
        call_hub_with_anonymous_retry(hub, BAD)
    assert hub.tokens == [BAD, False]


@pytest.mark.parametrize(
    "error",
    [
        _http_error(HfHubHTTPError, 403),
        _http_error(RepositoryNotFoundError, 404),
        _http_error(HfHubHTTPError, 500),
        httpx.ReadTimeout("slow"),
        OSError("disk"),
    ],
    ids = ["403", "404", "500", "timeout", "oserror"],
)
def test_only_a_401_is_retried(error):
    tokens = []

    def read(*, token):
        tokens.append(token)
        raise error

    with pytest.raises(type(error)):
        call_hub_with_anonymous_retry(read, BAD)
    assert tokens == [BAD]


def test_an_anonymous_caller_is_never_retried():
    hub = _Hub(private = True)
    with pytest.raises(RepositoryNotFoundError):
        call_hub_with_anonymous_retry(hub, False)
    assert hub.tokens == [False]


def test_ambient_without_any_host_token_is_not_retried(monkeypatch):
    monkeypatch.setattr(hf_tokens, "_ambient_hf_token", lambda: (True, None))
    hub = _Hub()
    with pytest.raises(RepositoryNotFoundError):
        call_hub_with_anonymous_retry(hub, None)
    assert hub.tokens == [None]


def test_ambient_host_token_is_retried(monkeypatch):
    monkeypatch.setattr(hf_tokens, "_ambient_hf_token", lambda: (True, BAD))
    hub = _Hub()
    assert call_hub_with_anonymous_retry(hub, None) == "ok"
    assert hub.tokens == [None, False]


def test_scope_forgets_the_verdict():
    with token_rejection_scope():
        call_hub_with_anonymous_retry(_Hub(), BAD)
        assert saved_token_rejected(BAD)
    assert not saved_token_rejected(BAD)


def test_a_different_token_is_not_marked_rejected():
    call_hub_with_anonymous_retry(_Hub(), BAD)
    assert not saved_token_rejected("hf_" + "y" * 34)


def test_a_401_rebuilt_from_a_child_process_is_recognised():
    # The download ladder re-raises a child's error from its "<Class>: <message>" text.
    # The same bypass the ladder uses when the constructor demands a response.
    rebuilt = RepositoryNotFoundError.__new__(RepositoryNotFoundError)
    BaseException.__init__(rebuilt, "RepositoryNotFoundError: 401 Client Error. (Request ID: x)")
    assert is_token_rejection(rebuilt)
    assert is_token_rejection(RuntimeError("RepositoryNotFoundError: 401 Client Error. (x)"))
    assert not is_token_rejection(RuntimeError("403 Client Error. (x)"))
    wrapped = RuntimeError("download failed")
    wrapped.__cause__ = _rejected()
    assert is_token_rejection(wrapped)


# The shared download entry point


def test_xet_fallback_download_retries_anonymously(monkeypatch, tmp_path):
    from utils import hf_xet_fallback

    seen = []

    def shared(repo_id, filename, token, **kwargs):
        seen.append(token)
        if token is False:
            return str(tmp_path / filename)
        raise _rejected()

    monkeypatch.setattr(hf_xet_fallback, "_shared_hf_hub_download_with_xet_fallback", shared)
    path = hf_xet_fallback.hf_hub_download_with_xet_fallback(
        REPO, "model.gguf", BAD, cache_dir = str(tmp_path)
    )
    assert path == str(tmp_path / "model.gguf")
    assert seen == [BAD, False]


def test_xet_fallback_download_of_a_private_repo_still_fails(monkeypatch, tmp_path):
    from utils import hf_xet_fallback

    seen = []

    def shared(repo_id, filename, token, **kwargs):
        seen.append(token)
        raise _rejected()

    monkeypatch.setattr(hf_xet_fallback, "_shared_hf_hub_download_with_xet_fallback", shared)
    with pytest.raises(RepositoryNotFoundError):
        hf_xet_fallback.hf_hub_download_with_xet_fallback(
            REPO, "model.gguf", BAD, cache_dir = str(tmp_path)
        )
    assert seen == [BAD, False]


# llama.cpp


def _listing(
    repo_id,
    *,
    token = None,
    **_kwargs,
):
    if token is not False:
        raise _rejected()
    return [MAIN, SHARD, "README.md"]


def test_variant_resolution_uses_the_anonymous_listing(monkeypatch):
    from core.inference import llama_cpp

    monkeypatch.setattr(llama_cpp, "_cached_variant_resolution", lambda *a, **k: (None, []))
    with patch("huggingface_hub.list_repo_files", _listing):
        name, shards = llama_cpp._resolve_variant_gguf_files(REPO, VARIANT, BAD)
    assert name == MAIN
    assert shards == [SHARD]


def _backend_download(monkeypatch, tmp_path, *, private: bool):
    from core.inference import llama_cpp
    from utils import hf_xet_fallback

    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: types.SimpleNamespace(hub_cache = tmp_path),
    )
    monkeypatch.setattr(llama_cpp, "cached_gguf_for_load", lambda *a, **k: None)
    monkeypatch.setattr(llama_cpp, "_cached_variant_resolution", lambda *a, **k: (None, []))
    downloads = []

    def shared(repo_id, filename, token, **kwargs):
        downloads.append((filename, token))
        if token is False and not private:
            return str(tmp_path / filename)
        raise _rejected()

    def paths_info(
        repo_id,
        paths,
        *,
        token = None,
        **_kwargs,
    ):
        if token is not False or private:
            raise _rejected()
        return [types.SimpleNamespace(path = p, size = 4) for p in paths]

    def listing(
        repo_id,
        *,
        token = None,
        **kwargs,
    ):
        if private:
            raise _rejected()
        return _listing(repo_id, token = token, **kwargs)

    monkeypatch.setattr(hf_xet_fallback, "_shared_hf_hub_download_with_xet_fallback", shared)
    with (
        patch("huggingface_hub.list_repo_files", listing),
        patch("huggingface_hub.get_paths_info", paths_info),
        patch("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None),
    ):
        backend = llama_cpp.LlamaCppBackend()
        try:
            return backend._download_gguf(hf_repo = REPO, hf_variant = VARIANT, hf_token = BAD), downloads
        except Exception as exc:
            return exc, downloads


def test_gguf_download_recovers_from_a_rejected_token(monkeypatch, tmp_path):
    out, downloads = _backend_download(monkeypatch, tmp_path, private = False)
    assert out == str(tmp_path / MAIN)
    # Main file and shard, both anonymous once the listing saw the token rejected.
    assert downloads == [(MAIN, False), (SHARD, False)]


def test_gguf_download_of_a_private_repo_names_the_token(monkeypatch, tmp_path):
    out, _ = _backend_download(monkeypatch, tmp_path, private = True)
    assert isinstance(out, RuntimeError)
    assert "token may have expired or been revoked" in str(out)


# Download worker (one job per process)


@pytest.fixture()
def worker(monkeypatch):
    from hub.workers import hf_download

    monkeypatch.setattr(hf_download, "_REJECTED_TOKEN", None)
    monkeypatch.setattr(hf_download, "_METADATA_RETRY_DELAY", 0)
    return hf_download


def test_worker_metadata_retries_and_the_download_follows(worker):
    seen = []

    def model_info(
        repo_id,
        *,
        token = None,
        **_kwargs,
    ):
        seen.append(token)
        if token is False:
            return types.SimpleNamespace(sha = "a" * 40, siblings = [])
        raise _rejected()

    with patch("huggingface_hub.model_info", model_info):
        info = worker._model_info_with_retry(REPO, BAD)
    assert info.sha == "a" * 40
    assert seen == [BAD, False]
    # snapshot_download / hf_hub_download in the worker all take this argument.
    assert worker._hf_token_arg(BAD) is False
    assert worker._hf_token_arg("hf_" + "y" * 34) == "hf_" + "y" * 34


def test_worker_private_repo_keeps_the_token(worker):
    seen = []

    def model_info(
        repo_id,
        *,
        token = None,
        **_kwargs,
    ):
        seen.append(token)
        raise _rejected()

    with patch("huggingface_hub.model_info", model_info):
        with pytest.raises(RepositoryNotFoundError):
            worker._model_info_with_retry(REPO, BAD)
    # Two metadata attempts, each asking with the token and without it (the second may ask
    # anonymously first, having learned the token is refused).
    assert sorted(map(str, seen)) == sorted(map(str, [BAD, False, BAD, False]))
    assert worker._hf_token_arg(BAD) == BAD


def test_worker_dataset_metadata_retries(worker):
    seen = []

    def dataset_info(
        self,
        repo_id,
        *,
        token = None,
        **_kwargs,
    ):
        seen.append(token)
        if token is False:
            return types.SimpleNamespace(sha = "b" * 40)
        raise _rejected()

    with patch("huggingface_hub.HfApi.dataset_info", dataset_info):
        info = worker._dataset_info_with_retry("unsloth/data", BAD)
    assert info.sha == "b" * 40
    assert seen == [BAD, False]
    assert worker._hf_token_arg(BAD) is False


# Inference worker child env


def test_inference_worker_drops_a_rejected_token_from_its_env(monkeypatch):
    from core.inference import worker

    env = {"HF_TOKEN": BAD, "HUGGING_FACE_HUB_TOKEN": BAD}
    monkeypatch.setattr(os, "environ", env)
    config = {"model_name": REPO, "hf_token": BAD}
    call_hub_with_anonymous_retry(_Hub(), BAD)
    worker._drop_a_rejected_token(config)
    assert worker._config_hf_token(config) is False
    assert "HF_TOKEN" not in env and "HUGGING_FACE_HUB_TOKEN" not in env
    assert env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"


def test_inference_worker_keeps_an_accepted_token(monkeypatch):
    from core.inference import worker

    env = {"HF_TOKEN": BAD}
    monkeypatch.setattr(os, "environ", env)
    config = {"model_name": REPO, "hf_token": BAD}
    worker._drop_a_rejected_token(config)
    assert worker._config_hf_token(config) == BAD
    assert env == {"HF_TOKEN": BAD}


def test_inference_worker_load_scope_resets(monkeypatch):
    from core.inference import worker

    seen = []

    def load(backend, config, resp_queue):
        call_hub_with_anonymous_retry(_Hub(), BAD)
        seen.append(saved_token_rejected(BAD))

    worker._in_token_rejection_scope(load)(None, {}, None)
    assert worker._handle_load.__wrapped__ is not None
    assert seen == [True]
    assert not saved_token_rejected(BAD)
