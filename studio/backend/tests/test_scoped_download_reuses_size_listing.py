# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A scoped download start reuses the size probe's fresh, credential-equivalent listing; otherwise it lists again."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from hub.schemas.downloads import DownloadModelRequest
from hub.services import download_lifecycle
from hub.services.models import downloads as dl

REPO = "org/scoped-reuse-model"
FILES = ["a.gguf", "b.json"]


def _sib(
    name,
    sha,
    size = 10,
):
    return SimpleNamespace(
        rfilename = name,
        size = size,
        blob_id = f"blob-{sha}",
        lfs = SimpleNamespace(sha256 = sha, size = size) if name.endswith(".gguf") else None,
    )


@pytest.fixture
def hub(monkeypatch):
    state = {"calls": [], "version": 1, "fail_repo_info": False}

    def listing():
        v = state["version"]
        return [
            _sib("a.gguf", f"sha-a-{v}"),
            _sib("b.json", f"sha-b-{v}"),
            _sib("c.gguf", f"sha-c-{v}"),
        ]

    class _Api:
        def __init__(
            self,
            token = None,
            **_,
        ):
            self.token = token

        def repo_info(
            self,
            repo_id,
            *,
            repo_type = None,
            files_metadata = False,
            timeout = None,
            **_,
        ):
            state["calls"].append(("repo_info", repo_id, self.token, files_metadata))
            if state["fail_repo_info"]:
                raise OSError("hub down")
            return SimpleNamespace(siblings = listing())

        def model_info(
            self,
            repo_id,
            *,
            files_metadata = False,
            token = None,
            **_,
        ):
            state["calls"].append(("model_info", repo_id, token, files_metadata))
            return SimpleNamespace(siblings = listing())

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    monkeypatch.setattr(download_lifecycle, "_REPO_SIBLINGS", {})
    monkeypatch.setattr(dl, "_reject_if_load_in_flight", lambda repo_id: None)
    monkeypatch.setattr(dl, "resolve_cached_repo_id_case", lambda repo, **k: repo)
    monkeypatch.setattr(download_lifecycle, "launch_worker", lambda *a, **k: "running")

    async def _no_gate(*a, **k):
        return None

    monkeypatch.setattr(dl, "_refuse_stale_scoped_download", _no_gate)
    real = dl.scoped_file_blob_hashes
    hashes = []

    def spy(*a, **k):
        out = real(*a, **k)
        hashes.append(out)
        return out

    monkeypatch.setattr(dl, "scoped_file_blob_hashes", spy)

    def run(hf_token = None, *, allow_ambient_token = True):
        state["calls"].clear()
        hashes.clear()
        body = DownloadModelRequest(
            repo_id = REPO, scope_id = "audio", files = list(FILES), use_xet = False
        )
        asyncio.run(
            dl.download_model_response(body, hf_token, allow_ambient_token = allow_ambient_token)
        )
        key = dl._download_job_key(REPO, dl._scope_variant("audio"))
        dl._registry.set_job(key, "complete")
        return list(state["calls"]), hashes[0]

    run.state = state
    return run


EXPECTED_V1 = frozenset({"sha-a-1", "blob-sha-b-1"})


@pytest.mark.parametrize("token", [None, "hf_abc"])
def test_one_listing_per_scoped_start(hub, token):
    calls, hashes = hub(token)
    assert [c[0] for c in calls] == ["repo_info"]
    assert hashes == EXPECTED_V1


@pytest.mark.parametrize(
    "token, ambient",
    [(None, False), (" hf_abc ", True), ("", True), ("hf_abc ", False)],
)
def test_a_different_credential_lists_again(hub, token, ambient):
    calls, hashes = hub(token, allow_ambient_token = ambient)
    assert [c[0] for c in calls] == ["repo_info", "model_info"]
    assert calls[1][2] == token
    assert hashes == EXPECTED_V1


def test_a_failed_size_probe_lists_again(hub):
    hub.state["fail_repo_info"] = True
    calls, hashes = hub("hf_abc")
    assert [c[0] for c in calls] == ["repo_info", "model_info"]
    assert hashes == EXPECTED_V1


def test_a_ttl_memo_from_an_earlier_request_is_not_reused(hub):
    hub("hf_abc")  # fills the 60 s size memo with version 1
    hub.state["version"] = 2  # the repo changed since
    calls, hashes = hub("hf_abc")
    assert [c[0] for c in calls] == ["model_info"]
    assert hashes == frozenset({"sha-a-2", "blob-sha-b-2"})


def test_a_failed_refresh_serving_the_old_listing_is_not_reused(hub, monkeypatch):
    hub("hf_abc")
    key = next(iter(download_lifecycle._REPO_SIBLINGS))
    sibs, _ = download_lifecycle._REPO_SIBLINGS[key]
    download_lifecycle._REPO_SIBLINGS[key] = (sibs, time.monotonic() - 3600)
    hub.state["fail_repo_info"] = True
    hub.state["version"] = 2
    calls, hashes = hub("hf_abc")
    assert [c[0] for c in calls] == ["repo_info", "model_info"]
    assert hashes == frozenset({"sha-a-2", "blob-sha-b-2"})


def test_an_empty_listing_is_not_reused():
    download_lifecycle._REPO_SIBLINGS.clear()
    since = time.monotonic()
    key = ("model", REPO.lower(), download_lifecycle.hf_cache_scan.token_fingerprint(None))
    download_lifecycle._REPO_SIBLINGS[key] = ((), since)
    try:
        assert (
            download_lifecycle.repo_siblings_fetched_since(
                "model", REPO, None, allow_ambient_token = True, since = since
            )
            is None
        )
    finally:
        download_lifecycle._REPO_SIBLINGS.pop(key, None)


def test_scoped_hashes_match_with_and_without_a_listing(monkeypatch):
    import huggingface_hub

    sibs = [_sib("a.gguf", "x"), _sib("b.json", "y"), _sib("c.gguf", "z")]

    class _Api:
        def model_info(self, *a, **k):
            return SimpleNamespace(siblings = sibs)

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    fetched = dl.scoped_file_blob_hashes(REPO, FILES, None)
    reused = dl.scoped_file_blob_hashes(REPO, FILES, None, siblings = tuple(sibs))
    assert fetched == reused == frozenset({"x", "blob-y"})
