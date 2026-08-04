# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hermetic tests for the gated companion-base preflight.

Hugging Face enforces gating on the BYTE endpoint only, so ``model_info`` answers anonymously
for FLUX / Krea / Ideogram and the download plan is built from a full file list before anything
401s. These stub ``HfApi`` / ``hf_hub_download`` to pin both halves of that asymmetry: the plan
must fail up front naming the repo and its licence page, and every non-access failure must fall
through so an offline or flaky host can still load.
"""

from __future__ import annotations

import types

import pytest

from core.inference.diffusion import (
    DiffusionBackend,
    _assert_base_repo_accessible,
    _LoadingState,
)

GATED_REPO = "black-forest-labs/FLUX.1-dev"


class _FakeInfo:
    def __init__(self, gated, siblings = ()):
        # The Hub reports "auto" / "manual" (a truthy STRING) or False, never True.
        self.gated = gated
        self.siblings = list(siblings)


class _FakeSibling:
    def __init__(self, rfilename, size):
        self.rfilename = rfilename
        self.size = size


def _stub_hub(monkeypatch, *, info = None, model_info_error = None, download_error = None):
    """Point HfApi.model_info / hf_hub_download at canned outcomes; returns the probe log."""
    probed: list = []

    class _Api:
        def model_info(self, repo_id, files_metadata = False, token = None):
            if model_info_error is not None:
                raise model_info_error
            return info

    def _download(repo_id, filename, token = None, cache_dir = None, **kwargs):
        probed.append((repo_id, filename))
        if download_error is not None:
            raise download_error
        return "/cache/model_index.json"

    monkeypatch.setattr("huggingface_hub.HfApi", lambda *a, **k: _Api())
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _download)
    return probed


def _gated_error():
    from huggingface_hub.errors import GatedRepoError

    return GatedRepoError("401 Client Error. Cannot access gated repo for url ...")


def test_a_gated_base_fails_at_plan_time_naming_the_repo_and_its_licence(monkeypatch):
    # The whole point: metadata says 18 files / 15.4 GiB, the first byte says 401. Fail here, with something to act on.
    probed = _stub_hub(
        monkeypatch, info = _FakeInfo("auto"), download_error = _gated_error()
    )

    with pytest.raises(ValueError) as excinfo:
        _assert_base_repo_accessible(GATED_REPO, None)

    detail = str(excinfo.value)
    assert GATED_REPO in detail
    assert f"https://huggingface.co/{GATED_REPO}" in detail
    assert "licence" in detail.lower() and "token" in detail.lower()
    # Probed the pipeline manifest the load fetches anyway, not a multi-GB shard.
    assert probed == [(GATED_REPO, "model_index.json")]


def test_an_open_base_is_never_probed(monkeypatch):
    # gated is False for the vast majority of repos, so the preflight costs one metadata call and no byte fetch.
    probed = _stub_hub(monkeypatch, info = _FakeInfo(False), download_error = _gated_error())

    _assert_base_repo_accessible("Tongyi-MAI/Z-Image-Turbo", None)

    assert probed == []


def test_a_network_error_fails_open(monkeypatch):
    # Offline / transient must never be the reason a load is refused: the download surfaces any real error.
    _stub_hub(monkeypatch, model_info_error = OSError("Connection reset by peer"))
    _assert_base_repo_accessible(GATED_REPO, None)

    # Same on the byte probe: only an access verdict counts, so a 500 or a repo without a manifest passes.
    from huggingface_hub.errors import EntryNotFoundError

    _stub_hub(monkeypatch, info = _FakeInfo("auto"), download_error = EntryNotFoundError("404"))
    _assert_base_repo_accessible(GATED_REPO, None)

    _stub_hub(monkeypatch, info = _FakeInfo("manual"), download_error = TimeoutError("read timed out"))
    _assert_base_repo_accessible(GATED_REPO, None)


def test_unreadable_metadata_is_named_too(monkeypatch):
    # A private / renamed / deleted base 401s on model_info, which the size estimate swallows: the plan would then stage zero bytes and the load fail with no explanation.
    from huggingface_hub.errors import RepositoryNotFoundError

    _stub_hub(monkeypatch, model_info_error = RepositoryNotFoundError("401 Client Error."))

    with pytest.raises(ValueError) as excinfo:
        _assert_base_repo_accessible("unsloth/not-published-yet", None)

    assert "unsloth/not-published-yet" in str(excinfo.value)
    assert "https://huggingface.co/unsloth/not-published-yet" in str(excinfo.value)

    # A gated repo that also withholds its metadata keeps the licence wording.
    _stub_hub(monkeypatch, model_info_error = _gated_error())
    with pytest.raises(ValueError) as gated:
        _assert_base_repo_accessible(GATED_REPO, None)
    assert "licence" in str(gated.value).lower()


def test_local_and_non_repo_bases_are_skipped(monkeypatch, tmp_path):
    # Only a remote 'org/name' can be gated; a local pipeline dir is already on disk.
    def _explode(*a, **k):
        pytest.fail("a local / non-repo base must never be probed")

    monkeypatch.setattr("huggingface_hub.HfApi", _explode)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _explode)

    local = tmp_path / "my-base"
    local.mkdir()
    _assert_base_repo_accessible(str(local), None)
    _assert_base_repo_accessible("", None)
    _assert_base_repo_accessible("bare-name", None)


def test_download_plan_refuses_a_gated_base_before_listing_files(monkeypatch):
    # End to end through the planner: the ValueError the route maps to a 400 replaces a confident 18-file plan.
    _stub_hub(
        monkeypatch,
        info = _FakeInfo("auto", [_FakeSibling("model_index.json", 1000)]),
        download_error = _gated_error(),
    )
    monkeypatch.setattr(
        "core.inference.diffusion._resolve_base_repo", lambda *a, **k: GATED_REPO
    )

    with pytest.raises(ValueError) as excinfo:
        DiffusionBackend().download_plan(
            "unsloth/FLUX.1-dev-GGUF", gguf_filename = "flux1-dev-Q4_K_M.gguf"
        )

    assert GATED_REPO in str(excinfo.value)


def test_run_load_stamps_the_gated_error_on_the_load(monkeypatch):
    # The load path takes the same preflight, so the failure reaches the UI as a load error instead of dying inside the prefetch.
    backend = DiffusionBackend()
    monkeypatch.setattr(
        backend, "validate_load_request", lambda *a, **k: types.SimpleNamespace(name = "flux.1")
    )
    monkeypatch.setattr(
        "core.inference.diffusion.detect_family_for_pick",
        lambda *a, **k: types.SimpleNamespace(name = "flux.1", single_file_is_pipeline = False),
    )
    monkeypatch.setattr(
        "core.inference.diffusion._resolve_base_repo", lambda *a, **k: GATED_REPO
    )
    _stub_hub(monkeypatch, info = _FakeInfo("auto"), download_error = _gated_error())

    def _no_prefetch(*a, **k):
        pytest.fail("the prefetch must not start once the base is known to be unreadable")

    monkeypatch.setattr(DiffusionBackend, "_prefetch_files", _no_prefetch)
    monkeypatch.setattr(
        DiffusionBackend, "_estimate_download_bytes", staticmethod(lambda *a, **k: (0, []))
    )
    # What begin_load() stamps before handing off to the worker thread.
    backend._loading = _LoadingState(
        repo_id = "unsloth/FLUX.1-dev-GGUF", base_repo = "black-forest-labs/FLUX.1-schnell"
    )

    backend._run_load(
        repo_id = "unsloth/FLUX.1-dev-GGUF",
        gguf_filename = "flux1-dev-Q4_K_M.gguf",
        hf_token = None,
        _load_token = backend._load_token,
    )

    assert GATED_REPO in (backend.load_progress().get("error") or "")
