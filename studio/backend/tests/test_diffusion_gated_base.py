# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hermetic tests for the gated companion-base preflight.

Hugging Face gates the BYTE endpoint only, so ``model_info`` answers anonymously for FLUX / Krea /
Ideogram and the download plan is built from a full file list before anything 401s. Stubbing
``HfApi`` / ``get_hf_file_metadata`` pins both halves: the plan fails up front naming the repo and
its licence page, and every non-access failure falls through so an offline host can still load.
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
    def __init__(
        self,
        gated,
        siblings = (),
    ):
        # The Hub reports "auto" / "manual" (a truthy STRING) or False, never True.
        self.gated = gated
        self.siblings = list(siblings)


class _FakeSibling:
    def __init__(self, rfilename, size):
        self.rfilename = rfilename
        self.size = size


def _stub_hub(
    monkeypatch,
    *,
    info = None,
    model_info_error = None,
    download_error = None,
):
    """Point model_info / the byte-URL HEAD at canned outcomes; returns the probe log."""
    probed: list = []

    class _Api:
        def model_info(
            self,
            repo_id,
            files_metadata = False,
            token = None,
        ):
            if model_info_error is not None:
                raise model_info_error
            return info

    def _metadata(
        url,
        token = None,
        **kwargs,
    ):
        probed.append(url)
        if download_error is not None:
            raise download_error
        return types.SimpleNamespace(etag = "abc", size = 1000)

    monkeypatch.setattr("huggingface_hub.HfApi", lambda *a, **k: _Api())
    monkeypatch.setattr("huggingface_hub.get_hf_file_metadata", _metadata)
    # The probe must never route through the download API: a cached manifest answers that from disk.
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda *a, **k: pytest.fail("the access probe must not be satisfiable from the cache"),
    )
    return probed


def _gated_error():
    from huggingface_hub.errors import GatedRepoError
    return GatedRepoError("401 Client Error. Cannot access gated repo for url ...")


def test_a_gated_base_fails_at_plan_time_naming_the_repo_and_its_licence(monkeypatch):
    # The whole point: metadata says 18 files / 15.4 GiB, the first byte says 401.
    probed = _stub_hub(monkeypatch, info = _FakeInfo("auto"), download_error = _gated_error())

    with pytest.raises(ValueError) as excinfo:
        _assert_base_repo_accessible(GATED_REPO, None)

    detail = str(excinfo.value)
    assert GATED_REPO in detail
    assert f"https://huggingface.co/{GATED_REPO}" in detail
    assert "licence" in detail.lower() and "token" in detail.lower()
    # Probed the manifest the load fetches anyway, not a multi-GB shard.
    assert probed == [f"https://huggingface.co/{GATED_REPO}/resolve/main/model_index.json"]


def test_an_open_base_is_never_probed(monkeypatch):
    # gated is False for almost every repo, so the preflight costs one metadata call.
    probed = _stub_hub(monkeypatch, info = _FakeInfo(False), download_error = _gated_error())

    _assert_base_repo_accessible("Tongyi-MAI/Z-Image-Turbo", None)

    assert probed == []


def test_a_network_error_fails_open(monkeypatch):
    # Offline / transient must never refuse a load: the download surfaces any real error.
    _stub_hub(monkeypatch, model_info_error = OSError("Connection reset by peer"))
    _assert_base_repo_accessible(GATED_REPO, None)

    # Same on the byte probe: only an access verdict counts.
    from huggingface_hub.errors import EntryNotFoundError

    _stub_hub(monkeypatch, info = _FakeInfo("auto"), download_error = EntryNotFoundError("404"))
    _assert_base_repo_accessible(GATED_REPO, None)

    _stub_hub(monkeypatch, info = _FakeInfo("manual"), download_error = TimeoutError("read timed out"))
    _assert_base_repo_accessible(GATED_REPO, None)


def test_unreadable_metadata_is_named_too(monkeypatch):
    # A private / renamed / deleted base 401s on model_info, which the size estimate swallows: the plan would stage zero bytes and the load fail with no explanation.
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


def test_a_cached_manifest_cannot_satisfy_the_probe(monkeypatch, tmp_path):
    # hf_hub_download keeps a 401 HEAD as head_call_error and returns the cached pointer anyway
    # (file_download._hf_hub_download_to_cache_dir), so a manifest on disk would clear the preflight
    # for a stale token and 401 again mid-prefetch. Only a bare HEAD verifies current access.
    root = tmp_path / "hub"
    folder = root / f"models--{GATED_REPO.replace('/', '--')}"
    commit = "c" * 40
    (folder / "refs").mkdir(parents = True)
    (folder / "refs" / "main").write_text(commit)
    (folder / "snapshots" / commit).mkdir(parents = True)
    (folder / "snapshots" / commit / "model_index.json").write_text("{}")
    monkeypatch.setattr("core.inference.diffusion.hub_cache_dir", lambda: str(root))

    probed = _stub_hub(monkeypatch, info = _FakeInfo("auto"), download_error = _gated_error())

    with pytest.raises(ValueError) as excinfo:
        _assert_base_repo_accessible(GATED_REPO, None)

    assert GATED_REPO in str(excinfo.value)
    assert probed == [f"https://huggingface.co/{GATED_REPO}/resolve/main/model_index.json"]


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
    # End to end: the ValueError the route maps to a 400 replaces a confident 18-file plan.
    _stub_hub(
        monkeypatch,
        info = _FakeInfo("auto", [_FakeSibling("model_index.json", 1000)]),
        download_error = _gated_error(),
    )
    monkeypatch.setattr("core.inference.diffusion._resolve_base_repo", lambda *a, **k: GATED_REPO)

    with pytest.raises(ValueError) as excinfo:
        DiffusionBackend().download_plan(
            "unsloth/FLUX.1-dev-GGUF", gguf_filename = "flux1-dev-Q4_K_M.gguf"
        )

    assert GATED_REPO in str(excinfo.value)


def test_run_load_stamps_the_gated_error_on_the_load(monkeypatch):
    # The load path takes the same preflight, so the failure reaches the UI as a load error.
    backend = DiffusionBackend()
    monkeypatch.setattr(
        backend, "validate_load_request", lambda *a, **k: types.SimpleNamespace(name = "flux.1")
    )
    monkeypatch.setattr(
        "core.inference.diffusion.detect_family_for_pick",
        lambda *a, **k: types.SimpleNamespace(name = "flux.1", single_file_is_pipeline = False),
    )
    monkeypatch.setattr("core.inference.diffusion._resolve_base_repo", lambda *a, **k: GATED_REPO)
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


def _auth_error(status):
    """The bare HfHubHTTPError hf_raise_for_status leaves for an unclassified 401/403.

    Its RepoNotFound branch excludes 401 "Invalid credentials in Authorization header" by name, and
    a permission-scoped 403 has no branch at all, so neither becomes GatedRepoError."""
    import requests
    from huggingface_hub.errors import HfHubHTTPError

    response = requests.Response()
    response.status_code = status
    return HfHubHTTPError(f"{status} Client Error.", response = response)


@pytest.mark.parametrize("status", [401, 403])
def test_an_invalid_token_is_an_access_error_not_a_transient_one(status, monkeypatch):
    """An expired token must not fail open: that is the case the probe exists for."""
    _stub_hub(monkeypatch, model_info_error = _auth_error(status))
    with pytest.raises(ValueError) as excinfo:
        _assert_base_repo_accessible(GATED_REPO, "stale-token")
    assert GATED_REPO in str(excinfo.value)


@pytest.mark.parametrize("status", [401, 403])
def test_an_invalid_token_on_the_byte_probe_is_an_access_error(status, monkeypatch):
    """Same on the second half, where metadata succeeded and only the HEAD carries the verdict."""
    _stub_hub(monkeypatch, info = _FakeInfo("auto"), download_error = _auth_error(status))
    with pytest.raises(ValueError) as excinfo:
        _assert_base_repo_accessible(GATED_REPO, "stale-token")
    assert GATED_REPO in str(excinfo.value)


@pytest.mark.parametrize("status", [500, 429])
def test_a_server_error_still_fails_open(status, monkeypatch):
    """A 5xx or a rate limit is not an access verdict, so an offline-ish host still loads."""
    _stub_hub(monkeypatch, info = _FakeInfo("auto"), download_error = _auth_error(status))
    _assert_base_repo_accessible(GATED_REPO, "token")


@pytest.mark.parametrize("token", ["", "   ", None])
def test_a_blank_token_is_not_sent_as_a_credential(token, monkeypatch):
    """build_hf_headers sends any str verbatim, so "" becomes a literal "Bearer " that the Hub
    answers 401 invalid-credentials. Left unnormalized, the 401 handling would turn an open base
    into a hard access error, which is the opposite of what this preflight is for."""
    seen: list = []

    class _Api:
        def model_info(self, repo_id, files_metadata = False, token = None):
            seen.append(token)
            return _FakeInfo(False)

    monkeypatch.setattr("huggingface_hub.HfApi", lambda *a, **k: _Api())
    _assert_base_repo_accessible("some-org/open-model", token)

    assert seen == [None]  # blank normalized away, so the cached login still applies
