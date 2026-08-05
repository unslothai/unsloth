# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hermetic tests for the gated companion-base preflight.

Hugging Face gates the BYTE endpoint only, so ``model_info`` answers anonymously for FLUX / Krea /
Ideogram and the download plan is built from a full file list before anything 401s. Stubbing
``HfApi`` / ``get_hf_file_metadata`` pins both halves: the plan fails up front naming the repo and
its licence page, and every non-access failure falls through so an offline host can still load.
"""

from __future__ import annotations

import os
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
    # No ambient cache: whether THIS box happens to hold the repo must not decide the test.
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    # The probe must never route through the download API: a cached manifest answers that from disk.
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda *a, **k: pytest.fail("the access probe must not be satisfiable from the cache"),
    )
    return probed


def _gated_error():
    from huggingface_hub.errors import GatedRepoError
    return _hub_http_error(
        GatedRepoError, "401 Client Error. Cannot access gated repo for url ...", 401
    )


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

    _stub_hub(
        monkeypatch,
        model_info_error = _hub_http_error(RepositoryNotFoundError, "401 Client Error.", 401),
    )

    with pytest.raises(ValueError) as excinfo:
        _assert_base_repo_accessible("unsloth/not-published-yet", None)

    assert "unsloth/not-published-yet" in str(excinfo.value)
    assert "https://huggingface.co/unsloth/not-published-yet" in str(excinfo.value)

    # A gated repo that also withholds its metadata keeps the licence wording.
    _stub_hub(monkeypatch, model_info_error = _gated_error())
    with pytest.raises(ValueError) as gated:
        _assert_base_repo_accessible(GATED_REPO, None)
    assert "licence" in str(gated.value).lower()


def test_an_already_downloaded_base_is_never_refused(monkeypatch, tmp_path):
    """A base whose bytes are on disk loads today with no token at all: hf_hub_download catches the
    gated 401 HEAD and returns the cached pointer. Probing live access there can only refuse a load
    that already works -- a token cleared from Studio settings, an expired one, or a fresh profile
    over an existing cache. The never-downloaded pick this preflight exists for still probes."""
    root = tmp_path / "hub"
    folder = root / f"models--{GATED_REPO.replace('/', '--')}"
    commit = "c" * 40
    (folder / "refs").mkdir(parents = True)
    (folder / "refs" / "main").write_text(commit)
    (folder / "snapshots" / commit).mkdir(parents = True)
    (folder / "snapshots" / commit / "model_index.json").write_text("{}")

    # Cached under the LIVE root, and separately under huggingface_hub's import-time constant: the
    # prefetch downloads under the latter, so checking only one root would still refuse the load.
    for live, imported in ((str(root), tmp_path / "other"), (str(tmp_path / "other"), root)):
        monkeypatch.setattr("core.inference.diffusion.hub_cache_dir", lambda live = live: live)
        monkeypatch.setenv("HF_HUB_CACHE", str(imported))
        probed = _stub_hub(monkeypatch, info = _FakeInfo("auto"), download_error = _gated_error())
        monkeypatch.setattr(
            "huggingface_hub.try_to_load_from_cache",
            lambda repo_id, filename, cache_dir = None, **k: (
                str(
                    root / f"models--{repo_id.replace('/', '--')}" / "snapshots" / commit / filename
                )
                if cache_dir in (str(root), None) and str(root) in (live, str(imported))
                else None
            ),
        )
        _assert_base_repo_accessible(GATED_REPO, "stale-token")
        assert probed == []  # served from disk, so not one Hub call was made

    # Nothing cached: the reported case still fails up front, naming the repo and its licence.
    monkeypatch.setattr("core.inference.diffusion.hub_cache_dir", lambda: str(tmp_path / "empty"))
    _stub_hub(monkeypatch, info = _FakeInfo("auto"), download_error = _gated_error())
    with pytest.raises(ValueError, match = "gated"):
        _assert_base_repo_accessible(GATED_REPO, "stale-token")


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


def test_a_base_whose_home_cannot_be_resolved_fails_open(monkeypatch):
    # '~someoneelse/models' and a plain '~/models' under a service account with no home both
    # carry exactly one slash, so they reach the local-path probe instead of short-circuiting,
    # and pathlib answers an unresolvable home with RuntimeError -- which is NOT an OSError.
    # This function documents "fails open on any non-access error", so it must fall through to
    # the Hub probe rather than escaping as a 500 on a load that has not started.
    probed = _stub_hub(monkeypatch, info = _FakeInfo(False))

    class _NoHomePath:
        def __init__(self, *a, **k):
            pass

        def expanduser(self):
            raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr("core.inference.diffusion.Path", _NoHomePath)

    _assert_base_repo_accessible("~ghost/my-base", None)
    # Fell through to the remote probe instead of raising: an open repo costs one metadata call.
    assert probed == []


def test_an_unknown_user_home_base_fails_open_for_real(monkeypatch):
    # The same failure without stubbing pathlib at all: no such user, so expanduser() raises.
    # POSIX only -- Windows expanduser() has no user database and silently rewrites '~ghost'
    # against USERPROFILE's parent, so it never reaches the RuntimeError branch there.
    if os.name == "nt":
        pytest.skip("POSIX-only: Windows expanduser() never fails for an unknown user")
    _stub_hub(monkeypatch, info = _FakeInfo(False))

    _assert_base_repo_accessible("~unsloth-no-such-user-4b1f/my-base", None)


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


def _hub_http_error(cls, message, status):
    """Build an HfHubHTTPError subclass portably.

    huggingface_hub 1.x made ``response`` a REQUIRED keyword-only argument, so a message-only
    construction raises TypeError there, and CI resolves 1.x (transformers pins
    huggingface-hub>=1.5 over studio.txt's 0.36.2). Passing it works on both."""
    import requests

    response = requests.Response()
    response.status_code = status
    return cls(message, response = response)


def _auth_error(status):
    """The bare HfHubHTTPError hf_raise_for_status leaves for an unclassified 401/403.

    Its RepoNotFound branch excludes 401 "Invalid credentials in Authorization header" by name, and
    a permission-scoped 403 has no branch at all, so neither becomes GatedRepoError."""
    from huggingface_hub.errors import HfHubHTTPError
    return _hub_http_error(HfHubHTTPError, f"{status} Client Error.", status)


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
        def model_info(
            self,
            repo_id,
            files_metadata = False,
            token = None,
        ):
            seen.append(token)
            return _FakeInfo(False)

    monkeypatch.setattr("huggingface_hub.HfApi", lambda *a, **k: _Api())
    _assert_base_repo_accessible("some-org/open-model", token)

    assert seen == [None]  # blank normalized away, so the cached login still applies


def test_the_native_plan_preflights_its_companion_repos_too(monkeypatch):
    """A GPU-less host routes /images/download-plan to the sd.cpp planner, whose asset list carries
    its own companion repos: flux.1's VAE is the gated black-forest-labs/FLUX.1-schnell. The size
    probe swallows the 401, so without the preflight that entry plans at 0 bytes and the manager's
    fetch dies on the bare token error this preflight exists to replace."""
    from core.inference.sd_cpp_backend import SdCppDiffusionBackend

    gated = "black-forest-labs/FLUX.1-schnell"
    b = SdCppDiffusionBackend(engine = None)
    monkeypatch.setattr(
        SdCppDiffusionBackend, "_plan_file_sizes", staticmethod(lambda by_repo, token: {})
    )

    # Only the VAE repo is gated, as on the Hub: the pick and the encoder repo answer normally.
    class _Api:
        def model_info(
            self,
            repo_id,
            files_metadata = False,
            token = None,
        ):
            return _FakeInfo("auto" if repo_id == gated else False)

    monkeypatch.setattr("huggingface_hub.HfApi", lambda *a, **k: _Api())
    monkeypatch.setattr(
        "huggingface_hub.get_hf_file_metadata",
        lambda url, token = None, **k: (_ for _ in ()).throw(_gated_error()),
    )

    with pytest.raises(ValueError) as excinfo:
        b.download_plan(
            "unsloth/FLUX.1-dev-GGUF",
            gguf_filename = "flux1-dev-Q4_K_M.gguf",
            model_kind = "gguf",
        )
    detail = str(excinfo.value)
    assert gated in detail and f"https://huggingface.co/{gated}" in detail

    # An open family is untouched: every companion answers, so the plan is built exactly as before.
    plan = b.download_plan(
        "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z-image-turbo-Q4_K_M.gguf",
        model_kind = "gguf",
    )
    assert {e["repo_id"] for e in plan["entries"]} == {
        "unsloth/Z-Image-Turbo-GGUF",
        "Comfy-Org/z_image_turbo",
    }


def test_the_native_plan_probes_the_asset_it_stages(monkeypatch):
    """flux.1's native VAE repo is read for ae.safetensors only. Probing the pipeline manifest
    there would neither verify access to that file nor see it in the cache, so a host that already
    downloaded the VAE would be refused a load that works today."""
    from core.inference.sd_cpp_backend import SdCppDiffusionBackend

    gated = "black-forest-labs/FLUX.1-schnell"
    b = SdCppDiffusionBackend(engine = None)
    monkeypatch.setattr(
        SdCppDiffusionBackend, "_plan_file_sizes", staticmethod(lambda by_repo, token: {})
    )

    class _Api:
        def model_info(
            self,
            repo_id,
            files_metadata = False,
            token = None,
        ):
            return _FakeInfo("auto" if repo_id == gated else False)

    monkeypatch.setattr("huggingface_hub.HfApi", lambda *a, **k: _Api())
    probed: list = []

    def _metadata(
        url,
        token = None,
        **k,
    ):
        probed.append(url)
        raise _gated_error()

    monkeypatch.setattr("huggingface_hub.get_hf_file_metadata", _metadata)

    # Nothing cached: the probe is the VAE file the plan stages, not the manifest.
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    with pytest.raises(ValueError, match = "gated"):
        b.download_plan(
            "unsloth/FLUX.1-dev-GGUF",
            gguf_filename = "flux1-dev-Q4_K_M.gguf",
            model_kind = "gguf",
        )
    assert probed == [f"https://huggingface.co/{gated}/resolve/main/ae.safetensors"]

    # That same VAE already on disk clears the plan without a single Hub call.
    probed.clear()
    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache",
        lambda repo_id, filename, cache_dir = None, **k: (
            "/cache/ae.safetensors" if filename == "ae.safetensors" else None
        ),
    )
    plan = b.download_plan(
        "unsloth/FLUX.1-dev-GGUF", gguf_filename = "flux1-dev-Q4_K_M.gguf", model_kind = "gguf"
    )
    assert probed == []
    assert gated in {e["repo_id"] for e in plan["entries"]}


def test_the_gguf_is_resolved_against_the_live_cache_root(monkeypatch, tmp_path):
    """The prefetch stages the GGUF under the LIVE root (hf_hub_download_with_xet_fallback
    defaults to it), so an unpinned resolve reads huggingface_hub's import-time constant and,
    after a mid-session cache change, pulls the whole multi-GB file again inside the load lock."""
    live = tmp_path / "live"
    calls: list = []

    def _download(
        repo_id,
        filename,
        token = None,
        cache_dir = None,
        **k,
    ):
        calls.append(cache_dir)
        return str(live / filename)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _download)
    monkeypatch.setattr("core.inference.diffusion.hub_cache_dir", lambda: str(live))

    b = DiffusionBackend()

    # Nothing cached anywhere: the download is pinned to the live root, not the import-time one.
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    b._resolve_gguf_path("unsloth/Z-Image-Turbo-GGUF", "z.gguf", None)
    assert calls == [str(live)]

    # A copy under the OTHER root is reused rather than re-fetched: pinning alone would miss it.
    other = tmp_path / "other" / "z.gguf"
    other.parent.mkdir(parents = True)
    other.write_bytes(b"gguf")
    calls.clear()
    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache",
        lambda repo_id, filename, cache_dir = None, **k: None if cache_dir else str(other),
    )
    assert b._resolve_gguf_path("unsloth/Z-Image-Turbo-GGUF", "z.gguf", None) == str(other)
    assert calls == []  # not one byte re-downloaded


def test_a_private_but_already_downloaded_base_is_not_refused(monkeypatch):
    """huggingface_hub folds 401 into RepositoryNotFoundError: hf_raise_for_status says "401 is
    misleading as it is returned for: private and gated repos if user is not authenticated,
    missing repos => for now, we process them as RepoNotFound anyway" (utils/_http.py). So a
    PRIVATE companion base that is already on disk arrives here indistinguishable from a deleted
    one, and refusing it blocks a load that works today: hf_hub_download serves the cached pointer
    once the token is cleared or expires, exactly as it does for a gated base."""
    from huggingface_hub.errors import RepositoryNotFoundError

    private = "unsloth/private-base"
    for status in (401, 403):
        probed = _stub_hub(
            monkeypatch,
            model_info_error = _hub_http_error(
                RepositoryNotFoundError, f"{status} Client Error.", status
            ),
        )
        monkeypatch.setattr(
            "huggingface_hub.try_to_load_from_cache",
            lambda repo_id, filename, cache_dir = None, **k: "/cache/model_index.json",
        )
        _assert_base_repo_accessible(private, "expired-token")
        assert probed == []  # served from disk, so not one byte probe was made


def test_a_deleted_or_renamed_base_still_raises_even_when_cached(monkeypatch):
    """The other side of the split. A 404 is not an access verdict a stale copy can excuse: the
    repo is gone, the size estimate swallows the 404 into a zero-byte plan, and the pick would
    download nothing and fail later with no explanation."""
    from huggingface_hub.errors import RepositoryNotFoundError

    _stub_hub(
        monkeypatch,
        model_info_error = _hub_http_error(RepositoryNotFoundError, "404 Client Error.", 404),
    )
    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache",
        lambda repo_id, filename, cache_dir = None, **k: "/cache/model_index.json",
    )
    with pytest.raises(ValueError) as excinfo:
        _assert_base_repo_accessible("unsloth/renamed-away", None)
    assert "unsloth/renamed-away" in str(excinfo.value)


def test_a_repo_not_found_with_no_response_still_raises(monkeypatch):
    """Older / newer hub versions may not attach a response, so the status is unreadable. Neither
    401 nor 404 can be proven, so the cache escape is not granted: fail safe, as before."""
    from huggingface_hub.errors import RepositoryNotFoundError

    _stub_hub(monkeypatch, model_info_error = RepositoryNotFoundError("no response attached"))
    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache",
        lambda repo_id, filename, cache_dir = None, **k: "/cache/model_index.json",
    )
    with pytest.raises(ValueError):
        _assert_base_repo_accessible("unsloth/mystery", None)


def _native_backend_ready(monkeypatch):
    """An SdCppDiffusionBackend whose binary resolution and asset fetch are stubbed out, so
    ``_run_load`` reaches (or fails to reach) the preflight without touching a real engine."""
    from core.inference.sd_cpp_backend import SdCppDiffusionBackend

    b = SdCppDiffusionBackend(engine = None)
    monkeypatch.setattr(
        SdCppDiffusionBackend,
        "_resolve_backend",
        lambda self: ("oneshot", None, types.SimpleNamespace(version = lambda: "master")),
    )
    monkeypatch.setattr(
        SdCppDiffusionBackend, "_set_expected_bytes", lambda self, assets, token: None
    )
    fetched: list = []

    def _fetch(self, assets, token, cancel_event = None):
        fetched.append(assets)
        raise AssertionError("the gated companion must be caught before any byte is fetched")

    monkeypatch.setattr(SdCppDiffusionBackend, "_fetch_assets", _fetch)
    return b, fetched


def test_the_native_load_preflights_its_companion_repos_too(monkeypatch):
    """The plan alone is not enough. images-page.tsx wraps getDiffusionDownloadPlan in a
    try/catch ("No plan (older backend, metadata hiccup): fall back to the load's own download")
    and then calls /images/load regardless, so the 400 the plan raises is swallowed and the load
    would run with no preflight at all -- the user gets the bare Hub token error this exists to
    replace. The diffusers backend runs the same check in both places; the native one must too."""
    from core.inference.diffusion_families import detect_family_for_pick
    from core.inference.sd_cpp_backend import _SdLoading

    gated = "black-forest-labs/FLUX.1-schnell"
    b, fetched = _native_backend_ready(monkeypatch)
    b._loading = _SdLoading(repo_id = "unsloth/FLUX.1-dev-GGUF", base_repo = "")

    class _Api:
        def model_info(
            self,
            repo_id,
            files_metadata = False,
            token = None,
        ):
            return _FakeInfo("auto" if repo_id == gated else False)

    monkeypatch.setattr("huggingface_hub.HfApi", lambda *a, **k: _Api())
    monkeypatch.setattr(
        "huggingface_hub.get_hf_file_metadata",
        lambda url, token = None, **k: (_ for _ in ()).throw(_gated_error()),
    )
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)

    fam = detect_family_for_pick("unsloth/FLUX.1-dev-GGUF", "flux1-dev-Q4_K_M.gguf", None)
    b._run_load(
        repo_id = "unsloth/FLUX.1-dev-GGUF",
        gguf_filename = "flux1-dev-Q4_K_M.gguf",
        base = "",
        fam = fam,
        hf_token = "no-access",
        _load_token = b._load_token,
    )

    assert fetched == []  # refused before the multi-GB pull, not 15 GiB into it
    detail = b._loading.error or ""
    assert gated in detail and "licence" in detail.lower()


def test_the_native_load_probes_the_asset_it_stages_and_honours_the_cache(monkeypatch):
    """Same two properties the plan half has, so the load half cannot drift from it: the probe is
    the file THIS pick stages (a VAE-only repo has no pipeline manifest), and a companion already
    on disk is never refused."""
    from core.inference.diffusion_families import detect_family_for_pick
    from core.inference.sd_cpp_backend import _SdLoading

    gated = "black-forest-labs/FLUX.1-schnell"
    b, fetched = _native_backend_ready(monkeypatch)
    b._loading = _SdLoading(repo_id = "unsloth/FLUX.1-dev-GGUF", base_repo = "")

    class _Api:
        def model_info(
            self,
            repo_id,
            files_metadata = False,
            token = None,
        ):
            return _FakeInfo("auto" if repo_id == gated else False)

    monkeypatch.setattr("huggingface_hub.HfApi", lambda *a, **k: _Api())
    probed: list = []

    def _metadata(
        url,
        token = None,
        **k,
    ):
        probed.append(url)
        raise _gated_error()

    monkeypatch.setattr("huggingface_hub.get_hf_file_metadata", _metadata)
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)

    fam = detect_family_for_pick("unsloth/FLUX.1-dev-GGUF", "flux1-dev-Q4_K_M.gguf", None)
    kwargs = dict(
        repo_id = "unsloth/FLUX.1-dev-GGUF",
        gguf_filename = "flux1-dev-Q4_K_M.gguf",
        base = "",
        fam = fam,
        hf_token = "no-access",
        _load_token = b._load_token,
    )
    b._run_load(**kwargs)
    # The VAE file the load actually opens, not model_index.json, which that repo would not serve.
    assert probed == [f"https://huggingface.co/{gated}/resolve/main/ae.safetensors"]
    assert fetched == []

    # That same VAE already on disk clears the preflight and the load proceeds to the fetch.
    probed.clear()
    b._loading = _SdLoading(repo_id = "unsloth/FLUX.1-dev-GGUF", base_repo = "")
    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache",
        lambda repo_id, filename, cache_dir = None, **k: (
            "/cache/ae.safetensors" if filename == "ae.safetensors" else None
        ),
    )
    b._run_load(**kwargs)
    assert probed == []
    assert len(fetched) == 1  # got past the preflight, exactly as before this check existed
