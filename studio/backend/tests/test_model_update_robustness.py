# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Tests for model-update detection and the GGUF force-download helper.

Covers:
  * get_gguf_variants degrades to update_available=False when the remote
    update check fails (offline / rate-limit / gated), instead of 500ing.
  * get_gguf_variants reports update_available correctly on success.
  * hf_hub_download_with_xet_fallback(force_download=True) bypasses the
    try_to_load_from_cache cache-first early-return.

The cache "Update" action now runs through the download manager as a normal
managed download (so it shows in the Downloads panel with progress + cancel),
so the old POST /api/models/update endpoint and its tests are gone. Update
*detection* — the "Update available" cue — is still exercised here.
"""

import asyncio
import sys
import types
from types import SimpleNamespace

if "structlog" not in sys.modules:

    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *a, **k: None

    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger = _DummyLogger, get_logger = lambda *a, **k: _DummyLogger()
    )

import pytest
import routes.models as M
from hub.services.models import cache_inventory as CI


def _variants():
    return [
        SimpleNamespace(filename = "model-Q4_K_M.gguf", quant = "Q4_K_M", size_bytes = 1000),
        SimpleNamespace(filename = "model-Q8_0.gguf", quant = "Q8_0", size_bytes = 2000),
    ]


def _seed_cache(tmp_path, repo_id, blob_ids, gguf_files):
    repo = tmp_path / f"models--{repo_id.replace('/', '--')}"
    snap = repo / "snapshots" / ("a" * 40)
    snap.mkdir(parents = True)
    for name, size in gguf_files.items():
        (snap / name).write_bytes(b"\0" * size)
    blobs = repo / "blobs"
    blobs.mkdir()
    for b in blob_ids:
        (blobs / b).write_bytes(b"x")
    return repo


@pytest.fixture
def patch_hf(monkeypatch):
    """Patch the huggingface_hub bits get_gguf_variants pulls in."""

    def _apply(tmp_path, *, paths_info_impl):
        import huggingface_hub as hf

        monkeypatch.setattr(hf.constants, "HF_HUB_CACHE", str(tmp_path), raising = False)
        monkeypatch.setattr(
            M, "list_gguf_variants", lambda r, hf_token = None: (_variants(), False), raising = True
        )
        monkeypatch.setattr(M, "is_local_path", lambda p: False, raising = False)
        monkeypatch.setattr(hf, "try_to_load_from_cache", lambda **k: None, raising = False)
        monkeypatch.setattr(hf, "get_paths_info", paths_info_impl, raising = False)

    return _apply


def _call(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── get_gguf_variants update detection ───────────────────────────────


def test_update_check_failure_does_not_500(tmp_path, patch_hf):
    """A failed update check must not break the variant listing."""
    repo = "unsloth/gemma-3-4b-it-GGUF"
    _seed_cache(tmp_path, repo, blob_ids = ["oldsha"], gguf_files = {"model-Q4_K_M.gguf": 1000})

    def boom(**kwargs):
        raise RuntimeError("429 Too Many Requests / offline")

    patch_hf(tmp_path, paths_info_impl = boom)
    resp = _call(M.get_gguf_variants(repo_id = repo, hf_token = None, current_subject = "t"))
    assert len(resp.variants) == 2
    q4 = next(v for v in resp.variants if v.quant == "Q4_K_M")
    assert q4.downloaded is True
    assert q4.update_available is False


def test_update_check_detects_update(tmp_path, patch_hf):
    repo = "unsloth/gemma-3-4b-it-GGUF"
    _seed_cache(tmp_path, repo, blob_ids = ["oldsha"], gguf_files = {"model-Q4_K_M.gguf": 1000})

    def ok(
        repo_id,
        paths,
        token = None,
    ):
        return [
            SimpleNamespace(
                path = "model-Q4_K_M.gguf", lfs = SimpleNamespace(sha256 = "NEWsha"), blob_id = None
            )
        ]

    patch_hf(tmp_path, paths_info_impl = ok)
    resp = _call(M.get_gguf_variants(repo_id = repo, hf_token = None, current_subject = "t"))
    q4 = next(v for v in resp.variants if v.quant == "Q4_K_M")
    assert q4.update_available is True


def test_update_check_no_update_when_blob_matches(tmp_path, patch_hf):
    repo = "unsloth/gemma-3-4b-it-GGUF"
    _seed_cache(tmp_path, repo, blob_ids = ["samesha"], gguf_files = {"model-Q4_K_M.gguf": 1000})

    def ok(
        repo_id,
        paths,
        token = None,
    ):
        return [
            SimpleNamespace(
                path = "model-Q4_K_M.gguf", lfs = SimpleNamespace(sha256 = "samesha"), blob_id = None
            )
        ]

    patch_hf(tmp_path, paths_info_impl = ok)
    resp = _call(M.get_gguf_variants(repo_id = repo, hf_token = None, current_subject = "t"))
    q4 = next(v for v in resp.variants if v.quant == "Q4_K_M")
    assert q4.update_available is False


# ── hf_hub_download_with_xet_fallback force_download bypass (X2/F2) ───


def test_force_download_bypasses_cache_first_early_return(monkeypatch):
    """force_download=True skips the try_to_load_from_cache early-return and
    proceeds to the real download path; force_download=False returns the cached
    path without ever attempting a download (X2/F2)."""
    import huggingface_hub as hf
    import utils.hf_xet_fallback as X

    cached_path = "/cache/blob/cached.gguf"

    # Pretend the blob IS cached on disk (try_to_load_from_cache is imported
    # inside the function from huggingface_hub, and os.path.exists must agree).
    monkeypatch.setattr(hf, "try_to_load_from_cache", lambda *a, **k: cached_path, raising = False)
    monkeypatch.setattr(X.os.path, "exists", lambda p: True, raising = False)

    attempts = []

    def fake_attempt(repo_id, filename, token, **kwargs):
        attempts.append(
            {"repo_id": repo_id, "filename": filename, "force": kwargs.get("force_download")}
        )
        return ("ok", "/freshly/downloaded/path")

    monkeypatch.setattr(X, "_run_download_attempt", fake_attempt, raising = True)

    # force_download=False: cache-first early-return, no download attempt.
    out = X.hf_hub_download_with_xet_fallback(
        "unsloth/repo", "model.gguf", token = None, force_download = False
    )
    assert out == cached_path
    assert attempts == []  # never reached the real download

    # force_download=True: bypass the early-return, run the real download.
    out2 = X.hf_hub_download_with_xet_fallback(
        "unsloth/repo", "model.gguf", token = None, force_download = True
    )
    assert out2 == "/freshly/downloaded/path"
    assert len(attempts) == 1
    assert attempts[0]["force"] is True


# ── repo_update_status_response: multi-revision GGUF blob comparison ──
#
# Regression for the phantom "Update available" cue that lingered AFTER a model
# was already updated. A re-download leaves BOTH the old and new revision
# snapshots in the HF cache, so the same gguf file resolves to several blobs.
# The local collection must keep ALL of them (a set per file), and the
# remote-vs-local diff must treat the file as current when the remote (`main`)
# blob is present in ANY cached revision, mirroring routes/models.py's
# `cached_blob_ids` membership test. The old code collapsed the revisions to one
# arbitrary blob (`setdefault`) and compared it with `!=`, so an up-to-date model
# kept reporting an update.


def _rev(*files):
    return SimpleNamespace(
        files = [SimpleNamespace(file_name = name, blob_path = f"/blobs/{blob}") for name, blob in files]
    )


def _paths_info(path, sha):
    def _impl(
        repo_id,
        paths,
        token = None,
    ):
        return [SimpleNamespace(path = path, lfs = SimpleNamespace(sha256 = sha), blob_id = None)]

    return _impl


def test_repo_gguf_blob_map_collects_all_revision_blobs():
    """Every cached revision's blob for a gguf file is kept as a set, not
    collapsed to one arbitrary blob."""
    repo_info = SimpleNamespace(
        revisions = [
            _rev(("lfm2-350m-q4_k_m.gguf", "OLDsha")),
            _rev(("lfm2-350m-q4_k_m.gguf", "NEWsha")),
        ]
    )
    assert CI._repo_gguf_blob_map(repo_info) == {"lfm2-350m-q4_k_m.gguf": {"OLDsha", "NEWsha"}}


def test_gguf_remote_update_false_when_main_blob_in_any_local_revision(monkeypatch):
    """Remote (main) blob present among the local revisions => no update, even
    though an older revision's blob differs."""
    import huggingface_hub as hf

    local_blobs = {"lfm2-350m-q4_k_m.gguf": {"OLDsha", "NEWsha"}}
    monkeypatch.setattr(
        hf, "get_paths_info", _paths_info("lfm2-350m-q4_k_m.gguf", "NEWsha"), raising = False
    )
    assert CI._gguf_remote_update("repo/x", local_blobs, None) is False


def test_gguf_remote_update_true_when_main_blob_absent(monkeypatch):
    """Remote (main) blob not among the local blobs => update available."""
    import huggingface_hub as hf

    local_blobs = {"lfm2-350m-q4_k_m.gguf": {"OLDsha"}}
    monkeypatch.setattr(
        hf, "get_paths_info", _paths_info("lfm2-350m-q4_k_m.gguf", "NEWsha"), raising = False
    )
    assert CI._gguf_remote_update("repo/x", local_blobs, None) is True
