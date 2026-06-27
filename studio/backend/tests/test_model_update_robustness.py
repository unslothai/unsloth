# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Tests for model-update detection and the GGUF force-download helper.

Covers:
  * GGUF variant listing computes update_available from the already-fetched
    sibling metadata instead of a second Hub call.
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
from hub.services.models import cache_inventory as CI
from hub.services.models import deletion as D
from hub.services.models import gguf_variants as GV


def _variants():
    return [
        SimpleNamespace(
            filename = "model-Q4_K_M.gguf",
            quant = "Q4_K_M",
            display_label = None,
            size_bytes = 1000,
        ),
        SimpleNamespace(
            filename = "model-Q8_0.gguf",
            quant = "Q8_0",
            display_label = None,
            size_bytes = 2000,
        ),
    ]


def _seed_cache(tmp_path, repo_id, blob_ids, gguf_files):
    repo = tmp_path / f"models--{repo_id.replace('/', '--')}"
    snap = repo / "snapshots" / ("a" * 40)
    snap.mkdir(parents = True, exist_ok = True)
    for name, size in gguf_files.items():
        (snap / name).write_bytes(b"\0" * size)
    blobs = repo / "blobs"
    blobs.mkdir(exist_ok = True)
    for b in blob_ids:
        (blobs / b).write_bytes(b"x")
    return repo, snap, blobs


@pytest.fixture
def patch_hub_gguf(monkeypatch):
    """Patch GGUF listing and cache scans for sibling-derived update checks."""

    def _sibling(path: str, size: int, sha = None, *, lfs_dict = False, blob_id = None):
        if lfs_dict:
            lfs = {"sha256": sha} if sha else {}
        else:
            lfs = SimpleNamespace(sha256 = sha) if sha else None
        return SimpleNamespace(rfilename = path, size = size, lfs = lfs, blob_id = blob_id)

    def _repo_info(repo_id: str, repo_path, files: list[tuple[str, str]]):
        return SimpleNamespace(
            repo_id = repo_id,
            repo_type = "model",
            repo_path = repo_path,
            revisions = [
                SimpleNamespace(
                    files = [
                        SimpleNamespace(
                            file_name = name,
                            blob_path = str(repo_path / "blobs" / blob),
                        )
                        for name, blob in files
                    ]
                )
            ],
        )

    def _apply(tmp_path, repo_id: str, *, local_blob: str, remote_sibling):
        with GV._VARIANT_HASH_LOCK:
            GV._VARIANT_HASH_CACHE.clear()
            GV._VARIANT_REQUIREMENT_CACHE.clear()
            GV._VARIANT_REQUIREMENT_NEG_CACHE.clear()
        repo, snap, _blobs = _seed_cache(
            tmp_path,
            repo_id,
            blob_ids = [local_blob],
            gguf_files = {"model-Q4_K_M.gguf": 1000},
        )
        monkeypatch.setattr(
            GV,
            "list_gguf_variants",
            lambda r, hf_token = None: (_variants(), False, [remote_sibling]),
            raising = True,
        )
        monkeypatch.setattr(GV, "iter_hf_cache_snapshots", lambda _repo_id: [snap])
        monkeypatch.setattr(
            CI,
            "all_hf_cache_scans",
            lambda: [
                SimpleNamespace(
                    repos = [
                        _repo_info(
                            repo_id,
                            repo,
                            [("model-Q4_K_M.gguf", local_blob)],
                        )
                    ]
                )
            ],
        )

    return SimpleNamespace(apply = _apply, sibling = _sibling)


def _call(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── GGUF variant update detection ───────────────────────────────


def test_variant_update_check_missing_remote_blob_id_is_not_phantom_update(tmp_path, patch_hub_gguf):
    """Missing sha/blob metadata is unknown, not update_available=True."""
    repo = "unsloth/gemma-3-4b-it-GGUF"
    patch_hub_gguf.apply(
        tmp_path,
        repo,
        local_blob = "oldsha",
        remote_sibling = patch_hub_gguf.sibling("model-Q4_K_M.gguf", 1000, None),
    )
    resp = _call(GV.get_gguf_variants_response(repo))
    assert len(resp.variants) == 2
    q4 = next(v for v in resp.variants if v.quant == "Q4_K_M")
    assert q4.downloaded is True
    assert q4.update_available is False


def test_variant_update_check_detects_update_from_existing_siblings(tmp_path, patch_hub_gguf):
    repo = "unsloth/gemma-3-4b-it-GGUF"
    patch_hub_gguf.apply(
        tmp_path,
        repo,
        local_blob = "oldsha",
        remote_sibling = patch_hub_gguf.sibling("model-Q4_K_M.gguf", 1000, "NEWsha"),
    )
    resp = _call(GV.get_gguf_variants_response(repo))
    q4 = next(v for v in resp.variants if v.quant == "Q4_K_M")
    assert q4.update_available is True


def test_variant_update_check_no_update_when_blob_matches(tmp_path, patch_hub_gguf):
    repo = "unsloth/gemma-3-4b-it-GGUF"
    patch_hub_gguf.apply(
        tmp_path,
        repo,
        local_blob = "samesha",
        remote_sibling = patch_hub_gguf.sibling("model-Q4_K_M.gguf", 1000, "samesha"),
    )
    resp = _call(GV.get_gguf_variants_response(repo))
    q4 = next(v for v in resp.variants if v.quant == "Q4_K_M")
    assert q4.update_available is False


def test_variant_update_check_accepts_lfs_dict_and_blob_id_fallback(tmp_path, patch_hub_gguf):
    repo = "unsloth/gemma-3-4b-it-GGUF"
    patch_hub_gguf.apply(
        tmp_path,
        repo,
        local_blob = "dictsha",
        remote_sibling = patch_hub_gguf.sibling(
            "model-Q4_K_M.gguf",
            1000,
            "dictsha",
            lfs_dict = True,
        ),
    )
    resp = _call(GV.get_gguf_variants_response(repo))
    assert next(v for v in resp.variants if v.quant == "Q4_K_M").update_available is False

    patch_hub_gguf.apply(
        tmp_path,
        repo,
        local_blob = "blobid",
        remote_sibling = patch_hub_gguf.sibling(
            "model-Q4_K_M.gguf",
            1000,
            None,
            blob_id = "blobid",
        ),
    )
    resp = _call(GV.get_gguf_variants_response(repo))
    assert next(v for v in resp.variants if v.quant == "Q4_K_M").update_available is False


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


# ── multi-revision GGUF blob comparison and update reclaim ──
#
# Regression for the phantom "Update available" cue that lingered AFTER a model
# was already updated. A re-download leaves BOTH the old and new revision
# snapshots in the HF cache, so the same gguf file resolves to several blobs.
# The local collection must keep ALL of them (a set per file), and stale hashes
# must be pruned only after the replacement revision verifies.


def _rev(*files):
    return SimpleNamespace(
        files = [
            SimpleNamespace(file_name = name, blob_path = f"/blobs/{blob}") for name, blob in files
        ]
    )


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


def test_reclaim_replaced_gguf_variant_prunes_old_revision_only(monkeypatch, tmp_path):
    """After a verified update, stale same-variant files/blobs are removed while
    the freshly downloaded hash and sibling variants remain cached."""
    repo_id = "org/repo-GGUF"
    repo_path = tmp_path / "models--org--repo-GGUF"
    old_snap = repo_path / "snapshots" / ("a" * 40) / "model-Q4_K_M.gguf"
    new_snap = repo_path / "snapshots" / ("b" * 40) / "model-Q4_K_M.gguf"
    sibling_snap = repo_path / "snapshots" / ("b" * 40) / "model-Q8_0.gguf"
    old_blob = repo_path / "blobs" / "OLDsha"
    new_blob = repo_path / "blobs" / "NEWsha"
    sibling_blob = repo_path / "blobs" / "Q8sha"
    for path, payload in (
        (old_snap, b"old"),
        (new_snap, b"new"),
        (sibling_snap, b"sibling"),
        (old_blob, b"old-blob"),
        (new_blob, b"new-blob"),
        (sibling_blob, b"sibling-blob"),
    ):
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_bytes(payload)

    repo_info = SimpleNamespace(
        repo_id = repo_id,
        repo_type = "model",
        repo_path = repo_path,
        revisions = [
            SimpleNamespace(
                files = [
                    SimpleNamespace(
                        file_name = "model-Q4_K_M.gguf",
                        file_path = str(old_snap),
                        blob_path = str(old_blob),
                    )
                ]
            ),
            SimpleNamespace(
                files = [
                    SimpleNamespace(
                        file_name = "model-Q4_K_M.gguf",
                        file_path = str(new_snap),
                        blob_path = str(new_blob),
                    ),
                    SimpleNamespace(
                        file_name = "model-Q8_0.gguf",
                        file_path = str(sibling_snap),
                        blob_path = str(sibling_blob),
                    ),
                ]
            ),
        ],
    )
    monkeypatch.setattr(
        CI,
        "all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [repo_info])],
    )
    invalidated = []
    monkeypatch.setattr(CI, "invalidate_hf_cache_scans", lambda: invalidated.append(True))

    result = D.reclaim_replaced_gguf_variant(repo_id, "Q4_K_M", frozenset({"NEWsha"}))

    assert result["removed_snapshots"] == 1
    assert result["deleted_blobs"] == 1
    assert result["removed_dirs"] == 1
    assert old_snap.exists() is False
    assert old_snap.parent.exists() is False
    assert old_blob.exists() is False
    assert new_snap.exists() is True
    assert new_blob.exists() is True
    assert sibling_snap.exists() is True
    assert sibling_blob.exists() is True
    assert invalidated == [True]
