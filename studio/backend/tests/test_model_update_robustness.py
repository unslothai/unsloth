# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Tests for model-update detection and the GGUF force-download helper.

Covers:
  * GGUF variant listing computes update_available from the already-fetched
    sibling metadata instead of a second Hub call.
  * hf_hub_download_with_xet_fallback forwards force_download through the shim to the
    shared unsloth_zoo helper (which owns the cache-first early-return and its bypass).

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

    def _sibling(
        path: str,
        size: int,
        sha = None,
        *,
        lfs_dict = False,
        blob_id = None,
    ):
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


def test_variant_update_check_missing_remote_blob_id_is_not_phantom_update(
    tmp_path, patch_hub_gguf
):
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


@pytest.mark.parametrize(
    ("companion_path", "has_vision"),
    [
        ("mmproj-F16.gguf", True),
        ("mtp-drafter-Q8_0.gguf", False),
    ],
)
def test_variant_update_check_detects_companion_only_update(
    monkeypatch, tmp_path, patch_hub_gguf, companion_path, has_vision
):
    repo_id = "unsloth/gemma-4-GGUF"
    with GV._VARIANT_HASH_LOCK:
        GV._VARIANT_HASH_CACHE.clear()
        GV._VARIANT_REQUIREMENT_CACHE.clear()
        GV._VARIANT_REQUIREMENT_NEG_CACHE.clear()
    repo, snap, _blobs = _seed_cache(
        tmp_path,
        repo_id,
        blob_ids = ["mainsha", "old-companion"],
        gguf_files = {
            "model-Q4_K_M.gguf": 1000,
            companion_path: 100,
        },
    )
    siblings = [
        patch_hub_gguf.sibling("model-Q4_K_M.gguf", 1000, "mainsha"),
        patch_hub_gguf.sibling(companion_path, 100, "new-companion"),
    ]
    monkeypatch.setattr(
        GV,
        "list_gguf_variants",
        lambda r, hf_token = None: (_variants(), has_vision, siblings),
        raising = True,
    )
    monkeypatch.setattr(GV, "iter_hf_cache_snapshots", lambda _repo_id: [snap])
    monkeypatch.setattr(
        CI,
        "all_hf_cache_scans",
        lambda: [
            SimpleNamespace(
                repos = [
                    SimpleNamespace(
                        repo_id = repo_id,
                        repo_type = "model",
                        repo_path = repo,
                        revisions = [
                            SimpleNamespace(
                                files = [
                                    SimpleNamespace(
                                        file_name = "model-Q4_K_M.gguf",
                                        blob_path = str(repo / "blobs" / "mainsha"),
                                    ),
                                    SimpleNamespace(
                                        file_name = companion_path,
                                        blob_path = str(repo / "blobs" / "old-companion"),
                                    ),
                                ]
                            )
                        ],
                    )
                ]
            )
        ],
    )

    resp = _call(GV.get_gguf_variants_response(repo_id))
    q4 = next(v for v in resp.variants if v.quant == "Q4_K_M")

    assert q4.downloaded is True
    assert q4.update_available is True


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


def test_cached_model_scan_keeps_local_safetensors_repo(monkeypatch, tmp_path):
    repo_path = tmp_path / "models--Org--SafeTensorRepo"
    repo = SimpleNamespace(
        repo_id = "Org/SafeTensorRepo",
        repo_type = "model",
        repo_path = repo_path,
        revisions = [
            SimpleNamespace(
                files = [
                    SimpleNamespace(
                        file_name = "config.json",
                        size_on_disk = 10,
                        blob_path = None,
                    ),
                    SimpleNamespace(
                        file_name = "model.safetensors",
                        size_on_disk = 100,
                        blob_path = str(repo_path / "blobs" / "modelsha"),
                        blob_last_modified = 3_000.0,
                    ),
                ]
            )
        ],
    )
    monkeypatch.setattr(
        CI,
        "all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [repo])],
    )
    monkeypatch.setattr(
        CI.hf_cache_scan,
        "is_snapshot_partial",
        lambda *args, **kwargs: False,
    )

    rows = CI._scan_cached_models()

    assert len(rows) == 1
    assert rows[0]["repo_id"] == "Org/SafeTensorRepo"
    assert rows[0]["model_format"] == "safetensors"
    assert rows[0]["size_bytes"] == 100
    assert rows[0]["last_modified"] == 3_000.0


def test_cached_gguf_scan_keeps_download_timestamp(monkeypatch, tmp_path):
    repo_path = tmp_path / "models--Org--GgufRepo"
    repo = SimpleNamespace(
        repo_id = "Org/GgufRepo",
        repo_type = "model",
        repo_path = repo_path,
        revisions = [
            SimpleNamespace(
                files = [
                    SimpleNamespace(
                        file_name = "model-Q4_K_M.gguf",
                        size_on_disk = 100,
                        blob_path = None,
                        blob_last_modified = 5_000.0,
                    ),
                ]
            )
        ],
    )
    monkeypatch.setattr(
        CI,
        "all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [repo])],
    )
    monkeypatch.setattr(
        CI.hf_cache_scan,
        "is_gguf_repo_partial",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        CI,
        "_gguf_variant_state_summary",
        lambda _repo_id: (False, 0),
    )

    rows = CI._scan_cached_gguf()

    assert len(rows) == 1
    assert rows[0]["repo_id"] == "Org/GgufRepo"
    assert rows[0]["model_format"] == "gguf"
    assert rows[0]["size_bytes"] == 100
    assert rows[0]["last_modified"] == 5_000.0


def test_cached_model_scan_hides_custom_whisper_repo(monkeypatch, tmp_path):
    repo_path = tmp_path / "models--Org--CustomWhisper"
    snapshot = repo_path / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents = True)
    (snapshot / "config.json").write_text(
        '{"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]}'
    )
    repo = SimpleNamespace(
        repo_id = "Org/CustomWhisper",
        repo_type = "model",
        repo_path = repo_path,
        revisions = [
            SimpleNamespace(
                files = [
                    SimpleNamespace(
                        file_name = "config.json",
                        size_on_disk = 10,
                        blob_path = None,
                    ),
                    SimpleNamespace(
                        file_name = "model.safetensors",
                        size_on_disk = 100,
                        blob_path = str(repo_path / "blobs" / "modelsha"),
                    ),
                ]
            )
        ],
    )
    monkeypatch.setattr(
        CI,
        "all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [repo])],
    )
    monkeypatch.setattr(
        CI,
        "_cached_model_snapshot_path",
        lambda _repo_path: snapshot,
    )
    monkeypatch.setattr(
        CI.hf_cache_scan,
        "is_snapshot_partial",
        lambda *args, **kwargs: False,
    )

    assert CI._scan_cached_models() == []


# ── hf_hub_download_with_xet_fallback force_download bypass (X2/F2) ───


def test_force_download_is_forwarded_through_the_shim(monkeypatch):
    """The shim's contract is to forward force_download unchanged to the shared helper (which owns the
    cache-first early-return and bypass). Verify both False and True reach it (X2/F2)."""
    import utils.hf_xet_fallback as X

    seen = []

    def fake_shared(repo_id, filename, token, **kwargs):
        seen.append(kwargs.get("force_download"))
        return "/downloaded/path"

    monkeypatch.setattr(X, "_shared_hf_hub_download_with_xet_fallback", fake_shared, raising = True)

    X.hf_hub_download_with_xet_fallback(
        "unsloth/repo", "model.gguf", token = None, force_download = False
    )
    X.hf_hub_download_with_xet_fallback(
        "unsloth/repo", "model.gguf", token = None, force_download = True
    )
    assert seen == [False, True]  # the shim forwards force_download to the shared helper unchanged


# ── multi-revision GGUF blob comparison and update reclaim ──
#
# Regression for the phantom "Update available" cue that lingered AFTER a model
# was already updated. A re-download leaves BOTH the old and new revision
# snapshots in the HF cache, so the same gguf file resolves to several blobs.
# The local collection must keep ALL of them (a set per file), and stale hashes
# must be pruned only after the replacement revision verifies.


def _rev(*files):
    return SimpleNamespace(
        files = [SimpleNamespace(file_name = name, blob_path = f"/blobs/{blob}") for name, blob in files]
    )


def test_repo_gguf_blob_map_collects_all_revision_blobs():
    """Every cached revision's blob for a gguf file is kept as a set, not
    collapsed to one arbitrary blob."""
    repo_info = SimpleNamespace(
        repo_path = "/",  # real blobs live at <repo_path>/blobs/<etag>
        revisions = [
            _rev(("lfm2-350m-q4_k_m.gguf", "OLDsha")),
            _rev(("lfm2-350m-q4_k_m.gguf", "NEWsha")),
        ],
    )
    assert CI._repo_gguf_blob_map(repo_info) == {"lfm2-350m-q4_k_m.gguf": {"OLDsha", "NEWsha"}}


# ── no-symlink (Windows without Developer Mode) GGUF update detection ──
#
# Regression for the phantom "Update available" that NEVER clears (#7060). Without
# the symlink privilege, hf_hub_download MOVES the blob into snapshots/ instead of
# symlinking it, so blobs/ is empty and scan_cache_dir reports blob_path = the
# snapshot file. Its name is the FILENAME, not an etag, so a remote-vs-local sha256
# comparison can never match and every cached GGUF reports an update forever --
# which re-downloading cannot fix, since the same file is rewritten with no blob.


def _rev_no_symlink(*files):
    """A revision whose GGUFs were MOVED into snapshots/ (no blobs/ entry)."""
    return SimpleNamespace(
        files = [
            SimpleNamespace(
                file_name = name,
                blob_path = f"/hf/models--org--repo/snapshots/{'a' * 40}/{name}",
                size_on_disk = size,
            )
            for name, size in files
        ]
    )


def _requirement(*expected):
    from hub.utils.download_manifest import ExpectedFile
    from hub.utils.gguf_plan import GgufVariantPlan

    expected_files = tuple(
        ExpectedFile(path = path, size = size, sha256 = sha) for path, size, sha in expected
    )
    return GgufVariantPlan(
        main_filenames = frozenset(e.path for e in expected_files),
        target_filenames = tuple(e.path for e in expected_files),
        main_hashes = frozenset(e.sha256 for e in expected_files if e.sha256),
        required_hashes = frozenset(e.sha256 for e in expected_files if e.sha256),
        companion_hashes = frozenset(),
        mmproj_filenames = frozenset(),
        mmproj_hashes = frozenset(),
        expected_files = expected_files,
        main_size_bytes = sum(e.size for e in expected_files),
        download_size_bytes = sum(e.size for e in expected_files),
    )


def test_repo_gguf_blob_map_uses_size_identity_when_cache_has_no_blob():
    """A snapshot-resident GGUF (no blobs/ entry) must NOT be recorded under its
    filename as if that were a hash -- it gets a size identity instead."""
    repo_info = SimpleNamespace(
        repo_path = "/hf/models--org--repo",
        revisions = [_rev_no_symlink(("model-Q4_K_M.gguf", 4096))],
    )

    assert CI._repo_gguf_blob_map(repo_info) == {
        "model-Q4_K_M.gguf": {CI.local_size_identity(4096)}
    }


def test_repo_gguf_blob_map_skips_snapshot_file_with_unknown_size():
    """No blob and no readable size means no identity at all, rather than a
    filename masquerading as a hash."""
    repo_info = SimpleNamespace(
        repo_path = "/hf/models--org--repo",
        revisions = [_rev_no_symlink(("model-Q4_K_M.gguf", 0))],
    )

    assert CI._repo_gguf_blob_map(repo_info) == {}


def test_repo_gguf_blob_map_ignores_repo_blobs_subdir_on_no_symlink():
    """A repo that ships a GGUF under its own blobs/ subdir lands at
    snapshots/<rev>/blobs/model.gguf on a no-symlink cache. Its parent is named
    'blobs' but it is NOT the cache blob store, so it gets a size identity rather
    than having its filename recorded as a hash (which would show a phantom update)."""
    repo_path = "/hf/models--org--repo"
    repo_info = SimpleNamespace(
        repo_path = repo_path,
        revisions = [
            SimpleNamespace(
                files = [
                    SimpleNamespace(
                        file_name = "model-Q4_K_M.gguf",
                        blob_path = f"{repo_path}/snapshots/{'a' * 40}/blobs/model-Q4_K_M.gguf",
                        size_on_disk = 4096,
                    )
                ]
            )
        ],
    )

    assert CI._repo_gguf_blob_map(repo_info) == {
        "model-Q4_K_M.gguf": {CI.local_size_identity(4096)}
    }


def test_no_symlink_cache_matching_remote_size_reports_no_update():
    """The #7060 repro: a GGUF stored directly in snapshots/ whose size matches the
    remote is CURRENT, and must not show a phantom 'update available'."""
    local_blobs = {"model-Q4_K_M.gguf": {CI.local_size_identity(4096)}}
    requirement = _requirement(("model-Q4_K_M.gguf", 4096, "REMOTEsha256"))

    assert (
        GV._variant_update_available_from_requirement(local_blobs, requirement, "Q4_K_M") is False
    )


def test_no_symlink_cache_with_different_remote_size_still_reports_update():
    """A genuine upstream change is still detected in the no-symlink layout."""
    local_blobs = {"model-Q4_K_M.gguf": {CI.local_size_identity(4096)}}
    requirement = _requirement(("model-Q4_K_M.gguf", 8192, "REMOTEsha256"))

    assert GV._variant_update_available_from_requirement(local_blobs, requirement, "Q4_K_M") is True


def test_symlinked_cache_with_stale_blob_still_reports_update():
    """The blob-hash path is untouched: a real blob that does not match the remote
    sha256 is still stale, and a size-identity fallback must not rescue it."""
    local_blobs = {"model-Q4_K_M.gguf": {"OLDsha"}}
    requirement = _requirement(("model-Q4_K_M.gguf", 4096, "NEWsha"))

    assert GV._variant_update_available_from_requirement(local_blobs, requirement, "Q4_K_M") is True


def test_symlinked_cache_with_current_blob_reports_no_update():
    """The blob-hash path is untouched: a matching blob is current."""
    local_blobs = {"model-Q4_K_M.gguf": {"OLDsha", "NEWsha"}}
    requirement = _requirement(("model-Q4_K_M.gguf", 4096, "NEWsha"))

    assert (
        GV._variant_update_available_from_requirement(local_blobs, requirement, "Q4_K_M") is False
    )


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


def test_reclaim_replaced_gguf_variant_keeps_no_symlink_current_file(monkeypatch, tmp_path):
    """No-symlink cache (Windows without Developer Mode): the moved GGUF lives
    directly in snapshots/ and blobs/ is empty, so scan_cache_dir reports
    blob_path == the snapshot file and its name is the FILENAME, not an etag.
    Reclaim must NOT mistake that filename for a stale hash and delete the
    freshly-downloaded current file."""
    repo_id = "org/repo-GGUF"
    repo_path = tmp_path / "models--org--repo-GGUF"
    snap = repo_path / "snapshots" / ("a" * 40) / "model-Q4_K_M.gguf"
    snap.parent.mkdir(parents = True, exist_ok = True)
    snap.write_bytes(b"current-download")
    (repo_path / "blobs").mkdir(parents = True, exist_ok = True)  # empty: moved, not linked

    repo_info = SimpleNamespace(
        repo_id = repo_id,
        repo_type = "model",
        repo_path = repo_path,
        revisions = [
            SimpleNamespace(
                files = [
                    SimpleNamespace(
                        file_name = "model-Q4_K_M.gguf",
                        file_path = str(snap),
                        blob_path = str(snap),  # no-symlink: blob_path == the snapshot file
                    )
                ]
            )
        ],
    )
    monkeypatch.setattr(CI, "all_hf_cache_scans", lambda: [SimpleNamespace(repos = [repo_info])])
    monkeypatch.setattr(CI, "invalidate_hf_cache_scans", lambda: None)

    result = D.reclaim_replaced_gguf_variant(repo_id, "Q4_K_M", frozenset({"REMOTEsha256"}))

    assert snap.exists() is True  # the current file must survive
    assert result["removed_snapshots"] == 0
    assert result["deleted_blobs"] == 0


def _mmproj_repo(*file_names: str):
    return SimpleNamespace(
        revisions = [SimpleNamespace(files = [SimpleNamespace(file_name = n) for n in file_names])]
    )


def test_repo_has_mmproj_requires_gguf_projector():
    # A non-GGUF sidecar whose name merely contains "mmproj" must NOT mark the
    # repo vision-capable; the runtime's projector detection is GGUF-only.
    assert CI._repo_has_mmproj(_mmproj_repo("model-Q4_K_M.gguf", "mmproj_config.json")) is False
    assert CI._repo_has_mmproj(_mmproj_repo("model-Q4_K_M.gguf", "README-mmproj.md")) is False
    # A real GGUF projector still marks the repo vision-capable.
    assert CI._repo_has_mmproj(_mmproj_repo("model-Q4_K_M.gguf", "mmproj-F16.gguf")) is True
