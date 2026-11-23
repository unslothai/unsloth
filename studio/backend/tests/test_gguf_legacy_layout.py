# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A GGUF repo that moves its previous layout into Unsloth_Layout/ and ships new bytes under the old
root names: nothing the user already has is listed twice, re-downloaded or flagged as an update."""

from __future__ import annotations

import os
import threading
from types import SimpleNamespace

import pytest

import core.inference.llama_cpp as llama_cpp
import hub.services.models.gguf_variants as GV
from hub.utils.download_manifest import ExpectedFile
from hub.utils.gguf_plan import build_gguf_variant_plans
from utils.gguf_legacy_layout import (
    cached_blob_sha256,
    is_legacy_layout_path,
    legacy_layout_alias,
)

OLD = "a" * 64
NEW = "b" * 64
DOLD = "c" * 64
DNEW = "d" * 64
MM = "e" * 64


def _sib(
    name,
    sha,
    size = 100,
):
    return SimpleNamespace(
        rfilename = name,
        size = size,
        blob_id = None,
        lfs = SimpleNamespace(sha256 = sha, size = size),
    )


MOVED = [
    _sib("Model-UD-IQ1_S.gguf", NEW),
    _sib("dspark-Model-Q8_0.gguf", DNEW),
    _sib("mmproj-F16.gguf", MM),
    _sib("Unsloth_Layout/Model-UD-IQ1_S.gguf", OLD),
    _sib("Unsloth_Layout/dspark-Model-Q8_0.gguf", DOLD),
]


def test_alias_names_the_root_file_and_only_for_the_legacy_folder():
    assert legacy_layout_alias("Unsloth_Layout/Model-UD-IQ1_S.gguf") == "Model-UD-IQ1_S.gguf"
    assert legacy_layout_alias("unsloth_layout\\sub/x.gguf") == "sub/x.gguf"
    assert legacy_layout_alias("Model-UD-IQ1_S.gguf") is None
    assert legacy_layout_alias("Unsloth_Layout") is None
    assert not is_legacy_layout_path("Mainline_Support/Model-UD-IQ1_S.gguf")


def test_cached_blob_sha256_reads_the_lfs_blob_name(tmp_path):
    blob = tmp_path / "blobs" / OLD
    blob.parent.mkdir()
    blob.write_bytes(b"x")
    snap = tmp_path / "snapshots" / ("1" * 40)
    snap.mkdir(parents = True)
    (snap / "m.gguf").symlink_to(os.path.relpath(blob, snap))
    (snap / "plain.gguf").write_bytes(b"x")
    assert cached_blob_sha256(str(snap / "m.gguf")) == OLD
    assert cached_blob_sha256(str(snap / "plain.gguf")) is None


def test_plans_skip_the_legacy_folder_and_record_published_and_alternate_hashes():
    plans = build_gguf_variant_plans(MOVED)
    assert set(plans) == {"ud-iq1_s"}
    plan = plans["ud-iq1_s"]
    assert plan.main_filenames == frozenset({"Model-UD-IQ1_S.gguf"})
    assert plan.published_hashes == frozenset({OLD, NEW, DOLD, DNEW, MM})
    assert plan.alternate_main_hashes == frozenset({OLD})


def test_plans_without_a_legacy_folder_are_unchanged_apart_from_published():
    plans = build_gguf_variant_plans([_sib("Model-UD-IQ1_S.gguf", NEW)])
    assert plans["ud-iq1_s"].alternate_main_hashes == frozenset()
    assert plans["ud-iq1_s"].published_hashes == frozenset({NEW})


def _requirement(published, path = "Model-UD-IQ1_S.gguf"):
    return SimpleNamespace(
        expected_files = (ExpectedFile(path = path, size = 100, sha256 = NEW),),
        published_hashes = frozenset(published),
    )


def test_a_cached_copy_of_bytes_the_repo_still_publishes_is_not_an_update():
    local = {"Model-UD-IQ1_S.gguf": {OLD}}
    assert (
        GV._variant_update_available_from_requirement(local, _requirement({NEW, OLD}), "UD-IQ1_S")
        is False
    )


def test_a_cached_copy_the_repo_no_longer_publishes_is_still_an_update():
    local = {"Model-UD-IQ1_S.gguf": {OLD}}
    assert (
        GV._variant_update_available_from_requirement(local, _requirement({NEW}), "UD-IQ1_S")
        is True
    )


def test_a_cached_legacy_folder_copy_stands_in_for_the_root_file():
    local = {"Unsloth_Layout/Model-UD-IQ1_S.gguf": {OLD}}
    assert (
        GV._variant_update_available_from_requirement(local, _requirement({NEW, OLD}), "UD-IQ1_S")
        is False
    )


def test_hub_variant_listing_hides_the_legacy_folder(monkeypatch):
    import hub.utils.gguf as G

    monkeypatch.setattr(G, "_env_offline", lambda: False)
    monkeypatch.setattr(G, "_hub_model_info", lambda repo, token: SimpleNamespace(siblings = MOVED))
    variants, has_vision, _siblings = G.list_gguf_variants("org/Model-GGUF")
    assert [v.filename for v in variants] == ["Model-UD-IQ1_S.gguf"]
    assert has_vision


def test_loader_resolution_prefers_the_current_layout_and_falls_back_to_a_cached_legacy_copy():
    both = ["Model-UD-IQ1_S.gguf", "Unsloth_Layout/Model-UD-IQ1_S.gguf"]
    assert llama_cpp._gguf_files_for_variant(both, "UD-IQ1_S") == ["Model-UD-IQ1_S.gguf"]
    only_legacy = ["Unsloth_Layout/Model-UD-IQ1_S.gguf"]
    assert llama_cpp._gguf_files_for_variant(only_legacy, "UD-IQ1_S") == only_legacy


def test_mmproj_pick_prefers_the_current_layout():
    names = ["Unsloth_Layout/mmproj-F16.gguf", "mmproj-F16.gguf"]
    assert llama_cpp._pick_mmproj(names) == "mmproj-F16.gguf"


class _NoWaitEvent(threading.Event):
    def wait(self, timeout = None):
        return self.is_set()


def _snapshot(repo_dir, rev, files):
    snap = repo_dir / "snapshots" / rev
    snap.mkdir(parents = True)
    for name, sha in files.items():
        blob = repo_dir / "blobs" / sha
        blob.parent.mkdir(parents = True, exist_ok = True)
        if not blob.exists():
            blob.write_bytes(b"gguf")
        link = snap / name
        link.parent.mkdir(parents = True, exist_ok = True)
        link.symlink_to(os.path.relpath(blob, link.parent))
    return snap


@pytest.fixture
def two_snapshots(tmp_path, monkeypatch):
    """Weight fetched at revision 1, drafter added and fetched at revision 2."""
    repo_dir = tmp_path / "models--org--Model-GGUF"
    weight = _snapshot(repo_dir, "1" * 40, {"Model-UD-IQ1_S.gguf": OLD}) / "Model-UD-IQ1_S.gguf"
    _snapshot(repo_dir, "2" * 40, {"dspark-Model-Q8_0.gguf": DOLD})
    downloads = []
    monkeypatch.setattr(llama_cpp, "_hub_download_in_flight", lambda repo: False)
    monkeypatch.setattr(llama_cpp, "_hf_env_offline", lambda: False)
    monkeypatch.setattr(
        llama_cpp, "_companion_repo_files", lambda repo, token: [s.rfilename for s in MOVED]
    )

    def _download(repo, name, token, **kwargs):
        downloads.append(name)
        return f"/hub/{repo}/{name}"

    monkeypatch.setattr(llama_cpp, "hf_hub_download_with_xet_fallback", _download)
    llama_cpp._published_hashes_cache.clear()
    return SimpleNamespace(weight = str(weight), downloads = downloads)


def _backend():
    backend = llama_cpp.LlamaCppBackend.__new__(llama_cpp.LlamaCppBackend)
    backend._cancel_event = threading.Event()
    return backend


def _lookup(weight):
    return _backend()._download_companion_gguf(
        hf_repo = "org/Model-GGUF",
        hf_token = None,
        pick = llama_cpp._pick_dspark,
        label = "DSpark drafter",
        cancel_event = _NoWaitEvent(),
        near_path = weight,
    )


def test_a_drafter_cached_at_another_revision_and_still_published_is_reused(
    two_snapshots, monkeypatch
):
    monkeypatch.setattr(
        llama_cpp, "_published_lfs_hashes", lambda repo, token: frozenset({NEW, DNEW, OLD, DOLD})
    )
    found = _lookup(two_snapshots.weight)
    assert found is not None and found.endswith("dspark-Model-Q8_0.gguf")
    assert "2" * 40 in found
    assert two_snapshots.downloads == []


def test_a_cached_drafter_the_repo_dropped_is_not_reused_online(two_snapshots, monkeypatch):
    monkeypatch.setattr(
        llama_cpp, "_published_lfs_hashes", lambda repo, token: frozenset({NEW, DNEW})
    )
    found = _lookup(two_snapshots.weight)
    assert two_snapshots.downloads == ["dspark-Model-Q8_0.gguf"]
    assert found == "/hub/org/Model-GGUF/dspark-Model-Q8_0.gguf"


def test_an_unanswered_published_listing_falls_back_to_the_existing_flow(
    two_snapshots, monkeypatch
):
    monkeypatch.setattr(llama_cpp, "_published_lfs_hashes", lambda repo, token: None)
    _lookup(two_snapshots.weight)
    assert two_snapshots.downloads == ["dspark-Model-Q8_0.gguf"]


def test_published_hashes_are_read_from_the_repo_tree_and_cached(monkeypatch):
    calls = []

    class _Api:
        def __init__(self, token = None):
            pass

        def list_repo_tree(
            self,
            repo,
            recursive = False,
        ):
            calls.append((repo, recursive))
            return [
                SimpleNamespace(path = "a.gguf", lfs = SimpleNamespace(sha256 = OLD)),
                SimpleNamespace(path = "README.md", lfs = None),
            ]

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    llama_cpp._published_hashes_cache.clear()
    assert llama_cpp._published_lfs_hashes("org/Model-GGUF", None) == frozenset({OLD})
    assert llama_cpp._published_lfs_hashes("org/Model-GGUF", None) == frozenset({OLD})
    assert calls == [("org/Model-GGUF", True)]


def test_dspark_pick_prefers_the_current_layout_whatever_the_listing_order():
    names = ["Unsloth_Layout/dspark-Model-Q8_0.gguf", "dspark-Model-Q8_0.gguf"]
    assert llama_cpp._pick_dspark(names) == "dspark-Model-Q8_0.gguf"
    assert llama_cpp._pick_dspark(list(reversed(names))) == "dspark-Model-Q8_0.gguf"
    assert llama_cpp._pick_dspark(names[:1]) == names[0]


def _revision_not_found():
    import httpx
    from huggingface_hub.errors import RevisionNotFoundError

    request = httpx.Request("POST", "https://huggingface.co/api/models/org/Model-GGUF/paths-info/x")
    return RevisionNotFoundError(
        "404 Revision Not Found", response = httpx.Response(404, request = request)
    )


@pytest.fixture
def squashed_snapshot(tmp_path, monkeypatch):
    """A complete cached quant whose revision the Hub no longer has (history super-squashed)."""
    repo_dir = tmp_path / "models--org--Model-GGUF"
    snap = _snapshot(
        repo_dir,
        "1" * 40,
        {
            "Q4/Model-Q4_K_M-00001-of-00002.gguf": OLD,
            "Q4/Model-Q4_K_M-00002-of-00002.gguf": NEW,
        },
    )
    calls = []

    def _paths_info(
        repo,
        paths,
        *,
        revision = None,
        token = None,
        **kwargs,
    ):
        calls.append(revision)
        raise _revision_not_found()

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "get_paths_info", _paths_info)
    main = "Q4/Model-Q4_K_M-00001-of-00002.gguf"
    candidate = (str(snap / main), main, ["Q4/Model-Q4_K_M-00002-of-00002.gguf"], snap)
    return SimpleNamespace(snap = snap, candidate = candidate, calls = calls, repo_dir = repo_dir)


def test_a_complete_copy_whose_revision_404s_stays_usable(squashed_snapshot):
    assert llama_cpp._cached_candidate_matches_revision_size(
        "org/Model-GGUF", squashed_snapshot.candidate, None
    )
    assert squashed_snapshot.calls == ["1" * 40]


def test_a_copy_with_an_incomplete_blob_is_not_reused_when_the_revision_404s(squashed_snapshot):
    (squashed_snapshot.repo_dir / "blobs" / (NEW + ".incomplete")).write_bytes(b"part")
    assert not llama_cpp._cached_candidate_matches_revision_size(
        "org/Model-GGUF", squashed_snapshot.candidate, None
    )


def test_a_copy_missing_a_shard_is_not_reused_when_the_hub_is_unreachable(
    squashed_snapshot, monkeypatch
):
    import huggingface_hub

    def _offline(*args, **kwargs):
        raise OSError("network unreachable")

    monkeypatch.setattr(huggingface_hub, "get_paths_info", _offline)
    (squashed_snapshot.snap / "Q4" / "Model-Q4_K_M-00002-of-00002.gguf").unlink()
    assert not llama_cpp._cached_candidate_matches_revision_size(
        "org/Model-GGUF", squashed_snapshot.candidate, None
    )


def test_an_empty_answer_for_the_revision_is_judged_on_disk(squashed_snapshot, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "get_paths_info", lambda *a, **k: [])
    assert llama_cpp._cached_candidate_matches_revision_size(
        "org/Model-GGUF", squashed_snapshot.candidate, None
    )
    (squashed_snapshot.snap / "Q4" / "Model-Q4_K_M-00001-of-00002.gguf").unlink()
    (squashed_snapshot.snap / "Q4" / "Model-Q4_K_M-00001-of-00002.gguf").write_bytes(b"")
    assert not llama_cpp._cached_candidate_matches_revision_size(
        "org/Model-GGUF", squashed_snapshot.candidate, None
    )
