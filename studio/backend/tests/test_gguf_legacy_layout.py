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


def test_a_withdrawn_root_copy_does_not_hide_a_published_legacy_copy():
    """Both the root and the legacy-folder copies are on disk: the root bytes are withdrawn (WITHDRAWN), the legacy
    bytes are still published (OLD). The user holds loadable published bytes, so this is not an update."""
    withdrawn = "f" * 64
    local = {"Model-UD-IQ1_S.gguf": {withdrawn}, "Unsloth_Layout/Model-UD-IQ1_S.gguf": {OLD}}
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


def _lookup(weight, reuse_published = True):
    return _backend()._download_companion_gguf(
        hf_repo = "org/Model-GGUF",
        hf_token = None,
        pick = llama_cpp._pick_dspark,
        label = "DSpark drafter",
        cancel_event = _NoWaitEvent(),
        near_path = weight,
        reuse_published_companion = reuse_published,
    )


def test_callers_that_do_not_opt_in_keep_the_live_pick(two_snapshots, monkeypatch):
    """MTP and DFlash companions keep their live-listing preference: a copy cached at another revision is not
    reused unless the caller opts in (mmproj does, DSpark runs its own loadability-checked reuse)."""
    monkeypatch.setattr(
        llama_cpp,
        "_published_lfs_hashes",
        lambda repo, token, **_kw: frozenset({NEW, DNEW, OLD, DOLD}),
    )
    _lookup(two_snapshots.weight, reuse_published = False)
    assert two_snapshots.downloads == ["dspark-Model-Q8_0.gguf"]


def test_only_mmproj_opts_in_to_cross_revision_reuse():
    import inspect

    src = inspect.getsource(llama_cpp.LlamaCppBackend)
    assert src.count("reuse_published_companion = not f16_unsafe") == 1
    assert (
        inspect.signature(llama_cpp.LlamaCppBackend._download_companion_gguf)
        .parameters["reuse_published_companion"]
        .default
        is False
    )


def test_a_drafter_cached_at_another_revision_and_still_published_is_reused(
    two_snapshots, monkeypatch
):
    monkeypatch.setattr(
        llama_cpp,
        "_published_lfs_hashes",
        lambda repo, token, **_kw: frozenset({NEW, DNEW, OLD, DOLD}),
    )
    found = _lookup(two_snapshots.weight)
    assert found is not None and found.endswith("dspark-Model-Q8_0.gguf")
    assert "2" * 40 in found
    assert two_snapshots.downloads == []


def test_a_cached_drafter_the_repo_dropped_is_not_reused_online(two_snapshots, monkeypatch):
    monkeypatch.setattr(
        llama_cpp, "_published_lfs_hashes", lambda repo, token, **_kw: frozenset({NEW, DNEW})
    )
    found = _lookup(two_snapshots.weight)
    assert two_snapshots.downloads == ["dspark-Model-Q8_0.gguf"]
    assert found == "/hub/org/Model-GGUF/dspark-Model-Q8_0.gguf"


def test_an_unanswered_published_listing_falls_back_to_the_existing_flow(
    two_snapshots, monkeypatch
):
    monkeypatch.setattr(llama_cpp, "_published_lfs_hashes", lambda repo, token, **_kw: None)
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


def _split_cache(monkeypatch, tmp_path, *, same_snapshot):
    """A repo whose split UD-IQ1_S moved into Unsloth_Layout/; the cached legacy shards sit in one snapshot or are
    spread over two (two partial downloads)."""
    from hub.utils import inventory_scan
    from utils import hf_cache_settings

    store = {}
    monkeypatch.setattr(hf_cache_settings, "_EXPLICIT_CACHE_ENV", {})
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(
        "storage.studio_db.get_app_setting", lambda key, fallback = None: store.get(key, fallback)
    )
    monkeypatch.setattr("storage.studio_db.upsert_app_settings", store.update)
    monkeypatch.setattr("hub.utils.paths.legacy_hf_cache_dir", lambda: tmp_path / "legacy")
    monkeypatch.setattr("hub.utils.paths.hf_default_cache_dir", lambda: tmp_path / "unused")
    root = hf_cache_settings.get_hf_cache_paths().hub_cache
    repo = root / "models--Org--Model-GGUF"
    blobs = repo / "blobs"
    blobs.mkdir(parents = True)
    shards = ["Model-UD-IQ1_S-00001-of-00002.gguf", "Model-UD-IQ1_S-00002-of-00002.gguf"]
    old = ["1" * 64, "2" * 64]
    for sha in old:
        (blobs / sha).write_bytes(b"\0" * 100)
    snaps = ["a" * 40] if same_snapshot else ["a" * 40, "b" * 40]
    for i, (name, sha) in enumerate(zip(shards, old)):
        snap = repo / "snapshots" / snaps[0 if same_snapshot else i] / "Unsloth_Layout"
        snap.mkdir(parents = True, exist_ok = True)
        os.symlink(os.path.relpath(blobs / sha, snap), snap / name)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(snaps[-1])
    inventory_scan.invalidate_hf_cache_scans()
    siblings = [_sib(n, s) for n, s in zip(shards, ["3" * 64, "4" * 64])] + [
        _sib(f"Unsloth_Layout/{n}", s) for n, s in zip(shards, old)
    ]
    return siblings


@pytest.mark.parametrize("same_snapshot", [True, False])
def test_a_split_legacy_copy_counts_only_when_one_snapshot_holds_every_shard(
    monkeypatch, tmp_path, same_snapshot
):
    import asyncio

    import hub.utils.gguf as G

    siblings = _split_cache(monkeypatch, tmp_path, same_snapshot = same_snapshot)
    monkeypatch.setattr(G, "_env_offline", lambda: False)
    monkeypatch.setattr(
        G, "_hub_model_info", lambda repo, token: SimpleNamespace(siblings = siblings)
    )
    monkeypatch.setattr(GV, "hub_cached_read_refused", lambda *a, **k: False)
    response = asyncio.run(GV.get_gguf_variants_response("Org/Model-GGUF"))
    row = next(v for v in response.variants if v.quant.upper() == "UD-IQ1_S")
    assert row.downloaded is same_snapshot


# ---- one rule across listings: hide a legacy copy only when its current file is listed too ----


LEGACY_ONLY = [
    _sib("Model-UD-IQ1_S.gguf", NEW),
    _sib("Unsloth_Layout/Model-UD-IQ1_S.gguf", OLD),
    _sib("Unsloth_Layout/Model-Q8_0.gguf", "9" * 64),
    _sib("Unsloth_Layout/mmproj-F16.gguf", MM),
]


def test_a_legacy_only_quant_and_projector_stay_listed(monkeypatch):
    """A quant (or projector) published only under the legacy folder is a file in its own right: listed once, under
    its root name, with vision on; the legacy copy of a quant that has a current file stays hidden."""
    import hub.utils.gguf as G

    monkeypatch.setattr(G, "_env_offline", lambda: False)
    monkeypatch.setattr(
        G, "_hub_model_info", lambda repo, token: SimpleNamespace(siblings = LEGACY_ONLY)
    )
    variants, has_vision, _siblings = G.list_gguf_variants("org/Model-GGUF")
    rows = {v.quant: v.filename for v in variants}
    assert rows == {"UD-IQ1_S": "Model-UD-IQ1_S.gguf", "Q8_0": "Unsloth_Layout/Model-Q8_0.gguf"}
    assert has_vision
    plans = build_gguf_variant_plans(LEGACY_ONLY)
    assert set(plans) == {"ud-iq1_s", "q8_0"}


def test_the_model_config_listing_follows_the_same_rule(monkeypatch):
    from utils.models import model_config as mc

    monkeypatch.setattr(mc, "_env_offline", lambda: False)
    info = SimpleNamespace(siblings = LEGACY_ONLY)
    monkeypatch.setattr(mc, "_hub_model_info_slow_link", lambda *a, **k: info)
    variants, has_vision = mc.list_gguf_variants("org/Model-GGUF")[:2]
    assert has_vision
    names = sorted(v.filename for v in variants)
    assert "Unsloth_Layout/Model-UD-IQ1_S.gguf" not in names
    assert "Unsloth_Layout/Model-Q8_0.gguf" in names


@pytest.mark.parametrize("lister", ["hub", "model_config"])
@pytest.mark.parametrize(
    "files, expect",
    [
        # both complete: one row, the current file
        (
            {"Model-UD-IQ1_S.gguf": 10, "Unsloth_Layout/Model-UD-IQ1_S.gguf": 10},
            "Model-UD-IQ1_S.gguf",
        ),
        # torn current copy: the complete legacy copy is the row
        (
            {"Model-UD-IQ1_S.gguf": 0, "Unsloth_Layout/Model-UD-IQ1_S.gguf": 10},
            "Unsloth_Layout/Model-UD-IQ1_S.gguf",
        ),
        # only the legacy copy: listed under the root quant name
        ({"Unsloth_Layout/Model-UD-IQ1_S.gguf": 10}, "Unsloth_Layout/Model-UD-IQ1_S.gguf"),
    ],
)
def test_on_disk_listings_show_one_row_per_quant(tmp_path, lister, files, expect):
    for name, size in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_bytes(b"\0" * size)
    if lister == "hub":
        import hub.utils.gguf as G
        variants, _ = G.list_local_gguf_variants(str(tmp_path))
    else:
        from utils.models import model_config as mc
        variants, _ = mc.list_local_gguf_variants(str(tmp_path))
    rows = [(v.quant, v.filename) for v in variants]
    assert len(rows) == 1, rows
    assert rows[0][1] == expect
    assert "Unsloth_Layout" not in rows[0][0]


def test_mmproj_precision_order_never_picks_a_legacy_copy_over_a_current_projector():
    names = ["mmproj-Model-BF16.gguf", "Unsloth_Layout/mmproj-Model-F16.gguf"]
    assert llama_cpp._pick_mmproj(names) == "mmproj-Model-BF16.gguf"
    assert llama_cpp._pick_mmproj(names, order = ("f16", "bf16", "f32")) == "mmproj-Model-BF16.gguf"
    assert llama_cpp._pick_mmproj(["Unsloth_Layout/mmproj-Model-F16.gguf"]) == (
        "Unsloth_Layout/mmproj-Model-F16.gguf"
    )


def test_a_torn_current_split_does_not_hide_a_complete_legacy_split(tmp_path, monkeypatch):
    """Same snapshot: shard 1 of the current split only, both shards of the legacy split. The loader uses the
    complete legacy family instead of skipping the snapshot."""
    repo_dir = tmp_path / "models--org--Model-GGUF"
    snap = _snapshot(
        repo_dir,
        "1" * 40,
        {
            "Model-UD-IQ1_S-00001-of-00002.gguf": "5" * 64,
            "Unsloth_Layout/Model-UD-IQ1_S-00001-of-00002.gguf": "6" * 64,
            "Unsloth_Layout/Model-UD-IQ1_S-00002-of-00002.gguf": "7" * 64,
        },
    )
    monkeypatch.setattr(llama_cpp, "_cached_gguf_snapshots", lambda repo: [snap])
    import hub.utils.gguf_sources as gs

    monkeypatch.setattr(gs, "cached_gguf_manifest_complete", lambda *a, **k: True)
    monkeypatch.setattr(gs, "cached_gguf_source_partial", lambda *a, **k: False)
    got = list(llama_cpp._cached_variant_candidates("org/Model-GGUF", "UD-IQ1_S"))
    assert got and got[0][1] == "Unsloth_Layout/Model-UD-IQ1_S-00001-of-00002.gguf"


def test_a_filesystem_error_in_cross_revision_reuse_falls_back_to_the_download(
    two_snapshots, monkeypatch
):
    def _boom(*a, **k):
        raise PermissionError("cache entry unreadable")

    monkeypatch.setattr(llama_cpp, "_published_still_listing", _boom)
    found = _lookup(two_snapshots.weight)
    assert two_snapshots.downloads == ["dspark-Model-Q8_0.gguf"]
    assert found == "/hub/org/Model-GGUF/dspark-Model-Q8_0.gguf"
