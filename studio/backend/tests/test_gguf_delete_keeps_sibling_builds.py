# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGUF listings expose one build per quant (#13327), but deletion must preserve cached siblings."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from hub.utils.gguf import gguf_variant_keys_by_checkpoint, list_local_gguf_variants

PREFIX = "Huihui-Qwen3.8-27B-abliterated-"
LISTED = f"{PREFIX}GSQ-RCO-IQ3_S-mtp.gguf"
SWIFT = f"{PREFIX}Swift-1.5-GSQ-RCO-IQ3_S-mtp.gguf"
UD_DW = f"{PREFIX}UD-DW-IQ3_S.gguf"
BUILDS = [LISTED, SWIFT, UD_DW]


def _cache_repo(tmp_path: Path, names: list[str]):
    """model an HF cache whose snapshot symlinks match the deletion scan's input."""
    repo_dir = tmp_path / "models--huihui-ai--Model-GGUF"
    snap = repo_dir / "snapshots" / "rev1"
    blobs = repo_dir / "blobs"
    snap.mkdir(parents = True)
    blobs.mkdir(parents = True)
    files = []
    for index, name in enumerate(names):
        blob = blobs / f"sha{index}"
        blob.write_bytes(b"x" * 64)
        link = snap / name
        link.parent.mkdir(parents = True, exist_ok = True)
        link.symlink_to(blob)
        files.append(
            SimpleNamespace(
                file_name = name,
                file_path = str(link),
                blob_path = str(blob),
                size_on_disk = 64,
            )
        )
    return SimpleNamespace(
        repo_id = "huihui-ai/Model-GGUF",
        repo_type = "model",
        repo_path = repo_dir,
        revisions = [SimpleNamespace(files = files, snapshot_path = str(snap))],
    ), snap


def test_only_a_different_build_leaves_the_key():
    keys = gguf_variant_keys_by_checkpoint(
        [*BUILDS, "QwQ-32B-BF16.gguf", "QwQ-32B.BF16.gguf", "model-Q6_K-00001-of-00002.gguf"]
    )
    assert keys[LISTED] == "IQ3_S"
    assert keys[SWIFT] == SWIFT[: -len(".gguf")]
    assert keys[UD_DW] == UD_DW[: -len(".gguf")]
    # alternate spellings and split shards belong to the same checkpoint.
    assert keys["QwQ-32B-BF16.gguf"] == keys["QwQ-32B.BF16.gguf"] == "BF16"
    assert keys["model-Q6_K-00001-of-00002.gguf"] == "Q6_K"


def test_the_listing_still_offers_one_build_per_quant(tmp_path):
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    for name in BUILDS:
        (snapshot / name).write_bytes(b"x" * 64)
    (snapshot / "config.json").write_text("{}")
    variants, _ = list_local_gguf_variants(str(snapshot))
    assert [(v.quant, v.filename) for v in variants] == [("IQ3_S", LISTED)]


def test_deleting_the_row_keeps_the_other_builds(tmp_path):
    from hub.services.models.deletion import _delete_gguf_variant_from_repos

    repo, snap = _cache_repo(tmp_path, BUILDS)
    _delete_gguf_variant_from_repos(repo.repo_id, "IQ3_S", [repo], None, root = tmp_path)
    assert not (snap / LISTED).is_symlink()
    assert (snap / SWIFT).is_symlink()
    assert (snap / UD_DW).is_symlink()
    assert (repo.repo_path / "blobs" / "sha1").exists()


def test_copies_of_the_listed_build_still_go_with_it(tmp_path):
    from hub.services.models.deletion import _delete_gguf_variant_from_repos

    repo, snap = _cache_repo(tmp_path, ["Model-Q4_K_M.gguf", "Model.Q4_K_M.gguf"])
    _delete_gguf_variant_from_repos(repo.repo_id, "Q4_K_M", [repo], None, root = tmp_path)
    assert not (snap / "Model-Q4_K_M.gguf").is_symlink()
    assert not (snap / "Model.Q4_K_M.gguf").is_symlink()


def test_an_unnamed_checkpoint_is_not_a_wildcard_for_named_builds(tmp_path):
    from hub.services.models.deletion import _delete_gguf_variant_from_repos

    names = ["Q4_K_M.gguf", "model-a-Q4_K_M.gguf", "model-b-Q4_K_M.gguf"]
    repo, snap = _cache_repo(tmp_path, names)
    _delete_gguf_variant_from_repos(repo.repo_id, "Q4_K_M", [repo], None, root = tmp_path)
    assert not (snap / names[0]).is_symlink()
    assert (snap / names[1]).is_symlink()
    assert (snap / names[2]).is_symlink()


def test_hidden_big_endian_file_cannot_take_the_visible_rows_key(tmp_path):
    from hub.services.models.deletion import _delete_gguf_variant_from_repos
    for index, hidden in enumerate(("model-Q4_K_M-be.gguf", "0-F16-be-checkpoint-Q4_K_M.gguf")):
        root = tmp_path / str(index)
        visible = "model-Q4_K_M.gguf"
        repo, snap = _cache_repo(root, [hidden, visible])
        _delete_gguf_variant_from_repos(repo.repo_id, "Q4_K_M", [repo], None, root = root)
        assert (snap / hidden).is_symlink()
        assert not (snap / visible).is_symlink()


def _revisioned_repo(tmp_path: Path, names_by_revision):
    """treat the final HF cache snapshot as newest."""
    import os

    repo_dir = tmp_path / "models--org--Model-GGUF"
    blobs = repo_dir / "blobs"
    blobs.mkdir(parents = True)
    revisions = []
    for age, (revision_name, names) in enumerate(names_by_revision):
        snapshot = repo_dir / "snapshots" / revision_name
        snapshot.mkdir(parents = True)
        files = []
        for name in names:
            blob = blobs / f"{revision_name}-{name}"
            blob.write_bytes(b"x" * 64)
            link = snapshot / name
            link.symlink_to(blob)
            files.append(
                SimpleNamespace(
                    file_name = name,
                    file_path = str(link),
                    blob_path = str(blob),
                    size_on_disk = 64,
                )
            )
        os.utime(snapshot, (1_000_000 + age, 1_000_000 + age))
        revisions.append(SimpleNamespace(files = files, snapshot_path = str(snapshot)))
    return SimpleNamespace(
        repo_id = "org/Model-GGUF",
        repo_type = "model",
        repo_path = repo_dir,
        revisions = revisions,
    ), repo_dir / "snapshots"


def test_the_newest_snapshot_names_the_rows_build(tmp_path):
    from hub.services.models.companion_cleanup import (
        _remaining_main_gguf_variants,
        _variant_bytes,
    )
    from hub.services.models.deletion import _delete_gguf_variant_from_repos

    repo, snapshots = _revisioned_repo(
        tmp_path,
        (
            ("old", ["alpha-Q4_K_M.gguf", "beta-Q4_K_M.gguf", "zeta-Q4_K_M.gguf"]),
            ("new", ["beta-Q4_K_M.gguf", "gamma-Q4_K_M.gguf"]),
        ),
    )
    assert _variant_bytes(repo, "Q4_K_M") == 128
    assert _remaining_main_gguf_variants(repo, excluding = "Q4_K_M") == {
        "alpha-q4_k_m",
        "zeta-q4_k_m",
        "gamma-q4_k_m",
    }
    _delete_gguf_variant_from_repos(repo.repo_id, "Q4_K_M", [repo], None, root = tmp_path)
    assert not (snapshots / "new/beta-Q4_K_M.gguf").is_symlink()
    assert not (snapshots / "old/beta-Q4_K_M.gguf").is_symlink()
    assert (snapshots / "old/alpha-Q4_K_M.gguf").is_symlink()
    assert (snapshots / "old/zeta-Q4_K_M.gguf").is_symlink()
    assert (snapshots / "new/gamma-Q4_K_M.gguf").is_symlink()


def test_a_build_alone_in_an_older_snapshot_is_not_the_rows(tmp_path):
    from hub.services.models.deletion import _delete_gguf_variant_from_repos

    repo, snapshots = _revisioned_repo(tmp_path, (("old", [SWIFT]), ("new", [LISTED])))
    _delete_gguf_variant_from_repos(repo.repo_id, "IQ3_S", [repo], None, root = tmp_path)
    assert not (snapshots / "new" / LISTED).is_symlink()
    assert (snapshots / "old" / SWIFT).is_symlink()


def test_an_incomplete_newer_build_does_not_replace_the_complete_rows_build(tmp_path):
    from hub.services.models.deletion import _delete_gguf_variant_from_repos
    from hub.utils.gguf import select_gguf_cache_snapshot_for_repo_dir

    old = "old-Q4_K_M.gguf"
    partial = "new-Q4_K_M-00001-of-00002.gguf"
    repo, snapshots = _revisioned_repo(tmp_path, (("old", [old]), ("new", [partial])))

    selected = select_gguf_cache_snapshot_for_repo_dir(repo.repo_path)
    assert selected is not None and selected[3] == snapshots / "old"
    _delete_gguf_variant_from_repos(repo.repo_id, "Q4_K_M", [repo], None, root = tmp_path)
    assert not (snapshots / "old" / old).is_symlink()
    assert (snapshots / "new" / partial).is_symlink()


def test_a_hidden_only_newer_snapshot_does_not_replace_the_visible_rows_build(tmp_path):
    from hub.services.models.deletion import _delete_gguf_variant_from_repos

    partial = "model-Q4_K_M-00001-of-00002.gguf"
    hidden = "model-Q4_K_M-be.gguf"
    repo, snapshots = _revisioned_repo(tmp_path, (("old", [partial]), ("new", [hidden])))

    _delete_gguf_variant_from_repos(repo.repo_id, "Q4_K_M", [repo], None, root = tmp_path)
    assert not (snapshots / "old" / partial).is_symlink()
    assert (snapshots / "new" / hidden).is_symlink()


def test_each_rows_build_is_selected_independently_across_snapshots(tmp_path):
    from hub.services.models.deletion import _delete_gguf_variant_from_repos

    alpha = "alpha-Q4_K_M-00001-of-00002.gguf"
    beta = "beta-Q4_K_M-00001-of-00002.gguf"
    repo, snapshots = _revisioned_repo(
        tmp_path,
        (("old", [alpha, "model-Q6_K.gguf"]), ("new", [beta])),
    )

    _delete_gguf_variant_from_repos(repo.repo_id, "Q4_K_M", [repo], None, root = tmp_path)
    assert not (snapshots / "new" / beta).is_symlink()
    assert (snapshots / "old" / alpha).is_symlink()
    assert (snapshots / "old/model-Q6_K.gguf").is_symlink()


def test_readiness_ranking_selects_the_healthy_rows_build(tmp_path):
    from hub.services.models.deletion import _delete_gguf_variant_from_repos
    from hub.utils import download_manifest
    from hub.utils.gguf_sources import cached_gguf_manifest_complete

    alpha = "alpha-Q4_K_M.gguf"
    beta = "beta-Q4_K_M.gguf"
    old_revision = "a" * 40
    new_revision = "b" * 40
    repo, snapshots = _revisioned_repo(tmp_path, ((old_revision, [alpha]), (new_revision, [beta])))
    assert download_manifest.write_manifest(
        "model",
        repo.repo_id,
        "Q4_K_M",
        [
            download_manifest.ExpectedFile(beta, 64),
            download_manifest.ExpectedFile("mmproj-F16.gguf", 128),
        ],
        hub_cache = tmp_path,
        commit_hash = new_revision,
        metadata_derived = True,
    )
    assert not cached_gguf_manifest_complete(repo.repo_id, "Q4_K_M", snapshots / new_revision)
    assert cached_gguf_manifest_complete(repo.repo_id, "Q4_K_M", snapshots / old_revision)

    _delete_gguf_variant_from_repos(repo.repo_id, "Q4_K_M", [repo], None, root = tmp_path)
    assert not (snapshots / old_revision / alpha).is_symlink()
    assert (snapshots / new_revision / beta).is_symlink()


def test_scoped_source_path_selects_the_rows_build(tmp_path):
    from hub.services.models.companion_cleanup import _variant_bytes
    from hub.services.models.deletion import _delete_gguf_variant_from_repos
    from hub.utils.gguf_sources import cached_gguf_action_path

    alpha = "alpha-Q4_K_M.gguf"
    beta = "beta-Q4_K_M.gguf"
    repo, snapshots = _revisioned_repo(tmp_path, (("old", [alpha]), ("new", [beta])))
    beta_entry = repo.revisions[1].files[0]
    Path(beta_entry.blob_path).write_bytes(b"x" * 128)
    beta_entry.size_on_disk = 128
    preferred_path = str(snapshots / "old" / alpha)

    assert cached_gguf_action_path(repo.repo_id, "Q4_K_M", preferred_path = preferred_path) == str(
        repo.repo_path
    )
    assert _variant_bytes(repo, "Q4_K_M") == 128
    assert _variant_bytes(repo, "Q4_K_M", preferred_path) == 64
    _delete_gguf_variant_from_repos(
        repo.repo_id,
        "Q4_K_M",
        [repo],
        None,
        root = tmp_path,
        preferred_path = preferred_path,
    )
    assert not (snapshots / "old" / alpha).is_symlink()
    assert (snapshots / "new" / beta).is_symlink()


def test_scoped_source_path_keeps_repo_ancestor_for_shared_blob(tmp_path):
    from hub.utils.gguf_sources import cached_gguf_action_path

    repo_dir = tmp_path / "hub" / "models--org--Model-GGUF"
    source = repo_dir / "snapshots" / "rev" / "model-Q4_K_M.gguf"
    shared = tmp_path / "xet" / "blobs" / "shared"
    source.parent.mkdir(parents = True)
    shared.parent.mkdir(parents = True)
    shared.write_bytes(b"x")
    source.symlink_to(shared)

    assert cached_gguf_action_path("org/Model-GGUF", "Q4_K_M", preferred_path = str(source)) == str(
        repo_dir
    )


def test_deleting_selected_build_preserves_sibling_partial_state(tmp_path, monkeypatch):
    from hub.services.models import gguf_variants
    from hub.services.models.deletion import _delete_gguf_variant_from_repos
    from hub.utils import download_manifest
    from utils import hf_cache_settings

    alpha = "alpha-Q4_K_M.gguf"
    beta = "beta-Q4_K_M.gguf"
    repo, snapshots = _revisioned_repo(tmp_path, (("old", [alpha]), ("new", [beta])))
    monkeypatch.setattr(
        hf_cache_settings,
        "get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = tmp_path),
    )
    monkeypatch.setattr(gguf_variants, "_fetch_gguf_variant_requirements", lambda *a, **k: {})
    partial_hash = "f" * 64
    incomplete = repo.repo_path / "blobs" / f"{partial_hash}.incomplete"
    incomplete.write_bytes(b"partial")
    empty_beta_dir = snapshots / "new" / "Q4_K_M"
    empty_beta_dir.mkdir()
    assert download_manifest.write_manifest(
        "model",
        repo.repo_id,
        "Q4_K_M",
        [download_manifest.ExpectedFile(beta, 128, partial_hash)],
        hub_cache = tmp_path,
        commit_hash = "new",
        metadata_derived = True,
    )
    assert download_manifest.write_cancel_marker(
        "model", repo.repo_id, "Q4_K_M", "http", hub_cache = tmp_path
    )

    _delete_gguf_variant_from_repos(
        repo.repo_id,
        "Q4_K_M",
        [repo],
        None,
        root = tmp_path,
        preferred_path = str(snapshots / "old" / alpha),
    )

    assert not (snapshots / "old" / alpha).is_symlink()
    assert (snapshots / "new" / beta).is_symlink()
    assert incomplete.exists()
    assert empty_beta_dir.is_dir()
    assert (
        download_manifest.read_manifest("model", repo.repo_id, "Q4_K_M", hub_cache = tmp_path)
        is not None
    )
    assert download_manifest.has_cancel_marker("model", repo.repo_id, "Q4_K_M", hub_cache = tmp_path)


def test_deleting_selected_build_preserves_unmanifested_sibling_partial(tmp_path, monkeypatch):
    from hub.services.models import gguf_variants
    from hub.services.models.deletion import _delete_gguf_variant_from_repos
    from hub.utils import download_manifest
    from utils import hf_cache_settings

    beta = "beta-Q4_K_M.gguf"
    repo, snapshots = _revisioned_repo(tmp_path, (("healthy", [beta]),))
    monkeypatch.setattr(
        hf_cache_settings,
        "get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = tmp_path),
    )
    partial_hash = "f" * 64
    incomplete = repo.repo_path / "blobs" / f"{partial_hash}.incomplete"
    incomplete.write_bytes(b"partial")
    empty_sibling_dir = snapshots / "healthy" / "Q4_K_M"
    empty_sibling_dir.mkdir()
    assert download_manifest.write_cancel_marker(
        "model", repo.repo_id, "Q4_K_M", "http", hub_cache = tmp_path
    )
    monkeypatch.setattr(
        gguf_variants,
        "gguf_variant_blob_hashes",
        lambda _repo_id, candidate, *_args, **_kwargs: (
            frozenset({partial_hash}) if candidate == "Q4_K_M" else frozenset()
        ),
    )

    _delete_gguf_variant_from_repos(
        repo.repo_id,
        "Q4_K_M",
        [repo],
        None,
        root = tmp_path,
    )

    assert not (snapshots / "healthy" / beta).is_symlink()
    assert incomplete.exists()
    assert empty_sibling_dir.is_dir()
    assert download_manifest.has_cancel_marker("model", repo.repo_id, "Q4_K_M", hub_cache = tmp_path)


def test_delete_preview_counts_the_whole_audio_package(tmp_path):
    from core.inference.audio_cpp_models import FAMILIES
    from hub.services.models.companion_cleanup import (
        _remaining_main_gguf_variants,
        _variant_bytes,
    )

    package = next(
        variant for variant in FAMILIES["minimax_music3"].package if variant.key == "Q8_0"
    )
    repo, _snap = _cache_repo(tmp_path, list(package.files))
    assert _variant_bytes(repo, "Q8_0") == 64 * len(package.files)
    assert _remaining_main_gguf_variants(repo, excluding = "Q8_0") == set()


def test_the_delete_preview_counts_only_the_listed_build(tmp_path):
    from hub.services.models.companion_cleanup import (
        _remaining_main_gguf_variants,
        _variant_bytes,
    )

    repo, _snap = _cache_repo(tmp_path, BUILDS)
    assert _variant_bytes(repo, "IQ3_S") == 64
    # surviving builds retain their companions, so those bytes are not reclaimable.
    assert _remaining_main_gguf_variants(repo, excluding = "IQ3_S") == {
        SWIFT[: -len(".gguf")].lower(),
        UD_DW[: -len(".gguf")].lower(),
    }


def test_the_delete_preview_excludes_a_blob_retained_by_a_sibling_build(tmp_path):
    from hub.services.models.companion_cleanup import _variant_bytes
    from hub.services.models.deletion import _delete_gguf_variant_from_repos

    repo, snap = _cache_repo(tmp_path, [LISTED, SWIFT])
    shared_blob = repo.repo_path / "blobs" / "sha0"
    (snap / SWIFT).unlink()
    (snap / SWIFT).symlink_to(shared_blob)
    repo.revisions[0].files[1].blob_path = str(shared_blob)

    assert _variant_bytes(repo, "IQ3_S") == 0
    _delete_gguf_variant_from_repos(repo.repo_id, "IQ3_S", [repo], None, root = tmp_path)
    assert shared_blob.exists()


def test_deleting_from_a_local_folder_keeps_the_other_builds(tmp_path):
    from routes.models import _delete_gguf_variant_files

    for name in BUILDS:
        (tmp_path / name).write_bytes(b"x" * 64)
    assert _delete_gguf_variant_files(tmp_path, "IQ3_S")[0] == 1
    assert not (tmp_path / LISTED).exists()
    assert (tmp_path / SWIFT).exists()
    assert (tmp_path / UD_DW).exists()


def test_reclaiming_an_update_leaves_the_other_builds(monkeypatch, tmp_path):
    from hub.services.models import cache_inventory as CI
    from hub.services.models import deletion as D

    repo_path = tmp_path / "models--huihui-ai--Model-GGUF"
    snap = repo_path / "snapshots" / ("b" * 40)
    entries = []
    for name, sha in ((LISTED, "NEWsha"), (SWIFT, "SWIFTsha")):
        link = snap / name
        blob = repo_path / "blobs" / sha
        for path in (link, blob):
            path.parent.mkdir(parents = True, exist_ok = True)
            path.write_bytes(sha.encode())
        entries.append(SimpleNamespace(file_name = name, file_path = str(link), blob_path = str(blob)))
    repo_info = SimpleNamespace(
        repo_id = "huihui-ai/Model-GGUF",
        repo_type = "model",
        repo_path = repo_path,
        revisions = [SimpleNamespace(files = entries)],
    )
    monkeypatch.setattr(CI, "all_hf_cache_scans", lambda: [SimpleNamespace(repos = [repo_info])])
    monkeypatch.setattr(CI, "invalidate_hf_cache_scans", lambda: None)

    D.reclaim_replaced_gguf_variant(
        repo_info.repo_id, "IQ3_S", frozenset({"NEWsha"}), hub_cache = tmp_path
    )
    assert (snap / SWIFT).exists()
    assert (repo_path / "blobs" / "SWIFTsha").exists()


def test_reclaiming_an_update_keeps_a_build_alone_in_an_older_snapshot(monkeypatch, tmp_path):
    import os

    from hub.services.models import cache_inventory as CI
    from hub.services.models import deletion as D

    repo_path = tmp_path / "models--huihui-ai--Model-GGUF"
    revisions = []
    layout = (
        ("a" * 40, LISTED, "OLDsha"),
        ("c" * 40, SWIFT, "SWIFTsha"),
        ("b" * 40, LISTED, "NEWsha"),
    )
    for age, (rev, name, sha) in enumerate(layout):
        snap = repo_path / "snapshots" / rev
        link = snap / name
        blob = repo_path / "blobs" / sha
        for path in (link, blob):
            path.parent.mkdir(parents = True, exist_ok = True)
            path.write_bytes(sha.encode())
        os.utime(snap, (1_000_000 + age, 1_000_000 + age))
        revisions.append(
            SimpleNamespace(
                files = [SimpleNamespace(file_name = name, file_path = str(link), blob_path = str(blob))],
                snapshot_path = str(snap),
            )
        )
    repo_info = SimpleNamespace(
        repo_id = "huihui-ai/Model-GGUF",
        repo_type = "model",
        repo_path = repo_path,
        revisions = revisions,
    )
    monkeypatch.setattr(CI, "all_hf_cache_scans", lambda: [SimpleNamespace(repos = [repo_info])])
    monkeypatch.setattr(CI, "invalidate_hf_cache_scans", lambda: None)

    D.reclaim_replaced_gguf_variant(
        repo_info.repo_id, "IQ3_S", frozenset({"NEWsha"}), hub_cache = tmp_path
    )
    assert not (repo_path / "blobs" / "OLDsha").exists()
    assert (repo_path / "snapshots" / ("c" * 40) / SWIFT).exists()
    assert (repo_path / "blobs" / "SWIFTsha").exists()
    assert (repo_path / "blobs" / "NEWsha").exists()


def test_builds_under_quant_named_directories_stay_apart(tmp_path):
    from hub.services.models.deletion import _delete_gguf_variant_from_repos

    keys = gguf_variant_keys_by_checkpoint(
        [
            "alpha-Q4_K_M/model.gguf",
            "beta-Q4_K_M/model.gguf",
            "Q6_K/model-Q6_K-00001-of-00002.gguf",
            "model-Q6_K/model-Q6_K-00001-of-00002.gguf",
        ]
    )
    assert keys["alpha-Q4_K_M/model.gguf"] == "Q4_K_M"
    assert keys["beta-Q4_K_M/model.gguf"] == "beta-Q4_K_M/model"
    # a directory repeating the file's own name is still a copy of the build.
    assert keys["model-Q6_K/model-Q6_K-00001-of-00002.gguf"] == "Q6_K"

    repo, snap = _cache_repo(tmp_path, ["alpha-Q4_K_M/model.gguf", "beta-Q4_K_M/model.gguf"])
    _delete_gguf_variant_from_repos(repo.repo_id, "Q4_K_M", [repo], None, root = tmp_path)
    assert not (snap / "alpha-Q4_K_M/model.gguf").is_symlink()
    assert (snap / "beta-Q4_K_M/model.gguf").is_symlink()
