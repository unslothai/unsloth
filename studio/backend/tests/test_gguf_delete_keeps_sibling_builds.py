# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGUF repos list one build per quant (#13327), so deletion must preserve cached siblings."""

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


def test_each_cache_revision_uses_the_checkpoint_its_lister_exposes(tmp_path):
    from hub.services.models.companion_cleanup import (
        _remaining_main_gguf_variants,
        _variant_bytes,
    )
    from hub.services.models.deletion import _delete_gguf_variant_from_repos

    repo_dir = tmp_path / "models--org--Model-GGUF"
    blobs = repo_dir / "blobs"
    blobs.mkdir(parents = True)
    revisions = []
    names_by_revision = (
        ("old", ["alpha-Q4_K_M.gguf", "zeta-Q4_K_M.gguf"]),
        ("new", ["beta-Q4_K_M.gguf", "gamma-Q4_K_M.gguf"]),
    )
    for revision_name, names in names_by_revision:
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
        revisions.append(SimpleNamespace(files = files, snapshot_path = str(snapshot)))
    repo = SimpleNamespace(
        repo_id = "org/Model-GGUF",
        repo_type = "model",
        repo_path = repo_dir,
        revisions = revisions,
    )

    assert _variant_bytes(repo, "Q4_K_M") == 128
    assert _remaining_main_gguf_variants(repo, excluding = "Q4_K_M") == {
        "zeta-q4_k_m",
        "gamma-q4_k_m",
    }
    _delete_gguf_variant_from_repos(repo.repo_id, "Q4_K_M", [repo], None, root = tmp_path)
    assert not (repo_dir / "snapshots/old/alpha-Q4_K_M.gguf").is_symlink()
    assert (repo_dir / "snapshots/old/zeta-Q4_K_M.gguf").is_symlink()
    assert not (repo_dir / "snapshots/new/beta-Q4_K_M.gguf").is_symlink()
    assert (repo_dir / "snapshots/new/gamma-Q4_K_M.gguf").is_symlink()


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
    """stale reclaim must preserve sibling-build blobs that share the listed build's quant."""
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
