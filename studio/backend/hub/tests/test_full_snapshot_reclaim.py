# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import errno
import os
import stat
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from hub.utils import download_manifest, download_registry, snapshot_reclaim
from hub.workers import hf_download


REPO_ID = "Org/Model"
OLD = "a" * 40
NEW = "b" * 40
DETACHED = "c" * 40


def _cache_repo(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "hub"
    repo = root / "models--Org--Model"
    (repo / "blobs").mkdir(parents = True)
    (repo / "refs").mkdir()
    (repo / "snapshots").mkdir()
    (repo / "refs" / "main").write_text(OLD, encoding = "utf-8")
    return root, repo


def _snapshot_file(repo: Path, revision: str, name: str, payload: bytes) -> Path:
    snapshot_file = repo / "snapshots" / revision / name
    snapshot_file.parent.mkdir(parents = True, exist_ok = True)
    snapshot_file.write_bytes(payload)
    return snapshot_file


def _displaced_refs(repo: Path) -> list[Path]:
    """Return the displaced previous refs, asserting none leaked into ``refs``.

    Scratch refs are staged beside ``refs`` because third-party cache scanners
    glob ``refs/**/*`` -- dotfiles included -- and read every entry as a ref.
    """
    stray = sorted((repo / "refs").glob(".unsloth-main-*"))
    assert stray == [], f"scratch refs leaked into refs/: {stray}"
    staging = repo / snapshot_reclaim._REFS_STAGING_DIRECTORY_NAME
    return sorted(staging.glob(".unsloth-main-previous-*")) if staging.is_dir() else []


def _capture(monkeypatch, root: Path):
    monkeypatch.setattr(
        snapshot_reclaim,
        "get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = root),
    )
    return snapshot_reclaim.capture_previous_main_ref(REPO_ID)


def _promotion_cache(tmp_path: Path, monkeypatch, previous_revision):
    if previous_revision is None:
        root = tmp_path / "hub"
        root.mkdir()
        previous = _capture(monkeypatch, root)
        repo = root / "models--Org--Model"
        (repo / "blobs").mkdir(parents = True)
        (repo / "refs").mkdir()
        (repo / "snapshots").mkdir()
        return repo, previous
    root, repo = _cache_repo(tmp_path)
    return repo, _capture(monkeypatch, root)


def test_first_download_captures_the_expected_root_and_promotes_it(tmp_path, monkeypatch):
    root = tmp_path / "hub"
    root.mkdir()
    previous = _capture(monkeypatch, root)
    repo = root / "models--Org--Model"
    (repo / "blobs").mkdir(parents = True)
    snapshot = repo / "snapshots" / NEW
    snapshot.mkdir(parents = True)
    (snapshot / "config.json").write_text("{}", encoding = "utf-8")

    result = snapshot_reclaim.promote_verified_snapshot(
        "model",
        REPO_ID,
        NEW,
        snapshot,
        previous,
    )

    assert (repo / "refs" / "main").read_text(encoding = "utf-8") == NEW
    assert result == snapshot_reclaim.SnapshotPromotionResult(None)


def test_model_promotion_preserves_the_previous_snapshot_for_exact_resume(tmp_path, monkeypatch):
    root, repo = _cache_repo(tmp_path)
    old_file = _snapshot_file(repo, OLD, "model.safetensors", b"old")
    new_file = _snapshot_file(repo, NEW, "model.safetensors", b"new")
    previous = _capture(monkeypatch, root)

    result = snapshot_reclaim.promote_verified_snapshot(
        "model",
        REPO_ID,
        NEW,
        new_file.parent,
        previous,
    )

    assert (repo / "refs" / "main").read_text(encoding = "utf-8") == NEW
    assert old_file.read_bytes() == b"old"
    assert new_file.read_bytes() == b"new"
    assert result == snapshot_reclaim.SnapshotPromotionResult(OLD)


def test_dataset_promotion_moves_main_without_reclaiming_the_previous_snapshot(
    tmp_path, monkeypatch
):
    root = tmp_path / "hub"
    repo = root / "datasets--Org--Model"
    old_snapshot = repo / "snapshots" / OLD
    new_snapshot = repo / "snapshots" / NEW
    old_snapshot.mkdir(parents = True)
    new_snapshot.mkdir(parents = True)
    (old_snapshot / "data.parquet").write_bytes(b"old")
    (new_snapshot / "data.parquet").write_bytes(b"new")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(OLD, encoding = "utf-8")
    monkeypatch.setattr(
        snapshot_reclaim,
        "get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = root),
    )
    previous = snapshot_reclaim.capture_previous_main_ref(
        REPO_ID,
        repo_type = "dataset",
    )

    result = snapshot_reclaim.promote_verified_snapshot(
        "dataset",
        REPO_ID,
        NEW,
        new_snapshot,
        previous,
    )

    assert (repo / "refs" / "main").read_text(encoding = "utf-8") == NEW
    assert old_snapshot.is_dir()
    assert new_snapshot.is_dir()
    assert result == snapshot_reclaim.SnapshotPromotionResult(OLD)


@pytest.mark.parametrize("line_ending", [b"\n", b"\r\n"])
def test_main_ref_line_ending_is_normalized_during_promotion(line_ending, tmp_path, monkeypatch):
    root, repo = _cache_repo(tmp_path)
    main = repo / "refs" / "main"
    main.write_bytes(OLD.encode("ascii") + line_ending)
    new_file = _snapshot_file(repo, NEW, "model.safetensors", b"new")

    previous = _capture(monkeypatch, root)
    result = snapshot_reclaim.promote_verified_snapshot(
        "model",
        REPO_ID,
        NEW,
        new_file.parent,
        previous,
    )

    assert previous.revision == OLD
    assert previous.promotion_safe is True
    assert result == snapshot_reclaim.SnapshotPromotionResult(OLD)
    assert main.read_bytes() == NEW.encode("ascii")


@pytest.mark.parametrize(
    "raw",
    [
        f" {OLD}",
        f"{OLD} ",
        f"{OLD}\n\n",
        f"{OLD}\r\n\r\n",
        f"{OLD}\r",
        f"{OLD}\n{NEW}",
    ],
)
def test_main_ref_parser_rejects_non_cosmetic_whitespace(raw):
    assert snapshot_reclaim._parse_main_ref(raw) is None


def test_line_terminated_main_ref_still_detects_a_concurrent_advance(tmp_path, monkeypatch):
    root, repo = _cache_repo(tmp_path)
    main = repo / "refs" / "main"
    main.write_bytes(f"{OLD}\n".encode("ascii"))
    new_file = _snapshot_file(repo, NEW, "model.safetensors", b"new")
    previous = _capture(monkeypatch, root)
    main.write_bytes(f"{DETACHED}\n".encode("ascii"))

    with pytest.raises(snapshot_reclaim.ConcurrentMainRefError, match = "changed during download"):
        snapshot_reclaim.promote_verified_snapshot(
            "model",
            REPO_ID,
            NEW,
            new_file.parent,
            previous,
        )

    assert main.read_bytes() == f"{DETACHED}\n".encode("ascii")


@pytest.mark.parametrize(
    ("repo_type", "repo_prefix"),
    [("model", "models"), ("dataset", "datasets")],
)
def test_redirected_repo_delegates_activation_to_huggingface(
    repo_type, repo_prefix, tmp_path, monkeypatch
):
    root = tmp_path / "hub"
    target = tmp_path / "relocated"
    root.mkdir()
    target.mkdir()
    redirected = root / f"{repo_prefix}--Org--Model"
    try:
        redirected.symlink_to(target, target_is_directory = True)
    except OSError:
        pytest.skip("directory symlinks unavailable on this host")
    monkeypatch.setattr(
        snapshot_reclaim,
        "get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = root),
    )

    previous = snapshot_reclaim.capture_previous_main_ref(
        REPO_ID,
        repo_type = repo_type,
    )

    assert previous.promotion_safe is False
    assert previous.allow_unpinned_download is True
    assert "Hub cache repo is redirected" in (previous.reason or "")
    assert hf_download._snapshot_activation_plan(repo_type, REPO_ID, NEW, True) == (None, None)


@pytest.mark.parametrize(
    "raw_ref",
    [b"invalid\nref", b"\xff", b"x" * 257],
)
@pytest.mark.parametrize(
    ("repo_type", "repo_prefix"),
    [("model", "models"), ("dataset", "datasets")],
)
def test_invalid_regular_main_ref_delegates_activation_to_huggingface(
    raw_ref, repo_type, repo_prefix, tmp_path, monkeypatch
):
    root = tmp_path / "hub"
    repo = root / f"{repo_prefix}--Org--Model"
    (repo / "refs").mkdir(parents = True)
    (repo / "refs" / "main").write_bytes(raw_ref)
    monkeypatch.setattr(
        snapshot_reclaim,
        "get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = root),
    )

    previous = snapshot_reclaim.capture_previous_main_ref(
        REPO_ID,
        repo_type = repo_type,
    )

    assert previous.promotion_safe is False
    assert previous.allow_unpinned_download is True
    assert hf_download._snapshot_activation_plan(repo_type, REPO_ID, NEW, True) == (None, None)


@pytest.mark.parametrize(
    ("repo_type", "repo_prefix"),
    [("model", "models"), ("dataset", "datasets")],
)
def test_empty_main_ref_is_read_as_absent_rather_than_unpinning(
    repo_type, repo_prefix, tmp_path, monkeypatch
):
    """An empty refs/main is debris from a promotion that died before its write.

    No writer publishes an empty ref, so reading it as invalid would hand the repo
    to an unpinned download; reading it as absent keeps the next download pinned
    and lets the promotion reclaim the file.
    """
    root = tmp_path / "hub"
    repo = root / f"{repo_prefix}--Org--Model"
    (repo / "refs").mkdir(parents = True)
    (repo / "refs" / "main").write_bytes(b"")
    monkeypatch.setattr(
        snapshot_reclaim,
        "get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = root),
    )

    previous = snapshot_reclaim.capture_previous_main_ref(REPO_ID, repo_type = repo_type)

    assert previous.promotion_safe is True
    assert previous.revision is None
    assert previous.allow_unpinned_download is False
    assert hf_download._snapshot_activation_plan(repo_type, REPO_ID, NEW, True) != (None, None)


@pytest.mark.parametrize(
    ("repo_type", "repo_prefix"),
    [("model", "models"), ("dataset", "datasets")],
)
def test_unreadable_regular_main_ref_delegates_activation_to_huggingface(
    repo_type, repo_prefix, tmp_path, monkeypatch
):
    root = tmp_path / "hub"
    repo = root / f"{repo_prefix}--Org--Model"
    (repo / "refs").mkdir(parents = True)
    (repo / "refs" / "main").write_text(OLD, encoding = "utf-8")
    monkeypatch.setattr(
        snapshot_reclaim,
        "get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = root),
    )
    monkeypatch.setattr(
        snapshot_reclaim,
        "_read_main_ref_text",
        lambda _path: (_ for _ in ()).throw(PermissionError("temporarily locked")),
    )

    previous = snapshot_reclaim.capture_previous_main_ref(
        REPO_ID,
        repo_type = repo_type,
    )

    assert previous.promotion_safe is False
    assert previous.allow_unpinned_download is True
    assert "refs/main could not be read" in (previous.reason or "")
    assert hf_download._snapshot_activation_plan(repo_type, REPO_ID, NEW, True) == (None, None)


@pytest.mark.parametrize(
    ("repo_type", "repo_prefix"),
    [("model", "models"), ("dataset", "datasets")],
)
@pytest.mark.parametrize("failure_point", ["repo", "refs_lstat", "refs_resolve", "main"])
def test_transient_cache_inspection_error_delegates_activation_to_huggingface(
    repo_type, repo_prefix, failure_point, tmp_path, monkeypatch
):
    root = tmp_path / "hub"
    repo = root / f"{repo_prefix}--Org--Model"
    refs = repo / "refs"
    main = refs / "main"
    refs.mkdir(parents = True)
    main.write_text(OLD, encoding = "utf-8")
    monkeypatch.setattr(
        snapshot_reclaim,
        "get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = root),
    )

    if failure_point == "repo":

        def transient_repo(*_args, **_kwargs):
            raise OSError(errno.EIO, "cache temporarily unavailable")

        monkeypatch.setattr(
            snapshot_reclaim,
            "_canonical_repo_dir",
            transient_repo,
        )
    elif failure_point in {"refs_lstat", "main"}:
        failed_path = refs if failure_point == "refs_lstat" else main
        real_lstat = snapshot_reclaim._lstat_or_none

        def transient_lstat(path):
            if path == failed_path:
                raise OSError(errno.EIO, "cache temporarily unavailable")
            return real_lstat(path)

        monkeypatch.setattr(snapshot_reclaim, "_lstat_or_none", transient_lstat)
    else:
        real_resolve = Path.resolve

        def transient_resolve(path, *args, **kwargs):
            if path == refs:
                raise OSError(errno.EIO, "cache temporarily unavailable")
            return real_resolve(path, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", transient_resolve)

    previous = snapshot_reclaim.capture_previous_main_ref(
        REPO_ID,
        repo_type = repo_type,
    )

    assert previous.promotion_safe is False
    assert previous.allow_unpinned_download is True
    assert "cache temporarily unavailable" in (previous.reason or "")
    assert hf_download._snapshot_activation_plan(repo_type, REPO_ID, NEW, True) == (None, None)


@pytest.mark.parametrize(
    ("repo_type", "repo_prefix"),
    [("model", "models"), ("dataset", "datasets")],
)
def test_structurally_unsafe_refs_still_block_activation(
    repo_type, repo_prefix, tmp_path, monkeypatch
):
    root = tmp_path / "hub"
    repo = root / f"{repo_prefix}--Org--Model"
    repo.mkdir(parents = True)
    (repo / "refs").write_text("not a directory", encoding = "utf-8")
    monkeypatch.setattr(
        snapshot_reclaim,
        "get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = root),
    )

    previous = snapshot_reclaim.capture_previous_main_ref(
        REPO_ID,
        repo_type = repo_type,
    )

    assert previous.promotion_safe is False
    assert previous.allow_unpinned_download is False
    with pytest.raises(RuntimeError, match = "refs is not a real directory"):
        hf_download._snapshot_activation_plan(
            repo_type,
            REPO_ID,
            NEW,
            True,
        )


def test_promotion_only_accepts_a_concurrent_promotion_to_the_same_revision(tmp_path, monkeypatch):
    root, repo = _cache_repo(tmp_path)
    new_file = _snapshot_file(repo, NEW, "model.safetensors", b"new")
    previous = _capture(monkeypatch, root)
    (repo / "refs" / "main").write_text(NEW, encoding = "utf-8")

    result = snapshot_reclaim.promote_verified_snapshot(
        "model",
        REPO_ID,
        NEW,
        new_file.parent,
        previous,
    )

    assert result == snapshot_reclaim.SnapshotPromotionResult(OLD)
    assert (repo / "refs" / "main").read_text(encoding = "utf-8") == NEW


def test_repo_lock_serializes_main_ref_promotion(tmp_path, monkeypatch):
    root, repo = _cache_repo(tmp_path)
    new_file = _snapshot_file(repo, NEW, "model.safetensors", b"new")
    detached_file = _snapshot_file(repo, DETACHED, "model.safetensors", b"detached")
    previous = _capture(monkeypatch, root)
    first_at_claim = threading.Event()
    release_first = threading.Event()
    second_done = threading.Event()
    errors: dict[str, Exception] = {}
    real_rename = os.rename

    def controlled_rename(source, target):
        if Path(source) == repo / "refs" / "main":
            first_at_claim.set()
            assert release_first.wait(5)
        return real_rename(source, target)

    def promote(
        label: str,
        revision: str,
        snapshot: Path,
        done = None,
    ):
        try:
            snapshot_reclaim.promote_verified_snapshot(
                "model", REPO_ID, revision, snapshot, previous
            )
        except Exception as exc:
            errors[label] = exc
        finally:
            if done is not None:
                done.set()

    monkeypatch.setattr(os, "rename", controlled_rename)
    first = threading.Thread(target = promote, args = ("first", NEW, new_file.parent))
    first.start()
    assert first_at_claim.wait(5)
    second = threading.Thread(
        target = promote,
        args = ("second", DETACHED, detached_file.parent, second_done),
    )
    second.start()
    assert not second_done.wait(0.2)
    release_first.set()
    first.join(5)
    second.join(5)

    assert not first.is_alive() and not second.is_alive()
    assert "first" not in errors
    assert isinstance(errors.get("second"), snapshot_reclaim.ConcurrentMainRefError)
    assert (repo / "refs" / "main").read_text(encoding = "utf-8") == NEW


def test_repo_lock_spans_post_promotion_work(tmp_path, monkeypatch):
    root, repo = _cache_repo(tmp_path)
    new_file = _snapshot_file(repo, NEW, "model.safetensors", b"new")
    detached_file = _snapshot_file(repo, DETACHED, "model.safetensors", b"detached")
    previous = _capture(monkeypatch, root)
    newer_previous = snapshot_reclaim.PreviousMainRef(repo, NEW, True)
    callback_started = threading.Event()
    release_callback = threading.Event()
    second_done = threading.Event()
    errors: list[Exception] = []

    def after_promotion():
        callback_started.set()
        assert release_callback.wait(5)

    def first_promotion():
        try:
            snapshot_reclaim.promote_verified_snapshot(
                "model",
                REPO_ID,
                NEW,
                new_file.parent,
                previous,
                after_promotion = after_promotion,
            )
        except Exception as exc:
            errors.append(exc)

    def second_promotion():
        try:
            snapshot_reclaim.promote_verified_snapshot(
                "model", REPO_ID, DETACHED, detached_file.parent, newer_previous
            )
        except Exception as exc:
            errors.append(exc)
        finally:
            second_done.set()

    first = threading.Thread(target = first_promotion)
    first.start()
    assert callback_started.wait(5)
    second = threading.Thread(target = second_promotion)
    second.start()
    assert not second_done.wait(0.2)
    release_callback.set()
    first.join(5)
    second.join(5)

    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert (repo / "refs" / "main").read_text(encoding = "utf-8") == DETACHED


def test_invalid_snapshot_path_does_not_switch_main(tmp_path, monkeypatch):
    root, repo = _cache_repo(tmp_path)
    wrong = tmp_path / "models--Other--Repo" / "snapshots" / NEW
    wrong.mkdir(parents = True)
    previous = _capture(monkeypatch, root)

    with pytest.raises(ValueError, match = "Unexpected model snapshot path"):
        snapshot_reclaim.promote_verified_snapshot("model", REPO_ID, NEW, wrong, previous)

    assert (repo / "refs" / "main").read_text(encoding = "utf-8") == OLD


def test_same_repo_shape_in_another_root_is_never_promoted(tmp_path, monkeypatch):
    active_root, active_repo = _cache_repo(tmp_path / "active")
    other_root, other_repo = _cache_repo(tmp_path / "other")
    other_snapshot = _snapshot_file(other_repo, NEW, "model.safetensors", b"new")
    previous = _capture(monkeypatch, active_root)

    with pytest.raises(ValueError, match = "captured active Hub cache"):
        snapshot_reclaim.promote_verified_snapshot(
            "model",
            REPO_ID,
            NEW,
            other_snapshot.parent,
            previous,
        )

    assert (active_repo / "refs" / "main").read_text(encoding = "utf-8") == OLD
    assert (other_repo / "refs" / "main").read_text(encoding = "utf-8") == OLD
    assert other_root != active_root


def test_external_main_advance_during_download_is_not_overwritten(tmp_path, monkeypatch):
    root, repo = _cache_repo(tmp_path)
    old_file = _snapshot_file(repo, OLD, "model.safetensors", b"old")
    new_file = _snapshot_file(repo, NEW, "model.safetensors", b"new")
    _snapshot_file(repo, DETACHED, "model.safetensors", b"external")
    previous = _capture(monkeypatch, root)
    (repo / "refs" / "main").write_text(DETACHED, encoding = "utf-8")

    with pytest.raises(snapshot_reclaim.ConcurrentMainRefError, match = "changed during download"):
        snapshot_reclaim.promote_verified_snapshot(
            "model",
            REPO_ID,
            NEW,
            new_file.parent,
            previous,
        )

    assert (repo / "refs" / "main").read_text(encoding = "utf-8") == DETACHED
    assert old_file.read_bytes() == b"old"
    assert new_file.read_bytes() == b"new"


def test_external_main_advance_at_promotion_boundary_is_not_overwritten(tmp_path, monkeypatch):
    root, repo = _cache_repo(tmp_path)
    new_file = _snapshot_file(repo, NEW, "model.safetensors", b"new")
    previous = _capture(monkeypatch, root)
    main = repo / "refs" / "main"
    real_rename = os.rename

    def advance_before_claim(source, target):
        if Path(source) == main:
            main.write_text(DETACHED, encoding = "utf-8")
        return real_rename(source, target)

    monkeypatch.setattr(snapshot_reclaim.os, "rename", advance_before_claim)

    with pytest.raises(snapshot_reclaim.ConcurrentMainRefError, match = "changed during download"):
        snapshot_reclaim.promote_verified_snapshot(
            "model",
            REPO_ID,
            NEW,
            new_file.parent,
            previous,
        )

    assert main.read_text(encoding = "utf-8") == DETACHED


def test_redirected_main_at_claim_is_not_restored_as_the_active_ref(tmp_path, monkeypatch):
    root, repo = _cache_repo(tmp_path)
    refs = repo / "refs"
    main = refs / "main"
    external = tmp_path / "external-main"
    external.write_text(DETACHED, encoding = "utf-8")
    new_file = _snapshot_file(repo, NEW, "model.safetensors", b"new")
    previous = _capture(monkeypatch, root)
    real_rename = os.rename

    def redirect_before_claim(source, destination):
        if Path(source) == main:
            main.unlink()
            try:
                main.symlink_to(external)
            except OSError:
                pytest.skip("file symlinks are unavailable on this host")
        return real_rename(source, destination)

    monkeypatch.setattr(snapshot_reclaim.os, "rename", redirect_before_claim)

    with pytest.raises(RuntimeError, match = "previous ref remains at"):
        snapshot_reclaim.promote_verified_snapshot(
            "model",
            REPO_ID,
            NEW,
            new_file.parent,
            previous,
        )

    assert not main.exists() and not main.is_symlink()
    displaced = _displaced_refs(repo)
    assert len(displaced) == 1
    assert displaced[0].is_symlink()
    assert displaced[0].resolve() == external
    assert external.read_text(encoding = "utf-8") == DETACHED


def test_external_main_advance_after_claim_is_not_overwritten(tmp_path, monkeypatch):
    root, repo = _cache_repo(tmp_path)
    new_file = _snapshot_file(repo, NEW, "model.safetensors", b"new")
    previous = _capture(monkeypatch, root)
    main = repo / "refs" / "main"
    real_link = os.link

    def advance_before_publish(source, target):
        if Path(target) == main:
            main.write_text(DETACHED, encoding = "utf-8")
        return real_link(source, target)

    monkeypatch.setattr(snapshot_reclaim.os, "link", advance_before_publish)

    with pytest.raises(snapshot_reclaim.ConcurrentMainRefError, match = "external revision"):
        snapshot_reclaim.promote_verified_snapshot(
            "model",
            REPO_ID,
            NEW,
            new_file.parent,
            previous,
        )

    assert main.read_text(encoding = "utf-8") == DETACHED


def test_external_main_creation_during_first_download_is_not_overwritten(tmp_path, monkeypatch):
    root = tmp_path / "hub"
    root.mkdir()
    previous = _capture(monkeypatch, root)
    repo = root / "models--Org--Model"
    new_file = _snapshot_file(repo, NEW, "model.safetensors", b"new")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(DETACHED, encoding = "utf-8")

    with pytest.raises(snapshot_reclaim.ConcurrentMainRefError, match = "absent"):
        snapshot_reclaim.promote_verified_snapshot(
            "model",
            REPO_ID,
            NEW,
            new_file.parent,
            previous,
        )

    assert (repo / "refs" / "main").read_text(encoding = "utf-8") == DETACHED
    assert new_file.exists()


def test_main_ref_claim_failure_keeps_the_old_revision(tmp_path, monkeypatch):
    root, repo = _cache_repo(tmp_path)
    old_file = _snapshot_file(repo, OLD, "model.safetensors", b"old")
    new_file = _snapshot_file(repo, NEW, "model.safetensors", b"new")
    previous = _capture(monkeypatch, root)
    real_rename = os.rename
    attempts = 0
    pauses: list[float] = []

    def locked_rename(source, target):
        nonlocal attempts
        if Path(source) == repo / "refs" / "main":
            attempts += 1
            raise PermissionError("refs/main is locked")
        return real_rename(source, target)

    monkeypatch.setattr(snapshot_reclaim, "_IS_WINDOWS", True)
    monkeypatch.setattr(snapshot_reclaim.time, "sleep", pauses.append)
    monkeypatch.setattr(os, "rename", locked_rename)

    with pytest.raises(PermissionError, match = "refs/main is locked"):
        snapshot_reclaim.promote_verified_snapshot(
            "model",
            REPO_ID,
            NEW,
            new_file.parent,
            previous,
        )

    assert (repo / "refs" / "main").read_text(encoding = "utf-8") == OLD
    assert old_file.exists()
    assert attempts == 3
    assert pauses == list(snapshot_reclaim._MAIN_REF_CHANGE_RETRY_DELAYS_SECONDS)


def test_main_ref_claim_does_not_retry_a_non_windows_permission_error(tmp_path, monkeypatch):
    root, repo = _cache_repo(tmp_path)
    new_file = _snapshot_file(repo, NEW, "model.safetensors", b"new")
    previous = _capture(monkeypatch, root)
    real_rename = os.rename
    attempts = 0
    pauses: list[float] = []

    def locked_rename(source, target):
        nonlocal attempts
        if Path(source) == repo / "refs" / "main":
            attempts += 1
            raise PermissionError("refs/main is locked")
        return real_rename(source, target)

    monkeypatch.setattr(snapshot_reclaim, "_IS_WINDOWS", False)
    monkeypatch.setattr(snapshot_reclaim.time, "sleep", pauses.append)
    monkeypatch.setattr(os, "rename", locked_rename)

    with pytest.raises(PermissionError, match = "refs/main is locked"):
        snapshot_reclaim.promote_verified_snapshot(
            "model",
            REPO_ID,
            NEW,
            new_file.parent,
            previous,
        )

    assert attempts == 1
    assert pauses == []
    assert (repo / "refs" / "main").read_text(encoding = "utf-8") == OLD


def test_main_ref_claim_retries_a_transient_windows_lock(tmp_path, monkeypatch):
    root, repo = _cache_repo(tmp_path)
    new_file = _snapshot_file(repo, NEW, "model.safetensors", b"new")
    previous = _capture(monkeypatch, root)
    real_rename = os.rename
    attempts = 0
    pauses: list[float] = []

    def transiently_locked_rename(source, target):
        nonlocal attempts
        if Path(source) == repo / "refs" / "main":
            attempts += 1
            if attempts < 3:
                raise PermissionError("refs/main is temporarily locked")
        return real_rename(source, target)

    monkeypatch.setattr(snapshot_reclaim, "_IS_WINDOWS", True)
    monkeypatch.setattr(snapshot_reclaim.time, "sleep", pauses.append)
    monkeypatch.setattr(os, "rename", transiently_locked_rename)

    snapshot_reclaim.promote_verified_snapshot(
        "model",
        REPO_ID,
        NEW,
        new_file.parent,
        previous,
    )

    assert attempts == 3
    assert pauses == list(snapshot_reclaim._MAIN_REF_CHANGE_RETRY_DELAYS_SECONDS)
    assert (repo / "refs" / "main").read_text(encoding = "utf-8") == NEW


def test_main_ref_claim_retry_does_not_overwrite_a_concurrent_advance(tmp_path, monkeypatch):
    root, repo = _cache_repo(tmp_path)
    new_file = _snapshot_file(repo, NEW, "model.safetensors", b"new")
    previous = _capture(monkeypatch, root)
    main = repo / "refs" / "main"
    real_rename = os.rename
    attempts = 0

    def locked_rename(source, target):
        nonlocal attempts
        if Path(source) == main:
            attempts += 1
            raise PermissionError("refs/main is temporarily locked")
        return real_rename(source, target)

    def advance_main(_delay):
        main.write_text(DETACHED, encoding = "utf-8")

    monkeypatch.setattr(snapshot_reclaim, "_IS_WINDOWS", True)
    monkeypatch.setattr(snapshot_reclaim.time, "sleep", advance_main)
    monkeypatch.setattr(snapshot_reclaim.os, "rename", locked_rename)

    with pytest.raises(snapshot_reclaim.ConcurrentMainRefError, match = "changed during download"):
        snapshot_reclaim.promote_verified_snapshot(
            "model",
            REPO_ID,
            NEW,
            new_file.parent,
            previous,
        )

    assert attempts == 1
    assert main.read_text(encoding = "utf-8") == DETACHED


def test_atomic_ref_replace_preserves_shared_cache_permissions(tmp_path, monkeypatch):
    root, repo = _cache_repo(tmp_path)
    main = repo / "refs" / "main"
    main.chmod(0o664)
    new_file = _snapshot_file(repo, NEW, "model.safetensors", b"new")
    previous = _capture(monkeypatch, root)

    snapshot_reclaim.promote_verified_snapshot(
        "model",
        REPO_ID,
        NEW,
        new_file.parent,
        previous,
    )

    if os.name == "nt":
        assert main.stat().st_mode & stat.S_IWRITE
    else:
        assert stat.S_IMODE(main.stat().st_mode) == 0o664


@pytest.mark.parametrize("previous_revision", [None, OLD])
@pytest.mark.parametrize(
    "link_errno",
    [errno.EPERM, getattr(errno, "EOPNOTSUPP", errno.EPERM)],
)
def test_promotion_falls_back_when_hardlinks_are_unsupported(
    previous_revision, link_errno, tmp_path, monkeypatch
):
    repo, previous = _promotion_cache(tmp_path, monkeypatch, previous_revision)
    if previous_revision is not None:
        main = repo / "refs" / "main"
        main.chmod(0o664)
    new_file = _snapshot_file(repo, NEW, "model.safetensors", b"new")

    def unsupported_link(*_args, **_kwargs):
        raise OSError(link_errno, "hard links are unavailable")

    monkeypatch.setattr(snapshot_reclaim.os, "link", unsupported_link)

    snapshot_reclaim.promote_verified_snapshot(
        "model",
        REPO_ID,
        NEW,
        new_file.parent,
        previous,
    )

    main = repo / "refs" / "main"
    assert main.read_text(encoding = "utf-8") == NEW
    if previous_revision is not None:
        if os.name == "nt":
            assert main.stat().st_mode & stat.S_IWRITE
        else:
            assert stat.S_IMODE(main.stat().st_mode) == 0o664


@pytest.mark.skipif(not snapshot_reclaim._IS_LINUX, reason = "Linux renameat2 integration")
def test_linux_native_noreplace_rename_is_atomic_when_supported(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.write_text(NEW, encoding = "utf-8")

    if not snapshot_reclaim._rename_noreplace(source, destination):
        pytest.skip("RENAME_NOREPLACE is unavailable on this filesystem")

    assert not source.exists()
    assert destination.read_text(encoding = "utf-8") == NEW

    occupied_source = tmp_path / "occupied-source"
    occupied_source.write_text(OLD, encoding = "utf-8")
    with pytest.raises(FileExistsError):
        snapshot_reclaim._rename_noreplace(occupied_source, destination)

    assert occupied_source.read_text(encoding = "utf-8") == OLD
    assert destination.read_text(encoding = "utf-8") == NEW


@pytest.mark.parametrize(
    ("error_code", "expected"),
    [
        (errno.EEXIST, "occupied"),
        (getattr(errno, "EOPNOTSUPP", errno.EINVAL), "unsupported"),
        (errno.EIO, "error"),
    ],
)
def test_libc_noreplace_result_classification(error_code, expected, tmp_path, monkeypatch):
    class FailedRename:
        argtypes = None
        restype = None

        def __call__(self, *_args):
            snapshot_reclaim.ctypes.set_errno(error_code)
            return -1

    operation = FailedRename()
    monkeypatch.setattr(
        snapshot_reclaim.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: SimpleNamespace(renameat2 = operation),
    )
    source = tmp_path / "source"
    destination = tmp_path / "destination"

    if expected == "occupied":
        with pytest.raises(FileExistsError):
            snapshot_reclaim._libc_rename_noreplace("renameat2", [], (), source, destination)
    elif expected == "unsupported":
        assert not snapshot_reclaim._libc_rename_noreplace("renameat2", [], (), source, destination)
    else:
        with pytest.raises(OSError) as caught:
            snapshot_reclaim._libc_rename_noreplace("renameat2", [], (), source, destination)
        assert caught.value.errno == errno.EIO


@pytest.mark.parametrize("native_supported", [True, False])
def test_main_ref_native_rename_source_lifecycle(native_supported, tmp_path, monkeypatch):
    refs = tmp_path / "refs"
    refs.mkdir()
    source = refs / "temporary"
    source.write_text(NEW, encoding = "utf-8")
    real_rename = os.rename

    def unsupported_link(*_args, **_kwargs):
        link_errno = getattr(errno, "EOPNOTSUPP", errno.EPERM)
        raise OSError(link_errno, "hard links are unavailable")

    def native_rename(source_path, destination_path):
        if not native_supported:
            return False
        real_rename(source_path, destination_path)
        return True

    monkeypatch.setattr(snapshot_reclaim.os, "link", unsupported_link)
    monkeypatch.setattr(snapshot_reclaim, "_rename_noreplace", native_rename)

    snapshot_reclaim._create_main_ref(refs, source, try_hardlink = True)

    assert (refs / "main").read_text(encoding = "utf-8") == NEW
    assert source.exists() is (not native_supported)


@pytest.mark.parametrize("previous_revision", [None, OLD])
def test_native_noreplace_does_not_overwrite_an_external_creation(
    previous_revision, tmp_path, monkeypatch
):
    repo, previous = _promotion_cache(tmp_path, monkeypatch, previous_revision)
    refs = repo / "refs"
    main = refs / "main"
    new_file = _snapshot_file(repo, NEW, "model.safetensors", b"new")
    real_native_rename = snapshot_reclaim._rename_noreplace
    advanced = False

    def unsupported_link(*_args, **_kwargs):
        link_errno = getattr(errno, "EOPNOTSUPP", errno.EPERM)
        raise OSError(link_errno, "hard links are unavailable")

    def advance_before_native_rename(source, destination):
        nonlocal advanced
        if destination == main and not advanced:
            advanced = True
            main.write_text(DETACHED, encoding = "utf-8")
        return real_native_rename(source, destination)

    monkeypatch.setattr(snapshot_reclaim.os, "link", unsupported_link)
    monkeypatch.setattr(snapshot_reclaim, "_rename_noreplace", advance_before_native_rename)

    with pytest.raises(snapshot_reclaim.ConcurrentMainRefError):
        snapshot_reclaim.promote_verified_snapshot(
            "model",
            REPO_ID,
            NEW,
            new_file.parent,
            previous,
        )

    assert advanced is True
    assert main.read_text(encoding = "utf-8") == DETACHED
    assert _displaced_refs(repo) == []


def test_windows_noreplace_retry_never_overwrites_a_created_destination(tmp_path, monkeypatch):
    source = tmp_path / "source"
    destination = tmp_path / "main"
    source.write_text(NEW, encoding = "utf-8")
    attempts = 0

    def locked_rename(_source, _destination):
        nonlocal attempts
        attempts += 1
        destination.write_text(DETACHED, encoding = "utf-8")
        raise PermissionError(errno.EACCES, "destination was locked")

    monkeypatch.setattr(snapshot_reclaim, "_IS_WINDOWS", True)
    monkeypatch.setattr(snapshot_reclaim.os, "rename", locked_rename)
    monkeypatch.setattr(snapshot_reclaim.time, "sleep", lambda _delay: None)

    with pytest.raises(snapshot_reclaim.ConcurrentMainRefError, match = "absent"):
        snapshot_reclaim._rename_noreplace(source, destination)

    assert attempts == 1
    assert source.read_text(encoding = "utf-8") == NEW
    assert destination.read_text(encoding = "utf-8") == DETACHED


def test_hardlink_winerror_classification_does_not_trust_lossy_errno():
    unrelated = SimpleNamespace(errno = errno.EINVAL, winerror = 12345)
    unsupported = SimpleNamespace(errno = errno.EIO, winerror = 50)

    assert snapshot_reclaim._hardlink_fallback_allowed(unrelated) is False
    assert snapshot_reclaim._hardlink_fallback_allowed(unsupported) is True


@pytest.mark.parametrize("previous_revision", [None, OLD])
def test_hardlink_fallback_exclusive_create_does_not_overwrite_an_external_advance(
    previous_revision, tmp_path, monkeypatch
):
    repo, previous = _promotion_cache(tmp_path, monkeypatch, previous_revision)
    main = repo / "refs" / "main"
    new_file = _snapshot_file(repo, NEW, "model.safetensors", b"new")
    real_open = os.open
    advanced = False

    def unsupported_link(*_args, **_kwargs):
        link_errno = getattr(errno, "EOPNOTSUPP", errno.EPERM)
        raise OSError(link_errno, "hard links are unavailable")

    def advance_before_exclusive_create(path, flags, *args, **kwargs):
        nonlocal advanced
        if Path(path) == main and flags & os.O_EXCL and not advanced:
            advanced = True
            main.write_text(DETACHED, encoding = "utf-8")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(snapshot_reclaim.os, "link", unsupported_link)
    monkeypatch.setattr(snapshot_reclaim.os, "open", advance_before_exclusive_create)
    monkeypatch.setattr(snapshot_reclaim, "_rename_noreplace", lambda *_args: False)

    with pytest.raises(snapshot_reclaim.ConcurrentMainRefError):
        snapshot_reclaim.promote_verified_snapshot(
            "model",
            REPO_ID,
            NEW,
            new_file.parent,
            previous,
        )

    assert advanced is True
    assert main.read_text(encoding = "utf-8") == DETACHED


@pytest.mark.parametrize("previous_revision", [None, OLD])
def test_hardlink_fallback_detects_a_path_replacement_after_exclusive_create(
    previous_revision, tmp_path, monkeypatch
):
    repo, previous = _promotion_cache(tmp_path, monkeypatch, previous_revision)
    main = repo / "refs" / "main"
    new_file = _snapshot_file(repo, NEW, "model.safetensors", b"new")
    real_open = os.open
    real_write = os.write
    main_fd = None

    def unsupported_link(*_args, **_kwargs):
        link_errno = getattr(errno, "EOPNOTSUPP", errno.EPERM)
        raise OSError(link_errno, "hard links are unavailable")

    def remember_exclusive_main(path, flags, *args, **kwargs):
        nonlocal main_fd
        fd = real_open(path, flags, *args, **kwargs)
        if Path(path) == main and flags & os.O_EXCL:
            main_fd = fd
        return fd

    def replace_before_write(fd, payload):
        if fd == main_fd:
            main.unlink()
            main.write_text(DETACHED, encoding = "utf-8")
        return real_write(fd, payload)

    monkeypatch.setattr(snapshot_reclaim.os, "link", unsupported_link)
    monkeypatch.setattr(snapshot_reclaim.os, "open", remember_exclusive_main)
    monkeypatch.setattr(snapshot_reclaim.os, "write", replace_before_write)
    monkeypatch.setattr(snapshot_reclaim, "_rename_noreplace", lambda *_args: False)

    with pytest.raises(snapshot_reclaim.ConcurrentMainRefError, match = "during promotion"):
        snapshot_reclaim.promote_verified_snapshot(
            "model",
            REPO_ID,
            NEW,
            new_file.parent,
            previous,
        )

    assert main.read_text(encoding = "utf-8") == DETACHED


@pytest.mark.parametrize("previous_revision", [None, OLD])
def test_hardlink_fallback_cleans_an_interrupted_exclusive_write(
    previous_revision, tmp_path, monkeypatch
):
    repo, previous = _promotion_cache(tmp_path, monkeypatch, previous_revision)
    main = repo / "refs" / "main"
    new_file = _snapshot_file(repo, NEW, "model.safetensors", b"new")
    real_write = os.write
    writes = 0

    def unsupported_link(*_args, **_kwargs):
        link_errno = getattr(errno, "EOPNOTSUPP", errno.EPERM)
        raise OSError(link_errno, "hard links are unavailable")

    def fail_first_write(fd, payload):
        nonlocal writes
        writes += 1
        if writes == 1:
            raise OSError(errno.EIO, "interrupted ref write")
        return real_write(fd, payload)

    monkeypatch.setattr(snapshot_reclaim.os, "link", unsupported_link)
    monkeypatch.setattr(snapshot_reclaim.os, "write", fail_first_write)
    monkeypatch.setattr(snapshot_reclaim, "_rename_noreplace", lambda *_args: False)

    with pytest.raises(OSError, match = "interrupted ref write"):
        snapshot_reclaim.promote_verified_snapshot(
            "model",
            REPO_ID,
            NEW,
            new_file.parent,
            previous,
        )

    if previous_revision is None:
        assert not main.exists()
    else:
        assert main.read_text(encoding = "utf-8") == OLD


def test_exclusive_copy_close_failure_restores_the_previous_ref(tmp_path, monkeypatch):
    root, repo = _cache_repo(tmp_path)
    main = repo / "refs" / "main"
    previous = _capture(monkeypatch, root)
    new_file = _snapshot_file(repo, NEW, "model.safetensors", b"new")
    real_open = os.open
    real_close = os.close
    main_fd = None
    close_failed = False

    def unsupported_link(*_args, **_kwargs):
        link_errno = getattr(errno, "EOPNOTSUPP", errno.EPERM)
        raise OSError(link_errno, "hard links are unavailable")

    def remember_exclusive_main(path, flags, *args, **kwargs):
        nonlocal main_fd
        fd = real_open(path, flags, *args, **kwargs)
        if Path(path) == main and flags & os.O_EXCL:
            main_fd = fd
        return fd

    def fail_first_main_close(fd):
        nonlocal close_failed
        if fd == main_fd and not close_failed:
            close_failed = True
            real_close(fd)
            raise OSError(errno.EIO, "delayed ref close failure")
        return real_close(fd)

    monkeypatch.setattr(snapshot_reclaim.os, "link", unsupported_link)
    monkeypatch.setattr(snapshot_reclaim.os, "open", remember_exclusive_main)
    monkeypatch.setattr(snapshot_reclaim.os, "close", fail_first_main_close)
    monkeypatch.setattr(snapshot_reclaim, "_rename_noreplace", lambda *_args: False)

    with pytest.raises(OSError, match = "delayed ref close failure"):
        snapshot_reclaim.promote_verified_snapshot(
            "model",
            REPO_ID,
            NEW,
            new_file.parent,
            previous,
        )

    assert close_failed is True
    assert main.read_text(encoding = "utf-8") == OLD


def test_failed_partial_ref_cleanup_preserves_the_displaced_previous_ref(tmp_path, monkeypatch):
    root, repo = _cache_repo(tmp_path)
    refs = repo / "refs"
    main = refs / "main"
    previous = _capture(monkeypatch, root)
    new_file = _snapshot_file(repo, NEW, "model.safetensors", b"new")
    real_unlink = Path.unlink

    def unsupported_link(*_args, **_kwargs):
        link_errno = getattr(errno, "EOPNOTSUPP", errno.EPERM)
        raise OSError(link_errno, "hard links are unavailable")

    def failed_write(*_args, **_kwargs):
        raise OSError(errno.EIO, "interrupted ref write")

    def fail_partial_main_unlink(path, *args, **kwargs):
        if path == main:
            raise PermissionError(errno.EACCES, "partial ref is locked")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(snapshot_reclaim.os, "link", unsupported_link)
    monkeypatch.setattr(snapshot_reclaim.os, "write", failed_write)
    monkeypatch.setattr(Path, "unlink", fail_partial_main_unlink)
    monkeypatch.setattr(snapshot_reclaim, "_rename_noreplace", lambda *_args: False)

    with pytest.raises(RuntimeError, match = "previous ref remains at") as caught:
        snapshot_reclaim.promote_verified_snapshot(
            "model",
            REPO_ID,
            NEW,
            new_file.parent,
            previous,
        )

    assert isinstance(caught.value.__cause__, snapshot_reclaim._MainRefCleanupError)
    assert main.read_bytes() == b""
    displaced = _displaced_refs(repo)
    assert len(displaced) == 1
    assert displaced[0].read_text(encoding = "utf-8") == OLD


def test_failed_partial_ref_cleanup_is_reported_when_main_was_absent(tmp_path, monkeypatch):
    repo, previous = _promotion_cache(tmp_path, monkeypatch, None)
    refs = repo / "refs"
    main = refs / "main"
    new_file = _snapshot_file(repo, NEW, "model.safetensors", b"new")
    real_unlink = Path.unlink

    def unsupported_link(*_args, **_kwargs):
        link_errno = getattr(errno, "EOPNOTSUPP", errno.EPERM)
        raise OSError(link_errno, "hard links are unavailable")

    def failed_write(*_args, **_kwargs):
        raise OSError(errno.EIO, "interrupted ref write")

    def fail_partial_main_unlink(path, *args, **kwargs):
        if path == main:
            raise PermissionError(errno.EACCES, "partial ref is locked")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(snapshot_reclaim.os, "link", unsupported_link)
    monkeypatch.setattr(snapshot_reclaim.os, "write", failed_write)
    monkeypatch.setattr(Path, "unlink", fail_partial_main_unlink)
    monkeypatch.setattr(snapshot_reclaim, "_rename_noreplace", lambda *_args: False)

    with pytest.raises(snapshot_reclaim._MainRefCleanupError, match = "could not be removed"):
        snapshot_reclaim.promote_verified_snapshot(
            "model",
            REPO_ID,
            NEW,
            new_file.parent,
            previous,
        )

    assert main.read_bytes() == b""
    assert _displaced_refs(repo) == []


def test_hardlink_fallback_preserves_a_mode_changed_at_the_claim_boundary(tmp_path, monkeypatch):
    root, repo = _cache_repo(tmp_path)
    main = repo / "refs" / "main"
    main.chmod(0o600)
    previous = _capture(monkeypatch, root)
    new_file = _snapshot_file(repo, NEW, "model.safetensors", b"new")
    real_rename = os.rename

    def unsupported_link(*_args, **_kwargs):
        link_errno = getattr(errno, "EOPNOTSUPP", errno.EPERM)
        raise OSError(link_errno, "hard links are unavailable")

    def change_mode_before_claim(source, destination):
        if Path(source) == main:
            main.chmod(0o664)
        return real_rename(source, destination)

    monkeypatch.setattr(snapshot_reclaim.os, "link", unsupported_link)
    monkeypatch.setattr(snapshot_reclaim.os, "rename", change_mode_before_claim)

    snapshot_reclaim.promote_verified_snapshot(
        "model",
        REPO_ID,
        NEW,
        new_file.parent,
        previous,
    )

    assert main.read_text(encoding = "utf-8") == NEW
    if os.name != "nt":
        assert stat.S_IMODE(main.stat().st_mode) == 0o664


def test_non_capability_hardlink_errors_do_not_use_the_fallback(tmp_path, monkeypatch):
    root, repo = _cache_repo(tmp_path)
    main = repo / "refs" / "main"
    previous = _capture(monkeypatch, root)
    new_file = _snapshot_file(repo, NEW, "model.safetensors", b"new")

    def failed_link(*_args, **_kwargs):
        raise OSError(errno.EIO, "hardlink I/O failure")

    monkeypatch.setattr(snapshot_reclaim.os, "link", failed_link)

    with pytest.raises(OSError, match = "hardlink I/O failure"):
        snapshot_reclaim.promote_verified_snapshot(
            "model",
            REPO_ID,
            NEW,
            new_file.parent,
            previous,
        )

    assert main.read_text(encoding = "utf-8") == OLD


def test_full_worker_pins_and_promotes_only_after_verification(monkeypatch, tmp_path):
    events: list[str] = []
    snapshot_calls: list[dict] = []
    previous = snapshot_reclaim.PreviousMainRef(tmp_path, OLD, True)

    monkeypatch.setattr(
        hf_download,
        "_model_info_with_retry",
        lambda *_args, **_kwargs: SimpleNamespace(
            sha = NEW,
            siblings = [SimpleNamespace(rfilename = "config.json", size = 2, lfs = None)],
        ),
    )
    monkeypatch.setattr(
        hf_download,
        "capture_previous_main_ref",
        lambda _repo_id: events.append("capture") or previous,
    )
    monkeypatch.setattr(
        hf_download,
        "_verify_completed_download",
        lambda *_args, **_kwargs: events.append("verify") or True,
    )
    monkeypatch.setattr(
        hf_download,
        "promote_verified_snapshot",
        lambda *_args, **_kwargs: events.append("promote"),
    )
    monkeypatch.setattr(download_registry, "prepare_cache_for_transport", lambda *_a, **_k: 0)
    monkeypatch.setattr(download_manifest, "clear_cancel_marker", lambda *_a, **_k: None)
    monkeypatch.setattr(download_manifest, "write_manifest", lambda *_a, **_k: True)
    monkeypatch.setattr(hf_download, "_preflight_disk_space", lambda *_a, **_k: None)

    def snapshot_download(**kwargs):
        events.append("snapshot")
        snapshot_calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download = snapshot_download),
    )

    hf_download._download_snapshot(REPO_ID, None, "http")

    assert events == ["capture", "snapshot", "verify", "promote"]
    assert snapshot_calls[0]["revision"] == NEW


def test_full_worker_allows_metadata_without_a_revision(monkeypatch, tmp_path, capsys):
    snapshot_calls = []
    manifest_calls = []
    monkeypatch.setattr(
        hf_download,
        "_model_info_with_retry",
        lambda *_args, **_kwargs: SimpleNamespace(
            siblings = [SimpleNamespace(rfilename = "config.json", size = 2, lfs = None)],
        ),
    )
    monkeypatch.setattr(
        hf_download,
        "capture_previous_main_ref",
        lambda _repo_id: snapshot_reclaim.PreviousMainRef(tmp_path, OLD, True),
    )
    monkeypatch.setattr(hf_download, "_verify_completed_download", lambda *_a, **_k: True)
    monkeypatch.setattr(
        hf_download,
        "promote_verified_snapshot",
        lambda *_a, **_k: pytest.fail("an unpinned fallback must not use manual promotion"),
    )
    monkeypatch.setattr(download_registry, "prepare_cache_for_transport", lambda *_a, **_k: 0)
    monkeypatch.setattr(download_manifest, "clear_cancel_marker", lambda *_a, **_k: None)
    monkeypatch.setattr(
        download_manifest,
        "write_manifest",
        lambda *args, **kwargs: manifest_calls.append((args, kwargs)) or True,
    )
    monkeypatch.setattr(hf_download, "_preflight_disk_space", lambda *_a, **_k: None)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download = lambda **kwargs: snapshot_calls.append(kwargs) or str(tmp_path)
        ),
    )

    hf_download._download_snapshot(REPO_ID, None, "http")

    assert "revision" not in snapshot_calls[0]
    assert manifest_calls[0][1] == {"commit_hash": None, "metadata_derived": False}
    assert "downloading without an immutable revision" in capsys.readouterr().err


def test_scoped_worker_allows_metadata_without_a_revision(monkeypatch, tmp_path):
    snapshot_calls = []
    monkeypatch.setattr(
        hf_download,
        "_model_info_with_retry",
        lambda *_args, **_kwargs: SimpleNamespace(
            siblings = [SimpleNamespace(rfilename = "weights.bin", size = 4, lfs = None)],
        ),
    )
    monkeypatch.setattr(
        hf_download,
        "capture_previous_main_ref",
        lambda _repo_id: snapshot_reclaim.PreviousMainRef(tmp_path, OLD, True),
    )
    monkeypatch.setattr(hf_download, "_verify_completed_download", lambda *_a, **_k: True)
    monkeypatch.setattr(hf_download, "_protected_blob_hashes", lambda: frozenset())
    monkeypatch.setattr(
        hf_download,
        "promote_verified_snapshot",
        lambda *_a, **_k: pytest.fail("an unpinned fallback must not use manual promotion"),
    )
    monkeypatch.setattr(download_registry, "prepare_cache_for_transport", lambda *_a, **_k: 0)
    monkeypatch.setattr(download_manifest, "clear_cancel_marker", lambda *_a, **_k: None)
    monkeypatch.setattr(download_manifest, "write_manifest", lambda *_a, **_k: True)
    monkeypatch.setattr(hf_download, "_preflight_disk_space", lambda *_a, **_k: None)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download = lambda **kwargs: snapshot_calls.append(kwargs) or str(tmp_path)
        ),
    )

    hf_download._download_scoped_snapshot(
        REPO_ID,
        "@diffusion",
        ["weights.bin"],
        None,
        "http",
    )

    assert "revision" not in snapshot_calls[0]


def test_unsafe_cache_layout_fails_before_download(monkeypatch, tmp_path):
    previous = snapshot_reclaim.PreviousMainRef(
        tmp_path,
        None,
        False,
        "refs/main is not a regular file",
    )
    monkeypatch.setattr(
        hf_download,
        "capture_previous_main_ref",
        lambda _repo_id: previous,
    )

    with pytest.raises(RuntimeError, match = "refs/main is not a regular file"):
        hf_download._snapshot_activation_plan("model", REPO_ID, NEW, True)


def test_verified_download_fails_when_activation_fails(monkeypatch, tmp_path):
    previous = snapshot_reclaim.PreviousMainRef(tmp_path, OLD, True)
    monkeypatch.setattr(
        hf_download,
        "promote_verified_snapshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("busy")),
    )

    with pytest.raises(
        RuntimeError,
        match = "Verified.*present on disk.*automatic activation failed.*remain cached",
    ):
        hf_download._promote_snapshot(
            "model",
            REPO_ID,
            NEW,
            str(tmp_path),
            previous,
            label = f"download for {REPO_ID}",
        )


@pytest.mark.parametrize(
    ("manifest_written", "verified"),
    [(False, True), (True, False)],
)
def test_full_worker_never_promotes_without_attested_verification(
    manifest_written, verified, monkeypatch, tmp_path
):
    promoted = []
    snapshot_calls = []
    monkeypatch.setattr(
        hf_download,
        "_model_info_with_retry",
        lambda *_args, **_kwargs: SimpleNamespace(
            sha = NEW,
            siblings = [SimpleNamespace(rfilename = "config.json", size = 2, lfs = None)],
        ),
    )
    monkeypatch.setattr(
        hf_download,
        "capture_previous_main_ref",
        lambda _repo_id: snapshot_reclaim.PreviousMainRef(tmp_path, OLD, True),
    )
    monkeypatch.setattr(
        hf_download,
        "_verify_completed_download",
        lambda *_args, **_kwargs: verified,
    )
    monkeypatch.setattr(
        hf_download,
        "promote_verified_snapshot",
        lambda *_args, **_kwargs: promoted.append(True),
    )
    monkeypatch.setattr(download_registry, "prepare_cache_for_transport", lambda *_a, **_k: 0)
    monkeypatch.setattr(download_manifest, "clear_cancel_marker", lambda *_a, **_k: None)
    monkeypatch.setattr(
        download_manifest,
        "write_manifest",
        lambda *_a, **_k: manifest_written,
    )
    monkeypatch.setattr(hf_download, "_preflight_disk_space", lambda *_a, **_k: None)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download = lambda **kwargs: snapshot_calls.append(kwargs) or str(tmp_path)
        ),
    )

    if manifest_written and not verified:
        with pytest.raises(RuntimeError, match = "completion could not be attested"):
            hf_download._download_snapshot(REPO_ID, None, "http")
    else:
        hf_download._download_snapshot(REPO_ID, None, "http")

    assert promoted == []
    assert snapshot_calls[0].get("revision") == (NEW if manifest_written else None)


def test_verification_failure_never_moves_main(monkeypatch, tmp_path):
    promoted = []
    monkeypatch.setattr(
        hf_download,
        "_model_info_with_retry",
        lambda *_args, **_kwargs: SimpleNamespace(
            sha = NEW,
            siblings = [SimpleNamespace(rfilename = "config.json", size = 2, lfs = None)],
        ),
    )
    monkeypatch.setattr(
        hf_download,
        "capture_previous_main_ref",
        lambda _repo_id: snapshot_reclaim.PreviousMainRef(tmp_path, OLD, True),
    )
    monkeypatch.setattr(
        hf_download,
        "_verify_completed_download",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(SystemExit(1)),
    )
    monkeypatch.setattr(
        hf_download,
        "promote_verified_snapshot",
        lambda *_args, **_kwargs: promoted.append(True),
    )
    monkeypatch.setattr(download_registry, "prepare_cache_for_transport", lambda *_a, **_k: 0)
    monkeypatch.setattr(download_manifest, "clear_cancel_marker", lambda *_a, **_k: None)
    monkeypatch.setattr(download_manifest, "write_manifest", lambda *_a, **_k: True)
    monkeypatch.setattr(hf_download, "_preflight_disk_space", lambda *_a, **_k: None)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download = lambda **_kwargs: str(tmp_path)),
    )

    with pytest.raises(SystemExit):
        hf_download._download_snapshot(REPO_ID, None, "http")

    assert promoted == []


@pytest.mark.parametrize(
    ("manifest_written", "verified", "should_promote"),
    [
        (True, True, True),
        (False, True, False),
        (True, False, False),
    ],
)
def test_scoped_worker_pins_and_promotes_only_an_attested_snapshot(
    manifest_written, verified, should_promote, monkeypatch, tmp_path
):
    previous = snapshot_reclaim.PreviousMainRef(tmp_path, OLD, True)
    captured = []
    promoted = []
    written = []
    snapshot_calls = []
    snapshot = tmp_path / "models--Org--Model" / "snapshots" / NEW
    snapshot.mkdir(parents = True)

    monkeypatch.setattr(
        hf_download,
        "_model_info_with_retry",
        lambda *_args, **_kwargs: SimpleNamespace(
            sha = NEW,
            siblings = [SimpleNamespace(rfilename = "weights.bin", size = 4, lfs = None)],
        ),
    )

    def capture(repo_id):
        captured.append(repo_id)
        return previous

    monkeypatch.setattr(hf_download, "capture_previous_main_ref", capture)
    monkeypatch.setattr(
        hf_download,
        "promote_verified_snapshot",
        lambda *args, **kwargs: promoted.append((args, kwargs)),
    )
    monkeypatch.setattr(
        hf_download,
        "_verify_completed_download",
        lambda *_args, **_kwargs: verified,
    )
    monkeypatch.setattr(hf_download, "_protected_blob_hashes", lambda: frozenset())
    monkeypatch.setattr(download_registry, "prepare_cache_for_transport", lambda *_a, **_k: 0)
    monkeypatch.setattr(download_manifest, "clear_cancel_marker", lambda *_a, **_k: None)
    monkeypatch.setattr(
        download_manifest,
        "write_manifest",
        lambda *args, **kwargs: written.append((args, kwargs)) or manifest_written,
    )
    monkeypatch.setattr(hf_download, "_preflight_disk_space", lambda *_a, **_k: None)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download = lambda **kwargs: snapshot_calls.append(kwargs) or str(snapshot)
        ),
    )

    if manifest_written and not verified:
        with pytest.raises(RuntimeError, match = "completion could not be attested"):
            hf_download._download_scoped_snapshot(
                "Org/Model",
                "@diffusion",
                ["weights.bin"],
                None,
                "http",
            )
    else:
        hf_download._download_scoped_snapshot(
            "Org/Model",
            "@diffusion",
            ["weights.bin"],
            None,
            "http",
        )

    assert snapshot_calls[0].get("revision") == (NEW if manifest_written else None)
    assert written[0][1] == {"commit_hash": NEW, "metadata_derived": True}
    assert captured == (["Org/Model"] if manifest_written else [])
    assert bool(promoted) is should_promote
    if should_promote:
        assert promoted == [(("model", "Org/Model", NEW, str(snapshot), previous), {})]


@pytest.mark.parametrize(
    ("manifest_written", "verified", "completion_written", "should_promote"),
    [
        (True, True, True, True),
        (False, True, True, False),
        (True, False, True, False),
        (True, True, False, True),
    ],
)
def test_dataset_worker_promotes_only_an_attested_exact_snapshot(
    manifest_written, verified, completion_written, should_promote, monkeypatch, tmp_path
):
    previous = snapshot_reclaim.PreviousMainRef(tmp_path, OLD, True)
    captured = []
    promoted = []
    snapshot_calls = []
    snapshot = tmp_path / "datasets--Org--Data" / "snapshots" / NEW
    snapshot.mkdir(parents = True)

    monkeypatch.setattr(
        hf_download,
        "_dataset_info_with_retry",
        lambda *_args, **_kwargs: SimpleNamespace(
            sha = NEW,
            siblings = [SimpleNamespace(rfilename = "data.parquet", size = 4)],
        ),
    )

    def capture(repo_id, *, repo_type = "model"):
        captured.append((repo_id, repo_type))
        return previous

    monkeypatch.setattr(hf_download, "capture_previous_main_ref", capture)
    monkeypatch.setattr(
        hf_download,
        "promote_verified_snapshot",
        lambda *args, **kwargs: promoted.append((args, kwargs)),
    )
    monkeypatch.setattr(
        hf_download,
        "_verify_completed_download",
        lambda *_args, **_kwargs: verified,
    )
    monkeypatch.setattr(
        hf_download,
        "_write_dataset_completion_from_metadata",
        lambda *_args, **_kwargs: completion_written,
    )
    monkeypatch.setattr(download_registry, "prepare_cache_for_transport", lambda *_a, **_k: 0)
    monkeypatch.setattr(download_manifest, "clear_cancel_marker", lambda *_a, **_k: None)
    monkeypatch.setattr(
        download_manifest,
        "write_manifest",
        lambda *_args, **_kwargs: manifest_written,
    )
    monkeypatch.setattr(hf_download, "_preflight_disk_space", lambda *_a, **_k: None)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download = lambda **kwargs: snapshot_calls.append(kwargs) or str(snapshot)
        ),
    )

    if manifest_written and not verified:
        with pytest.raises(RuntimeError, match = "completion could not be attested"):
            hf_download._download_dataset("Org/Data", None, "http")
    else:
        hf_download._download_dataset("Org/Data", None, "http")

    assert snapshot_calls[0].get("revision") == (NEW if manifest_written else None)
    assert captured == ([("Org/Data", "dataset")] if manifest_written else [])
    assert bool(promoted) is should_promote
    if should_promote:
        assert promoted == [(("dataset", "Org/Data", NEW, str(snapshot), previous), {})]
