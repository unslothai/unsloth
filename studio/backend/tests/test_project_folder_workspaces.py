# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Existing-folder project ownership, persistence, and execution boundaries."""

import base64
import hashlib
import hmac
import json
import os
import shutil
import threading
import time
from pathlib import Path

import pytest
from fastapi import HTTPException

from core.inference import tools
from routes import chat_history, inference
from storage import studio_db
from utils import native_path_leases as leases


_LEASE_SECRET = b"project-folder-test-secret-value"


@pytest.fixture(autouse = True)
def _isolated_project_state(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "managed-projects"))
    monkeypatch.setenv(
        leases.LEASE_SECRET_ENV,
        base64.urlsafe_b64encode(_LEASE_SECRET).decode("ascii").rstrip("="),
    )
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    monkeypatch.setattr(leases, "_CACHED_LEASE_SECRET", None)
    leases._USED_NONCES.clear()
    tools._workdirs.clear()
    with tools._sessions_free:
        tools._active_sessions.clear()
        tools._removing_sessions.clear()
    yield
    leases._USED_NONCES.clear()
    tools._workdirs.clear()
    with tools._sessions_free:
        tools._active_sessions.clear()
        tools._removing_sessions.clear()
        tools._sessions_free.notify_all()


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _sign_folder(
    path: Path,
    *,
    operation: str = "open-project",
    path_kind: str = "document-folder",
    path_type: str = "directory",
    nonce: str | None = None,
) -> str:
    metadata = path.stat()
    now_ms = int(time.time() * 1000)
    payload = {
        "version": 1,
        "operation": operation,
        "canonical_path": str(path.resolve()),
        "path_kind": path_kind,
        "path_type": path_type,
        "source_kind": "dialog",
        "token_id_hash": hashlib.sha256(b"project_path_token").hexdigest(),
        "issued_at_ms": now_ms,
        "expires_at_ms": now_ms + 120_000,
        "nonce": nonce or os.urandom(16).hex(),
        "display_label": path.name,
        "size_bytes": None,
        "modified_ms": None,
        "device_id": format(metadata.st_dev, "x"),
        "file_id": format(metadata.st_ino, "x"),
        "change_time_ns": str(metadata.st_ctime_ns),
    }
    payload_b64 = _b64(json.dumps(payload, separators = (",", ":")).encode("utf-8"))
    signature = hmac.new(
        _LEASE_SECRET,
        payload_b64.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return f"{payload_b64}.{_b64(signature)}"


def _managed_project(project_id: str, name: str = "Managed") -> dict:
    return {
        "id": project_id,
        "name": name,
        "instructions": "",
        "archived": False,
        "createdAt": 1_700_000_000_000,
        "updatedAt": 1_700_000_000_000,
    }


def _folder_claim(project_id: str, folder: Path, **extra) -> dict:
    metadata = folder.stat()
    return {
        "id": project_id,
        "name": folder.name,
        "instructions": "",
        "workspacePath": str(folder.resolve()),
        "workspaceDeviceId": metadata.st_dev,
        "workspaceFileId": metadata.st_ino,
        "createdAt": 1_700_000_000_000,
        "updatedAt": 1_700_000_000_001,
        **extra,
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"operation": "link-documents"},
        {"path_kind": "attachment"},
        {"path_type": "file"},
    ],
)
def test_project_folder_lease_requires_exact_purpose(tmp_path, overrides):
    folder = tmp_path / "repository"
    folder.mkdir()

    with pytest.raises(HTTPException) as caught:
        chat_history._resolve_project_folder_lease(_sign_folder(folder, **overrides))

    assert caught.value.status_code == 400


def test_folder_workspace_rejects_a_windows_junction_boundary(tmp_path, monkeypatch):
    folder = tmp_path / "repository"
    folder.mkdir()
    monkeypatch.setattr(
        Path,
        "is_junction",
        lambda self: self == folder,
        raising = False,
    )

    with pytest.raises(studio_db.ProjectWorkspaceError, match = "link"):
        studio_db._ensure_folder_workspace(
            str(folder),
            str(folder.stat().st_dev),
            str(folder.stat().st_ino),
        )


def test_folder_claim_requires_a_real_write_probe(tmp_path, monkeypatch):
    folder = tmp_path / "repository"
    folder.mkdir()
    project = _folder_claim("project-write-probe", folder)
    calls = []

    def deny_probe(resolved, *args, **kwargs):
        calls.append(str(resolved))
        raise PermissionError("ACL denied")

    monkeypatch.setattr(studio_db, "_probe_folder_writable", deny_probe)

    with pytest.raises(studio_db.ProjectWorkspaceError, match = "ACL denied"):
        studio_db.claim_chat_project_folder(project)

    assert calls == [str(folder.resolve())]
    assert studio_db.get_chat_project(project["id"]) is None


def test_folder_claim_probe_fails_closed_when_the_path_becomes_a_symlink(tmp_path, monkeypatch):
    folder = tmp_path / "repository"
    redirect = tmp_path / "redirect"
    moved = tmp_path / "repository-original"
    folder.mkdir()
    redirect.mkdir()
    project = _folder_claim("project-probe-race", folder)
    real_probe = studio_db._probe_folder_writable

    def swap_then_probe(resolved, *args, **kwargs):
        folder.rename(moved)
        folder.symlink_to(redirect, target_is_directory = True)
        return real_probe(resolved, *args, **kwargs)

    monkeypatch.setattr(studio_db, "_probe_folder_writable", swap_then_probe)

    with pytest.raises(studio_db.ProjectWorkspaceError):
        studio_db.claim_chat_project_folder(project)

    assert studio_db.get_chat_project(project["id"]) is None
    assert list(redirect.iterdir()) == []
    folder.unlink()
    moved.rmdir()


def test_workspace_health_does_not_probe_or_touch_the_folder(tmp_path, monkeypatch):
    folder = tmp_path / "repository"
    folder.mkdir()
    metadata = folder.stat()
    calls = []

    def deny_probe(*args, **kwargs):
        calls.append(args[0] if args else None)
        raise AssertionError("health checks must not write to the selected folder")

    monkeypatch.setattr(studio_db, "_probe_folder_writable", deny_probe)

    assert studio_db._ensure_folder_workspace(
        str(folder),
        str(metadata.st_dev),
        str(metadata.st_ino),
        str(metadata.st_ctime_ns),
    ) == str(folder.resolve())
    assert calls == []


@pytest.mark.parametrize(
    "request_type,fields",
    [
        (
            chat_history.ChatProjectCreate,
            {
                "id": "project-whitespace",
                "name": "   ",
                "createdAt": 1_700_000_000_000,
                "updatedAt": 1_700_000_000_000,
            },
        ),
        (
            chat_history.OpenProjectFolderRequest,
            {"nativePathLease": "lease", "name": "   "},
        ),
        (chat_history.ChatProjectPatch, {"name": "   "}),
    ],
)
def test_project_names_cannot_be_only_whitespace(request_type, fields):
    with pytest.raises(ValueError, match = "Project name cannot be empty"):
        request_type(**fields)


def test_project_folder_lease_is_single_use(tmp_path):
    folder = tmp_path / "repository"
    folder.mkdir()
    lease = _sign_folder(folder)

    assert chat_history._resolve_project_folder_lease(lease).canonical_path == folder.resolve()
    with pytest.raises(HTTPException) as caught:
        chat_history._resolve_project_folder_lease(lease)

    assert caught.value.status_code == 400
    assert "already used" in str(caught.value.detail)


def test_project_folder_lease_rejects_ctime_replacement(tmp_path):
    folder = tmp_path / "repository"
    folder.mkdir()
    lease = _sign_folder(folder)
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    folder.rmdir()
    replacement.rename(folder)

    with pytest.raises(HTTPException) as caught:
        chat_history._resolve_project_folder_lease(lease)

    assert caught.value.status_code == 400


def test_open_folder_project_persists_path_identity_and_public_state(tmp_path):
    folder = tmp_path / "repository"
    folder.mkdir()

    opened = chat_history.open_project_folder(
        chat_history.OpenProjectFolderRequest(
            nativePathLease = _sign_folder(folder),
            name = "Repository",
        ),
        current_subject = "tester",
    )
    stored = studio_db.get_chat_project(opened.id)

    assert opened.workspaceKind == "folder"
    assert opened.workspacePath == str(folder.resolve())
    assert opened.workspaceAvailable is True
    assert "rootPath" not in opened.model_dump()
    assert stored is not None
    assert stored["sandboxPath"] == str(folder.resolve())
    assert stored["workspaceDeviceId"] == str(folder.stat().st_dev)
    assert stored["workspaceFileId"] == str(folder.stat().st_ino)

    monkeypatch_project = stored["id"]
    studio_db._schema_ready = False
    reloaded = studio_db.get_chat_project(monkeypatch_project)
    assert reloaded is not None
    assert reloaded["workspacePath"] == str(folder.resolve())
    assert reloaded["workspaceAvailable"] is True


def test_reopening_same_folder_reuses_the_project(tmp_path):
    folder = tmp_path / "repository"
    folder.mkdir()

    first = chat_history.open_project_folder(
        chat_history.OpenProjectFolderRequest(
            nativePathLease = _sign_folder(folder), name = "Repository"
        ),
        current_subject = "tester",
    )
    second = chat_history.open_project_folder(
        chat_history.OpenProjectFolderRequest(
            nativePathLease = _sign_folder(folder), name = "Repository"
        ),
        current_subject = "tester",
    )

    assert second.id == first.id
    assert len(studio_db.list_chat_projects()) == 1


def test_managed_create_collision_cannot_mutate_a_folder_project(tmp_path):
    folder = tmp_path / "project-collision"
    folder.mkdir()
    original = studio_db.claim_chat_project_folder(_folder_claim("project-collision", folder))
    assert original is not None

    with pytest.raises(HTTPException) as caught:
        chat_history.save_project(
            chat_history.ChatProjectCreate(
                **_managed_project("project-collision", name = "Replacement")
            ),
            current_subject = "tester",
        )

    assert caught.value.status_code == 409
    stored = studio_db.get_chat_project("project-collision")
    assert stored is not None
    assert stored["name"] == folder.name
    assert stored["managedRootPath"] is None
    assert not (folder / "sandbox").exists()
    attempted_root = Path(
        studio_db._default_project_root(_managed_project("project-collision", name = "Replacement"))
    )
    assert not attempted_root.exists()


def test_legacy_upsert_never_materializes_inside_an_existing_folder_project(tmp_path):
    folder = tmp_path / "repository"
    folder.mkdir()
    original = studio_db.claim_chat_project_folder(_folder_claim("project-upsert-folder", folder))
    assert original is not None

    updated = studio_db.upsert_chat_project(
        {
            "id": original["id"],
            "name": "Imported metadata",
            "instructions": "Keep the selected repository.",
            "archived": False,
            "createdAt": original["createdAt"],
            "updatedAt": original["updatedAt"] + 1,
        }
    )

    assert updated["workspaceKind"] == "folder"
    assert updated["workspacePath"] == str(folder.resolve())
    assert updated["managedRootPath"] is None
    assert not (folder / "sandbox").exists()


def test_legacy_upsert_preserves_folder_instructions_when_import_omits_them(tmp_path):
    folder = tmp_path / "repository"
    folder.mkdir()
    original = studio_db.claim_chat_project_folder(
        _folder_claim(
            "project-upsert-instructions",
            folder,
            instructions = "Keep this repository's conventions.",
        )
    )
    assert original is not None

    updated = studio_db.upsert_chat_project(
        {
            "id": original["id"],
            "name": "Imported metadata",
            "archived": False,
            "createdAt": original["createdAt"],
            "updatedAt": original["updatedAt"] + 1,
        }
    )

    assert updated["instructions"] == "Keep this repository's conventions."


def test_distinct_project_ids_with_the_same_prefix_get_distinct_managed_roots():
    first = studio_db._default_project_root(
        _managed_project("shared-prefix-first", name = "Repository")
    )
    second = studio_db._default_project_root(
        _managed_project("shared-prefix-second", name = "Repository")
    )

    assert first != second


def test_orphan_lookup_recognizes_hashed_managed_workspace_names():
    project = _managed_project("project-hashed-orphan", name = "Repository")
    root = Path(studio_db._default_project_root(project))
    sandbox = root / "sandbox"
    sandbox.mkdir(parents = True)

    assert tools._orphaned_project_workdir(project["id"]) == str(sandbox.resolve())


def test_failed_create_cleanup_cannot_remove_a_replacement_workspace(tmp_path):
    root = tmp_path / "managed"
    first = studio_db._prepare_project_workspace(str(root))
    assert first.created_root is True

    (root / "sandbox").rmdir()
    (root / studio_db._PROJECT_WORKSPACE_IDENTITY_FILE).unlink()
    root.rmdir()
    replacement = studio_db._prepare_project_workspace(str(root))
    assert replacement.marker_token != first.marker_token

    studio_db._remove_empty_project_workspace_if_unclaimed(first)

    assert root.is_dir()
    assert (root / "sandbox").is_dir()


def test_failed_create_cleanup_never_follows_a_replacement_symlink(tmp_path):
    root = tmp_path / "managed"
    prepared = studio_db._prepare_project_workspace(str(root))
    original = tmp_path / "original-managed"
    root.rename(original)
    victim = tmp_path / "unrelated"
    (victim / "sandbox").mkdir(parents = True)
    root.symlink_to(victim, target_is_directory = True)

    studio_db._remove_empty_project_workspace_if_unclaimed(prepared)

    assert root.is_symlink()
    assert victim.is_dir()
    assert (victim / "sandbox").is_dir()
    assert original.is_dir()


def test_failed_create_cleanup_rejects_a_linked_identity_marker(tmp_path):
    root = tmp_path / "managed"
    prepared = studio_db._prepare_project_workspace(str(root))
    marker = root / studio_db._PROJECT_WORKSPACE_IDENTITY_FILE
    marker.unlink()
    external_marker = tmp_path / "external-marker"
    external_marker.write_text(prepared.marker_token, encoding = "ascii")
    marker.symlink_to(external_marker)

    studio_db._remove_empty_project_workspace_if_unclaimed(prepared)

    assert root.is_dir()
    assert marker.is_symlink()
    assert external_marker.read_text(encoding = "ascii") == prepared.marker_token


def test_managed_workspace_delete_never_follows_a_replacement_symlink(tmp_path):
    project = _managed_project("project-delete-link")
    root = Path(studio_db._default_project_root(project))
    root.parent.mkdir(parents = True)
    victim = tmp_path / "unrelated"
    (victim / "sandbox").mkdir(parents = True)
    (victim / "keep.txt").write_text("keep", encoding = "utf-8")
    root.symlink_to(victim, target_is_directory = True)

    studio_db.delete_project_workspace(
        {
            **project,
            "workspaceKind": "managed",
            "rootPath": str(root),
            "managedRootPath": str(root),
        }
    )

    assert root.is_symlink()
    assert victim.is_dir()
    assert (victim / "keep.txt").read_text(encoding = "utf-8") == "keep"


def test_immediate_managed_delete_rejects_a_replacement_directory(tmp_path, monkeypatch):
    """A same-name directory installed after capture is not ours to remove."""
    project = _managed_project("project-delete-replacement")
    managed = studio_db.upsert_chat_project(project)
    root = Path(managed["managedRootPath"])
    original = tmp_path / "original-managed"

    original_delete = studio_db._delete_project_workspace

    def replace_before_delete(target):
        root.rename(original)
        root.mkdir()
        (root / "sandbox").mkdir()
        (root / "replacement.txt").write_text("keep", encoding = "utf-8")
        original_delete(target)

    monkeypatch.setattr(studio_db, "_delete_project_workspace", replace_before_delete)

    deleted = studio_db.delete_chat_project(project["id"], delete_files = True)

    assert deleted is not None
    assert root.is_dir()
    assert (root / "sandbox").is_dir()
    assert (root / "replacement.txt").read_text(encoding = "utf-8") == "keep"
    assert original.is_dir()


def test_managed_delete_does_not_remove_a_quarantine_replacement(tmp_path, monkeypatch):
    project = studio_db.upsert_chat_project(_managed_project("project-quarantine-race"))
    root = Path(project["managedRootPath"])
    original = tmp_path / "quarantined-original"
    original_remove = studio_db._remove_directory_contents_fd
    swapped = False

    def swap_after_quarantine(directory_fd, root_device_id):
        nonlocal swapped
        if not swapped:
            quarantine = next(root.parent.glob(f".{root.name}.delete-*"))
            quarantine.rename(original)
            quarantine.mkdir()
            (quarantine / "sandbox").mkdir()
            (quarantine / "replacement.txt").write_text("keep", encoding = "utf-8")
            swapped = True
        return original_remove(directory_fd, root_device_id)

    monkeypatch.setattr(studio_db, "_remove_directory_contents_fd", swap_after_quarantine)

    studio_db.delete_chat_project(project["id"], delete_files = True)

    assert swapped is True
    assert (root / "replacement.txt").read_text(encoding = "utf-8") == "keep"
    assert original.is_dir()


def test_quarantine_cleanup_rejects_a_nested_mount_boundary(tmp_path, monkeypatch):
    root = tmp_path / "quarantine"
    child = root / "mounted"
    child.mkdir(parents = True)
    (child / "keep.txt").write_text("keep", encoding = "utf-8")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    root_fd = os.open(root, flags)
    original_fstat = os.fstat
    child_probe = False

    def report_foreign_device(fd):
        nonlocal child_probe
        metadata = original_fstat(fd)
        if not child_probe:
            child_probe = True
            values = list(metadata)
            values[2] += 1
            return os.stat_result(values)
        return metadata

    monkeypatch.setattr(os, "fstat", report_foreign_device)
    try:
        with pytest.raises(OSError, match = "mount boundary"):
            studio_db._remove_directory_contents_fd(root_fd, root.stat().st_dev)
    finally:
        os.close(root_fd)

    assert (child / "keep.txt").read_text(encoding = "utf-8") == "keep"


def test_direct_managed_workspace_delete_rejects_a_forged_orphan_bypass(tmp_path, monkeypatch):
    project = _managed_project("deniedid")
    root = tmp_path / "Managed-deniedid"
    (root / "sandbox").mkdir(parents = True)
    monkeypatch.setattr(studio_db, "_denied_path_prefixes", lambda: [str(tmp_path.resolve())])

    studio_db.delete_project_workspace(
        {
            **project,
            "workspaceKind": "managed",
            "rootPath": str(root),
            "managedRootPath": str(root),
            "_recordedOrphan": True,
        }
    )

    assert root.is_dir()
    assert (root / "sandbox").is_dir()


def test_managed_workspace_delete_rejects_a_junction_boundary(tmp_path, monkeypatch):
    project = _managed_project("junctionid")
    root = tmp_path / "Managed-junctionid"
    (root / "sandbox").mkdir(parents = True)
    monkeypatch.setattr(studio_db, "_denied_path_prefixes", lambda: [])
    monkeypatch.setattr(
        Path,
        "is_junction",
        lambda self: self == root,
        raising = False,
    )

    studio_db.delete_project_workspace(
        {
            **project,
            "workspaceKind": "managed",
            "rootPath": str(root),
            "managedRootPath": str(root),
        }
    )

    assert root.is_dir()


def test_managed_create_retries_when_prepared_identity_disappears(tmp_path, monkeypatch):
    project = _managed_project("project-create-retry")
    original_prepare = studio_db._prepare_project_workspace
    prepare_count = 0

    def prepare_then_remove(path, **kwargs):
        nonlocal prepare_count
        prepared = original_prepare(path, **kwargs)
        prepare_count += 1
        if prepare_count == 1:
            (Path(prepared.root_path) / "sandbox").rmdir()
            (Path(prepared.root_path) / studio_db._PROJECT_WORKSPACE_IDENTITY_FILE).unlink()
            Path(prepared.root_path).rmdir()
        return prepared

    monkeypatch.setattr(studio_db, "_prepare_project_workspace", prepare_then_remove)

    created = studio_db.create_chat_project(project)

    assert prepare_count == 2
    assert Path(created["rootPath"]).is_dir()
    assert Path(created["sandboxPath"]).is_dir()


def test_managed_create_does_not_adopt_a_root_won_between_check_and_mkdir(tmp_path, monkeypatch):
    project = _managed_project("project-create-reservation-race")
    expected = Path(studio_db._default_project_root(project))
    original_mkdir = Path.mkdir
    injected = False

    def race_mkdir(path, *args, **kwargs):
        nonlocal injected
        if path == expected and not injected and not kwargs.get("exist_ok"):
            injected = True
            original_mkdir(path, *args, **kwargs)
            (path / "foreign.txt").write_text("do not adopt", encoding = "utf-8")
            raise FileExistsError(str(path))
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", race_mkdir)

    created = studio_db.create_chat_project(project)

    assert injected is True
    assert Path(created["rootPath"]) != expected
    assert (expected / "foreign.txt").read_text(encoding = "utf-8") == "do not adopt"


def test_invalid_managed_workspace_marker_is_never_adopted():
    project = _managed_project("project-invalid-marker")
    root = Path(studio_db._default_project_root(project))
    (root / "sandbox").mkdir(parents = True)
    (root / studio_db._PROJECT_WORKSPACE_IDENTITY_FILE).write_text(
        "not-a-valid-marker",
        encoding = "ascii",
    )

    created = studio_db.create_chat_project(project)

    assert created["rootPath"] != str(root)
    assert (root / studio_db._PROJECT_WORKSPACE_IDENTITY_FILE).read_text(encoding = "ascii") == (
        "not-a-valid-marker"
    )


def test_managed_workspace_rejects_a_same_name_replacement_with_a_copied_marker(tmp_path):
    project = studio_db.create_chat_project(_managed_project("project-root-identity"))
    root = Path(project["managedRootPath"])
    original = tmp_path / "original-managed"
    marker = (root / studio_db._PROJECT_WORKSPACE_IDENTITY_FILE).read_text(encoding = "ascii")

    root.rename(original)
    root.mkdir()
    (root / "sandbox").mkdir()
    (root / studio_db._PROJECT_WORKSPACE_IDENTITY_FILE).write_text(marker, encoding = "ascii")

    with pytest.raises(studio_db.ProjectWorkspaceError, match = "identity changed"):
        studio_db.ensure_chat_project_workspace(project["id"])

    assert root.is_dir()
    assert original.is_dir()


def test_legacy_managed_workspace_identity_is_bound_during_schema_migration():
    project = studio_db.create_chat_project(_managed_project("project-legacy-bind"))
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE chat_projects SET root_device_id = NULL, root_file_id = NULL, "
            "root_change_time_ns = NULL, root_marker_token = NULL WHERE id = ?",
            (project["id"],),
        )
        conn.commit()
    finally:
        conn.close()

    studio_db._schema_ready = False
    migrated = studio_db.ensure_chat_project_workspace(project["id"])

    assert migrated is not None
    assert migrated["managedRootPath"] == project["managedRootPath"]
    assert migrated["rootPath"] == project["rootPath"]


def test_legacy_managed_workspace_without_marker_gets_fresh_root(tmp_path):
    project = studio_db.create_chat_project(_managed_project("project-legacy-fresh"))
    legacy_root = Path(project["managedRootPath"])
    (legacy_root / studio_db._PROJECT_WORKSPACE_IDENTITY_FILE).unlink()
    (legacy_root / "legacy-data.txt").write_text("keep", encoding = "utf-8")
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE chat_projects SET root_device_id = NULL, root_file_id = NULL, "
            "root_change_time_ns = NULL, root_marker_token = NULL WHERE id = ?",
            (project["id"],),
        )
        conn.commit()
    finally:
        conn.close()

    studio_db._schema_ready = False
    migrated = studio_db.ensure_chat_project_workspace(project["id"])

    assert migrated is not None
    assert Path(migrated["managedRootPath"]) != legacy_root
    assert (legacy_root / "legacy-data.txt").read_text(encoding = "utf-8") == "keep"
    assert any(
        record_id == project["id"] and root == str(legacy_root)
        for record_id, _workspace, root, _pending, _is_chat in tools.list_orphaned_projects()
    )
    assert tools._recorded_project_workdir(project["id"]) is None
    assert tools._orphaned_project_workdir(project["id"]) is None


def test_deleting_a_migrated_legacy_workspace_keeps_its_recovery_record(tmp_path):
    project = studio_db.create_chat_project(_managed_project("project-legacy-delete"))
    legacy_root = Path(project["managedRootPath"])
    (legacy_root / studio_db._PROJECT_WORKSPACE_IDENTITY_FILE).unlink()
    (legacy_root / "legacy-data.txt").write_text("keep", encoding = "utf-8")
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE chat_projects SET root_device_id = NULL, root_file_id = NULL, "
            "root_change_time_ns = NULL, root_marker_token = NULL WHERE id = ?",
            (project["id"],),
        )
        conn.commit()
    finally:
        conn.close()

    studio_db._schema_ready = False
    deleted = studio_db.delete_chat_project(project["id"], delete_files = True)

    assert deleted is not None
    assert (legacy_root / "legacy-data.txt").read_text(encoding = "utf-8") == "keep"
    assert any(
        record_id == project["id"] and root == str(legacy_root)
        for record_id, _workspace, root, _pending, _is_chat in tools.list_orphaned_projects()
    )


def test_unbound_managed_project_allocates_a_fresh_root_without_adopting_foreign_path(tmp_path):
    project = studio_db.create_chat_project(_managed_project("project-unbound-root"))
    original = Path(project["managedRootPath"])
    shutil.rmtree(original)
    original.mkdir()
    (original / "foreign.txt").write_text("keep", encoding = "utf-8")

    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE chat_projects SET root_path = NULL, root_device_id = NULL, "
            "root_file_id = NULL, root_change_time_ns = NULL, root_marker_token = NULL "
            "WHERE id = ?",
            (project["id"],),
        )
        conn.commit()
    finally:
        conn.close()

    initialized = studio_db.ensure_chat_project_workspace(project["id"])

    assert initialized is not None
    assert Path(initialized["managedRootPath"]) != original
    assert (original / "foreign.txt").read_text(encoding = "utf-8") == "keep"


def test_upsert_persists_identity_for_an_existing_unbound_managed_project():
    project = studio_db.create_chat_project(_managed_project("project-upsert-unbound"))
    old_root = Path(project["managedRootPath"])
    shutil.rmtree(old_root)
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE chat_projects SET root_path = NULL, root_device_id = NULL, "
            "root_file_id = NULL, root_change_time_ns = NULL, root_marker_token = NULL "
            "WHERE id = ?",
            (project["id"],),
        )
        conn.commit()
    finally:
        conn.close()

    restored = studio_db.upsert_chat_project(_managed_project(project["id"]))
    root = Path(restored["managedRootPath"])
    device_id, file_id = studio_db._workspace_identity(root)

    assert restored["_managedRootDeviceId"] == device_id
    assert restored["_managedRootFileId"] == file_id
    assert restored["_managedRootChangeTimeNs"] == studio_db._workspace_change_time(root)
    assert restored["_managedRootMarkerToken"] == studio_db._read_project_workspace_marker(root)


def test_upsert_does_not_overwrite_a_concurrent_folder_claim(tmp_path, monkeypatch):
    project = studio_db.create_chat_project(_managed_project("project-upsert-folder-race"))
    old_root = Path(project["managedRootPath"])
    shutil.rmtree(old_root)
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE chat_projects SET root_path = NULL, root_device_id = NULL, "
            "root_file_id = NULL, root_change_time_ns = NULL, root_marker_token = NULL "
            "WHERE id = ?",
            (project["id"],),
        )
        conn.commit()
    finally:
        conn.close()
    folder = tmp_path / "upsert-claimed-folder"
    folder.mkdir()
    original_prepare = studio_db._prepare_project_workspace
    prepared_candidates = []

    def race_prepare(path, **kwargs):
        prepared = original_prepare(path, **kwargs)
        prepared_candidates.append(prepared)
        claimed = studio_db.claim_chat_project_folder(
            _folder_claim(project["id"], folder),
            expected_workspace_revision = project["workspaceRevision"],
        )
        assert claimed is not None
        return prepared

    monkeypatch.setattr(studio_db, "_prepare_project_workspace", race_prepare)

    updated = studio_db.upsert_chat_project(_managed_project(project["id"]))

    assert updated["workspaceKind"] == "folder"
    assert updated["workspacePath"] == str(folder.resolve())
    assert prepared_candidates
    assert not Path(prepared_candidates[0].root_path).exists()


def test_unbound_managed_initialization_returns_a_concurrent_binding(tmp_path, monkeypatch):
    project = studio_db.create_chat_project(_managed_project("project-concurrent-bind"))
    old_root = Path(project["managedRootPath"])
    shutil.rmtree(old_root)
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE chat_projects SET root_path = NULL, root_device_id = NULL, "
            "root_file_id = NULL, root_change_time_ns = NULL, root_marker_token = NULL "
            "WHERE id = ?",
            (project["id"],),
        )
        conn.commit()
    finally:
        conn.close()

    concurrent_root = tmp_path / "concurrent-managed"
    concurrent = studio_db._prepare_project_workspace(str(concurrent_root))
    original_prepare = studio_db._prepare_project_workspace
    prepared_candidates = []

    def race_prepare(path, **kwargs):
        prepared = original_prepare(path, **kwargs)
        prepared_candidates.append(prepared)
        conn = studio_db.get_connection()
        try:
            conn.execute(
                "UPDATE chat_projects SET root_path = ?, root_device_id = ?, root_file_id = ?, "
                "root_change_time_ns = ?, root_marker_token = ? WHERE id = ?",
                (
                    concurrent.root_path,
                    concurrent.device_id,
                    concurrent.file_id,
                    concurrent.change_time_ns,
                    concurrent.marker_token,
                    project["id"],
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return prepared

    monkeypatch.setattr(studio_db, "_prepare_project_workspace", race_prepare)

    initialized = studio_db.ensure_chat_project_workspace(project["id"])

    assert initialized is not None
    assert initialized["managedRootPath"] == concurrent.root_path
    assert prepared_candidates
    assert not Path(prepared_candidates[0].root_path).exists()


def test_unbound_managed_initialization_does_not_overwrite_a_concurrent_folder_claim(
    tmp_path, monkeypatch
):
    project = studio_db.create_chat_project(_managed_project("project-concurrent-folder"))
    old_root = Path(project["managedRootPath"])
    shutil.rmtree(old_root)
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE chat_projects SET root_path = NULL, root_device_id = NULL, "
            "root_file_id = NULL, root_change_time_ns = NULL, root_marker_token = NULL "
            "WHERE id = ?",
            (project["id"],),
        )
        conn.commit()
    finally:
        conn.close()
    folder = tmp_path / "claimed-folder"
    folder.mkdir()
    original_prepare = studio_db._prepare_project_workspace
    prepared_candidates = []

    def race_prepare(path, **kwargs):
        prepared = original_prepare(path, **kwargs)
        prepared_candidates.append(prepared)
        claimed = studio_db.claim_chat_project_folder(
            _folder_claim(project["id"], folder),
            expected_workspace_revision = project["workspaceRevision"],
        )
        assert claimed is not None
        return prepared

    monkeypatch.setattr(studio_db, "_prepare_project_workspace", race_prepare)

    initialized = studio_db.ensure_chat_project_workspace(project["id"])

    assert initialized is not None
    assert initialized["workspaceKind"] == "folder"
    assert initialized["workspacePath"] == str(folder.resolve())
    assert prepared_candidates
    assert not Path(prepared_candidates[0].root_path).exists()


def test_concurrent_workspace_marker_publication_is_atomic(tmp_path, monkeypatch):
    root = tmp_path / "managed"
    original_link = os.link
    publication_barrier = threading.Barrier(2)
    prepared = []
    failures = []

    def synchronized_link(source, destination, *args, **kwargs):
        if Path(destination).name == studio_db._PROJECT_WORKSPACE_IDENTITY_FILE:
            publication_barrier.wait(timeout = 5)
        return original_link(source, destination, *args, **kwargs)

    def prepare():
        try:
            prepared.append(studio_db._prepare_project_workspace(str(root)))
        except Exception as exc:
            failures.append(exc)

    monkeypatch.setattr(os, "link", synchronized_link)
    workers = [threading.Thread(target = prepare) for _index in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout = 10)

    assert all(not worker.is_alive() for worker in workers)
    assert failures == []
    assert len(prepared) == 2
    assert prepared[0].marker_token == prepared[1].marker_token
    assert studio_db._read_project_workspace_marker(root) == prepared[0].marker_token


def test_short_marker_writes_are_completed_before_publication(tmp_path, monkeypatch):
    original_write = os.write

    def write_one_byte(descriptor, content):
        return original_write(descriptor, content[:1])

    monkeypatch.setattr(os, "write", write_one_byte)
    prepared = studio_db._prepare_project_workspace(str(tmp_path / "managed"))

    assert len(prepared.marker_token) == 32
    assert (
        studio_db._read_project_workspace_marker(Path(prepared.root_path)) == prepared.marker_token
    )


def test_chat_export_omits_local_workspace_state(tmp_path):
    folder = tmp_path / "private-repository"
    folder.mkdir()
    studio_db.claim_chat_project_folder(_folder_claim("project-export", folder))

    exported = chat_history.export_history(current_subject = "tester").model_dump()
    encoded = json.dumps(exported)

    assert "project-export" in encoded
    assert str(folder.resolve()) not in encoded
    assert "workspacePath" not in encoded
    assert "workspaceAvailable" not in encoded
    assert "workspaceRevision" not in encoded


def test_change_and_disconnect_preserve_managed_workspace(tmp_path):
    first_folder = tmp_path / "first"
    second_folder = tmp_path / "second"
    first_folder.mkdir()
    second_folder.mkdir()
    managed = studio_db.upsert_chat_project(_managed_project("project-one"))
    managed_root = Path(managed["managedRootPath"])
    marker = managed_root / "sandbox" / "keep.txt"
    marker.write_text("managed", encoding = "utf-8")

    first = studio_db.claim_chat_project_folder(
        _folder_claim("project-one", first_folder),
        expected_workspace_revision = 0,
    )
    assert first is not None
    assert first["managedRootPath"] == str(managed_root)
    assert first["workspaceRevision"] == 1

    second = studio_db.claim_chat_project_folder(
        _folder_claim("project-one", second_folder),
        expected_workspace_revision = 1,
    )
    assert second is not None
    assert second["workspacePath"] == str(second_folder.resolve())
    assert second["workspaceRevision"] == 2
    assert marker.read_text(encoding = "utf-8") == "managed"

    with pytest.raises(studio_db.ChatProjectWorkspaceRevisionConflictError):
        studio_db.disconnect_chat_project_folder(
            "project-one",
            expected_workspace_revision = 1,
            updated_at = 1_700_000_000_002,
        )

    disconnected = studio_db.disconnect_chat_project_folder(
        "project-one",
        expected_workspace_revision = 2,
        updated_at = 1_700_000_000_003,
    )
    assert disconnected is not None
    assert disconnected["workspaceKind"] == "managed"
    assert disconnected["rootPath"] == str(managed_root)
    assert disconnected["workspaceRevision"] == 3
    assert marker.read_text(encoding = "utf-8") == "managed"


def test_stale_disconnect_does_not_create_an_unowned_managed_workspace(tmp_path):
    folder = tmp_path / "repository"
    folder.mkdir()
    project = studio_db.claim_chat_project_folder(_folder_claim("project-stale-disconnect", folder))
    assert project is not None
    managed_root = Path(studio_db._default_project_root(project))
    assert not managed_root.exists()

    with pytest.raises(studio_db.ChatProjectWorkspaceRevisionConflictError):
        studio_db.disconnect_chat_project_folder(
            project["id"],
            expected_workspace_revision = project["workspaceRevision"] + 1,
            updated_at = 1_700_000_000_002,
        )

    assert not managed_root.exists()


def test_disconnect_skips_a_reserved_root_with_foreign_content(tmp_path, monkeypatch):
    folder = tmp_path / "repository"
    folder.mkdir()
    project = studio_db.claim_chat_project_folder(_folder_claim("project-reserved", folder))
    assert project is not None
    reserved = Path(studio_db._default_project_root(project))
    reserved.mkdir(parents = True)
    foreign = reserved / "someone-elses.txt"
    foreign.write_text("keep", encoding = "utf-8")

    disconnected = studio_db.disconnect_chat_project_folder(
        project["id"],
        expected_workspace_revision = project["workspaceRevision"],
        updated_at = 1_700_000_000_002,
    )

    assert disconnected is not None
    assert Path(disconnected["managedRootPath"]) != reserved
    assert foreign.read_text(encoding = "utf-8") == "keep"
    assert reserved.is_dir()


def test_disconnect_does_not_re_adopt_a_replaced_preserved_root(tmp_path):
    folder = tmp_path / "repository"
    folder.mkdir()
    managed = studio_db.upsert_chat_project(_managed_project("project-preserved-root"))
    managed_root = Path(managed["managedRootPath"])
    marker = (managed_root / studio_db._PROJECT_WORKSPACE_IDENTITY_FILE).read_text(encoding = "ascii")
    project = studio_db.claim_chat_project_folder(
        _folder_claim("project-preserved-root", folder),
        expected_workspace_revision = managed["workspaceRevision"],
    )
    replacement = managed_root / "foreign.txt"
    original = tmp_path / "original-managed"
    managed_root.rename(original)
    managed_root.mkdir()
    (managed_root / "sandbox").mkdir()
    (managed_root / studio_db._PROJECT_WORKSPACE_IDENTITY_FILE).write_text(marker, encoding = "ascii")
    replacement.write_text("keep", encoding = "utf-8")

    disconnected = studio_db.disconnect_chat_project_folder(
        project["id"],
        expected_workspace_revision = project["workspaceRevision"],
        updated_at = 1_700_000_000_002,
    )

    assert disconnected is not None
    assert Path(disconnected["managedRootPath"]) != managed_root
    assert replacement.read_text(encoding = "utf-8") == "keep"
    assert original.is_dir()


def test_new_managed_project_skips_a_preexisting_reserved_root():
    payload = _managed_project("project-create-reserved")
    reserved = Path(studio_db._default_project_root(payload))
    reserved.mkdir(parents = True)
    foreign = reserved / "someone-elses.txt"
    foreign.write_text("keep", encoding = "utf-8")

    created = studio_db.create_chat_project(payload)

    assert Path(created["managedRootPath"]) != reserved
    assert foreign.read_text(encoding = "utf-8") == "keep"


def test_workspace_change_time_rejects_same_identity_replacement(tmp_path, monkeypatch):
    folder = tmp_path / "repository"
    folder.mkdir()
    device_id, file_id = studio_db._workspace_identity(folder)
    old_change_time = studio_db._workspace_change_time(folder)
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    monkeypatch.setattr(
        studio_db,
        "_workspace_identity",
        lambda _root: (device_id, file_id),
    )

    assert not studio_db._workspace_identity_matches(
        replacement,
        device_id,
        file_id,
        old_change_time,
    )


def test_folder_workspace_rejects_an_alias_to_a_managed_root(tmp_path):
    managed = studio_db.upsert_chat_project(_managed_project("project-alias"))
    managed_root = Path(managed["managedRootPath"])
    selected = tmp_path / "repository"
    selected.mkdir()
    alias = selected / "managed-alias"
    try:
        alias.symlink_to(managed_root, target_is_directory = True)
    except OSError:
        pytest.skip("symlinks unavailable")

    with pytest.raises(studio_db.ProjectWorkspaceOverlapError):
        studio_db.claim_chat_project_folder(_folder_claim("project-alias-child", selected))


def test_workspace_session_rotates_and_stale_session_fails_closed(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    project = studio_db.claim_chat_project_folder(_folder_claim("project-session", first))
    assert project is not None
    old_session = project["workspaceSessionId"]

    changed = studio_db.claim_chat_project_folder(
        _folder_claim("project-session", second),
        expected_workspace_revision = project["workspaceRevision"],
    )
    assert changed is not None
    assert changed["workspaceSessionId"] != old_session
    with pytest.raises(RuntimeError, match = "workspace changed"):
        tools.get_sandbox_workdir(old_session)


def test_folder_change_route_releases_the_retired_session(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    project = studio_db.claim_chat_project_folder(_folder_claim("project-route-change", first))
    old_session = project["workspaceSessionId"]

    changed = chat_history.change_project_folder(
        project["id"],
        chat_history.ProjectFolderMutation(
            nativePathLease = _sign_folder(
                second,
                operation = "set-project-workspace",
                path_kind = "project-workspace",
            ),
            expectedWorkspaceRevision = project["workspaceRevision"],
        ),
        current_subject = "tester",
    )

    assert changed.workspacePath == str(second.resolve())
    assert tools._session_key(old_session) not in tools._removing_sessions


def test_disconnect_route_releases_the_retired_session(tmp_path):
    folder = tmp_path / "folder"
    folder.mkdir()
    project = studio_db.claim_chat_project_folder(_folder_claim("project-route-disconnect", folder))
    old_session = project["workspaceSessionId"]

    disconnected = chat_history.disconnect_project_folder(
        project["id"],
        chat_history.DisconnectProjectFolderRequest(
            expectedWorkspaceRevision = project["workspaceRevision"]
        ),
        current_subject = "tester",
    )

    assert disconnected.workspaceKind == "managed"
    assert tools._session_key(old_session) not in tools._removing_sessions


def test_recreated_project_id_never_reuses_an_old_managed_session():
    first = studio_db.create_chat_project(_managed_project("project-recreated"))
    old_session = first["workspaceSessionId"]

    deleted = studio_db.delete_chat_project("project-recreated")
    assert deleted is not None

    recreated = studio_db.create_chat_project(_managed_project("project-recreated"))
    assert recreated["workspaceSessionId"] != old_session
    assert recreated["workspaceSessionId"].startswith("project-workspace-")


def test_deleted_folder_project_records_its_external_workspace(tmp_path):
    folder = tmp_path / "external-project"
    folder.mkdir()
    try:
        project = studio_db.claim_chat_project_folder(
            _folder_claim("project-delete-folder", folder)
        )
        deleted = studio_db.delete_chat_project(project["id"])
        assert deleted is not None
        record = tools._read_orphan_record(tools._ORPHAN_PROJECT, project["id"])
        assert record is not None
        assert record["sessionId"] == project["workspaceSessionId"]
        assert record["path"] == str(folder.resolve())
        assert record["pendingDelete"] is False
    finally:
        if folder.exists():
            folder.rmdir()


def test_folder_delete_aborts_when_external_workspace_record_fails(tmp_path, monkeypatch):
    folder = tmp_path / "external-project"
    folder.mkdir()
    project = studio_db.claim_chat_project_folder(_folder_claim("project-record-fail", folder))
    monkeypatch.setattr(tools, "record_orphaned_project", lambda *args, **kwargs: False)

    with pytest.raises(studio_db.ProjectWorkspaceError):
        studio_db.delete_chat_project(project["id"])

    assert studio_db.get_chat_project(project["id"]) is not None


def test_retired_workspace_identity_blocks_replacement(tmp_path):
    folder = tmp_path / "retired"
    folder.mkdir()
    tools.record_orphaned_project("project-retired-identity", str(folder))
    session = tools.project_session_id("project-retired-identity")
    folder.rmdir()
    folder.mkdir()

    assert tools._recorded_project_workdir("project-retired-identity", session) is None


def test_retired_workspace_records_are_unique_per_incarnation(tmp_path):
    project_id = "project-retired-incarnations"
    first = tmp_path / "first"
    second = tmp_path / "second"
    (first / "sandbox").mkdir(parents = True)
    (second / "sandbox").mkdir(parents = True)
    first_session = "project-workspace-first-incarnation"
    second_session = "project-workspace-second-incarnation"

    assert tools.record_orphaned_project(
        project_id,
        str(first / "sandbox"),
        True,
        str(first),
        first_session,
    )
    assert tools.record_orphaned_project(
        project_id,
        str(second / "sandbox"),
        True,
        str(second),
        second_session,
    )

    assert tools._recorded_project_workdir(project_id, first_session) == str(
        (first / "sandbox").resolve()
    )
    assert tools._recorded_project_workdir(project_id, second_session) == str(
        (second / "sandbox").resolve()
    )
    assert len([entry for entry in tools.list_orphaned_projects() if entry[0] == project_id]) == 2


def test_retired_cleanup_and_runtime_evidence_are_session_scoped(tmp_path):
    project_id = "project-same-path-incarnations"
    root = tmp_path / "workspace"
    (root / "sandbox").mkdir(parents = True)
    first_session = "project-workspace-same-path-first"
    second_session = "project-workspace-same-path-second"

    first_evidence, first_token = studio_db.capture_recorded_orphan_evidence(
        project_id, str(root), first_session
    )
    second_evidence, second_token = studio_db.capture_recorded_orphan_evidence(
        project_id, str(root), second_session
    )
    assert first_evidence is not None and second_evidence is not None
    assert first_token is not second_token
    assert (
        studio_db.recorded_orphan_evidence_for(project_id, str(root), first_session).runtime_token
        is first_token
    )
    assert (
        studio_db.recorded_orphan_evidence_for(project_id, str(root), second_session).runtime_token
        is second_token
    )

    tools.record_orphaned_project(
        project_id,
        str(root / "sandbox"),
        True,
        str(root),
        first_session,
        (first_evidence["deviceId"], first_evidence["fileId"], first_evidence["changeTimeNs"]),
    )
    tools.record_orphaned_project(
        project_id,
        str(root / "sandbox"),
        True,
        str(root),
        second_session,
        (second_evidence["deviceId"], second_evidence["fileId"], second_evidence["changeTimeNs"]),
    )
    shutil.rmtree(root)
    tools.forget_orphaned_project_if_gone(
        project_id,
        str(root / "sandbox"),
        str(root),
        False,
        first_session,
    )
    assert tools._read_orphan_record(tools._ORPHAN_PROJECT, project_id, second_session) is not None


def test_switch_records_retired_folder_identity_when_drive_is_disconnected(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    project = studio_db.claim_chat_project_folder(_folder_claim("project-retired", first))
    assert project is not None
    old_session = project["workspaceSessionId"]
    old_identity = (
        project["workspaceDeviceId"],
        project["workspaceFileId"],
        project["workspaceChangeTimeNs"],
    )
    first.rmdir()

    changed = studio_db.claim_chat_project_folder(
        _folder_claim("project-retired", second),
        expected_workspace_revision = project["workspaceRevision"],
    )
    assert changed is not None
    record = tools._read_orphan_record(tools._ORPHAN_PROJECT, project["id"])
    assert record is not None
    assert record["sessionId"] == old_session
    assert record["pendingDelete"] is False
    assert record["rootIdentity"]["deviceId"] == str(old_identity[0])
    assert record["rootIdentity"]["fileId"] == str(old_identity[1])


def test_disconnect_cas_failure_cleans_the_workspace_prepared_by_that_attempt(
    tmp_path, monkeypatch
):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    project = studio_db.claim_chat_project_folder(_folder_claim("project-disconnect-race", first))
    assert project is not None
    managed_root = Path(studio_db._default_project_root(project))
    original_prepare = studio_db._prepare_project_workspace

    def prepare_then_change(path, **kwargs):
        prepared = original_prepare(path, **kwargs)
        studio_db.claim_chat_project_folder(
            _folder_claim("project-disconnect-race", second),
            expected_workspace_revision = project["workspaceRevision"],
        )
        return prepared

    monkeypatch.setattr(studio_db, "_prepare_project_workspace", prepare_then_change)

    with pytest.raises(studio_db.ChatProjectWorkspaceRevisionConflictError):
        studio_db.disconnect_chat_project_folder(
            project["id"],
            expected_workspace_revision = project["workspaceRevision"],
            updated_at = 1_700_000_000_003,
        )

    assert not managed_root.exists()
    stored = studio_db.get_chat_project(project["id"])
    assert stored is not None
    assert stored["workspacePath"] == str(second.resolve())


def test_folder_workspaces_cannot_overlap(tmp_path):
    parent = tmp_path / "repository"
    child = parent / "package"
    child.mkdir(parents = True)
    studio_db.claim_chat_project_folder(_folder_claim("parent-project", parent))

    with pytest.raises(studio_db.ProjectWorkspaceOverlapError):
        studio_db.claim_chat_project_folder(_folder_claim("child-project", child))


def test_folder_workspace_cannot_overlap_its_preserved_managed_root(tmp_path):
    managed = studio_db.upsert_chat_project(_managed_project("project-own-root"))
    managed_root = Path(managed["managedRootPath"])

    with pytest.raises(studio_db.ProjectWorkspaceOverlapError):
        studio_db.claim_chat_project_folder(
            _folder_claim("project-own-root", managed_root / "sandbox"),
            expected_workspace_revision = 0,
        )


def test_tools_use_folder_cwd_and_fail_closed_when_it_disappears(tmp_path):
    folder = tmp_path / "repository"
    folder.mkdir()
    project = studio_db.claim_chat_project_folder(_folder_claim("project-tools", folder))
    assert project is not None
    session_id = tools.project_session_id(project["id"])

    assert tools.get_sandbox_workdir(session_id) == str(folder.resolve())
    assert tools._tracks_workspace_artifacts(session_id) is False

    folder.rmdir()
    stored = studio_db.get_chat_project(project["id"])
    assert stored is not None
    assert stored["workspaceAvailable"] is False
    with pytest.raises(RuntimeError, match = "project folder is unavailable"):
        tools.get_sandbox_workdir(session_id)


def test_public_sandbox_routes_do_not_expose_existing_folder(tmp_path):
    folder = tmp_path / "repository"
    folder.mkdir()
    studio_db.claim_chat_project_folder(_folder_claim("project-public", folder))

    with pytest.raises(HTTPException) as caught:
        inference._sandbox_dir_for("project-project-public", create = False)

    assert caught.value.status_code == 403


def test_public_sandbox_authorization_and_path_use_one_project_snapshot(tmp_path, monkeypatch):
    managed_root = tmp_path / "managed"
    managed_sandbox = managed_root / "sandbox"
    managed_sandbox.mkdir(parents = True)
    external = tmp_path / "external"
    external.mkdir()
    reads = 0

    def get_project(_project_id):
        nonlocal reads
        reads += 1
        if reads == 1:
            return {
                "workspaceKind": "managed",
                "managedRootPath": str(managed_root),
                "rootPath": str(managed_root),
                "sandboxPath": str(managed_sandbox),
            }
        return {
            "workspaceKind": "folder",
            "managedRootPath": str(managed_root),
            "rootPath": str(external),
            "sandboxPath": str(external),
        }

    monkeypatch.setattr(studio_db, "get_chat_project", get_project)
    monkeypatch.setattr(studio_db, "get_chat_thread", lambda _session_id: None)

    resolved = inference._sandbox_dir_for("project-race", create = False)

    assert resolved == str(managed_sandbox.resolve())
    assert reads == 1


def test_public_sandbox_absence_cannot_be_redirected_by_a_new_folder_project(tmp_path, monkeypatch):
    external = tmp_path / "external"
    external.mkdir()
    project_reads = 0

    def get_project(_project_id):
        nonlocal project_reads
        project_reads += 1
        if project_reads == 1:
            return None
        return {
            "workspaceKind": "folder",
            "rootPath": str(external),
            "sandboxPath": str(external),
        }

    monkeypatch.setattr(studio_db, "get_chat_project", get_project)
    monkeypatch.setattr(studio_db, "get_chat_thread", lambda _session_id: None)

    resolved = inference._sandbox_dir_for("project-race", create = False)

    assert resolved != str(external.resolve())
    assert project_reads == 1


def test_public_sandbox_thread_snapshot_cannot_become_a_folder_project(tmp_path, monkeypatch):
    external = tmp_path / "external"
    external.mkdir()
    thread_reads = 0
    project_reads = 0

    def get_thread(_session_id):
        nonlocal thread_reads
        thread_reads += 1
        return {"id": "project-race"} if thread_reads == 1 else None

    def get_project(_project_id):
        nonlocal project_reads
        project_reads += 1
        return {
            "workspaceKind": "folder",
            "rootPath": str(external),
            "sandboxPath": str(external),
        }

    monkeypatch.setattr(studio_db, "get_chat_thread", get_thread)
    monkeypatch.setattr(studio_db, "get_chat_project", get_project)

    resolved = inference._sandbox_dir_for("project-race", create = False)

    assert resolved != str(external.resolve())
    assert thread_reads == 1
    assert project_reads == 0


def test_folder_update_does_not_recreate_a_concurrently_deleted_project(tmp_path):
    folder = tmp_path / "repository"
    folder.mkdir()

    updated = studio_db.claim_chat_project_folder(
        {
            "id": "project-deleted",
            "workspacePath": str(folder),
            "workspaceDeviceId": folder.stat().st_dev,
            "workspaceFileId": folder.stat().st_ino,
            "updatedAt": 1_700_000_000_001,
        },
        expected_workspace_revision = 0,
        require_existing = True,
    )

    assert updated is None
    assert studio_db.get_chat_project("project-deleted") is None


def test_unavailable_folder_metadata_patch_commits_without_a_500(tmp_path):
    folder = tmp_path / "repository"
    folder.mkdir()
    studio_db.claim_chat_project_folder(_folder_claim("project-rename", folder))
    folder.rmdir()

    patched = chat_history.patch_project(
        "project-rename",
        chat_history.ChatProjectPatch(name = "Renamed"),
        current_subject = "tester",
    )

    assert patched.name == "Renamed"
    assert patched.workspaceAvailable is False


def test_project_delete_never_removes_the_existing_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(studio_db, "_denied_path_prefixes", lambda: ())
    # Matches the legacy managed-root suffix check for id "project-delete".
    # Folder ownership, not a basename coincidence, must decide deletion.
    folder = tmp_path / "external-project"
    folder.mkdir()
    external_marker = folder / "source.txt"
    external_marker.write_text("user-owned", encoding = "utf-8")
    managed = studio_db.upsert_chat_project(_managed_project("project-delete"))
    managed_root = Path(managed["managedRootPath"])
    (managed_root / "sandbox" / "generated.txt").write_text("generated", encoding = "utf-8")
    studio_db.claim_chat_project_folder(
        _folder_claim("project-delete", folder),
        expected_workspace_revision = 0,
    )

    deleted = studio_db.delete_chat_project("project-delete", delete_files = True)

    assert deleted is not None
    assert external_marker.read_text(encoding = "utf-8") == "user-owned"
    assert folder.is_dir()
    assert not managed_root.exists()
