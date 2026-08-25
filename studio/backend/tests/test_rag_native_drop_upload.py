# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Desktop drops reach the RAG ingest through a signed native path grant (#7661)."""

import base64
import hashlib
import hmac
import json
import os
import time

import pytest
from fastapi import HTTPException

from core.rag import config
import utils.native_path_leases as leases
from routes.rag import _resolve_document_upload, _save_native_path_upload
from utils.paths import rag_uploads_root

SECRET = b"n" * 32


@pytest.fixture(autouse = True)
def _lease_secret(monkeypatch):
    monkeypatch.setenv(
        leases.LEASE_SECRET_ENV,
        base64.urlsafe_b64encode(SECRET).decode("ascii").rstrip("="),
    )
    monkeypatch.setattr(leases, "_CACHED_LEASE_SECRET", None, raising = False)
    yield
    monkeypatch.setattr(leases, "_CACHED_LEASE_SECRET", None, raising = False)


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _sign(
    path,
    *,
    operation = "attach",
    path_kind = "attachment",
    identity_options = None,
    nonce = None,
    secret = SECRET,
):
    """Mint the grant Rust would sign for a dropped file."""
    st = os.stat(path)
    path_type = "directory" if os.path.isdir(path) else "file"
    now_ms = int(time.time() * 1000)
    identities = identity_options or ((st.st_dev, st.st_ino),)
    payload = {
        "version": 1,
        "operation": operation,
        "canonical_path": str(path),
        "path_kind": path_kind,
        "path_type": path_type,
        "source_kind": "drop",
        "token_id_hash": hashlib.sha256(b"path_token").hexdigest(),
        "issued_at_ms": now_ms,
        "expires_at_ms": now_ms + 120_000,
        "nonce": nonce or os.urandom(16).hex(),
        "display_label": os.path.basename(path),
        "size_bytes": st.st_size if path_type == "file" else None,
        "modified_ms": int(st.st_mtime_ns // 1_000_000) if path_type == "file" else None,
        "device_id": ":".join(format(identity[0], "x") for identity in identities),
        "file_id": ":".join(format(identity[1], "x") for identity in identities),
    }
    payload_b64 = _b64(json.dumps(payload).encode("utf-8"))
    signature = hmac.new(secret, payload_b64.encode("ascii"), hashlib.sha256).digest()
    return f"{payload_b64}.{_b64(signature)}"


def _doc(
    tmp_path,
    name = "notes.txt",
    body = "alpha bravo charlie",
):
    path = tmp_path / name
    path.write_text(body, encoding = "utf-8")
    return path


def test_signed_drop_is_copied_into_the_uploads_root(rag_home, tmp_path):
    source = _doc(tmp_path)
    stored_path, filename = _save_native_path_upload(_sign(source))

    assert filename == "notes.txt"
    # Copied, not referenced: ingestion must not read from wherever the user dragged from.
    assert os.path.realpath(stored_path) != os.path.realpath(source)
    with open(stored_path, encoding = "utf-8") as handle:
        assert handle.read() == "alpha bravo charlie"


def test_forged_signature_is_rejected(rag_home, tmp_path):
    source = _doc(tmp_path)
    with pytest.raises(HTTPException) as excinfo:
        _save_native_path_upload(_sign(source, secret = b"z" * 32))
    assert excinfo.value.status_code == 400


@pytest.mark.parametrize(
    "kwargs",
    [
        {"operation": "load-model"},  # a model grant cannot be spent as an attachment
        {"path_kind": "model"},
    ],
)
def test_grant_for_another_purpose_is_rejected(rag_home, tmp_path, kwargs):
    source = _doc(tmp_path)
    with pytest.raises(HTTPException):
        _save_native_path_upload(_sign(source, **kwargs))


def test_unsupported_extension_is_rejected(rag_home, tmp_path):
    source = _doc(tmp_path, name = "payload.exe", body = "MZ")
    with pytest.raises(HTTPException):
        _save_native_path_upload(_sign(source))


def test_empty_file_is_rejected(rag_home, tmp_path):
    source = _doc(tmp_path, body = "")
    with pytest.raises(HTTPException) as excinfo:
        _save_native_path_upload(_sign(source))
    assert excinfo.value.status_code == 400


def test_native_drop_uses_the_shared_size_limit(rag_home, tmp_path, monkeypatch):
    source = _doc(tmp_path, body = "x" * 4096)
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 1024)

    with pytest.raises(HTTPException) as excinfo:
        _save_native_path_upload(_sign(source))

    assert excinfo.value.status_code == 413
    assert list(rag_uploads_root().glob("*.txt")) == []


def test_grant_is_single_use(rag_home, tmp_path):
    source = _doc(tmp_path)
    lease = _sign(source)
    _save_native_path_upload(lease)
    # Replaying the same nonce must not mint a second read of the path.
    with pytest.raises(HTTPException):
        _save_native_path_upload(lease)


def test_document_folder_grant_binds_identity_but_allows_content_changes(tmp_path):
    folder = tmp_path / "documents"
    folder.mkdir()
    lease = _sign(folder, operation = "link-documents", path_kind = "document-folder")
    (folder / "new.txt").write_text("new content", encoding = "utf-8")

    grant = leases.verify_native_path_lease(
        lease,
        operation = "link-documents",
        expected_kind = "document-folder",
        expected_path_type = "directory",
    )

    assert grant.canonical_path == folder

    replaced_lease = _sign(folder, operation = "link-documents", path_kind = "document-folder")
    old_folder = tmp_path / "old-documents"
    folder.rename(old_folder)
    folder.mkdir()
    with pytest.raises(leases.NativePathLeaseError, match = "changed"):
        leases.verify_native_path_lease(
            replaced_lease,
            operation = "link-documents",
            expected_kind = "document-folder",
            expected_path_type = "directory",
        )


def test_project_workspace_grant_is_scoped_and_binds_folder_identity(tmp_path):
    folder = tmp_path / "project"
    folder.mkdir()
    lease = _sign(
        folder,
        operation = "set-project-workspace",
        path_kind = "project-workspace",
    )
    original = folder.stat()
    (folder / "created-after-selection.txt").write_text("ok", encoding = "utf-8")
    os.utime(
        folder,
        ns = (original.st_atime_ns, original.st_mtime_ns + 1_000_000_000),
    )

    grant = leases.verify_native_path_lease(
        lease,
        operation = "set-project-workspace",
        expected_kind = "project-workspace",
        expected_path_type = "directory",
    )

    assert grant.canonical_path == folder

    wrong_operation = _sign(
        folder,
        operation = "set-project-workspace",
        path_kind = "project-workspace",
    )
    with pytest.raises(leases.NativePathLeaseError, match = "operation"):
        leases.verify_native_path_lease(
            wrong_operation,
            operation = "link-documents",
            expected_kind = "project-workspace",
            expected_path_type = "directory",
        )

    replaced = _sign(
        folder,
        operation = "set-project-workspace",
        path_kind = "project-workspace",
    )
    folder.rename(tmp_path / "old-project")
    folder.mkdir()
    with pytest.raises(leases.NativePathLeaseError, match = "changed"):
        leases.verify_native_path_lease(
            replaced,
            operation = "set-project-workspace",
            expected_kind = "project-workspace",
            expected_path_type = "directory",
        )


@pytest.mark.parametrize(("uses_extended_identity", "matching_index"), [(False, 0), (True, 1)])
def test_grant_uses_the_identity_exposed_by_its_python_runtime(
    tmp_path, monkeypatch, uses_extended_identity, matching_index
):
    folder = tmp_path / "documents"
    folder.mkdir()
    current = (folder.stat().st_dev, folder.stat().st_ino)
    other = (current[0] + 1, current[1] + 1)
    accepted = [other, other]
    accepted[matching_index] = current
    lease = _sign(
        folder,
        operation = "link-documents",
        path_kind = "document-folder",
        identity_options = accepted,
    )
    rejected = [other, other]
    rejected[1 - matching_index] = current
    rejected_lease = _sign(
        folder,
        operation = "link-documents",
        path_kind = "document-folder",
        identity_options = rejected,
    )
    monkeypatch.setattr(leases, "_WINDOWS_STAT_USES_FILE_ID_INFO", uses_extended_identity)

    grant = leases.verify_native_path_lease(
        lease,
        operation = "link-documents",
        expected_kind = "document-folder",
        expected_path_type = "directory",
    )

    assert (grant.device_id, grant.file_id) == current
    with pytest.raises(leases.NativePathLeaseError, match = "changed"):
        leases.verify_native_path_lease(
            rejected_lease,
            operation = "link-documents",
            expected_kind = "document-folder",
            expected_path_type = "directory",
        )


def test_resolver_needs_one_of_file_or_lease(rag_home):
    with pytest.raises(HTTPException) as excinfo:
        _resolve_document_upload(None, None)
    assert excinfo.value.status_code == 400
