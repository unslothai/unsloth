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
    nonce = None,
    secret = SECRET,
):
    """Mint the grant Rust would sign for a dropped file."""
    st = os.stat(path)
    now_ms = int(time.time() * 1000)
    payload = {
        "version": 1,
        "operation": operation,
        "canonical_path": str(path),
        "path_kind": path_kind,
        "path_type": "file",
        "source_kind": "drop",
        "token_id_hash": hashlib.sha256(b"path_token").hexdigest(),
        "issued_at_ms": now_ms,
        "expires_at_ms": now_ms + 120_000,
        "nonce": nonce or os.urandom(16).hex(),
        "display_label": os.path.basename(path),
        "size_bytes": st.st_size,
        "modified_ms": int(st.st_mtime_ns // 1_000_000),
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


def test_resolver_needs_one_of_file_or_lease(rag_home):
    with pytest.raises(HTTPException) as excinfo:
        _resolve_document_upload(None, None)
    assert excinfo.value.status_code == 400
