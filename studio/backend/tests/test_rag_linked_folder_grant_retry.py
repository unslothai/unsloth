# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A failed folder link must answer with its real error, readable by the desktop webview (#13093)."""

import base64
import hashlib
import hmac
import json
import os
import time
from contextlib import closing

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

import utils.native_path_leases as leases
from auth.authentication import get_current_subject
from core.rag import folder_sync, store
from routes import rag as rag_routes
from storage import rag_db

SECRET = b"f" * 32
ORIGIN = "tauri://localhost"

requires_sqlite_vec = pytest.mark.skipif(
    not rag_db.RAG_AVAILABLE, reason = "sqlite-vec is not installed"
)


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _sign(path) -> str:
    """The grant Rust's pick_native_document_folder signs."""
    st = os.stat(path)
    now_ms = int(time.time() * 1000)
    payload = {
        "version": 1,
        "operation": "link-documents",
        "canonical_path": str(path),
        "path_kind": "document-folder",
        "path_type": "directory",
        "source_kind": "dialog",
        "token_id_hash": hashlib.sha256(os.urandom(24)).hexdigest(),
        "issued_at_ms": now_ms,
        "expires_at_ms": now_ms + 120_000,
        "nonce": os.urandom(16).hex(),
        "display_label": os.path.basename(path),
        "size_bytes": None,
        "modified_ms": None,
        "device_id": format(st.st_dev, "x"),
        "file_id": format(st.st_ino, "x"),
    }
    payload_b64 = _b64(json.dumps(payload).encode("utf-8"))
    signature = hmac.new(SECRET, payload_b64.encode("ascii"), hashlib.sha256).digest()
    return f"{payload_b64}.{_b64(signature)}"


@pytest.fixture
def client(rag_home, monkeypatch):
    monkeypatch.setenv(leases.LEASE_SECRET_ENV, _b64(SECRET))
    monkeypatch.setattr(leases, "_CACHED_LEASE_SECRET", None, raising = False)
    with closing(rag_db.get_connection()) as conn:
        store.create_kb(conn, name = "Knowledge", kb_id = "kb")
    app = FastAPI()
    app.include_router(rag_routes.router, prefix = "/api/rag")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    # As in main.py: the desktop webview is cross-origin, so it reads only answers carrying CORS headers.
    app.add_middleware(CORSMiddleware, allow_origins = [ORIGIN], allow_methods = ["*"])
    yield TestClient(app, raise_server_exceptions = False)
    monkeypatch.setattr(leases, "_CACHED_LEASE_SECRET", None, raising = False)


def _folder(rag_home, name):
    folder = rag_home / name
    folder.mkdir()
    (folder / "notes.txt").write_text("alpha bravo", encoding = "utf-8")
    return folder


def _link(client, lease):
    return client.post(
        "/api/rag/knowledge-bases/kb/linked-folders",
        json = {"nativePathLease": lease, "displayName": "Docs"},
        headers = {"Origin": ORIGIN},
    )


@requires_sqlite_vec
def test_an_unexpected_failure_answers_with_cors_headers_and_its_error(
    client, rag_home, monkeypatch
):
    docs = _folder(rag_home, "docs")
    create = folder_sync.create_folder_with_sync
    calls = []

    def flaky(**kwargs):
        calls.append(kwargs["path"])
        if len(calls) == 1:
            raise RuntimeError("database is locked")
        return create(**kwargs)

    monkeypatch.setattr(folder_sync, "create_folder_with_sync", flaky)

    failed = _link(client, _sign(docs))
    assert failed.status_code == 500
    assert failed.json()["detail"] == "Could not link the folder: database is locked"
    # Without it the webview sees a network error and resends the spent grant.
    assert failed.headers.get("access-control-allow-origin") == ORIGIN

    relinked = _link(client, _sign(docs))
    assert relinked.status_code == 200, relinked.text
    assert len(folder_sync.list_folders(store.kb_scope("kb"))) == 1


@requires_sqlite_vec
def test_a_grant_that_linked_a_folder_still_refuses_a_replay(client, rag_home):
    lease = _sign(_folder(rag_home, "docs"))
    assert _link(client, lease).status_code == 200

    replay = _link(client, lease)
    assert replay.status_code == 400
    assert replay.json()["detail"] == "Native path grant was already used."
