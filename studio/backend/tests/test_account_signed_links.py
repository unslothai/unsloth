# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Signed links carry their own account, and die with it: with no auth dependency to bind one, each route resolves the account from the signed target and re-checks that it still exists."""

import secrets
from pathlib import Path

import pytest
from fastapi import HTTPException
from PIL import Image

from auth import policy, storage
from core.inference import image_gallery, video_gallery
from core.rag import store
from hub.services.models import account_access as access
from routes import rag as rag_routes
from storage import rag_db
from utils.account_context import OWNER, run_as
from utils.paths import ensure_dir, rag_uploads_root

ALICE_NAME = "alice"


@pytest.fixture
def accounts(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    rag_db.reset_schema_state_for_tests()
    policy.invalidate_account_cache()
    storage.create_initial_user("unsloth", "owner-password", secrets.token_urlsafe(32))
    alice = storage.issue_account_setup_code(username = ALICE_NAME)["account"]
    yield storage.get_account(ALICE_NAME), alice["account_id"]
    policy.invalidate_account_cache()


def _seed_document(account, document_id: str, text: bytes) -> None:
    def write():
        path = ensure_dir(rag_uploads_root()) / f"{document_id}.txt"
        path.write_bytes(text)
        conn = rag_db.get_connection()
        try:
            store.create_document(
                conn,
                scope = "kb:none",
                filename = f"{document_id}.txt",
                sha256 = document_id,
                status = "ready",
                stored_path = str(path),
                document_id = document_id,
            )
        finally:
            conn.close()

    run_as(account, write)


def test_a_signed_rag_link_serves_its_own_accounts_document(accounts):
    alice, _account_id = accounts
    _seed_document(OWNER, "shared-id", b"owner secret")
    _seed_document(alice, "shared-id", b"alice secret")

    token = run_as(alice, rag_routes._sign_document, "shared-id")
    response = rag_routes.document_file_signed("shared-id", token = token)
    assert Path(response.path).read_bytes() == b"alice secret"

    owner_token = rag_routes._sign_document("shared-id")
    owner_response = rag_routes.document_file_signed("shared-id", token = owner_token)
    assert Path(owner_response.path).read_bytes() == b"owner secret"


def test_a_signed_rag_link_dies_with_its_account(accounts):
    alice, account_id = accounts
    _seed_document(alice, "doc-1", b"alice secret")
    token = run_as(alice, rag_routes._sign_document, "doc-1")
    served = rag_routes.document_file_signed("doc-1", token = token)
    assert Path(served.path).read_bytes() == b"alice secret"

    storage.set_account_active(account_id, False)
    with pytest.raises(HTTPException) as refused:
        rag_routes.document_file_signed("doc-1", token = token)
    assert refused.value.status_code == 401


def test_a_signed_media_link_dies_with_its_account(accounts):
    alice, account_id = accounts
    meta = {"prompt": "p", "model": "m", "created_at": 100.0, "width": 8, "height": 8}
    image_id = run_as(alice, image_gallery.save, Image.new("RGB", (8, 8)), meta)["id"]
    video_id = run_as(
        alice,
        video_gallery.save,
        b"\x00\x00\x00\x18ftypmp42",
        {**meta, "num_frames": 1, "fps": 1, "duration_s": 1.0},
    )["id"]

    for media_id in (image_id, video_id):
        target = run_as(alice, access.media_link_target, media_id)
        assert access.media_link_account(target, media_id).account_id == account_id

    storage.set_account_active(account_id, False)
    for media_id in (image_id, video_id):
        target = f"{account_id}:{media_id}"
        assert access.media_link_account(target, media_id) is None
    assert access.media_link_account(image_id, image_id) is OWNER
