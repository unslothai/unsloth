# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A native dataset drop is confined to a managed account's workspace, like the RAG/seed/library drops."""

import base64
import hashlib
import hmac
import json
import os
import time
from pathlib import Path

import pytest
from fastapi import HTTPException

import utils.native_path_leases as leases
from auth import policy
from hub.services.datasets import local
from utils.account_context import OWNER, AccountContext, run_as
from utils.paths import workspace_root

SECRET = b"d" * 32
ALICE = AccountContext("alice-dataset-account", "alice")


@pytest.fixture(autouse = True)
def _setup(tmp_path, monkeypatch):
    monkeypatch.setenv(leases.LEASE_SECRET_ENV, _b64(SECRET))
    monkeypatch.setattr(leases, "_CACHED_LEASE_SECRET", None, raising = False)
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    store = tmp_path / "store"
    store.mkdir()
    monkeypatch.setattr(local, "DATASET_UPLOAD_DIR", store)
    yield store
    monkeypatch.setattr(leases, "_CACHED_LEASE_SECRET", None, raising = False)


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _sign(path: Path) -> str:
    st = os.stat(path)
    now_ms = int(time.time() * 1000)
    payload = {
        "version": 1,
        "operation": "dataset-import",
        "canonical_path": str(path),
        "path_kind": "dataset",
        "path_type": "file",
        "source_kind": "drop",
        "token_id_hash": hashlib.sha256(b"path_token").hexdigest(),
        "issued_at_ms": now_ms,
        "expires_at_ms": now_ms + 120_000,
        "nonce": os.urandom(16).hex(),
        "display_label": path.name,
        "size_bytes": st.st_size,
        "modified_ms": int(st.st_mtime_ns // 1_000_000),
        "device_id": format(st.st_dev, "x"),
        "file_id": format(st.st_ino, "x"),
    }
    payload_b64 = _b64(json.dumps(payload).encode("utf-8"))
    signature = hmac.new(SECRET, payload_b64.encode("ascii"), hashlib.sha256).digest()
    return f"{payload_b64}.{_b64(signature)}"


def _csv(directory: Path) -> Path:
    directory.mkdir(parents = True, exist_ok = True)
    path = directory / "rows.csv"
    path.write_bytes(b"a,b\n1,2\n")
    return path


def test_managed_account_cannot_import_a_file_outside_its_workspace(tmp_path, _setup):
    outside = _csv(tmp_path / "outside")
    with pytest.raises(HTTPException) as exc:
        run_as(ALICE, local._native_upload_dataset_response, _sign(outside))
    assert exc.value.status_code == 403
    assert list(_setup.iterdir()) == []


def test_managed_account_imports_a_file_inside_its_workspace(_setup):
    inside = _csv(run_as(ALICE, workspace_root) / "drops")
    response = run_as(ALICE, local._native_upload_dataset_response, _sign(inside))
    assert Path(response.stored_path).read_bytes() == b"a,b\n1,2\n"


def test_owner_import_is_unchanged(tmp_path):
    outside = _csv(tmp_path / "outside")
    response = run_as(OWNER, local._native_upload_dataset_response, _sign(outside))
    assert response.filename == "rows.csv"
    assert Path(response.stored_path).read_bytes() == b"a,b\n1,2\n"
