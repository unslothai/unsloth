# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A native dataset drop is confined to a managed account's workspace, like the RAG/seed/library drops."""

import asyncio
import base64
import hashlib
import hmac
import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi import HTTPException

import utils.native_path_leases as leases
from auth import policy
from hub.services.datasets import local
from utils.account_context import OWNER, AccountContext, arun_as, run_as
from utils.paths import workspace_root
from utils.paths.storage_roots import dataset_uploads_root

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


# --- Lease and source fixture helpers ---
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


# --- End fixture helpers ---


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


def test_inventory_excludes_in_progress_native_copy(monkeypatch):
    source = _csv(run_as(ALICE, workspace_root) / "drops")
    payload = b"prompt,response\n" + b"hello,world\n" * 200_000
    source.write_bytes(payload)
    lease = _sign(source)
    store = run_as(ALICE, dataset_uploads_root)
    store.mkdir(parents = True, exist_ok = True)
    monkeypatch.setattr(local, "DATASET_UPLOAD_DIR", store)
    existing = store / "existing.csv"
    existing.write_bytes(b"text\nkeep\n")
    resume = threading.Event()
    real_open = open

    async def exercise():
        loop = asyncio.get_running_loop()
        paused = asyncio.Event()

        @contextmanager
        def gated_open(path, mode):
            with real_open(path, mode) as handle:
                if Path(path) != source:
                    yield handle
                    return

                class Reader:
                    reads = 0

                    def read(self, size):
                        assert size == local.LOCAL_UPLOAD_CHUNK_BYTES
                        self.reads += 1
                        if self.reads == 2:
                            loop.call_soon_threadsafe(paused.set)
                            assert resume.wait(timeout = 5)
                        return handle.read(size)

                yield Reader()

        monkeypatch.setattr(local, "open", gated_open, raising = False)
        task = asyncio.create_task(arun_as(ALICE, local.upload_dataset_response(None, lease)))
        try:
            await asyncio.wait_for(paused.wait(), timeout = 5)
            pending = set(store.iterdir()) - {existing}
            assert len(pending) == 1
            assert pending.pop().stat().st_size == local.LOCAL_UPLOAD_CHUNK_BYTES
            during = run_as(ALICE, local.list_local_datasets_response).datasets
            assert [item.path for item in during] == [str(existing.resolve())]
        finally:
            resume.set()
            response = await task

        stored = Path(response.stored_path)
        assert response.model_dump() == {"filename": "rows.csv", "stored_path": str(stored)}
        assert stored.parent == store
        assert stored.name.endswith("_rows.csv")
        assert len(stored.name.split("_", 1)[0]) == 32
        assert stored.read_bytes() == payload
        assert source.read_bytes() == payload
        items = run_as(ALICE, local.list_local_datasets_response).datasets
        assert len(items) == 2
        item = next(item for item in items if item.path == str(stored.resolve()))
        assert (item.id, item.label, item.source) == (stored.name, "rows.csv", "upload")
        assert set(store.iterdir()) == {existing, stored}
        assert existing.read_bytes() == b"text\nkeep\n"

    asyncio.run(exercise())
    with pytest.raises(HTTPException) as exc:
        run_as(ALICE, local._native_upload_dataset_response, lease)
    assert exc.value.status_code == 400
    assert "already used" in exc.value.detail
    assert len(list(store.iterdir())) == 2


@pytest.mark.parametrize("failure", ["empty", "limit", "read", "write", "rename"])
def test_failed_native_copy_leaves_no_file(monkeypatch, failure):
    source = _csv(run_as(ALICE, workspace_root) / "drops")
    payload = b"x" * local.LOCAL_UPLOAD_CHUNK_BYTES + b"y"
    source.write_bytes(b"" if failure == "empty" else payload)
    lease = _sign(source)
    store = run_as(ALICE, dataset_uploads_root)
    store.mkdir(parents = True, exist_ok = True)
    monkeypatch.setattr(local, "DATASET_UPLOAD_DIR", store)
    existing = store / "existing.csv"
    existing.write_bytes(b"text\nkeep\n")
    if failure == "limit":
        monkeypatch.setattr(local, "get_upload_limit_mb", lambda: 1)
    if failure in {"read", "write"}:
        real_open = open

        @contextmanager
        def failing_open(path, mode):
            with real_open(path, mode) as handle:

                class File:
                    reads = 0

                    def read(self, size):
                        self.reads += 1
                        if failure == "read" and self.reads == 2:
                            raise OSError("read failed")
                        return handle.read(size)

                    def write(self, chunk):
                        handle.write(chunk)
                        if failure == "write":
                            raise OSError("write failed")

                yield File()

        monkeypatch.setattr(local, "open", failing_open, raising = False)
    if failure == "rename":

        def fail_rename(self, destination):
            assert self.read_bytes() == payload
            raise OSError("rename failed")

        monkeypatch.setattr(Path, "rename", fail_rename)

    with pytest.raises(HTTPException) as exc:
        run_as(ALICE, local._native_upload_dataset_response, lease)
    assert exc.value.status_code == (413 if failure == "limit" else 400)
    assert list(store.iterdir()) == [existing]
    items = run_as(ALICE, local.list_local_datasets_response).datasets
    assert [item.path for item in items] == [str(existing.resolve())]
    assert existing.read_bytes() == b"text\nkeep\n"
    assert source.read_bytes() == (b"" if failure == "empty" else payload)
