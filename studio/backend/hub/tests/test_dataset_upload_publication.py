# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Inventory must see complete uploads, with cleanup after failed publication."""

import asyncio
import threading
from contextlib import contextmanager, suppress
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import HTTPException, UploadFile

from hub.services.datasets import local


# --- Test helpers ---
@pytest.fixture
def upload_dir(tmp_path, monkeypatch):
    root = tmp_path / "uploads"
    root.mkdir()
    monkeypatch.setattr(local, "DATASET_UPLOAD_DIR", root)
    monkeypatch.setattr(local, "LOCAL_DATASETS_ROOT", tmp_path / "recipes")
    return root


def _upload(payload: bytes) -> UploadFile:
    return UploadFile(file = BytesIO(payload), filename = "rows.csv")


def _assert_completed(response, upload_dir, payload):
    stored = Path(response.stored_path)
    assert response.filename == "rows.csv"
    assert stored.parent == upload_dir
    assert stored.name.endswith("_rows.csv")
    assert len(stored.name.split("_", 1)[0]) == 32
    assert stored.read_bytes() == payload
    items = local.list_local_datasets_response().datasets
    item = next(item for item in items if item.path == str(stored.resolve()))
    assert (item.id, item.label, item.source) == (stored.name, "rows.csv", "upload")
    assert set(upload_dir.iterdir()) == {Path(item.path) for item in items}


# --- End test helpers ---


def test_inventory_excludes_in_progress_multipart_upload(upload_dir, monkeypatch):
    existing = upload_dir / "existing.csv"
    existing.write_bytes(b"a,b\n1,2\n")
    payload = b"prompt,response\n" + b"hello,world\n" * 200_000
    upload = _upload(payload)
    original_read = upload.read

    async def exercise():
        paused, resume = asyncio.Event(), asyncio.Event()
        reads = 0

        async def gated_read(size = -1):
            nonlocal reads
            assert size == local.LOCAL_UPLOAD_CHUNK_BYTES
            reads += 1
            if reads == 2:
                paused.set()
                await resume.wait()
            return await original_read(size)

        monkeypatch.setattr(upload, "read", gated_read)
        task = asyncio.create_task(local.upload_dataset_response(upload))
        try:
            await asyncio.wait_for(paused.wait(), timeout = 5)
            pending = set(upload_dir.iterdir()) - {existing}
            assert len(pending) == 1
            assert pending.pop().stat().st_size == local.LOCAL_UPLOAD_CHUNK_BYTES
            during = local.list_local_datasets_response().datasets
            assert [item.path for item in during] == [str(existing.resolve())]
        finally:
            resume.set()
            response = await task
            await upload.close()
        _assert_completed(response, upload_dir, payload)
        assert existing.read_bytes() == b"a,b\n1,2\n"
        assert len(local.list_local_datasets_response().datasets) == 2

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["empty", "limit", "read", "write", "rename", "cancel"])
def test_failed_multipart_upload_leaves_no_file(upload_dir, monkeypatch, failure):
    existing = upload_dir / "existing.csv"
    existing.write_bytes(b"text\nkeep\n")
    payload = b"x" * local.LOCAL_UPLOAD_CHUNK_BYTES + b"y"
    upload = _upload(b"" if failure == "empty" else payload)
    original_read = upload.read
    reads = 0

    async def failing_read(size = -1):
        nonlocal reads
        reads += 1
        if reads == 2:
            if failure == "read":
                raise OSError("read failed")
            if failure == "cancel":
                raise asyncio.CancelledError
        return await original_read(size)

    monkeypatch.setattr(upload, "read", failing_read)
    if failure == "limit":
        monkeypatch.setattr(local, "get_upload_limit_mb", lambda: 1)
    if failure == "write":
        real_open = open

        @contextmanager
        def failing_open(*args, **kwargs):
            with real_open(*args, **kwargs) as target:

                class Writer:
                    def write(self, chunk):
                        target.write(chunk)
                        raise OSError("write failed")

                yield Writer()

        monkeypatch.setattr(local, "open", failing_open, raising = False)
    if failure == "rename":

        def fail_rename(self, destination):
            assert self.read_bytes() == payload
            raise OSError("rename failed")

        monkeypatch.setattr(Path, "rename", fail_rename)

    expected = (
        asyncio.CancelledError
        if failure == "cancel"
        else HTTPException
        if failure in {"empty", "limit"}
        else OSError
    )

    async def exercise():
        try:
            with pytest.raises(expected) as exc:
                await local.upload_dataset_response(upload)
            if failure in {"empty", "limit"}:
                assert exc.value.status_code == (413 if failure == "limit" else 400)
        finally:
            await upload.close()

    asyncio.run(exercise())
    assert list(upload_dir.iterdir()) == [existing]
    assert existing.read_bytes() == b"text\nkeep\n"


def test_declared_oversize_upload_creates_no_file(upload_dir, monkeypatch):
    monkeypatch.setattr(local, "get_upload_limit_mb", lambda: 1)
    upload = _upload(b"unused")
    upload.size = 1024 * 1024 + 1

    async def exercise():
        try:
            with pytest.raises(HTTPException) as exc:
                await local.upload_dataset_response(upload)
            assert exc.value.status_code == 413
        finally:
            await upload.close()

    asyncio.run(exercise())
    assert list(upload_dir.iterdir()) == []


@pytest.mark.parametrize("write_fails", [False, True])
def test_cancelled_upload_waits_for_active_write(upload_dir, monkeypatch, write_fails):
    upload = _upload(b"text\nhello\n")
    real_open = open
    resume, finished = threading.Event(), threading.Event()
    closed = []

    async def exercise():
        loop = asyncio.get_running_loop()
        paused = asyncio.Event()

        @contextmanager
        def gated_open(*args, **kwargs):
            with real_open(*args, **kwargs) as target:

                class Writer:
                    def write(self, chunk):
                        loop.call_soon_threadsafe(paused.set)
                        try:
                            assert resume.wait(timeout = 5)
                            target.write(chunk)
                            if write_fails:
                                raise OSError("write failed after cancellation")
                        finally:
                            finished.set()

                try:
                    yield Writer()
                finally:
                    closed.append(finished.is_set())

        monkeypatch.setattr(local, "open", gated_open, raising = False)
        task = asyncio.create_task(local.upload_dataset_response(upload))
        try:
            await asyncio.wait_for(paused.wait(), timeout = 5)
            for _ in range(2):
                task.cancel()
                # FIFO callbacks let the cancellation run before inspecting its effects.
                checkpoint = loop.create_future()
                loop.call_soon(checkpoint.set_result, None)
                await checkpoint
                assert not task.done()
                assert not closed
                assert len(list(upload_dir.iterdir())) == 1
                assert local.list_local_datasets_response().datasets == []
        finally:
            resume.set()
            with suppress(asyncio.CancelledError):
                await task
            await upload.close()
        assert finished.is_set()
        assert closed == [True]

    asyncio.run(exercise())
    assert list(upload_dir.iterdir()) == []
