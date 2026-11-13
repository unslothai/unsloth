# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The dataset upload's file work runs on a worker thread, which owns the import lock until it ends."""

from __future__ import annotations

import asyncio
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from fastapi import FastAPI
from PIL import Image

import routes.training as training
from auth.authentication import get_current_subject


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (10, 20, 30)).save(buf, format = "PNG")
    return buf.getvalue()


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(training.router, prefix = "/api/train")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"

    @app.get("/tick")
    async def tick():
        return {"ok": True}

    return app


@pytest.fixture
def ds_root(monkeypatch, tmp_path):
    import utils.paths as up

    root = tmp_path / "assets" / "datasets"
    root.mkdir(parents = True)
    monkeypatch.setattr(up, "datasets_root", lambda: root)
    return root


def _parts(names):
    png = _png_bytes()
    return [
        ("files", (n, b"a caption" if n.endswith(".txt") else png, "application/octet-stream"))
        for n in names
    ]


def test_other_requests_are_served_while_an_upload_works(app, ds_root, monkeypatch):
    real_summary = training._diffusion_dataset_summary

    def slow_summary(folder):
        time.sleep(0.4)
        return real_summary(folder)

    monkeypatch.setattr(training, "_diffusion_dataset_summary", slow_summary)

    async def main():
        transport = httpx.ASGITransport(app = app)
        async with httpx.AsyncClient(transport = transport, base_url = "http://t") as client:
            upload = asyncio.ensure_future(
                client.post(
                    "/api/train/diffusion/dataset",
                    data = {"name": "styleset"},
                    files = _parts(["a.png", "a.txt"]),
                )
            )
            ticks = 0
            while not upload.done():
                await client.get("/tick")
                ticks += 1
                await asyncio.sleep(0.01)
            return await upload, ticks

    res, ticks = asyncio.run(main())
    assert res.status_code == 200, res.text
    assert res.json()["image_count"] == 1 and res.json()["caption_count"] == 1
    # Blocking on the loop would serve no tick during the 0.4 s summary.
    assert ticks >= 5


def test_cancelled_request_keeps_lock_until_worker_finishes():
    lock = threading.Lock()
    assert lock.acquire(blocking = False)
    entered = threading.Event()
    proceed = threading.Event()
    seen = {}

    def work(x, _lock, cancelled):
        try:
            entered.set()
            proceed.wait(5)
            seen["cancelled"] = cancelled.is_set()
            return x
        finally:
            _lock.release()

    async def main():
        task = asyncio.ensure_future(training._run_holding_import_lock(lock, work, 7))
        while not entered.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.sleep(0.05)
        # Pending request = training interlock and import lock still held.
        assert not task.done()
        assert lock.locked()
        proceed.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not lock.locked()

    asyncio.run(main())
    assert not lock.locked()
    assert seen == {"cancelled": True}


def test_cancel_before_worker_starts_releases_lock():
    lock = threading.Lock()
    assert lock.acquire(blocking = False)
    ran = []
    block = threading.Event()

    async def main():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers = 1))
        busy = loop.run_in_executor(None, block.wait, 5)
        task = asyncio.ensure_future(
            training._run_holding_import_lock(lock, lambda _lock, c: ran.append(1))
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not lock.locked()
        block.set()
        await busy
        await asyncio.sleep(0.05)

    asyncio.run(main())
    assert ran == []
    assert not lock.locked()


def test_cancel_during_staging_rolls_back(ds_root):
    from starlette.datastructures import UploadFile

    folder = ds_root / "newset"
    lock = threading.Lock()
    assert lock.acquire(blocking = False)
    cancelled = threading.Event()
    cancelled.set()
    files = [UploadFile(io.BytesIO(_png_bytes()), filename = "a.png")]
    with pytest.raises(asyncio.CancelledError):
        training._store_diffusion_dataset_upload(folder, "newset", files, True, lock, cancelled)
    assert not lock.locked()
    assert not folder.exists()


def test_worker_releases_lock_on_refusal(app, ds_root):
    from fastapi.testclient import TestClient

    client = TestClient(app)
    folder = ds_root / "styleset"
    folder.mkdir()
    (folder / "a.jpg").write_bytes(_png_bytes())
    res = client.post(
        "/api/train/diffusion/dataset", data = {"name": "styleset"}, files = _parts(["a.png"])
    )
    assert res.status_code == 400
    assert not training._dataset_import_lock(folder).locked()
    res = client.post(
        "/api/train/diffusion/dataset", data = {"name": "styleset"}, files = _parts(["b.png"])
    )
    assert res.status_code == 200
    assert sorted(p.name for p in folder.iterdir()) == ["a.jpg", "b.png"]
