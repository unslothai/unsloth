# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import threading
from pathlib import Path

import pytest
from anyio import CancelScope, sleep
from fastapi import BackgroundTasks
from starlette.background import BackgroundTask

from utils.temporary_file_response import TemporaryFileResponse


@pytest.fixture(params = ["jsonl", "parquet", "jsonl.zip", "webm", "gif"])
def exported_response(request, monkeypatch, tmp_path):
    payload = bytes(range(256)) * 1024
    path = tmp_path / f"download.{request.param}"
    path.write_bytes(payload)
    source = tmp_path / "original"
    source.write_bytes(b"keep the original")
    if request.param in {"webm", "gif"}:
        from core.inference import video_gallery
        from routes import video

        monkeypatch.setattr(video_gallery, "transcode_to_file", lambda *_a: path)
        response = asyncio.run(
            video.export_gallery_video("clip", format = request.param, current_subject = "user")
        )
    else:
        from routes.data_recipe import jobs

        monkeypatch.setattr(jobs, "_resolve_download_artifact_path", lambda **_kw: str(tmp_path))
        monkeypatch.setattr(
            jobs,
            "build_dataset_download",
            lambda **_kw: (path, "application/octet-stream", path.name),
        )
        background = BackgroundTasks()
        response = jobs.download_job_dataset(
            "job",
            background_tasks = background,
            export_format = "parquet" if request.param == "parquet" else "jsonl",
            artifact_path = None,
            filename = None,
        )
        if response.background is None:
            response.background = background
    yield response, path, payload
    assert source.read_bytes() == b"keep the original"


def _scope(range_header = None):
    return {
        "type": "http",
        "method": "GET",
        "path": "/download",
        "headers": [] if range_header is None else [(b"range", range_header.encode())],
        "asgi": {"spec_version": "2.4"},
    }


async def _receive():
    await asyncio.Event().wait()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("range_header", "status"), [("bytes=garbage", 400), ("bytes=999999-", 416)]
)
async def test_rejected_ranges_remove_the_generated_export(exported_response, range_header, status):
    response, path, _ = exported_response
    events = []

    async def send(message):
        events.append(message)

    await response(_scope(range_header), _receive, send)
    assert events[0]["status"] == status
    assert not path.exists()


@pytest.mark.parametrize("format", ["webm", "gif"])
@pytest.mark.parametrize(
    ("range_header", "status"), [("bytes=garbage", 400), ("bytes=999999-", 416)]
)
def test_video_export_route_cleans_up_rejected_ranges(
    monkeypatch, tmp_path, format, range_header, status
):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from core.inference import video_gallery
    from routes import video

    path = tmp_path / f"export.{format}"
    path.write_bytes(b"exported video")
    monkeypatch.setattr(video_gallery, "transcode_to_file", lambda *_a: path)
    app = FastAPI()
    app.include_router(video.router, prefix = "/api/inference")
    app.dependency_overrides[video.get_current_subject] = lambda: "user"
    with TestClient(app) as client:
        response = client.get(
            f"/api/inference/video/gallery/clip/export?format={format}",
            headers = {"Range": range_header},
        )
    assert response.status_code == status
    assert not path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["http.response.start", "http.response.body"])
async def test_send_errors_remove_the_generated_export(exported_response, phase):
    response, path, _ = exported_response

    async def send(message):
        if message["type"] == phase:
            raise OSError("client disconnected")

    with pytest.raises(OSError, match = "client disconnected"):
        await response(_scope(), _receive, send)
    assert not path.exists()


@pytest.mark.asyncio
async def test_cancelled_download_removes_the_generated_export(exported_response):
    response, path, _ = exported_response
    started = asyncio.Event()

    async def send(message):
        if message["type"] == "http.response.body":
            started.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(response(_scope(), _receive, send))
    try:
        await asyncio.wait_for(started.wait(), timeout = 5)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("range_header", [None, "bytes=2-7", "bytes=0-1,32-33"])
async def test_successful_downloads_keep_file_and_range_semantics(exported_response, range_header):
    response, path, payload = exported_response
    events = []

    async def send(message):
        events.append(message)

    await response(_scope(range_header), _receive, send)
    assert events[0]["status"] == (200 if range_header is None else 206)
    headers = dict(events[0]["headers"])
    body = b"".join(event.get("body", b"") for event in events)
    assert len(body) == int(headers[b"content-length"])
    if range_header is None:
        assert body == payload
    elif range_header == "bytes=2-7":
        assert body == payload[2:8]
        assert headers[b"content-range"] == f"bytes 2-7/{len(payload)}".encode()
    else:
        body = body.replace(b"\r\n", b"\n")
        assert b"\n\n" + payload[:2] + b"\n" in body
        assert b"\n\n" + payload[32:34] + b"\n" in body
    assert not path.exists()


@pytest.mark.asyncio
async def test_cancelled_scope_still_removes_the_generated_export(exported_response):
    response, path, _ = exported_response
    with CancelScope() as cancel_scope:

        async def send(message):
            if message["type"] == "http.response.body":
                cancel_scope.cancel()
                await sleep(0)

        await response(_scope(), _receive, send)
    assert not path.exists()


@pytest.mark.asyncio
async def test_cleanup_stays_off_the_event_loop(monkeypatch, tmp_path):
    path = tmp_path / "export"
    path.write_bytes(b"download")
    request_thread = threading.get_ident()
    cleanup_threads = []
    unlink = Path.unlink

    def checked_unlink(self, *args, **kwargs):
        if self == path:
            cleanup_threads.append(threading.get_ident())
        return unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", checked_unlink)

    async def send(_message):
        pass

    await TemporaryFileResponse(path)(_scope(), _receive, send)
    assert len(cleanup_threads) == 1
    assert cleanup_threads[0] != request_thread
    assert not path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("send_fails", [False, True])
async def test_cleanup_errors_do_not_replace_the_response_or_send_error(
    monkeypatch, tmp_path, caplog, send_fails
):
    path = tmp_path / "export"
    path.write_bytes(b"download")
    unlink = Path.unlink

    def locked_unlink(self, *args, **kwargs):
        if self == path:
            raise PermissionError("file is locked")
        return unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", locked_unlink)
    events = []

    async def send(message):
        events.append(message)
        if send_fails:
            raise OSError("client disconnected")

    response = TemporaryFileResponse(path)
    if send_fails:
        with pytest.raises(OSError, match = "client disconnected"):
            await response(_scope(), _receive, send)
    else:
        await response(_scope(), _receive, send)
        assert events[0]["status"] == 200
    assert "Could not remove temporary download" in caplog.text


@pytest.mark.asyncio
async def test_other_background_work_runs_once_before_the_file_is_removed(tmp_path):
    path = tmp_path / "export"
    path.write_bytes(b"download")
    calls = []

    async def send(_message):
        pass

    response = TemporaryFileResponse(
        path, background = BackgroundTask(lambda: calls.append(path.read_bytes()))
    )
    await response(_scope(), _receive, send)
    assert calls == [b"download"]
    assert not path.exists()
