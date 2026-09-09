# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A document upload must not occupy the event loop while it runs.

The upload routes copy the file and call start_ingestion inline, and start_ingestion
re-hashes the file and probes nvidia-smi through embedding_identity. Run on the event
loop that is seconds of dead backend: a streaming reply stops mid-token and every other
request queues behind the attachment. These tests count the trivial concurrent requests
answered while an upload is in flight, so they fail whenever the blocking work moves back
onto the loop, regardless of how the routes are spelled.
"""

import asyncio
import time

import httpx
import pytest

from core.rag import ingestion, store
from storage import rag_db
from .test_rag_native_drop_upload import SECRET, _sign

BLOCK_SECONDS = 0.6
PAYLOAD = b"alpha bravo charlie delta\n" * 320_000
POLL_INTERVAL = 0.005
# Blocked, the poller gets exactly two turns: one before the handler takes the loop and one
# after it hands it back. Free, it gets around a hundred. A count separates those by
# construction rather than by wall clock, so this file can stay in the -n 4 parallel run
# that test_scan_loras_off_event_loop, whose tick floor is loose for the same reason, is
# kept out of. The floor is 10x the blocked count and a fifth of the free one.
SERVED_FLOOR = 20


@pytest.fixture
def blocking_ingestion(monkeypatch):
    """start_ingestion stands in for the real re-hash + nvidia-smi probe."""

    def slow_start_ingestion(*args, **kwargs):
        time.sleep(BLOCK_SECONDS)
        return "doc-1", "job-1"

    monkeypatch.setattr(ingestion, "start_ingestion", slow_start_ingestion)


@pytest.fixture
def lease_secret(monkeypatch):
    import base64

    import utils.native_path_leases as leases

    monkeypatch.setenv(
        leases.LEASE_SECRET_ENV,
        base64.urlsafe_b64encode(SECRET).decode("ascii").rstrip("="),
    )
    monkeypatch.setattr(leases, "_CACHED_LEASE_SECRET", None, raising = False)
    yield
    monkeypatch.setattr(leases, "_CACHED_LEASE_SECRET", None, raising = False)


def _app():
    from fastapi import FastAPI

    from auth.authentication import get_current_subject
    from routes import rag as rag_routes

    app = FastAPI()
    app.include_router(rag_routes.router, prefix = "/api/rag")
    app.dependency_overrides[get_current_subject] = lambda: "tester"

    @app.get("/ping")
    async def ping() -> dict:
        return {"ok": True}

    return app


async def _upload_then_ping(path: str, **post_kwargs) -> tuple[httpx.Response, int, float]:
    """POST the upload while a trivial request is polled, counting the ones it answers."""
    latencies: list[float] = []
    done = asyncio.Event()

    async def poll_ping(client: httpx.AsyncClient) -> None:
        while not done.is_set():
            started = time.perf_counter()
            await asyncio.sleep(POLL_INTERVAL)
            ping = await client.get("/ping", timeout = 30.0)
            assert ping.status_code == 200
            latencies.append(time.perf_counter() - started - POLL_INTERVAL)

    transport = httpx.ASGITransport(app = _app())
    async with httpx.AsyncClient(transport = transport, base_url = "http://test") as client:
        poller = asyncio.ensure_future(poll_ping(client))
        try:
            response = await client.post(path, timeout = 30.0, **post_kwargs)
        finally:
            done.set()
            await poller
    return response, len(latencies), max(latencies, default = 0.0)


def _kb_id() -> str:
    conn = rag_db.get_connection()
    try:
        return store.create_kb(conn, name = "Latency")
    finally:
        conn.close()


def _files() -> dict:
    return {"file": ("notes.txt", PAYLOAD, "text/plain")}


def _assert_loop_stayed_free(
    response,
    served: int,
    worst: float,
    what: str = "upload",
) -> None:
    assert response.status_code == 200
    assert served >= SERVED_FLOOR, (
        f"only {served} of the polled requests completed during the {what}; "
        f"the worst waited {worst * 1000:.0f} ms"
    )


def test_kb_upload_leaves_the_event_loop_free(rag_home, blocking_ingestion):
    response, served, worst = asyncio.run(
        _upload_then_ping(f"/api/rag/knowledge-bases/{_kb_id()}/documents", files = _files())
    )
    _assert_loop_stayed_free(response, served, worst, "upload")


def test_thread_upload_leaves_the_event_loop_free(rag_home, blocking_ingestion):
    response, served, worst = asyncio.run(
        _upload_then_ping("/api/rag/threads/T1/documents", files = _files())
    )
    _assert_loop_stayed_free(response, served, worst, "upload")


def test_project_upload_leaves_the_event_loop_free(rag_home, blocking_ingestion, monkeypatch):
    from storage import studio_db

    monkeypatch.setattr(studio_db, "get_chat_project", lambda project_id: {"id": project_id})
    response, served, worst = asyncio.run(
        _upload_then_ping("/api/rag/projects/P1/documents", files = _files())
    )
    _assert_loop_stayed_free(response, served, worst, "upload")


def test_native_drop_leaves_the_event_loop_free(
    rag_home, blocking_ingestion, lease_secret, tmp_path
):
    dropped = tmp_path / "dropped.txt"
    dropped.write_bytes(PAYLOAD)
    response, served, worst = asyncio.run(
        _upload_then_ping("/api/rag/threads/T1/documents", data = {"nativePathLease": _sign(dropped)})
    )
    _assert_loop_stayed_free(response, served, worst, "drop")
