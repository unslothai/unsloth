# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import httpx
import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from utils import offline_docs_settings as docs


@pytest.fixture()
def settings_store(monkeypatch):
    store = {}
    monkeypatch.setattr(
        "storage.studio_db.get_app_setting",
        lambda key, fallback = None: store.get(key, fallback),
    )
    monkeypatch.setattr(
        "storage.studio_db.upsert_app_settings",
        lambda values: store.update(values) or values,
    )
    monkeypatch.setattr(docs, "_current_release", lambda: "2026.10.1")
    return store


def _pages(*bodies: str) -> httpx.Client:
    """A client serving ``bodies`` as part 0, 1, ... of llms-full.txt, each linking the next."""
    urls = [docs.DOCS_URL] + [f"{docs.DOCS_URL}/{index}" for index in range(1, len(bodies))]
    served = {}
    for index, body in enumerate(bodies):
        if index + 1 < len(bodies):
            body = f"{body}\n\n---\n\n[Next Page]({urls[index + 1]})"
        served[urls[index]] = body
    requested = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        if str(request.url) not in served:
            return httpx.Response(404, text = "Not found")
        return httpx.Response(200, text = served[str(request.url)])

    client = httpx.Client(transport = httpx.MockTransport(handler))
    client.requested = requested
    return client


def test_fetch_follows_every_next_page_link():
    client = _pages("# Part one", "# Part two", "# Part three")

    text = docs.fetch_docs_text(client)

    assert client.requested == [docs.DOCS_URL, f"{docs.DOCS_URL}/1", f"{docs.DOCS_URL}/2"]
    assert text.index("# Part one") < text.index("# Part two") < text.index("# Part three")
    assert text.endswith("\n")


def test_fetch_ignores_links_to_other_hosts_and_stops_on_a_loop():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text = (
                "# Docs\n[Next Page](https://example.com/llms-full.txt/1)\n"
                f"[Next Page]({docs.DOCS_URL})"
            ),
        )

    client = httpx.Client(transport = httpx.MockTransport(handler))

    # The only matching link points back at the first page, so one request is made.
    assert docs.fetch_docs_text(client).startswith("# Docs")


def test_fetch_raises_on_http_errors():
    client = httpx.Client(transport = httpx.MockTransport(lambda request: httpx.Response(503)))

    with pytest.raises(httpx.HTTPStatusError):
        docs.fetch_docs_text(client)


def test_folder_must_exist(settings_store, tmp_path):
    with pytest.raises(ValueError, match = "does not exist"):
        docs.set_offline_docs_settings(path = str(tmp_path / "missing"))
    file_path = tmp_path / "file.txt"
    file_path.write_text("x")
    with pytest.raises(ValueError, match = "must be a folder"):
        docs.set_offline_docs_settings(path = str(file_path))
    assert docs.get_offline_docs_dir() is None


def test_settings_round_trip_and_partial_updates(settings_store, tmp_path):
    docs.set_offline_docs_settings(path = str(tmp_path), auto_update = True)
    docs.set_offline_docs_settings(auto_update = False)

    status = docs.offline_docs_status()
    assert status["path"] == str(tmp_path.resolve())
    assert status["auto_update"] is False
    assert status["file"] is None

    docs.set_offline_docs_settings(path = None)
    assert docs.offline_docs_status()["path"] is None


def test_auto_update_defaults_off(settings_store):
    assert docs.get_offline_docs_auto_update() is False
    assert docs.offline_docs_refresh_due() is False


def test_refresh_writes_a_dated_file_and_replaces_older_copies(settings_store, tmp_path):
    old = tmp_path / docs.docs_file_name(date(2026, 1, 2))
    old.write_text("old docs")
    unrelated = tmp_path / "Unsloth Docs notes.md"
    unrelated.write_text("mine")
    docs.set_offline_docs_settings(path = str(tmp_path))

    status = docs.refresh_offline_docs(
        fetch = lambda: "# Unsloth docs\n", today = lambda: date(2026, 10, 26)
    )

    expected = tmp_path.resolve() / "Unsloth Docs LLMs Full Text [Documentation] (2026-10-26).md"
    assert status["file"] == str(expected)
    assert expected.read_text(encoding = "utf-8") == "# Unsloth docs\n"
    assert status["release"] == "2026.10.1"
    assert status["error"] is None
    assert not old.exists()
    assert unrelated.read_text() == "mine"
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


def test_failed_refresh_keeps_the_previous_copy_and_records_the_error(settings_store, tmp_path):
    docs.set_offline_docs_settings(path = str(tmp_path))
    docs.refresh_offline_docs(fetch = lambda: "# v1\n", today = lambda: date(2026, 10, 1))

    def offline():
        raise httpx.ConnectError("no network")

    with pytest.raises(httpx.ConnectError):
        docs.refresh_offline_docs(fetch = offline, today = lambda: date(2026, 10, 26))

    status = docs.offline_docs_status()
    assert Path(status["file"]).read_text(encoding = "utf-8") == "# v1\n"
    assert status["error"] == "Could not reach unsloth.ai to download the docs."
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


def test_refresh_reports_a_folder_that_was_removed(settings_store, tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    docs.set_offline_docs_settings(path = str(folder))
    folder.rmdir()

    with pytest.raises(ValueError, match = "no longer exists"):
        docs.refresh_offline_docs(fetch = lambda: "# docs\n")
    assert "no longer exists" in docs.offline_docs_status()["error"]


def test_refresh_is_due_once_per_release(settings_store, tmp_path, monkeypatch):
    docs.set_offline_docs_settings(path = str(tmp_path), auto_update = True)
    assert docs.offline_docs_refresh_due() is True

    docs.refresh_offline_docs(fetch = lambda: "# docs\n")
    assert docs.offline_docs_refresh_due() is False

    monkeypatch.setattr(docs, "_current_release", lambda: "2026.11.1")
    assert docs.offline_docs_refresh_due() is True

    # Deleting the copy, or pointing at another folder, also asks for a fresh one.
    monkeypatch.setattr(docs, "_current_release", lambda: "2026.10.1")
    Path(docs.offline_docs_status()["file"]).unlink()
    assert docs.offline_docs_refresh_due() is True


def test_startup_hook_refreshes_only_when_due(settings_store, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(docs, "refresh_offline_docs", lambda: calls.append(1))

    assert docs.refresh_offline_docs_if_due() is False
    docs.set_offline_docs_settings(path = str(tmp_path), auto_update = True)
    assert docs.refresh_offline_docs_if_due() is True
    assert calls == [1]


def test_startup_hook_swallows_failures(settings_store, tmp_path, monkeypatch):
    docs.set_offline_docs_settings(path = str(tmp_path), auto_update = True)

    def offline():
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(docs, "refresh_offline_docs", offline)
    assert docs.refresh_offline_docs_if_due() is False


def test_concurrent_refresh_is_refused(settings_store, tmp_path):
    docs.set_offline_docs_settings(path = str(tmp_path))
    with docs._refresh_lock:
        with pytest.raises(docs.OfflineDocsBusyError):
            docs.refresh_offline_docs(fetch = lambda: "# docs\n")
        assert docs.offline_docs_status()["refreshing"] is True


def test_settings_route_round_trip(settings_store, tmp_path):
    from routes import settings as settings_route

    saved = settings_route.update_offline_docs(
        settings_route.OfflineDocsPayload(path = str(tmp_path), auto_update = True),
        current_subject = "studio-user",
        via_api_key = False,
    )
    # Omitting the path leaves it alone.
    toggled = settings_route.update_offline_docs(
        settings_route.OfflineDocsPayload(auto_update = False),
        current_subject = "studio-user",
        via_api_key = False,
    )
    loaded = settings_route.get_offline_docs(current_subject = "studio-user")

    assert saved.path == str(tmp_path.resolve())
    assert saved.auto_update is True
    assert toggled.path == saved.path
    assert loaded == toggled
    assert loaded.auto_update is False


def test_settings_route_rejects_api_key_writes(settings_store, tmp_path):
    from fastapi import HTTPException
    from routes import settings as settings_route

    with pytest.raises(HTTPException) as exc_info:
        settings_route.update_offline_docs(
            settings_route.OfflineDocsPayload(path = str(tmp_path)),
            current_subject = "api-client",
            via_api_key = True,
        )
    assert exc_info.value.status_code == 403
    with pytest.raises(HTTPException) as exc_info:
        settings_route.refresh_offline_docs_now(current_subject = "api-client", via_api_key = True)
    assert exc_info.value.status_code == 403
    assert docs.get_offline_docs_dir() is None


def test_refresh_route_maps_errors(settings_store, tmp_path, monkeypatch):
    from fastapi import HTTPException
    from routes import settings as settings_route

    with pytest.raises(HTTPException) as exc_info:
        settings_route.refresh_offline_docs_now(current_subject = "studio-user", via_api_key = False)
    assert exc_info.value.status_code == 400

    docs.set_offline_docs_settings(path = str(tmp_path))

    def offline():
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(
        settings_route, "refresh_offline_docs", lambda: docs.refresh_offline_docs(fetch = offline)
    )
    with pytest.raises(HTTPException) as exc_info:
        settings_route.refresh_offline_docs_now(current_subject = "studio-user", via_api_key = False)
    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "Could not reach unsloth.ai to download the docs."
