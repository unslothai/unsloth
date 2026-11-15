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

from fastapi import HTTPException

from routes import settings as route
from utils import offline_docs_settings as docs

UI = {"current_subject": "studio-user", "via_api_key": False}


@pytest.fixture()
def store(monkeypatch):
    values = {}
    monkeypatch.setattr(
        "storage.studio_db.get_app_setting", lambda key, fallback = None: values.get(key, fallback)
    )
    monkeypatch.setattr(
        "storage.studio_db.upsert_app_settings", lambda new: values.update(new) or new
    )
    monkeypatch.setattr(docs, "_current_release", lambda: "2026.10.1")
    return values


def _client(pages: dict[str, str]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        body = pages.get(str(request.url))
        return httpx.Response(200, text = body) if body is not None else httpx.Response(404)

    return httpx.Client(transport = httpx.MockTransport(handler))


def _chain(*bodies: str) -> httpx.Client:
    urls = [docs.DOCS_URL] + [f"{docs.DOCS_URL}/{i}" for i in range(1, len(bodies))]
    return _client(
        {
            url: body + (f"\n\n---\n\n[Next Page]({urls[i + 1]})" if i + 1 < len(urls) else "")
            for i, (url, body) in enumerate(zip(urls, bodies))
        }
    )


def _offline():
    raise httpx.ConnectError("no network")


def test_fetch_joins_every_part_in_order():
    text = docs.fetch_docs_text(_chain("# One", "# Two", "# Three"))
    assert text.index("# One") < text.index("# Two") < text.index("# Three")
    assert text.endswith("\n")


def test_fetch_ignores_other_hosts():
    page = "# Docs\n[Next Page](https://example.com/llms-full.txt/1)"
    assert docs.fetch_docs_text(_client({docs.DOCS_URL: page})) == page + "\n"


def test_fetch_refuses_partial_documents(monkeypatch):
    loop = f"# Docs\n[Next Page]({docs.DOCS_URL}/1)"
    with pytest.raises(ValueError, match = "pagination"):
        docs.fetch_docs_text(_client({docs.DOCS_URL: loop, f"{docs.DOCS_URL}/1": loop}))
    monkeypatch.setattr(docs, "MAX_DOCS_PARTS", 2)
    with pytest.raises(ValueError, match = "pagination"):
        docs.fetch_docs_text(_chain("# One", "# Two", "# Three"))
    with pytest.raises(httpx.HTTPStatusError):
        docs.fetch_docs_text(_client({}))


def test_settings_validate_and_update_partially(store, tmp_path):
    with pytest.raises(ValueError, match = "does not exist"):
        docs.set_offline_docs_settings(path = str(tmp_path / "missing"))
    (tmp_path / "f.txt").write_text("x")
    with pytest.raises(ValueError, match = "must be a folder"):
        docs.set_offline_docs_settings(path = str(tmp_path / "f.txt"))
    assert docs.offline_docs_status()["auto_update"] is False

    docs.set_offline_docs_settings(path = str(tmp_path), auto_update = True)
    docs.set_offline_docs_settings(auto_update = False)
    status = docs.offline_docs_status()
    assert (status["path"], status["auto_update"], status["file"]) == (
        str(tmp_path.resolve()),
        False,
        None,
    )
    docs.set_offline_docs_settings(path = None)
    assert docs.offline_docs_status()["path"] is None


def test_refresh_writes_one_dated_copy(store, tmp_path):
    old = tmp_path / docs.docs_file_name(date(2026, 1, 2))
    old.write_text("old")
    mine = tmp_path / "Unsloth Docs notes.md"
    mine.write_text("mine")
    docs.set_offline_docs_settings(path = str(tmp_path))

    status = docs.refresh_offline_docs(fetch = lambda: "# docs\n", today = lambda: date(2026, 10, 26))

    expected = tmp_path.resolve() / "Unsloth Docs LLMs Full Text [Documentation] (2026-10-26).md"
    assert status["file"] == str(expected) and status["release"] == "2026.10.1"
    assert expected.read_text(encoding = "utf-8") == "# docs\n"
    assert not old.exists() and mine.read_text() == "mine"
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_refresh_keeps_the_previous_copy(store, tmp_path):
    docs.set_offline_docs_settings(path = str(tmp_path))
    docs.refresh_offline_docs(fetch = lambda: "# v1\n", today = lambda: date(2026, 10, 1))
    with pytest.raises(httpx.ConnectError):
        docs.refresh_offline_docs(fetch = _offline, today = lambda: date(2026, 10, 26))

    status = docs.offline_docs_status()
    assert Path(status["file"]).read_text(encoding = "utf-8") == "# v1\n"
    assert status["error"] == "Could not reach unsloth.ai to download the docs."

    tmp_path.joinpath(Path(status["file"]).name).unlink()
    tmp_path.rmdir()
    with pytest.raises(ValueError, match = "no longer exists"):
        docs.refresh_offline_docs(fetch = lambda: "# docs\n")


def test_status_survives_an_unreadable_folder(store, tmp_path, monkeypatch):
    docs.set_offline_docs_settings(path = str(tmp_path))
    docs.refresh_offline_docs(fetch = lambda: "# docs\n")

    def denied(self):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "is_file", denied)
    assert docs.offline_docs_status()["file"] is None


def test_refresh_is_due_once_per_build(store, tmp_path, monkeypatch):
    assert docs.offline_docs_refresh_due() is False
    docs.set_offline_docs_settings(path = str(tmp_path), auto_update = True)
    assert docs.offline_docs_refresh_due() is True
    docs.refresh_offline_docs(fetch = lambda: "# docs\n")
    assert docs.offline_docs_refresh_due() is False

    monkeypatch.setattr(docs, "_current_release", lambda: "2026.11.1")
    assert docs.offline_docs_refresh_due() is True

    monkeypatch.setattr(docs, "_current_release", lambda: "dev")
    monkeypatch.setattr(docs, "_package_version", lambda: "2026.10.1")
    docs.refresh_offline_docs(fetch = lambda: "# docs\n")
    assert docs.offline_docs_refresh_due() is False
    monkeypatch.setattr(docs, "_package_version", lambda: "2026.11.1")
    assert docs.offline_docs_refresh_due() is True

    Path(docs.offline_docs_status()["file"]).unlink()
    monkeypatch.setattr(docs, "_package_version", lambda: "2026.10.1")
    assert docs.offline_docs_refresh_due() is True


def test_startup_hook(store, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(docs, "refresh_offline_docs", lambda: calls.append(1))
    assert docs.refresh_offline_docs_if_due() is False
    docs.set_offline_docs_settings(path = str(tmp_path), auto_update = True)
    assert docs.refresh_offline_docs_if_due() is True and calls == [1]

    monkeypatch.setattr(docs, "refresh_offline_docs", _offline)
    assert docs.refresh_offline_docs_if_due() is False


def test_concurrent_refresh_is_refused(store, tmp_path):
    docs.set_offline_docs_settings(path = str(tmp_path))
    with docs._refresh_lock:
        with pytest.raises(docs.OfflineDocsBusyError):
            docs.refresh_offline_docs(fetch = lambda: "# docs\n")
        assert docs.offline_docs_status()["refreshing"] is True


def test_routes(store, tmp_path, monkeypatch):
    for write in (
        lambda: route.update_offline_docs(
            route.OfflineDocsPayload(path = str(tmp_path)), current_subject = "k", via_api_key = True
        ),
        lambda: route.refresh_offline_docs_now(current_subject = "k", via_api_key = True),
    ):
        with pytest.raises(HTTPException) as exc:
            write()
        assert exc.value.status_code == 403
    with pytest.raises(HTTPException) as exc:
        route.refresh_offline_docs_now(**UI)
    assert exc.value.status_code == 400

    saved = route.update_offline_docs(
        route.OfflineDocsPayload(path = str(tmp_path), auto_update = True), **UI
    )
    toggled = route.update_offline_docs(route.OfflineDocsPayload(auto_update = False), **UI)
    assert saved.path == toggled.path == str(tmp_path.resolve())
    assert route.get_offline_docs(current_subject = "studio-user") == toggled

    monkeypatch.setattr(
        route, "refresh_offline_docs", lambda: docs.refresh_offline_docs(fetch = _offline)
    )
    with pytest.raises(HTTPException) as exc:
        route.refresh_offline_docs_now(**UI)
    assert (exc.value.status_code, exc.value.detail) == (
        502,
        "Could not reach unsloth.ai to download the docs.",
    )
