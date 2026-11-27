# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The unauthenticated frontend route resolves the request path on the event loop; a very deep
path costs the square of its parts there, so it is answered 404 before any resolve()."""

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


def _client(tmp_path):
    import main

    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents = True)
    (dist / "index.html").write_text("<!doctype html><title>shell</title>", encoding = "utf-8")
    (dist / "assets" / "app.js").write_text("export {};", encoding = "utf-8")
    app = FastAPI()
    assert main.setup_frontend(app, dist)
    return TestClient(app, base_url = "http://127.0.0.1:8888", client = ("127.0.0.1", 40000))


def test_a_deep_path_is_refused_before_it_is_resolved(tmp_path, monkeypatch):
    client = _client(tmp_path)
    resolved = []
    original = Path.resolve
    monkeypatch.setattr(
        Path,
        "resolve",
        lambda self, *a, **k: resolved.append(len(str(self))) or original(self, *a, **k),
    )
    assert client.get("/" + "a/" * 2000).status_code == 404
    assert not any(length > 2000 for length in resolved)


def test_ordinary_frontend_paths_still_work(tmp_path):
    client = _client(tmp_path)
    assert client.get("/assets/app.js").status_code == 200
    assert "shell" in client.get("/chat/some-thread-id").text
