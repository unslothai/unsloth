# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The unauthenticated frontend catch-all never resolves a network share path (Windows NTLM sign-in)."""

import asyncio
from pathlib import Path

import pytest
from fastapi import FastAPI
from starlette.requests import Request

import main
from utils.datasets import remote_images

# GET /%5C%5Cevil.example%5Cshare%5Cx.png and GET ///evil.example/share/x.png
SHARES = [
    "\\\\evil.example\\share\\x.png",
    "//evil.example/share/x.png",
]


def _serve_frontend(tmp_path):
    (tmp_path / "index.html").write_text("<html></html>")
    app = FastAPI()
    main.setup_frontend(app, tmp_path)
    route = next(r for r in app.routes if getattr(r, "path", "") == "/{full_path:path}")
    return route.endpoint


def _request():
    return Request(
        {"type": "http", "method": "GET", "path": "/", "headers": [], "query_string": b""}
    )


@pytest.mark.parametrize("full_path", SHARES)
def test_catchall_never_resolves_a_share(monkeypatch, tmp_path, full_path):
    monkeypatch.setattr(remote_images, "_WINDOWS", True)
    endpoint = _serve_frontend(tmp_path)
    touched = []
    resolve = Path.resolve
    monkeypatch.setattr(
        Path, "resolve", lambda self, *a, **k: touched.append(str(self)) or resolve(self, *a, **k)
    )
    response = asyncio.run(endpoint(_request(), full_path))
    assert response.status_code in (403, 404)
    assert not any("evil.example" in p for p in touched), touched


def test_catchall_still_serves_the_app_shell(monkeypatch, tmp_path):
    monkeypatch.setattr(remote_images, "_WINDOWS", True)
    endpoint = _serve_frontend(tmp_path)
    (tmp_path / "favicon.ico").write_bytes(b"ico")
    response = asyncio.run(endpoint(_request(), "favicon.ico"))
    assert response.status_code == 200
