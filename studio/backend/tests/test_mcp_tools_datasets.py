# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json

from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from mcp_server import create_studio_mcp

from .mcp_harness import call_tool, fake_studio, served

PAYLOADS = {
    ("GET", "/api/hub/datasets/local"): {
        "datasets": [
            {
                "id": "my-chats",
                "label": "My chats",
                "path": "/srv/unsloth/datasets/my-chats",
                "rows": 120,
            }
        ]
    },
    ("GET", "/api/hub/datasets/cached"): {
        "cached": [
            {
                "repo_id": "mlabonne/FineTome-100k",
                "size_bytes": 1024,
                "cache_path": "/srv/hf/datasets--mlabonne--FineTome-100k",
                "load_cache_path": "/srv/hf/datasets/mlabonne___fine_tome-100k",
            }
        ]
    },
    ("POST", "/api/hub/datasets/check-format"): {
        "requires_manual_mapping": False,
        "detected_format": "sharegpt",
        "columns": ["conversations", "source"],
        "suggested_mapping": {"conversations": "messages"},
        "preview_samples": [{"conversations": "hi"}],
        "total_rows": 100000,
        "warning": None,
    },
    ("POST", "/api/hub/datasets/download"): {
        "repo_id": "mlabonne/FineTome-100k",
        "state": "running",
        "accepted": True,
        "generation": 1,
    },
    ("GET", "/api/hub/datasets/download-status"): {
        "state": "error",
        "error": "401 gated dataset",
        "generation": 1,
        "attempt": 1,
    },
}


def _studio(overrides = None):
    routes = {
        key: (lambda payload: lambda request, body: payload)(value)
        for key, value in PAYLOADS.items()
    }
    routes[("POST", "/api/hub/datasets/download")] = lambda request, body: JSONResponse(
        PAYLOADS[("POST", "/api/hub/datasets/download")], status_code = 202
    )
    routes.update(overrides or {})
    return fake_studio(routes)


def _call(
    monkeypatch,
    studio,
    args,
    headers = None,
):
    with TestClient(served(create_studio_mcp(), studio, monkeypatch = monkeypatch)) as http:
        return call_tool(http, "datasets", args, headers = headers)


def test_list_names_datasets_without_their_paths(monkeypatch):
    studio = _studio()
    result = _call(monkeypatch, studio, {"action": "list"})
    assert result["structuredContent"] == {
        "local": [{"id": "my-chats", "label": "My chats", "source": "local", "rows": 120}],
        "cached": [{"repo_id": "mlabonne/FineTome-100k", "size_bytes": 1024}],
        "format": None,
        "download": None,
    }
    assert "/srv" not in json.dumps(result)


def test_check_format_sends_the_hub_token_header(monkeypatch):
    studio = _studio()
    result = _call(
        monkeypatch,
        studio,
        {"action": "check_format", "name": "mlabonne/FineTome-100k", "hf_token": "hf_arg"},
    )
    assert result["structuredContent"]["format"] == {
        "detected_format": "sharegpt",
        "requires_manual_mapping": False,
        "columns": ["conversations", "source"],
        "suggested_mapping": {"conversations": "messages"},
        "is_image": False,
        "is_audio": False,
        "total_rows": 100000,
        "warning": None,
    }
    (_m, _p, headers, body) = next(c for c in studio.state.calls if c[1].endswith("/check-format"))
    assert headers["x-unsloth-hf-token"] == "hf_arg"
    assert json.loads(body) == {
        "dataset_name": "mlabonne/FineTome-100k",
        "is_vlm": False,
        "train_split": "train",
    }


def test_check_format_needs_a_name(monkeypatch):
    studio = _studio()
    result = _call(monkeypatch, studio, {"action": "check_format"})
    assert result["isError"] is True
    assert "check_format needs name" in result["content"][0]["text"]
    assert studio.state.calls == []


def test_download_starts_and_returns_the_202_state(monkeypatch):
    studio = _studio()
    result = _call(
        monkeypatch,
        studio,
        {"action": "download", "repo_id": "mlabonne/FineTome-100k"},
        headers = {"X-Unsloth-HF-Token": "hf_h"},
    )
    assert result["structuredContent"]["download"] == {
        "repo_id": "mlabonne/FineTome-100k",
        "state": "running",
        "error": None,
    }
    (_m, _p, headers, body) = next(c for c in studio.state.calls if c[1].endswith("/download"))
    assert headers["x-unsloth-hf-token"] == "hf_h"
    assert json.loads(body) == {"repo_id": "mlabonne/FineTome-100k"}


def test_status_reads_the_download_state(monkeypatch):
    queries = []

    def status(request, body):
        queries.append(dict(request.query_params))
        return PAYLOADS[("GET", "/api/hub/datasets/download-status")]

    studio = _studio({("GET", "/api/hub/datasets/download-status"): status})
    result = _call(monkeypatch, studio, {"action": "status", "repo_id": "a/b"})
    assert result["structuredContent"]["download"] == {
        "repo_id": "a/b",
        "state": "error",
        "error": "401 gated dataset",
    }
    assert queries == [{"repo_id": "a/b"}]


def test_download_and_status_need_a_repo(monkeypatch):
    studio = _studio()
    for action in ("download", "status"):
        assert _call(monkeypatch, studio, {"action": action})["isError"] is True
    assert studio.state.calls == []


def test_datasets_annotations():
    tool = {t.name: t for t in asyncio.run(create_studio_mcp().list_tools())}["datasets"]
    assert tool.annotations.readOnlyHint is False
    assert tool.annotations.destructiveHint is False
    assert tool.annotations.openWorldHint is False
    assert tool.output_schema is not None
