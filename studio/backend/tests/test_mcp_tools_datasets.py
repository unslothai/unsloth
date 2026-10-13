# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json

from fastapi.responses import JSONResponse

from .mcp_harness import call_to, queries, run_tool

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


def _call(
    monkeypatch,
    args,
    headers = None,
):
    download = JSONResponse(PAYLOADS[("POST", "/api/hub/datasets/download")], status_code = 202)
    routes = {**PAYLOADS, ("POST", "/api/hub/datasets/download"): download}
    return run_tool(monkeypatch, routes, "datasets", args, headers = headers)


def test_list_names_datasets_without_their_paths(monkeypatch):
    result, _studio = _call(monkeypatch, {"action": "list"})
    assert result["structuredContent"] == {
        "local": [{"id": "my-chats", "label": "My chats", "source": "local", "rows": 120}],
        "cached": [{"repo_id": "mlabonne/FineTome-100k", "size_bytes": 1024}],
        "format": None,
        "download": None,
    }
    assert "/srv" not in json.dumps(result)


def test_check_format_sends_the_hub_token_header(monkeypatch):
    result, studio = _call(
        monkeypatch,
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
    (_m, _p, headers, body) = call_to(studio, "/api/hub/datasets/check-format")
    assert headers["x-unsloth-hf-token"] == "hf_arg"
    assert json.loads(body) == {
        "dataset_name": "mlabonne/FineTome-100k",
        "is_vlm": False,
        "train_split": "train",
    }


def test_download_starts_and_returns_the_202_state(monkeypatch):
    result, studio = _call(
        monkeypatch,
        {"action": "download", "repo_id": "mlabonne/FineTome-100k"},
        headers = {"X-Unsloth-HF-Token": "hf_h"},
    )
    assert result["structuredContent"]["download"] == {
        "repo_id": "mlabonne/FineTome-100k",
        "state": "running",
        "error": None,
    }
    (_m, _p, headers, body) = call_to(studio, "/api/hub/datasets/download")
    assert headers["x-unsloth-hf-token"] == "hf_h"
    assert json.loads(body) == {"repo_id": "mlabonne/FineTome-100k"}


def test_status_reads_the_download_state(monkeypatch):
    result, studio = _call(monkeypatch, {"action": "status", "repo_id": "a/b"})
    assert result["structuredContent"]["download"] == {
        "repo_id": "a/b",
        "state": "error",
        "error": "401 gated dataset",
    }
    assert queries(studio, "/api/hub/datasets/download-status") == [{"repo_id": "a/b"}]
