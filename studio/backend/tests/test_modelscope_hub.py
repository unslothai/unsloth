# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The ModelScope adapter answers the Hub routes huggingface_hub and the browser use."""

from __future__ import annotations

from pathlib import Path
import sys
import types as _types

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hub.modelscope import upstream
from hub.modelscope.router import build_router

SHA = "a" * 40
TAG = "b" * 40


def _pkt(line: str) -> bytes:
    return f"{len(line) + 4:04x}".encode() + line.encode()


REFS = (
    _pkt("# service=git-upload-pack\n")
    + b"0000"
    + _pkt(f"{SHA} HEAD\0multi_ack\n")
    + _pkt(f"{SHA} refs/heads/master\n")
    + _pkt(f"{TAG} refs/tags/v1\n")
    + b"0000"
)
MODEL_FILES = [
    {"Path": "config.json", "Size": 659, "Sha256": "c" * 64, "IsLFS": False},
    {"Path": "model.safetensors", "Size": 1000, "Sha256": "d" * 64, "IsLFS": True},
    {"Path": "sub", "Type": "tree"},
    {"Path": "sub/tokenizer.json", "Size": 5, "Sha256": "e" * 64, "IsLFS": False},
]
DATASET_FILES = [
    {"Path": "dataset_infos.json", "Size": 9, "Sha256": "f" * 64, "IsLFS": False},
    {"Path": "data.jsonl", "Size": 90, "Sha256": "9" * 64, "IsLFS": False},
]
ITEM = {
    "id": "Qwen/Tiny",
    "downloads": 7,
    "likes": 2,
    "params": 494032768,
    "last_modified": "2025-02-26T18:24:53Z",
    "tags": [
        "model_type:qwen2",
        "library:transformer",
        "library:safetensors",
        "task:text-generation",
    ],
    "tasks": ["text-generation"],
}


def _upstream(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path in ("/Qwen/Tiny.git/info/refs", "/datasets/org/ds.git/info/refs"):
        return httpx.Response(200, content = REFS)
    if path.endswith(".git/info/refs"):
        return httpx.Response(500 if "boom" in path else 404)
    if path == "/api/v1/models/Qwen/Tiny/repo/files":
        return httpx.Response(200, json = {"Data": {"Files": MODEL_FILES}})
    if path == "/api/v1/datasets/org/ds/repo/tree":
        return httpx.Response(200, json = {"Data": {"Files": DATASET_FILES}})
    if path == "/openapi/v1/models/Qwen/Tiny":
        return httpx.Response(200, json = {"data": ITEM})
    if path == f"/models/Qwen/Tiny/resolve/{SHA}/config.json":
        return httpx.Response(200, content = b'{"model_type": "qwen2"}')
    if path == "/openapi/v1/models":
        return httpx.Response(200, json = {"data": {"models": [ITEM], "total_count": 120}})
    return httpx.Response(404)


@pytest.fixture
def hub(monkeypatch):
    monkeypatch.setattr(upstream, "transport", httpx.MockTransport(_upstream))
    monkeypatch.setattr(upstream, "_clients", upstream.weakref.WeakKeyDictionary())
    upstream._cache.clear()
    app = FastAPI()
    app.include_router(build_router(browser = False))
    return TestClient(app, follow_redirects = False)


def test_search_answers_in_hub_shape_with_absolute_paging(hub):
    response = hub.get("/api/models", params = {"search": "tiny", "sort": "downloads"})
    [entry] = response.json()
    assert entry["id"] == "Qwen/Tiny" and entry["lastModified"] == "2025-02-26T18:24:53Z"
    assert entry["library_name"] == "transformers" and entry["pipeline_tag"] == "text-generation"
    assert entry["safetensors"]["total"] == 494032768 and entry["config"] == {"model_type": "qwen2"}
    assert response.headers["link"] == (
        '<http://testserver/api/models?search=tiny&sort=downloads&p=2>; rel="next"'
    )
    assert hub.get("/api/models", params = {"filter": "gguf"}).json() == []
    assert len(hub.get("/api/models", params = {"pipeline_tag": "Text-Generation"}).json()) == 1
    assert hub.get("/api/models", params = {"pipeline_tag": "image-classification"}).json() == []


def test_info_names_the_commit_and_files(hub):
    body = hub.get("/api/models/Qwen/Tiny").json()
    assert body["sha"] == SHA
    assert {
        "rfilename": "model.safetensors",
        "size": 1000,
        "blobId": "d" * 64,
        "lfs": {"sha256": "d" * 64, "size": 1000, "pointerSize": 134},
    } in body["siblings"]
    assert hub.get("/api/models/Qwen/Tiny/revision/v1").json()["sha"] == TAG
    for path, code in [
        ("/api/models/Qwen/Tiny/revision/nope", "RevisionNotFound"),
        ("/api/models/no/one", "RepoNotFound"),
        ("/api/models/no/one/auth-check", "RepoNotFound"),
    ]:
        response = hub.get(path)
        assert (response.status_code, response.headers.get("x-error-code")) == (404, code)


def test_resolve_describes_then_redirects_to_the_pinned_file(hub):
    head = hub.head("/Qwen/Tiny/resolve/main/model.safetensors")
    assert head.headers["x-repo-commit"] == SHA
    assert (
        head.headers["location"]
        == f"https://www.modelscope.cn/models/Qwen/Tiny/resolve/{SHA}/model.safetensors"
    )
    assert (
        head.headers["x-linked-etag"] == f'"{"d" * 64}"' and head.headers["x-linked-size"] == "1000"
    )
    get = hub.get("/Qwen/Tiny/resolve/main/sub/tokenizer.json")
    assert get.status_code == 302
    assert (
        get.headers["location"]
        == f"https://www.modelscope.cn/models/Qwen/Tiny/resolve/{SHA}/sub/tokenizer.json"
    )
    missing = hub.head("/Qwen/Tiny/resolve/main/nope.json")
    assert (missing.status_code, missing.headers["x-error-code"]) == (404, "EntryNotFound")


def test_browser_resolve_relays_to_a_session_and_redirects_anyone_else(hub, monkeypatch):
    import hub.modelscope.router as router

    seen = []
    monkeypatch.setattr(
        upstream,
        "transport",
        httpx.MockTransport(lambda request: seen.append(request) or _upstream(request)),
    )
    signed_in = False

    async def session(_request) -> bool:
        return signed_in

    monkeypatch.setattr(router, "signed_in", session)
    app = FastAPI()
    app.include_router(build_router(browser = True))
    browser = TestClient(app, follow_redirects = False)
    response = browser.get("/datasets/org/ds/resolve/main/img/a.png")
    assert (
        response.headers["location"]
        == "https://www.modelscope.cn/datasets/org/ds/resolve/master/img/a.png"
    )
    assert seen == []
    assert (
        browser.get(
            "/Qwen/Tiny/resolve/main/config.json", headers = {"Authorization": "Bearer stale"}
        ).status_code
        == 401
    )
    signed_in = True
    relayed = browser.get("/Qwen/Tiny/resolve/main/config.json")
    assert (relayed.status_code, relayed.json()) == (200, {"model_type": "qwen2"})
    assert relayed.headers["x-repo-commit"] == SHA


def test_tree_and_paths_info_walk_the_listing(hub):
    top = hub.get(f"/api/models/Qwen/Tiny/tree/{SHA}").json()
    assert [(e["path"], e["type"]) for e in top] == [
        ("config.json", "file"),
        ("model.safetensors", "file"),
        ("sub", "directory"),
    ]
    assert "sub/tokenizer.json" in [
        e["path"] for e in hub.get("/api/models/Qwen/Tiny/tree/main?recursive=1").json()
    ]
    found = hub.post(
        "/api/models/Qwen/Tiny/paths-info/main", data = {"paths": ["config.json", "sub", "nope"]}
    )
    assert [e["path"] for e in found.json()] == ["config.json", "sub"]


def test_modelscope_dataset_metadata_is_hidden_from_datasets(hub):
    assert [e["path"] for e in hub.get("/api/datasets/org/ds/tree/main").json()] == ["data.jsonl"]
    assert hub.head("/datasets/org/ds/resolve/main/dataset_infos.json").status_code == 404
    assert hub.head("/datasets/org/ds/resolve/main/data.jsonl").status_code == 302


def test_an_aliased_repo_is_served_under_its_hugging_face_id(hub, monkeypatch):
    assert upstream.upstream_id("model", "minimaxai/minimax-music3") == "MiniMax/MiniMax-Music3"
    monkeypatch.setattr(
        upstream,
        "_ALIAS_INDEX",
        {"model": {"hf/tiny": "Qwen/Tiny"}, "dataset": {"hf/ds": "org/ds"}},
    )
    body = hub.get("/api/models/HF/Tiny").json()
    assert (body["id"], body["sha"], len(body["siblings"])) == ("HF/Tiny", SHA, 3)
    location = hub.head("/HF/Tiny/resolve/main/config.json").headers["location"]
    assert location == f"https://www.modelscope.cn/models/Qwen/Tiny/resolve/{SHA}/config.json"
    assert hub.get("/HF/Tiny").headers["location"] == "https://www.modelscope.cn/models/Qwen/Tiny"
    assert [e["path"] for e in hub.get("/api/datasets/hf/ds/tree/main").json()] == ["data.jsonl"]
    assert (
        hub.get("/datasets/hf/ds").headers["location"]
        == "https://www.modelscope.cn/datasets/org/ds"
    )


def test_a_listed_hugging_face_pin_is_served_from_the_matching_modelscope_commit(hub, monkeypatch):
    dac = ("model", "ibm-research/DAC.speech.v1.0", "1ea7f64cd0678415e2d8c32d67b190722cb9b149")
    assert upstream.upstream_commit(*dac) == "3268c083add6c07ae3c4a36650863eb2c3bb6717"
    monkeypatch.setattr(upstream, "_PINNED_COMMITS", {("model", "qwen/tiny", "c" * 40): SHA})
    head = hub.head(f"/Qwen/Tiny/resolve/{'C' * 40}/config.json")
    assert head.headers["x-repo-commit"] == "c" * 40
    assert head.headers["location"].endswith(f"/Qwen/Tiny/resolve/{SHA}/config.json")
    assert hub.get(f"/api/models/Qwen/Tiny/revision/{'c' * 40}").json()["sha"] == "c" * 40
    assert upstream.branch_url("model", "Qwen/Tiny", "c" * 40, "a").endswith(f"/{SHA}/a")
    assert upstream.upstream_commit("model", "Qwen/Tiny", "D" * 40) == "d" * 40


def test_writes_bad_ids_and_upstream_failures(hub):
    assert hub.post("/api/repos/create", json = {}).status_code == 403
    assert hub.get("/api/models/a/b..c").status_code == 400
    assert hub.get("/Qwen/Tiny/resolve/main/sub/%2E%2E/config.json").status_code == 400
    assert hub.get("/api/models/boom/repo/auth-check").status_code == 502
