# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json

import pytest

from hub.utils.host_paths import resolve_host_path_reference
from models.training import TrainingStartRequest

from .mcp_harness import CONFIG, b64, form, run_tool

UPLOAD = "/api/hub/datasets/upload"
STORED = "/srv/unsloth/assets/datasets/uploads/0123abcd_chats.jsonl"
PAYLOADS = {("POST", UPLOAD): {"filename": "chats.jsonl", "stored_path": STORED}}
ROWS = [{"text": "héllo"}, {"text": "bye", "n": 2}]


def test_rows_upload_as_jsonl_and_come_back_as_a_ref_training_resolves(monkeypatch):
    args = {"action": "upload", "rows": ROWS, "filename": "../chats.csv"}
    result, studio = run_tool(monkeypatch, PAYLOADS, "datasets", args)
    uploaded = result["structuredContent"]["uploaded"]
    assert uploaded["id"] == "0123abcd_chats.jsonl"
    assert uploaded["filename"] == "chats.jsonl"
    assert uploaded["ref"].startswith("ref:")
    assert resolve_host_path_reference(uploaded["ref"]) == STORED
    request = TrainingStartRequest.model_validate({**CONFIG, "local_datasets": [uploaded["ref"]]})
    assert request.local_datasets == [STORED]
    expected = '{"text": "héllo"}\n{"text": "bye", "n": 2}'.encode()
    assert form(studio, UPLOAD)["file"] == ("chats.jsonl", expected)
    assert "/srv" not in json.dumps(result)


def test_rows_default_to_rows_jsonl_and_base64_keeps_its_name(monkeypatch):
    args = {"action": "upload", "rows": ROWS}
    _result, studio = run_tool(monkeypatch, PAYLOADS, "datasets", args)
    assert form(studio, UPLOAD)["file"][0] == "rows.jsonl"
    args = {"action": "upload", "data_base64": b64(b"a,b\n1,2\n"), "filename": "C:\\x\\t.CSV"}
    result, studio = run_tool(monkeypatch, PAYLOADS, "datasets", args)
    assert result["isError"] is False
    assert form(studio, UPLOAD)["file"] == ("t.CSV", b"a,b\n1,2\n")


@pytest.mark.parametrize(
    "args,message",
    [
        ({}, "exactly one of"),
        ({"rows": ROWS, "data_base64": b64(b"x"), "filename": "a.csv"}, "exactly one of"),
        ({"rows": []}, None),
        ({"rows": ["not an object"]}, None),
        ({"data_base64": b64(b"x")}, "needs filename"),
        ({"data_base64": b64(b"x"), "filename": "a.txt"}, "Unsupported file type: .txt"),
        ({"data_base64": b64(b"x"), "filename": "noext"}, "Unsupported file type"),
        ({"data_base64": "@@@", "filename": "a.csv"}, "not valid base64"),
        ({"data_base64": "", "filename": "a.csv"}, "is empty"),
    ],
)
def test_a_refused_upload_never_reaches_a_route(monkeypatch, args, message):
    args = {"action": "upload", **args}
    result, studio = run_tool(monkeypatch, {}, "datasets", args, fallback = {})
    assert result["isError"] is True
    assert message is None or message in result["content"][0]["text"]
    assert studio.state.calls == []


def test_list_gives_each_local_dataset_a_ref(monkeypatch):
    path = "/srv/unsloth/assets/datasets/uploads/ff_chats.jsonl"
    local = {"datasets": [{"id": "ff_chats.jsonl", "path": path}, {"id": "no-path"}]}
    routes = {("GET", "/api/hub/datasets/local"): local, ("GET", "/api/hub/datasets/cached"): {}}
    result, _studio = run_tool(monkeypatch, routes, "datasets", {"action": "list"})
    first, second = result["structuredContent"]["local"]
    assert resolve_host_path_reference(first["ref"]) == path
    assert second["ref"] is None
