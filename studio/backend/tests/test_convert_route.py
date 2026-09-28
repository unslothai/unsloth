# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routes import convert


def _client():
    app = FastAPI()
    app.include_router(convert.router)
    return TestClient(app)


def test_status_tracks_stages_and_completion():
    convert._jobs.clear()
    fake = mock.MagicMock(
        returncode = 0, stdout = iter(["noise\n", "STAGE loading\n", "STAGE saving\n"])
    )
    with mock.patch.object(convert.subprocess, "Popen", return_value = fake) as popen:
        client = _client()
        assert (
            client.post("/api/convert", json = {"model_id": "a/b", "format": "ov_int4"}).status_code
            == 200
        )
        argv = popen.call_args.args[0]
        assert argv[-2:] == ["a/b", "ov_int4"]  # passed as argv, not spliced into source
        assert client.get("/api/convert/status", params = {"model_id": "a/b"}).json() == {
            "state": "done",
            "stage": "saving",
        }


def test_failed_process_reports_error():
    convert._jobs.clear()
    fake = mock.MagicMock(returncode = 1, stdout = iter([]))
    with mock.patch.object(convert.subprocess, "Popen", return_value = fake):
        client = _client()
        client.post("/api/convert", json = {"model_id": "x", "format": "ov_int8"})
        assert (
            client.get("/api/convert/status", params = {"model_id": "x"}).json()["state"] == "error"
        )


def test_rejects_unknown_format_and_unknown_job():
    convert._jobs.clear()
    client = _client()
    assert client.post("/api/convert", json = {"model_id": "x", "format": "gguf"}).status_code == 400
    assert client.get("/api/convert/status", params = {"model_id": "nope"}).status_code == 404
