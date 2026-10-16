# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routes import convert


def _client(tmp_path = None, owner = True):
    app = FastAPI()
    app.include_router(convert.router)
    if owner:
        app.dependency_overrides[convert._require_installation_owner] = lambda: None
    if tmp_path is not None:
        mock.patch.object(convert, "exports_root", return_value = tmp_path).start()
    # The Popen fakes have no real pid to adopt.
    mock.patch.object(convert, "adopt_pid").start()
    return TestClient(app)


def teardown_function():
    mock.patch.stopall()


def test_status_tracks_stages_and_completion(tmp_path):
    convert._jobs.clear()
    fake = mock.MagicMock(
        returncode = 0, stdout = iter(["noise\n", "STAGE loading\n", "STAGE saving\n"])
    )
    with mock.patch.object(convert.subprocess, "Popen", return_value = fake) as popen:
        client = _client(tmp_path)
        assert (
            client.post("/api/convert", json = {"model_id": "a/b", "format": "ov_int4"}).status_code
            == 200
        )
        argv = popen.call_args.args[0]
        # passed as argv, not spliced into source; written under exports, not the server's cwd
        assert argv[-3:] == ["a/b", "ov_int4", str(tmp_path / "b-openvino" / "ov_int4")]
        assert argv[1] == "-P"  # cwd off sys.path
        assert client.get("/api/convert/status", params = {"model_id": "a/b"}).json() == {
            "state": "done",
            "stage": "saving",
        }


def test_failed_process_reports_error(tmp_path):
    convert._jobs.clear()
    fake = mock.MagicMock(returncode = 1, stdout = iter([]))
    with mock.patch.object(convert.subprocess, "Popen", return_value = fake):
        client = _client(tmp_path)
        client.post("/api/convert", json = {"model_id": "x", "format": "ov_int8"})
        assert (
            client.get("/api/convert/status", params = {"model_id": "x"}).json()["state"] == "error"
        )


def test_rejects_unknown_format_and_unknown_job():
    convert._jobs.clear()
    client = _client()
    assert client.post("/api/convert", json = {"model_id": "x", "format": "gguf"}).status_code == 400
    assert client.get("/api/convert/status", params = {"model_id": "nope"}).status_code == 404


def test_output_dir_stays_under_exports(tmp_path):
    with mock.patch.object(convert, "exports_root", return_value = tmp_path):
        for model_id in ("../../etc", "/abs/path/My Model/", "C:\\models\\x", ".."):
            out = convert.output_dir_for(model_id, "ov_int8")
            assert out.parent.parent == tmp_path and out.name == "ov_int8"


def test_routes_require_the_installation_owner():
    with mock.patch.object(convert.subprocess, "Popen") as popen:
        client = _client(owner = False)
        assert (
            client.post("/api/convert", json = {"model_id": "a/b", "format": "ov_int4"}).status_code
            == 401
        )
        assert client.get("/api/convert/status", params = {"model_id": "a/b"}).status_code == 401
    popen.assert_not_called()


def test_reads_child_output_as_utf8(tmp_path):
    # A locale codec (cp1252 on Windows) raises on tqdm's partial blocks, e.g. U+258F.
    with mock.patch.object(convert.subprocess, "Popen") as popen:
        convert._jobs.clear()
        convert._jobs["m"] = {"state": "running", "stage": "starting"}
        popen.return_value = mock.MagicMock(returncode = 0, stdout = iter([]))
        with mock.patch.object(convert, "adopt_pid"):
            convert.run_conversion_task("m", "ov_int4", tmp_path / "m" / "ov_int4")
        kwargs = popen.call_args.kwargs
        assert kwargs["encoding"] == "utf-8" and kwargs["errors"] == "replace"


def test_failed_reader_stops_the_child(tmp_path):
    def broken():
        yield "STAGE loading\n"
        raise UnicodeDecodeError("cp1252", b"\x8f", 0, 1, "undefined")

    convert._jobs.clear()
    convert._jobs["m"] = {"state": "running", "stage": "starting"}
    fake = mock.MagicMock(pid = 4242, stdout = broken())
    fake.poll.return_value = None
    with (
        mock.patch.object(convert.subprocess, "Popen", return_value = fake),
        mock.patch.object(convert, "adopt_pid"),
        mock.patch.object(convert, "terminate_pid") as term,
    ):
        convert.run_conversion_task("m", "ov_int4", tmp_path / "m" / "ov_int4")
    assert convert._jobs["m"]["state"] == "error"
    term.assert_called_once_with(4242, timeout = 5.0, owner_verified = True)
