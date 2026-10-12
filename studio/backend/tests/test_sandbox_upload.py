# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Uploading a file straight into a chat's sandbox folder (POST /sandbox/{session_id}/upload)."""

import asyncio
import io
import os
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile


@pytest.fixture
def tools(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "sb"))
    from core.inference import tools

    tools._workdirs.clear()
    return tools


@pytest.mark.parametrize("dir_fd_writes", [True, False])
def test_upload_lands_at_the_root_and_never_replaces_unless_asked(
    tools, monkeypatch, dir_fd_writes
):
    monkeypatch.setattr(tools, "_DIR_FD_WRITES", tools._DIR_FD_WRITES and dir_fd_writes)
    session = "__LOCALID_upload1"
    saved = tools.save_sandbox_upload(session, "report.xlsx", [b"PK\x03\x04", b"rest"])
    workdir = Path(tools.get_sandbox_workdir(session))
    assert saved == {"name": "report.xlsx", "path": str(workdir / "report.xlsx"), "replaced": False}
    assert (workdir / "report.xlsx").read_bytes() == b"PK\x03\x04rest"

    with pytest.raises(tools.SandboxUploadExists):
        tools.save_sandbox_upload(session, "report.xlsx", [b"second"])
    assert (workdir / "report.xlsx").read_bytes() == b"PK\x03\x04rest"

    # Replacing twice gives the same folder: the upload is idempotent.
    for _ in range(2):
        again = tools.save_sandbox_upload(session, "report.xlsx", [b"second"], overwrite = True)
        assert again["replaced"] is True
        assert (workdir / "report.xlsx").read_bytes() == b"second"

    (workdir / "out").mkdir()
    with pytest.raises(tools.SandboxUploadExists):
        tools.save_sandbox_upload(session, "out", [b"x"], overwrite = True)
    assert not [n for n in os.listdir(workdir) if n.startswith(".tmp-")], "temp file left behind"


def test_a_failed_upload_leaves_nothing_behind(tools):
    session = "__LOCALID_upload2"

    def chunks():
        yield b"part"
        raise HTTPException(status_code = 413)

    with pytest.raises(HTTPException):
        tools.save_sandbox_upload(session, "big.bin", chunks())
    workdir = Path(tools.get_sandbox_workdir(session))
    assert [n for n in os.listdir(workdir) if not n.startswith(".unsloth")] == []


def test_upload_names_stay_inside_the_folder(tools):
    assert tools.sandbox_upload_name("../../etc/passwd") == "passwd"
    assert tools.sandbox_upload_name("C:\\Users\\me\\report.xlsx") == "report.xlsx"
    assert tools.sandbox_upload_name("NUL.txt") == "_NUL.txt"
    assert tools.sandbox_upload_name("a:b?.csv") == "a_b_.csv"
    assert tools.sandbox_upload_name("..") == "upload"
    assert tools.sandbox_upload_name(None) == "upload"
    long_name = tools.sandbox_upload_name("季度" * 60 + ".xlsx")
    assert long_name.endswith(".xlsx") and len(long_name.encode()) <= 255


def test_replacing_a_link_does_not_write_through_it(tools, tmp_path):
    session = "__LOCALID_upload3"
    workdir = Path(tools.get_sandbox_workdir(session))
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"keep")
    (workdir / "data.csv").symlink_to(outside)
    with pytest.raises(tools.SandboxUploadExists):
        tools.save_sandbox_upload(session, "data.csv", [b"new"])
    tools.save_sandbox_upload(session, "data.csv", [b"new"], overwrite = True)
    assert outside.read_bytes() == b"keep"
    assert not (workdir / "data.csv").is_symlink()
    assert (workdir / "data.csv").read_bytes() == b"new"


def test_each_account_uploads_into_its_own_folder(tools):
    from utils.account_context import AccountContext, run_as

    alice, bob = AccountContext("alice-id", "alice"), AccountContext("bob-id", "bob")
    session = "__LOCALID_shared"
    run_as(alice, tools.save_sandbox_upload, session, "notes.txt", [b"alice"])
    run_as(bob, tools.save_sandbox_upload, session, "notes.txt", [b"bob"])
    alice_dir = Path(run_as(alice, tools.get_sandbox_workdir, session))
    bob_dir = Path(run_as(bob, tools.get_sandbox_workdir, session))
    assert alice_dir != bob_dir
    assert (alice_dir / "notes.txt").read_bytes() == b"alice"
    assert (bob_dir / "notes.txt").read_bytes() == b"bob"


def _upload(
    inference,
    session_id,
    name,
    data,
    overwrite = False,
):
    upload = UploadFile(file = io.BytesIO(data), filename = name)
    return asyncio.run(
        inference.upload_sandbox_file(
            session_id, file = upload, overwrite = overwrite, session = None, current_subject = "u"
        )
    )


def test_route_lists_the_upload_and_maps_conflicts_and_the_cap(tools, monkeypatch):
    from routes import inference
    from utils import upload_limits

    session = "thread-upload"
    saved = _upload(inference, session, "report.xlsx", b"xlsx-bytes")
    assert saved["name"] == "report.xlsx" and saved["replaced"] is False
    listed = [f["name"] for f in inference._sandbox_listing(inference._sandbox_dir_for(session))]
    assert "report.xlsx" in listed

    with pytest.raises(HTTPException) as conflict:
        _upload(inference, session, "report.xlsx", b"other")
    assert conflict.value.status_code == 409
    assert _upload(inference, session, "report.xlsx", b"other", overwrite = True)["replaced"]

    monkeypatch.setattr(upload_limits, "SANDBOX_UPLOAD_MAX_BYTES", 4)
    with pytest.raises(HTTPException) as too_large:
        _upload(inference, session, "big.bin", b"12345")
    assert too_large.value.status_code == 413
    assert not (Path(inference._sandbox_dir_for(session)) / "big.bin").exists()


def test_a_sandbox_file_named_upload_is_still_served():
    from routes.inference import router

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/sandbox/thread-1/upload",
        "headers": [],
        "root_path": "",
    }
    matched = [r for r in router.routes if r.matches(scope)[0].name == "FULL"]
    assert [r.endpoint.__name__ for r in matched] == ["serve_sandbox_file"]
    scope["method"] = "POST"
    matched = [r for r in router.routes if r.matches(scope)[0].name == "FULL"]
    assert [r.endpoint.__name__ for r in matched] == ["upload_sandbox_file"]
