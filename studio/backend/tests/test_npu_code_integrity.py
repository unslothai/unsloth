# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Smart App Control / WDAC refusing lemond.exe or flm.exe surfaces as a clear NPU error, not a 500."""

from __future__ import annotations

import asyncio
import errno

import pytest
from fastapi import HTTPException

from core.inference import lemonade_server as ls
from core.inference import npu_backend as nb


def _refusal(
    winerror: int | None,
    code: int = errno.EACCES,
    text: str = "blocked",
) -> OSError:
    exc = OSError(code, text)
    # Set by hand so the case runs on every OS; only Windows fills winerror itself.
    exc.winerror = winerror
    return exc


@pytest.fixture
def popen(monkeypatch):
    calls = []

    def install(exc: BaseException):
        def fake(*args, **kwargs):
            calls.append(args)
            raise exc

        monkeypatch.setattr(ls.subprocess, "Popen", fake)
        monkeypatch.setattr(nb.subprocess, "Popen", fake)
        return calls

    return install


def _server(tmp_path) -> ls.LemonadeServer:
    binary = tmp_path / "lemonade" / "2026.44.1" / "lemond.exe"
    binary.parent.mkdir(parents = True)
    binary.write_bytes(b"MZ")
    return ls.LemonadeServer(
        binary,
        cache_dir = tmp_path / "cache",
        config_dir = tmp_path / "config",
        flm_model_dir = tmp_path / "flm",
    )


@pytest.mark.parametrize(
    "winerror, expect",
    [
        (4551, "Smart App Control"),
        (1260, "administers this device"),
        (577, "Reinstalling the local runtime"),
    ],
)
def test_a_refused_lemond_names_the_policy_and_is_not_retried(tmp_path, popen, winerror, expect):
    calls = popen(_refusal(winerror))
    server = _server(tmp_path)
    with pytest.raises(ls.LemonadeUnavailable) as info:
        server.start(timeout = 5)
    message = str(info.value)
    assert expect in message
    assert str(server.binary.parent) in message
    assert isinstance(info.value.__cause__, OSError)
    assert len(calls) == 1
    assert not server.is_alive() and server._process is None


def test_the_policy_text_alone_is_recognised(tmp_path, popen):
    popen(_refusal(None, text = "An Application Control policy has blocked this file"))
    with pytest.raises(ls.LemonadeUnavailable, match = "administers this device"):
        _server(tmp_path).start(timeout = 5)


@pytest.mark.parametrize(
    "exc",
    [
        FileNotFoundError(errno.ENOENT, "No such file or directory"),
        PermissionError(errno.EACCES, "Permission denied"),
        _refusal(5, text = "Access is denied"),
    ],
)
def test_other_spawn_errors_are_left_alone(tmp_path, popen, exc):
    popen(exc)
    with pytest.raises(type(exc)) as info:
        _server(tmp_path).start(timeout = 5)
    assert info.value is exc


class _Installer:
    def __init__(self, binary):
        self.binary = binary

    def installed_lemond(self, root):
        return self.binary

    def install(
        self,
        root,
        cancel = None,
    ):
        return self.binary


@pytest.fixture
def backend(tmp_path, monkeypatch):
    binary = tmp_path / "lemonade" / "2026.44.1" / "lemond.exe"
    binary.parent.mkdir(parents = True)
    binary.write_bytes(b"MZ")
    installer = _Installer(binary)
    monkeypatch.setattr(nb, "_installer_module", lambda: installer)
    monkeypatch.setattr(
        nb,
        "detect_amd_npu",
        lambda: {"present": True, "supported": True, "family": "XDNA2", "name": "NPU Strix Halo"},
    )
    npu = nb.LemonadeNpuBackend(root = tmp_path / "lemonade")
    yield npu
    npu.shutdown()


def test_the_model_list_route_answers_409_with_the_reason(backend, popen, monkeypatch):
    from routes import npu as npu_routes

    popen(_refusal(4551))
    monkeypatch.setattr(npu_routes, "get_npu_backend", lambda: backend)
    with pytest.raises(HTTPException) as info:
        asyncio.run(npu_routes.list_npu_models())
    assert info.value.status_code == 409
    assert "Smart App Control" in info.value.detail


def test_enable_fails_with_the_reason_in_status(backend, popen):
    popen(_refusal(4551))
    with pytest.raises(nb.NpuError):
        backend.enable()
    status = backend.status()
    assert status["state"] == "failed"
    assert "Smart App Control" in status["error"]


def test_a_refused_flm_validate_names_the_policy(backend, popen, monkeypatch):
    flm = backend.root / "cache" / "bin" / "flm" / "npu" / "flm.exe"
    monkeypatch.setattr(backend, "_flm_binary", lambda: flm)
    popen(_refusal(4551))
    report = backend._validate()
    assert report["ready"] is False
    assert "Smart App Control" in report["problems"][0]
    assert str(flm.parent) in report["problems"][0]


def test_a_missing_flm_validate_keeps_the_old_wording(backend, popen, monkeypatch):
    flm = backend.root / "flm.exe"
    monkeypatch.setattr(backend, "_flm_binary", lambda: flm)
    popen(FileNotFoundError(errno.ENOENT, "No such file or directory"))
    assert backend._validate()["problems"][0].startswith("flm validate could not run:")


class _Exited:
    pid = 4242

    def __init__(
        self,
        returncode: int,
        stdout: str = "",
    ):
        self.returncode = returncode
        self._stdout = stdout

    def communicate(self, timeout = None):
        return self._stdout, ""

    def poll(self):
        return self.returncode


@pytest.mark.parametrize("returncode", [0xC0E90002, -1058471934])
def test_flm_refused_a_dll_names_the_policy(backend, monkeypatch, returncode):
    # flm.exe links every model DLL, so one refused DLL ends it with no output.
    flm = backend.root / "flm" / "npu" / "flm.exe"
    monkeypatch.setattr(backend, "_flm_binary", lambda: flm)
    monkeypatch.setattr(nb.subprocess, "Popen", lambda *a, **k: _Exited(returncode))
    monkeypatch.setattr(nb, "adopt_pid", lambda pid: None)
    monkeypatch.setattr(nb, "forget_pid", lambda pid: None)
    problem = backend._validate()["problems"][0]
    assert "Smart App Control" in problem and str(flm.parent) in problem


@pytest.mark.parametrize("returncode", [1, -9, 0xC0000005])
def test_other_silent_flm_exits_keep_the_old_wording(backend, monkeypatch, returncode):
    monkeypatch.setattr(backend, "_flm_binary", lambda: backend.root / "flm.exe")
    monkeypatch.setattr(nb.subprocess, "Popen", lambda *a, **k: _Exited(returncode, "garbage"))
    monkeypatch.setattr(nb, "adopt_pid", lambda pid: None)
    monkeypatch.setattr(nb, "forget_pid", lambda pid: None)
    assert backend._validate()["problems"] == ["flm validate returned no report: garbage"]
