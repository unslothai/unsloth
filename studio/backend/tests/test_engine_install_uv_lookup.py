# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""#18492: vLLM / SGLang installs find a per-user uv off PATH, and every attempt is told apart."""

import json
import os
import stat
import threading

import pytest

from core.inference import engine_install as install
from core.inference import wsl_host


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    root = tmp_path / "engines"
    monkeypatch.setattr(install, "engine_root", lambda: root)
    monkeypatch.setattr(install, "support_reason", lambda engine = "vllm", **_: None)
    monkeypatch.setattr(install, "_jobs", {})
    monkeypatch.setattr(install, "_cancels", {})
    monkeypatch.setattr(install, "_studio_packages", lambda: {})
    monkeypatch.setattr(wsl_host, "active", lambda: False)
    return root


def job(root):
    return json.loads((root / "vllm.job.json").read_text(encoding = "utf-8"))


def per_user_uv(monkeypatch, home):
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    uv = home / ".local" / "bin" / ("uv.exe" if os.name == "nt" else "uv")
    uv.parent.mkdir(parents = True)
    uv.write_text("")
    uv.chmod(uv.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(install.shutil, "which", lambda _: None)
    return uv


def test_install_uses_a_per_user_uv_that_is_not_on_path(isolated, monkeypatch, tmp_path):
    uv = per_user_uv(monkeypatch, tmp_path / "home")
    calls = []

    def run(engine, argv, cancel):
        calls.append(argv)
        raise RuntimeError("stop after the first uv call")

    monkeypatch.setattr(install, "_run", run)
    install._install("vllm", threading.Event())
    assert calls and calls[0][:2] == [str(uv), "venv"]
    assert job(isolated)["state"] == "error"
    assert job(isolated)["message"] == "stop after the first uv call"


def test_uv_on_path_still_wins(isolated, monkeypatch, tmp_path):
    per_user_uv(monkeypatch, tmp_path / "home")
    monkeypatch.setattr(
        install.shutil, "which", lambda name: "/on/path/uv" if name == "uv" else None
    )
    assert install._uv_executable() == "/on/path/uv"


def test_each_attempt_records_when_it_ended(isolated, monkeypatch):
    clock = iter([1000.0, 1007.0])
    monkeypatch.setattr(install.time, "time", lambda: next(clock))
    message = "uv is unavailable. Repair the Unsloth installation and retry."

    install._update("vllm", state = "running", phase = "queued", message = "Preparing installation")
    assert job(isolated)["finished_at"] is None
    install._update("vllm", state = "error", phase = None, message = message)
    first = job(isolated)
    assert first["finished_at"] == 1000.0

    install._update("vllm", state = "running", phase = "queued", message = "Preparing installation")
    assert job(isolated)["finished_at"] is None
    install._update("vllm", state = "error", phase = None, message = message)
    second = job(isolated)
    assert second["message"] == first["message"]
    assert second["finished_at"] == 1007.0


def test_progress_updates_keep_a_running_attempt_unfinished(isolated, monkeypatch):
    monkeypatch.setattr(install.time, "time", lambda: 1000.0)
    install._update("vllm", state = "running", phase = "queued", message = "Preparing installation")
    install._update(
        "vllm", phase = "installing", message = "Downloading and installing engine packages"
    )
    install._update("vllm", activity = "torch 2.11")
    assert job(isolated)["finished_at"] is None
    install._update("vllm", state = "success", phase = "ready", message = "Engine installed")
    assert job(isolated)["finished_at"] == 1000.0
