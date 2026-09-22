# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Diffusers self-heal: an install updated by an installer without the Diffusers main step gets the
pinned build from the backend on its next start, instead of needing a second update."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.diffusers_repair as dr  # noqa: E402


class _Dist:
    def __init__(self, direct_url):
        self._direct_url = direct_url

    def read_text(self, name):
        if name != "direct_url.json" or self._direct_url is None:
            return None
        return json.dumps(self._direct_url)


@pytest.fixture(autouse = True)
def _reset(monkeypatch):
    monkeypatch.setattr(dr, "_thread", None)
    monkeypatch.setattr(dr, "_installed", False)
    monkeypatch.delenv(dr.DISABLE_ENV_VAR, raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSERS_MAIN", raising = False)


def _installed_diffusers(monkeypatch, direct_url):
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: _Dist(direct_url))


@pytest.mark.parametrize(
    "direct_url, expected",
    [
        (None, True),  # a PyPI wheel records no direct_url.json
        ({"url": "https://files.example/x.whl", "dir_info": {}}, True),
        ({"url": "https://github.com/huggingface/diffusers", "vcs_info": {"vcs": "git"}}, False),
        (
            {"url": "https://github.com/huggingface/diffusers/archive/x.zip", "archive_info": {}},
            False,
        ),
    ],
)
def test_only_an_index_release_is_a_candidate(monkeypatch, direct_url, expected):
    _installed_diffusers(monkeypatch, direct_url)
    assert dr._diffusers_is_an_index_install() is expected


class _Proc:
    """A finished installer, for the Popen the repair starts."""

    def __init__(
        self,
        returncode,
        wait = None,
    ):
        self.pid, self.returncode, self._wait = 0, returncode, wait

    def communicate(self, timeout = None):
        if self._wait is not None:
            self._wait.wait(5)
        return "", None

    def poll(self):
        return self.returncode


def test_a_release_install_starts_one_repair_and_records_success(monkeypatch):
    _installed_diffusers(monkeypatch, None)
    calls = []
    release = threading.Event()

    def fake_popen(argv, **kwargs):
        calls.append((argv, kwargs["env"]))
        return _Proc(dr._INSTALLED, wait = release)

    monkeypatch.setattr(dr.subprocess, "Popen", fake_popen)
    assert dr.start_diffusers_autorepair_if_needed() is True
    assert dr.diffusers_repair_in_flight() is True
    assert dr.start_diffusers_autorepair_if_needed() is False, "at most once per process"
    release.set()
    dr._thread.join(5)
    assert dr.diffusers_repair_in_flight() is False
    assert dr.diffusers_repair_installed() is True
    argv, env = calls[0]
    assert argv == [sys.executable, str(dr._INSTALLER), "--repair-diffusers-main"]
    assert env["VIRTUAL_ENV"] == sys.prefix


def test_nothing_to_do_and_failure_do_not_report_an_install(monkeypatch):
    _installed_diffusers(monkeypatch, None)
    for code in (dr._NOTHING_TO_DO, 2):
        monkeypatch.setattr(dr, "_thread", None)
        monkeypatch.setattr(dr.subprocess, "Popen", lambda argv, code = code, **kw: _Proc(code))
        assert dr.start_diffusers_autorepair_if_needed() is True
        dr._thread.join(5)
        assert dr.diffusers_repair_installed() is False


@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX process tree")
def test_a_timed_out_repair_stops_the_installers_children_too(monkeypatch, tmp_path):
    """uv, not the installer, rewrites diffusers, so killing only the installer on timeout would
    reopen the load gate while the files are still being replaced."""
    from utils.process_lifetime import _pid_alive, _pid_is_zombie

    pid_file = tmp_path / "child.pid"
    installer = tmp_path / "installer.py"
    installer.write_text(
        "import pathlib, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))\n"
        "time.sleep(60)\n",
        encoding = "utf-8",
    )
    monkeypatch.setattr(dr, "_INSTALLER", installer)
    monkeypatch.setattr(dr, "_REPAIR_TIMEOUT_S", 3)
    dr._run_repair()
    child = int(pid_file.read_text())
    for _ in range(50):
        if not _pid_alive(child) or _pid_is_zombie(child):
            break
        time.sleep(0.1)
    else:
        os.kill(child, 9)
        pytest.fail("the installer's child outlived the timed-out repair")
    assert dr.diffusers_repair_installed() is False


@pytest.mark.parametrize(
    "env",
    [{dr.DISABLE_ENV_VAR: "1"}, {"UNSLOTH_DIFFUSERS_MAIN": "0"}, {"UNSLOTH_DIFFUSERS_MAIN": "off"}],
)
def test_opt_outs_start_nothing(monkeypatch, env):
    _installed_diffusers(monkeypatch, None)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(dr.subprocess, "Popen", lambda *a, **k: pytest.fail("started a repair"))
    assert dr.start_diffusers_autorepair_if_needed() is False


def test_the_pinned_build_starts_nothing(monkeypatch):
    _installed_diffusers(
        monkeypatch, {"url": "https://github.com/huggingface/diffusers", "vcs_info": {}}
    )
    monkeypatch.setattr(dr.subprocess, "Popen", lambda *a, **k: pytest.fail("started a repair"))
    assert dr.start_diffusers_autorepair_if_needed() is False


def test_secrets_and_index_redirects_stay_out_of_the_installer_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("UV_INDEX_URL", "https://evil.example/simple")
    monkeypatch.setenv("UNSLOTH_DIFFUSERS_MAIN", "1")
    env = dr._repair_env()
    assert "HF_TOKEN" not in env and "UV_INDEX_URL" not in env
    assert env["UNSLOTH_DIFFUSERS_MAIN"] == "1"


def test_the_load_gate_waits_for_a_running_repair(monkeypatch):
    """Importing diffusers mid-install reads half-replaced files and pins the release for the
    session, so a load during the repair is refused with a retry message instead."""
    from core.inference import diffusion_families as fam

    monkeypatch.setattr(dr, "diffusers_repair_in_flight", lambda: True)
    monkeypatch.delitem(sys.modules, "diffusers", raising = False)
    with pytest.raises(ValueError, match = "installing the pinned diffusers build"):
        fam.assert_pipeline_class_available("QwenImage21Pipeline", "qwen-image-2.1")


def test_a_finished_repair_behind_a_loaded_release_asks_for_a_restart(monkeypatch):
    from core.inference import diffusion_families as fam

    monkeypatch.setattr(dr, "_installed", True)
    message = fam._too_old_message("QwenImage21Pipeline", "qwen-image-2.1", "0.40.0")
    assert "Restart Unsloth Studio" in message


def test_the_installer_exposes_the_repair_flag():
    source = dr._INSTALLER.read_text(encoding = "utf-8")
    assert '["--repair-diffusers-main"]' in source
