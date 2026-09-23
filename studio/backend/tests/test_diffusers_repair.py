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

_REAL_PEER_HOLDS_PASS = dr._peer_holds_pass


class _Dist:
    def __init__(self, direct_url):
        self._direct_url = direct_url

    def read_text(self, name):
        if name != "direct_url.json" or self._direct_url is None:
            return None
        return json.dumps(self._direct_url)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(dr, "_thread", None)
    monkeypatch.setattr(dr, "_installed", False)
    monkeypatch.delenv(dr.DISABLE_ENV_VAR, raising=False)
    monkeypatch.delenv("UNSLOTH_DIFFUSERS_MAIN", raising=False)
    monkeypatch.setattr(dr, "_peer_holds_pass", lambda: False)


def _installed_diffusers(monkeypatch, direct_url):
    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: _Dist(direct_url))


@pytest.mark.parametrize(
    "direct_url, expected",
    [
        (None, True),  # a PyPI wheel records no direct_url.json
        # A local checkout, editable or not, is the user's own build.
        ({"url": "file:///src/diffusers", "dir_info": {"editable": True}}, False),
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
        wait=None,
    ):
        self.pid, self.returncode, self._wait = 0, returncode, wait

    def communicate(self, timeout=None):
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
        return _Proc(dr._INSTALLED, wait=release)

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
        monkeypatch.setattr(dr.subprocess, "Popen", lambda argv, code=code, **kw: _Proc(code))
        assert dr.start_diffusers_autorepair_if_needed() is True
        dr._thread.join(5)
        assert dr.diffusers_repair_installed() is False


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process tree")
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
        encoding="utf-8",
    )
    monkeypatch.setattr(dr, "_INSTALLER", installer)
    monkeypatch.setattr(dr, "_REPAIR_TIMEOUT_S", 3)
    waited = []
    monkeypatch.setattr(dr, "_wait_for_peer_pass", lambda: waited.append(True))
    dr._run_repair()
    assert waited == [True], "a timed-out repair must hold the gate for a peer still in the pass"
    child = int(pid_file.read_text())
    for _ in range(50):
        if not _pid_alive(child) or _pid_is_zombie(child):
            break
        time.sleep(0.1)
    else:
        os.kill(child, 9)
        pytest.fail("the installer's child outlived the timed-out repair")
    assert dr.diffusers_repair_installed() is False


def _peer_pass(monkeypatch, *uncontended):
    import contextlib
    import types

    monkeypatch.setattr(dr, "_peer_holds_pass", _REAL_PEER_HOLDS_PASS)
    held = iter(uncontended)
    fake = types.SimpleNamespace(pass_lock=lambda: contextlib.nullcontext(next(held)))
    monkeypatch.setitem(sys.modules, "studio.install_manifest", fake)


def test_the_gate_waits_for_a_peer_to_leave_the_pass(monkeypatch):
    """The repair's installer may have been waiting behind a sibling's repair or an update when it
    timed out; that peer can still be rewriting diffusers."""
    _peer_pass(monkeypatch, False, False, True)
    polls = []
    monkeypatch.setattr(dr.time, "sleep", lambda seconds: polls.append(seconds))
    dr._wait_for_peer_pass()
    assert polls == [dr._PEER_POLL_S] * 2


def test_a_start_during_a_peers_pass_waits_it_out(monkeypatch):
    """A peer swapping diffusers can have removed its metadata, which reads as no index release."""
    _peer_pass(monkeypatch, False)
    _installed_diffusers(
        monkeypatch, {"url": "https://github.com/huggingface/diffusers", "vcs_info": {}}
    )
    started = []
    monkeypatch.setattr(dr.subprocess, "Popen", lambda argv, **kw: started.append(argv) or _Proc(1))
    assert dr.start_diffusers_autorepair_if_needed() is True
    dr._thread.join(5)
    assert started and started[0][-1] == "--repair-diffusers-main"


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
    monkeypatch.setenv("UV_OFFLINE", "1")
    env = dr._repair_env()
    assert "HF_TOKEN" not in env and "UV_INDEX_URL" not in env
    assert env["UNSLOTH_DIFFUSERS_MAIN"] == "1"
    assert env["UV_OFFLINE"] == "1", "an offline install must not reach GitHub from the repair"


def test_the_load_gate_waits_for_a_running_repair(monkeypatch):
    """Importing diffusers mid-install reads half-replaced files and pins the release for the
    session, so a load during the repair is refused with a retry message instead."""
    from core.inference import diffusion_families as fam

    monkeypatch.setattr(dr, "diffusers_repair_in_flight", lambda: True)
    monkeypatch.delitem(sys.modules, "diffusers", raising=False)
    with pytest.raises(ValueError, match="installing the pinned diffusers build"):
        fam.assert_pipeline_class_available("QwenImage21Pipeline", "qwen-image-2.1")


def test_minimax_music3_is_refused_before_eviction_while_the_repair_runs(monkeypatch):
    """Its worker is a separate process that imports diffusers and never sees this one's repair."""
    import asyncio
    import types

    from fastapi import HTTPException

    import routes.inference as ri

    monkeypatch.setattr(dr, "diffusers_repair_in_flight", lambda: True)
    config = types.SimpleNamespace(audio_type="minimax_music3", is_lora=False, identifier="x/y")
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            ri._preflight_native_audio_placement(
                config,
                types.SimpleNamespace(audio_device=None),
                types.SimpleNamespace(requested_gpu_ids=None),
            )
        )
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == dr.IN_FLIGHT_MESSAGE


def test_the_repair_is_decided_before_the_socket_binds():
    """The lifespan yields, and the server accepts requests, before the post-warm thread runs, so a
    repair started there leaves a window where a load imports the release being replaced. Checked on
    the source because running the real lifespan brings up the whole backend."""
    import ast

    tree = ast.parse((_BACKEND / "main.py").read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    def first_call(function, name):
        lines = [
            node.lineno
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == name
        ]
        return min(lines) if lines else None

    lifespan = functions["lifespan"]
    start = first_call(lifespan, "start_diffusers_autorepair_if_needed")
    assert start is not None
    assert start < first_call(lifespan, "_start_post_warm_thread")
    assert start < min(n.lineno for n in ast.walk(lifespan) if isinstance(n, ast.Yield))
    post_warm = functions["_post_warm_background_work"]
    assert first_call(post_warm, "start_diffusers_autorepair_if_needed") is None


def test_a_finished_repair_behind_a_loaded_release_asks_for_a_restart(monkeypatch):
    from core.inference import diffusion_families as fam

    monkeypatch.setattr(dr, "_installed", True)
    message = fam._too_old_message("QwenImage21Pipeline", "qwen-image-2.1", "0.40.0")
    assert "Restart Unsloth Studio" in message


def test_the_installer_exposes_the_repair_flag():
    source = dr._INSTALLER.read_text(encoding="utf-8")
    assert '["--repair-diffusers-main"]' in source
