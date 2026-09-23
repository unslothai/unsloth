# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Test startup repair of the pinned Diffusers build before app imports."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.diffusers_repair as dr  # noqa: E402

_REAL_PEER_HOLDS_PASS = dr._peer_holds_pass
_REAL_INSTALLER_WOULD_SKIP = dr._installer_would_skip
_REAL_LOADED_REPLACEABLE = dr._loaded_replaceable_modules


class _Dist:
    def __init__(self, direct_url):
        self._direct_url = direct_url

    def read_text(self, name):
        if name != "direct_url.json" or self._direct_url is None:
            return None
        return json.dumps(self._direct_url)


@pytest.fixture(autouse = True)
def _reset(monkeypatch):
    monkeypatch.delenv(dr.DISABLE_ENV_VAR, raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSERS_MAIN", raising = False)
    monkeypatch.setattr(dr, "_peer_holds_pass", lambda: False)
    monkeypatch.setattr(dr, "_installer_would_skip", lambda: False)
    monkeypatch.setattr(dr, "_loaded_replaceable_modules", lambda: [])


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

    def __init__(self, returncode):
        self.pid, self.returncode = 0, returncode

    def communicate(self, timeout = None):
        return "installer output", None

    def poll(self):
        return self.returncode


def test_a_release_install_runs_the_installer_step_and_reports_it(monkeypatch):
    _installed_diffusers(monkeypatch, None)
    calls, lines = [], []

    def fake_popen(argv, **kwargs):
        calls.append((argv, kwargs["env"]))
        return _Proc(dr._INSTALLED)

    monkeypatch.setattr(dr.subprocess, "Popen", fake_popen)
    assert dr.repair_diffusers_before_imports(lines.append) is True
    # The slow fetch first, into uv's cache only, then the install from it.
    assert [argv for argv, _env in calls] == [
        [sys.executable, str(dr._INSTALLER), "--prefetch-diffusers-main"],
        [sys.executable, str(dr._INSTALLER), "--repair-diffusers-main"],
    ]
    assert all(env["VIRTUAL_ENV"] == sys.prefix for _argv, env in calls)
    assert lines == [
        "  - installing the pinned Diffusers build (first start after an update)...",
        "  - installed the pinned Diffusers build",
    ]


def test_nothing_to_do_and_failure_do_not_report_an_install(monkeypatch):
    _installed_diffusers(monkeypatch, None)
    for code, says_retry in ((dr._NOTHING_TO_DO, False), (2, True)):
        lines = []
        monkeypatch.setattr(dr.subprocess, "Popen", lambda argv, code = code, **kw: _Proc(code))
        assert dr.repair_diffusers_before_imports(lines.append) is False
        assert any("unsloth studio update" in line for line in lines) is says_retry


def _slow_installer(
    monkeypatch,
    tmp_path,
    timeout_s = 3,
):
    """An installer that outlives the budget, with a child standing in for uv."""
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
    monkeypatch.setattr(dr, "_REPAIR_TIMEOUT_S", timeout_s)
    monkeypatch.setattr(dr, "_INSTALL_MIN_TIMEOUT_S", timeout_s)
    # The tree kill names descendants before it signals, so a child forked in between is missed.
    # A slow runner's installer can still be starting at the deadline: let it spawn first.
    import utils.process_lifetime as pl

    real_terminate = pl.terminate_pid

    def terminate_after_spawn(pid, *args, **kwargs):
        for _ in range(300):
            if pid_file.exists() and pid_file.read_text():
                break
            time.sleep(0.1)
        return real_terminate(pid, *args, **kwargs)

    monkeypatch.setattr(pl, "terminate_pid", terminate_after_spawn)
    started = []
    real_popen = dr.subprocess.Popen
    monkeypatch.setattr(
        dr.subprocess,
        "Popen",
        lambda argv, **kw: started.append(argv[-1]) or real_popen(argv, **kw),
    )
    return pid_file, started


def _assert_child_stopped(pid_file):
    from utils.process_lifetime import _pid_alive, _pid_is_zombie

    child = int(pid_file.read_text())
    for _ in range(50):
        if not _pid_alive(child) or _pid_is_zombie(child):
            return
        time.sleep(0.1)
    os.kill(child, 9)
    pytest.fail("the installer's child outlived the timed-out repair")


@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX process tree")
def test_a_timed_out_repair_stops_the_installers_children_and_records_it(monkeypatch, tmp_path):
    """Stop uv children before app imports and record the timeout to prevent repeated retries."""
    pid_file, started = _slow_installer(monkeypatch, tmp_path)
    recorded = []
    monkeypatch.setattr(dr, "_record_failure", lambda: recorded.append(True))
    lines = []
    assert dr._run_repair(lines.append) is False
    assert recorded == [True]
    assert "unsloth studio update" in lines[-1]
    assert started == ["--prefetch-diffusers-main"], "a slow fetch must never reach the install"
    _assert_child_stopped(pid_file)


@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX process tree")
def test_our_own_install_stopped_at_the_deadline_stops_startup(monkeypatch, tmp_path):
    """Stopped mid-install, packages may be half replaced: never import them, never record."""
    pid_file, started = _slow_installer(monkeypatch, tmp_path, timeout_s = 2)
    recorded = []
    monkeypatch.setattr(dr, "_record_failure", lambda: recorded.append(True))
    with pytest.raises(dr.InstallInterrupted, match = "Start Unsloth Studio again"):
        dr._run_repair(lambda _line: None, prefetch = False)
    assert started == ["--repair-diffusers-main"]
    assert recorded == [], "the next start has to retry, not skip, a half-finished install"
    _assert_child_stopped(pid_file)


@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX process tree")
def test_a_peer_still_in_the_pass_at_the_deadline_stops_startup_in_time(monkeypatch, tmp_path):
    """Abort at timeout if a peer is still installing, without changing its manifest."""
    pid_file, _started = _slow_installer(monkeypatch, tmp_path, timeout_s = 2)
    recorded = []
    monkeypatch.setattr(dr, "_record_failure", lambda: recorded.append(True))
    monkeypatch.setattr(dr, "_peer_holds_pass", lambda: True)
    started = time.monotonic()
    with pytest.raises(dr.PeerInstallInProgress, match = "Start Unsloth Studio again"):
        dr._run_repair(lambda _line: None, prefetch = False)
    assert time.monotonic() - started < 2 + 40
    assert recorded == []
    _assert_child_stopped(pid_file)


@pytest.mark.parametrize(
    "error, message",
    [
        (dr.PeerInstallInProgress, dr.PEER_INSTALL_MESSAGE),
        (dr.InstallInterrupted, dr.INTERRUPTED_MESSAGE),
    ],
)
def test_run_server_exits_with_the_message_when_the_environment_is_unsafe(
    monkeypatch, capsys, error, message
):
    import run

    def blocked(echo):
        raise error(message)

    monkeypatch.setattr(dr, "repair_diffusers_before_imports", blocked)
    with pytest.raises(SystemExit) as excinfo:
        run._repair_pinned_diffusers(silent = True)
    assert excinfo.value.code == 1
    assert message in capsys.readouterr().err, "shown even under --silent"


@pytest.mark.parametrize("loaded", ["huggingface_hub", "diffusers"])
def test_an_embedding_host_that_imported_the_packages_is_not_repaired_under_them(
    monkeypatch, loaded
):
    """A notebook kernel calling run_server() after importing them would mix old and new files."""
    _installed_diffusers(monkeypatch, None)
    monkeypatch.setattr(dr, "_loaded_replaceable_modules", _REAL_LOADED_REPLACEABLE)
    for name in ("huggingface_hub", "diffusers"):
        monkeypatch.delitem(sys.modules, name, raising = False)
    monkeypatch.setitem(sys.modules, loaded, object())
    monkeypatch.setattr(dr.subprocess, "Popen", lambda *a, **k: pytest.fail("started a repair"))
    lines = []
    assert dr.repair_diffusers_before_imports(lines.append) is False
    assert len(lines) == 1 and loaded in lines[0] and "unsloth studio update" in lines[0]


def test_any_other_repair_error_lets_startup_continue(monkeypatch, capsys):
    import run

    def broken(echo):
        raise OSError("disk full")

    monkeypatch.setattr(dr, "repair_diffusers_before_imports", broken)
    run._repair_pinned_diffusers(silent = False)
    assert "diffusers self-heal skipped: disk full" in capsys.readouterr().out


def test_a_recorded_failure_uses_the_key_the_installer_reads(monkeypatch):
    import types

    recorded = {}
    fake = types.SimpleNamespace(update_manifest = lambda **extra: recorded.update(extra))
    monkeypatch.setitem(sys.modules, "studio.install_manifest", fake)
    dr._record_failure()
    source = dr._INSTALLER.read_text(encoding = "utf-8")
    assert recorded == {"diffusers_main_repair": "failed"}
    assert '_DIFFUSERS_MAIN_REPAIR_KEY = "diffusers_main_repair"' in source


def _peer_pass(monkeypatch, *uncontended):
    import contextlib
    import types

    monkeypatch.setattr(dr, "_peer_holds_pass", _REAL_PEER_HOLDS_PASS)
    held = iter(uncontended)
    fake = types.SimpleNamespace(pass_lock = lambda: contextlib.nullcontext(next(held)))
    monkeypatch.setitem(sys.modules, "studio.install_manifest", fake)


def test_a_start_during_a_peers_pass_waits_it_out(monkeypatch):
    """A peer swapping diffusers can have removed its metadata, which reads as no index release."""
    _peer_pass(monkeypatch, False)
    _installed_diffusers(
        monkeypatch, {"url": "https://github.com/huggingface/diffusers", "vcs_info": {}}
    )
    started, lines = [], []
    monkeypatch.setattr(dr.subprocess, "Popen", lambda argv, **kw: started.append(argv) or _Proc(1))
    assert dr.repair_diffusers_before_imports(lines.append) is False
    assert started and started[0][-1] == "--repair-diffusers-main"
    assert lines == ["  - waiting for another Unsloth install or update to finish..."]


def _manifest(monkeypatch, manifest):
    import types

    monkeypatch.setattr(dr, "_installer_would_skip", _REAL_INSTALLER_WOULD_SKIP)
    fake = types.SimpleNamespace(read_manifest = lambda: manifest)
    monkeypatch.setitem(sys.modules, "studio.install_manifest", fake)


@pytest.mark.parametrize(
    "min_python, manifest",
    [
        ((99, 0), {}),  # Python 3.9 against the real (3, 10) floor
        ((3, 0), {"diffusers_main_repair": "failed"}),
        ((3, 0), {"step_results": {"diffusers-main.txt": "failed"}}),
    ],
)
def test_a_start_the_installer_would_skip_spawns_nothing(monkeypatch, min_python, manifest):
    """Every start would otherwise block on a no-op installer and claim it is installing."""
    _installed_diffusers(monkeypatch, None)
    _manifest(monkeypatch, manifest)
    monkeypatch.setattr(dr, "_MAIN_MIN_PYTHON", min_python)
    monkeypatch.setattr(dr.subprocess, "Popen", lambda *a, **k: pytest.fail("started a repair"))
    lines = []
    assert dr.repair_diffusers_before_imports(lines.append) is False
    assert lines == []


def test_a_skipped_start_still_waits_out_a_peers_pass(monkeypatch):
    _installed_diffusers(monkeypatch, None)
    _manifest(monkeypatch, {"diffusers_main_repair": "failed"})
    monkeypatch.setattr(dr, "_peer_holds_pass", lambda: True)
    started, lines = [], []
    monkeypatch.setattr(dr.subprocess, "Popen", lambda argv, **kw: started.append(argv) or _Proc(1))
    assert dr.repair_diffusers_before_imports(lines.append) is False
    assert started
    assert lines == ["  - waiting for another Unsloth install or update to finish..."]


def test_the_python_floor_matches_the_installer():
    source = dr._INSTALLER.read_text(encoding = "utf-8")
    assert f"DIFFUSERS_MAIN_MIN_PYTHON = {dr._MAIN_MIN_PYTHON!r}" in source


@pytest.mark.parametrize(
    "env",
    [{dr.DISABLE_ENV_VAR: "1"}, {"UNSLOTH_DIFFUSERS_MAIN": "0"}, {"UNSLOTH_DIFFUSERS_MAIN": "off"}],
)
def test_opt_outs_start_nothing(monkeypatch, env):
    _installed_diffusers(monkeypatch, None)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(dr.subprocess, "Popen", lambda *a, **k: pytest.fail("started a repair"))
    assert dr.repair_diffusers_before_imports() is False


def test_the_pinned_build_starts_nothing(monkeypatch):
    _installed_diffusers(
        monkeypatch, {"url": "https://github.com/huggingface/diffusers", "vcs_info": {}}
    )
    monkeypatch.setattr(dr.subprocess, "Popen", lambda *a, **k: pytest.fail("started a repair"))
    assert dr.repair_diffusers_before_imports() is False


def test_secrets_and_index_redirects_stay_out_of_the_installer_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("UV_INDEX_URL", "https://evil.example/simple")
    monkeypatch.setenv("UNSLOTH_DIFFUSERS_MAIN", "1")
    monkeypatch.setenv("UV_OFFLINE", "1")
    env = dr._repair_env()
    assert "HF_TOKEN" not in env and "UV_INDEX_URL" not in env
    assert env["UNSLOTH_DIFFUSERS_MAIN"] == "1"
    assert env["UV_OFFLINE"] == "1", "an offline install must not reach GitHub from the repair"


def test_run_server_repairs_before_it_imports_the_app():
    """Check import ordering in the AST to avoid starting the full backend."""
    import ast

    tree = ast.parse((_BACKEND / "run.py").read_text(encoding = "utf-8"))
    run_server = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_server"
    )
    repair = min(
        node.lineno
        for node in ast.walk(run_server)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "_repair_pinned_diffusers"
    )
    app_import = min(
        node.lineno
        for node in ast.walk(run_server)
        if isinstance(node, ast.ImportFrom) and node.module == "main"
    )
    assert repair < app_import
    main_source = (_BACKEND / "main.py").read_text(encoding = "utf-8")
    assert "diffusers_repair" not in main_source, "the app itself must never install under itself"


def test_the_repair_path_imports_nothing_it_can_replace():
    """Use a fresh interpreter to exclude imports made by the test suite."""
    code = (
        "import sys\n"
        "sys.argv = ['run.py']\n"
        "import run\n"
        "import utils.diffusers_repair as dr\n"
        "from utils.child_stdio import utf8_child_env\n"
        "from utils.process_lifetime import adopt_pid, child_popen_kwargs, forget_pid, terminate_pid\n"
        "dr._repair_env()\n"
        "dr._diffusers_is_an_index_install()\n"
        "names = ('huggingface_hub', 'diffusers', 'torch', 'transformers', 'safetensors', 'numpy')\n"
        "print('LOADED=' + ','.join(n for n in names if n in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd = _BACKEND,
        capture_output = True,
        text = True,
        timeout = 120,
    )
    assert result.returncode == 0, result.stderr
    assert "LOADED=\n" in result.stdout, result.stdout


def test_the_installer_exposes_the_repair_flag():
    source = dr._INSTALLER.read_text(encoding = "utf-8")
    assert '["--repair-diffusers-main"]' in source
    assert '["--prefetch-diffusers-main"]' in source
