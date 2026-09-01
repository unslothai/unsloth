# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression coverage for the startup profiler's budget gate, teardown and triggers."""

from __future__ import annotations

import ast
import fnmatch
import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "profile_startup.py"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "startup-profile-ci.yml"
PROCESS_RS = REPO_ROOT / "studio" / "src-tauri" / "src" / "process.rs"

# Checkout files that build the venv the workflow profiles.
INSTALLER_INPUTS = (
    "studio/setup.sh",
    "studio/setup.ps1",
    "studio/install_python_stack.py",
)
# Checkout file that defines the argv the profiler reproduces.
LAUNCH_INPUTS = ("studio/src-tauri/src/process.rs",)


def _load():
    spec = importlib.util.spec_from_file_location("profile_startup", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _no_subprocesses(mod, monkeypatch):
    # Keep the gate tests off the real interpreter and CLI.
    monkeypatch.setattr(mod, "find_bin", lambda: None)
    monkeypatch.setattr(mod, "profile_imports", lambda python, top = 15: {"ok": False, "error": ""})
    monkeypatch.setattr(mod, "python_version_of", lambda python: "3.13.0")


class _Proc:
    """Stand-in for a still-running Popen."""

    def __init__(self):
        self.pid = 4321
        self.terminated = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True


def _nt(mod, monkeypatch, returncode):
    calls: list[list[str]] = []

    def _run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, returncode, "", "")

    # Patch the module's own references, not the real os/subprocess the session shares.
    monkeypatch.setattr(mod, "os", SimpleNamespace(name = "nt"))
    monkeypatch.setattr(mod, "subprocess", SimpleNamespace(run = _run))
    return calls


def test_budget_fails_when_no_launch_was_measured(capsys, monkeypatch):
    """A requested budget must not pass just because the CLI was never found."""
    mod = _load()
    _no_subprocesses(mod, monkeypatch)
    rc = mod.main(["--max-healthz-seconds", "30"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "::error::" in out and "no healthz measurement" in out
    assert "no unsloth CLI found" in out


def _healthy_launch(
    mod,
    monkeypatch,
    healthz = 1.5,
):
    monkeypatch.setattr(mod, "find_bin", lambda: "unsloth")
    monkeypatch.setattr(
        mod,
        "profile_launch",
        lambda bin_path, port, **kw: {
            "spawn_seconds": 0.1,
            "healthz_seconds": healthz,
            "lifespan_ms": 100.0,
            "reached_healthz": True,
            "log_tail": [],
        },
    )


def test_budget_still_passes_when_a_launch_was_measured(monkeypatch):
    """The fail-closed branch must not swallow a genuinely healthy run."""
    mod = _load()
    _no_subprocesses(mod, monkeypatch)
    _healthy_launch(mod, monkeypatch)
    assert mod.main(["--max-healthz-seconds", "30"]) == 0
    assert mod.main(["--max-healthz-seconds", "1"]) == 1


# "=" form for -inf: a bare "-inf" is an option token to argparse, not a value.
@pytest.mark.parametrize(
    "bad", ["--max-healthz-seconds=nan", "--max-healthz-seconds=inf", "--max-healthz-seconds=-inf"]
)
def test_budget_rejects_non_finite_values(bad, capsys, monkeypatch):
    """`med > nan` and `med > inf` are always False, so the gate would never bind."""
    mod = _load()
    _no_subprocesses(mod, monkeypatch)
    _healthy_launch(mod, monkeypatch)
    with pytest.raises(SystemExit) as exc:
        mod.main([bad])
    assert exc.value.code == 2
    assert "finite" in capsys.readouterr().err


def test_budget_rejects_import_only(capsys):
    """--import-only launches nothing, so a budget on it could only ever pass."""
    mod = _load()
    with pytest.raises(SystemExit) as exc:
        mod.main(["--import-only", "--max-healthz-seconds", "30"])
    assert exc.value.code == 2
    assert "--import-only" in capsys.readouterr().err


def test_terminate_tree_falls_back_when_taskkill_fails(monkeypatch):
    """A nonzero taskkill must still reach terminate(), not return silently."""
    mod = _load()
    calls = _nt(mod, monkeypatch, returncode = 1)
    proc = _Proc()
    mod._terminate_tree(proc)
    assert calls == [["taskkill", "/PID", "4321", "/T", "/F"]]
    assert proc.terminated


def test_terminate_tree_falls_back_when_taskkill_raises(monkeypatch):
    """A missing or hung taskkill must reach terminate() too."""
    mod = _load()
    monkeypatch.setattr(mod, "os", SimpleNamespace(name = "nt"))

    def _boom(argv, **kwargs):
        raise FileNotFoundError(argv)

    monkeypatch.setattr(mod, "subprocess", SimpleNamespace(run = _boom))
    proc = _Proc()
    mod._terminate_tree(proc)
    assert proc.terminated


def test_terminate_tree_returns_on_successful_taskkill(monkeypatch):
    mod = _load()
    _nt(mod, monkeypatch, returncode = 0)
    proc = _Proc()
    mod._terminate_tree(proc)
    assert not proc.terminated


def test_terminate_tree_skips_an_exited_process(monkeypatch):
    mod = _load()
    calls = _nt(mod, monkeypatch, returncode = 0)
    proc = _Proc()
    proc.poll = lambda: 0
    mod._terminate_tree(proc)
    assert calls == [] and not proc.terminated


def _trigger_paths():
    wf = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    # YAML 1.1 turns the bare `on:` key into True.
    on = wf.get("on") or wf[True]
    return [p for p in on["pull_request"]["paths"] if not p.startswith("!")]


@pytest.mark.parametrize("rel", INSTALLER_INPUTS)
def test_workflow_triggers_on_studio_installer_inputs(rel):
    """A setup script that changes the profiled venv must schedule a measurement."""
    assert (REPO_ROOT / rel).is_file(), f"{rel} moved; revisit the trigger list"
    paths = _trigger_paths()
    assert any(fnmatch.fnmatch(rel, p) for p in paths), f"{rel} not covered by {paths}"


def test_studio_installer_inputs_are_on_the_local_install_path():
    """Anchor the list above: these files are what --local actually executes."""
    # install.ps1 reaches setup.ps1 through the editable install, not by name.
    assert "studio/setup.sh" in (REPO_ROOT / "install.sh").read_text(encoding = "utf-8")
    for setup in ("studio/setup.sh", "studio/setup.ps1"):
        text = (REPO_ROOT / setup).read_text(encoding = "utf-8", errors = "replace")
        assert "install_python_stack.py" in text


@pytest.mark.parametrize("rel", LAUNCH_INPUTS)
def test_workflow_triggers_on_the_desktop_launch_command(rel):
    """The profiler copies process.rs's argv, so a change there must be measured."""
    assert (REPO_ROOT / rel).is_file(), f"{rel} moved; revisit the trigger list"
    paths = _trigger_paths()
    assert any(fnmatch.fnmatch(rel, p) for p in paths), f"{rel} not covered by {paths}"


def _desktop_backend_argv():
    body = re.search(
        r"fn backend_args\(port: u16\) -> Vec<String> \{(.*?)\n\}",
        PROCESS_RS.read_text(encoding = "utf-8"),
        re.S,
    )
    assert body, "backend_args moved; revisit the trigger list"
    return re.findall(r'"([^"]+)"', body.group(1))


def _profiler_argv():
    tree = ast.parse(SCRIPT.read_text(encoding = "utf-8"))
    fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "profile_launch"
    )
    call = next(
        n for n in ast.walk(fn) if isinstance(n, ast.Call) and ast.unparse(n.func).endswith("Popen")
    )
    return [e.value for e in call.args[0].elts if isinstance(e, ast.Constant)]


def test_profiler_spawns_the_desktop_backend_argv():
    """Anchor the trigger above: these two argv lists must stay identical."""
    assert _profiler_argv() == _desktop_backend_argv()


@pytest.mark.skipif(sys.platform == "win32", reason = "posix branch")
def test_terminate_tree_posix_uses_terminate():
    mod = _load()
    proc = _Proc()
    mod._terminate_tree(proc)
    assert proc.terminated


# The budget is only a gate if the workflow asks for it and can see it fail
def _profile_job() -> dict:
    wf = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    return wf["jobs"]["profile"]


def _profile_step() -> dict:
    for step in _profile_job()["steps"]:
        if step.get("name") == "Profile startup":
            return step
    raise AssertionError("no 'Profile startup' step in the workflow")


def test_every_platform_carries_a_budget():
    """A matrix entry without one would profile and assert nothing."""
    entries = _profile_job()["strategy"]["matrix"]["include"]
    assert entries, "the matrix no longer lists platforms by include"
    for entry in entries:
        budget = entry.get("max_healthz_seconds")
        assert budget, f"{entry.get('os')} has no max_healthz_seconds"
        assert float(budget) > 0


def test_the_profile_step_passes_the_budget():
    assert "--max-healthz-seconds" in _profile_step()["run"]


def test_the_profile_step_sets_pipefail():
    """`shell: bash` already implies -o pipefail, so this is belt and braces: the
    gate's exit code only reaches the step through the pipe into tee, and it has to
    survive that shell key being dropped or changed to `bash {0}`."""
    run = _profile_step()["run"]
    assert "| tee" in run, "no pipe left; this guard can go"
    assert "set -o pipefail" in run


def test_the_job_is_not_advisory():
    """continue-on-error would let the gate fail without failing the check."""
    assert _profile_job().get("continue-on-error") is not True
