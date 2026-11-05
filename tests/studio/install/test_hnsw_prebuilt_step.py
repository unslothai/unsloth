# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""install_python_stack step 11e: the optional HNSW pair never fails an install, and a later
runtime install of it never voids the known-unmet evidence of the last dependency pass."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest
from unsloth_pwsh_runner import run_pwsh

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
STACK_PATH = REPO_ROOT / "studio" / "install_python_stack.py"


def _load():
    sys.path.insert(0, str(REPO_ROOT / "studio"))
    try:
        spec = importlib.util.spec_from_file_location("studio_stack_hnsw_step", STACK_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        if sys.path and sys.path[0] == str(REPO_ROOT / "studio"):
            sys.path.pop(0)
    return module


stack = _load()


@pytest.fixture
def step(monkeypatch, tmp_path):
    monkeypatch.delenv("RAG_VEC_ANN_AUTO_INSTALL", raising = False)
    monkeypatch.setattr(stack, "_progress", lambda label: progress.append(label))
    monkeypatch.setattr(stack, "_note", lambda message, color_fn = None: notes.append(message))
    stack._STEP_RESULTS.clear()
    progress, notes = [], []

    def run(installer_source):
        installer = tmp_path / "install_hnsw_prebuilt.py"
        installer.write_text(installer_source, encoding = "utf-8")
        monkeypatch.setattr(stack, "HNSW_INSTALLER", installer)
        stack._hnsw_prebuilt_step()
        return progress, notes, stack._STEP_RESULTS.get("hnsw-prebuilt")

    return run


def test_a_ready_pair_is_recorded_without_a_note(step):
    progress, notes, result = step(
        "import sys\n"
        "def ensure(python):\n"
        "    assert python == sys.executable\n"
        "    return {'status': 'ready', 'detail': ''}\n"
    )
    assert progress == ["HNSW search"] and notes == [] and result == "ready"


@pytest.mark.parametrize(
    "source",
    [
        "def ensure(python):\n    return {'status': 'unsupported', 'detail': 'glibc 2.27'}\n",
        "def ensure(python):\n    raise RuntimeError('network down')\n",
        "import sys\ndef ensure(python):\n    sys.exit(3)\n",
        "raise ImportError('broken installer')\n",
    ],
)
def test_any_failure_is_a_note_never_an_install_failure(step, source):
    progress, notes, result = step(source)
    assert progress == ["HNSW search"]
    assert result in {"unsupported", "failed"}
    assert len(notes) == 1 and "exact search" in notes[0]


def test_a_missing_installer_is_a_note(step, monkeypatch, tmp_path):
    monkeypatch.setattr(stack, "HNSW_INSTALLER", tmp_path / "absent.py")
    stack._hnsw_prebuilt_step()
    assert stack._STEP_RESULTS["hnsw-prebuilt"] == "failed"


def test_the_opt_out_loads_nothing(step, monkeypatch):
    monkeypatch.setenv("RAG_VEC_ANN_AUTO_INSTALL", "0")
    progress, notes, result = step("raise AssertionError('loaded despite the opt-out')\n")
    assert progress == ["HNSW search (disabled)"] and notes == [] and result == "disabled"


def test_the_step_runs_after_the_studio_deps_on_every_path():
    source = STACK_PATH.read_text(encoding = "utf-8")
    body = source[source.index("def install_python_stack(") :]
    assert body.index("_hnsw_prebuilt_step()") > body.index('REQ_ROOT / "studio.txt"')
    # Not inside an if: the slot is counted unconditionally (base_total += 1).
    line = next(l for l in body.splitlines() if "_hnsw_prebuilt_step()" in l)
    assert line == "    _hnsw_prebuilt_step()"


def _index(**versions):
    return {name: (version, []) for name, version in versions.items()}


def test_installing_the_hnsw_pair_keeps_the_known_unmet_evidence(monkeypatch):
    before = _index(numpy = "2.3.0", click = "8.3.0")
    after = _index(numpy = "2.3.0", click = "8.3.0", usearch = "2.26.6", numkong = "7.8.6")
    monkeypatch.setattr(stack, "_installed_index", lambda: before)
    digest = stack._installed_index_digest()
    monkeypatch.setattr(stack, "_installed_index", lambda: after)
    assert stack._installed_index_digest() == digest


def test_any_other_change_still_voids_it(monkeypatch):
    monkeypatch.setattr(stack, "_installed_index", lambda: _index(numpy = "2.3.0", click = "8.3.0"))
    digest = stack._installed_index_digest()
    monkeypatch.setattr(stack, "_installed_index", lambda: _index(numpy = "2.3.0", click = "8.3.1"))
    assert stack._installed_index_digest() != digest


# setup.sh / setup.ps1 fast path: the Python stack is current and skipped, so only the HNSW check runs.

STUB = """import json, sys
open(sys.argv[0] + ".calls", "a").write(" ".join(sys.argv[1:]) + "\\n")
check = "--check-only" in sys.argv
print("noise")
print(json.dumps({"status": CHECK if check else AFTER, "detail": ""}))
"""


def _stub(tmp_path, check, after):
    installer = tmp_path / "install_hnsw_prebuilt.py"
    installer.write_text(
        STUB.replace("CHECK", repr(check)).replace("AFTER", repr(after)), encoding = "utf-8"
    )
    return installer


def _calls(installer):
    path = pathlib.Path(str(installer) + ".calls")
    return path.read_text(encoding = "utf-8").splitlines() if path.exists() else []


def _run_sh(tmp_path, env = None):
    source = (REPO_ROOT / "studio" / "setup.sh").read_text(encoding = "utf-8")
    func = re.search(r"^_hnsw_backfill\(\) \{\n.*?^\}\n", source, re.M | re.S).group(0)
    script = (
        f'SCRIPT_DIR="{tmp_path}"; C_WARN=W; C_OK=G\n'
        f'python() {{ "{sys.executable}" "$@"; }}\n'
        'step() { printf "STEP|%s|%s|%s\\n" "$1" "$2" "${3:-}"; }\n'
        f"{func}_hnsw_backfill\n"
    )
    full_env = {k: v for k, v in os.environ.items() if k != "RAG_VEC_ANN_AUTO_INSTALL"}
    full_env.update(env or {})
    return subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script], capture_output = True, text = True, env = full_env
    )


# setup.sh never runs on native Windows, where `bash` is the WSL stub and runs nothing.
_needs_bash = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None, reason = "setup.sh runs on POSIX only"
)


@_needs_bash
@pytest.mark.parametrize("check", ["already", "disabled", "unsupported", "offline"])
def test_setup_sh_stays_local_when_nothing_is_owed(tmp_path, check):
    installer = _stub(tmp_path, check, "ready")
    out = _run_sh(tmp_path)
    assert out.returncode == 0, out.stderr
    assert _calls(installer) == ["--check-only --json"] and "STEP" not in out.stdout


@_needs_bash
@pytest.mark.parametrize("after,colour", [("ready", ""), ("failed", "W")])
def test_setup_sh_backfills_a_missing_pair_and_never_fails(tmp_path, after, colour):
    installer = _stub(tmp_path, "missing", after)
    out = _run_sh(tmp_path)
    assert out.returncode == 0, out.stderr
    assert _calls(installer) == ["--check-only --json", "--json --timeout 300"]
    assert f"STEP|HNSW search|{after}|{colour}" in out.stdout


@_needs_bash
def test_setup_sh_opt_out_and_missing_installer_run_nothing(tmp_path):
    installer = _stub(tmp_path, "missing", "ready")
    assert _run_sh(tmp_path, {"RAG_VEC_ANN_AUTO_INSTALL": "0"}).returncode == 0
    assert _calls(installer) == []
    installer.unlink()
    assert _run_sh(tmp_path).returncode == 0


def test_setup_sh_calls_it_only_on_the_skipped_branch():
    source = (REPO_ROOT / "studio" / "setup.sh").read_text(encoding = "utf-8")
    branch = source[source.index('step "python" "dependencies up to date"') :]
    assert branch.index("_hnsw_backfill || true") < branch.index("\nfi\n")


def _run_ps1(tmp_path, env = None):
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("pwsh not installed")
    source = (REPO_ROOT / "studio" / "setup.ps1").read_text(encoding = "utf-8")
    func = re.search(r"^function Invoke-HnswBackfill \{\n.*?^\}\n", source, re.M | re.S).group(0)
    func = func.replace("$PSScriptRoot", f"'{tmp_path}'")
    script = (
        f"function python {{ & '{sys.executable}' @args }}\n"
        'function step { param($Label, $Value, $Color = "Green") Write-Output "STEP|$Label|$Value|$Color" }\n'
        f"{func}Invoke-HnswBackfill\n"
    )
    full_env = {k: v for k, v in os.environ.items() if k != "RAG_VEC_ANN_AUTO_INSTALL"}
    full_env.update(env or {})
    return run_pwsh(
        [pwsh, "-NoProfile", "-Command", script], capture_output = True, text = True, env = full_env
    )


@pytest.mark.parametrize(
    "check,after,expected_calls,step_line",
    [
        ("already", "ready", ["--check-only --json"], None),
        (
            "missing",
            "ready",
            ["--check-only --json", "--json --timeout 300"],
            "STEP|HNSW search|ready|Green",
        ),
        (
            "missing",
            "failed",
            ["--check-only --json", "--json --timeout 300"],
            "STEP|HNSW search|failed|Yellow",
        ),
    ],
)
def test_setup_ps1_backfill_matches_setup_sh(tmp_path, check, after, expected_calls, step_line):
    installer = _stub(tmp_path, check, after)
    out = _run_ps1(tmp_path)
    assert out.returncode == 0, out.stderr
    assert _calls(installer) == expected_calls
    if step_line:
        assert step_line in out.stdout
    else:
        assert "STEP" not in out.stdout


def test_setup_ps1_opt_out_runs_nothing(tmp_path):
    installer = _stub(tmp_path, "missing", "ready")
    assert _run_ps1(tmp_path, {"RAG_VEC_ANN_AUTO_INSTALL": "0"}).returncode == 0
    assert _calls(installer) == []
