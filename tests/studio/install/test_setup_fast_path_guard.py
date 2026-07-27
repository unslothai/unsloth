# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""setup.sh / setup.ps1 must not skip the dependency pass on a half-built venv.

Both short-circuit all dependency work when the installed unsloth version equals
PyPI's latest, which is true on an interrupted install: unsloth goes in early and
studio.txt never finishes. So update, and the desktop Repair button behind it,
said "up to date" while the server kept dying on `import structlog`.

That branch only runs for a non-local update, which reinstalls from PyPI and
clobbers the tree under test, so assert the guard structurally instead.
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SETUP_SH = REPO_ROOT / "studio" / "setup.sh"
SETUP_PS1 = REPO_ROOT / "studio" / "setup.ps1"


@pytest.mark.parametrize("script", [SETUP_SH, SETUP_PS1], ids = ["setup.sh", "setup.ps1"])
def test_fast_path_consults_the_install_manifest(script: pathlib.Path):
    text = script.read_text(encoding = "utf-8")
    assert "install_manifest" in text, (
        f"{script.name} no longer consults studio/install_manifest.py. Without it "
        "the 'up to date' fast path skips the dependency pass on an interrupted "
        "install, and `unsloth studio update` becomes a silent no-op."
    )
    assert "verify_install" in text, (
        f"{script.name} must call install_manifest.verify_install() so the check "
        "matches what `unsloth studio verify-install` and the desktop preflight use."
    )


@pytest.mark.parametrize("script", [SETUP_SH, SETUP_PS1], ids = ["setup.sh", "setup.ps1"])
def test_guard_can_still_force_the_dependency_pass(script: pathlib.Path):
    """The guard has to clear the skip flag, not merely log a warning."""
    text = script.read_text(encoding = "utf-8")
    if script.name.endswith(".ps1"):
        pattern = r"studio install incomplete[\s\S]{0,200}?\$SkipPythonDeps\s*=\s*\$false"
    else:
        pattern = r"studio install incomplete[\s\S]{0,200}?_SKIP_PYTHON_DEPS=false"
    assert re.search(pattern, text), (
        f"{script.name} detects an incomplete install but does not clear the "
        "skip flag, so the dependency pass would still be skipped."
    )


def test_ps1_drops_the_manifest_before_its_first_install():
    """Nothing may mutate the venv while the marker still says "install finished".

    install_python_stack.py drops it before its own dependency pass, which is
    enough for setup.sh: the stack is the first thing that pass runs. setup.ps1
    replaces pip, torch and triton first, so a run killed there would leave a
    manifest that still verifies and a venv with half a PyTorch.
    """
    text = SETUP_PS1.read_text(encoding = "utf-8")
    pass_start = text.index("if (-not $SkipPythonDeps) {")
    removal = text.find("remove_manifest", pass_start)
    first_install = text.index("Fast-Install", pass_start)
    stack = text.index(r'python "$PSScriptRoot\install_python_stack.py"', pass_start)

    assert removal != -1, (
        "setup.ps1 never drops the install manifest; install_python_stack.py "
        "only does so after setup.ps1 has already replaced pip and torch"
    )
    assert removal < first_install < stack, (
        "setup.ps1 must invalidate the install manifest before its first "
        "Fast-Install, not leave it to install_python_stack.py"
    )


def test_sh_dependency_pass_mutates_nothing_before_the_stack():
    """setup.sh relies on install_python_stack.py dropping the marker, which only
    holds while the stack is the first thing its dependency pass runs."""
    text = SETUP_SH.read_text(encoding = "utf-8")
    pass_start = text.index('if [ "$_SKIP_PYTHON_DEPS" = false ]')
    body = text[pass_start : text.index("install_python_stack", pass_start)]
    assert "fast_install" not in body and "pip install" not in body, (
        "setup.sh installs something before install_python_stack.py drops the "
        "manifest, so an interrupted run would keep a marker that verifies"
    )


def test_sh_guard_runs_before_the_skip_decision():
    text = SETUP_SH.read_text(encoding = "utf-8")
    guard = text.find("studio install incomplete")
    decision = text.find('if [ "$_SKIP_PYTHON_DEPS" = false ]')
    assert guard != -1 and decision != -1
    assert guard < decision, (
        "the incomplete-install guard must run before setup.sh acts on "
        "_SKIP_PYTHON_DEPS, otherwise it can never change the outcome"
    )
