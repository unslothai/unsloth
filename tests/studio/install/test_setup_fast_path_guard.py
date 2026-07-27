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


def test_sh_guard_runs_before_the_skip_decision():
    text = SETUP_SH.read_text(encoding = "utf-8")
    guard = text.find("studio install incomplete")
    decision = text.find('if [ "$_SKIP_PYTHON_DEPS" = false ]')
    assert guard != -1 and decision != -1
    assert guard < decision, (
        "the incomplete-install guard must run before setup.sh acts on "
        "_SKIP_PYTHON_DEPS, otherwise it can never change the outcome"
    )
