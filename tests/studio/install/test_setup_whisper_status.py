# SPDX-License-Identifier: AGPL-3.0-only
"""Setup no longer installs or updates whisper.cpp; audio.cpp and uninstall cleanup stay."""

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SETUP_SCRIPTS = ("setup.sh", "setup.ps1")


def _setup(name: str) -> str:
    return (ROOT / "studio" / name).read_text(encoding = "utf-8")


@pytest.mark.parametrize("name", SETUP_SCRIPTS)
def test_setup_never_runs_a_whisper_installer(name):
    script = _setup(name)
    assert "install_whisper_prebuilt.py" not in script
    assert "build_whisper_cpp.sh" not in script
    assert not re.search(r'^\s*step "whisper\.cpp" ', script, re.M)


@pytest.mark.parametrize("name", SETUP_SCRIPTS)
def test_setup_still_installs_audio_cpp(name):
    script = _setup(name)
    assert "install_audio_cpp_prebuilt.py" in script
    assert re.search(r'^\s*step "audio\.cpp" "prebuilt installed"', script, re.M)


@pytest.mark.parametrize("name", SETUP_SCRIPTS)
def test_setup_leaves_an_existing_whisper_tree_alone(name):
    script = _setup(name)
    assert "whisper.cpp: retired; Unsloth Studio no longer installs or updates it" in script


@pytest.mark.parametrize("name", ("uninstall.sh", "uninstall.ps1"))
def test_uninstall_still_removes_whisper_cpp(name):
    script = (ROOT / "scripts" / name).read_text(encoding = "utf-8")
    assert re.search(r"whisper\\?\.cpp", script)
