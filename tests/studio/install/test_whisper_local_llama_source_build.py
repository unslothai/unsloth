# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""#9179: beside a --with-llama-cpp-dir tree no slim whisper bundle can pair, so setup.sh
falls back to the static whisper.cpp source build instead of leaving dictation unavailable."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SETUP_SH = Path(__file__).resolve().parents[3] / "studio" / "setup.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason = "setup.sh needs bash")


def _whisper_block() -> str:
    text = SETUP_SH.read_text(encoding = "utf-8")
    start = text.index('WHISPER_CPP_DIR="$UNSLOTH_HOME/whisper.cpp"')
    end = text.index("# ── audio.cpp", start)
    return text[start:end]


def _run(tmp_path: Path, *, linked: bool, force_compile: bool) -> tuple[str, bool]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # Installer exit 2: no slim bundle pairs with an unmanaged llama tree.
    for name, body in (("python", "exit 2"), ("cmake", "exit 0"), ("git", "exit 0")):
        stub = bin_dir / name
        stub.write_text(f"#!/bin/sh\n{body}\n", encoding = "utf-8")
        stub.chmod(0o755)
    (tmp_path / "studio").mkdir()
    (tmp_path / "scripts").mkdir()
    built = tmp_path / "built"
    (tmp_path / "scripts" / "build_whisper_cpp.sh").write_text(f": > '{built}'\n", encoding = "utf-8")
    script = f"""
set -e
step() {{ echo "STEP $1: $2"; }}
substep() {{ echo "SUB $1"; }}
verbose_substep() {{ :; }}
_is_verbose() {{ return 1; }}
_filter_download_output() {{ cat; }}
_assert_studio_owned_or_absent() {{ :; }}
run_quiet_no_exit() {{ shift; "$@"; }}
SCRIPT_DIR='{tmp_path / "studio"}'
UNSLOTH_HOME='{tmp_path / "home"}'
_RUNTIME_ROOT_IS_CUSTOM=false
_LOCAL_LLAMA_CPP_LINKED={"true" if linked else "false"}
{_whisper_block()}
"""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("WHISPER", "UNSLOTH_"))}
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    if force_compile:
        env["UNSLOTH_WHISPER_FORCE_COMPILE"] = "1"
    result = subprocess.run(
        ["bash", "-c", script], env = env, capture_output = True, text = True, timeout = 60
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout, built.exists()


def test_linked_local_llama_falls_back_to_source_build(tmp_path):
    out, built = _run(tmp_path, linked = True, force_compile = False)
    assert built, out
    assert "local llama.cpp build linked" in out
    assert "STEP whisper.cpp: source build installed" in out


def test_managed_llama_keeps_source_build_opt_in(tmp_path):
    out, built = _run(tmp_path, linked = False, force_compile = False)
    assert not built, out
    assert "curated whisper.cpp dictation is unavailable" in out


def test_force_compile_still_builds(tmp_path):
    out, built = _run(tmp_path, linked = False, force_compile = True)
    assert built, out
    assert "UNSLOTH_WHISPER_FORCE_COMPILE=1" in out
