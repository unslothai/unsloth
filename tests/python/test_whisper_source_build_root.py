# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_whisper_cpp.sh"
SETUP_SH = REPO_ROOT / "studio" / "setup.sh"


def _setup_sh_build_env(master_root: Path) -> dict[str, str]:
    """The VAR=VALUE prefix setup.sh puts on the source build, read from setup.sh.

    Hard-coding it would keep passing after a refactor moved the blanking that makes
    UNSLOTH_HOME win over the inherited UNSLOTH_STUDIO_HOME.
    """
    match = re.search(
        r'env (UNSLOTH_HOME=[^\n]*?)\s*\\?\s*sh "\$_WHISPER_BUILD"',
        SETUP_SH.read_text(encoding = "utf-8"),
    )
    assert match, "could not find the whisper.cpp source build invocation in setup.sh"
    overrides = {}
    for token in shlex.split(match.group(1).replace("$UNSLOTH_HOME", str(master_root))):
        name, _, value = token.partition("=")
        overrides[name] = value
    assert overrides["UNSLOTH_HOME"] == str(master_root)
    return overrides


def _stub_tools(tmp_path: Path) -> Path:
    tool_dir = tmp_path / "bin"
    tool_dir.mkdir()
    git = tool_dir / "git"
    git.write_text(
        "#!/bin/sh\n"
        "for argument do destination=$argument; done\n"
        'mkdir -p "$destination/.git"\n',
        encoding = "utf-8",
    )
    cmake = tool_dir / "cmake"
    cmake.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-S" ]; then\n'
        "  while [ $# -gt 0 ]; do\n"
        '    if [ "$1" = "-B" ]; then shift; build=$1; break; fi\n'
        "    shift\n"
        "  done\n"
        '  mkdir -p "$build/bin"\n'
        "  printf '#!/bin/sh\\nexit 0\\n' > \"$build/bin/whisper-server\"\n"
        '  chmod +x "$build/bin/whisper-server"\n'
        "fi\n",
        encoding = "utf-8",
    )
    git.chmod(0o755)
    cmake.chmod(0o755)
    return tool_dir


@pytest.mark.skipif(os.name == "nt", reason = "the source builder is POSIX-only")
def test_source_build_uses_the_staged_managed_root(tmp_path):
    tool_dir = _stub_tools(tmp_path)

    live_home = tmp_path / "live-home"
    stage_root = tmp_path / "stage"
    env = {
        **os.environ,
        "HOME": str(live_home),
        "PATH": f"{tool_dir}{os.pathsep}{os.environ['PATH']}",
        "UNSLOTH_HOME": str(stage_root),
    }
    env.pop("UNSLOTH_STUDIO_HOME", None)
    env.pop("STUDIO_HOME", None)

    subprocess.run(["sh", str(BUILD_SCRIPT)], env = env, check = True, capture_output = True)

    assert (stage_root / "whisper.cpp" / "build" / "bin" / "whisper-server").is_file()
    assert not (live_home / ".unsloth" / "whisper.cpp").exists()


@pytest.mark.skipif(os.name == "nt", reason = "the source builder is POSIX-only")
def test_source_build_lands_in_the_portable_master_root(tmp_path):
    """A nested portable install runs setup.sh with UNSLOTH_STUDIO_HOME=<root>/studio.

    build_whisper_cpp.sh resolves that before UNSLOTH_HOME, so the build used to land at
    <root>/studio/whisper.cpp while stt_ggml_sidecar searched <root>/whisper.cpp: setup
    reported success and dictation stayed unavailable.
    """
    tool_dir = _stub_tools(tmp_path)

    live_home = tmp_path / "live-home"
    master_root = tmp_path / "root"
    studio_root = master_root / "studio"
    studio_root.mkdir(parents = True)

    env = {
        **os.environ,
        "HOME": str(live_home),
        "PATH": f"{tool_dir}{os.pathsep}{os.environ['PATH']}",
        "UNSLOTH_STUDIO_HOME": str(studio_root),
        **_setup_sh_build_env(master_root),
    }

    subprocess.run(["sh", str(BUILD_SCRIPT)], env = env, check = True, capture_output = True)

    assert (master_root / "whisper.cpp" / "build" / "bin" / "whisper-server").is_file()
    assert not (studio_root / "whisper.cpp").exists()
    assert not (live_home / ".unsloth" / "whisper.cpp").exists()
