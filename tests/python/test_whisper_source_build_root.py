# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_whisper_cpp.sh"


@pytest.mark.skipif(os.name == "nt", reason = "the source builder is POSIX-only")
def test_source_build_uses_the_staged_managed_root(tmp_path):
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
