# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""docker/build.sh must hand docker a COMMIT, not a branch name.

Docker matches a RUN layer on the command string alone -- "the files updated in the
container aren't examined to determine if a cache hit exists" -- so a build arg that
stays "main" makes the pip-install-from-git layer a cache hit on every rebuild and the
image keeps the commits of the first build while the build reports success. The publish
workflow already freezes both refs with git ls-remote; the local script has to as well.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_SH = REPO_ROOT / "docker" / "build.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason = "needs bash")

UNSLOTH_SHA = "a" * 40
ZOO_SHA = "b" * 40


def _stub(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\n" + body, encoding = "utf-8")
    path.chmod(0o755)


def _run(
    tmp_path: Path,
    git_body: str,
    env_extra: dict[str, str] | None = None,
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    args_file = tmp_path / "docker-args.txt"
    _stub(bin_dir / "git", git_body)
    _stub(bin_dir / "docker", f'printf "%s\\n" "$@" > {args_file}\n')
    # no network: the llama.cpp tag lookup must not reach github from a unit test
    _stub(bin_dir / "curl", "exit 1\n")

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env.update(env_extra or {})
    proc = subprocess.run(
        ["bash", str(BUILD_SH)],
        env = env,
        capture_output = True,
        text = True,
        cwd = str(tmp_path),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    argv = args_file.read_text(encoding = "utf-8").splitlines() if args_file.exists() else []
    return proc, argv


def _build_arg(argv: list[str], name: str) -> str:
    for index, item in enumerate(argv):
        if item == "--build-arg" and argv[index + 1].startswith(f"{name}="):
            return argv[index + 1].split("=", 1)[1]
    raise AssertionError(f"--build-arg {name} was never passed: {argv}")


LS_REMOTE_STUB = f"""
if [ "$1" = "ls-remote" ]; then
    case "$2" in
        *unsloth-zoo*) echo -e "{ZOO_SHA}\\tHEAD" ;;
        *) echo -e "{UNSLOTH_SHA}\\tHEAD" ;;
    esac
    exit 0
fi
exit 0
"""


def test_the_default_main_refs_are_frozen_to_commits(tmp_path):
    _proc, argv = _run(tmp_path, LS_REMOTE_STUB)
    assert _build_arg(argv, "UNSLOTH_REF") == UNSLOTH_SHA, (
        "a mutable 'main' build arg is byte-identical across rebuilds, so docker "
        "reuses the cached install layer and silently ships a stale image"
    )
    assert _build_arg(argv, "UNSLOTH_ZOO_REF") == ZOO_SHA


def test_an_explicit_tag_is_frozen_too(tmp_path):
    _proc, argv = _run(
        tmp_path, LS_REMOTE_STUB, {"UNSLOTH_REF": "v2026.5.6", "UNSLOTH_ZOO_REF": "v2026.5.4"}
    )
    assert _build_arg(argv, "UNSLOTH_REF") == UNSLOTH_SHA
    assert _build_arg(argv, "UNSLOTH_ZOO_REF") == ZOO_SHA


def test_a_sha_is_passed_through_without_a_lookup(tmp_path):
    sha = "c" * 40
    _proc, argv = _run(
        tmp_path,
        'if [ "$1" = "ls-remote" ]; then echo "ls-remote should not run" >&2; exit 3; fi\nexit 0\n',
        {"UNSLOTH_REF": sha, "UNSLOTH_ZOO_REF": sha},
    )
    assert _build_arg(argv, "UNSLOTH_REF") == sha
    assert _build_arg(argv, "UNSLOTH_ZOO_REF") == sha


def test_an_unreachable_remote_warns_and_still_builds(tmp_path):
    proc, argv = _run(tmp_path, 'if [ "$1" = "ls-remote" ]; then exit 128; fi\nexit 0\n')
    assert _build_arg(argv, "UNSLOTH_REF") == "main", "offline: the name is passed through"
    assert "unreachable" in proc.stdout + proc.stderr


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
