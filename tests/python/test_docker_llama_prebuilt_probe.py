# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""The llama-server sanity probe must not accept the loader's failure message.

The probe asserts on the substring "version" in the binary's output, which is exactly
the word the dynamic loader uses when it refuses to start one:

    ./llama-server: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.38' not
    found (required by ./llama-server)

The program never reaches main and exits nonzero, so the exit code is what separates
the two cases. Driven end to end against the real probe with stub binaries on disk.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import stat
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FETCH = REPO_ROOT / "docker" / "fetch_llama_prebuilt.py"

GLIBC_FAILURE = (
    "./llama-server: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.38' "
    "not found (required by ./llama-server)"
)

behavioural = pytest.mark.skipif(
    shutil.which("bash") is None, reason = "needs bash for the stub binaries"
)


@pytest.fixture()
def fetch():
    assert FETCH.is_file(), f"missing {FETCH}"
    spec = importlib.util.spec_from_file_location("fetch_llama_under_test", FETCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stub(path: Path, output: str, rc: int) -> None:
    # quoted heredoc: the message has a backtick an `echo` would run as a substitution
    path.write_text(
        "#!/usr/bin/env bash\n"
        "cat >&2 <<'UNSLOTH_EOF'\n"
        f"{output}\n"
        "UNSLOTH_EOF\n"
        f"exit {rc}\n",
        encoding = "utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _stubs(install_dir: Path, *, server_output: str, server_rc: int) -> Path:
    build_bin = install_dir / "build" / "bin"
    build_bin.mkdir(parents = True, exist_ok = True)
    _stub(install_dir / "llama-server", server_output, server_rc)
    _stub(install_dir / "llama-quantize", "usage: llama-quantize [options]", 1)
    _stub(build_bin / "llama-quantize", "usage: llama-quantize [options]", 1)
    return build_bin


@behavioural
def test_a_loader_failure_is_not_accepted_as_a_version_banner(fetch, tmp_path):
    build_bin = _stubs(tmp_path, server_output = GLIBC_FAILURE, server_rc = 1)
    with pytest.raises(SystemExit) as exc:
        fetch.sanity_check_binaries(str(tmp_path), str(build_bin))
    assert "llama-server" in str(exc.value)
    assert "exited 1" in str(exc.value)


@behavioural
def test_a_healthy_server_passes(fetch, tmp_path):
    build_bin = _stubs(tmp_path, server_output = "version: 4589 (b9a9e6d)", server_rc = 0)
    fetch.sanity_check_binaries(str(tmp_path), str(build_bin))


@behavioural
def test_the_quantize_probes_are_not_required_to_exit_zero(fetch, tmp_path):
    build_bin = _stubs(tmp_path, server_output = "version: 4589 (b9a9e6d)", server_rc = 0)
    _stub(tmp_path / "llama-quantize", "usage: llama-quantize [options]", 7)
    fetch.sanity_check_binaries(str(tmp_path), str(build_bin))


@behavioural
def test_a_server_with_no_banner_at_all_still_fails(fetch, tmp_path):
    build_bin = _stubs(tmp_path, server_output = "", server_rc = 0)
    with pytest.raises(SystemExit) as exc:
        fetch.sanity_check_binaries(str(tmp_path), str(build_bin))
    assert "did not print 'version'" in str(exc.value)


def test_the_loader_message_really_does_contain_the_substring():
    # the premise: without it the exit-code check above is only belt and braces
    assert "version" in GLIBC_FAILURE
