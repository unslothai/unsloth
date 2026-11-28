# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Settings > Security file size limit: env first, then the owner's saved value, then 4 GB; a call that runs
into it is told so, because ulimit cannot raise the hard limit from inside the sandbox."""

from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import tools
from storage import studio_db
from utils import sandbox_file_size_limit as limit
from utils.account_context import AccountContext, bind_account, reset_account

ALICE = AccountContext("a" * 32, "alice")
MB = 1024 * 1024
_SIGXFSZ = getattr(signal, "SIGXFSZ", 25)
_POSIX = pytest.mark.skipif(not hasattr(signal, "SIGXFSZ"), reason = "RLIMIT_FSIZE is POSIX-only")


@pytest.fixture(autouse = True)
def isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    monkeypatch.delenv(limit.FILE_SIZE_LIMIT_ENV, raising = False)


def test_default_lets_builds_link_past_the_old_fixed_100_mb():
    assert limit.saved_file_size_limit_mb() == 4096
    assert limit.file_size_limit_bytes() == 4096 * MB
    tools._refresh_sandbox_memory_limit()
    assert tools._sandbox_fsize_bytes == 4096 * MB


def test_a_saved_value_applies_to_the_next_launch():
    for _ in range(2):
        assert limit.set_file_size_limit_mb(100) == 100
    assert limit.file_size_limit_bytes() == 100 * MB
    tools._refresh_sandbox_memory_limit()
    assert tools._sandbox_fsize_bytes == 100 * MB


def test_the_value_is_installation_wide():
    limit.set_file_size_limit_mb(2048)
    token = bind_account(ALICE)
    try:
        assert limit.saved_file_size_limit_mb() == 2048
    finally:
        reset_account(token)


@pytest.mark.parametrize("value", [0, -1, limit.MAX_FILE_SIZE_LIMIT_MB + 1, "big", None, True, 1.5])
def test_a_bad_value_is_refused_and_keeps_the_old_one(value):
    limit.set_file_size_limit_mb(500)
    with pytest.raises(ValueError):
        limit.set_file_size_limit_mb(value)
    assert limit.saved_file_size_limit_mb() == 500


def test_the_environment_wins_and_a_bad_one_keeps_the_default(monkeypatch):
    limit.set_file_size_limit_mb(500)
    monkeypatch.setenv(limit.FILE_SIZE_LIMIT_ENV, "8192")
    assert limit.locked_by_environment()
    assert limit.effective_file_size_limit_mb() == 8192
    assert limit.saved_file_size_limit_mb() == 500
    monkeypatch.setenv(limit.FILE_SIZE_LIMIT_ENV, "lots")
    assert limit.effective_file_size_limit_mb() == limit.DEFAULT_FILE_SIZE_LIMIT_MB


def test_the_linux_prlimit_flags_carry_the_saved_limit(monkeypatch):
    if tools._resource is None:
        pytest.skip("no resource module")
    limit.set_file_size_limit_mb(3000)
    tools._refresh_sandbox_memory_limit()
    args = tools._sandbox_rlimit_args()
    if args is None:
        pytest.skip("this host keeps the pre-exec")
    assert f"--fsize={3000 * MB}:{3000 * MB}" in args


def _proc(returncode):
    return SimpleNamespace(returncode = returncode)


@_POSIX
@pytest.mark.parametrize(
    "returncode, output",
    [
        # bash on macOS (the report), then Linux, then a Python writer (it ignores SIGXFSZ, so it sees EFBIG).
        (1, "error: Filesize limit exceeded: 25\n"),
        (153, "bash: line 1:  42 File size limit exceeded (core dumped) dd if=/dev/zero of=big\n"),
        (1, "OSError: [Errno 27] File too large\n"),
        (1, "bash: line 1: ulimit: file size: cannot modify limit: Operation not permitted\n"),
        (-_SIGXFSZ, ""),
        (128 + _SIGXFSZ, ""),
        (0, "bash: line 1:  42 File size limit exceeded head -c 157286400 /dev/zero > big.bin\n"),
    ],
)
def test_a_call_stopped_by_the_limit_is_told_which_limit(monkeypatch, returncode, output):
    monkeypatch.setattr(tools, "_sandbox_fsize_bytes", 100 * MB)
    note = tools._file_size_exceeded_note(_proc(returncode), output, True)
    assert note.startswith("Stopped: this call hit the 100 MB sandbox file size limit")
    assert "Settings > Security > File size limit" in note
    assert "ulimit cannot raise it" in note


@_POSIX
@pytest.mark.parametrize(
    "returncode, output, sandboxed",
    [
        (0, "File too large is a fine thing to print", True),
        (1, "error: no such module 'Foo'\n", True),
        (1, "Filesize limit exceeded: 25\n", False),
    ],
)
def test_other_results_get_no_note(returncode, output, sandboxed):
    assert tools._file_size_exceeded_note(_proc(returncode), output, sandboxed) == ""


def test_windows_has_no_file_size_limit_to_name(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert tools._file_size_exceeded_note(_proc(1), "File too large", True) == ""


_WRITE_150_MB = "head -c 157286400 /dev/zero > big.bin && echo SIZE=$(( $(wc -c < big.bin) ))"


@pytest.fixture
def workdir(monkeypatch, tmp_path):
    folder = tmp_path / "chat"
    folder.mkdir()
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(folder))
    return folder


def _sandboxed_terminal(command):
    # Software mode: the rlimits are what is under test, on any POSIX host with or without an OS sandbox.
    return tools._bash_exec(command, timeout = 60, tool_execution_mode = "software")


@_POSIX
@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX terminal")
def test_a_real_terminal_call_past_the_limit_is_stopped_and_told_why(workdir):
    limit.set_file_size_limit_mb(100)
    result = _sandboxed_terminal(_WRITE_150_MB)
    assert "Stopped: this call hit the 100 MB sandbox file size limit" in result, result
    assert "SIZE=157286400" not in result
    assert (workdir / "big.bin").stat().st_size == 100 * MB


@_POSIX
@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX terminal")
def test_the_default_lets_the_same_call_finish(workdir):
    result = _sandboxed_terminal(_WRITE_150_MB)
    assert "SIZE=157286400" in result, result
    assert "Stopped:" not in result


@_POSIX
@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX terminal")
def test_ulimit_inside_the_sandbox_cannot_raise_it(workdir):
    limit.set_file_size_limit_mb(100)
    result = _sandboxed_terminal("ulimit -f unlimited; echo status=$?; ulimit -Hf; exit 1")
    assert "status=1" in result, result
    assert "Stopped: this call hit the 100 MB sandbox file size limit" in result
    # bash reports RLIMIT_FSIZE in 1024-byte blocks.
    assert str(100 * 1024) in result


@_POSIX
@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX python")
def test_a_real_python_call_past_the_limit_is_told_why(workdir):
    limit.set_file_size_limit_mb(100)
    code = "open('big.bin', 'wb').write(b'0' * (150 * 1024 * 1024))\nprint('wrote')\n"
    result = tools._python_exec(code, timeout = 60, tool_execution_mode = "software")
    assert "File too large" in result, result
    assert "Stopped: this call hit the 100 MB sandbox file size limit" in result
    assert "\nwrote\n" not in result
    assert (workdir / "big.bin").stat().st_size <= 100 * MB


@pytest.mark.skipif(tools._resource is None, reason = "RLIMIT_FSIZE is POSIX-only")
def test_an_inherited_hard_limit_below_the_setting_is_what_applies(monkeypatch):
    real = tools._resource.getrlimit
    monkeypatch.setattr(
        tools._resource,
        "getrlimit",
        lambda rid: (MB, 200 * MB) if rid == tools._resource.RLIMIT_FSIZE else real(rid),
    )
    tools._refresh_sandbox_memory_limit()
    assert tools._sandbox_fsize_bytes == 200 * MB
    note = tools._file_size_exceeded_note(_proc(128 + _SIGXFSZ), "", True)
    assert "200 MB sandbox file size limit" in note
