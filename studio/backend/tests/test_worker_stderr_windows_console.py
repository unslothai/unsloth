# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""#15583: a worker whose sys.stderr is a Windows console must keep writing once the mirror owns fd 2."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)

_CHILD = r"""
import atexit, json, os, sys, tempfile
backend, out_path, sink_dir = sys.argv[1:4]
sys.path.insert(0, backend)
from utils.worker_stderr import install_worker_stderr_mirror

def raw_name(stream):
    buffer = getattr(stream, "buffer", None)
    return type(getattr(buffer, "raw", buffer)).__name__

def write_stderr(text):
    try:
        print(text, file = sys.stderr, flush = True)
        return None
    except OSError as exc:
        return repr(exc)

result = {"raw_before": raw_name(sys.stderr)}
original = sys.stderr
handle, sink = tempfile.mkstemp(dir = sink_dir, suffix = ".stderr")
os.close(handle)
result["installed"] = install_worker_stderr_mirror(sink)
result["same_stream"] = sys.stderr is original
result["error"] = write_stderr("written through sys.stderr")
with open(sink, "rb") as handle:
    result["sink"] = handle.read().decode("utf-8", "replace")
atexit._run_exitfuncs()
result["restored"] = sys.stderr is original
result["error_after_stop"] = write_stderr("written after the mirror stopped")
with open(out_path, "w", encoding = "utf-8") as handle:
    json.dump(result, handle)
os._exit(0)
"""


def _run_child(
    tmp_path,
    unbuffered = False,
    **popen_kwargs,
) -> dict:
    out = tmp_path / "result.json"
    env = dict(os.environ)
    for key in ("PYTHONLEGACYWINDOWSSTDIO", "PYTHONIOENCODING", "PYTHONUNBUFFERED"):
        env.pop(key, None)
    if unbuffered:
        env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(
        [sys.executable, "-c", _CHILD, _BACKEND_DIR, str(out), str(tmp_path)],
        env = env,
        timeout = 120,
        check = True,
        **popen_kwargs,
    )
    return json.loads(out.read_text(encoding = "utf-8"))


@pytest.mark.skipif(sys.platform != "win32", reason = "a Windows console is Windows only")
@pytest.mark.parametrize("unbuffered", [False, True], ids = ["buffered", "unbuffered"])
def test_a_console_stderr_keeps_working_while_the_mirror_owns_fd_2(tmp_path, unbuffered):
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0  # SW_HIDE
    # No std handles passed: the child gets the new console's, as a worker spawned from `unsloth studio` in a terminal.
    result = _run_child(
        tmp_path,
        unbuffered = unbuffered,
        creationflags = subprocess.CREATE_NEW_CONSOLE,
        startupinfo = startupinfo,
    )

    # Fails rather than skips: without a console this proves nothing.
    assert result["raw_before"] == "_WindowsConsoleIO", result
    assert result["installed"] is True, result
    assert result["error"] is None, result
    assert "written through sys.stderr" in result["sink"], result
    assert result["restored"] is True, result
    assert result["error_after_stop"] is None, result


@pytest.mark.parametrize("unbuffered", [False, True], ids = ["buffered", "unbuffered"])
def test_a_stderr_that_is_not_a_console_is_left_as_it_was(tmp_path, unbuffered):
    result = _run_child(tmp_path, unbuffered = unbuffered, stderr = subprocess.DEVNULL)

    assert result["raw_before"] != "_WindowsConsoleIO", result
    assert result["installed"] is True, result
    assert result["same_stream"] is True, result
    assert result["error"] is None, result
    assert "written through sys.stderr" in result["sink"], result
    assert result["restored"] is True, result
