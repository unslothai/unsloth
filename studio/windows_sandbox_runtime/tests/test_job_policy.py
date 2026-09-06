# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Real Job controls isolate process-limit enforcement from LPAC child denial.

The fixed parent/child controls are intentionally ordinary host Python. This
tests the shared Job owner, not the complete bootstrap security boundary.
"""

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import subprocess
import sys

import pytest

BACKEND = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND))
from core.inference import windows_lpac

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason = "Native Windows Job Object tests")


@pytest.mark.parametrize(
    "limit,breakaway,child_runs", [(2, False, True), (1, False, False), (1, True, False)]
)
def test_live_job_enforces_single_process_before_payload_entry(
    limit, breakaway, child_runs, tmp_path, monkeypatch
):
    import _winapi

    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_NPROC", "10000")
    sentinel = tmp_path / "child-ran"
    diagnostic = tmp_path / "parent-diagnostic"
    child_code = f"from pathlib import Path; Path({str(sentinel)!r}).write_text('executed', encoding='utf-8')"
    child_args = [sys.executable, "-I", "-S", "-c", child_code]
    source = (
        "import subprocess, sys\n"
        "try:\n"
        f"    child = subprocess.run({child_args!r}, creationflags={0x1000000 if breakaway else 0}, "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)\n"
        "except OSError as exc:\n"
        # ActiveProcessLimit reports ERROR_NOT_ENOUGH_QUOTA (1816); a prohibited
        # breakaway reports ERROR_ACCESS_DENIED (5). Do not accept arbitrary errors.
        f"    assert exc.winerror == {5 if breakaway else 1816}, repr(exc)\n"
        f"    assert {not child_runs!r}\n"
        "    sys.exit(0)\n"
        # If creation returns a process, require an unsuccessful exit as well
        # as absence of its sentinel. The unrestricted control must write it.
        f"assert (child.returncode == 0) is {child_runs!r}, child.returncode\n"
    )
    source = (
        "try:\n"
        + "\n".join("    " + line for line in source.splitlines())
        + "\nexcept Exception:\n"
        + "    import traceback\n"
        + f"    open({str(diagnostic)!r}, 'w', encoding='utf-8').write(traceback.format_exc())\n"
        + "    raise\n"
    )
    startup = subprocess.STARTUPINFO()
    process, thread, _, _ = _winapi.CreateProcess(
        sys.executable,
        subprocess.list2cmdline([sys.executable, "-I", "-S", "-c", source]),
        None,
        None,
        False,
        0x4 | 0x08000000,
        {"SystemRoot": os.environ["SystemRoot"], "TEMP": str(tmp_path), "TMP": str(tmp_path)},
        str(tmp_path),
        startup,
    )
    api = windows_lpac._api()
    job = None
    try:
        job = windows_lpac._create_job(process, active_process_limit = limit)
        query = api.kernel32.QueryInformationJobObject
        query.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.c_void_p,
        ]
        query.restype = wintypes.BOOL
        info = windows_lpac._JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        assert query(job._handle, 9, ctypes.byref(info), ctypes.sizeof(info), None)
        assert info.BasicLimitInformation.ActiveProcessLimit == limit
        flags = info.BasicLimitInformation.LimitFlags
        assert flags & 0x2000  # KILL_ON_JOB_CLOSE
        assert not flags & (0x800 | 0x1000)  # no explicit or silent breakaway
        assert api.kernel32.ResumeThread(thread) == 1
        assert api.kernel32.WaitForSingleObject(process, 10000) == 0
        code = wintypes.DWORD()
        assert api.kernel32.GetExitCodeProcess(process, ctypes.byref(code))
        assert code.value == 0, (
            diagnostic.read_text(encoding = "utf-8") if diagnostic.exists() else code.value
        )
        assert sentinel.exists() is child_runs
        if child_runs:
            assert sentinel.read_text(encoding = "utf-8") == "executed"
    finally:
        if job is not None:
            job.close()
        api.kernel32.TerminateProcess(process, 1)
        try:
            assert api.kernel32.WaitForSingleObject(process, 5000) == 0
        finally:
            api.kernel32.CloseHandle(thread)
            api.kernel32.CloseHandle(process)
