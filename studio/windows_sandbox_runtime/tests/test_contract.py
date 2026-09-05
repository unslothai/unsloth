# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Bootstrap contract tests. These are not live LPAC qualification tests."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import subprocess
import sys

import pytest

BACKEND = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND))
from core.inference.windows_sandbox.profiles import (
    ABI_ADAPTERS,
    PYTHON_PROFILE,
    WindowsRuntimeError,
    select_abi_adapter,
)


@pytest.mark.parametrize("minor", [11, 12, 13])
@pytest.mark.parametrize("patch", [0, 10, 99])
def test_supported_minor_selects_one_matching_adapter(minor, patch):
    adapter = select_abi_adapter(
        implementation = "cpython", version = (3, minor, patch), architecture = "x64"
    )
    assert adapter.identity == f"cpython-3{minor}-x64-release"
    assert adapter.configuration_api == "PyConfig"
    # A supported ABI does not qualify an installed runtime or a patch release.
    assert not hasattr(adapter, "qualified")
    assert not hasattr(adapter, "available")


@pytest.mark.parametrize(
    "override",
    [
        {"implementation": "pypy"},
        {"version": (3, 9, 0)},
        {"version": (3, 10, 0)},
        {"version": (3, 14, 0)},
        {"version": (3, 12, -1)},
        {"version": (3, 12, True)},
        {"version": (3, 12)},
        {"architecture": "arm64"},
        {"architecture": "x86"},
        {"debug": True},
        {"free_threaded": True},
    ],
)
def test_unknown_abi_never_selects_nearby_adapter(override):
    args = dict(implementation = "cpython", version = (3, 12, 10), architecture = "x64")
    args.update(override)
    with pytest.raises(WindowsRuntimeError) as error:
        select_abi_adapter(**args)
    assert error.value.code == "WINDOWS_SANDBOX_ABI_UNSUPPORTED"


def test_profile_is_immutable_versioned_and_has_no_payload_capabilities():
    assert PYTHON_PROFILE.active_process_limit == 1
    assert PYTHON_PROFILE.payload_capabilities == ()
    assert PYTHON_PROFILE.startup_capabilities == ("registryRead",)
    assert PYTHON_PROFILE.abi_adapters == ABI_ADAPTERS
    assert len({adapter.identity for adapter in ABI_ADAPTERS}) == 3
    assert PYTHON_PROFILE.schema_version == PYTHON_PROFILE.protocol_version == 1
    assert len(PYTHON_PROFILE.digest) == 64
    assert replace(PYTHON_PROFILE, protocol_version = 2).digest != PYTHON_PROFILE.digest
    with pytest.raises(FrozenInstanceError):
        PYTHON_PROFILE.active_process_limit = 2


def _run_policy(code):
    # The shim changes process APIs, so never install it in the pytest process.
    source = (
        f"import sys; sys.path.insert(0, {str(BACKEND)!r})\n"
        "from core.inference.windows_sandbox.policy import install_single_process_policy, "
        "WindowsSandboxChildProcessDisabled\n"
        "install_single_process_policy()\n"
        "install_single_process_policy()\n" + code
    )
    return subprocess.run(
        [sys.executable, "-I", "-S", "-c", source],
        capture_output = True,
        text = True,
        encoding = "utf-8",
        timeout = 15,
        check = False,
    )


@pytest.mark.parametrize(
    "call",
    [
        "multiprocessing.Process(target=print).start()",
        "multiprocessing.get_context('spawn').Process(target=print).start()",
        "multiprocessing.Pool(1)",
        "multiprocessing.pool.Pool(1)",
        "multiprocessing.get_context('spawn').Pool(1)",
        "multiprocessing.Manager()",
        "multiprocessing.get_context('spawn').Manager()",
        "multiprocessing.managers.SyncManager().start()",
        "concurrent.futures.ProcessPoolExecutor(1)",
        "subprocess.run([sys.executable, '-c', 'print(123)'])",
        "asyncio.run(asyncio.create_subprocess_exec(sys.executable, '-c', 'print(123)'))",
        "asyncio.run(asyncio.create_subprocess_shell('echo 123'))",
    ],
)
def test_worker_apis_fail_early_with_specific_error(call):
    result = _run_policy(
        "import multiprocessing, multiprocessing.pool, multiprocessing.managers\n"
        "import concurrent.futures, subprocess, asyncio\n"
        "try:\n"
        f"    {call}\n"
        "except WindowsSandboxChildProcessDisabled as exc:\n"
        "    assert exc.code == 'WINDOWS_SANDBOX_CHILD_PROCESS_DISABLED'\n"
        "    assert 'single-process or threading mode' in str(exc)\n"
        "else:\n"
        "    raise AssertionError('worker API was allowed')\n"
        "print('POLICY_ERROR_OK')\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "POLICY_ERROR_OK"


def test_imports_threads_thread_pool_and_asyncio_still_work():
    result = _run_policy(
        "import asyncio, threading, concurrent.futures, multiprocessing.pool\n"
        "import sqlite3, ssl, _ctypes, _bz2, _lzma\n"
        "out = []\n"
        "thread = threading.Thread(target=lambda: out.append(7))\n"
        "thread.start(); thread.join(timeout=2)\n"
        "assert out == [7] and not thread.is_alive()\n"
        "with concurrent.futures.ThreadPoolExecutor(2) as pool:\n"
        "    assert list(pool.map(abs, [-1, -2])) == [1, 2]\n"
        "with multiprocessing.pool.ThreadPool(2) as pool:\n"
        "    assert pool.map(abs, [-3, -4]) == [3, 4]\n"
        "async def main():\n"
        "    await asyncio.sleep(0)\n"
        "    return 5\n"
        "assert asyncio.run(main()) == 5\n"
        "print('THREADS_AND_IMPORTS_OK')\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "THREADS_AND_IMPORTS_OK"


def test_importing_policy_does_not_install_it_in_host():
    import subprocess

    original = subprocess.Popen.__init__
    from core.inference.windows_sandbox import policy

    assert (
        policy.WindowsSandboxChildProcessDisabled.code == "WINDOWS_SANDBOX_CHILD_PROCESS_DISABLED"
    )
    assert subprocess.Popen.__init__ is original
