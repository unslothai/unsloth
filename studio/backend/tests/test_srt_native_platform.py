# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Opt-in checks against the installed Windows/macOS SRT backend."""

import os
import sys
import socket
import subprocess
import threading
import time

import pytest

pytestmark = [
    pytest.mark.allow_network,
    pytest.mark.skipif(
        sys.platform not in ("win32", "darwin")
        or os.environ.get("UNSLOTH_SRT_NATIVE_TESTS") != "1",
        reason = "requires installed native Windows or macOS SRT",
    ),
]


@pytest.fixture
def native_workdir(tmp_path, monkeypatch):
    from core.inference import os_sandbox, tools

    capability = os_sandbox.capability_snapshot(force = True)
    assert capability.available, capability.reason
    assert capability.profile_id == "srt-0.0.75-native-v1"
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id: str(tmp_path))
    return tmp_path


@pytest.mark.parametrize("kind", ["python", "terminal"])
def test_selected_native_tool_executes_with_receipt(native_workdir, kind):
    from core.inference import tools

    records = []
    execute = tools._python_exec if kind == "python" else tools._bash_exec
    code = "print('NATIVE_PLATFORM_OK')" if kind == "python" else "printf NATIVE_PLATFORM_OK"
    output = execute(code, None, 30, "native-platform", launch_record_callback = records.append)
    assert "NATIVE_PLATFORM_OK" in output, output
    assert len(records) == 1
    assert records[0].os_isolation and records[0].backend == "srt"


def test_native_python_selected_packages_and_file(native_workdir):
    from core.inference import tools

    records = []
    output = tools._python_exec(
        "from PIL import Image\nImage.new('RGB',(4,4)).save('native.png')\nprint('IMAGE_OK')",
        None,
        30,
        "native-platform",
        launch_record_callback = records.append,
    )
    assert "IMAGE_OK" in output, output
    assert (native_workdir / "native.png").is_file()
    assert len(records) == 1


def test_native_failure_cannot_attest_success(native_workdir):
    from core.inference import tools

    records = []
    output = tools._python_exec(
        "raise ValueError('EXPECTED_NATIVE_FAILURE')",
        None,
        30,
        "native-platform",
        launch_record_callback = records.append,
    )
    assert "EXPECTED_NATIVE_FAILURE" in output, output
    assert records == []


def test_native_srt_blocks_direct_host_connection(native_workdir):
    from core.inference import srt_adapter, tools
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        address = listener.getsockname()
        with socket.create_connection(address, timeout = 2):
            accepted, _ = listener.accept()
            accepted.close()
        code = """
import socket, sys
with socket.socket() as connection:
    connection.settimeout(2)
    try:
        connection.connect(('127.0.0.1', int(sys.argv[1])))
    except OSError:
        print('NATIVE_NETWORK_DENIED')
    else:
        raise RuntimeError('Direct host connection escaped SRT network policy')
"""
        request = srt_adapter.request_for(
            [sys.executable, "-I", "-S", "-c", code, str(address[1])],
            str(native_workdir),
            tools._build_safe_env(str(native_workdir)),
            30,
        )
        proc = srt_adapter.spawn(request, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
        try:
            output, _ = proc.communicate(timeout = 45)
            assert proc.returncode == 0, output.decode(errors = "replace")
            srt_adapter.verify_success(proc)
            assert b"NATIVE_NETWORK_DENIED" in output
        finally:
            if proc.poll() is None:
                srt_adapter._stop_failed_helper(proc, getattr(proc, "_srt_control_socket", None))
            srt_adapter.release_control(proc)


def test_native_cancellation_returns_promptly(native_workdir):
    from core.inference import tools

    cancel = threading.Event()
    started_file = native_workdir / "cancel-started.txt"
    observed = []

    def cancel_running_payload():
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and not cancel.is_set():
            if started_file.exists():
                observed.append(time.monotonic())
                cancel.set()
                return
            time.sleep(0.05)
        cancel.set()

    watcher = threading.Thread(target = cancel_running_payload, daemon = True)
    records = []
    watcher.start()
    try:
        tools._python_exec(
            "from pathlib import Path\nimport time\nPath('cancel-started.txt').write_text('running')\ntime.sleep(60)",
            timeout = 30,
            session_id = "native-platform",
            cancel_event = cancel,
            launch_record_callback = records.append,
        )
    finally:
        cancel.set()
        watcher.join(timeout = 1)
    assert observed, "Cancellation control never reached the native payload"
    assert time.monotonic() - observed[0] < 15
    assert records == []
