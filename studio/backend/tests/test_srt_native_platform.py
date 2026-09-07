# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Opt-in checks against the installed Windows/macOS SRT backend."""

import os
import sys
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


def test_native_cancellation_returns_promptly(native_workdir):
    from core.inference import tools

    cancel = threading.Event()
    timer = threading.Timer(3, cancel.set)
    records = []
    started = time.monotonic()
    timer.start()
    try:
        tools._python_exec(
            "import time\ntime.sleep(60)",
            None,
            30,
            "native-platform",
            cancel_event = cancel,
            launch_record_callback = records.append,
        )
    finally:
        timer.cancel()
    assert time.monotonic() - started < 15
    assert records == []
