# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys

import pytest

from utils.prebuilt import update_flow as flow


def test_format_installer_failure_prefers_fallback_reason_over_path_noise():
    huge_path = "C:\\Windows\\" + "VeryLongDir\\" * 120 + "torch\\lib"
    lines = [
        "[llama-prebuilt] prebuilt install path failed; falling back to source build\n",
        (
            "[llama-prebuilt] prebuilt fallback reason: failed to inspect published "
            "releases in unslothai/llama.cpp: GitHub API returned 403 for "
            "https://api.github.com/repos/unslothai/llama.cpp/releases/tags/b10679; "
            "set GH_TOKEN or GITHUB_TOKEN to avoid GitHub API rate limits\n"
        ),
        f"windows_runtime_dirs={huge_path}\n",
    ]
    message = flow.format_installer_failure_message(2, lines)
    assert "GH_TOKEN" in message
    assert "rate limit" in message.lower()
    assert "windows_runtime_dirs" not in message


def test_format_installer_failure_does_not_name_llama_for_a_whisper_install():
    # whisper.cpp updates share stream_installer, and prebuilt_core raises the same
    # 403 text for both, so the message must not name the wrong component.
    lines = [
        "[whisper-prebuilt] prebuilt install failed: GitHub API returned 403 for "
        "https://api.github.com/repos/unslothai/whisper.cpp/releases/latest; "
        "set GH_TOKEN or GITHUB_TOKEN to avoid GitHub API rate limits\n",
    ]
    message = flow.format_installer_failure_message(1, lines)
    assert "GH_TOKEN" in message
    assert "llama" not in message.lower()


def test_format_installer_failure_falls_back_to_tail_without_actionable_line():
    lines = ["line one\n", "line two\n"]
    message = flow.format_installer_failure_message(1, lines)
    assert message == "installer exited 1: line one\nline two"


def test_stream_installer_keeps_reason_a_long_system_report_pushes_out_of_the_tail(tmp_path):
    # The report (selection log, nvidia-smi, ldd) outruns the bounded tail on a
    # Linux CUDA host, so the reason has to be kept as it streams, not looked up
    # in the tail afterwards.
    script = tmp_path / "fake_installer.py"
    script.write_text(
        "import sys\n"
        "print('[llama-prebuilt] prebuilt fallback reason: failed to inspect published "
        "releases in unslothai/llama.cpp: GitHub API returned 403; set GH_TOKEN or "
        "GITHUB_TOKEN to avoid GitHub API rate limits')\n"
        "for i in range(300):\n"
        "    print(f'linux_runtime_dirs_{i}=/usr/lib/x86_64-linux-gnu')\n"
        "sys.exit(2)\n",
        encoding = "utf-8",
    )
    with pytest.raises(flow.InstallerExit) as excinfo:
        flow.stream_installer(
            [sys.executable, str(script)],
            {},
            timeout_seconds = 60,
            set_progress = lambda _fraction: None,
        )
    assert excinfo.value.returncode == 2
    assert "GH_TOKEN" in str(excinfo.value)
    assert "linux_runtime_dirs" not in str(excinfo.value)
