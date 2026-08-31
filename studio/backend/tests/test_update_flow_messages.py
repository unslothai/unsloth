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


def test_format_installer_failure_prefers_the_verdict_over_an_earlier_rate_limit_retry():
    # The installer retries a rate-limited fetch and can still get through, so a
    # retry line must never outrank the reason it actually exited on.
    lines = [
        "[llama-prebuilt] fetch failed (1/4) for "
        "https://api.github.com/repos/unslothai/llama.cpp/releases/tags/b10679: "
        "HTTP Error 403: rate limit exceeded; retrying\n",
        "[llama-prebuilt] prebuilt install refused: unknown backend request 'cuda14'\n",
    ]
    message = flow.format_installer_failure_message(1, lines)
    assert message == "installer exited 1: unknown backend request 'cuda14'"


def _run_fake_installer(tmp_path, body: str) -> flow.InstallerExit:
    script = tmp_path / "fake_installer.py"
    script.write_text("import sys\n" + body, encoding = "utf-8")
    with pytest.raises(flow.InstallerExit) as excinfo:
        flow.stream_installer(
            [sys.executable, str(script)],
            {},
            timeout_seconds = 60,
            set_progress = lambda _fraction: None,
        )
    return excinfo.value


def test_stream_installer_keeps_reason_a_long_system_report_pushes_out_of_the_tail(tmp_path):
    # The report (selection log, nvidia-smi, ldd) outruns the bounded tail on a
    # Linux CUDA host, so the reason has to be kept as it streams, not looked up
    # in the tail afterwards.
    exc = _run_fake_installer(
        tmp_path,
        "print('[llama-prebuilt] prebuilt fallback reason: failed to inspect published "
        "releases in unslothai/llama.cpp: GitHub API returned 403; set GH_TOKEN or "
        "GITHUB_TOKEN to avoid GitHub API rate limits')\n"
        "for i in range(300):\n"
        "    print(f'linux_runtime_dirs_{i}=/usr/lib/x86_64-linux-gnu')\n"
        "sys.exit(2)\n",
    )
    assert exc.returncode == 2
    assert "GH_TOKEN" in str(exc)
    assert "linux_runtime_dirs" not in str(exc)


def test_stream_installer_keeps_the_verdict_behind_a_run_of_rate_limit_retries(tmp_path):
    # Retries are unbounded in principle and the verdict comes last, so the two
    # cannot share one capped buffer.
    exc = _run_fake_installer(
        tmp_path,
        "url = 'https://api.github.com/repos/unslothai/llama.cpp/releases/tags/b10679'\n"
        "for i in range(1, 11):\n"
        "    print(f'[llama-prebuilt] fetch failed ({i}/4) for {url}: "
        "HTTP Error 403: rate limit exceeded; retrying')\n"
        "print('[llama-prebuilt] prebuilt fallback reason: no published prebuilt asset "
        "matches this host (windows-rocm gfx803)')\n"
        "for i in range(300):\n"
        "    print(f'windows_runtime_dirs_{i}=C:/x')\n"
        "sys.exit(2)\n",
    )
    assert exc.returncode == 2
    assert str(exc) == (
        "installer exited 2: no published prebuilt asset matches this host (windows-rocm gfx803)"
    )
