# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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


def test_format_installer_failure_falls_back_to_tail_without_actionable_line():
    lines = ["line one\n", "line two\n"]
    message = flow.format_installer_failure_message(1, lines)
    assert message == "installer exited 1: line one\nline two"
