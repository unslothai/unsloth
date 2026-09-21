# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys

import pytest

from utils.prebuilt import update_flow as flow


@pytest.fixture(autouse = True)
def _no_github_token(monkeypatch):
    # The rate-limit advice depends on it, and a CI runner may export either.
    monkeypatch.delenv("GH_TOKEN", raising = False)
    monkeypatch.delenv("GITHUB_TOKEN", raising = False)


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


def test_stream_installer_keeps_a_multiline_verdict_whole(tmp_path):
    # The linux preflight failure names one binary per line; truncating at the
    # first line leaves "preflight failed:" and nothing to act on.
    exc = _run_fake_installer(
        tmp_path,
        "print('[llama-prebuilt] prebuilt install path failed; falling back to source build')\n"
        "for line in ['prebuilt fallback reason: linux extracted binary preflight failed:',\n"
        "             'llama-server: missing=libcuda.so.1 ld_library_path=none',\n"
        "             'llama-quantize: missing=libgomp.so.1 ld_library_path=none']:\n"
        "    print('[llama-prebuilt] ' + line)\n"
        "print('platform=Linux machine=x86_64')\n"
        "for i in range(300):\n"
        "    print(f'linux_runtime_dirs_{i}=/usr/lib')\n"
        "sys.exit(2)\n",
    )
    assert exc.returncode == 2
    assert "missing=libcuda.so.1" in str(exc)
    assert "missing=libgomp.so.1" in str(exc)
    # The unprefixed system report is where the verdict ends.
    assert "linux_runtime_dirs" not in str(exc)
    assert "platform=Linux" not in str(exc)


def test_stream_installer_reports_a_fatal_error_that_follows_a_survived_retry(tmp_path):
    # The retry succeeded and the run died of something else; the stale hint must
    # not replace the error the installer actually exited on.
    exc = _run_fake_installer(
        tmp_path,
        "url = 'https://api.github.com/repos/unslothai/llama.cpp/releases/tags/b10679'\n"
        "print(f'[llama-prebuilt] fetch failed (1/4) for {url}: "
        "HTTP Error 429: Too Many Requests; retrying')\n"
        "print('[llama-prebuilt] resolved published release b10679-mix-67dfc8b')\n"
        "print('[llama-prebuilt] fatal helper error: staged install could not be "
        "activated: [Errno 13] Permission denied')\n"
        "sys.exit(1)\n",
    )
    assert exc.returncode == 1
    assert str(exc) == (
        "installer exited 1: staged install could not be activated: [Errno 13] Permission denied"
    )


def test_stream_installer_keeps_the_output_when_only_a_rate_limit_hint_is_present(tmp_path):
    # No verdict line at all: the hint annotates the output rather than replacing
    # it, so a crash is still readable.
    exc = _run_fake_installer(
        tmp_path,
        "url = 'https://api.github.com/repos/unslothai/llama.cpp/releases/tags/b10679'\n"
        "print(f'[llama-prebuilt] fetch failed (1/4) for {url}: "
        "HTTP Error 403: rate limit exceeded; retrying')\n"
        "print('Traceback (most recent call last):')\n"
        "print('MemoryError')\n"
        "sys.exit(1)\n",
    )
    assert exc.returncode == 1
    assert "GH_TOKEN" in str(exc)
    assert "MemoryError" in str(exc)


def test_format_installer_failure_tells_an_authenticated_run_to_wait(monkeypatch):
    # The token is already set, so its quota is what ran out: setting it again is
    # advice the user cannot act on. fetch_json omits its own hint here too.
    monkeypatch.setenv("GH_TOKEN", "x")
    lines = [
        "[llama-prebuilt] prebuilt fallback reason: failed to inspect published "
        "releases in unslothai/llama.cpp: GitHub API returned 429 for "
        "https://api.github.com/repos/unslothai/llama.cpp/releases/tags/b10679\n",
    ]
    message = flow.format_installer_failure_message(2, lines)
    assert "rate limit" in message.lower()
    assert "GH_TOKEN" not in message
    assert "Wait for the limit to reset" in message


def test_stream_installer_reads_the_token_from_the_installer_environment(tmp_path, monkeypatch):
    # The advice must follow the env the child ran with, not this process's.
    monkeypatch.delenv("GH_TOKEN", raising = False)
    script = tmp_path / "fake_installer.py"
    script.write_text(
        "import sys\n"
        "print('[llama-prebuilt] prebuilt fallback reason: GitHub API returned 403 for "
        "https://api.github.com/repos/unslothai/llama.cpp/releases/latest')\n"
        "sys.exit(2)\n",
        encoding = "utf-8",
    )
    with pytest.raises(flow.InstallerExit) as excinfo:
        flow.stream_installer(
            [sys.executable, str(script)],
            {"GH_TOKEN": "x"},
            timeout_seconds = 60,
            set_progress = lambda _fraction: None,
        )
    assert "GH_TOKEN" not in str(excinfo.value)
    assert "Wait for the limit to reset" in str(excinfo.value)


def test_stream_installer_keeps_a_verdict_a_child_announcement_interrupts(tmp_path):
    # The installer announces the servers it starts on the same stream; that
    # protocol line must not be mistaken for the start of the system report.
    exc = _run_fake_installer(
        tmp_path,
        "print('[llama-prebuilt] prebuilt fallback reason: linux extracted binary "
        "preflight failed:')\n"
        "print('UNSLOTH_INSTALLER_CHILD started 4242')\n"
        "print('[llama-prebuilt] llama-server: missing=libcuda.so.1 ld_library_path=none')\n"
        "print('UNSLOTH_INSTALLER_CHILD stopped 4242')\n"
        "sys.exit(2)\n",
    )
    assert exc.returncode == 2
    assert "missing=libcuda.so.1" in str(exc)
    assert "UNSLOTH_INSTALLER_CHILD" not in str(exc)
