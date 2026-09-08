# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""A code integrity refusal must be told apart from an ordinary launch failure.

The two need opposite advice. Reinstalling fixes a corrupt download and does
nothing for a policy block, and users repair repeatedly when the error does not
distinguish them.
"""

import pytest

from utils.code_integrity import (
    code_integrity_block_reason,
    code_integrity_user_message,
    is_bad_image_text,
)


class _WinError(OSError):
    def __init__(self, winerror: int):
        super().__init__("launch failed")
        self.winerror = winerror


def test_smart_app_control_status_from_text():
    """0xc0e90002 is what the Bad Image dialog shows for a Smart App Control block."""
    text = (
        r"C:\Users\x\.unsloth\llama.cpp\build\bin\Release\llama-common.dll is either "
        r"not designed to run on Windows or it contains an error. Error status 0xc0e90002."
    )
    reason = code_integrity_block_reason(text)
    assert reason is not None
    assert "Smart App Control" in reason


@pytest.mark.parametrize("winerror", [577, 1260, 4551])
def test_win32_error_numbers_are_recognised(winerror: int):
    assert code_integrity_block_reason(_WinError(winerror)) is not None


@pytest.mark.parametrize("status", [0xC0E90002, 0xC0000428, 0xC0000602])
def test_ntstatus_return_codes_are_recognised(status: int):
    # Popen reports the same status as a negative int once it is read as signed.
    assert code_integrity_block_reason(status) is not None
    assert code_integrity_block_reason(status - (1 << 32)) is not None


def test_application_control_phrasing():
    """The wording from unslothai/unsloth#8490, where unsloth.exe was blocked."""
    text = (
        "Program 'unsloth.exe' failed to run: An Application Control policy has blocked this file"
    )
    assert code_integrity_block_reason(text) is not None


def test_ordinary_failures_are_not_misreported():
    """A missing file or a plain nonzero exit must not read as a policy block."""
    assert code_integrity_block_reason(FileNotFoundError("no such file")) is None
    assert code_integrity_block_reason(_WinError(2)) is None
    assert code_integrity_block_reason("llama-server: unknown argument --nope") is None
    assert code_integrity_block_reason(1) is None
    assert code_integrity_block_reason("") is None
    assert code_integrity_block_reason(None) is None


def test_user_message_names_the_binary_and_rules_out_reinstalling():
    message = code_integrity_user_message(r"C:\Users\x\.unsloth\llama.cpp", "blocked")
    assert r"C:\Users\x\.unsloth\llama.cpp" in message
    # The two things users try that cannot work.
    assert "reinstalling" in message
    assert "administrator" in message


def test_an_administrator_policy_block_is_not_sent_to_smart_app_control():
    """1260 and the AppLocker/Group Policy wording identify an admin's policy.

    Turning Smart App Control off does not lift a WDAC/AppLocker/Group Policy
    block: it is a different feature, usually already off on a managed device,
    and switching it off is a security downgrade the user may not be able to
    undo. Advertising it as "the only local workaround" there leaves the runtime
    blocked and the user worse off.
    """
    for error in (_WinError(1260), "An Application Control policy has blocked this file"):
        reason = code_integrity_block_reason(error)
        assert reason is not None
        message = code_integrity_user_message(r"C:\Users\x\.unsloth\llama.cpp", reason)
        assert "administers this device" in message
        assert "only local workaround" not in message

    # A confirmed Smart App Control block still gets the Smart App Control remedy.
    sac_reason = code_integrity_block_reason("This app was blocked by Smart App Control")
    assert sac_reason is not None
    sac_message = code_integrity_user_message(r"C:\x", sac_reason)
    assert "only local workaround" in sac_message
    assert "administers this device" not in sac_message

    # The ambiguous status (SAC and WDAC both report 0xC0E90002) offers both.
    ambiguous = code_integrity_user_message(r"C:\x", code_integrity_block_reason(0xC0E90002))
    assert "Smart App Control" in ambiguous
    assert "managed by an administrator" in ambiguous


def test_bad_image_without_a_status_is_not_called_a_policy_block():
    """The stock Bad Image sentence is not evidence of Application Control.

    Windows prints the same wording for a corrupt DLL, one built for another
    architecture, and one whose own dependencies are missing. Those are ordinary
    broken installs, and the advice this module gives for a policy block
    ("reinstalling will not help") is the exact opposite of what they need.
    """
    corrupt = (
        r"C:\Users\x\.unsloth\llama.cpp\build\bin\Release\ggml-base.dll is either not "
        r"designed to run on Windows or it contains an error."
    )
    assert code_integrity_block_reason(corrupt) is None
    # Still recognisable as an image load failure, just not as a policy one.
    assert is_bad_image_text(corrupt) is True

    # The same sentence WITH the status is a block, and still classified.
    assert code_integrity_block_reason(corrupt + " Error status 0xc0e90002.") is not None


def test_a_blocked_start_is_explained_to_the_user_not_blamed_on_the_model():
    """The classifier has to reach a user-facing message, not just a log line.

    Every other branch of _classify_start_failure_text gives advice that is
    actively wrong for a policy refusal: reinstall, free memory, install a
    missing library. The file is present and Windows will not load it.
    """
    import sys
    from pathlib import Path

    backend = Path(__file__).resolve().parents[1]
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    from core.inference.llama_cpp import LlamaCppBackend

    message = LlamaCppBackend._classify_start_failure_text(
        output = (
            r"C:\Users\x\.unsloth\llama.cpp\build\bin\Release\llama-common.dll is either "
            r"not designed to run on Windows or it contains an error. Error status 0xc0e90002."
        ),
        gguf_path = "C:\\models\\qwen.gguf",
        model_identifier = "unsloth/Qwen3.5-2B-MTP-GGUF",
        binary = r"C:\Users\x\.unsloth\llama.cpp",
    )
    assert "Smart App Control" in message
    assert "reinstalling" in message
    # It must NOT fall through to the generic file-or-memory advice.
    assert "out of memory" not in message.lower()

    # The same refusal delivered as an exit status, with no output at all.
    by_status = LlamaCppBackend._classify_start_failure_text(
        output = "",
        gguf_path = None,
        model_identifier = None,
        returncode = 0xC0E90002,
        binary = r"C:\Users\x\.unsloth\llama.cpp",
    )
    assert "Smart App Control" in by_status
