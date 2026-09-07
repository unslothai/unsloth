# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""A code integrity refusal must be told apart from an ordinary launch failure.

The two need opposite advice. Reinstalling fixes a corrupt download and does
nothing for a policy block, and users repair repeatedly when the error does not
distinguish them.
"""

import pytest

from utils.code_integrity import code_integrity_block_reason, code_integrity_user_message


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
    text = "Program 'unsloth.exe' failed to run: An Application Control policy has blocked this file"
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
