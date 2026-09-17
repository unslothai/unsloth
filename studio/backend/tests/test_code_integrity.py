# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""A code integrity refusal needs the opposite advice to a corrupt download."""

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
    # Popen reports the same status as a negative int, read as signed.
    assert code_integrity_block_reason(status) is not None
    assert code_integrity_block_reason(status - (1 << 32)) is not None


def test_application_control_phrasing():
    """The wording from unslothai/unsloth#8490."""
    text = (
        "Program 'unsloth.exe' failed to run: An Application Control policy has blocked this file"
    )
    assert code_integrity_block_reason(text) is not None


def test_ordinary_failures_are_not_misreported():
    assert code_integrity_block_reason(FileNotFoundError("no such file")) is None
    assert code_integrity_block_reason(_WinError(2)) is None
    assert code_integrity_block_reason("llama-server: unknown argument --nope") is None
    assert code_integrity_block_reason(1) is None
    assert code_integrity_block_reason("") is None
    assert code_integrity_block_reason(None) is None


def test_user_message_names_the_binary_and_rules_out_reinstalling():
    message = code_integrity_user_message(r"C:\Users\x\.unsloth\llama.cpp", "blocked")
    assert r"C:\Users\x\.unsloth\llama.cpp" in message
    assert "reinstalling" in message
    assert "administrator" in message


def test_an_administrator_policy_block_is_not_sent_to_smart_app_control():
    """winerror 1260 and the AppLocker wording mean an admin policy, which
    turning Smart App Control off does not lift."""
    for error in (_WinError(1260), "An Application Control policy has blocked this file"):
        reason = code_integrity_block_reason(error)
        assert reason is not None
        message = code_integrity_user_message(r"C:\Users\x\.unsloth\llama.cpp", reason)
        assert "administers this device" in message
        assert "only local workaround" not in message

    sac_reason = code_integrity_block_reason("This app was blocked by Smart App Control")
    assert sac_reason is not None
    sac_message = code_integrity_user_message(r"C:\x", sac_reason)
    assert "only local workaround" in sac_message
    assert "administers this device" not in sac_message

    # SAC and WDAC both report 0xC0E90002, so it offers both remedies.
    ambiguous = code_integrity_user_message(r"C:\x", code_integrity_block_reason(0xC0E90002))
    assert "Smart App Control" in ambiguous
    assert "managed by an administrator" in ambiguous


def test_an_invalid_image_hash_does_not_deny_corruption():
    """0xC0000428 / winerror 577 are a hash mismatch, which Windows reports for a
    damaged file too, so the message must not rule out a reinstall."""
    for error in (0xC0000428, 0xC0000428 - (1 << 32), _WinError(577)):
        reason = code_integrity_block_reason(error)
        assert reason is not None
        message = code_integrity_user_message(r"C:\Users\x\.unsloth\llama.cpp", reason)
        assert r"C:\Users\x\.unsloth\llama.cpp" in message
        assert "not a corrupt download" not in message
        assert "Reinstalling" in message
        assert "Smart App Control" in message
        assert "managed by an administrator" in message

    policy = code_integrity_user_message(r"C:\x", code_integrity_block_reason(0xC0E90002))
    assert "not a corrupt download" in policy


def test_bad_image_without_a_status_is_not_called_a_policy_block():
    """Windows prints the same sentence for a corrupt or wrong-architecture DLL,
    which does need the reinstall this module rules out."""
    corrupt = (
        r"C:\Users\x\.unsloth\llama.cpp\build\bin\Release\ggml-base.dll is either not "
        r"designed to run on Windows or it contains an error."
    )
    assert code_integrity_block_reason(corrupt) is None
    assert is_bad_image_text(corrupt) is True
    assert code_integrity_block_reason(corrupt + " Error status 0xc0e90002.") is not None


def test_a_blocked_start_is_explained_to_the_user_not_blamed_on_the_model():
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
    assert "out of memory" not in message.lower()

    by_status = LlamaCppBackend._classify_start_failure_text(
        output = "",
        gguf_path = None,
        model_identifier = None,
        returncode = 0xC0E90002,
        binary = r"C:\Users\x\.unsloth\llama.cpp",
    )
    assert "Smart App Control" in by_status
