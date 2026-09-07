# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Recognise a Windows code integrity refusal in a failed process launch.

Windows Smart App Control, WDAC and AppLocker all refuse to load code through
the same kernel path, so a blocked binary does not fail like a missing or
corrupt one. It fails with a code integrity status, and the user sees a modal
"Bad Image" dialog naming whichever dependent DLL was refused:

    llama-server.exe - Bad Image
    ...\\llama-common.dll is either not designed to run on Windows or it
    contains an error. Error status 0xc0e90002.

Nothing in user space can make a blocked binary load. Telling the difference
matters because the remedies are opposite: a corrupt install is worth
reinstalling, while a policy refusal is not, and repairing it repeatedly is what
users end up doing when the error does not say which one it is.

The distinction is worth logging even when it cannot be surfaced, because the
alternative symptom is a probe that simply never answers.
"""

from __future__ import annotations

import re


# Statuses Windows reports when code integrity refuses an image.
#
# 0xC0E90002  the Smart App Control / system integrity policy facility, which is
#             what the "Bad Image" dialog shows for a SAC block
# 0xC0000428  STATUS_INVALID_IMAGE_HASH, an unsigned or tampered image under an
#             enforced policy
# 0xC0000602  STATUS_FAIL_FAST_EXCEPTION raised from a code integrity failure
_BLOCK_STATUS_CODES = {
    0xC0E90002: "Smart App Control or an Application Control policy blocked the image",
    0xC0000428: "the image failed code integrity validation (invalid or missing signature)",
    0xC0000602: "the image was refused by a code integrity fail-fast",
}

# Win32 error numbers surfaced through OSError.winerror for the same refusals.
#
# 577  ERROR_INVALID_IMAGE_HASH, "Windows cannot verify the digital signature"
# 1260 ERROR_ACCESS_DISABLED_BY_POLICY, the AppLocker/WDAC phrasing
# 4551 ERROR_CI_BLOCKED, seen against unsigned ROCm DLLs in unslothai/unsloth#6648
_BLOCK_WINERRORS = {
    577: "Windows could not verify the digital signature of the image",
    1260: "an Application Control policy blocked this program",
    4551: "code integrity blocked the image",
}

# Matches the status in text form wherever it reaches us as a string: a child's
# stderr, a Rust-side error, or the repr of an exception we did not raise.
_STATUS_TEXT_RE = re.compile(r"0x(c0e90002|c0000428|c0000602)\b", re.IGNORECASE)
_BAD_IMAGE_RE = re.compile(
    r"is either not designed to run on Windows or it contains an error", re.IGNORECASE
)
_POLICY_TEXT_RE = re.compile(
    r"(application control policy has blocked|blocked by (?:smart app control|group policy))",
    re.IGNORECASE,
)


def code_integrity_block_reason(error: object) -> str | None:
    """Return a human reason when ``error`` is a code integrity refusal, else None.

    Accepts an exception, a completed-process return code, or free text, since
    the same refusal reaches us through all three depending on whether we
    spawned the process directly or read a child's output.
    """
    winerror = getattr(error, "winerror", None)
    if isinstance(winerror, int):
        reason = _BLOCK_WINERRORS.get(winerror)
        if reason is not None:
            return reason
        # winerror can also carry the raw NTSTATUS on some launch failures.
        reason = _BLOCK_STATUS_CODES.get(winerror & 0xFFFFFFFF)
        if reason is not None:
            return reason

    returncode = getattr(error, "returncode", None)
    if isinstance(error, int):
        returncode = error
    if isinstance(returncode, int):
        # A negative return code is the signed reading of the same 32-bit status.
        reason = _BLOCK_STATUS_CODES.get(returncode & 0xFFFFFFFF)
        if reason is not None:
            return reason

    text = error if isinstance(error, str) else str(error)
    if not text:
        return None
    match = _STATUS_TEXT_RE.search(text)
    if match is not None:
        return _BLOCK_STATUS_CODES[int(match.group(1), 16)]
    if _POLICY_TEXT_RE.search(text):
        return "an Application Control policy blocked this program"
    if _BAD_IMAGE_RE.search(text):
        return "Windows refused to load the image (Bad Image)"
    return None


def code_integrity_user_message(binary: str, reason: str) -> str:
    """The message to show a user whose llama.cpp runtime will not load."""
    return (
        f"Windows blocked part of the local model runtime: {reason}. "
        f"The blocked file is under {binary}. "
        "This is Smart App Control or an Application Control policy refusing to load "
        "code it does not recognise, not a corrupt download, so reinstalling or "
        "running as administrator will not clear it. "
        "Smart App Control has no per-application exception; turning it off in "
        "Windows Security under App & browser control is the only local workaround."
    )
