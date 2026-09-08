# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Recognise a Windows code integrity refusal in a failed process launch.

Smart App Control, WDAC and AppLocker refuse through the same kernel path: a
"Bad Image" dialog names a dependent DLL, and no reinstall can repair it.
"""

from __future__ import annotations

import re


_REASON_SAC_OR_POLICY = "Smart App Control or an Application Control policy blocked the image"
_REASON_ADMIN_POLICY = "an Application Control policy blocked this program"
_REASON_SMART_APP_CONTROL = "Smart App Control blocked this program"
_REASON_INVALID_HASH_STATUS = (
    "the image failed code integrity validation (invalid or missing signature)"
)
_REASON_INVALID_HASH_WINERROR = "Windows could not verify the digital signature of the image"

# NTSTATUS refusals: SAC facility, INVALID_IMAGE_HASH, FAIL_FAST_EXCEPTION.
_BLOCK_STATUS_CODES = {
    0xC0E90002: _REASON_SAC_OR_POLICY,
    0xC0000428: _REASON_INVALID_HASH_STATUS,
    0xC0000602: "the image was refused by a code integrity fail-fast",
}

# winerror equivalents; CI_BLOCKED is from unslothai/unsloth#6648.
_BLOCK_WINERRORS = {
    577: _REASON_INVALID_HASH_WINERROR,
    1260: _REASON_ADMIN_POLICY,
    4551: "code integrity blocked the image",
}

# Windows reports 0xC0E90002 for a Smart App Control block AND for an admin's
# WDAC policy, so that status is ambiguous and must offer both remedies. Only
# winerror 1260 and the AppLocker / Group Policy wording positively identify an
# admin-owned policy, where turning off SAC fixes nothing and downgrades security.
_ADMIN_POLICY_REASONS = frozenset({_REASON_ADMIN_POLICY})
_SMART_APP_CONTROL_REASONS = frozenset({_REASON_SMART_APP_CONTROL})

# 0xC0000428 and winerror 577 report a HASH MISMATCH, not a policy verdict.
# Microsoft's own text for both is "signed incorrectly or damaged", and event
# 5038 names disk error and unauthorized modification beside it, so a truncated
# or damaged copy of an otherwise acceptable file lands here too -- and there
# replacing the file IS the remedy. These two must not deny corruption.
_INVALID_HASH_REASONS = frozenset(
    {_REASON_INVALID_HASH_STATUS, _REASON_INVALID_HASH_WINERROR}
)

_STATUS_TEXT_RE = re.compile(r"0x(c0e90002|c0000428|c0000602)\b", re.IGNORECASE)
_BAD_IMAGE_RE = re.compile(
    r"is either not designed to run on Windows or it contains an error", re.IGNORECASE
)
_SAC_TEXT_RE = re.compile(r"blocked by smart app control", re.IGNORECASE)
_ADMIN_POLICY_TEXT_RE = re.compile(
    r"(application control policy has blocked|blocked by group policy)",
    re.IGNORECASE,
)


def is_bad_image_text(error: object) -> bool:
    """True when Windows reported a Bad Image, whatever the cause."""
    text = error if isinstance(error, str) else str(error)
    return bool(text) and _BAD_IMAGE_RE.search(text) is not None


def code_integrity_block_reason(error: object) -> str | None:
    """Return a human reason when ``error`` is a code integrity refusal, else None."""
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
    if _SAC_TEXT_RE.search(text):
        return _REASON_SMART_APP_CONTROL
    if _ADMIN_POLICY_TEXT_RE.search(text):
        return _REASON_ADMIN_POLICY
    # "Bad Image" alone is NOT a block: Windows prints it for a corrupt or
    # wrong-architecture DLL too, where reinstalling IS the right remedy.
    return None


def code_integrity_user_message(binary: str, reason: str) -> str:
    if reason in _INVALID_HASH_REASONS:
        return (
            f"Windows refused to load part of the local model runtime: {reason}. "
            f"The refused file is under {binary}. "
            "Windows reports this both for a file a code integrity policy will not "
            "accept and for one that is damaged or was downloaded incompletely, so "
            "the error alone does not say which. Reinstalling the local runtime "
            "replaces the files and clears the damaged case. If it fails again after "
            "that, it is a policy: Smart App Control has no per-application exception "
            "and turning it off in Windows Security under App & browser control is "
            "the only local workaround, while on a device managed by an administrator "
            "the policy is theirs to change."
        )
    opening = (
        f"Windows blocked part of the local model runtime: {reason}. "
        f"The blocked file is under {binary}. "
        "This is a Windows code integrity policy refusing to load code it does not "
        "recognise, not a corrupt download, so reinstalling or running as "
        "administrator will not clear it. "
    )
    if reason in _ADMIN_POLICY_REASONS:
        return opening + (
            "The policy is set by whoever administers this device (AppLocker, WDAC "
            "or Group Policy) and can only be changed there, so ask them to allow "
            "the files in that folder. Turning off Smart App Control does not "
            "affect an administrator policy."
        )
    if reason in _SMART_APP_CONTROL_REASONS:
        return opening + (
            "Smart App Control has no per-application exception; turning it off in "
            "Windows Security under App & browser control is the only local workaround."
        )
    # Ambiguous status: name both remedies rather than asserting either.
    return opening + (
        "If Smart App Control is on, it has no per-application exception and turning "
        "it off in Windows Security under App & browser control is the only local "
        "workaround. If this device is managed by an administrator, the policy is "
        "theirs to change and turning off Smart App Control will not help."
    )
