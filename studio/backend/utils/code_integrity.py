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


_REASON_SAC_OR_POLICY = "Smart App Control or an Application Control policy blocked the image"
_REASON_ADMIN_POLICY = "an Application Control policy blocked this program"
_REASON_SMART_APP_CONTROL = "Smart App Control blocked this program"

# Statuses Windows reports when code integrity refuses an image.
#
# 0xC0E90002  the Smart App Control / system integrity policy facility, which is
#             what the "Bad Image" dialog shows for a SAC block
# 0xC0000428  STATUS_INVALID_IMAGE_HASH, an unsigned or tampered image under an
#             enforced policy
# 0xC0000602  STATUS_FAIL_FAST_EXCEPTION raised from a code integrity failure
_BLOCK_STATUS_CODES = {
    0xC0E90002: _REASON_SAC_OR_POLICY,
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
    1260: _REASON_ADMIN_POLICY,
    4551: "code integrity blocked the image",
}

# Which remedy each reason earns. Windows reports 0xC0E90002 for a Smart App
# Control block AND for an administrator's WDAC policy, so most classifications
# are genuinely ambiguous and have to offer both remedies. Only ERROR_ACCESS_
# DISABLED_BY_POLICY (1260) and the AppLocker / Group Policy wording positively
# identify a policy some administrator owns, and there turning off Smart App
# Control changes nothing: it is a different feature, it is normally already off
# on a managed device, and switching it off is a security downgrade the user
# cannot necessarily undo. So that case is sent to the administrator instead.
_ADMIN_POLICY_REASONS = frozenset({_REASON_ADMIN_POLICY})
_SMART_APP_CONTROL_REASONS = frozenset({_REASON_SMART_APP_CONTROL})

# Matches the status in text form wherever it reaches us as a string: a child's
# stderr, a Rust-side error, or the repr of an exception we did not raise.
_STATUS_TEXT_RE = re.compile(r"0x(c0e90002|c0000428|c0000602)\b", re.IGNORECASE)
# Kept, but deliberately NOT a classifier on its own; see the end of
# code_integrity_block_reason. Exposed so a caller can say "this was an image
# load failure" without claiming to know why.
_BAD_IMAGE_RE = re.compile(
    r"is either not designed to run on Windows or it contains an error", re.IGNORECASE
)
# Split, because the two phrasings earn different advice: one names Smart App
# Control, the others name a policy only an administrator can change.
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
    if _SAC_TEXT_RE.search(text):
        return _REASON_SMART_APP_CONTROL
    if _ADMIN_POLICY_TEXT_RE.search(text):
        return _REASON_ADMIN_POLICY
    # "Bad Image" alone is NOT enough. Windows prints that same sentence for a
    # genuinely corrupt DLL, one built for another architecture, and one whose
    # own dependencies are missing. Those are ordinary broken installs, and the
    # remedy for them (reinstall) is the exact opposite of the advice this
    # module exists to give. So the wording only counts when it arrives with a
    # code integrity status or a policy phrase, both handled above; on its own
    # it is left unclassified rather than reported as a policy block.
    return None


def code_integrity_user_message(binary: str, reason: str) -> str:
    """The message to show a user whose llama.cpp runtime will not load.

    The workaround sentence follows the policy the classifier actually
    identified. Naming Smart App Control at an AppLocker/WDAC block sends the
    user to turn off an unrelated feature that will not unblock anything.
    """
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
    # Ambiguous: the same status covers both, so name both remedies rather than
    # asserting one of them is the only one.
    return opening + (
        "If Smart App Control is on, it has no per-application exception and turning "
        "it off in Windows Security under App & browser control is the only local "
        "workaround. If this device is managed by an administrator, the policy is "
        "theirs to change and turning off Smart App Control will not help."
    )
