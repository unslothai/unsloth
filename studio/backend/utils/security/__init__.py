# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Security helpers for the ``trust_remote_code`` boundary.

Two orthogonal questions: ``trusted_org.is_trusted_org_repo`` (may we AUTO-enable
remote code for this name?) and ``remote_code_scan`` (WHAT would run if the user
opts in?). The load paths try ``trust_remote_code=False`` first and, on the
transformers "requires trust_remote_code" error, scan the repo's ``auto_map``,
surface findings + a pinning fingerprint, and require explicit consent before
retrying with it enabled. Detection (is-vision / version / size) reads raw
``config.json`` and never enters this flow.
"""

from utils.security.consent import (  # noqa: F401
    RemoteCodeDecision,
    evaluate_remote_code_consent,
    evaluate_remote_code_consent_for_targets,
)
from utils.security.file_security import (  # noqa: F401
    FileSecurityDecision,
    evaluate_file_security,
    record_embedding_verdict,
    security_load_subdirs,
)
from utils.security.remote_code_scan import (  # noqa: F401
    CRITICAL,
    HIGH,
    MEDIUM,
    Finding,
    RemoteCodeUnscannable,
    ScanResult,
    remote_code_fingerprint,
    repo_remote_code_files,
    scan_remote_code_files,
)
from utils.security.trusted_org import is_trusted_org_repo  # noqa: F401

__all__ = [
    "is_trusted_org_repo",
    "scan_remote_code_files",
    "repo_remote_code_files",
    "RemoteCodeUnscannable",
    "remote_code_fingerprint",
    "should_block_remote_code",
    "evaluate_remote_code_consent",
    "evaluate_remote_code_consent_for_targets",
    "preflight_remote_code_consent",
    "preflight_remote_code_consent_for_targets",
    "evaluate_file_security",
    "record_embedding_verdict",
    "security_load_subdirs",
    "FileSecurityDecision",
    "RemoteCodeDecision",
    "ScanResult",
    "Finding",
    "CRITICAL",
    "HIGH",
    "MEDIUM",
]


def preflight_remote_code_consent(
    model_name: str,
    hf_token = None,
    *,
    trust_remote_code: bool = True,
    approved_fingerprint = None,
    trusted_org = None,
    subject = None,
) -> "RemoteCodeDecision":
    """Scan a model's ``auto_map`` for the consent dialog. Thin wrapper over
    ``evaluate_remote_code_consent`` defaulting ``trust_remote_code=True`` so the scan
    runs whenever the repo declares custom code; the start routes pass the user's real
    value + approved fingerprint to enforce consent before any state mutation.
    """
    return evaluate_remote_code_consent(
        model_name,
        hf_token,
        trust_remote_code = trust_remote_code,
        approved_fingerprint = approved_fingerprint,
        trusted_org = trusted_org,
        subject = subject,
    )


def preflight_remote_code_consent_for_targets(
    targets,
    hf_token = None,
    *,
    trust_remote_code: bool = True,
    approved_fingerprint = None,
    subject = None,
) -> "RemoteCodeDecision":
    """Preflight consent over multiple repos (a LoRA adapter plus its base) scanned as
    one combined unit with a single pinning fingerprint. Wrapper defaulting
    ``trust_remote_code=True``; the load passes the user's real value + fingerprint.
    """
    return evaluate_remote_code_consent_for_targets(
        targets,
        hf_token,
        trust_remote_code = trust_remote_code,
        approved_fingerprint = approved_fingerprint,
        subject = subject,
    )


def should_block_remote_code(result: "ScanResult") -> bool:
    """Recommend blocking by default on CRITICAL/HIGH findings. Advisory only: the
    caller still surfaces findings and takes explicit consent.
    """
    sev = result.max_severity
    return sev in (CRITICAL, HIGH)
