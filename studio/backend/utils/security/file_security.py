# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Malware / unsafe-file gate for model loads.

The ``trust_remote_code`` consent gate (``consent.py``) covers one load-time RCE
vector: a repo's ``auto_map`` Python. It does NOT cover the other one -- a
malicious *pickle* inside a weight file (``pytorch_model.bin``, ``*.pkl``,
``*.dat``), which executes during ``from_pretrained`` deserialization *even with*
``trust_remote_code=False``. A repo with a normal ``config.json`` plus a poisoned
pickle would slip straight past the remote-code gate.

This module closes that gap using Hugging Face's OWN security scan (picklescan +
ClamAV), which the Hub runs on every repo and exposes as metadata via
``HfApi.model_info(repo, securityStatus=True).security_repo_status``::

    {"scansDone": true, "filesWithIssues": [{"path": "x.pkl", "level": "unsafe"}, ...]}

The check is strictly METADATA-ONLY: it reads the Hub's scan verdict and surfaces
the flagged file names + levels. It NEVER downloads, opens, ``torch.load``s, or
unpickles the flagged files.

Policy (confirmed product decisions):
  * Hard block, non-approvable -- a flagged file cannot be overridden by the user.
  * Fail-open -- block only when the Hub reports ``scansDone`` true AND a blocking
    level is present. Missing scan / ``scansDone`` false / offline / any error =>
    allow (subprocess isolation + transformers ``weights_only`` remain the
    backstop), matching the remote-code gate's "unscannable => allow".
  * No first-party exemption -- a poisoned pickle in a compromised trusted repo is
    exactly the CRITICAL-class threat we block even for first parties.
  * Local paths / GGUF are skipped (no Hub scan exists; GGUF is not a pickle).
"""

from dataclasses import dataclass, field
from typing import Optional

from loggers import get_logger

logger = get_logger(__name__)

# Hub security levels that mean "do not deserialize this file".
_BLOCKING_LEVELS = frozenset({"malicious", "unsafe", "suspicious"})

# Two-timeout metadata fetch, mirroring hub.workers.hf_download._retry_metadata_fetch.
_REQUEST_TIMEOUT = 10.0
_RETRY_TIMEOUT = 20.0


@dataclass
class FileSecurityDecision:
    """Outcome of the Hub security scan for one model repo."""

    model_name: str
    blocked: bool
    unsafe_files: list = field(default_factory = list)  # [{"path", "level"}]
    reason: str = ""

    def response_payload(self) -> dict:
        """Machine-readable detail merged into the preflight payload the dialog reads."""
        return {
            "unsafe_files": self.unsafe_files,
            "security_blocked": self.blocked,
            "reason": self.reason,
        }


def _is_gguf(model_name: str) -> bool:
    return model_name.lower().endswith(".gguf")


def _fetch_security_status(model_name: str, hf_token: Optional[str]):
    """Return ``security_repo_status`` (a dict) or ``None`` if unavailable.

    Reads Hub metadata only (no file download). Retries once on a transient error,
    then gives up and returns ``None`` so the caller fails open.
    """
    from huggingface_hub import model_info as hf_model_info

    token_arg = hf_token if hf_token else False
    last_exc = None
    for attempt, timeout in enumerate((_REQUEST_TIMEOUT, _RETRY_TIMEOUT)):
        try:
            info = hf_model_info(
                model_name,
                token = token_arg,
                securityStatus = True,
                timeout = timeout,
            )
            return getattr(info, "security_repo_status", None)
        except Exception as exc:  # network/offline/gated/404/unsupported-client
            last_exc = exc
            if attempt == 0:
                continue
    logger.debug(
        "HF security scan unavailable for '%s' (%s); failing open.",
        model_name,
        type(last_exc).__name__ if last_exc else "unknown",
    )
    return None


def evaluate_file_security(model_name: str, hf_token: Optional[str] = None) -> FileSecurityDecision:
    """Block a load when Hugging Face's security scan flags unsafe serialized files.

    Call this UNCONDITIONALLY before any load (independent of trust_remote_code),
    because a malicious pickle deserializes during ``from_pretrained`` regardless.
    Metadata-only: never touches the flagged file bytes. Fails open when the scan
    cannot be obtained.
    """
    # Local folders and GGUF files have no Hub security scan; nothing to check.
    try:
        from utils.paths import is_local_path
        if is_local_path(model_name) or _is_gguf(model_name):
            return FileSecurityDecision(model_name, False, reason = "local or gguf; no Hub scan")
    except Exception:
        # If we cannot even classify the path, do not block on that account.
        return FileSecurityDecision(model_name, False, reason = "path check failed; not blocked")

    status = _fetch_security_status(model_name, hf_token)
    if not isinstance(status, dict):
        return FileSecurityDecision(
            model_name, False, reason = "scan unavailable; allowed (fail-open)"
        )

    # Block on any file the Hub has ALREADY flagged with a blocking level. We do
    # NOT gate on ``scansDone``: it is frequently false even for fully-clean
    # safetensors repos, and a file already flagged ``unsafe`` is unsafe whether or
    # not the repo's remaining files have finished scanning. ``scansDone`` only
    # tells us how complete the scan is; an unavailable status (above) is the sole
    # fail-open path.
    unsafe = []
    for entry in status.get("filesWithIssues") or []:
        if not isinstance(entry, dict):
            continue
        level = str(entry.get("level", "")).lower()
        if level in _BLOCKING_LEVELS:
            unsafe.append({"path": entry.get("path", ""), "level": level})

    if not unsafe:
        return FileSecurityDecision(model_name, False, reason = "no unsafe files")

    names = ", ".join(u["path"] for u in unsafe if u["path"]) or "unknown files"
    logger.warning(
        "Blocking load of '%s': Hugging Face security scan flagged unsafe files (%s).",
        model_name,
        names,
    )
    return FileSecurityDecision(
        model_name,
        True,
        unsafe_files = unsafe,
        reason = f"Hugging Face security scan flagged unsafe files: {names}",
    )
