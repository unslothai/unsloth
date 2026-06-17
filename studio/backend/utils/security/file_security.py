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
  * Block whenever ``filesWithIssues`` lists a blocking level, regardless of
    ``scansDone``: ``scansDone`` is frequently false even for fully-clean
    safetensors repos, and a file the Hub has ALREADY flagged ``unsafe`` is unsafe
    whether or not the rest of the scan has finished. Unknown/future non-``safe``
    levels fail CLOSED (block) so Hub schema drift cannot silently allow a
    newly-named bad verdict; only a small allowlist of "clean / not-yet-scanned"
    levels is treated as non-blocking. The ONLY fail-open path is an unavailable
    status (missing scan field / offline / any error) => allow (subprocess
    isolation + transformers ``weights_only`` remain the backstop).
  * Scope the block to the LOAD-PATH RCE vector. The gate exists to stop a
    malicious PICKLE deserializing during ``from_pretrained``; that loader reads
    weight files at the repo ROOT. A flag only blocks when it lands on a
    root-level file in a code-executing format. Two exclusions, because they are
    NOT an RCE vector for a Studio load:
      - Non-pickle formats can't execute: ``.safetensors`` is tensor-only by
        design, ``.gguf`` is a non-pickle llama.cpp format, configs/text/images
        cannot deserialize code. (Hub sometimes flags a repo's safetensors at
        ``unsafe`` when picklescan trips on a *sibling* pickle; that safetensors
        still cannot run code.)
      - Files in SUBDIRECTORIES are not read by ``from_pretrained`` (e.g. NeMo
        ``nemo/weights/*.distcp`` / ``common.pt``, ONNX/TF exports), so a flag
        there is never loaded. This keeps real malware blocked (eicar ships its
        ``*.pkl`` / ``*.dat`` / ``eicar_test_file`` at the repo root) while not
        false-blocking legitimate first-party repos like
        ``nvidia/Nemotron-H-8B-Base-8K`` (root safetensors + flagged NeMo pickle
        checkpoints in ``nemo/`` that the loader never touches).
  * No first-party exemption -- a poisoned pickle in a compromised trusted repo is
    exactly the CRITICAL-class threat we block even for first parties. (The scoping
    above is by load-path/format, not by org, so a root pickle in an
    ``unsloth``/``nvidia`` repo still blocks.)
  * Local paths are skipped (no Hub scan exists; a local ``.gguf`` lands here too).
    A REMOTE reference is always scanned, even one whose name ends in ``.gguf``, so
    a repo cannot dodge the scan by suffixing its name (``evil/model.gguf``); a
    reference that is not a real repo id simply fails open.
"""

from dataclasses import dataclass, field
from typing import Optional

from loggers import get_logger

logger = get_logger(__name__)

# Scan levels that are NOT a block: clean, or a not-yet-finished/non-verdict state.
# ANYTHING else (unsafe/suspicious/malicious, or a future label like "infected")
# is treated as blocking, so Hub schema drift fails CLOSED instead of silently
# allowing a newly-named bad verdict. "pending"/"scanning"/"error"/"" stay
# non-blocking so an in-progress or errored per-file scan does not false-block.
_NONBLOCKING_LEVELS = frozenset(
    {"", "safe", "pending", "scanning", "queued", "unscanned", "error", "unknown", "none"}
)

# File suffixes that cannot execute code when a model loads them, so a flag on one
# is never an RCE vector. safetensors is tensor-only by design; gguf is a non-pickle
# llama.cpp format; the rest are text/markup/images that no loader deserializes.
_INERT_SUFFIXES = frozenset(
    {
        ".safetensors",
        ".gguf",
        ".json",
        ".txt",
        ".md",
        ".rst",
        ".yaml",
        ".yml",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".svg",
        ".bmp",
        ".gitattributes",
        ".gitignore",
    }
)


def _is_load_rce_vector(path: str) -> bool:
    """Whether a flagged repo file is one a model load could DESERIALIZE as a pickle.

    ``from_pretrained`` reads weight files at the repo ROOT (root
    ``pytorch_model*.bin`` / ``*.pkl`` / sharded bins). It does not read pickle
    artifacts in SUBDIRECTORIES (NeMo ``nemo/weights/*.distcp``, ONNX/TF exports)
    nor non-pickle formats (safetensors/gguf cannot execute). So a flag is a real
    load-path RCE vector only when it lands on a root-level, non-inert file.
    """
    p = (path or "").strip()
    while p.startswith("./"):
        p = p[2:]
    if not p or "/" in p:
        return False  # subdirectory artifact (or empty): not read by from_pretrained
    suffix = "." + p.rsplit(".", 1)[1].lower() if "." in p else ""
    return suffix not in _INERT_SUFFIXES


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
    # Local folders/files (including a local .gguf) have no Hub security scan;
    # nothing to check. A REMOTE reference is scanned even if it ends in .gguf, so a
    # repo cannot evade the scan by naming itself "*.gguf".
    try:
        from utils.paths import is_local_path
        if is_local_path(model_name):
            return FileSecurityDecision(model_name, False, reason = "local path; no Hub scan")
    except Exception:
        # If we cannot even classify the path, do not block on that account.
        return FileSecurityDecision(model_name, False, reason = "path check failed; not blocked")

    status = _fetch_security_status(model_name, hf_token)
    if not isinstance(status, dict):
        return FileSecurityDecision(
            model_name, False, reason = "scan unavailable; allowed (fail-open)"
        )

    # Block on a file the Hub has flagged with a non-``safe`` level, scoped to the
    # actual load-path RCE vector (root-level, code-executing format). We do NOT
    # gate on ``scansDone`` (frequently false even for clean repos; an already-
    # flagged file is flagged regardless). Unknown levels fail closed (block);
    # in-progress/clean levels do not. A flag on a subdirectory pickle or an inert
    # format (safetensors/gguf) is not loaded by ``from_pretrained`` and does not
    # block. An unavailable status (above) is the sole fail-open path.
    unsafe = []
    skipped = []  # flagged, but not a load-path RCE vector (subdir / inert format)
    for entry in status.get("filesWithIssues") or []:
        if not isinstance(entry, dict):
            continue
        level = str(entry.get("level", "")).lower()
        if level in _NONBLOCKING_LEVELS:
            continue
        path = entry.get("path", "")
        if _is_load_rce_vector(path):
            unsafe.append({"path": path, "level": level})
        else:
            skipped.append({"path": path, "level": level})

    if not unsafe:
        if skipped:
            # The repo has flagged files, but only ones the model load never
            # deserializes (a subdirectory pickle, or an inert format like a
            # safetensors). Not an RCE vector for a from_pretrained load -> allow,
            # but log the flagged-but-not-loaded files so they remain visible.
            logger.info(
                "'%s': Hugging Face flagged files, but none are a load-path RCE "
                "vector (subdir/inert); allowing the load. Flagged: %s",
                model_name,
                ", ".join(f"{s['path']}({s['level']})" for s in skipped),
            )
        return FileSecurityDecision(model_name, False, reason = "no unsafe files in the load path")

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
