# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Classify llama-server stdout lines for backend log level.

At default LOG_LEVEL=INFO, tensor-load spam stays at DEBUG while failures,
warnings, and readiness stay visible in the server session log (#10793).
"""

from __future__ import annotations

# Substrings that mark a line as worth INFO in the session log (not routine load).
_FAILURE_MARKERS = (
    "error",
    "fail",
    "fatal",
    "panic",
    "abort",
    "exception",
    "cannot ",
    "can't ",
    "unable",
    "invalid",
    "warning",
    " warn:",
    "out of memory",
    "oom",
    "killed",
    "overflow",
    "refused",
    "timed out",
    "timeout",
    "exited",
    "signal",
    "not found",
    "not loaded",
)


def _line_looks_like_failure(lower: str) -> bool:
    if any(token in lower for token in _FAILURE_MARKERS):
        return True
    if "cuda" in lower or "ggml" in lower:
        return any(
            token in lower
            for token in (
                "error",
                "fail",
                "fatal",
                "warning",
                "invalid",
                "unable",
                "cannot",
                "can't",
                "not found",
                "not loaded",
            )
        )
    return False


def llama_server_line_uses_info_level(line: str) -> bool:
    """True when a llama-server line should use logger.info in the backend stream."""
    if not line:
        return False
    lower = line.lower()
    if "server is listening" in lower or "model loaded" in lower:
        return True
    # High-volume progress during weight load; the dedicated llama-*.log keeps the full tee.
    if "%" in line and any(token in lower for token in ("load", "offload", "progress", "tensor")):
        return False
    # Routine metadata spam during load (dozens of lines per attempt).
    if "llama_model_loader:" in lower:
        return _line_looks_like_failure(lower)
    if lower.startswith("build:") or lower.startswith("system_info:"):
        return True
    return _line_looks_like_failure(lower)
