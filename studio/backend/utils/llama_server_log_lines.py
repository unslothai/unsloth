# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Classify llama-server stdout lines for backend log level.

At default LOG_LEVEL=INFO, tensor-load spam stays at DEBUG while failures,
warnings, and readiness stay visible in the server session log (#10793).
"""

from __future__ import annotations


def llama_server_line_uses_info_level(line: str) -> bool:
    """True when a llama-server line should use logger.info in the backend stream."""
    if not line:
        return False
    lower = line.lower()
    if "server is listening" in lower or "model loaded" in lower:
        return True
    if lower.startswith(("main:", "llama_", "system_info", "build:")):
        return True
    # High-volume progress during weight load; the dedicated llama-*.log keeps the full tee.
    if "%" in line and any(token in lower for token in ("load", "offload", "progress", "tensor")):
        return False
    markers = (
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
        "cuda",
        "ggml",
        "out of memory",
        "oom",
        "killed",
        "overflow",
        "refused",
        "timed out",
        "timeout",
        "exited",
        "signal",
    )
    return any(token in lower for token in markers)
