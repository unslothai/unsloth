# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared assertions for packaged Linux AppImage smoke tests."""

from __future__ import annotations

from pathlib import Path


LOADER_ERROR_MARKERS = (
    "Failed to load module:",
    "undefined symbol:",
)


def assert_no_loader_errors(*logs: Path) -> None:
    """Reject mixed host/bundle loader failures even when the UI path succeeds."""

    failures: list[str] = []
    for log in logs:
        if not log.is_file():
            continue
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if any(marker in line for marker in LOADER_ERROR_MARKERS):
                failures.append(f"{log.name}: {line}")
    if failures:
        raise RuntimeError("AppImage mixed host and bundled runtime libraries:\n" + "\n".join(failures))
