# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared assertions for packaged Linux AppImage smoke tests."""

from __future__ import annotations

from pathlib import Path


LOADER_ERROR_MARKERS = (
    "Failed to load module:",
    "undefined symbol:",
)

# What a fixture reports as the managed backend's version.
# Not MIN_DESKTOP_BACKEND_VERSION, which is only the right answer for a build made here.
# The floor is not the only gate: managed_backend_version_stale_reason() compares against expected_backend_version(),
# which is option_env!("UNSLOTH_DESKTOP_BACKEND_VERSION").unwrap_or(MIN_DESKTOP_BACKEND_VERSION) and the release
# workflow stamps that from the same pypi_version as the updater manifest.
# v0.1.804-beta is pinned to backend 2026.8.22 and read the floor as `desktop_backend_version_outdated`, so it ran its
# bundled installer instead of doing what the test was watching for.
# Every AppImage lane of the nightly failed on that while the pull_request lane, testing an unstamped build of the same
# commit, passed.
# It lives here rather than in either test because #10001 fixed only the portability smoke and the model-download E2E
# kept failing for exactly this.
FIXTURE_BACKEND_VERSION = "9999.12.31"


def assert_fixture_version_clears_floor(repo_root: Path) -> None:
    """Fail by name if the floor ever catches up with the sentinel.

    Compared on the leading component: a string compare would read "999" as above
    "9999", which is the one answer this must not get wrong.
    """
    source = (repo_root / "studio/src-tauri/src/preflight/version.rs").read_text(encoding = "utf-8")
    marker = 'MIN_DESKTOP_BACKEND_VERSION: &str = "'
    start = source.find(marker)
    if start < 0:
        raise RuntimeError("Could not read the minimum desktop backend version")
    start += len(marker)
    floor = source[start : source.index('"', start)]
    if int(floor.split(".")[0]) >= int(FIXTURE_BACKEND_VERSION.split(".")[0]):
        raise RuntimeError(
            f"fixture version {FIXTURE_BACKEND_VERSION} no longer clears the floor {floor}"
        )


def assert_no_loader_errors(*logs: Path) -> None:
    """Reject mixed host/bundle loader failures even when the UI path succeeds."""

    failures: list[str] = []
    for log in logs:
        if not log.is_file():
            continue
        for line in log.read_text(encoding = "utf-8", errors = "replace").splitlines():
            if any(marker in line for marker in LOADER_ERROR_MARKERS):
                failures.append(f"{log.name}: {line}")
    if failures:
        raise RuntimeError(
            "AppImage mixed host and bundled runtime libraries:\n" + "\n".join(failures)
        )
