#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Create a 100,000-path fixture and enforce repository-map responsiveness."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.agent_workspace.discovery import build_repository_map


def _create_fixture(root: Path, path_count: int) -> None:
    files_per_directory = 1_000
    for start in range(0, path_count, files_per_directory):
        directory = root / f"bucket-{start // files_per_directory:04}"
        directory.mkdir()
        for index in range(start, min(start + files_per_directory, path_count)):
            descriptor = os.open(
                directory / f"file-{index:06}.txt",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            try:
                os.write(descriptor, b"x")
            finally:
                os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type = int, default = 100_000)
    parser.add_argument("--max-paths", type = int, default = 20_000)
    parser.add_argument("--max-seconds", type = float, default = 30.0)
    arguments = parser.parse_args()
    if arguments.paths < 100_000:
        parser.error("--paths must retain the 100,000-path release fixture")
    if not 0 < arguments.max_paths < arguments.paths:
        parser.error("--max-paths must be positive and lower than --paths")

    with tempfile.TemporaryDirectory(prefix = "unsloth-repository-map-") as temporary:
        root = Path(temporary)
        _create_fixture(root, arguments.paths)
        started = time.monotonic()
        result = build_repository_map(
            root,
            max_paths = arguments.max_paths,
            max_total_bytes = arguments.max_paths,
            max_file_bytes = 1,
        )
        elapsed = time.monotonic() - started

    assert result["source"] == "filesystem"
    assert result["pathsScanned"] == arguments.max_paths
    assert result["fileCount"] <= arguments.max_paths
    assert result["bytesIncluded"] <= arguments.max_paths
    assert result["truncated"] is True
    assert "path-limit" in result["truncationReasons"]
    assert elapsed <= arguments.max_seconds
    print(
        json.dumps(
            {
                "fixturePaths": arguments.paths,
                "maxPaths": arguments.max_paths,
                "pathsScanned": result["pathsScanned"],
                "filesReturned": result["fileCount"],
                "bytesIncluded": result["bytesIncluded"],
                "elapsedSeconds": round(elapsed, 3),
                "truncationReasons": result["truncationReasons"],
            },
            sort_keys = True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
