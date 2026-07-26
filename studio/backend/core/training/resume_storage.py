# SPDX-License-Identifier: AGPL-3.0-only
"""Crash-safe staging and publication of resumed checkpoints."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable

COMPLETE_MARKER = ".unsloth-sync-complete"


def _copy_tree_atomic(source: Path, destination: Path, progress: Callable[[str], None]) -> Path:
    """Copy a directory and publish it only after every byte is durable."""
    destination.parent.mkdir(parents = True, exist_ok = True)
    partial = Path(tempfile.mkdtemp(prefix = f".{destination.name}.partial-", dir = destination.parent))
    try:
        progress(f"Copying checkpoint to local storage: {source.name}")
        shutil.copytree(source, partial / "payload", symlinks = False)
        payload = partial / "payload"
        (payload / COMPLETE_MARKER).write_text("complete\n", encoding = "utf-8")
        if destination.exists():
            raise FileExistsError(f"Refusing to replace existing checkpoint: {destination}")
        os.replace(payload, destination)
        progress(f"Checkpoint copy complete: {source.name}")
        return destination
    finally:
        shutil.rmtree(partial, ignore_errors = True)


def stage_checkpoint(source: str, working_root: str | None, progress: Callable[[str], None]) -> str:
    root = Path(working_root) if working_root else Path(tempfile.gettempdir()) / "unsloth-resume"
    root.mkdir(parents = True, exist_ok = True)
    destination = root / f"{Path(source).name}-{os.getpid()}"
    return str(_copy_tree_atomic(Path(source), destination, progress))


def synchronize_checkpoints(local_output: str, persistent_output: str, progress: Callable[[str], None]) -> None:
    """Atomically publish completed checkpoint directories to persistent storage."""
    source_root, destination_root = Path(local_output), Path(persistent_output)
    destination_root.mkdir(parents = True, exist_ok = True)
    for source in sorted(source_root.glob("checkpoint-*")):
        if not source.is_dir():
            continue
        destination = destination_root / source.name
        if (destination / COMPLETE_MARKER).is_file():
            continue
        progress(f"Synchronizing checkpoint: {source.name}")
        _copy_tree_atomic(source, destination, progress)


def latest_synchronized_checkpoint(output_dir: str) -> str | None:
    """Return the newest atomically finalized checkpoint, ignoring partial copies."""
    def step(path: Path) -> int:
        try:
            return int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            return -1
    valid = [p for p in Path(output_dir).glob("checkpoint-*") if (p / COMPLETE_MARKER).is_file()]
    return str(max(valid, key = step)) if valid else None
