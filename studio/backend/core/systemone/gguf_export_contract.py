# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""On-disk contract between decision-model GGUF export and native serving.

Standard library only: Unsloth's save path loads this file by path, without importing Studio.
"""

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Optional

EXPORT_DIR = "gguf"
EXPORT_FILE = "export.json"
FORMAT = "unsloth-decision-gguf"
VERSION = 1
LAYOUTS = ("clef", "laya")

_CLEF_FILES = ("unsloth_decision_config.json", "joint_head_config.json", "joint_head.safetensors")
_LAYA_FILES = ("rl_agent_config.json",)


def _fingerprint_files(folder: Path, layout: str) -> list:
    if layout == "clef":
        return [folder / name for name in _CLEF_FILES]
    if layout == "laya":
        weights = sorted(folder.glob("model*.safetensors")) + sorted(
            folder.glob("model.safetensors.index.json")
        )
        return [folder / name for name in _LAYA_FILES] + weights
    raise ValueError(f"unknown decision layout {layout!r}")


def fingerprint(folder, layout: str) -> str:
    """sha256 over the files that decide what a decision GGUF serves."""
    folder = Path(folder)
    digest = hashlib.sha256()
    for path in _fingerprint_files(folder, layout):
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing, so {folder} cannot be fingerprinted")
        digest.update(path.name.encode("utf-8") + b"\0")
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _valid(data) -> bool:
    if not isinstance(data, dict):
        return False
    if data.get("format") != FORMAT or data.get("version") != VERSION:
        return False
    if data.get("layout") not in LAYOUTS or not isinstance(data.get("source_fingerprint"), str):
        return False
    files = data.get("files")
    quants = data.get("quantizations")
    if not isinstance(files, dict) or not isinstance(quants, list) or not quants:
        return False
    for quant in quants:
        entry = files.get(quant)
        if not isinstance(entry, dict) or not isinstance(entry.get("model"), str):
            return False
        if entry.get("mmproj") is not None and not isinstance(entry.get("mmproj"), str):
            return False
    return True


def read_export(folder) -> Optional[dict]:
    """export.json of a run folder (or of its gguf/ directory), or None if absent or invalid."""
    folder = Path(folder)
    path = folder / EXPORT_FILE if folder.name == EXPORT_DIR else folder / EXPORT_DIR / EXPORT_FILE
    try:
        data = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None
    return data if _valid(data) else None


def write_export(export_dir, layout: str, files: dict, source_fingerprint: str) -> dict:
    """Writes export.json atomically; call it after every GGUF it names is in place."""
    export_dir = Path(export_dir)
    data = {
        "format": FORMAT,
        "version": VERSION,
        "layout": layout,
        "quantizations": list(files),
        "files": {
            quant: {"model": entry["model"], "mmproj": entry.get("mmproj")}
            for quant, entry in files.items()
        },
        "source_fingerprint": source_fingerprint,
    }
    if not _valid(data):
        raise ValueError(f"invalid decision GGUF export: {data}")
    # open(..., "x") rather than mkstemp: the file gets the umask's mode, not 0600.
    tmp = export_dir / f".export-{os.getpid()}-{uuid.uuid4().hex[:8]}.json"
    try:
        with open(tmp, "x", encoding = "utf-8") as f:
            json.dump(data, f, indent = 2)
        os.replace(tmp, export_dir / EXPORT_FILE)
    except BaseException:
        Path(tmp).unlink(missing_ok = True)
        raise
    return data
