# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""On-disk contract for a decision fine-tune's GGUF export, shared by the exporter and the Decision API.

An export lives in ``<run folder>/gguf/``: ``model-<QUANT>.gguf``, an optional ``mmproj-<QUANT>.gguf``
and ``export.json``::

    {"format": "unsloth-decision-gguf", "version": 1, "layout": "clef" | "laya",
     "quantizations": ["Q8_0", ...],
     "files": {"Q8_0": {"model": "model-Q8_0.gguf", "mmproj": "mmproj-Q8_0.gguf" | null}},
     "source_fingerprint": fingerprint(run folder, layout)}

The fingerprint covers what the served probabilities depend on, so retraining or recalibrating the
folder makes an older export stale.
"""

from __future__ import annotations

import hashlib
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EXPORT_DIR = "gguf"
EXPORT_FILE = "export.json"
FORMAT = "unsloth-decision-gguf"
VERSION = 1
LAYOUTS = ("clef", "laya")
FINGERPRINT_FILES = {
    "clef": ("unsloth_decision_config.json", "joint_head_config.json", "joint_head.safetensors"),
    "laya": ("rl_agent_config.json", "model.safetensors"),
}
# Served quantization, best first; any other listed one after these.
PREFERRED = ("Q8_0", "F16", "BF16")
_CHUNK = 1 << 20
_stale_logged: set[tuple[str, str]] = set()


@lru_cache(maxsize = 64)
def _file_digest(path: str, size: int, mtime_ns: int) -> bytes:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while chunk := source.read(_CHUNK):
            digest.update(chunk)
    return digest.digest()


def fingerprint(folder: str | Path, layout: str) -> str:
    """sha256 over each fingerprint file as name, NUL, then its sha256 (or ``missing``), in order."""
    if layout not in FINGERPRINT_FILES:
        raise ValueError(f"Unknown decision layout: {layout!r}")
    folder = Path(folder)
    digest = hashlib.sha256()
    for name in FINGERPRINT_FILES[layout]:
        digest.update(name.encode() + b"\0")
        path = folder / name
        try:
            stat = path.stat()
            digest.update(_file_digest(str(path.resolve()), stat.st_size, stat.st_mtime_ns))
        except OSError:
            digest.update(b"missing")
    return digest.hexdigest()


def _plain_name(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value == Path(value).name
        and value not in (".", "..")
        and "\\" not in value
    )


def read_export(folder: str | Path) -> dict[str, Any] | None:
    """The folder's ``gguf/export.json`` if it is well formed and names files that exist, else None."""
    directory = Path(folder) / EXPORT_DIR
    try:
        data = json.loads((directory / EXPORT_FILE).read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("format") != FORMAT or data.get("version") != VERSION:
        return None
    if data.get("layout") not in LAYOUTS or not isinstance(data.get("source_fingerprint"), str):
        return None
    files, quantizations = data.get("files"), data.get("quantizations")
    if not isinstance(files, dict) or not isinstance(quantizations, list):
        return None
    for quant in quantizations:
        entry = files.get(quant) if isinstance(quant, str) else None
        if not isinstance(entry, dict) or not _plain_name(entry.get("model")):
            return None
        mmproj = entry.get("mmproj")
        if mmproj is not None and not _plain_name(mmproj):
            return None
    return data


def served_files(folder: str | Path, layout: str) -> tuple[str, Path, Path | None] | None:
    """Current export's (quantization, model, mmproj) or None; a stale fingerprint is ignored."""
    folder = Path(folder)
    data = read_export(folder)
    if data is None or data["layout"] != layout:
        return None
    try:
        current = fingerprint(folder, layout)
    except OSError:
        return None
    if data["source_fingerprint"] != current:
        key = (str(folder), data["source_fingerprint"])
        if key not in _stale_logged:
            _stale_logged.add(key)
            logger.info("Ignoring the stale GGUF export in %s: the folder changed since", folder)
        return None
    listed = [q for q in data["quantizations"] if isinstance(q, str)]
    order = [q for q in PREFERRED if q in listed] + [q for q in listed if q not in PREFERRED]
    directory = folder / EXPORT_DIR
    for quant in order:
        entry = data["files"][quant]
        model = directory / entry["model"]
        mmproj = directory / entry["mmproj"] if entry.get("mmproj") else None
        if model.is_file() and (mmproj is None or mmproj.is_file()):
            return quant, model, mmproj
    return None
