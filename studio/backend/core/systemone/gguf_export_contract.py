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
folder makes an older export stale. Standard library only: Unsloth's save path loads this file by
path, without importing Studio.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EXPORT_DIR = "gguf"
EXPORT_FILE = "export.json"
FORMAT = "unsloth-decision-gguf"
VERSION = 1
LAYOUTS = ("clef", "laya")
_CLEF_FILES = ("unsloth_decision_config.json", "joint_head_config.json", "joint_head.safetensors")
_LAYA_FILES = ("rl_agent_config.json",)
# A stock Clef checkpoint has no Unsloth config: it serves at temperature 1.
_OPTIONAL_FILES = ("unsloth_decision_config.json",)
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


def _fingerprint_files(folder: Path, layout: str) -> list[Path]:
    if layout == "clef":
        return [folder / name for name in _CLEF_FILES]
    if layout == "laya":
        weights = sorted(folder.glob("model*.safetensors")) + sorted(
            folder.glob("model.safetensors.index.json")
        )
        return [folder / name for name in _LAYA_FILES] + weights
    raise ValueError(f"Unknown decision layout: {layout!r}")


def fingerprint(folder: str | Path, layout: str) -> str:
    """sha256 over each fingerprint file as name, NUL, then its sha256 (``missing`` if optional)."""
    folder = Path(folder)
    digest = hashlib.sha256()
    for path in _fingerprint_files(folder, layout):
        digest.update(path.name.encode() + b"\0")
        if not path.is_file():
            if path.name in _OPTIONAL_FILES:
                digest.update(b"missing")
                continue
            raise FileNotFoundError(f"{path} is missing, so {folder} cannot be fingerprinted")
        stat = path.stat()
        digest.update(_file_digest(str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    return digest.hexdigest()


def _plain_name(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value == Path(value).name
        and value not in (".", "..")
        and "\\" not in value
    )


def _valid(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    if data.get("format") != FORMAT or data.get("version") != VERSION:
        return False
    if data.get("layout") not in LAYOUTS or not isinstance(data.get("source_fingerprint"), str):
        return False
    files, quantizations = data.get("files"), data.get("quantizations")
    if not isinstance(files, dict) or not isinstance(quantizations, list) or not quantizations:
        return False
    for quant in quantizations:
        entry = files.get(quant) if isinstance(quant, str) else None
        if not isinstance(entry, dict) or not _plain_name(entry.get("model")):
            return False
        mmproj = entry.get("mmproj")
        if mmproj is not None and not _plain_name(mmproj):
            return False
    return True


def read_export(folder: str | Path) -> dict[str, Any] | None:
    """export.json of a run folder (or of its gguf/ directory) if well formed, else None."""
    folder = Path(folder)
    path = folder / EXPORT_FILE if folder.name == EXPORT_DIR else folder / EXPORT_DIR / EXPORT_FILE
    try:
        data = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None
    return data if _valid(data) else None


def write_export(export_dir: str | Path, layout: str, files: dict, source_fingerprint: str) -> dict:
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


def served_files(folder: str | Path, layout: str) -> tuple[str, Path, Path | None] | None:
    """(quantization, model path, mmproj path or None) of the folder's current export, else None.

    An export whose fingerprint no longer matches the folder is stale and ignored (logged once).
    """
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
