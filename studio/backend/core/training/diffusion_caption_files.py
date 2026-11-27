# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

CAPTION_EXTS = (".txt", ".caption")
METADATA_NAMES = ("metadata.jsonl", "captions.jsonl")


def canonical_caption_file_name(name: str) -> str:
    stem, ext = os.path.splitext(name)
    if name.lower() in METADATA_NAMES:
        return name.lower()
    if ext.lower() in CAPTION_EXTS:
        return stem + ext.lower()
    return name


def caption_sidecar_index(folder: Path) -> dict[tuple[str, str], list[str]]:
    index: dict[tuple[str, str], list[str]] = {}
    try:
        entries = list(os.scandir(folder))
    except OSError:
        return index
    for entry in entries:
        stem, ext = os.path.splitext(entry.name)
        if ext.lower() in CAPTION_EXTS and entry.is_file():
            index.setdefault((stem, ext.lower()), []).append(entry.name)
    return index


def _same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def _iter_caption_sidecars(media: Path, index: Optional[dict[tuple[str, str], list[str]]]):
    if index is None:
        index = caption_sidecar_index(media.parent)
    for ext in CAPTION_EXTS:
        exact = media.with_suffix(ext)
        names = sorted(index.get((media.stem, ext), ()))
        if exact.name in names:
            names.remove(exact.name)
            yield exact
            yield from (media.parent / n for n in names)
            continue
        # Not listed under its own name: a case-insensitive filesystem may still resolve it to a listed entry.
        exact_found = exact.is_file()
        if exact_found:
            yield exact
        for n in names:
            path = media.parent / n
            if not (exact_found and _same_file(path, exact)):
                yield path


def caption_sidecars(
    media: Path, index: Optional[dict[tuple[str, str], list[str]]] = None
) -> list[Path]:
    return list(_iter_caption_sidecars(media, index))


def find_caption_sidecar(
    media: Path, index: Optional[dict[tuple[str, str], list[str]]] = None
) -> Optional[Path]:
    return next(_iter_caption_sidecars(media, index), None)


def metadata_caption_files(folder: Path) -> list[Path]:
    try:
        names = sorted(e.name for e in os.scandir(folder) if e.is_file())
    except OSError:
        return []
    found: list[Path] = []
    for canonical in METADATA_NAMES:
        name = (
            canonical
            if canonical in names
            else next((n for n in names if n.lower() == canonical), None)
        )
        if name is not None:
            found.append(folder / name)
    return found
