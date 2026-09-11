# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bounded, read-only workspace browsing. No indexing or file execution."""

import base64
import os
import stat
from pathlib import Path, PurePosixPath

MAX_ENTRIES = 2000
MAX_SEARCH_ENTRIES = 10000
MAX_TEXT_BYTES = 256 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024
IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def _identity(st: os.stat_result) -> tuple[int, int]:
    return st.st_dev, st.st_ino


def _regular_path(path: Path) -> os.stat_result:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise ValueError("Symbolic links and junctions cannot be browsed")
    if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
        raise ValueError("Only regular files and folders can be browsed")
    return info


def resolve_path(folder: dict, relative_path: str = "") -> Path:
    from utils.paths.sensitive import contains_sensitive_path_component

    if "\x00" in relative_path or "\\" in relative_path or ":" in relative_path:
        raise ValueError("Invalid relative path")
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Path must stay inside the workspace")
    root = Path(folder["path"])
    root_info = _regular_path(root)

    def stored(value):
        return (
            int(value[1:], 16) if isinstance(value, str) and value.startswith("x") else int(value)
        )

    expected = stored(folder["root_device"]), stored(folder["root_inode"])
    if not stat.S_ISDIR(root_info.st_mode) or _identity(root_info) != expected:
        raise ValueError("Workspace changed, please refresh")
    if os.path.normcase(os.path.realpath(root)) != os.path.normcase(str(root)):
        raise ValueError("Workspace changed, please refresh")
    if relative.parts and relative.parts[0] in {".unsloth_sandbox", ".unsloth_sandbox_remap.json"}:
        raise ValueError("Internal workspace file")
    current = root
    for part in relative.parts:
        current /= part
        _regular_path(current)
        if contains_sensitive_path_component(str(current.relative_to(root))):
            raise ValueError("This folder cannot be browsed")
    if os.path.commonpath([str(root), os.path.realpath(current)]) != str(root):
        raise ValueError("Path must stay inside the workspace")
    return current


def list_directory(folder: dict, relative_path: str = "") -> dict:
    directory = resolve_path(folder, relative_path)
    if not directory.is_dir():
        raise ValueError("Select a folder")
    entries = []
    truncated = False
    with os.scandir(directory) as children:
        for child in children:
            relative = (PurePosixPath(relative_path) / child.name).as_posix()
            try:
                path = resolve_path(folder, relative)
                info = _regular_path(path)
            except (OSError, ValueError):
                continue
            if len(entries) == MAX_ENTRIES:
                truncated = True
                break
            entries.append(
                {
                    "name": child.name,
                    "path": relative,
                    "isDirectory": stat.S_ISDIR(info.st_mode),
                    "size": info.st_size,
                }
            )
    resolve_path(folder, relative_path)
    entries.sort(
        key=lambda entry: (not entry["isDirectory"], entry["name"].casefold(), entry["name"])
    )
    return {"entries": entries, "truncated": truncated}


def preview_file(folder: dict, relative_path: str) -> dict:
    path = resolve_path(folder, relative_path)
    before = _regular_path(path)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("Select a file")
    mime = IMAGE_TYPES.get(path.suffix.lower())
    limit = MAX_IMAGE_BYTES if mime else MAX_TEXT_BYTES
    if mime and before.st_size > limit:
        return {"kind": "unsupported", "message": "Image is too large to preview"}
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or _identity(opened) != _identity(before):
            raise ValueError("File changed, please select it again")
        resolve_path(folder, relative_path)
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read(limit + 1)
        after = os.fstat(fd)
        resolve_path(folder, relative_path)
        if (opened.st_size, opened.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("File changed while reading, please try again")
    finally:
        os.close(fd)
    if mime:
        return {"kind": "image", "mimeType": mime, "data": base64.b64encode(data).decode("ascii")}
    truncated = len(data) > limit
    data = data[:limit]
    if b"\x00" in data:
        return {"kind": "unsupported", "message": "No inline preview for this file type"}
    try:
        # A truncated UTF-8 sequence at the boundary is not a binary file.
        import codecs

        content = codecs.getincrementaldecoder("utf-8-sig")().decode(data, final=not truncated)
    except UnicodeDecodeError:
        return {"kind": "unsupported", "message": "No inline preview for this file type"}
    return {"kind": "text", "content": content, "truncated": truncated}


def search_files(folder: dict, query: str) -> dict:
    """Search names across the workspace with explicit traversal/result bounds."""
    from collections import deque

    pending = deque([""])
    matches = []
    visited = 0
    inspected = 0
    truncated = False
    while pending and visited < 200:
        directory = pending.popleft()
        visited += 1
        try:
            listing = list_directory(folder, directory)
        except (OSError, ValueError):
            if not directory:
                raise
            continue
        truncated |= listing["truncated"]
        for entry in listing["entries"]:
            inspected += 1
            if inspected > MAX_SEARCH_ENTRIES:
                return {"entries": matches, "truncated": True}
            if entry["isDirectory"]:
                if entry["path"].count("/") < 16 and len(pending) < 200:
                    pending.append(entry["path"])
                else:
                    truncated = True
            elif query.casefold() in entry["name"].casefold():
                matches.append(entry)
                if len(matches) == MAX_ENTRIES:
                    return {"entries": matches, "truncated": True}
    matches.sort(key=lambda entry: entry["path"].casefold())
    return {"entries": matches, "truncated": truncated or bool(pending)}
