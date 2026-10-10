# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import base64
import json
import os
import stat

from .mcp_images import SENTINEL, _png_data_url

MAX_FILE_BYTES = 12 * 1024 * 1024

VIEW_IMAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "view_image",
        "description": (
            "Open an existing image in this conversation's working directory and see its pixels. "
            "Use this to inspect screenshots, plots, or other images, including files you created "
            "with Python or terminal. Accepts a relative path or an absolute path inside the "
            "working directory. Files outside it are not accessible, even with Full access."
        ),
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to the image file"}},
            "required": ["path"],
        },
    },
}


def _strip_habit_prefix(path: str, root: str) -> str:
    from .tools import _MISSING_PATH_PREFIXES

    # Not gated on isabs: Windows treats drive-less /mnt/data as relative; commonpath can raise.
    try:
        if os.path.isabs(path) and os.path.commonpath([root, os.path.realpath(path)]) == root:
            return path
    except ValueError:
        pass
    for prefix in _MISSING_PATH_PREFIXES:
        if path == prefix or path.startswith(prefix + "/"):
            return path[len(prefix) :].lstrip("/") or "."
    return path


def view_image(
    path,
    workdir: str,
    cancel_event = None,
) -> str:
    if not isinstance(path, str) or not path.strip():
        return "Error: 'path' must be a non-empty string."
    if cancel_event is not None and cancel_event.is_set():
        return "Error: image viewing was cancelled."
    root = os.path.realpath(workdir)
    try:
        path = _strip_habit_prefix(path, root)
        target = os.path.realpath(os.path.join(root, path))
        if os.path.commonpath([root, target]) != root:
            return "Error: image is outside this conversation's working directory."
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        flags |= getattr(os, "O_BINARY", 0)
        if os.open in os.supports_dir_fd:
            # Walk the resolved path from an open root: swapping a parent for a symlink must not escape it.
            directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                parts = os.path.relpath(target, root).split(os.sep)
                for part in parts[:-1]:
                    child = os.open(
                        part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd = directory
                    )
                    os.close(directory)
                    directory = child
                handle = os.open(parts[-1], flags, dir_fd = directory)
            finally:
                os.close(directory)
        else:
            handle = os.open(target, flags)
        try:
            source = os.fdopen(handle, "rb")
        except BaseException:
            os.close(handle)
            raise
        with source:
            info = os.fstat(source.fileno())
            checked_path = os.path.realpath(target)
            checked = os.stat(checked_path)
            if (
                os.path.commonpath([root, checked_path]) != root
                or not stat.S_ISREG(info.st_mode)
                or (info.st_dev, info.st_ino) != (checked.st_dev, checked.st_ino)
            ):
                return "Error: image must be a regular file inside this conversation's working directory."
            if info.st_size > MAX_FILE_BYTES:
                return "Error: image exceeds the 12 MiB file limit."
            raw = source.read(MAX_FILE_BYTES + 1)
        if len(raw) > MAX_FILE_BYTES:
            return "Error: image exceeds the 12 MiB file limit."
    except (OSError, ValueError):
        return "Error: could not read the image file inside this conversation's working directory."
    if cancel_event is not None and cancel_event.is_set():
        return "Error: image viewing was cancelled."
    url = _png_data_url(base64.b64encode(raw).decode("ascii"))
    if url is None:
        return "Error: file is not a supported image or exceeds the 40 megapixel limit."
    return (
        "Opened image."
        + "\n"
        + SENTINEL
        + json.dumps([{"data": url.split(",", 1)[1], "mimeType": "image/png"}])
    )
