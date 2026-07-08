# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MCP tools for Studio's dataset catalog (mostly read-only)."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any, Optional

from fastmcp import FastMCP

from loggers import get_logger
from mcp_server.auth import HINT_READ_ONLY, resolve_hf_token

logger = get_logger(__name__)

GROUP = "data"

_READ_ONLY = {"readOnlyHint": True}


def data_list_local() -> dict[str, Any]:
    """List datasets visible to Studio (uploaded, recipe-generated, and local).

    Returns the same local-dataset listing the Studio UI shows. Use these
    paths/ids in ``train_start``'s ``local_datasets`` argument.
    """
    from hub.services.datasets.local import list_local_datasets_response

    response = list_local_datasets_response()
    return {"datasets": response.model_dump().get("datasets", [])}


def data_check_format(
    dataset_name: str,
    is_vlm: bool = False,
    subset: Optional[str] = None,
    train_split: str = "train",
    hf_token: Optional[str] = None,
) -> dict[str, Any]:
    """Inspect a dataset's columns and detect its chat format.

    Accepts either a HuggingFace dataset id or a local dataset name/path (as
    shown by ``data_list_local``). Returns the detected format, columns, and a
    suggested column mapping when one is needed -- the same check the Studio
    UI runs before training.
    """
    from fastapi import HTTPException

    from hub.schemas.datasets import CheckFormatRequest
    from hub.services.datasets.formatting import check_format_response

    request = CheckFormatRequest(
        dataset_name = dataset_name,
        is_vlm = is_vlm,
        subset = subset,
        train_split = train_split,
    )
    try:
        result = check_format_response(request, hf_token = resolve_hf_token(hf_token))
    except HTTPException as exc:
        return {"success": False, "error": str(exc.detail)}
    return {"success": True, **result.model_dump()}


def data_register(path: str) -> dict[str, Any]:
    """Register a local file as a Studio dataset.

    Copies the file at ``path`` (an absolute path on this host -- ``.csv``,
    ``.json``, ``.jsonl`` or ``.parquet``) into Studio's dataset store so it
    appears in ``data_list_local`` and can be passed to ``train_start``.

    This is the stdio equivalent of Studio's dataset-upload box.
    """
    from hub.services.datasets.local import (
        DATASET_UPLOAD_DIR,
        LOCAL_UPLOAD_EXTS,
        LOCAL_UPLOAD_MAX_BYTES,
        _sanitize_filename,
    )
    from hub.utils.paths import ensure_dir

    src = Path(path).expanduser()
    if not src.is_file():
        return {"success": False, "error": f"Not a file: {path}"}

    ext = src.suffix.lower()
    if ext not in LOCAL_UPLOAD_EXTS:
        return {
            "success": False,
            "error": f"Unsupported file type {ext!r}. Allowed: {sorted(LOCAL_UPLOAD_EXTS)}",
        }

    size = src.stat().st_size
    if size > LOCAL_UPLOAD_MAX_BYTES:
        return {
            "success": False,
            "error": f"File too large ({size:,} bytes; max {LOCAL_UPLOAD_MAX_BYTES:,}).",
        }
    if size == 0:
        return {"success": False, "error": "File is empty."}

    display_name = _sanitize_filename(src.name)
    stored_name = f"{uuid.uuid4().hex}_{Path(display_name).stem}{ext}"
    ensure_dir(DATASET_UPLOAD_DIR)
    stored_path = DATASET_UPLOAD_DIR / stored_name
    try:
        shutil.copy2(src, stored_path)
    except OSError as exc:
        # Remove any partial copy so a failed registration can't leave a
        # corrupt dataset that data_list_local would later surface (matches
        # the HTTP upload route's cleanup).
        stored_path.unlink(missing_ok = True)
        return {"success": False, "error": f"Failed to copy file: {exc}"}

    logger.info("Registered dataset %s -> %s", path, stored_path)
    return {"success": True, "filename": display_name, "stored_path": str(stored_path)}


def register(mcp: FastMCP) -> list[str]:
    """Register the data tools onto ``mcp``; return the tool names added."""
    names: list[str] = []
    mcp.tool(data_list_local, annotations = _READ_ONLY)
    names.append("data_list_local")
    mcp.tool(data_check_format, annotations = _READ_ONLY)
    names.append("data_check_format")
    mcp.tool(data_register)
    names.append("data_register")
    return names
