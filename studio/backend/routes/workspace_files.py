# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Read-only Files panel for the same workspace used by chat tools."""

from contextlib import contextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from auth.authentication import get_current_subject
from core.workspace_files import list_directory, preview_file, search_files

router = APIRouter(dependencies=[Depends(get_current_subject)])


def workspace_path(session: str) -> Path:
    from core.inference.tools import resolve_sandbox_workdir

    return Path(resolve_sandbox_workdir(session))


@contextmanager
def browse_errors():
    try:
        yield
    except FileNotFoundError as exc:
        raise HTTPException(404, "File or folder is no longer available") from exc
    except PermissionError as exc:
        raise HTTPException(403, "File or folder is not readable") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(400, "Could not access this file or folder") from exc


def folder_for(root: Path) -> dict:
    info = root.lstat()
    return {"path": str(root), "root_device": info.st_dev, "root_inode": info.st_ino}


@router.get("/files")
def files(
    session: str = Query(min_length=1, max_length=4096),
    path: str = Query(default="", max_length=4096),
    q: str = Query(default="", max_length=256),
) -> dict:
    with browse_errors():
        root = workspace_path(session)
        if not root.exists() and not path:
            return {"entries": [], "truncated": False}
        folder = folder_for(root)
        return search_files(folder, q) if q else list_directory(folder, path)


@router.get("/preview")
def preview(
    session: str = Query(min_length=1, max_length=4096),
    path: str = Query(min_length=1, max_length=4096),
) -> dict:
    with browse_errors():
        return preview_file(folder_for(workspace_path(session)), path)
