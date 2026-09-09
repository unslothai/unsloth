# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Owner-managed account lifecycle. Private roots are retired, never erased."""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response, status

from auth import policy, storage
from auth.authentication import get_current_subject
from models.auth import (
    AccountActiveRequest,
    AccountListResponse,
    AccountResponse,
    AccountSetupResponse,
    CreateAccountRequest,
)
from state import active_generations
from utils.account_context import AccountContext, run_as
from utils.paths import storage_roots


router = APIRouter(dependencies = [Depends(get_current_subject), Depends(policy.require_owner)])


@contextmanager
def _account_errors():
    try:
        yield
    except LookupError:
        raise HTTPException(status_code = 404, detail = "Account not found")
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc))
    except sqlite3.IntegrityError:
        raise HTTPException(status_code = 409, detail = "Username is unavailable")


def retire_account_roots(account: AccountContext) -> None:
    """Signal this account's work, then rename each private root aside; roots may nest, so resolve all first and move children first."""
    if account.is_owner or account.account_id == "owner":
        raise ValueError("The installation owner cannot be retired")
    active_generations.cancel_all(account.account_id)
    from core.inference.mcp_client import close_mcp_sessions, invalidate_tool_cache
    from core.training.account_jobs import retire_account_jobs

    retire_account_jobs(account)
    from core.inference.video import generation_account_in_flight, get_video_backend
    from core.training.account_jobs import AccountRetirementError

    if generation_account_in_flight() == account.account_id:
        get_video_backend().cancel_generate(expected_account = account.account_id)
        raise AccountRetirementError("Video generation is still active; retry deletion")
    run_as(account, close_mcp_sessions)
    run_as(account, invalidate_tool_cache)
    from storage.studio_db import close_wal_keeper_for

    close_wal_keeper_for(run_as(account, storage_roots.workspace_root) / "studio.db")
    roots = {
        run_as(account, root).absolute()
        for root in (
            storage_roots.workspace_root,
            storage_roots.project_workspaces_root,
            storage_roots.tmp_root,
        )
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    moved: list[tuple[Path, Path]] = []
    # Same lock as ensure_account_dir: the rename never lands between its check and mkdir.
    with storage_roots.root_retirement_lock:
        try:
            for root in sorted(roots, key = lambda path: len(path.parts), reverse = True):
                # Rename a symlink itself; never resolve it into another account's data.
                if not root.exists() and not root.is_symlink():
                    continue
                destination = root.with_name(f"{root.name}-deleted-{stamp}")
                suffix = 0
                while destination.exists() or destination.is_symlink():
                    suffix += 1
                    destination = root.with_name(f"{root.name}-deleted-{stamp}-{suffix}")
                Path.rename(root, destination)
                moved.append((root, destination))
        except OSError:
            # All or nothing: reactivation restores no roots, so a half-retired account comes back with an empty workspace.
            for root, destination in reversed(moved):
                try:
                    Path.rename(destination, root)
                except OSError:
                    pass
            raise


@router.get("", response_model = AccountListResponse)
def list_accounts():
    return {"accounts": storage.list_accounts()}


@router.post("", response_model = AccountSetupResponse, status_code = status.HTTP_201_CREATED)
def create_account(payload: CreateAccountRequest):
    with _account_errors():
        return storage.issue_account_setup_code(username = payload.username)


@router.post("/{account_id}/setup-code", response_model = AccountSetupResponse)
def regenerate_setup_code(account_id: str):
    with _account_errors():
        return storage.issue_account_setup_code(account_id = account_id)


@router.patch("/{account_id}", response_model = AccountResponse)
def set_account_active(account_id: str, payload: AccountActiveRequest):
    with _account_errors():
        result = storage.set_account_active(account_id, payload.is_active)
        if payload.is_active:
            # A delete that failed after retiring the jobs left the id tombstoned in-process.
            from core.training.account_jobs import restore_account_jobs
            restore_account_jobs(account_id)
        else:
            active_generations.cancel_all(account_id)
        return result


@router.delete("/{account_id}", status_code = status.HTTP_204_NO_CONTENT)
def delete_account(account_id: str):
    from core.training.account_jobs import AccountRetirementError
    with _account_errors():
        try:
            storage.delete_account(account_id, retire_account_roots)
        except (OSError, AccountRetirementError):
            raise HTTPException(
                status_code = 409,
                detail = "Could not retire account files. The account is disabled; retry deletion.",
            )
    return Response(status_code = status.HTTP_204_NO_CONTENT)
