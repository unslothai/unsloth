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
from hub.services.models import account_access
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


def retire_account_roots(account: AccountContext):
    """Signal this account's work, then rename each private root aside (children first, as roots
    may nest). Returns a callable that renames them back if a later step fails."""
    if account.is_owner or account.account_id == "owner":
        raise ValueError("The installation owner cannot be retired")
    active_generations.fence(account.account_id)
    active_generations.cancel_all(account.account_id)
    account_access.retire_resident_shares(account.account_id)
    from core.inference.mcp_client import close_mcp_sessions, invalidate_tool_cache
    from core.training.account_jobs import retire_account_jobs

    retire_account_jobs(account)
    from core.inference.video import generation_account_in_flight, get_video_backend
    from core.training.account_jobs import AccountRetirementError

    from core.inference.diffusion_engine_router import cancel_generation_for_account

    # Signal both media generations before raising, so one delete retry stops them all.
    video_active = generation_account_in_flight() == account.account_id
    if video_active:
        get_video_backend().cancel_generate(expected_account = account.account_id)
    image_active = cancel_generation_for_account(account.account_id)
    if video_active or image_active:
        raise AccountRetirementError("Media generation is still active; retry deletion")
    # A load the account started before its tombstone is torn down here; one after it is refused.
    from core.inference import video as video_module
    from core.inference.diffusion_engine_router import retire_load_for_account
    from routes.inference import retire_account_loads

    retire_load_for_account(account.account_id)
    video_module.retire_load_for_account(account.account_id)
    retire_account_loads(account.account_id)
    run_as(account, close_mcp_sessions)
    run_as(account, invalidate_tool_cache)
    from storage.studio_db import close_wal_keeper_for

    close_wal_keeper_for(run_as(account, storage_roots.workspace_root) / "studio.db")
    roots = {root.absolute() for root in run_as(account, storage_roots.managed_account_roots)}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    moved: list[tuple[Path, Path]] = []

    def restore() -> None:
        # A root that will not come back stays listed, so the caller hears where its data is.
        stranded: list[tuple[Path, Path, OSError]] = []
        with storage_roots.root_retirement_lock:
            for root, destination in reversed(moved):
                try:
                    Path.rename(destination, root)
                except OSError as exc:
                    stranded.append((root, destination, exc))
            moved[:] = [(root, destination) for root, destination, _ in stranded]
        if stranded:
            raise AccountRetirementError(
                "Could not restore retired directories; the data remains at "
                + ", ".join(str(destination) for _, destination, _ in stranded)
            ) from stranded[0][2]

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
            # All or nothing: reactivation restores no roots, so a half-retired account is empty.
            restore()
            raise
    return restore


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
            active_generations.lift_fence(account_id)
        else:
            # Fence first, so a request registering after the sweep is cancelled too.
            active_generations.fence(account_id)
            active_generations.cancel_all(account_id)
            account_access.retire_resident_shares(account_id)
        return result


@router.delete("/{account_id}", status_code = status.HTTP_204_NO_CONTENT)
def delete_account(account_id: str):
    from core.training.account_jobs import AccountRetirementError
    with _account_errors():
        try:
            storage.delete_account(account_id, retire_account_roots)
        except (OSError, AccountRetirementError) as exc:
            detail = "Could not retire account files. The account is disabled; retry deletion."
            if isinstance(exc, AccountRetirementError):
                detail = f"{detail} {exc}"
            raise HTTPException(status_code = 409, detail = detail)
    return Response(status_code = status.HTTP_204_NO_CONTENT)
