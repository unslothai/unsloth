# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Account boundaries shared by model services and inference routes."""

from __future__ import annotations

import re
import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from contextlib import closing
from functools import wraps
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from huggingface_hub import HfApi

from auth import policy
from core.inference.gpu_arbiter import GpuBusyForAnotherAccountError
from utils.paths.storage_roots import studio_db_path, workspace_root

from utils.account_context import OWNER, AccountContext, current_account, current_account_id, is_owner_context

_ACCOUNT_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def media_link_target(media_id: str) -> str:
    """Bind a bearer media link to its workspace without changing owner links."""
    if is_owner_context():
        return media_id
    return f"{current_account().account_id}:{media_id}"


def media_link_account(target: str | None, media_id: str) -> AccountContext | None:
    """Resolve an already signature-verified target, never an unsigned account selector."""
    if target == media_id:
        return OWNER
    if not target:
        return None
    account_id, sep, signed_id = target.partition(":")
    if not sep or signed_id != media_id or not _ACCOUNT_ID.fullmatch(account_id):
        return None
    if account_id == OWNER.account_id:
        return OWNER
    return AccountContext(account_id, "", "user")


def managed_account() -> bool:
    """Keep account policy off the owner's model and credential paths."""
    if is_owner_context():
        return False
    return policy.installation_is_multi_user()


def account_scope() -> str | None:
    """None preserves installation-wide legacy queries on one-account installs."""
    return current_account_id() if policy.installation_is_multi_user() else None


_resident_accounts: dict[str, tuple[str, frozenset[str]]] = {}


def note_resident_account(modality: str, *references: str) -> None:
    """CPU residents have no GPU lease, so retain their load provenance at the route boundary."""
    if policy.installation_is_multi_user():
        _resident_accounts[modality] = (current_account_id(), frozenset(references))


def resident_hidden(modality: str | None = None, reference: str | None = None) -> bool:
    if not managed_account():
        return False
    from core.inference import gpu_arbiter
    owner = gpu_arbiter.current_owner()
    if owner is not None and (modality is None or owner == modality):
        return gpu_arbiter.owner_account() != current_account_id()
    if reference:
        account, references = _resident_accounts.get(modality, (OWNER.account_id, frozenset()))
        return account != current_account_id() or reference not in references
    return False


def hidden_resident_response():
    return JSONResponse(content = {"loaded": True, "yours": False})


def gpu_busy_error() -> HTTPException:
    return HTTPException(
        status_code = 409,
        detail = {"error": "gpu_busy", "retry_after": 1},
        headers = {"Retry-After": "1"},
    )


def require_idle_other_accounts() -> None:
    if policy.installation_is_multi_user():
        from state import active_generations
        if active_generations.foreign_count(current_account_id()):
            raise gpu_busy_error()


def require_resident_control(modality: str, reference: str | None = None) -> None:
    require_idle_other_accounts()
    if resident_hidden(modality, reference):
        raise HTTPException(status_code = 404, detail = "Model not found")


def gpu_busy_route(handler):
    """Keep the retry response identical on Studio and OpenAI load surfaces."""
    @wraps(handler)
    async def wrapped(*args, **kwargs):
        try:
            return await handler(*args, **kwargs)
        except GpuBusyForAnotherAccountError:
            error = gpu_busy_error()
            return JSONResponse(status_code = 409, content = error.detail, headers = error.headers)
        except HTTPException as exc:
            if isinstance(exc.detail, dict) and exc.detail.get("error") == "gpu_busy":
                return JSONResponse(status_code = 409, content = exc.detail, headers = exc.headers)
            raise
    return wrapped


def ambient_hf_token():
    return False if managed_account() else os.environ.get("HF_TOKEN")


def account_hf_token(token):
    """False is Hugging Face's explicit anonymous sentinel; None lends the ambient token."""
    if managed_account() and not token:
        return False
    return token


def require_installation_owner() -> None:
    if managed_account():
        raise HTTPException(status_code = 403, detail = "Only the installation owner can do this")


_PUBLIC_TTL = 300.0
_PRIVATE_TTL = 30.0
_public_repos: dict[tuple[str, str], tuple[float, bool]] = {}
_public_lock = threading.Lock()


def repo_is_public(repo_id: str, repo_type: str = "model") -> bool:
    """Only an anonymous Hub answer proves that a shared-cache repo is public."""
    key = (repo_type, repo_id.lower())
    now = time.monotonic()
    with _public_lock:
        cached = _public_repos.get(key)
        if cached is not None and cached[0] > now:
            return cached[1]
    try:
        info = HfApi().repo_info(repo_id, repo_type = repo_type, token = False, timeout = 5.0)
        public = getattr(info, "private", None) is False and not getattr(info, "gated", False)
    except Exception:  # noqa: BLE001 - an unreachable Hub never establishes public access
        public = False
    with _public_lock:
        if len(_public_repos) >= 4096:
            _public_repos.clear()
        _public_repos[key] = (now + (_PUBLIC_TTL if public else _PRIVATE_TTL), public)
    return public


def _grant_key(repo_id: str, repo_type: str) -> str:
    return f"{repo_type}:{repo_id.strip().lower()}"


def model_grants() -> set[str]:
    """Read only this account's grants; absent or malformed records confer no access."""
    path = studio_db_path()
    if not path.is_file():
        return set()
    try:
        with closing(sqlite3.connect(str(path))) as conn:
            row = conn.execute("SELECT value_json FROM app_settings WHERE key = 'model_grants'").fetchone()
        grants = json.loads(row[0]) if row else []
        return {key for key in grants if isinstance(key, str)} if isinstance(grants, list) else set()
    except (sqlite3.Error, ValueError, TypeError):
        return set()


def record_model_grant(repo_id: str, repo_type: str = "model") -> None:
    """Record a successfully authorized download in the initiating account's studio.db.

    Called from the account-bound download watcher for both models and datasets. A
    transaction preserves simultaneous completions without depending on storage's
    installation-era schema cache. Owner downloads need no grants.
    """
    if not managed_account() or not repo_id:
        return
    path = studio_db_path()
    path.parent.mkdir(parents = True, exist_ok = True)
    with closing(sqlite3.connect(str(path), timeout = 5.0)) as conn, conn:
        conn.execute("CREATE TABLE IF NOT EXISTS app_settings (key TEXT NOT NULL PRIMARY KEY, value_json TEXT NOT NULL, updated_at TEXT NOT NULL)")
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT value_json FROM app_settings WHERE key = 'model_grants'").fetchone()
        try:
            prior = json.loads(row[0]) if row else []
        except (ValueError, TypeError):
            prior = []
        grants = {key for key in prior if isinstance(key, str)} if isinstance(prior, list) else set()
        grants.add(_grant_key(repo_id, repo_type))
        conn.execute(
            "INSERT INTO app_settings (key, value_json, updated_at) VALUES ('model_grants', ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json, updated_at = excluded.updated_at",
            (json.dumps(sorted(grants)), datetime.now(timezone.utc).isoformat()),
        )


def repo_visible(repo_id: str, repo_type: str = "model", *, grants: set[str] | None = None) -> bool:
    if not managed_account():
        return True
    if not repo_id:
        return False
    granted = model_grants() if grants is None else grants
    return _grant_key(repo_id, repo_type) in granted or repo_is_public(repo_id, repo_type)


def _cached_repo(path: Path) -> tuple[str, str] | None:
    for part in reversed(path.parts):
        for prefix, repo_type in (("models--", "model"), ("datasets--", "dataset")):
            if part.startswith(prefix):
                pieces = part[len(prefix):].split("--")
                if len(pieces) == 2 and all(pieces):
                    return "/".join(pieces), repo_type
    return None


def model_visible(reference: str, *, grants: set[str] | None = None, repo_type: str = "model") -> bool:
    """Apply grants equally to repo ids and cache snapshot/file spellings.

    Arbitrary local paths are private to the current workspace. A symlink cannot
    turn another account's private files into a visible local model.
    """
    if not managed_account():
        return True
    if not isinstance(reference, str) or not reference:
        return False
    reference = reference.strip()
    path = Path(reference).expanduser()
    if path.is_absolute() or reference.startswith(("./", "../", "~")) or path.exists():
        try:
            resolved = path.resolve()
            if resolved.is_relative_to(workspace_root().resolve()):
                return True
            from utils.hf_cache_settings import known_hf_hub_caches
            if not any(resolved.is_relative_to(root.resolve()) for root in known_hf_hub_caches()):
                return False
            cached = _cached_repo(path)
            if cached is not None:
                # HF snapshots point to the same repository's blobs; cross-repo links are refused.
                actual = _cached_repo(resolved)
                return actual == cached and repo_visible(cached[0], cached[1], grants = grants)
        except (OSError, RuntimeError, ValueError):
            return False
        return False
    # Quant suffixes and an HF filename still belong to the parent repository.
    repo_id = reference.split(":", 1)[0]
    parts = repo_id.split("/")
    if len(parts) < 2 or not all(parts[:2]):
        return False
    repo_id = "/".join(parts[:2])
    return repo_visible(repo_id, repo_type, grants = grants)


def require_model_access(reference: str, repo_type: str = "model") -> None:
    if not model_visible(reference, repo_type = repo_type):
        raise HTTPException(status_code = 404, detail = "Model not found")


def filter_model_rows(rows, *, repo_type: str = "model"):
    """Filter after shared scans/caches, never store a caller's filtered catalog globally."""
    if not managed_account():
        return rows
    grants = model_grants()
    visible = []
    for row in rows:
        get = row.get if isinstance(row, dict) else lambda key, default = None: getattr(row, key, default)
        ref = get("path") or get("local_path") or get("repo_id") or get("model_id") or get("id")
        if model_visible(ref, grants = grants, repo_type = repo_type):
            visible.append(row)
    return visible


def private_directory(path: str, folder: str) -> str:
    """Rebase import-time owner defaults and refuse arbitrary account-external scans."""
    if not managed_account():
        return path
    from utils.paths.storage_roots import studio_root
    legacy = studio_root() / folder
    target = workspace_root() / folder if Path(path).resolve() == legacy.resolve() else Path(path)
    if not target.resolve().is_relative_to(workspace_root().resolve()):
        raise HTTPException(status_code = 404, detail = "Directory not found")
    return str(target)


def authorize_download(repo_id: str, repo_type: str, hf_token) -> None:
    """A cache hit is not proof the requester may download private Hub content."""
    if not managed_account():
        return
    try:
        api = HfApi()
        token = account_hf_token(hf_token)
        info = api.repo_info(repo_id, repo_type = repo_type, token = token, timeout = 5.0)
        if getattr(info, "gated", False):
            # Gated repo metadata is public even when the caller cannot read its files.
            api.auth_check(repo_id, repo_type = repo_type, token = token)
    except Exception as exc:  # noqa: BLE001 - never convert another account's cached files into a grant
        raise HTTPException(status_code = 404, detail = "Repository not found") from exc


_schema_paths: set[tuple[str, str]] = set()
_schema_lock = threading.Lock()


def ensure_account_schema(module) -> None:
    """Initialize a private DB for callers of storage modules with legacy global schema flags.

    Storage owns the DDL. This account-aware caller bridge can be retired when all
    storage modules track schema readiness by database path.
    """
    if not managed_account():
        return
    path = studio_db_path()
    key = module.__name__, str(path)
    with _schema_lock:
        if key in _schema_paths:
            return
        path.parent.mkdir(parents = True, exist_ok = True)
        with closing(sqlite3.connect(str(path))) as conn, conn:
            conn.row_factory = sqlite3.Row
            module._ensure_schema(conn)
        _schema_paths.add(key)


def require_media_references(request) -> None:
    """Companion file overrides obey the same policy as the primary model."""
    require_media_adapters(request)
    for name in ("gguf_filename", "transformer_prequant_path"):
        reference = getattr(request, name, None)
        if not isinstance(reference, str) or not reference:
            continue
        path = Path(reference).expanduser()
        if name == "transformer_prequant_path":
            require_model_access(str(path.resolve()))
        elif path.is_absolute():
            require_model_access(reference)
        elif ".." in path.parts:
            raise HTTPException(status_code = 404, detail = "Model not found")
        elif Path(request.model_path).is_absolute():
            require_model_access(str(Path(request.model_path) / path))


def require_media_generation_access(status: dict) -> None:
    """Recheck the actual resident before a cache-only generation can reuse it."""
    if managed_account() and status.get("loaded"):
        require_model_access(status.get("repo_id"))


def foreign_work_active() -> bool:
    if not policy.installation_is_multi_user():
        return False
    from state import active_generations
    return bool(active_generations.foreign_count(current_account_id()))


def require_download_progress_access(registry, repo_id: str) -> None:
    if not managed_account():
        return
    from hub.services import download_lifecycle
    if not any(download_lifecycle.download_belongs_to_account(registry, ref.key) for ref in registry.active_job_refs(repo_id)):
        require_model_access(repo_id)


def require_media_adapters(request) -> None:
    """Apply model grants to catalog aliases as well as raw adapter repo ids."""
    if not managed_account():
        return
    loras = getattr(request, "loras", None)
    controlnet = getattr(request, "controlnet", None)
    groups = []
    if loras:
        from core.inference import diffusion_lora
        groups.append((loras, diffusion_lora.list_loras()))
    if controlnet:
        from core.inference import diffusion_controlnet
        groups.append(([controlnet], diffusion_controlnet.list_controlnets()))
    for selections, entries in groups:
        by_id = {entry.id: entry for entry in entries}
        for selection in selections:
            entry = by_id.get(selection.id)
            reference = (entry.local_path or entry.repo_id) if entry is not None else selection.id
            require_model_access(reference)
