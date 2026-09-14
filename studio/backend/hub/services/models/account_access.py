# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Account boundaries shared by model services and inference routes."""

from __future__ import annotations

import re
import json
import os
import sqlite3
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future
from datetime import datetime, timezone
from contextlib import closing, contextmanager
from functools import wraps
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from huggingface_hub import HfApi

from auth import policy
from core.inference.gpu_arbiter import GpuBusyForAnotherAccountError
from utils.paths import storage_roots
from utils.paths.storage_roots import project_workspaces_root, studio_db_path, workspace_root

from utils.account_context import (
    OWNER,
    AccountContext,
    account_thread,
    current_account,
    current_account_id,
    is_owner_context,
)

_ACCOUNT_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def media_link_target(media_id: str) -> str:
    if is_owner_context():
        return media_id
    return f"{current_account().account_id}:{media_id}"


_link_accounts: dict[str, tuple[int, AccountContext | None]] = {}
_link_accounts_lock = threading.Lock()


def _signed_link_account(account_id: str) -> AccountContext | None:
    """Keyed on policy generation so deactivation revokes links at once, not at TTL."""
    generation = policy.account_generation()
    with _link_accounts_lock:
        cached = _link_accounts.get(account_id)
        if cached is not None and cached[0] == generation:
            return cached[1]
    from auth.storage import get_account_by_id

    account = get_account_by_id(account_id)
    with _link_accounts_lock:
        if len(_link_accounts) >= 256:
            _link_accounts.clear()
        _link_accounts[account_id] = (generation, account)
    return account


def media_link_account(target: str | None, media_id: str) -> AccountContext | None:
    if target == media_id:
        return OWNER
    if not target:
        return None
    account_id, sep, signed_id = target.partition(":")
    if not sep or signed_id != media_id or not _ACCOUNT_ID.fullmatch(account_id):
        return None
    if account_id == OWNER.account_id:
        return OWNER
    account = _signed_link_account(account_id)
    return None if account is None or account.is_owner else account


def managed_account() -> bool:
    return not is_owner_context()


def account_scope() -> str | None:
    """None keeps legacy install-wide scope; any managed-account record scopes owner work too."""
    if not is_owner_context():
        return current_account_id()
    return current_account_id() if policy.installation_has_managed_accounts() else None


_resident_accounts: dict[str, tuple[str, frozenset[str]]] = {}

_generation_accounts: dict[str, dict[str, int]] = {}
_generation_lock = threading.Lock()


@contextmanager
def media_generation(modality: str):
    """Recorded even on one-account installs: an account created mid-generation sees it foreign."""
    account_id = current_account_id()
    with _generation_lock:
        counts = _generation_accounts.setdefault(modality, {})
        counts[account_id] = counts.get(account_id, 0) + 1
    try:
        yield
    finally:
        with _generation_lock:
            counts = _generation_accounts.get(modality) or {}
            remaining = counts.get(account_id, 0) - 1
            if remaining > 0:
                counts[account_id] = remaining
            else:
                counts.pop(account_id, None)
            if not counts:
                _generation_accounts.pop(modality, None)


_generation_holders: dict[str, list[str]] = {}


@contextmanager
def media_generation_slot(modality: str):
    """Entered only with the slot held, so a queued request is not the running one."""
    account_id = current_account_id()
    with _generation_lock:
        _generation_holders.setdefault(modality, []).append(account_id)
    try:
        yield
    finally:
        with _generation_lock:
            holders = _generation_holders.get(modality) or []
            if account_id in holders:
                holders.remove(account_id)
            if not holders:
                _generation_holders.pop(modality, None)


def tracked_generation_account() -> Optional[str]:
    return account_scope()


def generation_is_mine(modality: str) -> bool:
    if account_scope() is None:
        return False
    account_id = current_account_id()
    with _generation_lock:
        holders = _generation_holders.get(modality)
        if holders:
            return account_id in holders
        return bool(_generation_accounts.get(modality, {}).get(account_id))


def generation_is_foreign(modality: str) -> bool:
    # account_scope, not login mode: a deactivated account's job keeps running and stays foreign.
    if account_scope() is None:
        return False
    account_id = current_account_id()
    with _generation_lock:
        holders = _generation_holders.get(modality)
        if holders:
            return any(holder != account_id for holder in holders)
        return any(
            account != account_id and count
            for account, count in _generation_accounts.get(modality, {}).items()
        )


def foreign_media_generations(account_id: str) -> int:
    """Other accounts' image and video work, tracked outside ``active_generations``."""
    total = 0
    with _generation_lock:
        for modality, counts in _generation_accounts.items():
            holders = _generation_holders.get(modality)
            if holders:
                continue
            total += sum(count for account, count in counts.items() if account != account_id)
        for holders in _generation_holders.values():
            total += sum(1 for holder in holders if holder != account_id)
    # sys.modules, not an import: no video job is in flight before its module loads.
    video = sys.modules.get("core.inference.video")
    reserved = video.generation_account_in_flight() if video is not None else None
    if reserved is not None and reserved != account_id:
        total += 1
    return total


# A failed load leaves the previous model resident, still owned by its account.
_prior_resident_accounts: dict[str, tuple[str, frozenset[str]]] = {}
# What each publish displaced, so a load that never commits can restore it.
_uncommitted_resident: dict[str, tuple] = {}
_uncommitted_components: dict[str, tuple] = {}


def note_resident_account(modality: str, *references: str) -> None:
    """CPU residents have no GPU lease, so record load provenance at the route boundary."""
    if policy.installation_has_managed_accounts():
        previous = _resident_accounts.get(modality)
        _uncommitted_resident[modality] = (
            current_account_id(),
            previous,
            _prior_resident_accounts.get(modality),
        )
        if previous is not None and previous[1] != frozenset(references):
            _prior_resident_accounts[modality] = previous
        _resident_accounts[modality] = (current_account_id(), frozenset(references))


# Admits a media load and scans for one at retirement, so neither slips between the other's steps.
media_load_lock = threading.Lock()


def require_live_account() -> None:
    """A tombstoned or disabled account starts nothing and shares nothing."""
    if managed_account():
        from core.training.account_jobs import account_is_retired
        from state import active_generations

        if account_is_retired():
            raise HTTPException(status_code = 403, detail = "Account is retired")
        if active_generations.fenced(current_account_id()):
            raise HTTPException(status_code = 403, detail = "Account is disabled")


def admit_media_load(modality: str, start, *references: str):
    """Start a load under the retirement scan's lock, so a tombstoned account starts none."""
    with media_load_lock:
        require_live_account()
        result = start()
        note_resident_account(modality, *references)
        return result


def retire_media_load(modality: str, account_id: str, engine) -> bool:
    """Tear down ``engine``'s in-flight load when ``account_id`` started it; True when one was."""
    with media_load_lock:
        resident = _resident_accounts.get(modality)
        if engine is None or resident is None or resident[0] != account_id:
            return False
        if not engine.loading_repo_ids():
            return False
        engine.unload()
        return True


# Accounts sharing the resident: its loader plus every account whose matching load reused it.
_resident_sharers: dict[str, set[str]] = {}
_sharers_lock = threading.Lock()


def publish_resident(modality: str, *references: str) -> None:
    """A new resident: recorded as before, with its loader as the only sharer."""
    if not policy.installation_has_managed_accounts():
        return
    require_live_account()
    note_resident_account(modality, *references)
    with _sharers_lock:
        _resident_sharers[modality] = {current_account_id()}


def join_resident(modality: str) -> None:
    """A matching load reused the resident: the caller shares it from now on."""
    if not policy.installation_has_managed_accounts():
        return
    require_live_account()
    with _sharers_lock:
        _resident_sharers.setdefault(modality, set()).add(current_account_id())


def release_shared_resident(modality: str) -> bool:
    """Drop the caller's share while others keep the model; False leaves the share for a real unload."""
    if not policy.installation_has_managed_accounts():
        return False
    account_id = current_account_id()
    with _sharers_lock:
        sharers = _resident_sharers.get(modality)
        if not sharers or account_id not in sharers or len(sharers) == 1:
            return False
        sharers.discard(account_id)
        return True


def clear_resident(modality: str) -> None:
    """The backend was torn down; the next load publishes afresh."""
    with _sharers_lock:
        _resident_sharers.pop(modality, None)


def retire_resident_shares(account_id: str) -> None:
    """A deactivated or deleted account shares nothing; other sharers keep the model."""
    with _sharers_lock:
        for sharers in _resident_sharers.values():
            sharers.discard(account_id)


def resident_shared_with(modality: str, account_id: str) -> bool:
    with _sharers_lock:
        return account_id in _resident_sharers.get(modality, ())


_resident_components: dict[str, tuple[str, frozenset[str]]] = {}


def note_resident_components(modality: str, primary: str, *references: str) -> None:
    """A generation on a shared resident must clear its base repo and baked adapters too."""
    if policy.installation_has_managed_accounts():
        _uncommitted_components[modality] = (
            current_account_id(),
            _resident_components.get(modality),
            None,
        )
        _resident_components[modality] = (
            str(primary or ""),
            frozenset(r for r in references if isinstance(r, str) and r),
        )


def restore_resident_metadata(modality: str) -> bool:
    """Undo a failed load's records; no-op once another load took residency."""
    if not policy.installation_has_managed_accounts():
        return False
    account_id = current_account_id()
    restored = False
    for published, live in (
        (_uncommitted_resident, _resident_accounts),
        (_uncommitted_components, _resident_components),
    ):
        entry = published.get(modality)
        if entry is None or entry[0] != account_id:
            continue
        del published[modality]
        _, previous, prior = entry
        if previous is None:
            live.pop(modality, None)
        else:
            live[modality] = previous
        if live is _resident_accounts:
            if prior is None:
                _prior_resident_accounts.pop(modality, None)
            else:
                _prior_resident_accounts[modality] = prior
        restored = True
    return restored


def resident_hidden(modality: str | None = None, reference: str | None = None) -> bool:
    if not managed_account():
        return False
    if modality is not None and resident_shared_with(modality, current_account_id()):
        return False
    from core.inference import gpu_arbiter

    owner = gpu_arbiter.current_owner()
    if owner is not None and (modality is None or owner == modality):
        return gpu_arbiter.owner_account() != current_account_id()
    if reference:
        account, references = _resident_accounts.get(modality, (OWNER.account_id, frozenset()))
        if reference not in references:
            prior = _prior_resident_accounts.get(modality)
            if prior is not None and reference in prior[1]:
                account, references = prior
        return account != current_account_id() or reference not in references
    return False


def hidden_resident_response():
    return JSONResponse(content = {"loaded": True, "yours": False})


def hidden_chat_status_response():
    """The chat status shape (``loaded`` is a list there), with nothing of the resident."""
    return JSONResponse(content = {"loaded": [], "loading": [], "yours": False})


def gpu_busy_error(path: str | None = None) -> HTTPException:
    from core.inference import gpu_arbiter
    from state import active_generations

    account_id = current_account_id()
    error = gpu_arbiter.GpuBusyForAnotherAccountError(
        gpu_arbiter.current_owner() or gpu_arbiter.CHAT,
        active_generations.foreign_count(account_id),
    )
    return error.as_http_exception(path)


def require_idle_other_accounts(path: str | None = None) -> None:
    # Managed accounts, not login mode: deactivating the last drops the count mid-generation.
    if policy.installation_has_managed_accounts():
        from core.inference.gpu_arbiter import require_no_foreign_generations
        require_no_foreign_generations(current_account_id(), path = path)


def require_resident_control(modality: str, reference: str | None = None) -> None:
    require_idle_other_accounts()
    if resident_hidden(modality, reference):
        raise HTTPException(status_code = 404, detail = "Model not found")


def gpu_busy_route(handler):
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
    """False is the Hub's anonymous sentinel; None would lend the installation token."""
    if managed_account() and (not token or (isinstance(token, str) and not token.strip())):
        return False
    return token


def require_installation_owner() -> None:
    if managed_account():
        raise HTTPException(status_code = 403, detail = "Only the installation owner can do this")


_PUBLIC_TTL = 300.0
_PRIVATE_TTL = 30.0
_public_repos: dict[tuple[str, str], tuple[float, bool]] = {}
_public_lock = threading.Lock()
_public_flights: dict[tuple[str, str], Future] = {}
_PROBE_FANOUT = 8


_UNKNOWN_TTL = 30.0
_PROOF_TTL = 7 * 24 * 3600.0
_DEFINITIVE_HUB_STATUSES = (401, 403, 404)


def _public_verdicts_path() -> Path:
    from utils.paths.storage_roots import cache_root
    return cache_root() / "public_repos.json"


def _load_public_verdicts() -> dict[str, float]:
    try:
        data = json.loads(_public_verdicts_path().read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    oldest = time.time() - _PROOF_TTL
    return {
        name: float(stamp)
        for name, stamp in data.items()
        if isinstance(name, str) and isinstance(stamp, (int, float)) and float(stamp) > oldest
    }


def _remember_public_verdict(name: str, public: bool) -> None:
    verdicts = _load_public_verdicts()
    if public:
        verdicts[name] = time.time()
    elif name in verdicts:
        del verdicts[name]
    else:
        return
    path = _public_verdicts_path()
    try:
        path.parent.mkdir(parents = True, exist_ok = True)
        staging = path.with_name(path.name + ".tmp")
        staging.write_text(json.dumps(verdicts, sort_keys = True), encoding = "utf-8")
        os.replace(staging, path)
    except OSError:
        pass


def _hub_public_answer(repo_id: str, repo_type: str) -> bool | None:
    """True when the Hub says public, False for private/gated/missing, None when unaskable."""
    try:
        info = HfApi().repo_info(repo_id, repo_type = repo_type, token = False, timeout = 5.0)
    except Exception as exc:  # noqa: BLE001 - classified below, never trusted as public
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in _DEFINITIVE_HUB_STATUSES:
            return False
        return None
    return getattr(info, "private", None) is False and not getattr(info, "gated", False)


def _public_verdict(repo_id: str, repo_type: str) -> bool | None:
    key = (repo_type, repo_id.lower())
    with _public_lock:
        cached = _public_repos.get(key)
    return cached[1] if cached is not None and cached[0] > time.monotonic() else None


def repo_is_public(repo_id: str, repo_type: str = "model") -> bool:
    """Only an anonymous Hub answer proves a shared-cache repo public."""
    key = (repo_type, repo_id.lower())
    name = f"{repo_type}:{repo_id.lower()}"
    with _public_lock:
        cached = _public_repos.get(key)
        if cached is not None and cached[0] > time.monotonic():
            return cached[1]
        flight = _public_flights.get(key)
        leading = flight is None
        if leading:
            flight = _public_flights[key] = Future()
    if not leading:
        return flight.result()
    try:
        answer = _hub_public_answer(repo_id, repo_type)
        with _public_lock:
            if answer is None:
                public = name in _load_public_verdicts()
                ttl = _UNKNOWN_TTL
            else:
                public = answer
                _remember_public_verdict(name, public)
                ttl = _PUBLIC_TTL if public else _PRIVATE_TTL
            if len(_public_repos) >= 4096:
                _public_repos.clear()
            _public_repos[key] = (time.monotonic() + ttl, public)
    except BaseException as exc:
        flight.set_exception(exc)
        raise
    else:
        flight.set_result(public)
        return public
    finally:
        with _public_lock:
            _public_flights.pop(key, None)


def _hub_probe_targets(references, repo_type: str, grants: set[str]) -> set[str]:
    """Distinct repo ids ``model_visible`` would ask the Hub about, cheapest checks first."""
    candidates = {}
    for reference in references:
        if not isinstance(reference, str) or not reference:
            continue
        reference = reference.strip()
        if reference.startswith(("./", "../", "~")):
            continue
        cached = _cached_repo(Path(reference))
        if cached is not None and cached[1] == repo_type:
            repo_id = cached[0]
            if _grant_key(repo_id, repo_type) not in grants:
                candidates.setdefault(repo_id, repo_id)
            continue
        parts = reference.split(":", 1)[0].split("/")
        if len(parts) < 2 or not all(parts[:2]):
            continue
        repo_id = "/".join(parts[:2])
        if _grant_key(repo_id, repo_type) not in grants:
            candidates.setdefault(repo_id, reference)
    if not candidates:
        return set()
    now = time.monotonic()
    with _public_lock:
        unknown = {
            repo_id: reference
            for repo_id, reference in candidates.items()
            if (entry := _public_repos.get((repo_type, repo_id.lower()))) is None or entry[0] <= now
        }
    # A local path that happens to spell a repo id resolves against the cache, not the Hub.
    return {
        repo_id
        for repo_id, reference in unknown.items()
        if not Path(reference).expanduser().exists()
    }


def _warm_public_repos(repo_ids: set[str], repo_type: str) -> None:
    """Probe unknown repos concurrently; helpers are joined so none outlives the request."""
    if len(repo_ids) < 2:
        return
    pending = deque(repo_ids)

    def drain() -> None:
        while True:
            try:
                repo_id = pending.popleft()
            except IndexError:
                return
            try:
                repo_is_public(repo_id, repo_type)
            except Exception:  # noqa: BLE001 - the serial pass below asks again and decides
                pass

    helpers = [
        account_thread(target = drain, daemon = True, name = "studio-repo-probe")
        for _ in range(min(_PROBE_FANOUT, len(pending)) - 1)
    ]
    for helper in helpers:
        helper.start()
    try:
        drain()
    finally:
        for helper in helpers:
            helper.join()


def _grant_key(repo_id: str, repo_type: str) -> str:
    return f"{repo_type}:{repo_id.strip().lower()}"


def model_grants() -> set[str]:
    """This account's grants only; absent or malformed records confer no access."""
    path = studio_db_path()
    if not path.is_file():
        return set()
    try:
        with closing(sqlite3.connect(str(path))) as conn:
            row = conn.execute(
                "SELECT value_json FROM app_settings WHERE key = 'model_grants'"
            ).fetchone()
        grants = json.loads(row[0]) if row else []
        return (
            {key for key in grants if isinstance(key, str)} if isinstance(grants, list) else set()
        )
    except (sqlite3.Error, ValueError, TypeError):
        return set()


def record_model_grant(repo_id: str, repo_type: str = "model") -> None:
    """Record a grant in the initiating account's studio.db; transactional for concurrent writes."""
    if not managed_account() or not repo_id:
        return
    from core.training.account_jobs import account_is_retired

    if account_is_retired():
        return
    # Held across the write so a late completion cannot recreate a retired account's workspace.
    with storage_roots.root_retirement_lock:
        path = studio_db_path()
        try:
            storage_roots.ensure_account_dir(path.parent)
        except storage_roots.RetiredAccountError:
            return
        _write_grant(path, _grant_key(repo_id, repo_type))


def _write_grant(path: Path, key: str) -> None:
    with closing(sqlite3.connect(str(path), timeout = 5.0)) as conn, conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS app_settings (key TEXT NOT NULL PRIMARY KEY, value_json TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT value_json FROM app_settings WHERE key = 'model_grants'"
        ).fetchone()
        try:
            prior = json.loads(row[0]) if row else []
        except (ValueError, TypeError):
            prior = []
        grants = (
            {key for key in prior if isinstance(key, str)} if isinstance(prior, list) else set()
        )
        grants.add(key)
        conn.execute(
            "INSERT INTO app_settings (key, value_json, updated_at) VALUES ('model_grants', ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json, updated_at = excluded.updated_at",
            (json.dumps(sorted(grants)), datetime.now(timezone.utc).isoformat()),
        )


def repo_visible(
    repo_id: str,
    repo_type: str = "model",
    *,
    grants: set[str] | None = None,
) -> bool:
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
                pieces = part[len(prefix) :].split("--")
                if len(pieces) in (1, 2) and all(pieces):
                    return "/".join(pieces), repo_type
    return None


def model_visible(
    reference: str,
    *,
    grants: set[str] | None = None,
    repo_type: str = "model",
) -> bool:
    """Grants cover repo ids and cache snapshot/file spellings; other local paths stay private."""
    if not managed_account():
        return True
    if not isinstance(reference, str) or not reference:
        return False
    reference = reference.strip()
    path = Path(reference).expanduser()
    if path.is_absolute() or reference.startswith(("./", "../", "~")) or path.exists():
        try:
            resolved = path.resolve()
            own_roots = (workspace_root(), project_workspaces_root())
            if any(resolved.is_relative_to(root.resolve()) for root in own_roots):
                return True
            from utils.hf_cache_settings import known_hf_hub_caches

            if not any(resolved.is_relative_to(root.resolve()) for root in known_hf_hub_caches()):
                return False
            cached = _cached_repo(path)
            if cached is not None:
                # Snapshots point at their own repo's blobs; cross-repo links are refused.
                actual = _cached_repo(resolved)
                return actual == cached and repo_visible(cached[0], cached[1], grants = grants)
        except (OSError, RuntimeError, ValueError):
            return False
        return False
    repo_id = reference.split(":", 1)[0]
    parts = repo_id.split("/")
    if not all(parts[:2]):
        return False
    repo_id = "/".join(parts[:2])
    return repo_visible(repo_id, repo_type, grants = grants)


def require_model_access(reference: str, repo_type: str = "model") -> None:
    if not model_visible(reference, repo_type = repo_type):
        raise HTTPException(status_code = 404, detail = "Model not found")


def _row_reference(row):
    get = row.get if isinstance(row, dict) else lambda key, default = None: getattr(row, key, default)
    return get("path") or get("local_path") or get("repo_id") or get("model_id") or get("id")


def filter_model_rows(rows, *, repo_type: str = "model"):
    """Filter after shared scans/caches; never store a caller's filtered catalog globally."""
    if not managed_account():
        return rows
    grants = model_grants()
    rows = list(rows)
    references = [_row_reference(row) for row in rows]
    _warm_public_repos(_hub_probe_targets(references, repo_type, grants), repo_type)
    return [
        row
        for row, reference in zip(rows, references)
        if model_visible(reference, grants = grants, repo_type = repo_type)
    ]


def private_directory(path: str, folder: str) -> str:
    """Rebase import-time owner defaults; refuse account-external scans."""
    if not managed_account():
        return path
    from utils.paths.storage_roots import studio_root

    legacy = studio_root() / folder
    target = workspace_root() / folder if Path(path).resolve() == legacy.resolve() else Path(path)
    # The project workspace counts as the account's own, matching within_account().
    own_roots = (workspace_root(), project_workspaces_root())
    resolved = target.resolve()
    if not any(resolved.is_relative_to(root.resolve()) for root in own_roots):
        raise HTTPException(status_code = 404, detail = "Directory not found")
    return str(target)


def authorize_download(repo_id: str, repo_type: str, hf_token) -> None:
    """A cache hit is not proof the requester may read private Hub content."""
    if not managed_account():
        return
    try:
        api = HfApi()
        token = account_hf_token(hf_token)
        info = api.repo_info(repo_id, repo_type = repo_type, token = token, timeout = 5.0)
        if getattr(info, "gated", False):
            # Gated repo metadata is public even when the caller cannot read its files.
            api.auth_check(repo_id, repo_type = repo_type, token = token)
    except Exception as exc:  # noqa: BLE001 - a cached file is never a grant
        raise HTTPException(status_code = 404, detail = "Repository not found") from exc


def require_media_references(request) -> None:
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


def resident_components(status: dict, modality: str | None = None) -> list[str]:
    repo_id = status.get("repo_id")
    references = [repo_id, status.get("base_repo")]
    primary, components = _resident_components.get(modality, ("", frozenset()))
    # Baked adapters are absent from status(), so the load's authorized list stands in.
    if primary and repo_id and primary == repo_id:
        references.extend(sorted(components))
    return [ref for ref in references if isinstance(ref, str) and ref]


def require_media_generation_access(status: dict, modality: str | None = None) -> None:
    """Recheck the actual resident before a cache-only generation reuses it."""
    if managed_account() and status.get("loaded"):
        for reference in resident_components(status, modality):
            require_model_access(reference)


def foreign_work_active() -> bool:
    if not policy.installation_has_managed_accounts():
        return False
    from state import active_generations
    return bool(active_generations.foreign_count(current_account_id()))


def require_download_progress_access(
    registry,
    repo_id: str,
    repo_type: str = "model",
) -> None:
    if not managed_account():
        return
    from hub.services import download_lifecycle
    if not any(
        download_lifecycle.download_belongs_to_account(registry, ref.key)
        for ref in registry.active_job_refs(repo_id)
    ):
        require_model_access(repo_id, repo_type)


def require_media_adapters(request) -> None:
    if not managed_account():
        return
    for reference in media_adapter_references(request):
        require_model_access(reference)


def media_adapter_references(request) -> list[str]:
    """Resolve catalog aliases to the repo or path each adapter loads."""
    references: list[str] = []
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
            references.append(
                (entry.local_path or entry.repo_id) if entry is not None else selection.id
            )
    return references
