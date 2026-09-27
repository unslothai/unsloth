# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Account ownership and execution boundaries shared by long-running Studio jobs."""

from __future__ import annotations

import importlib
import os
import threading
import weakref
from functools import wraps
from pathlib import Path

from fastapi import HTTPException

from utils.account_context import AccountContext, OWNER, current_account, run_as
from utils.paths.storage_roots import (
    project_workspaces_root,
    resolve_export_write_dir,
    tmp_root,
    workspace_root,
)

_services = weakref.WeakSet()
_services_lock = threading.RLock()
_retired: set[str] = set()


def _multi_user() -> bool:
    # Lazy: the export worker imports this module on hosts without the auth stack.
    from auth import policy
    return policy.installation_is_multi_user()


def _has_managed_accounts() -> bool:
    from auth import policy
    return policy.installation_has_managed_accounts()


def managed_account() -> bool:
    return not current_account().is_owner


def account_key(value: str):
    """Keep legacy owner keys; namespace in-memory work by immutable account id."""
    account = current_account()
    return value if account.is_owner else (account.account_id, value)


def account_path(
    value,
    *,
    reference: bool = False,
    shared_cache: bool = False,
):
    """Validate a local path (symlinks resolved). Remote Hub ids only for ``reference`` fields."""
    if not value or not managed_account():
        return value
    raw = str(value)
    path = Path(raw).expanduser()
    if reference and not path.is_absolute() and not path.exists():
        parts = raw.split("/")
        if (
            len(parts) in (1, 2)
            and all(
                part and part not in {".", ".."} and all(c.isalnum() or c in "-_." for c in part)
                for part in parts
            )
            and not raw.startswith((".", "~"))
        ):
            return value
    resolved = path.resolve()

    roots = (workspace_root(), project_workspaces_root(), tmp_root())
    if shared_cache:
        from utils.hf_cache_settings import active_hf_hub_cache
        roots += (Path(active_hf_hub_cache()),)
    if not any(resolved.is_relative_to(root.resolve()) for root in roots):
        raise HTTPException(status_code = 403, detail = "Path is outside this account's workspace")
    return value


def export_write_path(value):
    if not value or not managed_account():
        return value
    if Path(str(value)).expanduser().is_absolute():
        return account_path(value)
    try:
        resolve_export_write_dir(str(value))
    except ValueError as exc:
        raise HTTPException(status_code = 403, detail = str(exc))
    return value


def visible_cached_path(value, repo_type: str = "model"):
    if not value or not managed_account():
        return value
    from hub.services.models import account_access

    if account_access.model_visible(str(value), repo_type = repo_type):
        return value
    return account_path(value)


def account_hf_token(token):
    """False explicitly disables the Hub's environment and saved-token fallback."""
    if managed_account():
        return token.strip() if isinstance(token, str) and token.strip() else False
    return token


def require_explicit_credentials(values: dict) -> None:
    if not managed_account():
        return
    if values.get("enable_wandb") and not str(values.get("wandb_token") or "").strip():
        raise HTTPException(status_code = 403, detail = "Supply your own W&B API key")
    for key in ("model_name", "base_model", "hf_dataset"):
        reference = values.get(key)
        if (
            reference
            and not Path(str(reference)).is_absolute()
            and not Path(str(reference)).exists()
        ):
            if not isinstance(values.get("hf_token"), str) or not values["hf_token"].strip():
                raise HTTPException(
                    status_code = 403,
                    detail = "Supply your own Hugging Face token for remote training resources",
                )
    if values.get("push_to_hub") and not values.get("hf_token"):
        raise HTTPException(status_code = 403, detail = "Supply your own Hugging Face token to publish")
    s3 = values.get("s3_dataset") or values.get("s3_config")
    if hasattr(s3, "model_dump"):
        s3 = s3.model_dump()
    if s3 and (
        s3.get("use_iam_role") or not s3.get("access_key_id") or not s3.get("secret_access_key")
    ):
        raise HTTPException(status_code = 403, detail = "Supply your own AWS access keys")


def validate_job_paths(values: dict, *, cached_resources: bool = False) -> None:
    if not managed_account():
        return
    for key in ("model_name", "base_model", "base_model_id", "hf_dataset"):
        account_path(values.get(key), reference = True)
    for key, repo_type in (
        ("model_local_path", "model"),
        ("dataset_local_path", "dataset"),
        ("model_snapshot_path", "model"),
        ("dataset_snapshot_path", "dataset"),
    ):
        value = values.get(key)
        if value and not cached_resources:
            visible_cached_path(value, repo_type)
            continue
        account_path(value, shared_cache = cached_resources)
    for key, claimed, repo_type in (
        ("model_name", "model_known_cached", "model"),
        ("hf_dataset", "dataset_known_cached", "dataset"),
    ):
        reference = values.get(key)
        # A cached claim pins the shared cache, so the caller's token never authorizes it.
        if values.get(claimed) and reference:
            from hub.services.models import account_access
            account_access.require_model_access(str(reference), repo_type)
    for key in (
        "checkpoint_path",
        "resume_from_checkpoint",
        "output_dir",
        "tensorboard_dir",
        "imatrix_path",
        "data_dir",
        "conditioning_cache_dir",
        "cond_cache_dir",
        "vae_path",
        "text_encoder_path",
        "dataset_path",
    ):
        account_path(values.get(key))
    export_write_path(values.get("save_directory"))
    if isinstance(values.get("imatrix_file"), (str, Path)):
        account_path(values["imatrix_file"])
    for key in ("local_datasets", "local_eval_datasets"):
        for value in values.get(key) or ():
            account_path(value)
    require_explicit_credentials(values)


def init_job_owner(
    service,
    active,
    cancel,
    clear = None,
) -> None:
    service.job_account = None
    service._result_account = OWNER
    service._account_job_lock = threading.RLock()
    service._account_inflight = 0
    service._account_active = active
    service._account_cancel = cancel
    service._account_clear = clear
    with _services_lock:
        _services.add(service)


def job_is_foreign(service) -> bool:
    owner = getattr(service, "_result_account", OWNER)
    return (
        isinstance(owner, AccountContext)
        and owner.account_id != current_account().account_id
        and _has_managed_accounts()
    )


def require_job_owner(service) -> None:
    if job_is_foreign(service):
        raise HTTPException(status_code = 404, detail = "Job not found")


def refresh_job_owner(service) -> None:
    with service._account_job_lock:
        if not service._account_inflight and not service._account_active():
            service.job_account = None


def owned_job(*, continuation: bool = False):
    """Hold ownership across validation/spawn and while work is live. Skipped when no managed
    account ever existed."""

    def decorate(fn):
        @wraps(fn)
        def wrapped(self, *args, **kwargs):
            if not _has_managed_accounts():
                return fn(self, *args, **kwargs)
            account = current_account()
            with self._account_job_lock:
                if account.account_id in _retired:
                    raise HTTPException(status_code = 403, detail = "Account is retired")
                if continuation:
                    require_job_owner(self)
                if job_is_foreign(self) and (self._account_inflight or self._account_active()):
                    raise HTTPException(
                        status_code = 409, detail = {"code": "job_busy", "message": "Busy"}
                    )
                if job_is_foreign(self) and self._account_clear is not None:
                    self._account_clear()
                self._result_account = self.job_account = account
                self._account_inflight += 1
            try:
                return fn(self, *args, **kwargs)
            finally:
                with self._account_job_lock:
                    self._account_inflight -= 1
                refresh_job_owner(self)

        return wrapped

    return decorate


def job_control(fn):
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        lock = getattr(self, "_account_job_lock", None)
        if lock is None or not _has_managed_accounts():
            return fn(self, *args, **kwargs)
        with lock:
            require_job_owner(self)
            self._account_inflight += 1
        try:
            return fn(self, *args, **kwargs)
        finally:
            with lock:
                self._account_inflight -= 1
            refresh_job_owner(self)

    return wrapped


def job_read(neutral):
    def decorate(fn):
        @wraps(fn)
        def wrapped(self, *args, **kwargs):
            lock = getattr(self, "_account_job_lock", None)
            if lock is None or not _has_managed_accounts():
                return fn(self, *args, **kwargs)
            with lock:
                if self.job_account is not None:
                    refresh_job_owner(self)
                if job_is_foreign(self):
                    return neutral(self, *args, **kwargs)
                return fn(self, *args, **kwargs)

        return wrapped

    return decorate


def job_pump(fn):
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        finally:
            if getattr(self, "job_account", None) is not None:
                refresh_job_owner(self)

    return wrapped


def account_process_spec(module: str, target: str, env: dict, kwargs: dict):
    if account_is_retired():
        raise HTTPException(status_code = 403, detail = "Account is retired")
    if not managed_account():
        return (module, target, env), kwargs
    return ("core.training.account_jobs", "run_account_child", env), {
        **kwargs,
        "account": current_account(),
        "job_module": module,
        "job_target": target,
    }


# Model provider credentials the recipe engine reads straight from the environment.
_PROVIDER_SECRET_ENV_VARS = frozenset(
    {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "NVIDIA_API_KEY",
        "NGC_API_KEY",
        "OPENROUTER_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "MISTRAL_API_KEY",
        "TOGETHER_API_KEY",
        "GROQ_API_KEY",
        "XAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "FIREWORKS_API_KEY",
    }
)


def run_account_child(*, account: AccountContext, job_module: str, job_target: str, **kwargs):
    def execute():
        if not account.is_owner:
            # Child-only: never mutate credentials in the multithreaded server.
            for key in tuple(os.environ):
                if (
                    key.startswith(("AWS_", "WANDB_"))
                    or key
                    in {
                        "HF_TOKEN",
                        "HF_HUB_TOKEN",
                        "HUGGINGFACE_HUB_TOKEN",
                        "HUGGINGFACEHUB_API_TOKEN",
                        "HUGGING_FACE_HUB_TOKEN",
                        "GH_TOKEN",
                        "GITHUB_TOKEN",
                    }
                    or key in _PROVIDER_SECRET_ENV_VARS
                ):
                    os.environ.pop(key, None)
            private_tmp = tmp_root()
            private_tmp.mkdir(parents = True, exist_ok = True)
            os.environ.update(
                HF_HUB_DISABLE_IMPLICIT_TOKEN = "1",
                HF_TOKEN_PATH = str(private_tmp / ".no-ambient-hf-token"),
                AWS_EC2_METADATA_DISABLED = "true",
                AWS_SHARED_CREDENTIALS_FILE = str(private_tmp / ".no-ambient-aws-credentials"),
                AWS_CONFIG_FILE = str(private_tmp / ".no-ambient-aws-config"),
                TMPDIR = str(private_tmp),
            )
        target = getattr(importlib.import_module(job_module), job_target)
        return target(**kwargs)

    return run_as(account, execute)


class AccountRetirementError(RuntimeError):
    """A worker or job of the account could not be stopped; its files stay in place."""


def retire_account_jobs(account: AccountContext) -> None:
    """Revoke new starts and cancel only this account's work. Call before renaming its dirs."""
    with _services_lock:
        _retired.add(account.account_id)
        services = list(_services)
    errors = []
    for service in services:
        try:
            with service._account_job_lock:
                if service._result_account.account_id == account.account_id:
                    if service._account_inflight:
                        raise RuntimeError("Account job operation is in flight; retry retirement")
                    try:
                        run_as(account, service._account_cancel)
                    finally:
                        proc = getattr(service, "_proc", None)
                        if proc is not None and proc.is_alive():
                            proc.terminate()
                            proc.join(timeout = 5)
                            if proc.is_alive():
                                proc.kill()
                                proc.join(timeout = 3)
                            if proc.is_alive():
                                raise RuntimeError("Retired account worker has not stopped")
        except Exception as exc:
            errors.append(exc)
    from hub.services.datasets.downloads import retire_account_downloads
    from hub.services.models.downloads import (
        retire_account_downloads as retire_account_model_downloads,
    )
    from core.rag import folder_sync, ingestion
    from core.research_runs import retire_account_research
    from routes.inference import retire_stt_downloads

    for cancel in (
        retire_account_downloads,
        retire_account_model_downloads,
        retire_stt_downloads,
        ingestion.retire_account_ingestions,
        folder_sync.retire_account_sync,
        lambda: retire_account_research(account),
    ):
        try:
            run_as(account, cancel)
        except Exception as exc:
            errors.append(exc)
    if errors:
        raise AccountRetirementError(
            "Could not retire every account job; keep its directories in place"
        ) from errors[0]


def restore_account_jobs(account_id: str) -> None:
    with _services_lock:
        _retired.discard(account_id)


def account_is_retired() -> bool:
    return current_account().account_id in _retired


def job_accounts() -> list[AccountContext]:
    if not _multi_user():
        return [OWNER]
    from auth import storage

    conn = storage.get_connection()
    try:
        rows = conn.execute(
            "SELECT account_id, username, role FROM auth_user WHERE is_active = 1"
        ).fetchall()
        return [AccountContext(*row) for row in rows if row[0] not in _retired]
    finally:
        conn.close()


def _inactive_job_accounts() -> list[AccountContext]:
    if not _has_managed_accounts():
        return []
    from auth import storage

    conn = storage.get_connection()
    try:
        rows = conn.execute(
            "SELECT account_id, username, role FROM auth_user WHERE is_active = 0"
        ).fetchall()
        return [AccountContext(*row) for row in rows if row[0] not in _retired]
    finally:
        conn.close()


def startup_reconciliation_accounts() -> list[AccountContext]:
    """Accounts boot-time reconciliation visits; one with no db is skipped (opening creates it)."""
    from utils.paths.storage_roots import studio_db_path

    accounts: list[AccountContext] = []
    seen: set[str] = set()
    for account in (*job_accounts(), *_inactive_job_accounts()):
        if account.account_id in seen:
            continue
        seen.add(account.account_id)
        if account.is_owner:
            accounts.append(account)
            continue
        try:
            if run_as(account, studio_db_path).is_file():
                accounts.append(account)
        except (OSError, ValueError):
            continue
    return accounts


def sweepable_job_accounts() -> list[AccountContext]:
    """Boot-reconcile set: a deactivated account still holds a registration and GPU reservation."""
    return startup_reconciliation_accounts()


def validate_recipe_access(recipe) -> None:
    if not managed_account():
        return
    if isinstance(recipe, list):
        for value in recipe:
            validate_recipe_access(value)
    elif isinstance(recipe, dict):
        seed_type = recipe.get("seed_type")
        if seed_type in {"hf", "github_repo"} and not str(recipe.get("token") or "").strip():
            raise HTTPException(status_code = 403, detail = "Supply your own seed-source token")
        for key, value in recipe.items():
            if key in {"api_key_env", "token_env", "hf_token_env"} and value:
                raise HTTPException(status_code = 403, detail = "Supply your own provider credential")
            # Env-resolved before literals: a bare variable name would send the host credential.
            if key in {"api_key", "token", "hf_token"} and isinstance(value, str):
                if value in os.environ:
                    raise HTTPException(
                        status_code = 403, detail = "Supply your own provider credential"
                    )
            if key in {
                "path",
                "paths",
                "file_path",
                "file_paths",
                "data_files",
                "local_path",
                "artifact_path",
                "base_path",
            }:
                for path in value if isinstance(value, list) else [value]:
                    if isinstance(path, str):
                        if seed_type == "hf" and key == "path":
                            remote = path.removeprefix("hf://")
                            parts = remote.split("/")
                            if (
                                not remote.startswith("datasets/")
                                or len(parts) < 3
                                or any(part in {"", ".", ".."} for part in parts)
                                or "\\" in remote
                            ):
                                raise HTTPException(
                                    status_code = 403, detail = "Invalid Hugging Face seed path"
                                )
                        else:
                            account_path(path)
            if isinstance(value, (dict, list)):
                validate_recipe_access(value)


async def account_event_stream(service, events):
    """Recheck ownership for every frame, including streams opened before a start."""
    try:
        async for event in events:
            if job_is_foreign(service):
                yield 'event: busy\ndata: {"status":"busy"}\n\n'
                return
            yield event
    finally:
        await events.aclose()


def job_busy(service) -> bool:
    active = getattr(service, "_account_active", None)
    return bool(getattr(service, "_account_inflight", 0) or (active and active()))


def worker_alive(
    service,
    *,
    process: str = "_proc",
    pump: str = "_pump_thread",
) -> bool:
    """A finalizer still writing the prior result also owns the service slot."""
    proc = getattr(service, process, None)
    thread = getattr(service, pump, None)
    return bool(
        (proc is not None and proc.is_alive())
        or (thread is not None and thread is not threading.current_thread() and thread.is_alive())
    )
