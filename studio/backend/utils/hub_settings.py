# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The install's model source and Hugging Face endpoint, saved in settings.

The saved values are applied as ``HF_ENDPOINT`` / ``HF_DATASETS_SERVER``, which
everything in Unsloth -- huggingface_hub, datasets, the browser via /api/health,
and every worker process spawned afterwards -- already follows. Until the owner
saves, whatever the operator exported stays in effect. ModelScope as the source
points ``HF_ENDPOINT`` at the loopback adapter in ``hub.modelscope``.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
from contextlib import closing
from dataclasses import dataclass
from urllib.parse import urlsplit

from loggers import get_logger
from utils.hf_endpoint import (
    DEFAULTS_BY_HEALTH_KEY,
    normalize_hf_endpoint_env,
    validate_hub_endpoint,
)

logger = get_logger(__name__)

HF_ENDPOINT_KEY = "hub_hf_endpoint"
DATASETS_SERVER_FOLLOWS_KEY = "hub_datasets_server_follows_endpoint"
SOURCE_KEY = "hub_source"
HUGGINGFACE = "huggingface"
MODELSCOPE = "modelscope"
SOURCES = (HUGGINGFACE, MODELSCOPE)

_ENV_VARS = ("HF_ENDPOINT", "HF_DATASETS_SERVER")
SOURCE_ENV = "UNSLOTH_STUDIO_HUB_SOURCE"
_operator_env: dict[str, str | None] | None = None
_apply_lock = threading.Lock()


@dataclass(frozen = True)
class HubSettings:
    hf_endpoint: str
    datasets_server_follows_endpoint: bool
    saved: bool
    source: str = HUGGINGFACE


def _capture_operator_env() -> dict[str, str | None]:
    global _operator_env
    if _operator_env is None:
        _operator_env = {name: os.environ.get(name) for name in _ENV_VARS}
    return _operator_env


def _operator_endpoint() -> str:
    raw = _capture_operator_env()["HF_ENDPOINT"] or ""
    try:
        return validate_hub_endpoint(raw)
    except ValueError:
        return ""


def operator_hf_endpoint() -> str:
    """The endpoint the environment configured: the one state recorded before saved settings refers to."""
    return _operator_endpoint() or DEFAULTS_BY_HEALTH_KEY["hf_endpoint"]


def _read_stored() -> dict:
    keys = [HF_ENDPOINT_KEY, DATASETS_SERVER_FOLLOWS_KEY, SOURCE_KEY]
    try:
        from utils.account_context import OWNER, run_as

        # get_app_settings creates and migrates studio.db; the startup read must leave it untouched.
        if "storage.studio_db" not in sys.modules:
            from utils.paths.storage_roots import studio_db_path

            path = run_as(OWNER, studio_db_path)
            with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri = True)) as conn:
                rows = conn.execute(
                    "SELECT key, value_json FROM app_settings WHERE key IN (?, ?, ?)", keys
                ).fetchall()
            return {key: json.loads(value) for key, value in rows}
        from storage.studio_db import get_app_settings

        return run_as(OWNER, get_app_settings, keys)
    except Exception as exc:  # noqa: BLE001 - a missing or unreadable db keeps the environment's values
        logger.debug("hub settings read failed (%s)", exc)
        return {}


def get_hub_settings() -> HubSettings:
    stored = _read_stored()
    source = stored.get(SOURCE_KEY) if stored.get(SOURCE_KEY) in SOURCES else HUGGINGFACE
    endpoint = stored.get(HF_ENDPOINT_KEY)
    if not isinstance(endpoint, str):
        return HubSettings(_operator_endpoint(), False, saved = False, source = source)
    try:
        endpoint = validate_hub_endpoint(endpoint)
    except ValueError:
        endpoint = ""
    return HubSettings(
        endpoint, stored.get(DATASETS_SERVER_FOLLOWS_KEY) is True, saved = True, source = source
    )


def active_source() -> str:
    return MODELSCOPE if os.environ.get(SOURCE_ENV) == MODELSCOPE else HUGGINGFACE


def hugging_face_endpoint() -> str:
    """The Hugging Face endpoint the settings select, also while ModelScope serves."""
    return (get_hub_settings().hf_endpoint or DEFAULTS_BY_HEALTH_KEY["hf_endpoint"]).rstrip("/")


def set_hub_source(source: str) -> HubSettings:
    """Persist and apply the model source. Raises ValueError on an unknown one."""
    if source not in SOURCES:
        raise ValueError(f"Unknown model source {source!r}.")
    from storage.studio_db import upsert_app_settings

    upsert_app_settings({SOURCE_KEY: source}, read_back = False)
    apply_hub_settings()
    return get_hub_settings()


def set_hub_settings(hf_endpoint: str, datasets_server_follows_endpoint: bool) -> HubSettings:
    """Validate, persist and apply. Raises ValueError on an unusable endpoint."""
    endpoint = validate_hub_endpoint(hf_endpoint)
    from storage.studio_db import upsert_app_settings

    upsert_app_settings(
        {
            HF_ENDPOINT_KEY: endpoint,
            DATASETS_SERVER_FOLLOWS_KEY: bool(datasets_server_follows_endpoint),
        },
        read_back = False,
    )
    apply_hub_settings()
    return get_hub_settings()


def _effective_env(settings: HubSettings) -> tuple[dict[str, str | None], str]:
    """The environment for ``settings``, and the source it actually selects."""
    operator = _capture_operator_env()
    if settings.source == MODELSCOPE:
        try:
            from hub.modelscope.router import internal_endpoint
            env = {
                "HF_ENDPOINT": internal_endpoint(),
                "HF_DATASETS_SERVER": operator["HF_DATASETS_SERVER"],
            }
            return env, MODELSCOPE
        except Exception:  # noqa: BLE001 - a dead adapter must not take the backend down
            logger.exception("ModelScope adapter failed to start; using Hugging Face")
    if not settings.saved:
        return dict(operator), HUGGINGFACE
    datasets_server = operator["HF_DATASETS_SERVER"]
    if settings.datasets_server_follows_endpoint and settings.hf_endpoint:
        datasets_server = settings.hf_endpoint
    env = {"HF_ENDPOINT": settings.hf_endpoint or None, "HF_DATASETS_SERVER": datasets_server}
    return env, HUGGINGFACE


def apply_hub_settings() -> None:
    """Point this process, and every worker it spawns from now on, at the saved endpoints.

    Safe before huggingface_hub is imported (startup) and after (a save): the
    library reads the endpoint once at import, so its copies are refreshed too.
    Workers already running keep the endpoint they started with.
    """
    with _apply_lock:
        env, source = _effective_env(get_hub_settings())
        for name, value in env.items():
            if value:
                os.environ[name] = value
            else:
                os.environ.pop(name, None)
        os.environ[SOURCE_ENV] = source
        normalize_hf_endpoint_env()
        _refresh_imported_hub_libraries()
        utils_module = sys.modules.get("utils.utils")
        if utils_module is not None:
            utils_module.reset_hf_reachability_cache()


def _refresh_imported_hub_libraries() -> None:
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is None or getattr(constants, "_staging_mode", False):
        return
    endpoint = os.environ.get("HF_ENDPOINT", constants._HF_DEFAULT_ENDPOINT).rstrip("/")
    constants.ENDPOINT = endpoint
    constants.HUGGINGFACE_CO_URL_TEMPLATE = endpoint + "/{repo_id}/resolve/{revision}/{filename}"
    host = urlsplit(endpoint).hostname
    hosts = getattr(constants, "HF_URL_HOSTS", None)
    if host and isinstance(hosts, frozenset):
        constants.HF_URL_HOSTS = hosts | {host.lower()}
    hf_api = sys.modules.get("huggingface_hub.hf_api")
    if hf_api is not None and getattr(hf_api, "api", None) is not None:
        hf_api.api.endpoint = endpoint
    hf_file_system = sys.modules.get("huggingface_hub.hf_file_system")
    if hf_file_system is not None:
        # fsspec hands back the cached instance, which kept the old endpoint.
        hf_file_system.HfFileSystem.clear_instance_cache()
    datasets_config = sys.modules.get("datasets.config")
    if datasets_config is not None:
        datasets_config.HF_ENDPOINT = endpoint
        datasets_config.HUB_DATASETS_URL = (
            endpoint + "/datasets/{repo_id}/resolve/{revision}/{path}"
        )
