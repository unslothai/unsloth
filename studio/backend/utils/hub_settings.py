# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The install's Hugging Face endpoint, saved in settings.

The saved values are applied as ``HF_ENDPOINT`` / ``HF_DATASETS_SERVER``, which
everything in Unsloth -- huggingface_hub, datasets, the browser via /api/health,
and every worker process spawned afterwards -- already follows. Until the owner
saves, whatever the operator exported stays in effect.
"""

from __future__ import annotations

import os
import sys
import threading
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

_ENV_VARS = ("HF_ENDPOINT", "HF_DATASETS_SERVER")
# What the operator exported before any saved value was applied over it.
_operator_env: dict[str, str | None] | None = None
_apply_lock = threading.Lock()


@dataclass(frozen = True)
class HubSettings:
    hf_endpoint: str
    datasets_server_follows_endpoint: bool
    saved: bool


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
    # get_app_settings creates and migrates studio.db; do not build one just to read two keys.
    if "storage.studio_db" not in sys.modules:
        try:
            from utils.paths.storage_roots import studio_db_path
            os.stat(studio_db_path())
        except FileNotFoundError:
            return {}
        except Exception:  # noqa: BLE001 - on any doubt, try the read
            pass
    try:
        from storage.studio_db import get_app_settings
        from utils.account_context import OWNER, run_as
        return run_as(OWNER, get_app_settings, [HF_ENDPOINT_KEY, DATASETS_SERVER_FOLLOWS_KEY])
    except Exception as exc:  # noqa: BLE001 - an unreadable db keeps the environment's values
        logger.debug("hub settings read failed (%s)", exc)
        return {}


def get_hub_settings() -> HubSettings:
    stored = _read_stored()
    endpoint = stored.get(HF_ENDPOINT_KEY)
    if not isinstance(endpoint, str):
        return HubSettings(_operator_endpoint(), False, saved = False)
    try:
        endpoint = validate_hub_endpoint(endpoint)
    except ValueError:
        endpoint = ""
    return HubSettings(endpoint, stored.get(DATASETS_SERVER_FOLLOWS_KEY) is True, saved = True)


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


def _effective_env(settings: HubSettings) -> dict[str, str | None]:
    operator = _capture_operator_env()
    if not settings.saved:
        return dict(operator)
    datasets_server = operator["HF_DATASETS_SERVER"]
    if settings.datasets_server_follows_endpoint and settings.hf_endpoint:
        datasets_server = settings.hf_endpoint
    return {"HF_ENDPOINT": settings.hf_endpoint or None, "HF_DATASETS_SERVER": datasets_server}


def apply_hub_settings() -> None:
    """Point this process, and every worker it spawns from now on, at the saved endpoints.

    Safe before huggingface_hub is imported (startup) and after (a save): the
    library reads the endpoint once at import, so its copies are refreshed too.
    Workers already running keep the endpoint they started with.
    """
    with _apply_lock:
        for name, value in _effective_env(get_hub_settings()).items():
            if value:
                os.environ[name] = value
            else:
                os.environ.pop(name, None)
        normalize_hf_endpoint_env()
        _refresh_imported_hub_libraries()
        utils_module = sys.modules.get("utils.utils")
        if utils_module is not None:
            # The memoised verdict was about the previous endpoint.
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
