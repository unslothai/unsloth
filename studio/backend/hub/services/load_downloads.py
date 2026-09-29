# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from typing import Optional, Sequence

from hub.services import download_lifecycle
from hub.utils import download_registry
from hub.utils.hf_cache_state import TRANSPORT_HTTP, TRANSPORT_XET

LOAD_OWNER = "load"
_ACTIVE = ("running", "cancelling")


def _job_key(repo_id: str) -> str:
    return download_registry.normalize_job_key(f"{download_registry.normalize_repo_key(repo_id)}::")


def is_load_owned(registry: download_registry.DownloadRegistry, key: str) -> bool:
    metadata = registry.get_job_metadata(key)
    if metadata is None or metadata.owner != LOAD_OWNER:
        return False
    return registry.get_job(key).state in _ACTIVE


def claim_load_downloads(
    repo_ids: Sequence[str],
    *,
    xet_disabled: bool = False,
    hub_cache: Optional[str] = None,
) -> list[str]:
    registry = download_registry.get_models_registry()
    transport = TRANSPORT_HTTP if xet_disabled else TRANSPORT_XET
    keys: list[str] = []
    for repo_id in repo_ids:
        key = _job_key(repo_id)
        accepted, _state = registry.claim(
            key,
            transport,
            repo_type = "model",
            repo_id = repo_id,
            hub_cache = hub_cache,
            owner = LOAD_OWNER,
        )
        if accepted:
            # The load runs as its caller, so a managed account sees its own load's downloads.
            download_lifecycle.record_download_account(registry, key)
            keys.append(key)
            continue
        for ref in registry.active_job_refs(repo_id):
            registry.mark_load_attached(ref.key, True)
            if ref.key not in keys:
                keys.append(ref.key)
    return keys


def release_load_downloads(keys: Sequence[str]) -> None:
    registry = download_registry.get_models_registry()
    for key in keys:
        if not registry.release_owned(key, LOAD_OWNER):
            registry.mark_load_attached(key, False)
