# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Training downloads share the Hub registry; cancel kills the loader subprocess, not a queued
signal; progress covers current files only and leaving a load phase never verifies the full repo."""

from __future__ import annotations

import weakref
import re
from typing import Optional

from hub.services import download_lifecycle
from hub.utils import download_registry
from hub.utils.paths import is_valid_repo_id

OWNER = "training"
PREFIX = "@training-"
_services: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


def is_training_variant(variant: Optional[str]) -> bool:
    return bool(re.fullmatch(r"@training-[0-9a-f]{32}", variant or ""))


def claim(service, job_id: str, repo_id: Optional[str]) -> Optional[str]:
    if not repo_id or not is_valid_repo_id(repo_id) or not is_training_variant(f"{PREFIX}{job_id}"):
        return None
    registry = download_registry.get_models_registry()
    variant = f"{PREFIX}{job_id}"
    key = download_registry.normalize_job_key(f"{repo_id}::{variant}")
    metadata = registry.get_job_metadata(key)
    if (
        metadata is not None
        and metadata.owner == OWNER
        and registry.get_job(key).state == "running"
    ):
        return key
    accepted, state = registry.claim(
        key,
        download_registry.TRANSPORT_AUTO,
        repo_type = "model",
        repo_id = repo_id,
        variant = variant,
        owner = OWNER,
    )
    if not accepted:
        raise RuntimeError(f"Model files are already in use by another download ({state}).")
    download_lifecycle.record_download_account(registry, key)
    _services[key] = service
    return key


def settle(
    key: Optional[str],
    outcome: str = "idle",
    error: Optional[str] = None,
) -> None:
    if key is None:
        return
    registry = download_registry.get_models_registry()
    metadata = registry.get_job_metadata(key)
    if metadata is None or metadata.owner != OWNER:
        return
    if outcome == "idle":
        registry.release_owned(key, OWNER)
    else:
        registry.set_job(key, outcome, error)
    _services.pop(key, None)


def cancel(key: str, generation: Optional[int]) -> str:
    registry = download_registry.get_models_registry()
    download_lifecycle.require_download_account(registry, key)
    if generation is None or generation != registry.current_generation(key):
        return registry.get_job(key).state
    metadata = registry.get_job_metadata(key)
    service = _services.get(key)
    if metadata is None or metadata.owner != OWNER or service is None:
        return registry.get_job(key).state
    job_id = metadata.variant[len(PREFIX) :]
    service.stop(save = False, job_id = job_id, download_only = True, download_generation = generation)
    return registry.get_job(key).state


def progress(key: str) -> dict:
    registry = download_registry.get_models_registry()
    download_lifecycle.require_download_account(registry, key)
    service = _services.get(key)
    snapshot = service.status() if service is not None else {}
    metadata = registry.get_job_metadata(key)
    same_job = metadata is not None and metadata.variant == f"{PREFIX}{snapshot.get('job_id')}"
    reading = (
        snapshot.get("download_progress") if same_job and snapshot.get("in_model_load") else None
    )
    done = reading.get("bytes_done", 0) if reading else 0
    total = reading.get("bytes_total", 0) if reading else 0
    return {
        "downloaded_bytes": done,
        "expected_bytes": total,
        "completed_bytes": 0,
        "complete_on_disk": False,
        "progress": done / total if total else 0,
        "cache_path": None,
        "cache_measured": True,
        "target_present": bool(reading),
        "progress_scope": "active_files",
    }
