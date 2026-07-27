# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local-only snapshot resolution for background model loads."""

import os
from typing import Optional


def _snapshot_has_weights(rev: str) -> bool:
    """Whether a snapshot dir holds at least one safetensors weight file.

    Background loads only target safetensors-format rows (adapters and
    pickle checkpoints are excluded upstream), so this is the runnable-weight
    signal that made the inventory row eligible.
    """
    try:
        return any(name.endswith(".safetensors") for name in os.listdir(rev))
    except OSError:
        return False


def _snapshot_dir_fallback(repo_id: str, cache_dir: Optional[str]) -> Optional[str]:
    """Newest snapshots/* dir (by mtime) holding a config.json, preferring
    revisions that also hold weights.

    This mirrors the inventory scanner's selection (newest revision that
    actually carries the inventoried weights), so the load targets the
    snapshot that made the row eligible: a newest metadata-only revision must
    not shadow an older complete one. It also covers revision-only layouts
    (refs/ missing or pruned) that snapshot_download(local_files_only = True)
    cannot map through refs/main.
    """
    if cache_dir is None:
        try:
            from huggingface_hub.constants import HF_HUB_CACHE
            cache_dir = HF_HUB_CACHE
        except Exception:
            return None
    folder = os.path.join(cache_dir, "models--" + repo_id.replace("/", "--"), "snapshots")
    try:
        revisions = [
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, name))
        ]
    except OSError:
        return None
    candidates = [rev for rev in revisions if os.path.isfile(os.path.join(rev, "config.json"))]
    if not candidates:
        return None
    weightful = [rev for rev in candidates if _snapshot_has_weights(rev)]
    return max(weightful or candidates, key = os.path.getmtime)


def resolve_local_snapshot_path(
    repo_id: str,
    hf_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Optional[str]:
    """Resolve a Hub repo id to its snapshot directory in the local HF cache
    without any network access; None when the repo is not cached.

    The newest snapshot dir (by mtime) holding a config.json is preferred:
    that is the same selection the inventory scanner surfaces, so the load
    targets the revision that made the row eligible. refs/main can lag it
    when a non-main commit was downloaded later, so consulting refs first
    could load an older revision or fail on its missing files.
    ``snapshot_download(local_files_only = True)`` stays as the fallback for
    layouts the directory scan cannot interpret. Either way resolution never
    touches the network, and a missing-file snapshot still resolves so the
    subsequent weight load fails instead of downloading the gaps, which is
    the fail-closed behavior background loads need.
    """
    resolved = _snapshot_dir_fallback(repo_id, cache_dir)
    if resolved is not None:
        return resolved
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(
            repo_id,
            local_files_only = True,
            token = hf_token or None,
            cache_dir = cache_dir or None,
        )
    except Exception:
        return None
