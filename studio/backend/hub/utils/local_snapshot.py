# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local-only snapshot resolution for background model loads."""

import os
from typing import Optional


def _snapshot_dir_fallback(repo_id: str, cache_dir: Optional[str]) -> Optional[str]:
    """Newest snapshots/* dir holding a config.json for a cache entry whose
    refs/ are missing or pruned. snapshot_download(local_files_only = True)
    needs refs/main to map the ref to a revision, but the inventory scanner
    accepts revision-only layouts, so background loads must resolve them too.
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
    return max(candidates, key = os.path.getmtime)


def resolve_local_snapshot_path(
    repo_id: str,
    hf_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Optional[str]:
    """Resolve a Hub repo id to its snapshot directory in the local HF cache
    without any network access; None when the repo is not cached.

    ``snapshot_download(local_files_only = True)`` reads only the on-disk
    refs/snapshots, so a cache populated outside Studio that is missing files
    still resolves to its snapshot directory; the subsequent weight load on
    that local path then fails instead of downloading the gaps, which is the
    fail-closed behavior background loads need.
    """
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(
            repo_id,
            local_files_only = True,
            token = hf_token or None,
            cache_dir = cache_dir or None,
        )
    except Exception:
        return _snapshot_dir_fallback(repo_id, cache_dir)
