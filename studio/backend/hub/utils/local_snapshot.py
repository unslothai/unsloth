# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local-only snapshot resolution for background model loads."""

from typing import Optional


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
        return None
