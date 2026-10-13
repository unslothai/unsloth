# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Precision folders of one Hub repo (``2-bit/``, ``6-bit/``, ``8bit/``), each a whole MLX
checkpoint, offered and loaded as variants of that repo the way GGUF quants are."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from hub.utils.snapshot_filters import PRECISION_FOLDER_RE


def is_precision_folder(variant: Optional[str]) -> bool:
    return bool(variant) and PRECISION_FOLDER_RE.fullmatch(str(variant).strip()) is not None


def cached_precision_folder(repo_id: str, folder: str) -> Optional[Path]:
    """The cached ``<snapshot>/<folder>`` when it holds a complete checkpoint, else None. No network."""
    if not is_precision_folder(folder):
        return None
    from utils.utils import checkpoint_directory_is_complete, hf_cache_snapshot_dir_for_repo

    snapshot = hf_cache_snapshot_dir_for_repo(repo_id)
    if snapshot is None:
        return None
    path = snapshot / folder
    if (path / "config.json").is_file() and checkpoint_directory_is_complete(path):
        return path
    return None


def precision_folder_config_dir(
    repo_id: str, folder: str, hf_token: Optional[str]
) -> Optional[Path]:
    """The local directory holding ``<folder>/config.json`` (fetching only that file), or None
    when the repo has no such folder, so a GGUF quant name never takes this path."""
    if not is_precision_folder(folder):
        return None
    cached = cached_precision_folder(repo_id, folder)
    if cached is not None:
        return cached
    from huggingface_hub import hf_hub_download

    from hub.utils.hf_tokens import call_with_anonymous_retry
    from utils.hf_cache_settings import active_hf_hub_cache

    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    try:
        config_path = call_with_anonymous_retry(
            lambda token: hf_hub_download(
                repo_id,
                f"{folder}/config.json",
                token = token,
                cache_dir = active_hf_hub_cache(),
            ),
            hf_token,
        )
    except EntryNotFoundError as exc:
        # Only a Hub that answered "no such file" means no folder. Any other failure must not
        # quietly load the root, which is a different precision.
        if isinstance(exc, LocalEntryNotFoundError):
            raise
        return None
    return Path(config_path).parent


def download_precision_folder(repo_id: str, folder: str, hf_token: Optional[str]) -> str:
    """Fetch only ``<folder>/*`` and return its local directory; files already cached are reused."""
    cached = cached_precision_folder(repo_id, folder)
    if cached is not None:
        return str(cached)
    from huggingface_hub import snapshot_download

    from hub.utils.hf_tokens import call_with_anonymous_retry
    from utils.hf_cache_settings import active_hf_hub_cache

    snapshot = call_with_anonymous_retry(
        lambda token: snapshot_download(
            repo_id,
            allow_patterns = [f"{folder}/*"],
            token = token,
            cache_dir = active_hf_hub_cache(),
        ),
        hf_token,
    )
    path = os.path.join(snapshot, folder)
    if not os.path.isfile(os.path.join(path, "config.json")):
        raise FileNotFoundError(f"{repo_id} has no checkpoint in its '{folder}' folder.")
    return path
