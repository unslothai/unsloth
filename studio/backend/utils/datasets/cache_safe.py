# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Permission-safe wrapper around datasets.load_dataset.

A shared HF datasets cache can contain subtrees owned by another user (for
example populated by an earlier root-run job). datasets then raises
"[Errno 13] Permission denied: ..._builder.lock" while locking the cached
builder, killing the training run even though the dataset itself is fine.
Retry such loads in an Unsloth-owned cache so the run proceeds; the worst case
is one rebuild of the dataset in the fallback location.
"""

import logging
import os

from utils.paths.storage_roots import cache_root

logger = logging.getLogger(__name__)


def studio_datasets_cache() -> str:
    path = cache_root() / "hf-datasets"
    path.mkdir(parents = True, exist_ok = True)
    return str(path)


def load_dataset_cache_safe(*args, **kwargs):
    """datasets.load_dataset, retried in an Unsloth-owned cache on EACCES."""
    from datasets import load_dataset
    try:
        return load_dataset(*args, **kwargs)
    except PermissionError as error:
        fallback = studio_datasets_cache()
        logger.warning(
            "HF datasets cache is not writable (%s); rebuilding in %s",
            error,
            fallback,
        )
        kwargs["cache_dir"] = fallback
        # Nested builders consult the env var while the load runs; restore it
        # after so other datasets keep trying the shared cache first.
        old_env = os.environ.get("HF_DATASETS_CACHE")
        os.environ["HF_DATASETS_CACHE"] = fallback
        try:
            return load_dataset(*args, **kwargs)
        finally:
            if old_env is None:
                os.environ.pop("HF_DATASETS_CACHE", None)
            else:
                os.environ["HF_DATASETS_CACHE"] = old_env
