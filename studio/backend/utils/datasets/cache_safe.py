# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Permission-safe wrapper around datasets.load_dataset.

A shared HF datasets cache can contain subtrees owned by another user (for
example populated by an earlier root-run job). datasets then raises
"[Errno 13] Permission denied: ..._builder.lock" while locking the cached
builder, killing the training run even though the dataset itself is fine.
Retry such loads in an Unsloth-owned cache so the run proceeds; the worst case
is one rebuild of the dataset in the fallback location.

On Windows, huggingface_hub's concurrent symlink capability probe can also
briefly publish a false-positive result and raise WinError 1314. Only after
that exact failure, retry with its regular-file cache mode for this worker.
"""

import logging
import os

from utils.paths.storage_roots import cache_root

logger = logging.getLogger(__name__)

_WINDOWS_SYMLINK_PRIVILEGE_ERROR = 1314


def _is_native_windows() -> bool:
    return os.name == "nt"


def _is_windows_symlink_privilege_error(error: OSError) -> bool:
    return _is_native_windows() and (
        getattr(error, "winerror", None) == _WINDOWS_SYMLINK_PRIVILEGE_ERROR
    )


def _disable_hf_symlinks_for_process() -> None:
    """Switch an affected worker to HF's regular-file cache fallback."""
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    # huggingface_hub reads this environment variable at import time. datasets
    # has already imported it by the time a cache link fails, so update the live
    # constant before retrying too.
    from huggingface_hub import constants

    constants.HF_HUB_DISABLE_SYMLINKS = True


def studio_datasets_cache() -> str:
    path = cache_root() / "hf-datasets"
    path.mkdir(parents = True, exist_ok = True)
    return str(path)


def load_dataset_cache_safe(*args, **kwargs):
    """Load a dataset with narrow retries for known cache permission failures."""
    from datasets import load_dataset

    # datasets is in sys.modules exactly now, which is what lets its bar class be
    # patched; the server never imports it at boot, so this shared entry point is
    # where its "Generating train split" bar stops reaching the structured log.
    from loggers.config import quiet_third_party_progress_bars

    quiet_third_party_progress_bars()

    try:
        return load_dataset(*args, **kwargs)
    except PermissionError as error:
        return _retry_in_studio_cache(load_dataset, args, kwargs, error)
    except OSError as error:
        if not _is_windows_symlink_privilege_error(error):
            raise
        logger.warning(
            "Windows denied a Hugging Face cache symlink (%s); retrying with regular files",
            error,
        )
        _disable_hf_symlinks_for_process()
        try:
            return load_dataset(*args, **kwargs)
        except PermissionError as retry_error:
            return _retry_in_studio_cache(load_dataset, args, kwargs, retry_error)


def _retry_in_studio_cache(load_dataset, args, kwargs, error):
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
