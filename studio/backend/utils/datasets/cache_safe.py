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
    # datasets has already imported huggingface_hub by the time a cache link
    # fails, so update its live state before retrying too. Hub 1.x exposes a
    # disable constant; the Python 3.9 pin (0.36.2) does not and decides from
    # this per-directory capability cache instead.
    from huggingface_hub import constants, file_download

    if hasattr(constants, "HF_HUB_DISABLE_SYMLINKS"):
        constants.HF_HUB_DISABLE_SYMLINKS = True
    symlink_support = getattr(file_download, "_are_symlinks_supported_in_dir", None)
    if isinstance(symlink_support, dict):
        for cache_dir in tuple(symlink_support):
            symlink_support[cache_dir] = False


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
    except OSError as error:
        # Classify the exact Windows failure before generic PermissionError:
        # OSError's selected subclass for winerror=1314 varies by Python/runtime.
        if _is_windows_symlink_privilege_error(error):
            logger.warning(
                "Windows denied a Hugging Face cache symlink (%s); "
                "retrying with regular files",
                error,
            )
            _disable_hf_symlinks_for_process()
            try:
                return load_dataset(*args, **kwargs)
            except PermissionError as retry_error:
                return _retry_in_studio_cache(load_dataset, args, kwargs, retry_error)
        if isinstance(error, PermissionError):
            return _retry_in_studio_cache(load_dataset, args, kwargs, error)
        raise


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
