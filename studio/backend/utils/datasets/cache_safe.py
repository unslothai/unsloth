# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Retries load_dataset in an Unsloth-owned cache on Errno 13 from a shared HF cache; on Windows,
only WinError 1314 retries in regular-file cache mode for this worker. A local JSON load whose later
rows add keys is retried with a schema covering every row."""

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


def _is_retryable_cache_error(error: OSError) -> bool:
    return isinstance(error, PermissionError) or _is_windows_symlink_privilege_error(error)


class _NoSymlinkSupport(dict):
    """Answers "already probed, unsupported" for every cache dir.

    Hub before 1.9 has no disable flag and re-probes any dir missing from this
    mapping, losing the same race again, so leave it nothing to probe.
    """

    def __contains__(self, cache_dir) -> bool:
        return True

    def __missing__(self, cache_dir) -> bool:
        return False


def _disable_hf_symlinks_for_process() -> None:
    """Switch an affected worker to HF's regular-file cache fallback."""
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    # huggingface_hub is already imported, so its live state must be updated too; Hub 1.9 added the constant.
    try:
        from huggingface_hub import constants, file_download
    except ImportError:  # never mask the load error we are recovering
        return

    if hasattr(constants, "HF_HUB_DISABLE_SYMLINKS"):
        constants.HF_HUB_DISABLE_SYMLINKS = True
    symlink_support = getattr(file_download, "_are_symlinks_supported_in_dir", None)
    if isinstance(symlink_support, dict):
        # Flipped in place too, for anything already holding the old dict.
        for cache_dir in tuple(symlink_support):
            symlink_support[cache_dir] = False
        file_download._are_symlinks_supported_in_dir = _NoSymlinkSupport(symlink_support)


def studio_datasets_cache() -> str:
    path = cache_root() / "hf-datasets"
    path.mkdir(parents = True, exist_ok = True)
    return str(path)


def load_dataset_cache_safe(*args, **kwargs):
    """Load a dataset with narrow retries for known cache permission and JSON schema failures."""
    from datasets import load_dataset

    # datasets is in sys.modules exactly now.
    from loggers.config import quiet_third_party_progress_bars

    quiet_third_party_progress_bars()

    try:
        return _load_with_cache_retries(load_dataset, args, kwargs)
    except Exception as error:
        features = _json_retry_features(error, args, kwargs)
        if features is None:
            raise
        logger.warning(
            "Rows of %s do not share one structure (%s); reloading with a schema covering every row",
            kwargs["data_files"],
            error.__cause__ or error,
        )
        return _load_with_cache_retries(load_dataset, args, {**kwargs, "features": features})


def _json_retry_features(error, args, kwargs):
    """Features for a local JSON load that failed on rows typed unlike the first block, else None."""
    loader = args[0] if args else kwargs.get("path")
    files = kwargs.get("data_files")
    if loader != "json" or kwargs.get("features") is not None or kwargs.get("streaming"):
        return None
    if isinstance(files, (str, os.PathLike)):
        files = [files]
    if not isinstance(files, (list, tuple)) or not files:
        return None
    if not all(isinstance(path, (str, os.PathLike)) and os.path.isfile(path) for path in files):
        return None
    from datasets.exceptions import DatasetGenerationError

    if not isinstance(error, DatasetGenerationError):
        return None
    from utils.datasets.json_rows import json_rows_features

    return json_rows_features(files)


def _load_with_cache_retries(load_dataset, args, kwargs):
    try:
        return load_dataset(*args, **kwargs)
    except OSError as error:
        # Classify winerror 1314 first: the subclass Python picks for it varies.
        if _is_windows_symlink_privilege_error(error):
            logger.warning(
                "Windows denied a Hugging Face cache symlink (%s); retrying with regular files",
                error,
            )
            _disable_hf_symlinks_for_process()
            try:
                return load_dataset(*args, **kwargs)
            except OSError as retry_error:
                # A second 1314 is a cache dir Hub had not probed; the Unsloth-owned cache is probed fresh too.
                if _is_retryable_cache_error(retry_error):
                    return _retry_in_studio_cache(load_dataset, args, kwargs, retry_error)
                raise
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
    # Nested builders read the env var during the load; restore it after so other datasets still try the shared cache first.
    old_env = os.environ.get("HF_DATASETS_CACHE")
    os.environ["HF_DATASETS_CACHE"] = fallback
    try:
        return load_dataset(*args, **kwargs)
    finally:
        if old_env is None:
            os.environ.pop("HF_DATASETS_CACHE", None)
        else:
            os.environ["HF_DATASETS_CACHE"] = old_env
