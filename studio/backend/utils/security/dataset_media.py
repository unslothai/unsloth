# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep a managed account's dataset media to files it may read, and validate Arrow data files.

A ``datasets`` Image / Audio / Video / Pdf / Nifti cell holding a ``path`` and no ``bytes`` is opened by the
feature's decoder in whichever process reads the row: a local path is read from disk, and a Hub URL is fetched
with ``token_per_repo_id[repo]``, which falls back to the process's own Hub credentials for any repo the dataset
was not loaded from. Any dataset can carry such cells, so for a managed account a local path must be one the
account may read (its own roots, a Hub snapshot it can see or its training dataset's, an archive ``datasets``
extracted), a Hub URL is fetched with the token the caller supplied for its repo or anonymously, and any other
URL is refused. The owner is unchanged.

pyarrow's IPC reader trusts the lengths and offsets an ``.arrow`` file declares, so the ``arrow`` builder validates
each table in full before ``datasets`` converts its rows: a malformed file raises ``ArrowInvalid`` (a ValueError)
instead of reading past its buffers. ``datasets``' own cache files are memory-mapped by its reader, not this builder.

``install()`` patches ``datasets`` once it is imported, without importing it at startup; it is idempotent.
"""

import functools
import importlib.abc
import importlib.util
import os
import re
import sys

_DECODERS = {
    "datasets.features.image": "Image",
    "datasets.features.audio": "Audio",
    "datasets.features.video": "Video",
    "datasets.features.pdf": "Pdf",
    "datasets.features.nifti": "Nifti",
}
_ARROW = "datasets.packaged_modules.arrow.arrow"
_MARK = "_unsloth_dataset_media_guard"
_SNAPSHOT_REPO = re.compile(r"[\\/]datasets--([^\\/]+?)--([^\\/]+)[\\/]")
# Hub dataset repos the process was started to train on (a worker serves one account's job).
_dataset_repos: set[str] = set()


class MediaPathRefused(PermissionError, ValueError):
    """A ValueError too, so check-format answers 400 like any other unreadable dataset."""


def _is_owner() -> bool:
    try:
        from utils.account_context import is_owner_context
    except ImportError:  # outside Studio's backend there are no managed accounts
        return True
    return is_owner_context()


def allow_dataset_repos(*repo_ids) -> None:
    """Snapshot files of these Hub datasets may be decoded: the job loaded them with the account's own token."""
    _dataset_repos.update(
        str(repo).strip() for repo in repo_ids if isinstance(repo, str) and repo.strip()
    )


def _refuse(path) -> None:
    raise MediaPathRefused(f"Dataset media path is not readable by this account: {path}")


def _training_snapshot(path: str) -> bool:
    if not _dataset_repos:
        return False
    from utils.hf_cache_settings import known_hf_hub_caches

    real = os.path.realpath(path)
    lexical, resolved = _SNAPSHOT_REPO.search(path), _SNAPSHOT_REPO.search(real)
    if not (lexical and resolved) or lexical.groups() != resolved.groups():
        return False
    caches = [os.path.realpath(root) for root in known_hf_hub_caches()]
    return "/".join(resolved.groups()) in _dataset_repos and any(
        real.startswith(root.rstrip(os.sep) + os.sep) for root in caches
    )


def _extracted_archive(path: str) -> bool:
    from datasets import config
    root = os.path.realpath(str(config.EXTRACTED_DATASETS_PATH))
    return os.path.realpath(path).startswith(root.rstrip(os.sep) + os.sep)


def _require_readable(path: str) -> None:
    from fastapi import HTTPException

    from core.training.account_jobs import visible_cached_path

    path = os.path.abspath(os.path.expanduser(path))
    try:
        visible_cached_path(path, "dataset")
    except HTTPException:
        if not (_training_snapshot(path) or _extracted_archive(path)):
            _refuse(path)


def require_readable_media(path) -> None:
    """Refuse a local media path the managed account may not read; a no-op for the owner."""
    if not _is_owner():
        _require_readable(str(path))


def _permitted_tokens(value, token_per_repo_id):
    """The token mapping the decoder may use for ``value``; raises for a path the account may not read."""
    if isinstance(value, str):
        path, raw = value, None
    elif isinstance(value, dict):
        path, raw = value.get("path"), value.get("bytes")
    else:
        return token_per_repo_id
    if raw is not None or not isinstance(path, str) or not path:
        return token_per_repo_id
    from datasets import config
    from datasets.utils.file_utils import is_local_path
    from datasets.utils.py_utils import string_to_dict

    # A chained URL ("zip://inner::<outer>") reads the outer location.
    source = path.split("::")[-1]
    if is_local_path(source):
        _require_readable(source)
        return token_per_repo_id
    # The decoders' own lookup: whatever key they will read is the one given a token here.
    pattern = (
        config.HUB_DATASETS_URL
        if source.startswith(config.HF_ENDPOINT)
        else config.HUB_DATASETS_HFFS_URL
    )
    try:
        fields = string_to_dict(source, pattern)
    except ValueError:
        fields = None
    if fields is None:
        # Not a Hub file the decoder can key a token on: it would dial it with the server's credentials.
        _refuse(path)
    tokens = dict(token_per_repo_id or {})
    if not isinstance(tokens.get(fields["repo_id"]), str):
        # None and True both mean the server's own token; only a token the caller supplied is sent.
        tokens[fields["repo_id"]] = False
    return tokens


def _decode_guard(decode_example):
    @functools.wraps(decode_example)
    def guarded(self, value, *args, **kwargs):
        if _is_owner():
            return decode_example(self, value, *args, **kwargs)
        tokens = kwargs.pop("token_per_repo_id", args[0] if args else None)
        return decode_example(
            self, value, *args[1:], token_per_repo_id = _permitted_tokens(value, tokens), **kwargs
        )

    return guarded


def _validated_cast(cast_table):
    @functools.wraps(cast_table)
    def guarded(self, pa_table, *args, **kwargs):
        pa_table.validate(full = True)
        return cast_table(self, pa_table, *args, **kwargs)

    return guarded


def guard_decoder(decode_example):
    """``decode_example`` behind the account check; for a decoder installed after ``install()``."""
    if getattr(decode_example, _MARK, False):
        return decode_example
    guarded = _decode_guard(decode_example)
    setattr(guarded, _MARK, True)
    return guarded


def _patch_attribute(owner, attribute, wrapper) -> None:
    original = getattr(owner, attribute, None)
    if original is not None and not getattr(original, _MARK, False):
        guarded = wrapper(original)
        setattr(guarded, _MARK, True)
        setattr(owner, attribute, guarded)


def _patch(module) -> None:
    if module.__name__ == _ARROW:
        owner = getattr(module, "Arrow", None)
        if owner is not None:
            _patch_attribute(owner, "_cast_table", _validated_cast)
        return
    owner = getattr(module, _DECODERS[module.__name__], None)
    if owner is not None:
        _patch_attribute(owner, "decode_example", _decode_guard)


class _Loader(importlib.abc.Loader):
    def __init__(self, loader):
        self._loader = loader

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        _patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        name,
        path,
        target = None,
    ):
        if name not in _DECODERS and name != _ARROW:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _Loader(spec.loader)
        return spec


def install() -> None:
    for name in (*_DECODERS, _ARROW):
        if name in sys.modules:
            _patch(sys.modules[name])
    if not any(isinstance(finder, _Finder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
