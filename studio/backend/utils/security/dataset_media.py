# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep a managed account's dataset media to files it may read, and validate Arrow data files.

Media decoders open a cell's ``path`` locally or fetch Hub URLs with the server's token as fallback, so for a
managed account local paths must be readable by it (extracted archive members only via a recorded, readable
source archive), Hub URLs use only a caller-supplied token, and other URLs are refused. ``.arrow`` tables are
validated in full because pyarrow's IPC reader trusts declared offsets. ``install()`` patches lazily, idempotent.
"""

import functools
import importlib.abc
import importlib.util
import os
import re
import stat
import sys
import tempfile

_DECODERS = {
    "datasets.features.image": "Image",
    "datasets.features.audio": "Audio",
    "datasets.features.video": "Video",
    "datasets.features.pdf": "Pdf",
    "datasets.features.nifti": "Nifti",
}
_ARROW = "datasets.packaged_modules.arrow.arrow"
_EXTRACT = "datasets.utils.extract"
_SOURCE_SUFFIX = ".unsloth-source"
_SOURCE_MAX_BYTES = 16384
# Protocols ``datasets`` chains in front of an outer location ("zip://member::<outer>").
_MARK = "_unsloth_dataset_media_guard"
_SNAPSHOT_REPO = re.compile(r"[\\/]datasets--([^\\/]+?)--([^\\/]+)[\\/]")
_dataset_repos: set[str] = set()
# Extraction dir (real path) -> source archive.
_extracted_sources: dict[str, str] = {}


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


def _extracted_source(path: str):
    """The archive an extracted file came from, or None when it is not under the extraction cache."""
    from datasets import config
    from datasets.utils.extract import ExtractManager

    root = os.path.realpath(str(config.EXTRACTED_DATASETS_PATH)).rstrip(os.sep) + os.sep
    real = os.path.realpath(path)
    if not real.startswith(root):
        return None
    top = root + real[len(root) :].split(os.sep, 1)[0]
    source = _extracted_sources.get(top)
    if source is None:
        try:
            # A record planted as a FIFO or a link (an archive member can name it) must not block or redirect the read.
            flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(top + _SOURCE_SUFFIX, flags)
            with os.fdopen(fd, "rb") as f:
                if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                    return ""
                raw = f.read(_SOURCE_MAX_BYTES + 1)
            if len(raw) > _SOURCE_MAX_BYTES:
                return ""
            source = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            return ""
        # The directory is named by a hash of the archive's path, so a recorded source can not name another one.
        expected = ExtractManager(cache_dir = None)._get_output_path(source)
        if os.path.realpath(expected) != top:
            return ""
        _extracted_sources[top] = source
    return source


def _require_readable(path: str, depth: int = 0) -> None:
    from fastapi import HTTPException

    from core.training.account_jobs import visible_cached_path

    path = os.path.abspath(os.path.expanduser(path))
    try:
        visible_cached_path(path, "dataset")
        return
    except HTTPException:
        pass
    if _training_snapshot(path):
        return
    source = _extracted_source(path)
    if not source or depth >= 4:
        _refuse(path)
    try:
        _require_readable(source, depth + 1)
    except MediaPathRefused:
        _refuse(path)


def _record_extraction(extract):
    @functools.wraps(extract)
    def recorded(self, input_path, *args, **kwargs):
        output_path = extract(self, input_path, *args, **kwargs)
        if output_path != input_path:
            source, top = os.path.abspath(str(input_path)), os.path.realpath(str(output_path))
            _extracted_sources[top] = source
            # Atomic rename: readers never see a partial record.
            temp = None
            try:
                fd, temp = tempfile.mkstemp(
                    dir = os.path.dirname(top), prefix = os.path.basename(top) + _SOURCE_SUFFIX + "."
                )
                with os.fdopen(fd, "w", encoding = "utf-8") as f:
                    f.write(source)
                os.chmod(
                    temp, 0o644
                )  # mkstemp's 0600 would hide the record from other users of the cache
                os.replace(temp, top + _SOURCE_SUFFIX)
                temp = None
            except OSError:
                pass
            finally:
                if temp is not None:
                    try:
                        os.unlink(temp)
                    except OSError:
                        pass
        return output_path

    return recorded


def require_readable_media(path) -> None:
    """Refuse a local media path the managed account may not read; a no-op for the owner."""
    if not _is_owner():
        _require_readable(str(path))


_READ_THROUGH_PROTOCOLS = frozenset(
    {"zip", "tar", "gzip", "bz2", "xz", "lz4", "zstd", "simplecache", "filecache", "blockcache"}
)


def _anchored(pattern: str):
    """``pattern``'s ``{field}`` placeholders as groups and the rest literal, matched against a whole string."""
    parts = re.split(r"{([^:}]+)(?::[^}]+)?}", pattern)
    regex = "".join(
        f"(?P<{part}>.+)" if index % 2 else re.escape(part) for index, part in enumerate(parts)
    )
    return re.compile(regex)


def _hub_fields(source: str):
    """Repo fields of a Hub dataset file URL, else None. Matched whole: string_to_dict searches anywhere."""
    from datasets import config
    from datasets.utils.py_utils import string_to_dict

    endpoint = str(config.HF_ENDPOINT).rstrip("/")
    if source.startswith(endpoint + "/"):
        pattern = config.HUB_DATASETS_URL
    elif source.startswith("hf://"):
        pattern = config.HUB_DATASETS_HFFS_URL
    else:
        return None
    match = _anchored(pattern).fullmatch(source)
    if match is None:
        return None
    # The decoders key the token on string_to_dict's answer: it must name the same repo.
    try:
        theirs = string_to_dict(source, pattern)
    except ValueError:
        theirs = None
    if not theirs or theirs.get("repo_id") != match.group("repo_id"):
        return None
    return match.groupdict()


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
    from datasets.utils.file_utils import is_local_path

    # A chained URL ("zip://inner::<outer>") reads the outer location, but only an archive,
    # compression or cache hop reads through to the next one: `file://x::hf://...` opens x.
    *hops, source = path.split("::")
    if any(hop.split("://", 1)[0] not in _READ_THROUGH_PROTOCOLS for hop in hops):
        _refuse(path)
    if is_local_path(source):
        _require_readable(source)
        return token_per_repo_id
    fields = _hub_fields(source)
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
    if module.__name__ == _EXTRACT:
        owner = getattr(module, "ExtractManager", None)
        if owner is not None:
            _patch_attribute(owner, "extract", _record_extraction)
        return
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
        if name not in _DECODERS and name not in (_ARROW, _EXTRACT):
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
    for name in (*_DECODERS, _ARROW, _EXTRACT):
        if name in sys.modules:
            _patch(sys.modules[name])
    if not any(isinstance(finder, _Finder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
