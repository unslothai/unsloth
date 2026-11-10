# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Refuses Hub cards whose data_files or data_dir point at host paths or file:// URLs; managed
accounts hf:// only; install() patches datasets lazily, never at startup, idempotent."""

import functools
import importlib.abc
import importlib.util
import os
import re
import sys

_HUB_URL_PROTOCOLS = frozenset({"hf", "http", "https"})
_MANAGED_URL_PROTOCOLS = frozenset({"hf"})
_PATCHES = {
    "datasets.data_files": ("resolve_pattern", "_pattern_guard"),
    "datasets.load": ("create_builder_configs_from_metadata_configs", "_card_guard"),
}
_MARK = "_unsloth_hub_card_guard"
# A cached Hub snapshot is loaded as a local folder, but its card is still the Hub's.
_SNAPSHOT = re.compile(r"[\\/]datasets--[^\\/]+[\\/]snapshots[\\/]")


def _hub_cache_roots() -> list[str]:
    roots = [os.environ.get("HF_HUB_CACHE")]
    try:
        from huggingface_hub import constants
        roots.append(constants.HF_HUB_CACHE)
    except Exception:  # noqa: BLE001 - a root we cannot read is simply not one
        pass
    try:
        from utils.hf_cache_settings import active_hf_hub_cache
        roots.append(active_hf_hub_cache())
    except Exception:  # noqa: BLE001
        pass
    return [os.path.realpath(root) for root in roots if root]


def _hub_base(base_path) -> bool:
    text = str(base_path or "")
    if text.startswith("hf://"):
        return True
    # Spelling alone is not enough: an owner's own folder may be named like a snapshot.
    if not _SNAPSHOT.search(text + "/"):
        return False
    real = os.path.realpath(text)
    return any(real.startswith(root.rstrip(os.sep) + os.sep) for root in _hub_cache_roots())


def _stays_on_the_hub(value, *, urls: bool) -> bool:
    from datasets.utils.file_utils import is_relative_path
    from fsspec.core import split_protocol

    text = str(value)
    if "::" in text:
        return False
    protocol, _ = split_protocol(text)
    if protocol is None:
        # Relative, but ".." would climb out of a cached snapshot onto the host.
        return is_relative_path(text) and ".." not in re.split(r"[\\/]", text)
    return urls and protocol in (_HUB_URL_PROTOCOLS if _is_owner() else _MANAGED_URL_PROTOCOLS)


def _is_owner() -> bool:
    try:
        from utils.account_context import is_owner_context
    except ImportError:  # outside Studio's backend there are no managed accounts
        return True
    return is_owner_context()


def _refuse(value) -> None:
    raise FileNotFoundError(f"Unable to find '{value}': a Hub dataset card cannot point here")


def _patterns(data_files):
    if isinstance(data_files, dict):
        for value in data_files.values():
            yield from _patterns(value)
    elif isinstance(data_files, (list, tuple)):
        for value in data_files:
            yield from _patterns(value)
    elif data_files is not None:
        yield data_files


def _pattern_guard(resolve_pattern):
    @functools.wraps(resolve_pattern)
    def guarded(pattern, base_path, *args, **kwargs):
        if _hub_base(base_path) and not _stays_on_the_hub(pattern, urls = True):
            _refuse(pattern)
        return resolve_pattern(pattern, base_path, *args, **kwargs)

    return guarded


def _card_guard(create_builder_configs):
    @functools.wraps(create_builder_configs)
    def guarded(
        module_path,
        metadata_configs,
        base_path = None,
        *args,
        **kwargs,
    ):
        # The card's own values, while their Hub origin is still known: a later join drops it.
        if _hub_base(base_path):
            for params in (metadata_configs or {}).values():
                data_dir = params.get("data_dir")
                if data_dir and not _stays_on_the_hub(data_dir, urls = False):
                    _refuse(data_dir)
                for pattern in _patterns(params.get("data_files")):
                    if not _stays_on_the_hub(pattern, urls = True):
                        _refuse(pattern)
        return create_builder_configs(module_path, metadata_configs, base_path, *args, **kwargs)

    return guarded


def _patch(module) -> None:
    attribute, wrapper = _PATCHES[module.__name__]
    original = getattr(module, attribute)
    if not getattr(original, _MARK, False):
        guarded = globals()[wrapper](original)
        setattr(guarded, _MARK, True)
        setattr(module, attribute, guarded)


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
        if name not in _PATCHES:
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
    for name in _PATCHES:
        if name in sys.modules:
            _patch(sys.modules[name])
    if not any(isinstance(finder, _Finder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
