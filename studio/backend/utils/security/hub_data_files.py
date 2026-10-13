# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Refuse Hub dataset cards whose data_files name a host path.

datasets joins a card's relative patterns to the repo (``hf://datasets/<repo>@<rev>``) but keeps an
absolute one as given, so a Hub repo whose README lists ``/home/owner/private.csv`` makes
``load_dataset("org/repo")`` read that host file. No real card does that, and it would let anyone
who can name a Hub dataset to Studio read host files through a preview or a load.

``install()`` wraps ``datasets.data_files.resolve_pattern`` once ``datasets`` is imported, without
importing it at startup; it is idempotent.
"""

import functools
import importlib.abc
import importlib.util
import sys

_MODULE = "datasets.data_files"


def _guard(resolve_pattern):
    if getattr(resolve_pattern, "_unsloth_hub_card_guard", False):
        return resolve_pattern
    from datasets.utils.file_utils import is_relative_path

    @functools.wraps(resolve_pattern)
    def guarded(pattern, base_path, *args, **kwargs):
        if (
            str(base_path).startswith("hf://")
            and "://" not in str(pattern)
            and not is_relative_path(str(pattern))
        ):
            raise FileNotFoundError(
                f"Unable to find '{pattern}': a Hub dataset cannot list a local file"
            )
        return resolve_pattern(pattern, base_path, *args, **kwargs)

    guarded._unsloth_hub_card_guard = True
    return guarded


def _patch(module) -> None:
    module.resolve_pattern = _guard(module.resolve_pattern)


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
        if name != _MODULE:
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
    module = sys.modules.get(_MODULE)
    if module is not None:
        _patch(module)
    elif not any(isinstance(finder, _Finder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
