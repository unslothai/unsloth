# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Compatibility accessors for formerly import-time path constants."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable


class LazyPath(os.PathLike[str]):
    """Resolve a root in the acting account whenever a caller uses the path.

    Existing imports retain a live accessor, including across account switches.
    Calling it explicitly returns a concrete Path for a job to retain. Do not
    retain this accessor as a job's account identity; bind that account instead.
    """

    def __init__(self, resolve: Callable[[], Path]):
        self._resolve = resolve

    def __call__(self) -> Path:
        return self._resolve()

    def __fspath__(self) -> str:
        return os.fspath(self._resolve())

    def __str__(self) -> str:
        return str(self._resolve())

    def __truediv__(self, other: str | os.PathLike[str]) -> Path:
        return self._resolve() / other

    def __rtruediv__(self, other: str | os.PathLike[str]) -> Path:
        return other / self._resolve()

    def __getattr__(self, name: str):
        return getattr(self._resolve(), name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LazyPath):
            other = other()
        return self._resolve() == other

    __hash__ = None
