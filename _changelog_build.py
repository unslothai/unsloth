# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Snapshot CHANGELOG.md into the studio package at build time.

CHANGELOG.md at the repo root stays the one file to edit. Copying it here,
rather than in build.sh, means every packaging path ships it, so release notes
still render when the popup cannot reach GitHub."""

from __future__ import annotations

import shutil
from pathlib import Path

from setuptools.command.build_py import build_py as _build_py

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "CHANGELOG.md"
SNAPSHOT = ROOT / "studio" / "CHANGELOG.md"


class build_py(_build_py):
    def run(self) -> None:
        # Written into the source tree only when that tree is writable, and
        # always into the staging directory. A PEP 517 build may run against an
        # immutable checkout (Nix, Bazel, a read-only container mount), where
        # copying beside the sources raised PermissionError before build_py had
        # even started and no wheel could be produced at all.
        if SOURCE.is_file():
            try:
                shutil.copyfile(SOURCE, SNAPSHOT)
            except OSError:
                pass
        super().run()
        if not SOURCE.is_file():
            return
        staged = Path(self.build_lib) / "studio" / "CHANGELOG.md"
        staged.parent.mkdir(parents = True, exist_ok = True)
        shutil.copyfile(SOURCE, staged)
