# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A torchcodec placeholder for the Colab-shaped smoke job, which has no CPU wheel to install.

Installed as a real (tiny) distribution on sys.path rather than as an entry poked into
sys.modules, because both of the obvious sys.modules shapes fail against the transformers the
Colab image pins:

  * `types.ModuleType("torchcodec")` leaves `__spec__` None, and `importlib.util.find_spec`
    RAISES for a module sitting in sys.modules with no spec instead of returning None. That is
    `ValueError: torchcodec.__spec__ is None`, which is how the smoke matrix was failing.

  * the same module carrying a hand-made ModuleSpec answers find_spec, but then
    `is_torchcodec_available()` is True -- transformers 5.16.1 calls `_is_package_available`
    without `return_version`, so it never looks at distribution metadata and reports presence
    from the spec alone -- and `audio_utils` line 61 immediately does
    `version.parse(importlib.metadata.version("torchcodec"))` at import time. With no
    distribution installed that is PackageNotFoundError: one import failure traded for another.

A distribution on disk answers both questions honestly. It also answers a third one the right
way: the version is deliberately below the 0.3.0 floor `load_audio` requires, so the "auto"
backend resolves to librosa and nothing ever asks this placeholder to decode anything.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import tempfile
from pathlib import Path

NAME = "torchcodec"
# Deliberately below transformers' 0.3.0 torchcodec floor; see the module docstring.
VERSION = "0.0.0"

_METADATA = f"""Metadata-Version: 2.1
Name: {NAME}
Version: {VERSION}
Summary: Placeholder installed by the Unsloth notebooks smoke job; no CPU wheel is published.
"""


def install(target_dir: "str | Path | None" = None) -> "str | None":
    """Put the placeholder on sys.path unless a real torchcodec is already importable.

    Returns the directory it was written to, or None when a real one was found.
    """
    if _already_present():
        # A genuine torchcodec, or a placeholder an earlier step already installed. Either way
        # replacing it would be the opposite of what this is for.
        return None

    root = Path(target_dir) if target_dir else Path(tempfile.mkdtemp(prefix = "unsloth-torchcodec-stub-"))
    package = root / NAME
    package.mkdir(parents = True, exist_ok = True)
    (package / "__init__.py").write_text(
        f'"""Placeholder; the real torchcodec publishes no CPU wheel."""\n\n__version__ = "{VERSION}"\n',
        encoding = "utf-8",
    )
    dist_info = root / f"{NAME}-{VERSION}.dist-info"
    dist_info.mkdir(parents = True, exist_ok = True)
    (dist_info / "METADATA").write_text(_METADATA, encoding = "utf-8")
    # Names who put it there, so anyone reading the venv can tell this from a real install.
    (dist_info / "INSTALLER").write_text("unsloth-notebooks-smoke\n", encoding = "utf-8")
    (dist_info / "RECORD").write_text("", encoding = "utf-8")

    sys.path.insert(0, str(root))
    # The path entry is new, so the finders' directory caches have to be dropped or the
    # package stays invisible to find_spec for the rest of the process.
    importlib.invalidate_caches()
    return str(root)


def _already_present() -> bool:
    """Whether importing torchcodec would find something without our help."""
    if NAME in sys.modules:
        return True
    try:
        return importlib.util.find_spec(NAME) is not None
    except (ImportError, ValueError):
        # ValueError is the spec-less sys.modules entry this file exists to avoid; treat a
        # broken pre-existing entry as present rather than shadowing it with a second one.
        return True
