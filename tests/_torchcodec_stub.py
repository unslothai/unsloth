# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Ship torchcodec as a real on-disk distribution; a sys.modules stub breaks find_spec and metadata."""

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
    """Put the placeholder on sys.path unless a real torchcodec is already present; returns None then."""
    if _already_present():
        return None

    root = (
        Path(target_dir)
        if target_dir
        else Path(tempfile.mkdtemp(prefix = "unsloth-torchcodec-stub-"))
    )
    package = root / NAME
    package.mkdir(parents = True, exist_ok = True)
    (package / "__init__.py").write_text(
        f'"""Placeholder; the real torchcodec publishes no CPU wheel."""\n\n__version__ = "{VERSION}"\n',
        encoding = "utf-8",
    )
    dist_info = root / f"{NAME}-{VERSION}.dist-info"
    dist_info.mkdir(parents = True, exist_ok = True)
    (dist_info / "METADATA").write_text(_METADATA, encoding = "utf-8")
    (dist_info / "INSTALLER").write_text("unsloth-notebooks-smoke\n", encoding = "utf-8")
    (dist_info / "RECORD").write_text("", encoding = "utf-8")

    sys.path.insert(0, str(root))
    # Drop finder caches, or the new path entry stays invisible to find_spec.
    importlib.invalidate_caches()
    return str(root)


def _already_present() -> bool:
    """Whether importing torchcodec would find something without our help."""
    if NAME in sys.modules:
        return True
    try:
        return importlib.util.find_spec(NAME) is not None
    except (ImportError, ValueError):
        # ValueError is a spec-less sys.modules entry; treat it as present.
        return True
