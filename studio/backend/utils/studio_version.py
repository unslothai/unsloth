# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Network-free Studio release version resolution for display-only UI."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from utils import _studio_release_build

_DEV_VERSION = "dev"
_GIT_TIMEOUT_SECONDS = 1.0
_STUDIO_TAG_RE = re.compile(r"^v\d+\.\d+\.\d+(?:-[0-9A-Za-z.][0-9A-Za-z.-]*)?$")
_GIT_DESCRIBE_SUFFIX_RE = re.compile(r"-\d+-g[0-9A-Fa-f]+(?:-dirty)?$")
_MAX_VERSION_LENGTH = 64


def is_valid_studio_release_version(value: object) -> bool:
    """Return True for Studio release tags such as ``v0.1.39-beta``."""
    if not isinstance(value, str):
        return False
    version = value.strip()
    if not version or len(version) > _MAX_VERSION_LENGTH:
        return False
    if version.endswith("-dirty") or _GIT_DESCRIBE_SUFFIX_RE.search(version):
        return False
    return _STUDIO_TAG_RE.fullmatch(version) is not None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _path_is_in_site_packages(path: Path) -> bool:
    return any(part in {"site-packages", "dist-packages"} for part in path.parts)


def _is_source_checkout(repo_root: Path) -> bool:
    return (repo_root / ".git").exists() and not _path_is_in_site_packages(
        Path(__file__).resolve()
    )


def _exact_git_studio_tag(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            [
                "git",
                "describe",
                "--tags",
                "--exact-match",
                "--match",
                "v[0-9]*",
                "HEAD",
            ],
            cwd = repo_root,
            check = False,
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            timeout = _GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    tag = result.stdout.strip()
    return tag if is_valid_studio_release_version(tag) else None


def get_studio_version(repo_root: Path | None = None) -> str:
    """Return the installed Studio release tag for display, or ``dev``.

    This value is intentionally separate from the PyPI ``unsloth`` package
    version used by update checks. It never performs network requests.
    """
    resolved_repo_root = repo_root or _repo_root()

    if _is_source_checkout(resolved_repo_root):
        git_tag = _exact_git_studio_tag(resolved_repo_root)
        return git_tag if git_tag is not None else _DEV_VERSION

    stamped_version = _studio_release_build.STUDIO_RELEASE_VERSION
    if is_valid_studio_release_version(stamped_version):
        return stamped_version.strip()

    return _DEV_VERSION
