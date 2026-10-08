# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Git on demand for ``git+`` installs. A working system Git is used as is; on Windows a missing one
is fetched once as portable MinGit into the Studio home (no admin, no winget) and put on the child's
PATH only. Elsewhere installing Git needs root, so callers get an error with instructions."""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

from filelock import FileLock

from utils.paths.storage_roots import studio_root

# Bump all three together; digests are GitHub's published asset digests for that release.
MINGIT_VERSION = "2.56.0.2"
_MINGIT_TAG = "v2.56.0.windows.2"
_MINGIT_SHA256 = {
    "64-bit": "da35e72aa21c005a5a0d298cfbae110bc1609a815730ea0dde84b01a1b3cd3be",
    "arm64": "38b33dc6024026e3315cf88ab2cfea65205bbd7bb3a8e824bd21c8ad4fe609a7",
}


_IS_WINDOWS = os.name == "nt"


class GitUnavailable(RuntimeError):
    pass


def _git_works(exe: str) -> bool:
    try:
        return (
            subprocess.run(
                [exe, "--version"],
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
                timeout = 30,
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


def _mingit_flavor() -> str:
    arch = (
        os.environ.get("PROCESSOR_ARCHITEW6432")
        or os.environ.get("PROCESSOR_ARCHITECTURE")
        or platform.machine()
    )
    return "arm64" if arch.strip().lower() == "arm64" else "64-bit"


def _mingit_dir(flavor: str) -> Path:
    return studio_root() / "tools" / f"mingit-{MINGIT_VERSION}-{flavor}"


def _mingit_git(root: Path) -> Path:
    return root / "cmd" / "git.exe"


def _download_mingit(flavor: str, target: Path) -> None:
    name = f"MinGit-{MINGIT_VERSION}-{flavor}.zip"
    url = f"https://github.com/git-for-windows/git/releases/download/{_MINGIT_TAG}/{name}"
    target.parent.mkdir(parents = True, exist_ok = True)
    with tempfile.TemporaryDirectory(dir = target.parent, prefix = ".mingit-") as tmp:
        archive = Path(tmp) / name
        digest = hashlib.sha256()
        request = urllib.request.Request(url, headers = {"User-Agent": "unsloth-studio"})
        with urllib.request.urlopen(request, timeout = 300) as response, open(archive, "wb") as out:
            while chunk := response.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != _MINGIT_SHA256[flavor]:
            raise GitUnavailable(f"{name} failed its sha256 check; not installing it")
        staged = Path(tmp) / "mingit"
        with zipfile.ZipFile(archive) as zf:
            base = staged.resolve()
            for member in zf.namelist():
                if not (staged / member).resolve().is_relative_to(base):
                    raise GitUnavailable(f"{name} has an entry outside its root: {member}")
            zf.extractall(staged)
        if not _git_works(str(_mingit_git(staged))):
            raise GitUnavailable(f"git from {name} does not run on this machine")
        os.replace(staged, target)


def ensure_git(*, allow_download: bool = True) -> Optional[str]:
    """None when Git on PATH works, else a dir to prepend to a child's PATH (Windows MinGit).
    Raises GitUnavailable when Git is missing and cannot be provided here."""
    exe = shutil.which("git")
    if exe and _git_works(exe):
        return None
    if not _IS_WINDOWS:
        raise GitUnavailable(
            "Git is required for this step but is not installed. Install it with your package "
            "manager (for example `sudo apt install git` or `xcode-select --install`) and retry."
        )
    flavor = _mingit_flavor()
    root = _mingit_dir(flavor)
    if _git_works(str(_mingit_git(root))):
        return str(root / "cmd")
    if not allow_download:
        raise GitUnavailable("Git is required for this step and this process is offline.")
    root.parent.mkdir(parents = True, exist_ok = True)
    with FileLock(str(root.parent / ".mingit.lock"), timeout = 600):
        if not _git_works(str(_mingit_git(root))):
            if root.exists():
                shutil.rmtree(root, ignore_errors = True)
            try:
                _download_mingit(flavor, root)
            except GitUnavailable:
                raise
            except Exception as exc:  # network, disk, bad zip
                raise GitUnavailable(f"Could not download portable Git (MinGit): {exc}") from exc
    return str(root / "cmd")


def with_git_on_path(env: dict, git_dir: Optional[str]) -> dict:
    if not git_dir:
        return env
    env = dict(env)
    key = next((k for k in env if k.upper() == "PATH"), "PATH")
    env[key] = git_dir + os.pathsep + env.get(key, "")
    return env
